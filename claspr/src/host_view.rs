//! `HostAccessible` — make a buffer's bytes host-visible in queue
//! order so a host closure (typically inside
//! [`AndThenHost`](crate::AndThenHost)) can read or mutate them
//! in place, then hand the buffer back to subsequent device stages.
//!
//! Per buffer kind, acquire/release map to different CL primitives:
//!
//! | Buffer | Acquire | Release |
//! |---|---|---|
//! | [`DeviceSlice<T>`] | non-blocking `clEnqueueMapBuffer` (READ \| WRITE or READ only) | `clEnqueueUnmapMemObject` |
//! | [`MappedSlice<T>`] | non-blocking `clEnqueueSVMMap` | `clEnqueueSVMUnmap` |
//!
//! All three view types implement [`crate::mappable::Mappable`] so
//! the closure inside `and_then_host` receives the underlying slice
//! directly — `&mut [T]` for the read/write variants, `&[T]` for the
//! read-only variant. The closure does not need to call any method
//! on the view; passing it through is enough.
//!
//! ## Two patterns, one closure shape
//!
//! Direct (one stage of host work, framework auto-maps + auto-unmaps):
//!
//! ```ignore
//! .and_then(|buf| kernels.foo(buf))
//! .and_then_host(|slice: &mut [u32]| { slice[0] = 99; Ok(()) })
//! .and_then(|buf| kernels.bar(buf))
//! ```
//!
//! Explicit (the view persists across multiple stages — useful when
//! the host computation is split into pieces, or when caller wants
//! the map lifecycle to be visible in the chain):
//!
//! ```ignore
//! .and_then(|buf| kernels.foo(buf))
//! .and_then(|buf| buf.acquire_host_view())       // -> DeviceSliceHostView
//! .and_then_host(|slice: &mut [u32]| { slice[0] = 99; Ok(()) })
//! .and_then(|view| view.release_to_device())     // -> DeviceSlice
//! .and_then(|buf| kernels.bar(buf))
//! ```
//!
//! For pure host inspection without writing back, use
//! [`DeviceSlice::acquire_host_view_read`] (alias
//! [`HostReadableExt::acquire_host_view_read`]): the view passes
//! `&[T]` to the closure, the underlying map uses `CL_MAP_READ` only
//! (no writeback on unmap).

use crate::exec_ctx::ExecutionContext;
use crate::mappable::Mappable;
use crate::device_op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use crate::access::{HostReadable, HostWritable, MemMode};
use crate::map_primitive;
use crate::util::{RetainedQueue, mapped_slice, mapped_slice_mut};
use crate::{Buffer, DeviceSlice, Event, Launcher, MappedSlice, Result};
use opencl3::command_queue::CommandQueue;
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE, ClMem};
use opencl3::types::{cl_event, cl_map_flags};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

// ── Map-flag markers (host-side map access mode) ───────────────────
//
// These are `cl_map_flags` markers for `clEnqueueMapBuffer` — distinct
// from the `cl_mem_flags` markers in `crate::access` (which describe
// the buffer's creation-time kernel/host access). `MapReadOnly` says
// "the map call uses CL_MAP_READ only"; `MapReadWrite` says "CL_MAP_READ |
// CL_MAP_WRITE." The two layers compose: a `DeviceSlice<T, crate::ReadOnly>`
// (buffer-level kernel-RO) can be acquired with a `MapReadOnly` view
// (host-level read), but never with a `MapReadWrite` view (host can't
// write a buffer-level-kernel-RO buffer through the typed surface;
// the buffer-marker scheme will gate that statically).

mod map_access {
    use super::*;

    /// Read-only host map: `clEnqueueMapBuffer(CL_MAP_READ)`. Closure
    /// inside `and_then_host` sees `&[T]`. No writeback on unmap, so
    /// the cheapest way to inspect device data from the host inside
    /// a chain.
    pub struct MapReadOnly;
    /// Read/write host map: `clEnqueueMapBuffer(CL_MAP_READ | CL_MAP_WRITE)`.
    /// Closure inside `and_then_host` sees `&mut [T]`. Unmap commits
    /// writes back to the device.
    pub struct MapReadWrite;

    /// Trait abstracting [`MapReadOnly`] / [`MapReadWrite`] into a
    /// `cl_map_flags` constant.
    pub trait MapAccess: Send + Sync + 'static {
        const MAP_FLAGS: cl_map_flags;
    }
    impl MapAccess for MapReadOnly {
        const MAP_FLAGS: cl_map_flags = CL_MAP_READ;
    }
    impl MapAccess for MapReadWrite {
        const MAP_FLAGS: cl_map_flags = CL_MAP_READ | CL_MAP_WRITE;
    }
}

pub use map_access::{MapAccess, MapReadOnly, MapReadWrite};

// ── HostReadableExt / HostWritableExt ──────────────────────────────
//
// Two traits, split per host-access direction. Used to be one trait
// (`HostAccessibleExt`) with both methods on it, but that prevented
// per-marker gating — `Frozen` should expose `acquire_host_view_read`
// but not `acquire_host_view` (the mut version), which is unrepresent-
// able with a single trait that requires both methods.

/// Adds [`acquire_host_view_read`](Self::acquire_host_view_read) to
/// buffer types whose marker permits host reads
/// ([`crate::access::HostReadable`]).
pub trait HostReadableExt: Sized {
    /// The acquire op type for the read-only variant.
    type AcquireReadOp: DeviceOperation;

    /// Acquire a read-only host view. Closure inside `and_then_host`
    /// receives `&[T]`. Cheaper for inspection-only patterns since
    /// the unmap doesn't commit writes back to the device.
    fn acquire_host_view_read(self) -> Self::AcquireReadOp;
}

/// Adds [`acquire_host_view`](Self::acquire_host_view) (the mutating
/// variant) to buffer types whose marker permits host writes
/// ([`crate::access::HostWritable`]).
pub trait HostWritableExt: Sized {
    /// The acquire op type for the read/write variant.
    type AcquireOp: DeviceOperation;

    /// Acquire a read/write host view. Closure inside
    /// `and_then_host` receives `&mut [T]`.
    fn acquire_host_view(self) -> Self::AcquireOp;
}

/// Legacy compound trait — kept as an alias for callers that want
/// both directions at once. Users should prefer the split traits
/// going forward; this convenience trait is satisfied automatically
/// whenever both halves are.
pub trait HostAccessibleExt: HostReadableExt + HostWritableExt {}
impl<X: HostReadableExt + HostWritableExt> HostAccessibleExt for X {}

// ── DeviceSlice: real map/unmap ─────────────────────────────────────

impl<T, M: MemMode + HostReadable> HostReadableExt for DeviceSlice<T, M>
where
    T: Send + 'static,
{
    type AcquireReadOp = AcquireDeviceSliceOp<T, M, MapReadOnly>;
    fn acquire_host_view_read(self) -> Self::AcquireReadOp {
        AcquireDeviceSliceOp {
            buf: Some(self),
            _access: PhantomData,
        }
    }
}

impl<T, M: MemMode + HostWritable + HostReadable> HostWritableExt for DeviceSlice<T, M>
where
    T: Send + 'static,
{
    // HostWritable + HostReadable: the mut acquire issues
    // CL_MAP_READ | CL_MAP_WRITE, so the marker must permit both.
    // A buffer that's host-write-only (CL_MEM_HOST_WRITE_ONLY) isn't
    // expressible in our marker set, so HostWritable always implies
    // HostReadable for the markers we have today.
    type AcquireOp = AcquireDeviceSliceOp<T, M, MapReadWrite>;
    fn acquire_host_view(self) -> Self::AcquireOp {
        AcquireDeviceSliceOp {
            buf: Some(self),
            _access: PhantomData,
        }
    }
}

/// Combinator returned by [`DeviceSlice::acquire_host_view`] /
/// [`DeviceSlice::acquire_host_view_read`]. Enqueues a non-blocking
/// `clEnqueueMapBuffer` with `deps` as the wait-list and produces a
/// [`DeviceSliceHostView`].
pub struct AcquireDeviceSliceOp<T, M: MemMode, A: MapAccess> {
    buf: Option<DeviceSlice<T, M>>,
    _access: PhantomData<A>,
}

impl<T, M, A> DeviceOperation for AcquireDeviceSliceOp<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    type Output = DeviceSliceHostView<T, M, A>;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let buf = self
            .buf
            .take()
            .expect("AcquireDeviceSliceOp::execute called twice");
        let queue: &CommandQueue = ctx.cl_queue();
        // Retain the queue for the view's defensive Drop-time unmap
        // (it might outlive ctx if downstream stages don't release
        // explicitly).
        let map_queue = RetainedQueue::from_queue(queue)?;
        let cl_mem = buf.buffer().get();
        let len = Buffer::len(&buf);
        let size = len * std::mem::size_of::<T>();
        let wait_list: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SAFETY: cl_mem is live (we hold the DeviceSlice); the
        // map size matches the allocation's byte length; wait_list
        // stays alive for the call.
        let (host_ptr_raw, map_event) = unsafe {
            map_primitive::map_buffer(
                map_queue.raw(),
                cl_mem,
                false,
                A::MAP_FLAGS,
                0,
                size,
                &wait_list,
            )?
        };
        let view = DeviceSliceHostView {
            buf: Some(buf),
            host_ptr: host_ptr_raw.cast::<T>(),
            len,
            map_queue,
            unmap_done: false,
            _access: PhantomData,
        };
        Ok((view, vec![wrap_event(map_event)]))
    }
}

// ── DeviceSliceHostView ─────────────────────────────────────────────

/// Host view of a [`DeviceSlice<T>`] — the buffer is mapped into
/// host memory until `release_to_device` (or `Drop`) unmaps it.
///
/// `Deref<Target = [T]>` for both access modes;
/// `DerefMut` only for [`MapReadWrite`]. When used as the input to
/// [`AndThenHost`](crate::AndThenHost), the closure receives
/// `&'_ [T]` ([`MapReadOnly`]) or `&'_ mut [T]` ([`MapReadWrite`])
/// directly — no extra method call on the view needed.
///
/// Carries a [`RetainedQueue`] so the defensive `Drop`-time unmap
/// (fired if `release_to_device` was never called and the inner
/// [`DeviceSlice`] is about to drop) has a valid queue handle even
/// if the original Launcher is long gone. The retain pair is owned
/// by the field's `Drop` — see the impl below.
pub struct DeviceSliceHostView<T, M: MemMode, A: MapAccess> {
    buf: Option<DeviceSlice<T, M>>,
    host_ptr: *mut T,
    len: usize,
    map_queue: RetainedQueue,
    unmap_done: bool,
    _access: PhantomData<A>,
}

// SAFETY: `*mut T` is a mapped pointer that worker code accesses
// serially under unique borrow rules (see the doc on
// `DeviceSliceMapHandle` in mappable.rs). `RetainedQueue` is Send
// on its own.
unsafe impl<T: Send, M: MemMode, A: MapAccess> Send for DeviceSliceHostView<T, M, A> {}

impl<T, M: MemMode, A: MapAccess> Deref for DeviceSliceHostView<T, M, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: host_ptr is the mapped pointer; remains valid
        // between map (in acquire) and unmap (in release/Drop). The
        // OpenCL map gives at least CL_MAP_READ access, so reads are
        // permitted in both MapReadOnly and MapReadWrite.
        unsafe { mapped_slice(self.host_ptr, self.len) }
    }
}

impl<T, M: MemMode> DerefMut for DeviceSliceHostView<T, M, MapReadWrite> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: same as Deref; the map was acquired with
        // CL_MAP_WRITE so mutation is permitted. `&mut self` upgrades
        // the shared-ref guarantee to unique.
        unsafe { mapped_slice_mut(self.host_ptr, self.len) }
    }
}

impl<T, M: MemMode, A: MapAccess> Drop for DeviceSliceHostView<T, M, A> {
    fn drop(&mut self) {
        if !self.unmap_done
            && let Some(buf) = self.buf.as_ref()
        {
            // Defensive sync unmap: release_to_device was never
            // called, but we still need to unmap before the inner
            // DeviceSlice's clReleaseMemObject (in its own Drop)
            // fires, or the cl_mem is in a "still mapped" state on
            // release which strict implementations reject.
            //
            // SAFETY: host_ptr came from our own acquire; map_queue
            // and cl_mem are live (we hold the DeviceSlice).
            let res = unsafe {
                map_primitive::unmap_mem_object(
                    self.map_queue.raw(),
                    buf.buffer().get(),
                    self.host_ptr.cast(),
                    &[],
                )
            };
            match res {
                Ok(ev) => {
                    let _ = ev.wait();
                    // `ev` drops here, releasing the cl_event.
                }
                Err(_) => buf.ctx().record_err(),
            }
        }
        // The `map_queue: RetainedQueue` field drops after this body
        // returns and releases the queue handle.
    }
}

impl<T, M, A> DeviceSliceHostView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    /// Enqueue the matching `clEnqueueUnmapMemObject` and yield the
    /// underlying [`DeviceSlice`] back so subsequent device stages
    /// can use it.
    pub fn release_to_device(self) -> ReleaseDeviceSliceOp<T, M, A> {
        ReleaseDeviceSliceOp { view: Some(self) }
    }
}

/// Combinator returned by
/// [`DeviceSliceHostView::release_to_device`]. Enqueues
/// `clEnqueueUnmapMemObject` waiting on `deps`, returns the
/// [`DeviceSlice`] and the unmap event.
pub struct ReleaseDeviceSliceOp<T, M: MemMode, A: MapAccess> {
    view: Option<DeviceSliceHostView<T, M, A>>,
}

impl<T, M, A> DeviceOperation for ReleaseDeviceSliceOp<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    type Output = DeviceSlice<T, M>;

    fn execute(
        mut self,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(DeviceSlice<T, M>, Deps)> {
        let mut view = self
            .view
            .take()
            .expect("ReleaseDeviceSliceOp::execute called twice");
        let buf = view
            .buf
            .take()
            .expect("DeviceSliceHostView already released");
        let q_raw = ctx.cl_queue().get();
        let wait_list: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SAFETY: host_ptr is the mapped pointer from acquire;
        // cl_mem is the buffer it was mapped from.
        let unmap_event = unsafe {
            map_primitive::unmap_mem_object(
                q_raw,
                buf.buffer().get(),
                view.host_ptr.cast(),
                &wait_list,
            )?
        };
        view.unmap_done = true; // suppress Drop's defensive unmap
        // view drops here — only the RetainedQueue release fires.
        Ok((buf, vec![wrap_event(unmap_event)]))
    }
}

// ── Mappable for DeviceSliceHostView (closure-direct integration) ───

/// `MapHandle` for the host-view Mappable impls. The view is already
/// mapped; this just carries the pointer + len into the worker
/// thread so the closure can construct a slice.
pub struct HostViewHandle<T> {
    ptr: *mut T,
    len: usize,
    _t: PhantomData<T>,
}

unsafe impl<T: Send> Send for HostViewHandle<T> {}

impl<T, M: MemMode> Mappable for DeviceSliceHostView<T, M, MapReadWrite>
where
    T: Send + 'static,
{
    type View<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type MapHandle = HostViewHandle<T>;

    fn map(
        &self,
        _queue: &CommandQueue,
        _deps: &[cl_event],
    ) -> Result<(Self::MapHandle, Vec<Event>)> {
        // Already mapped via acquire — no CL command. The map event
        // that gated coherence is in source_evts (the worker waits
        // on it before calling view()).
        Ok((
            HostViewHandle {
                ptr: self.host_ptr,
                len: self.len,
                _t: PhantomData,
            },
            Vec::new(),
        ))
    }
    fn enqueue_unmap(
        _handle: &mut Self::MapHandle,
        _queue: &CommandQueue,
        _wait_for: &[cl_event],
    ) -> Result<Vec<Event>> {
        // View stays mapped past the and_then_host closure; the user
        // calls release_to_device when ready.
        Ok(Vec::new())
    }
    fn view<'a>(handle: &'a mut Self::MapHandle) -> Self::View<'a> {
        // SAFETY: see DeviceSliceHostView::deref_mut. The MapHandle
        // borrow is unique on the worker thread.
        unsafe { mapped_slice_mut(handle.ptr, handle.len) }
    }
    fn mark_unmap_not_done(_handle: &mut Self::MapHandle) {
        // No unmap was enqueued by us — nothing to do on error path.
    }
}

impl<T, M: MemMode> Mappable for DeviceSliceHostView<T, M, MapReadOnly>
where
    T: Send + Sync + 'static,
{
    type View<'a>
        = &'a [T]
    where
        Self: 'a;
    type MapHandle = HostViewHandle<T>;

    fn map(
        &self,
        _queue: &CommandQueue,
        _deps: &[cl_event],
    ) -> Result<(Self::MapHandle, Vec<Event>)> {
        Ok((
            HostViewHandle {
                ptr: self.host_ptr,
                len: self.len,
                _t: PhantomData,
            },
            Vec::new(),
        ))
    }
    fn enqueue_unmap(
        _handle: &mut Self::MapHandle,
        _queue: &CommandQueue,
        _wait_for: &[cl_event],
    ) -> Result<Vec<Event>> {
        Ok(Vec::new())
    }
    fn view<'a>(handle: &'a mut Self::MapHandle) -> Self::View<'a> {
        // SAFETY: see DeviceSliceHostView::deref. The buffer was
        // mapped with CL_MAP_READ; we hand out an immutable slice.
        unsafe { mapped_slice(handle.ptr, handle.len) }
    }
    fn mark_unmap_not_done(_handle: &mut Self::MapHandle) {}
}

// ── MappedSlice: non-blocking SVM map/unmap ────────────────────────

impl<T, M: MemMode + HostReadable> HostReadableExt for MappedSlice<T, M>
where
    T: Send + Sync + 'static,
{
    type AcquireReadOp = AcquireMappedSliceOp<T, M, MapReadOnly>;
    fn acquire_host_view_read(self) -> Self::AcquireReadOp {
        AcquireMappedSliceOp {
            buf: Some(self),
            _access: PhantomData,
        }
    }
}

impl<T, M: MemMode + HostWritable + HostReadable> HostWritableExt for MappedSlice<T, M>
where
    T: Send + Sync + 'static,
{
    type AcquireOp = AcquireMappedSliceOp<T, M, MapReadWrite>;
    fn acquire_host_view(self) -> Self::AcquireOp {
        AcquireMappedSliceOp {
            buf: Some(self),
            _access: PhantomData,
        }
    }
}

/// Combinator returned by `MappedSlice::acquire_host_view{,_read}`.
/// Issues a non-blocking `clEnqueueSVMMap` with `deps` as the
/// wait-list and produces a [`MappedSliceHostView`]. The map event
/// is returned in the output `Deps` so downstream stages (including
/// an `and_then_host` worker) gate on it before touching the SVM
/// memory.
pub struct AcquireMappedSliceOp<T, M: MemMode, A: MapAccess> {
    buf: Option<MappedSlice<T, M>>,
    _access: PhantomData<A>,
}

impl<T, M, A> DeviceOperation for AcquireMappedSliceOp<T, M, A>
where
    T: Send + Sync + 'static,
    M: MemMode,
    A: MapAccess,
{
    type Output = MappedSliceHostView<T, M, A>;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let buf = self
            .buf
            .take()
            .expect("AcquireMappedSliceOp::execute called twice");
        let size = buf.len() * std::mem::size_of::<T>();
        let ptr = buf.ptr();
        // Retain the queue for the view's defensive Drop-time unmap.
        let queue = RetainedQueue::from_queue(ctx.cl_queue())?;
        let wait_list: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SAFETY: non-blocking SVM map. `ptr` came from clSVMAlloc on
        // the same context; `size` is the allocation's exact byte
        // length. Wait-list events stay alive across the call via the
        // `deps` Vec.
        let map_event = unsafe {
            map_primitive::svm_map(
                queue.raw(),
                false,
                A::MAP_FLAGS,
                ptr.cast(),
                size,
                &wait_list,
            )?
        };
        let view = MappedSliceHostView {
            buf: Some(buf),
            queue,
            unmap_done: false,
            _access: PhantomData,
        };
        Ok((view, vec![wrap_event(map_event)]))
    }
}

/// Host view of a [`MappedSlice<T>`] — a live SVM map.
///
/// Deref behaviour mirrors [`DeviceSliceHostView`]: `&[T]` always;
/// `&mut [T]` only for the [`MapReadWrite`] access mode (type-system
/// enforcement against accidental writes through a read-only map).
pub struct MappedSliceHostView<T, M: MemMode, A: MapAccess> {
    buf: Option<MappedSlice<T, M>>,
    /// Retained `cl_command_queue` handle for the matching SVM unmap.
    queue: RetainedQueue,
    /// Set to `true` once `release_to_device` enqueued the unmap, so
    /// the view's `Drop` skips the defensive synchronous unmap.
    unmap_done: bool,
    _access: PhantomData<A>,
}

// SAFETY: MappedSlice is itself Send (per its own impl). The
// queue is wrapped in RetainedQueue which is independently Send.
unsafe impl<T: Send, M: MemMode, A: MapAccess> Send for MappedSliceHostView<T, M, A> {}

impl<T, M: MemMode, A: MapAccess> Drop for MappedSliceHostView<T, M, A> {
    fn drop(&mut self) {
        if !self.unmap_done
            && let Some(buf) = self.buf.as_ref()
        {
            // Defensive sync unmap on the error path between acquire
            // and release. Issue the unmap, wait for it, register the
            // event on the MappedSlice so its own clEnqueueSVMFree
            // (in MappedSlice::drop) doesn't race against an unmap
            // still in flight on this queue.
            //
            // SAFETY: ptr was mapped in acquire; unmap exactly once
            // per acquire (we never reach this branch if unmap_done).
            let res = unsafe { map_primitive::svm_unmap(self.queue.raw(), buf.ptr().cast(), &[]) };
            match res {
                Ok(evt) => {
                    let _ = evt.wait();
                    buf.register_use(std::sync::Arc::new(evt));
                }
                Err(_) => {
                    buf.ctx().record_err();
                }
            }
        }
        // The `queue: RetainedQueue` field drops after this body
        // returns and releases the queue handle.
    }
}

impl<T, M: MemMode, A: MapAccess> Deref for MappedSliceHostView<T, M, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        let buf = self
            .buf
            .as_ref()
            .expect("MappedSliceHostView already released");
        // SAFETY: the SVM pointer is valid and mapped between
        // acquire's clEnqueueSVMMap and release's / Drop's unmap.
        // CL_MAP_READ is always granted (both access modes set it).
        unsafe { mapped_slice(buf.ptr(), buf.len()) }
    }
}

impl<T, M: MemMode> DerefMut for MappedSliceHostView<T, M, MapReadWrite> {
    fn deref_mut(&mut self) -> &mut [T] {
        let buf = self
            .buf
            .as_ref()
            .expect("MappedSliceHostView already released");
        // SAFETY: same as Deref — plus the SVM map was acquired with
        // CL_MAP_WRITE (MapReadWrite::MAP_FLAGS includes it) so mutation
        // is permitted by the OpenCL runtime.
        unsafe { mapped_slice_mut(buf.ptr(), buf.len()) }
    }
}

impl<T, M, A> MappedSliceHostView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    /// Enqueue the matching `clEnqueueSVMUnmap` waiting on `deps`,
    /// and yield the [`MappedSlice`] back. The unmap event ends up
    /// in the chain's `Deps` so downstream device commands wait on
    /// it before touching the SVM allocation, and is also recorded
    /// on the [`MappedSlice`] so its eventual
    /// `clEnqueueSVMFree` (on drop) ordering is preserved.
    pub fn release_to_device(self) -> ReleaseMappedSliceOp<T, M, A> {
        ReleaseMappedSliceOp { view: Some(self) }
    }
}

/// Combinator returned by [`MappedSliceHostView::release_to_device`].
pub struct ReleaseMappedSliceOp<T, M: MemMode, A: MapAccess> {
    view: Option<MappedSliceHostView<T, M, A>>,
}

impl<T, M, A> DeviceOperation for ReleaseMappedSliceOp<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    type Output = MappedSlice<T, M>;

    fn execute(
        mut self,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(MappedSlice<T, M>, Deps)> {
        let mut view = self
            .view
            .take()
            .expect("ReleaseMappedSliceOp::execute called twice");
        let buf = view
            .buf
            .take()
            .expect("MappedSliceHostView already released");
        let q_raw = ctx.cl_queue().get();
        let wait_list: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SAFETY: ptr was mapped in acquire; unmap exactly once.
        let unmap_event = unsafe { map_primitive::svm_unmap(q_raw, buf.ptr().cast(), &wait_list)? };
        view.unmap_done = true; // suppress Drop's defensive unmap
        // Build one Arc<Event> reused as both the chain's Dep and the
        // MappedSlice's use-list entry — so its eventual SVMFree
        // queue-orders after the unmap regardless of when the
        // MappedSlice ends up dropping.
        let arc_event = std::sync::Arc::new(unmap_event);
        buf.register_use(std::sync::Arc::clone(&arc_event));
        // view drops here — only the retained queue release fires.
        Ok((buf, vec![arc_event]))
    }
}

/// `Mappable` impls so a `MappedSliceHostView` can pass its inner
/// slice straight into an [`AndThenHost`](crate::AndThenHost) closure.
/// The SVM map is already in place from acquire; `map`/`unmap` are
/// no-ops here, and the map event reaches the worker via
/// `source_evts` (the `Deps` carried through the chain).
impl<T, M: MemMode> Mappable for MappedSliceHostView<T, M, MapReadWrite>
where
    T: Send + 'static,
{
    type View<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type MapHandle = HostViewHandle<T>;

    fn map(
        &self,
        _queue: &CommandQueue,
        _deps: &[cl_event],
    ) -> Result<(Self::MapHandle, Vec<Event>)> {
        let buf = self
            .buf
            .as_ref()
            .expect("MappedSliceHostView already released");
        Ok((
            HostViewHandle {
                ptr: buf.ptr(),
                len: buf.len(),
                _t: PhantomData,
            },
            Vec::new(),
        ))
    }
    fn enqueue_unmap(
        _handle: &mut Self::MapHandle,
        _queue: &CommandQueue,
        _wait_for: &[cl_event],
    ) -> Result<Vec<Event>> {
        Ok(Vec::new())
    }
    fn view<'a>(handle: &'a mut Self::MapHandle) -> Self::View<'a> {
        unsafe { mapped_slice_mut(handle.ptr, handle.len) }
    }
    fn mark_unmap_not_done(_handle: &mut Self::MapHandle) {}
}

impl<T, M: MemMode> Mappable for MappedSliceHostView<T, M, MapReadOnly>
where
    T: Send + Sync + 'static,
{
    type View<'a>
        = &'a [T]
    where
        Self: 'a;
    type MapHandle = HostViewHandle<T>;

    fn map(
        &self,
        _queue: &CommandQueue,
        _deps: &[cl_event],
    ) -> Result<(Self::MapHandle, Vec<Event>)> {
        let buf = self
            .buf
            .as_ref()
            .expect("MappedSliceHostView already released");
        Ok((
            HostViewHandle {
                ptr: buf.ptr(),
                len: buf.len(),
                _t: PhantomData,
            },
            Vec::new(),
        ))
    }
    fn enqueue_unmap(
        _handle: &mut Self::MapHandle,
        _queue: &CommandQueue,
        _wait_for: &[cl_event],
    ) -> Result<Vec<Event>> {
        Ok(Vec::new())
    }
    fn view<'a>(handle: &'a mut Self::MapHandle) -> Self::View<'a> {
        unsafe { mapped_slice(handle.ptr, handle.len) }
    }
    fn mark_unmap_not_done(_handle: &mut Self::MapHandle) {}
}

// Suppress the unused import warning for `deps_as_events` — kept
// available for future variants that need it.
#[allow(dead_code)]
fn _unused_deps_as_events_keepalive(d: &Deps) -> impl Iterator<Item = &Event> {
    deps_as_events(d)
}
