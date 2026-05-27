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
//! | [`HostBuffer<T>`] | no-op — buffer is permanently mapped | no-op |
//! | [`SharedBuffer<T>`] | blocking `clEnqueueSVMMap` | `clEnqueueSVMUnmap` |
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
//! [`HostAccessibleExt::acquire_host_view_read`]): the view passes
//! `&[T]` to the closure, the underlying map uses `CL_MAP_READ` only
//! (no writeback on unmap).

use crate::exec_ctx::ExecutionContext;
use crate::mappable::Mappable;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use claspr::{Buffer, DeviceSlice, Error, Event, HostBuffer, Launcher, Result, SharedBuffer};
use claspr::util::{RetainedQueue, mapped_slice, mapped_slice_mut};
use opencl3::command_queue::{
    CommandQueue, enqueue_map_buffer, enqueue_svm_map, enqueue_svm_unmap,
    enqueue_unmap_mem_object,
};
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE, ClMem};
use opencl3::types::{CL_NON_BLOCKING, cl_event, cl_map_flags};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr;

/// Wrap a raw cl3 status code into our typed [`Error`].
fn cl_to_err(code: opencl3::types::cl_int) -> Error {
    Error::OpenCl(opencl3::error_codes::ClError(code))
}

// ── Access mode markers ─────────────────────────────────────────────

/// Marker types selecting which `cl_map_flags` an acquire uses and
/// whether the resulting view's `Mappable::View` is `&[T]` or
/// `&mut [T]`. Implementors of [`MapAccess`] are zero-sized.
mod access {
    use super::*;

    /// Read-only access: `clEnqueueMapBuffer(CL_MAP_READ)`; closure
    /// sees `&[T]`. No writeback on unmap, so a read-only view is
    /// the cheapest way to look at device data from the host inside
    /// a chain.
    pub struct ReadOnly;
    /// Read/write access: `clEnqueueMapBuffer(CL_MAP_READ | CL_MAP_WRITE)`;
    /// closure sees `&mut [T]`. Unmap commits writes back.
    pub struct ReadWrite;

    /// Trait abstracting [`ReadOnly`] / [`ReadWrite`] into a
    /// `cl_map_flags` constant.
    pub trait MapAccess: Send + Sync + 'static {
        const MAP_FLAGS: cl_map_flags;
    }
    impl MapAccess for ReadOnly {
        const MAP_FLAGS: cl_map_flags = CL_MAP_READ;
    }
    impl MapAccess for ReadWrite {
        const MAP_FLAGS: cl_map_flags = CL_MAP_READ | CL_MAP_WRITE;
    }
}

pub use access::{MapAccess, ReadOnly, ReadWrite};

// ── HostAccessibleExt ───────────────────────────────────────────────

/// Adds [`acquire_host_view`](Self::acquire_host_view) to types that
/// can yield a host-side view of their data.
///
/// Bring into scope with `use claspr_async::HostAccessibleExt;` (or
/// via a future prelude).
pub trait HostAccessibleExt: Sized {
    /// The acquire op type for this buffer kind (read/write).
    type AcquireOp: DeviceOperation;
    /// The acquire op type for the read-only variant. Same kind for
    /// types where read/write is a no-op (e.g. [`HostBuffer`]).
    type AcquireReadOp: DeviceOperation;

    /// Acquire a read/write host view. Closure inside
    /// `and_then_host` receives `&mut [T]`.
    fn acquire_host_view(self) -> Self::AcquireOp;

    /// Acquire a read-only host view. Closure inside `and_then_host`
    /// receives `&[T]`. Cheaper for inspection-only patterns since
    /// the unmap doesn't commit writes back to the device.
    fn acquire_host_view_read(self) -> Self::AcquireReadOp;
}

// ── DeviceSlice: real map/unmap ─────────────────────────────────────

impl<T> HostAccessibleExt for DeviceSlice<T>
where
    T: Send + 'static,
{
    type AcquireOp = AcquireDeviceSliceOp<T, ReadWrite>;
    type AcquireReadOp = AcquireDeviceSliceOp<T, ReadOnly>;
    fn acquire_host_view(self) -> Self::AcquireOp {
        AcquireDeviceSliceOp {
            buf: Some(self),
            _access: PhantomData,
        }
    }
    fn acquire_host_view_read(self) -> Self::AcquireReadOp {
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
pub struct AcquireDeviceSliceOp<T, A: MapAccess> {
    buf: Option<DeviceSlice<T>>,
    _access: PhantomData<A>,
}

impl<T, A> DeviceOperation for AcquireDeviceSliceOp<T, A>
where
    T: Send + 'static,
    A: MapAccess,
{
    type Output = DeviceSliceHostView<T, A>;

    fn execute(
        mut self,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(Self::Output, Deps)> {
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
        let (wait_count, wait_ptr) = if wait_list.is_empty() {
            (0, ptr::null())
        } else {
            (wait_list.len() as u32, wait_list.as_ptr())
        };
        let mut host_ptr_raw: opencl3::types::cl_mem = ptr::null_mut();
        // SAFETY: cl_mem is live (we hold the DeviceSlice); the
        // map size matches the allocation's byte length; deps stays
        // alive for the call.
        let map_event = unsafe {
            enqueue_map_buffer(
                map_queue.raw(),
                cl_mem,
                CL_NON_BLOCKING,
                A::MAP_FLAGS,
                0,
                size,
                &mut host_ptr_raw,
                wait_count,
                wait_ptr,
            )
            .map_err(cl_to_err)?
        };
        let view = DeviceSliceHostView {
            buf: Some(buf),
            host_ptr: host_ptr_raw.cast::<T>(),
            len,
            map_queue,
            unmap_done: false,
            _access: PhantomData,
        };
        Ok((view, vec![wrap_event(Event::new(map_event))]))
    }
}

// ── DeviceSliceHostView ─────────────────────────────────────────────

/// Host view of a [`DeviceSlice<T>`] — the buffer is mapped into
/// host memory until `release_to_device` (or `Drop`) unmaps it.
///
/// `Deref<Target = [T]>` for both access modes;
/// `DerefMut` only for [`ReadWrite`]. When used as the input to
/// [`AndThenHost`](crate::AndThenHost), the closure receives
/// `&'_ [T]` ([`ReadOnly`]) or `&'_ mut [T]` ([`ReadWrite`])
/// directly — no extra method call on the view needed.
///
/// Carries a [`RetainedQueue`] so the defensive `Drop`-time unmap
/// (fired if `release_to_device` was never called and the inner
/// [`DeviceSlice`] is about to drop) has a valid queue handle even
/// if the original Launcher is long gone. The retain pair is owned
/// by the field's `Drop` — see the impl below.
pub struct DeviceSliceHostView<T, A: MapAccess> {
    buf: Option<DeviceSlice<T>>,
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
unsafe impl<T: Send, A: MapAccess> Send for DeviceSliceHostView<T, A> {}

impl<T, A: MapAccess> Deref for DeviceSliceHostView<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: host_ptr is the mapped pointer; remains valid
        // between map (in acquire) and unmap (in release/Drop). The
        // OpenCL map gives at least CL_MAP_READ access, so reads are
        // permitted in both ReadOnly and ReadWrite.
        unsafe { mapped_slice(self.host_ptr, self.len) }
    }
}

impl<T> DerefMut for DeviceSliceHostView<T, ReadWrite> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: same as Deref; the map was acquired with
        // CL_MAP_WRITE so mutation is permitted. `&mut self` upgrades
        // the shared-ref guarantee to unique.
        unsafe { mapped_slice_mut(self.host_ptr, self.len) }
    }
}

impl<T, A: MapAccess> Drop for DeviceSliceHostView<T, A> {
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
            // and cl_mem are live (we hold the DeviceSlice). Wrap
            // the resulting cl_event in claspr::Event so its Drop
            // releases it without an explicit release_event call.
            let res = unsafe {
                enqueue_unmap_mem_object(
                    self.map_queue.raw(),
                    buf.buffer().get(),
                    self.host_ptr.cast(),
                    0,
                    ptr::null(),
                )
            };
            if let Ok(ev) = res {
                let _ = opencl3::event::wait_for_events(&[ev]);
                let _ = Event::new(ev); // drops, releases the event
            }
        }
        // The `map_queue: RetainedQueue` field drops after this body
        // returns and releases the queue handle.
    }
}

impl<T, A> DeviceSliceHostView<T, A>
where
    T: Send + 'static,
    A: MapAccess,
{
    /// Enqueue the matching `clEnqueueUnmapMemObject` and yield the
    /// underlying [`DeviceSlice`] back so subsequent device stages
    /// can use it.
    pub fn release_to_device(self) -> ReleaseDeviceSliceOp<T, A> {
        ReleaseDeviceSliceOp { view: Some(self) }
    }
}

/// Combinator returned by
/// [`DeviceSliceHostView::release_to_device`]. Enqueues
/// `clEnqueueUnmapMemObject` waiting on `deps`, returns the
/// [`DeviceSlice`] and the unmap event.
pub struct ReleaseDeviceSliceOp<T, A: MapAccess> {
    view: Option<DeviceSliceHostView<T, A>>,
}

impl<T, A> DeviceOperation for ReleaseDeviceSliceOp<T, A>
where
    T: Send + 'static,
    A: MapAccess,
{
    type Output = DeviceSlice<T>;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(DeviceSlice<T>, Deps)> {
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
        let (wait_count, wait_ptr) = if wait_list.is_empty() {
            (0, ptr::null())
        } else {
            (wait_list.len() as u32, wait_list.as_ptr())
        };
        // SAFETY: host_ptr is the mapped pointer from acquire;
        // cl_mem is the buffer it was mapped from.
        let unmap_event = unsafe {
            enqueue_unmap_mem_object(
                q_raw,
                buf.buffer().get(),
                view.host_ptr.cast(),
                wait_count,
                wait_ptr,
            )
            .map_err(cl_to_err)?
        };
        view.unmap_done = true; // suppress Drop's defensive unmap
        // view drops here — only the RetainedQueue release fires.
        Ok((buf, vec![wrap_event(Event::new(unmap_event))]))
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

impl<T> Mappable for DeviceSliceHostView<T, ReadWrite>
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

impl<T> Mappable for DeviceSliceHostView<T, ReadOnly>
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

// ── HostBuffer: zero-copy — acquire/release are no-ops ──────────────

impl<T> HostAccessibleExt for HostBuffer<T>
where
    T: Send + 'static,
{
    type AcquireOp = AcquireHostBufferOp<T>;
    type AcquireReadOp = AcquireHostBufferOp<T>;
    fn acquire_host_view(self) -> AcquireHostBufferOp<T> {
        AcquireHostBufferOp { buf: Some(self) }
    }
    fn acquire_host_view_read(self) -> AcquireHostBufferOp<T> {
        AcquireHostBufferOp { buf: Some(self) }
    }
}

/// Combinator returned by `HostBuffer::acquire_host_view{,_read}`. No
/// CL command — the buffer is permanently mapped already
/// (`CL_MEM_ALLOC_HOST_PTR` + persistent map), so the view just wraps
/// the buf.
pub struct AcquireHostBufferOp<T> {
    buf: Option<HostBuffer<T>>,
}

impl<T> DeviceOperation for AcquireHostBufferOp<T>
where
    T: Send + 'static,
{
    type Output = HostBufferHostView<T>;

    fn execute(
        mut self,
        _ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(HostBufferHostView<T>, Deps)> {
        let buf = self
            .buf
            .take()
            .expect("AcquireHostBufferOp::execute called twice");
        // No CL command (zero-copy persistent map), but we still need
        // to wait on pending device writes before the host derefs the
        // map. Drain deps synchronously and forward an empty list.
        for ev in &deps {
            ev.as_ref().wait()?;
        }
        Ok((HostBufferHostView { buf }, Vec::new()))
    }
}

/// Host view of a [`HostBuffer<T>`]. Same shape as
/// [`DeviceSliceHostView`] but the underlying pointer is the
/// HostBuffer's always-mapped one.
pub struct HostBufferHostView<T> {
    buf: HostBuffer<T>,
}

impl<T> Deref for HostBufferHostView<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // HostBuffer derefs to [T] via its persistent map.
        &self.buf
    }
}

impl<T> DerefMut for HostBufferHostView<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.buf
    }
}

impl<T> HostBufferHostView<T>
where
    T: Send + 'static,
{
    /// Symmetric counterpart of [`HostAccessibleExt::acquire_host_view`].
    /// Since [`HostBuffer`] is zero-copy, release is also a no-op —
    /// just hand the buffer back.
    pub fn release_to_device(self) -> ReleaseHostBufferOp<T> {
        ReleaseHostBufferOp {
            view: Some(self.buf),
        }
    }
}

/// Combinator returned by [`HostBufferHostView::release_to_device`].
pub struct ReleaseHostBufferOp<T> {
    view: Option<HostBuffer<T>>,
}

impl<T> DeviceOperation for ReleaseHostBufferOp<T>
where
    T: Send + 'static,
{
    type Output = HostBuffer<T>;
    fn execute(mut self, _ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(HostBuffer<T>, Deps)> {
        // No CL command. Forward deps unchanged so downstream device
        // ops still wait on anything pending elsewhere in the chain.
        Ok((
            self.view
                .take()
                .expect("ReleaseHostBufferOp::execute called twice"),
            deps,
        ))
    }
}

/// Mappable impl so a `HostBufferHostView` can pass its inner
/// slice straight into an [`AndThenHost`](crate::AndThenHost) closure.
impl<T> Mappable for HostBufferHostView<T>
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
        let ptr = self.buf.as_ptr() as *mut T;
        Ok((
            HostViewHandle {
                ptr,
                len: self.buf.len(),
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

// ── SharedBuffer: non-blocking SVM map/unmap ────────────────────────

impl<T> HostAccessibleExt for SharedBuffer<T>
where
    T: Send + Sync + 'static,
{
    type AcquireOp = AcquireSharedBufferOp<T, ReadWrite>;
    type AcquireReadOp = AcquireSharedBufferOp<T, ReadOnly>;
    fn acquire_host_view(self) -> Self::AcquireOp {
        AcquireSharedBufferOp {
            buf: Some(self),
            _access: PhantomData,
        }
    }
    fn acquire_host_view_read(self) -> Self::AcquireReadOp {
        AcquireSharedBufferOp {
            buf: Some(self),
            _access: PhantomData,
        }
    }
}

/// Combinator returned by `SharedBuffer::acquire_host_view{,_read}`.
/// Issues a non-blocking `clEnqueueSVMMap` with `deps` as the
/// wait-list and produces a [`SharedBufferHostView`]. The map event
/// is returned in the output `Deps` so downstream stages (including
/// an `and_then_host` worker) gate on it before touching the SVM
/// memory.
pub struct AcquireSharedBufferOp<T, A: MapAccess> {
    buf: Option<SharedBuffer<T>>,
    _access: PhantomData<A>,
}

impl<T, A> DeviceOperation for AcquireSharedBufferOp<T, A>
where
    T: Send + Sync + 'static,
    A: MapAccess,
{
    type Output = SharedBufferHostView<T, A>;

    fn execute(
        mut self,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(Self::Output, Deps)> {
        let buf = self
            .buf
            .take()
            .expect("AcquireSharedBufferOp::execute called twice");
        let size = buf.len() * std::mem::size_of::<T>();
        let ptr = buf.ptr();
        // Retain the queue for the view's defensive Drop-time unmap.
        let queue = RetainedQueue::from_queue(ctx.cl_queue())?;
        let wait_list: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let (wait_count, wait_ptr) = if wait_list.is_empty() {
            (0, ptr::null())
        } else {
            (wait_list.len() as u32, wait_list.as_ptr())
        };
        // SAFETY: non-blocking SVM map. `ptr` came from clSVMAlloc on
        // the same context; `size` is the allocation's exact byte
        // length. Wait-list events stay alive across the call via the
        // `deps` Vec.
        let map_event = unsafe {
            enqueue_svm_map(
                queue.raw(),
                CL_NON_BLOCKING,
                A::MAP_FLAGS,
                ptr.cast(),
                size,
                wait_count,
                wait_ptr,
            )
            .map_err(cl_to_err)?
        };
        let view = SharedBufferHostView {
            buf: Some(buf),
            queue,
            unmap_done: false,
            _access: PhantomData,
        };
        Ok((view, vec![wrap_event(Event::new(map_event))]))
    }
}

/// Host view of a [`SharedBuffer<T>`] — a live SVM map.
///
/// Deref behaviour mirrors [`DeviceSliceHostView`]: `&[T]` always;
/// `&mut [T]` only for the [`ReadWrite`] access mode (type-system
/// enforcement against accidental writes through a read-only map).
pub struct SharedBufferHostView<T, A: MapAccess> {
    buf: Option<SharedBuffer<T>>,
    /// Retained `cl_command_queue` handle for the matching SVM unmap.
    queue: RetainedQueue,
    /// Set to `true` once `release_to_device` enqueued the unmap, so
    /// the view's `Drop` skips the defensive synchronous unmap.
    unmap_done: bool,
    _access: PhantomData<A>,
}

// SAFETY: SharedBuffer is itself Send (per its own impl). The
// queue is wrapped in RetainedQueue which is independently Send.
unsafe impl<T: Send, A: MapAccess> Send for SharedBufferHostView<T, A> {}

impl<T, A: MapAccess> Drop for SharedBufferHostView<T, A> {
    fn drop(&mut self) {
        if !self.unmap_done
            && let Some(buf) = self.buf.as_ref()
        {
            // Defensive sync unmap on the error path between acquire
            // and release. Issue the unmap, wait for it, register the
            // event on the SharedBuffer so its own clEnqueueSVMFree
            // (in SharedBuffer::drop) doesn't race against an unmap
            // still in flight on this queue.
            //
            // SAFETY: ptr was mapped in acquire; unmap exactly once
            // per acquire (we never reach this branch if unmap_done).
            let res = unsafe {
                enqueue_svm_unmap(self.queue.raw(), buf.ptr().cast(), 0, ptr::null())
            };
            match res {
                Ok(evt) => {
                    let _ = opencl3::event::wait_for_events(&[evt]);
                    buf.register_use(std::sync::Arc::new(Event::new(evt)));
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

impl<T, A: MapAccess> Deref for SharedBufferHostView<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        let buf = self
            .buf
            .as_ref()
            .expect("SharedBufferHostView already released");
        // SAFETY: the SVM pointer is valid and mapped between
        // acquire's clEnqueueSVMMap and release's / Drop's unmap.
        // CL_MAP_READ is always granted (both access modes set it).
        unsafe { mapped_slice(buf.ptr(), buf.len()) }
    }
}

impl<T> DerefMut for SharedBufferHostView<T, ReadWrite> {
    fn deref_mut(&mut self) -> &mut [T] {
        let buf = self
            .buf
            .as_ref()
            .expect("SharedBufferHostView already released");
        // SAFETY: same as Deref — plus the SVM map was acquired with
        // CL_MAP_WRITE (ReadWrite::MAP_FLAGS includes it) so mutation
        // is permitted by the OpenCL runtime.
        unsafe { mapped_slice_mut(buf.ptr(), buf.len()) }
    }
}

impl<T, A> SharedBufferHostView<T, A>
where
    T: Send + 'static,
    A: MapAccess,
{
    /// Enqueue the matching `clEnqueueSVMUnmap` waiting on `deps`,
    /// and yield the [`SharedBuffer`] back. The unmap event ends up
    /// in the chain's `Deps` so downstream device commands wait on
    /// it before touching the SVM allocation, and is also recorded
    /// on the [`SharedBuffer`] so its eventual
    /// `clEnqueueSVMFree` (on drop) ordering is preserved.
    pub fn release_to_device(self) -> ReleaseSharedBufferOp<T, A> {
        ReleaseSharedBufferOp { view: Some(self) }
    }
}

/// Combinator returned by [`SharedBufferHostView::release_to_device`].
pub struct ReleaseSharedBufferOp<T, A: MapAccess> {
    view: Option<SharedBufferHostView<T, A>>,
}

impl<T, A> DeviceOperation for ReleaseSharedBufferOp<T, A>
where
    T: Send + 'static,
    A: MapAccess,
{
    type Output = SharedBuffer<T>;

    fn execute(
        mut self,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(SharedBuffer<T>, Deps)> {
        let mut view = self
            .view
            .take()
            .expect("ReleaseSharedBufferOp::execute called twice");
        let buf = view
            .buf
            .take()
            .expect("SharedBufferHostView already released");
        let q_raw = ctx.cl_queue().get();
        let wait_list: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let (wait_count, wait_ptr) = if wait_list.is_empty() {
            (0, ptr::null())
        } else {
            (wait_list.len() as u32, wait_list.as_ptr())
        };
        // SAFETY: ptr was mapped in acquire; unmap exactly once.
        let unmap_event = unsafe {
            enqueue_svm_unmap(q_raw, buf.ptr().cast(), wait_count, wait_ptr).map_err(cl_to_err)?
        };
        view.unmap_done = true; // suppress Drop's defensive unmap
        // Build one Arc<Event> reused as both the chain's Dep and the
        // SharedBuffer's use-list entry — so its eventual SVMFree
        // queue-orders after the unmap regardless of when the
        // SharedBuffer ends up dropping.
        let arc_event = std::sync::Arc::new(Event::new(unmap_event));
        buf.register_use(std::sync::Arc::clone(&arc_event));
        // view drops here — only the retained queue release fires.
        Ok((buf, vec![arc_event]))
    }
}

/// `Mappable` impls so a `SharedBufferHostView` can pass its inner
/// slice straight into an [`AndThenHost`](crate::AndThenHost) closure.
/// The SVM map is already in place from acquire; `map`/`unmap` are
/// no-ops here, and the map event reaches the worker via
/// `source_evts` (the `Deps` carried through the chain).
impl<T> Mappable for SharedBufferHostView<T, ReadWrite>
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
            .expect("SharedBufferHostView already released");
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

impl<T> Mappable for SharedBufferHostView<T, ReadOnly>
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
            .expect("SharedBufferHostView already released");
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
