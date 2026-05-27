//! `HostAccessible` — three-stage acquire / host-work / release for
//! host code that needs to look at device data.
//!
//! Spike scenario 16. The pattern:
//!
//! ```ignore
//! upload(host_vec)
//!     .and_then(|buf| /* GPU work */)
//!     .and_then(|buf| buf.acquire_host_view())     // d2h into a scratch Vec
//!     .and_then_host(|mut view| {                   // DerefMut on the scratch
//!         view[0] += 100.0;
//!         Ok(view)
//!     })
//!     .and_then(|view| view.release_to_device())   // h2d the scratch back
//!     .and_then(|buf| /* more GPU work */)
//!     .and_then(download)
//!     .sync(&ctx)?
//! ```
//!
//! Per buffer type, acquire/release map to different CL primitives:
//!
//! | Buffer | Acquire | Release |
//! |---|---|---|
//! | `DeviceSlice<T>` | `clEnqueueReadBuffer` into a host `Vec<T>` | `clEnqueueWriteBuffer` from the Vec back into the buffer |
//! | `HostBuffer<T>` (planned) | no-op — already host-mapped | no-op |
//! | `SharedBuffer<T>` (planned) | `clEnqueueSVMMap` | `clEnqueueSVMUnmap` |
//!
//! **Correctness**: the read inside `acquire`'s `execute` must complete
//! before the host gets to deref the view — otherwise `and_then_host`
//! sees stale data. Similarly, `release`'s `execute` must complete the
//! write before the view (and its `Vec`) drops. Both stages use
//! `.wait()` on the underlying [`ReadOp`](claspr::ReadOp) /
//! [`WriteOp`](claspr::WriteOp) builders for that. The chain still
//! gets queue-ordered pipelining at the boundaries with neighbouring
//! ops — the acquire's read is queue-ordered after prior kernels, the
//! release's write is queue-ordered before subsequent kernels.

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use claspr::{Buffer, DeviceSlice, Error, HostBuffer, Launcher, Result, SharedBuffer};
use opencl3::command_queue::{
    enqueue_svm_map, enqueue_svm_unmap, release_command_queue, retain_command_queue,
};
use opencl3::event::release_event;
use opencl3::types::{CL_BLOCKING, cl_command_queue};
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::slice;

/// Wrap a raw cl3 status code into our typed [`Error`].
fn cl_to_err(code: opencl3::types::cl_int) -> Error {
    Error::OpenCl(opencl3::error_codes::ClError(code))
}

// ── Extension trait for the user-facing entry point ─────────────────

/// Adds [`acquire_host_view`](Self::acquire_host_view) to types that
/// can yield a host-side scratch view of their data. Only
/// [`DeviceSlice`] today; `HostBuffer` and `SharedBuffer` will impl
/// this trait in a follow-up.
///
/// Bring into scope with `use claspr_async::HostAccessibleExt;` (or
/// via a future prelude).
pub trait HostAccessibleExt: Sized {
    /// The acquire op type for this buffer kind.
    type AcquireOp: DeviceOperation;

    /// Build a [`DeviceOperation`] whose `Output` is a host-side view
    /// into the buffer. The view DerefMut-s to `[T]` so the user can
    /// read or write through it. Compose with
    /// [`DeviceOperationHostExt::and_then_host`](crate::DeviceOperationHostExt::and_then_host).
    fn acquire_host_view(self) -> Self::AcquireOp;
}

impl<T> HostAccessibleExt for DeviceSlice<T>
where
    T: Clone + Default + Send + 'static,
{
    type AcquireOp = AcquireDeviceSliceOp<T>;
    fn acquire_host_view(self) -> AcquireDeviceSliceOp<T> {
        AcquireDeviceSliceOp { buf: Some(self) }
    }
}

// ── DeviceSlice acquire ──────────────────────────────────────────────

/// Combinator returned by `DeviceSlice::acquire_host_view`. Allocates
/// a host scratch `Vec<T>`, enqueues a blocking read into it, hands
/// back a [`DeviceSliceHostView`] holding both the device buffer and
/// the populated scratch.
pub struct AcquireDeviceSliceOp<T> {
    buf: Option<DeviceSlice<T>>,
}

impl<T> DeviceOperation for AcquireDeviceSliceOp<T>
where
    T: Clone + Default + Send + 'static,
{
    type Output = DeviceSliceHostView<T>;

    fn execute(
        mut self,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(DeviceSliceHostView<T>, Deps)> {
        let buf = self
            .buf
            .take()
            .expect("AcquireDeviceSliceOp::execute called twice");
        let mut data = vec![T::default(); buf.len()];
        // .wait() is correctness-required: the host can't deref a
        // stale Vec when `and_then_host` next runs the closure. CL_TRUE
        // inside the driver makes this efficient. Deps are passed in
        // as wait-list so the read is queue-ordered after prior
        // commands (e.g. an upload or kernel that produced `buf`'s
        // data).
        buf.read(ctx, &mut data)
            .after_all(deps_as_events(&deps))
            .wait()?;
        // Blocking read means no new outbound event — the data is
        // already in `data` by the time we return. The `and_then_host`
        // that follows runs synchronously on the host.
        Ok((DeviceSliceHostView { buf, data }, Vec::new()))
    }
}

// ── DeviceSliceHostView ─────────────────────────────────────────────

/// Host-side view of a [`DeviceSlice<T>`]. Carries the (now-populated)
/// host scratch alongside the source device buffer so
/// [`release_to_device`](Self::release_to_device) can write the scratch
/// back without re-allocating the buffer.
///
/// `Deref<Target = [T]>` + `DerefMut` for in-place host work via
/// [`DeviceOperationHostExt::and_then_host`](crate::DeviceOperationHostExt::and_then_host).
pub struct DeviceSliceHostView<T> {
    buf: DeviceSlice<T>,
    data: Vec<T>,
}

impl<T> Deref for DeviceSliceHostView<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.data
    }
}

impl<T> DerefMut for DeviceSliceHostView<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> DeviceSliceHostView<T>
where
    T: Send + 'static,
{
    /// Yield a [`DeviceOperation`] that writes the host scratch back
    /// into the device buffer and returns the buffer.
    pub fn release_to_device(self) -> ReleaseDeviceSliceOp<T> {
        ReleaseDeviceSliceOp { view: Some(self) }
    }
}

// ── DeviceSlice release ─────────────────────────────────────────────

/// Combinator returned by [`DeviceSliceHostView::release_to_device`].
/// Enqueues a blocking write from the host scratch back into the
/// device buffer, then yields the buffer.
pub struct ReleaseDeviceSliceOp<T> {
    view: Option<DeviceSliceHostView<T>>,
}

impl<T> DeviceOperation for ReleaseDeviceSliceOp<T>
where
    T: Send + 'static,
{
    type Output = DeviceSlice<T>;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(DeviceSlice<T>, Deps)> {
        let view = self
            .view
            .take()
            .expect("ReleaseDeviceSliceOp::execute called twice");
        let DeviceSliceHostView { mut buf, data } = view;
        // Non-blocking write — the keep-alive on `data` is provided
        // by `register_drop_callback` so the Vec lives until OpenCL
        // is done reading.
        let event = buf
            .write(ctx, &data)
            .after_all(deps_as_events(&deps))
            .submit()?;
        claspr::register_drop_callback(&event, Box::new(data))?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// ── HostBuffer: zero-copy — acquire/release are no-ops ──────────────

impl<T> HostAccessibleExt for HostBuffer<T>
where
    T: Send + 'static,
{
    type AcquireOp = AcquireHostBufferOp<T>;
    fn acquire_host_view(self) -> AcquireHostBufferOp<T> {
        AcquireHostBufferOp { buf: Some(self) }
    }
}

/// Combinator returned by `HostBuffer::acquire_host_view`. No CL
/// command — the buffer is permanently mapped already
/// (`CL_MEM_ALLOC_HOST_PTR` + persistent map), so the view just wraps
/// the buf and lets the host Deref-access the existing mapped pointer.
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
            ev.wait()?;
        }
        Ok((HostBufferHostView { buf }, Vec::new()))
    }
}

/// Host-side view of a [`HostBuffer<T>`]. Same shape as
/// [`DeviceSliceHostView`] but DerefMut goes straight to the
/// always-mapped host pointer — no extra scratch buffer.
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

// ── SharedBuffer: clEnqueueSVMMap (blocking) + clEnqueueSVMUnmap ────

impl<T> HostAccessibleExt for SharedBuffer<T>
where
    T: Send + 'static,
{
    type AcquireOp = AcquireSharedBufferOp<T>;
    fn acquire_host_view(self) -> AcquireSharedBufferOp<T> {
        AcquireSharedBufferOp { buf: Some(self) }
    }
}

/// Combinator returned by `SharedBuffer::acquire_host_view`. Issues a
/// blocking `clEnqueueSVMMap(CL_TRUE, CL_MAP_READ|WRITE)` so the host
/// has coherent access to the SVM allocation, retains the queue handle
/// so the matching unmap on release uses the same queue, then wraps
/// everything in a [`SharedBufferHostView`].
pub struct AcquireSharedBufferOp<T> {
    buf: Option<SharedBuffer<T>>,
}

impl<T> DeviceOperation for AcquireSharedBufferOp<T>
where
    T: Send + 'static,
{
    type Output = SharedBufferHostView<T>;

    fn execute(
        mut self,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(SharedBufferHostView<T>, Deps)> {
        let buf = self
            .buf
            .take()
            .expect("AcquireSharedBufferOp::execute called twice");
        let q_raw: cl_command_queue = ctx.cl_queue().get();
        let size = buf.len() * std::mem::size_of::<T>();
        let ptr = buf.ptr();
        // Pass deps as the map's wait-list — the blocking map then
        // returns only after prior device work is done.
        let wait_list: Vec<opencl3::types::cl_event> =
            deps.iter().map(|d| d.as_ref().get()).collect();
        let (wait_count, wait_ptr) = if wait_list.is_empty() {
            (0, ptr::null())
        } else {
            (wait_list.len() as u32, wait_list.as_ptr())
        };
        // SAFETY: blocking SVM map (CL_TRUE). `ptr` came from
        // clSVMAlloc on the same context; `size` is the allocation's
        // exact byte length. CL_MAP_READ | CL_MAP_WRITE gives full
        // coherent host access. Map flags `3` = READ|WRITE per the CL
        // spec (CL_MAP_READ=1, CL_MAP_WRITE=2).
        let evt = unsafe {
            enqueue_svm_map(
                q_raw,
                CL_BLOCKING,
                opencl3::memory::CL_MAP_READ | opencl3::memory::CL_MAP_WRITE,
                ptr.cast(),
                size,
                wait_count,
                wait_ptr,
            )
            .map_err(cl_to_err)?
        };
        // Keep deps alive across the enqueue.
        drop(deps);
        // SAFETY: blocking-map returned an event — release it immediately
        // (we don't need to track it; the map is already done).
        unsafe { release_event(evt).map_err(cl_to_err)? };
        // Retain the queue handle so it stays alive for the matching
        // unmap on release. Mirrors what claspr's
        // `SharedReadGuard::new` does for the same reason.
        unsafe { retain_command_queue(q_raw).map_err(cl_to_err)? };
        // Map was blocking, so no outbound event — host work next.
        Ok((
            SharedBufferHostView {
                buf: Some(buf),
                queue: q_raw,
            },
            Vec::new(),
        ))
    }
}

/// Host-side view of a [`SharedBuffer<T>`] — a live SVM map.
///
/// **Drop hazard**: the inner [`SharedBuffer`]'s own Drop fires a
/// `clEnqueueSVMFree` on the Context's default in-order queue (per
/// Phase 0 fix). The map we issued lives on a *different* queue
/// (the chain's OOO queue), so there's no implicit ordering between
/// the unmap and the free. We *must* unmap (and wait for the unmap
/// event) before letting the inner SharedBuffer drop. Drop below
/// does that; [`release_to_device`](Self::release_to_device) takes
/// the same path, then yields the buffer back.
///
/// `buf` is wrapped in `Option<>` so `release_to_device` can take it
/// out — Drop then becomes a no-op on the post-release path.
pub struct SharedBufferHostView<T> {
    buf: Option<SharedBuffer<T>>,
    /// Retained `cl_command_queue` handle for the matching SVM unmap.
    queue: cl_command_queue,
}

// SAFETY: cl_command_queue is an opaque handle; we hold a retained
// reference. The SharedBuffer inside is itself Send (its impl above).
unsafe impl<T: Send> Send for SharedBufferHostView<T> {}

impl<T> SharedBufferHostView<T> {
    /// Shared helper for the unmap sequence used by both the explicit
    /// release path and the implicit Drop path. Returns the buffer
    /// back; the queue handle is released as a side effect. Errors are
    /// returned (release) or sunk into the sticky-error counter (Drop).
    ///
    /// Records the unmap event as the buffer's `last_use` so its own
    /// Drop's `clEnqueueSVMFree` queue-orders after the unmap. No
    /// host-side wait — the cross-queue dependency is expressed
    /// device-side via the free's wait-list.
    fn unmap(&mut self) -> Result<Option<SharedBuffer<T>>> {
        let Some(buf) = self.buf.take() else {
            return Ok(None);
        };
        // SAFETY: ptr was mapped in acquire; unmap exactly once.
        let evt = unsafe { enqueue_svm_unmap(self.queue, buf.ptr().cast(), 0, ptr::null()) }
            .map_err(cl_to_err)?;
        // Wrap the raw cl_event so its Drop releases the reference we
        // got back from enqueue. Push an Arc<Event> onto the SharedBuffer's
        // in-flight-use list so its `clEnqueueSVMFree` waits on the
        // unmap before freeing.
        let event = std::sync::Arc::new(opencl3::event::Event::new(evt));
        buf.register_use(event);
        // SAFETY: the queue handle was retained in acquire; release
        // exactly once here.
        unsafe { release_command_queue(self.queue) }.map_err(cl_to_err)?;
        Ok(Some(buf))
    }
}

impl<T> Drop for SharedBufferHostView<T> {
    fn drop(&mut self) {
        // If the user already called release_to_device, `buf` is None
        // and unmap is a no-op. Otherwise we're on the panic / early-
        // exit path: do the unmap synchronously so the inner buf can
        // safely drop afterwards.
        match self.unmap() {
            Ok(_) => {}
            Err(_) => {
                // Errors here are unrecoverable. The buf, if any, is
                // still around (because unmap took it). Stash an
                // error on its context before dropping it.
                if let Some(buf) = self.buf.take() {
                    buf.ctx().record_err();
                }
            }
        }
    }
}

impl<T> Deref for SharedBufferHostView<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        let buf = self
            .buf
            .as_ref()
            .expect("SharedBufferHostView already released");
        // SAFETY: the SVM pointer is valid and mapped for read+write
        // for the lifetime of this view (between acquire's map and
        // release's / Drop's unmap).
        unsafe { slice::from_raw_parts(buf.ptr(), buf.len()) }
    }
}

impl<T> DerefMut for SharedBufferHostView<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        let buf = self
            .buf
            .as_ref()
            .expect("SharedBufferHostView already released");
        // SAFETY: same as Deref — plus the SVM map was acquired with
        // CL_MAP_WRITE so mutation is permitted.
        unsafe { slice::from_raw_parts_mut(buf.ptr(), buf.len()) }
    }
}

impl<T> SharedBufferHostView<T>
where
    T: Send + 'static,
{
    /// Symmetric counterpart of [`HostAccessibleExt::acquire_host_view`].
    /// Issues the `clEnqueueSVMUnmap`, waits for it to complete (so
    /// the buffer is safe to drop on a different queue afterwards),
    /// and yields the [`SharedBuffer`] back.
    pub fn release_to_device(self) -> ReleaseSharedBufferOp<T> {
        ReleaseSharedBufferOp { view: Some(self) }
    }
}

/// Combinator returned by [`SharedBufferHostView::release_to_device`].
pub struct ReleaseSharedBufferOp<T> {
    view: Option<SharedBufferHostView<T>>,
}

impl<T> DeviceOperation for ReleaseSharedBufferOp<T>
where
    T: Send + 'static,
{
    type Output = SharedBuffer<T>;

    fn execute(
        mut self,
        _ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(SharedBuffer<T>, Deps)> {
        let mut view = self
            .view
            .take()
            .expect("ReleaseSharedBufferOp::execute called twice");
        // Drain any deps from parallel branches before unmapping —
        // unmap is queue-local and won't otherwise see them.
        for ev in &deps {
            ev.wait()?;
        }
        let buf = view.unmap()?.expect("view's buf was already released");
        // view's Drop runs now — buf is None, no-op. unmap already
        // waited internally, so we return no outbound events.
        Ok((buf, Vec::new()))
    }
}
