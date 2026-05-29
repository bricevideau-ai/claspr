//! Lazy buffer allocs as [`DeviceOperation`]s — `device_slice_alloc`
//! and `shared_buffer_alloc`.
//!
//! Where the Tier 1 constructors [`DeviceSlice::alloc`] /
//! [`SharedBuffer::alloc`] are synchronous
//! fallible host calls that need a `Context` (or `Launcher`) in scope,
//! these defer the alloc until execute time. Two ergonomic wins:
//!
//! 1. **Hoist allocs to the top of a chain** via [`bundle!`](crate::bundle!):
//!    ```ignore
//!    bundle!(upload(input), device_slice_alloc::<u32>(N), device_slice_alloc::<u32>(M))
//!        .and_then(|(input, hidden, output)| kernels.first_stage([N], input, hidden, output))
//!    ```
//!    Buffers materialize when the chain executes, in the running
//!    context. No need to thread `&ctx` through closures.
//!
//! 2. **Compose with [`and_then_with_context`](crate::DeviceOperation::and_then_with_context)**:
//!    ```ignore
//!    upload(input).and_then_with_context(|_ec, buf| {
//!        device_slice_alloc::<u32>(N).and_then(move |tmp|
//!            kernels.do_something([N], buf, tmp))
//!    })
//!    ```
//!    The closure returns an op (not a `Result<value>`), so the alloc
//!    and the kernel launch flow lazily without the `.wait()` /
//!    `Ok(...)` boilerplate the synchronous form needs.
//!
//! ## What these don't buy
//!
//! GPU-side parallelism. `clCreateBuffer` / `clSVMAlloc` /
//! `clEnqueueMapBuffer` are synchronous host calls; a `bundle!` of
//! three allocs runs the three constructors sequentially on the
//! executing thread. The gain is purely ergonomic — chain composition
//! plus deferring "where does this buffer live" to the chain's
//! running context.
//!
//! ## Deps semantics
//!
//! Alloc doesn't depend on prior queue work. The new buffer's
//! contents are uninitialised — there's nothing to gate on. Each
//! op's `execute` passes `deps` through unchanged so downstream
//! stages still see whatever event chain led into this point.
//!
//! [`DeviceSlice::alloc`]: claspr::DeviceSlice::alloc
//! [`SharedBuffer::alloc`]: claspr::SharedBuffer::alloc

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use crate::transfer::UploadSource;
use claspr::{DeviceSlice, Launcher, Result, SharedBuffer, register_drop_callback};
use opencl3::event::{Event, retain_event};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

// ── DeviceSlice ─────────────────────────────────────────────────────

/// Lazy [`DeviceSlice<T>`] alloc. Built by [`device_slice_alloc`].
pub struct DeviceSliceAlloc<T> {
    len: usize,
    _phantom: PhantomData<fn() -> T>,
}

/// Allocate a [`DeviceSlice<T>`] of `len` uninitialised elements when
/// the chain reaches this op. See module docs for the rationale.
pub fn device_slice_alloc<T>(len: usize) -> DeviceSliceAlloc<T>
where
    T: Send + 'static,
{
    DeviceSliceAlloc {
        len,
        _phantom: PhantomData,
    }
}

impl<T> DeviceOperation for DeviceSliceAlloc<T>
where
    T: Send + 'static,
{
    type Output = DeviceSlice<T>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(DeviceSlice<T>, Deps)> {
        let buf = DeviceSlice::<T>::alloc(ec.context(), self.len)?;
        Ok((buf, deps))
    }
}

// ── DeviceSlice fill (alloc + clEnqueueFillBuffer) ─────────────────

/// Lazy alloc + fill — produces a [`DeviceSlice<T>`] of `len`
/// elements all set to `value`, via `clEnqueueFillBuffer` (no host
/// allocation, no host→device transfer). Built by
/// [`device_slice_filled`] and by the [`device_slice!`](crate::device_slice!)
/// macro's `[value; count]` arm.
pub struct DeviceSliceFilled<T: Copy> {
    value: T,
    len: usize,
}

/// Allocate a [`DeviceSlice<T>`] of `len` elements all set to
/// `value`. Argument order mirrors [`vec!`](std::vec)'s `[value; count]`
/// shape.
///
/// Compared to `upload(vec![value; len])`: no host allocation of `len`
/// elements, no host→device transfer — just `clEnqueueFillBuffer` on
/// the chain's queue with the single-element pattern repeated across
/// the buffer.
pub fn device_slice_filled<T>(value: T, len: usize) -> DeviceSliceFilled<T>
where
    T: Copy + Send + 'static,
{
    DeviceSliceFilled { value, len }
}

impl<T> DeviceOperation for DeviceSliceFilled<T>
where
    T: Copy + Send + 'static,
{
    type Output = DeviceSlice<T>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(DeviceSlice<T>, Deps)> {
        let mut buf = DeviceSlice::<T>::alloc(ec.context(), self.len)?;
        // Fill on the chain's queue with upstream deps as wait-list.
        // Same shape as Upload: downstream waits on the fill event,
        // which transitively gates on upstream.
        let event = buf
            .fill(ec, self.value)
            .after_all(deps_as_events(&deps))
            .submit()?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// HostBuffer Tier 2 (alloc / filled / upload) removed 2026-05-29
// when HostBuffer itself was deleted — see commit log.

// ── SharedBuffer fill (alloc + clEnqueueSVMMemFill) ────────────────

/// Lazy alloc + fill for SVM — produces a [`SharedBuffer<T>`] of
/// `len` elements all set to `value`, via `clEnqueueSVMMemFill`
/// (no host allocation, no host→device transfer). Built by
/// [`shared_buffer_filled`]. SVM analog of [`DeviceSliceFilled`].
pub struct SharedBufferFilled<T: Copy> {
    value: T,
    len: usize,
}

/// Allocate a [`SharedBuffer<T>`] of `len` elements all set to
/// `value`. Argument order mirrors [`vec!`](std::vec)'s
/// `[value; count]` shape — same as
/// [`device_slice_filled`](crate::device_slice_filled), just SVM.
///
/// Surfaces [`Error::SvmNotAvailable`](claspr::Error::SvmNotAvailable)
/// at execute time on devices without SVM (same gate as
/// [`SharedBuffer::alloc`]).
pub fn shared_buffer_filled<T>(value: T, len: usize) -> SharedBufferFilled<T>
where
    T: Copy + Send + 'static,
{
    SharedBufferFilled { value, len }
}

impl<T> DeviceOperation for SharedBufferFilled<T>
where
    T: Copy + Send + 'static,
{
    type Output = SharedBuffer<T>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(SharedBuffer<T>, Deps)> {
        let buf = SharedBuffer::<T>::alloc(ec.context(), self.len)?;
        let event = buf
            .fill(ec, self.value)
            .after_all(deps_as_events(&deps))
            .submit()?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// ── SharedBuffer upload (alloc + clEnqueueSVMMemcpy from host) ─────

/// Lazy alloc + memcpy from a host source into a fresh
/// [`SharedBuffer<T>`]. SVM analog of [`crate::upload`] — wraps
/// `clEnqueueSVMMemcpy` instead of `clEnqueueWriteBuffer`. The host
/// source is kept alive by a `clSetEventCallback(CL_COMPLETE)` drop
/// holder until the copy finishes.
pub struct SharedBufferUpload<T> {
    source: Option<UploadSource<T>>,
}

/// Allocate a [`SharedBuffer<T>`] of `source.len()` elements and
/// memcpy `source` into it via `clEnqueueSVMMemcpy`. Mirrors
/// [`upload`](crate::upload) on the SVM side.
///
/// `T: Copy` because the memcpy is bytewise — types with non-trivial
/// Drop or owned heap data would alias on copy. Same constraint as
/// [`shared_buffer_filled`].
pub fn shared_buffer_upload<T, S>(source: S) -> SharedBufferUpload<T>
where
    T: Copy + Send + Sync + 'static,
    S: Into<UploadSource<T>>,
{
    SharedBufferUpload {
        source: Some(source.into()),
    }
}

impl<T> DeviceOperation for SharedBufferUpload<T>
where
    T: Copy + Send + Sync + 'static,
{
    type Output = SharedBuffer<T>;

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(SharedBuffer<T>, Deps)> {
        let source = self
            .source
            .take()
            .expect("SharedBufferUpload::execute called twice — internal claspr-async bug");
        let len = source.len();
        let buf = SharedBuffer::<T>::alloc(ec.context(), len)?;

        let raw_deps: Vec<opencl3::types::cl_event> =
            deps.iter().map(|d| d.as_ref().get()).collect();
        let size = len * std::mem::size_of::<T>();
        // SAFETY: buf.ptr() is a fresh, valid SVM allocation in the
        // queue's context. source.as_slice().as_ptr() is stable for
        // the lifetime of `source`; the drop callback below keeps
        // `source` alive until the copy event fires. CL_NON_BLOCKING
        // so we return immediately and chain on the event.
        let event = unsafe {
            ec.cl_queue().enqueue_svm_mem_cpy(
                opencl3::types::CL_NON_BLOCKING,
                buf.ptr() as *mut c_void,
                source.as_slice().as_ptr() as *const c_void,
                size,
                &raw_deps,
            )?
        };
        register_drop_callback(&event, Box::new(source))?;

        // Auto-register on the buffer's last_use so Drop's
        // clEnqueueSVMFree waits for the memcpy. Need clRetainEvent
        // since we hand the original Event to the chain's deps_out
        // and an independent Event to the buffer's last_use list.
        // SAFETY: event.get() is live; retain is paired with the
        // Event::drop inside the Arc.
        unsafe {
            retain_event(event.get())
                .map_err(|code| claspr::Error::OpenCl(opencl3::error_codes::ClError(code)))?;
        }
        buf.register_use(Arc::new(Event::new(event.get())));

        Ok((buf, vec![wrap_event(event)]))
    }
}

// ── SharedBuffer ────────────────────────────────────────────────────

/// Lazy [`SharedBuffer<T>`] (SVM) alloc. Built by [`shared_buffer_alloc`].
pub struct SharedBufferAlloc<T> {
    len: usize,
    _phantom: PhantomData<fn() -> T>,
}

/// Allocate a [`SharedBuffer<T>`] (SVM coarse-grain) of `len`
/// elements when the chain reaches this op.
///
/// Surfaces [`Error::SvmNotAvailable`] at execute time if the running
/// device doesn't support SVM. Same gate as the synchronous
/// constructor.
///
/// [`Error::SvmNotAvailable`]: claspr::Error::SvmNotAvailable
pub fn shared_buffer_alloc<T>(len: usize) -> SharedBufferAlloc<T>
where
    T: Send + 'static,
{
    SharedBufferAlloc {
        len,
        _phantom: PhantomData,
    }
}

impl<T> DeviceOperation for SharedBufferAlloc<T>
where
    T: Send + 'static,
{
    type Output = SharedBuffer<T>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(SharedBuffer<T>, Deps)> {
        let buf = SharedBuffer::<T>::alloc(ec.context(), self.len)?;
        Ok((buf, deps))
    }
}
