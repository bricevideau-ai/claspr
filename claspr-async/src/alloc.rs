//! Lazy buffer allocs as [`DeviceOperation`]s — `device_slice_alloc`,
//! `host_buffer_alloc`, `shared_buffer_alloc`.
//!
//! Where the Tier 1 constructors [`DeviceSlice::alloc`] /
//! [`HostBuffer::alloc`] / [`SharedBuffer::alloc`] are synchronous
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
//! [`HostBuffer::alloc`]: claspr::HostBuffer::alloc
//! [`SharedBuffer::alloc`]: claspr::SharedBuffer::alloc

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use claspr::{DeviceSlice, HostBuffer, Result, SharedBuffer};
use std::marker::PhantomData;

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

// ── HostBuffer ──────────────────────────────────────────────────────

/// Lazy [`HostBuffer<T>`] alloc. Built by [`host_buffer_alloc`].
pub struct HostBufferAlloc<T> {
    len: usize,
    _phantom: PhantomData<fn() -> T>,
}

/// Allocate a [`HostBuffer<T>`] of `len` elements (uninitialised on
/// the device side; the persistent host map is established as part
/// of the alloc) when the chain reaches this op.
///
/// Uses the [`ExecutionContext`] itself as the [`Launcher`] for the
/// underlying `clEnqueueMapBuffer` — same queue the chain runs on.
///
/// [`Launcher`]: claspr::Launcher
pub fn host_buffer_alloc<T>(len: usize) -> HostBufferAlloc<T>
where
    T: Send + Sync + 'static,
{
    HostBufferAlloc {
        len,
        _phantom: PhantomData,
    }
}

impl<T> DeviceOperation for HostBufferAlloc<T>
where
    T: Send + Sync + 'static,
{
    type Output = HostBuffer<T>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(HostBuffer<T>, Deps)> {
        let buf = HostBuffer::<T>::alloc(ec, self.len)?;
        Ok((buf, deps))
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
