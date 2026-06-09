//! In-place transforms of existing buffers as [`DeviceOperation`]s
//! — `device_slice_fill` / `device_slice_copy` / `device_slice_write`
//! and the SVM analogues `mapped_slice_fill` / `mapped_slice_copy`.
//!
//! The Tier 2 alloc family in [`crate::alloc`] hands you a *new*
//! buffer; these ops take a buffer that already exists upstream in
//! the chain and transform it in place. Two ergonomic wins:
//!
//! 1. **Stay in the chain.** Without these, in-chain fill/copy/write
//!    requires falling back to
//!    [`and_then_with_context`](crate::DeviceOperation::and_then_with_context)
//!    + the synchronous Tier 1 terminal:
//!    ```ignore
//!    .and_then_with_context(|ec, buf| {
//!        buf.fill(0u32).wait_on(ec)?;
//!        Ok(buf)
//!    })
//!    ```
//!    With these ops the same shape is a one-liner:
//!    ```ignore
//!    .and_then(|buf| device_slice_fill(buf, 0u32))
//!    ```
//! 2. **Preserve the marker.** `device_slice_fill` carries `M` through
//!    — a [`HostReadOnly`](claspr::HostReadOnly) buffer stays
//!    `HostReadOnly` after the fill. The Tier 1 bounds
//!    (`KernelWritable` for fill, `HostWritable` for write) propagate
//!    so misuse rejects at compile time (see the
//!    `compile_fail/buffer_ops_*` fixtures).
//!
//! ## Naming vs. alloc family
//!
//! The past-participle names in [`crate::alloc`] —
//! [`device_slice_filled`](crate::device_slice_filled) /
//! [`mapped_slice_filled`](crate::mapped_slice_filled) — produce a
//! *new* buffer filled with a value. The verb names here
//! (`device_slice_fill` etc.) take an *existing* buffer as input and
//! emit it again as output. The difference is one letter; the
//! signatures are unambiguous (`(value, len)` vs. `(buf, value)`).
//!
//! ## Deps semantics
//!
//! Each op enqueues a single Tier 1 op with the upstream `deps` as
//! its wait-list, then returns the resulting Event as the only entry
//! in the downstream `Deps` — the same shape as the alloc-uninit-based
//! [`FillUninit`](crate::FillUninit) /
//! [`WriteUninit`](crate::WriteUninit) chain ops. Copy
//! returns a single Event covering both buffers; the buffers' own
//! `Drop` waits track the event independently for SVM buffers (via
//! `last_use` registration inside [`SvmFillOp`] / [`SvmCopyOp`]
//! `into_event`) so cross-queue free is queue-ordered correctly.
//!
//! [`SvmFillOp`]: claspr::SvmFillOp
//! [`SvmCopyOp`]: claspr::SvmCopyOp

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use crate::transfer::UploadSource;
use claspr::{
    DeviceSlice, Fillable, HostWritable, MappedSlice, MemMode, Result, register_drop_callback,
};

// ── DeviceSlice fill (in-place clEnqueueFillBuffer) ────────────────

/// Lazy in-place [`DeviceSlice<T, M>`] fill. Built by
/// [`device_slice_fill`].
pub struct DeviceSliceFillOp<T: Copy, M: MemMode> {
    buf: Option<DeviceSlice<T, M>>,
    value: T,
}

/// Fill an existing [`DeviceSlice<T, M>`] with `value` via
/// `clEnqueueFillBuffer` on the chain's queue, threading upstream
/// events into the wait-list. The buffer passes through as the op's
/// output so the chain can keep using it.
///
/// Bound `M: KernelWritable` — `clEnqueueFillBuffer` counts as a
/// kernel-side write. [`ReadOnly`](claspr::ReadOnly) and
/// [`Frozen`](claspr::Frozen) markers reject at compile time (see
/// `compile_fail/buffer_ops_fill_on_*`).
pub fn device_slice_fill<T, M>(buf: DeviceSlice<T, M>, value: T) -> DeviceSliceFillOp<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable,
{
    DeviceSliceFillOp {
        buf: Some(buf),
        value,
    }
}

impl<T, M> DeviceOperation for DeviceSliceFillOp<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn execute(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(DeviceSlice<T, M>, Deps)> {
        let mut buf = self
            .buf
            .take()
            .expect("DeviceSliceFillOp::execute called twice — internal claspr-async bug");
        let event = buf
            .fill(self.value)
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// DeviceSlice copy moved to claspr-async/src/copy.rs as the
// polymorphic `CopyTo` trait — covers all (src, dst) buffer-kind
// pairs uniformly. Use `src.copy_to(dst).and_then(...)`.

// ── DeviceSlice write (in-place clEnqueueWriteBuffer from host) ────

/// Lazy in-place write into an existing [`DeviceSlice<T, M>`]. Built
/// by [`device_slice_write`].
pub struct DeviceSliceWriteOp<T, M: MemMode> {
    state: Option<(DeviceSlice<T, M>, UploadSource<T>)>,
}

/// Write `source` into an existing [`DeviceSlice<T, M>`] via a
/// non-blocking `clEnqueueWriteBuffer`. The host source is kept alive
/// by a `clSetEventCallback(CL_COMPLETE)` drop holder until the write
/// finishes — same keep-alive trick the [`upload!`](crate::upload!)
/// macro uses. The buffer passes through as the op's output.
///
/// Bound `M: HostWritable` — excludes
/// [`HostReadOnly`](claspr::HostReadOnly),
/// [`Frozen`](claspr::Frozen), and
/// [`DeviceScratch`](claspr::DeviceScratch). `Vec<T>` / `Box<[T]>` /
/// `Arc<[T]>` all coerce via [`UploadSource`].
pub fn device_slice_write<T, M, S>(buf: DeviceSlice<T, M>, source: S) -> DeviceSliceWriteOp<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable,
    S: Into<UploadSource<T>>,
{
    DeviceSliceWriteOp {
        state: Some((buf, source.into())),
    }
}

impl<T, M> DeviceOperation for DeviceSliceWriteOp<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn execute(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(DeviceSlice<T, M>, Deps)> {
        let (mut buf, source) = self
            .state
            .take()
            .expect("DeviceSliceWriteOp::execute called twice — internal claspr-async bug");
        let event = buf
            .write(source.as_slice())
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        // Move `source` into a Box, hand to OpenCL's user_data. The
        // thunk drops it when CL_COMPLETE fires — exactly when the
        // runtime is done reading from the host heap.
        register_drop_callback(&event, Box::new(source))?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// ── MappedSlice fill (in-place clEnqueueSVMMemFill) ────────────────

/// Lazy in-place [`MappedSlice<T, M>`] fill. Built by
/// [`mapped_slice_fill`]. SVM analog of [`DeviceSliceFillOp`].
pub struct MappedSliceFillOp<T: Copy, M: MemMode> {
    buf: Option<MappedSlice<T, M>>,
    value: T,
}

/// Fill an existing [`MappedSlice<T, M>`] with `value` via
/// `clEnqueueSVMMemFill` on the chain's queue. The buffer passes
/// through as the op's output. Bound `M: KernelWritable` — same gate
/// as [`device_slice_fill`].
///
/// Drop-ordering: the Tier 1 [`SvmFillOp`](claspr::SvmFillOp)
/// `into_event` already registers the fill event on the buffer's
/// `last_use` list, so the buffer's eventual `clEnqueueSVMFree` will
/// wait for this op.
pub fn mapped_slice_fill<T, M>(buf: MappedSlice<T, M>, value: T) -> MappedSliceFillOp<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable,
{
    MappedSliceFillOp {
        buf: Some(buf),
        value,
    }
}

impl<T, M> DeviceOperation for MappedSliceFillOp<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn execute(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(MappedSlice<T, M>, Deps)> {
        let buf = self
            .buf
            .take()
            .expect("MappedSliceFillOp::execute called twice — internal claspr-async bug");
        let event = buf
            .fill(self.value)
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// MappedSlice copy moved to claspr-async/src/copy.rs (`CopyTo`
// trait). Use `src.copy_to(dst).and_then(...)`.
