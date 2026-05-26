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
use crate::op::DeviceOperation;
use claspr::{Buffer, DeviceSlice, Result};
use std::ops::{Deref, DerefMut};

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

    fn execute(mut self, ctx: &ExecutionContext<'_>) -> Result<DeviceSliceHostView<T>> {
        let buf = self
            .buf
            .take()
            .expect("AcquireDeviceSliceOp::execute called twice");
        let mut data = vec![T::default(); buf.len()];
        // .wait() here is correctness-required: the host can't deref
        // a stale Vec. The underlying ReadOp uses CL_TRUE so the
        // driver blocks internally rather than enqueue + event.wait().
        buf.read(ctx, &mut data).wait()?;
        Ok(DeviceSliceHostView { buf, data })
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

    fn execute(mut self, ctx: &ExecutionContext<'_>) -> Result<DeviceSlice<T>> {
        let view = self
            .view
            .take()
            .expect("ReleaseDeviceSliceOp::execute called twice");
        let DeviceSliceHostView { mut buf, data } = view;
        // .wait() here is correctness-required: the host `data` Vec
        // drops at end of scope; we can't release it until OpenCL is
        // done reading. Future optimisation: use the keep-alive
        // callback trick (same as transfer::upload) for a non-blocking
        // release.
        buf.write(ctx, &data).wait()?;
        Ok(buf)
    }
}
