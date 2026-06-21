//! [`OnDevice`] — per-op device routing combinator.
//!
//! Wrap any [`DeviceOperation`] in `.on_device(&Device)` to make its
//! `execute` run against the named device's default OOO queue instead
//! of the chain's primary queue. Use for kernel launches on a
//! non-default device in a multi-device chain.
//!
//! Pair with [`transfer_to_device`](crate::transfer_to_device()) to
//! explicitly migrate buffers between devices — that primitive
//! handles the `cl_mem` residency; this one handles where the kernel
//! actually enqueues.
//!
//! ## Why a separate combinator (rather than per-buffer device tagging)
//!
//! Production `DeviceSlice<T>` is per-context (one `cl_mem` valid on
//! any device of the context), not per-device. Tagging buffers would
//! be a bigger surface change. Per-op routing matches OpenCL's
//! actual decomposition: kernel launches pick their device from the
//! queue they're enqueued on.
//!
//! ## Idiom: resolve devices from `ec`, not external captures
//!
//! ```ignore
//! upload!(input).and_then_with_context(|ec, buf|
//!     kernels.foo([N], buf).on_device(ec.device_at(1)))
//! ```
//!
//! Pulling the device handle from `ec.context().devices()` (or its
//! shortcut `ec.device_at(i)`) keeps the chain portable across
//! contexts and avoids the trap of assuming "upload landed buf on
//! `devs[0]`" (it landed on the context's *default* device, which
//! may be either).

use crate::exec_ctx::ExecutionContext;
use crate::device_op::{Deps, DeviceOperation};
use crate::{Device, Result};

/// Combinator built by [`DeviceOperation::on_device`].
pub struct OnDevice<S> {
    source: S,
    device: Device,
}

impl<S: DeviceOperation> OnDevice<S> {
    pub(crate) fn new(source: S, device: Device) -> Self {
        OnDevice { source, device }
    }
}

impl<S> DeviceOperation for OnDevice<S>
where
    S: DeviceOperation,
{
    type Output = S::Output;

    fn execute(self, parent: &ExecutionContext<'_>, deps: Deps) -> Result<(S::Output, Deps)> {
        // Resolve the target queue from the chain's running context.
        // Lazy via Context's cache (and cached in turn — so the
        // chain's terminal's flush_all_outoforder_queues will pick it
        // up and push it). Bubbles up any allocation error.
        let target_q = parent.context().default_outoforder_queue(&self.device)?;
        // Build a sibling EC: same context + same host-error slot,
        // different device + queue. The Arc<Queue> lives on this
        // stack frame; `target_q.raw()` borrows for the duration of
        // the inner execute().
        let child = ExecutionContext::with_host_error_slot(
            parent.context(),
            self.device.clone(),
            target_q.raw(),
            parent.host_error_slot(),
        );
        // Inner op runs against the child EC, threading deps through.
        // Its events are valid in any queue of the same context (per
        // OpenCL's shared-context event semantics), so downstream
        // stages on the parent's queue can wait on them cross-device.
        self.source.execute(&child, deps)
    }
}
