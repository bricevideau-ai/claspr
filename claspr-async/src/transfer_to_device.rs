//! [`transfer_to_device`] — explicit `DeviceSlice<T>` migration as a
//! Tier 2 chain stage.
//!
//! Wraps `clEnqueueMigrateMemObjects` on the target device's default
//! OOO queue. Non-blocking — the migration is enqueued as a queue
//! command, the chain's `deps` flow through it, downstream stages
//! wait on the migrate event.
//!
//! ## What this actually does
//!
//! For two devices sharing a single `cl_context`, the OpenCL runtime
//! may or may not perform real data movement:
//!
//! - Single iGPU with sub-devices, or any topology where all devices
//!   share physical memory: typically a no-op.
//! - Two dGPUs in one context: real `cl_mem` migration between
//!   device memories.
//!
//! Either way, the migrate is a queue command (not a host-side wait),
//! so the chain stays non-blocking. The cost (if any) is in queue-side
//! timing.
//!
//! Cross-*context* transfer (separate `cl_context`s) is **not** what
//! this primitive does — those go through host bounce
//! (`download → upload`) which already composes via the existing
//! [`crate::download`] + [`crate::upload`] combinators.
//!
//! ## Idiom
//!
//! ```ignore
//! .and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(1)))
//! .and_then_with_context(|ec, buf|
//!     kernels.foo([N], buf).on_device(ec.device_at(1)))
//! ```
//!
//! Together, `transfer_to_device` + `.on_device` are the two primitives
//! that decompose cross-device pipelines into the OpenCL operations
//! they actually map to: migrate (data residency) + enqueue-on-queue
//! (kernel device).

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use claspr::{Device, DeviceSlice, Result};

/// Combinator built by [`transfer_to_device`].
pub struct TransferToDevice<T> {
    buf: Option<DeviceSlice<T>>,
    device: Device,
}

/// Enqueue a `clEnqueueMigrateMemObjects` for `buf` on `device`'s
/// default OOO queue. Returns a Tier 2 op whose `Output` is the
/// (now-migrated) buffer.
///
/// Non-blocking: the migrate is queued, the chain's upstream events
/// thread through as wait-list, downstream stages wait on the migrate
/// event. The actual cost depends on device topology (see module
/// docs).
///
/// See [`crate::OnDevice`] for the matching per-op routing combinator
/// kernels need after the buffer's been migrated.
pub fn transfer_to_device<T>(buf: DeviceSlice<T>, device: &Device) -> TransferToDevice<T>
where
    T: Send + 'static,
{
    TransferToDevice {
        buf: Some(buf),
        device: device.clone(),
    }
}

impl<T> DeviceOperation for TransferToDevice<T>
where
    T: Send + 'static,
{
    type Output = DeviceSlice<T>;

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(DeviceSlice<T>, Deps)> {
        let buf = self
            .buf
            .take()
            .expect("TransferToDevice::execute called twice — internal claspr-async bug");
        // Resolve the target device's default OOO queue. Lazy +
        // cached on Context; the chain terminal's
        // flush_all_outoforder_queues will push it.
        let target_q = ec.context().default_outoforder_queue(&self.device)?;
        // Enqueue migrate with upstream deps as the wait-list.
        let event = buf
            .migrate()
            .after_all(deps_as_events(&deps))
            .submit(&*target_q)?;
        Ok((buf, vec![wrap_event(event)]))
    }
}
