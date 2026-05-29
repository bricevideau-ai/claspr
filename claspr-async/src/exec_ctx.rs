//! [`ExecutionContext`] — what every [`DeviceOperation`]'s `execute`
//! method receives.
//!
//! Carries the [`Context`], the current [`Device`], and a borrowed
//! [`CommandQueue`] to enqueue on. Implements [`Launcher`] so any
//! existing Tier 1 op (e.g. proc-macro-generated `kernels.foo(...)`)
//! composes directly inside a chain:
//!
//! ```ignore
//! with_context(move |ctx| {
//!     // ctx: &ExecutionContext implements Launcher
//!     kernels.foo(ctx, [N], &buf).wait()?;  // Tier 1 inside Tier 2
//!     Ok(buf)
//! })
//! ```
//!
//! [`DeviceOperation`]: crate::op::DeviceOperation
//! [`Context`]: claspr::Context
//! [`Device`]: claspr::Device
//! [`CommandQueue`]: opencl3::command_queue::CommandQueue
//! [`Launcher`]: claspr::Launcher

use claspr::{CommandQueue, Context, Device, Error, Launcher};
use std::sync::{Arc, Mutex};

/// Execution-time environment for a [`DeviceOperation`].
///
/// Built by [`DeviceOperation::sync`] (and later, by the async terminal
/// in Phase 3.4); op authors don't construct this directly. The
/// `'ctx` lifetime is the lifetime of the borrow into the parent
/// [`Context`] and its per-device default OOO queue.
///
/// [`DeviceOperation`]: crate::op::DeviceOperation
/// [`DeviceOperation::sync`]: crate::op::DeviceOperation::sync
pub struct ExecutionContext<'ctx> {
    context: &'ctx Context,
    device: Device,
    cl_queue: &'ctx CommandQueue,
    /// Slot that `and_then_host` workers stash their failing
    /// [`Error`] into before signalling `clSetUserEventStatus(_, -1)`.
    /// Terminals read it after the marker event resolves with Err
    /// and prefer the rich variant over the CL cascade. `Arc<Mutex<_>>`
    /// because workers run on per-call threads. First-writer-wins
    /// when multiple `and_then_host`s in a bundle/fan-out fail
    /// concurrently — subsequent writers leave the slot alone.
    host_error: Arc<Mutex<Option<Error>>>,
}

impl<'ctx> ExecutionContext<'ctx> {
    /// Construct an `ExecutionContext` bound to `device`'s default
    /// out-of-order queue from `context`. Crate-internal — terminals
    /// in [`crate::op`] call this.
    pub(crate) fn new(
        context: &'ctx Context,
        device: Device,
        cl_queue: &'ctx CommandQueue,
    ) -> Self {
        ExecutionContext {
            context,
            device,
            cl_queue,
            host_error: Arc::new(Mutex::new(None)),
        }
    }

    /// Cheap `Arc` clone of the host-error slot for an `and_then_host`
    /// worker to carry across `thread::spawn`. Workers populate it
    /// (first-writer-wins) before signalling negative user-event
    /// status; terminals drain it via [`take_host_error`](Self::take_host_error).
    pub(crate) fn host_error_slot(&self) -> Arc<Mutex<Option<Error>>> {
        Arc::clone(&self.host_error)
    }

    /// Take the stashed host error (if any). Crate-internal — called
    /// by terminals (`sync` / `run`'s poll) after the chain's events
    /// resolve with Err, so the original Rust variant surfaces instead
    /// of the `Error::OpenCl(-1)` cascade from the user-event signal.
    pub(crate) fn take_host_error(&self) -> Option<Error> {
        self.host_error.lock().unwrap().take()
    }

    /// The [`Context`] this op-chain is running against.
    pub fn context(&self) -> &Context {
        self.context
    }

    /// The [`Device`] this op currently targets. Cross-device
    /// re-routing (`.on_device(&other)`) lands in a later phase;
    /// for now the device is fixed for the whole chain.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// Letting `ExecutionContext` itself act as a Launcher means a Tier 1
// op inside a Tier 2 closure (e.g. `kernels.foo(ctx, ...)`) routes
// transparently through the chain's queue — no need for the user to
// dig out the queue handle.
impl<'ctx> Launcher for ExecutionContext<'ctx> {
    fn cl_queue(&self) -> &CommandQueue {
        self.cl_queue
    }

    fn context(&self) -> &Context {
        self.context
    }
}
