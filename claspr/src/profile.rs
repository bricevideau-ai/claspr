//! Tier 2 `.profiled(|info| ...)` — wall-clock timing for an arbitrary
//! sub-chain.
//!
//! Where Tier 1's [`LaunchOp::profiled`] times a single kernel launch,
//! this combinator times *whatever the source op enqueued*, whether
//! that's one kernel, a [`Bundle`], a [`FanOut`], or a nested chain.
//! Under the hood it enqueues an `clEnqueueMarkerWithWaitList` after
//! the source op runs and registers the user callback on the marker
//! using the same FFI shim Tier 1 uses
//! ([`crate::register_profiling_callback`]).
//!
//! Like Tier 1, requires the chain's OOO queue to have
//! `CL_QUEUE_PROFILING_ENABLE` — build the [`Context`](crate::Context)
//! with [`.profiling(true)`](crate::context::ContextBuilder::profiling).
//! Otherwise the closure receives `Err(Error::ProfilingDisabled)` (the
//! source op still executes, and the chain continues — profiling is a
//! side-effect on the host, not data flow).
//!
//! ## Example
//!
//! ```ignore
//! use claspr_async::{DeviceOperation, DeviceOperationProfileExt, value};
//!
//! let (tx, rx) = std::sync::mpsc::channel();
//! kernels.foo_op(/* ... */)
//!     .and_then(|buf| kernels.bar_op(buf))
//!     .profiled(move |info_result| {
//!         tx.send(info_result).unwrap();
//!     })
//!     .sync(&ctx)?;
//!
//! let info = rx.recv().unwrap()?;
//! println!("foo + bar took {:?}", info.duration());
//! ```
//!
//! [`Bundle`]: crate::Bundle2
//! [`FanOut`]: crate::FanOut
//! [`LaunchOp::profiled`]: crate::LaunchOp::profiled

use crate::exec_ctx::ExecutionContext;
use crate::device_op::{Deps, DeviceOperation, wrap_event};
use crate::{
    CL_QUEUE_PROFILING_ENABLE, Error, Launcher, ProfilingInfo, Result, register_profiling_callback,
};

/// Combinator built by [`DeviceOperationProfileExt::profiled`].
pub struct Profiled<S, F> {
    source: S,
    cb: Option<F>,
}

/// Extension trait adding [`profiled`](Self::profiled) to every
/// [`DeviceOperation`].
pub trait DeviceOperationProfileExt: DeviceOperation {
    /// Register `cb` to receive the wall-clock [`ProfilingInfo`] for
    /// everything `self` enqueued onto the chain's queue. The closure
    /// fires on an OpenCL callback thread when the marker event
    /// completes; panics inside it are caught and dropped (FFI
    /// safety, same as Tier 1).
    fn profiled<F>(self, cb: F) -> Profiled<Self, F>
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        Profiled {
            source: self,
            cb: Some(cb),
        }
    }
}

impl<S: DeviceOperation> DeviceOperationProfileExt for S {}

impl<S, F> DeviceOperation for Profiled<S, F>
where
    S: DeviceOperation,
    F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
{
    // Profiling is a side-effect on the host; the chain's data flow is
    // unchanged.
    type Output = S::Output;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(S::Output, Deps)> {
        let (out, source_evts) = self.source.execute(ctx, deps)?;
        // Same up-front check as Tier 1: the queue needs profiling
        // enabled before we waste a marker + callback registration.
        if (ctx.cl_queue().properties()? & CL_QUEUE_PROFILING_ENABLE) == 0 {
            return Err(Error::ProfilingDisabled);
        }
        // The marker waits for the source op's events, so the
        // timestamps reflect the source op's wall-clock duration
        // (start of first command to end of last).
        let wait_list: Vec<opencl3::types::cl_event> =
            source_evts.iter().map(|d| d.as_ref().get()).collect();
        let marker = unsafe { ctx.cl_queue().enqueue_marker_with_wait_list(&wait_list) }
            .map_err(Error::OpenCl)?;
        // source_evts keeps the underlying cl_events alive across the
        // enqueue; safe to drop after.
        drop(source_evts);
        register_profiling_callback(
            &marker,
            Box::new(
                self.cb
                    .take()
                    .expect("Profiled::execute called twice — internal claspr-async bug"),
            ),
        )?;
        // The marker also becomes the source op's "completion event"
        // for downstream chaining — anything after the .profiled()
        // waits on the marker (which subsumes source's events).
        Ok((out, vec![wrap_event(marker)]))
    }
}
