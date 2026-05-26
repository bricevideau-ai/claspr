//! [`AndThenHost`] — in-queue host work between two device ops.
//!
//! `.and_then(|out| op)` chains by passing the source's output to a
//! closure that returns *another [`DeviceOperation`]*. `.and_then_host(|out| Result<U>)`
//! chains by passing the output to a closure that returns a *host value*
//! — useful for reductions, format conversions, validation, etc. that
//! sit between two GPU stages without needing the GPU itself.
//!
//! Compared to splitting the chain (`.await; do_host_thing(); .run().await`),
//! this keeps the whole pipeline as a single composable expression —
//! the host work is just another node in the chain.
//!
//! ## Example
//!
//! ```ignore
//! use claspr_async::{DeviceOperation, DeviceOperationHostExt, value};
//!
//! let summary = upload_data()
//!     .and_then(|buf| run_kernel(buf))
//!     .and_then(|buf| download(buf))
//!     .and_then_host(|host_vec| Ok(host_vec.iter().sum::<u32>()))
//!     .and_then(|total| value(total))
//!     .sync(&ctx)?;
//! ```
//!
//! The trait is implemented as an extension on [`DeviceOperation`]
//! (via [`DeviceOperationHostExt`]) so it lives in its own module
//! without bloating the core trait.

use crate::exec_ctx::ExecutionContext;
use crate::op::DeviceOperation;
use claspr::Result;

/// Combinator built by [`DeviceOperationHostExt::and_then_host`].
pub struct AndThenHost<S, F> {
    source: S,
    f: Option<F>,
}

/// Extension trait adding [`and_then_host`](Self::and_then_host) to
/// every [`DeviceOperation`].
pub trait DeviceOperationHostExt: DeviceOperation {
    /// Run `f` on this op's output, returning a `Result<U>` of plain
    /// host data. Useful for reductions / format conversions /
    /// validation between two device stages — the closure runs on the
    /// host thread driving the chain (not inside an OpenCL command
    /// queue), so it can do anything Rust can.
    ///
    /// The closure must be [`FnOnce`] + [`Send`].
    fn and_then_host<F, U>(self, f: F) -> AndThenHost<Self, F>
    where
        F: FnOnce(Self::Output) -> Result<U> + Send,
        U: Send,
    {
        AndThenHost {
            source: self,
            f: Some(f),
        }
    }
}

impl<S: DeviceOperation> DeviceOperationHostExt for S {}

impl<S, F, U> DeviceOperation for AndThenHost<S, F>
where
    S: DeviceOperation,
    F: FnOnce(S::Output) -> Result<U> + Send,
    U: Send,
{
    type Output = U;

    fn execute(mut self, ctx: &ExecutionContext<'_>) -> Result<U> {
        let prior = self.source.execute(ctx)?;
        (self
            .f
            .take()
            .expect("AndThenHost::execute called twice — internal claspr-async bug"))(prior)
    }
}
