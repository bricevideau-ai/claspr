//! Async terminal — [`DeviceOperation::run`] returns a [`ChainFuture`]
//! that resolves when every command the chain enqueued has finished.
//!
//! ## How it works
//!
//! [`run`](DeviceOperation::run) builds an [`ExecutionContext`],
//! calls `execute`, then enqueues an OpenCL marker via
//! [`clEnqueueMarkerWithWaitList`] on the chain's out-of-order queue.
//! The marker completes after every previously-submitted command on
//! the same queue does. The returned [`ChainFuture`] wraps that marker
//! in a [`claspr::EventFuture`] — the existing Tier 1
//! `clSetEventCallback` machinery (with `catch_unwind` + `AtomicWaker`)
//! does the actual waker dispatch when the marker fires.
//!
//! The chain's host-side `Output` value is materialised eagerly by
//! `execute` (handles, `Vec`s, etc.); the future just gates *when*
//! the user gets to see it on whether the queue work is done. If a
//! chain ends with a blocking download op, `execute` will already
//! have waited, and the marker fires immediately — `.await` is then
//! roughly equivalent to `.sync()`. Non-blocking chains let `.await`
//! genuinely overlap with other host work.
//!
//! [`DeviceOperation::run`]: crate::op::DeviceOperation::run
//! [`ExecutionContext`]: crate::ExecutionContext
//! [`clEnqueueMarkerWithWaitList`]: https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#clEnqueueMarkerWithWaitList

use crate::exec_ctx::ExecutionContext;
use crate::op::DeviceOperation;
use claspr::{Context, Error, EventFuture, EventFutureExt, Result};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context as TaskCx, Poll};

/// Future returned by [`DeviceOperation::run`]. Resolves to
/// `Result<T>` when the chain's commands have all completed on the
/// device (or immediately, with an error, if the chain failed to
/// submit).
pub enum ChainFuture<T> {
    /// Chain failed during setup or `execute`. The error surfaces on
    /// the first `poll`.
    Errored(Option<Error>),
    /// Chain submitted successfully; waiting for the trailing marker
    /// event to complete.
    Running {
        output: Option<T>,
        event_future: EventFuture,
    },
}

// `T: Unpin` covers every realistic chain output (`Vec<u8>`,
// `DeviceSlice<T>`, `Arc<T>`, tuples of those, ...) and lets us pin-
// project via the cheap `Pin::get_mut`. If a user ever needs a `!Unpin`
// output, they can `Box::pin(chain.run(&ctx))`.
impl<T: Unpin> Future for ChainFuture<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut TaskCx<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        match this {
            ChainFuture::Errored(slot) => Poll::Ready(Err(slot
                .take()
                .expect("ChainFuture polled after Ready (Errored)"))),
            ChainFuture::Running {
                output,
                event_future,
            } => match Pin::new(event_future).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Ready(Ok(())) => Poll::Ready(Ok(output
                    .take()
                    .expect("ChainFuture polled after Ready (Running)"))),
            },
        }
    }
}

/// Crate-internal worker: build a [`ChainFuture`] from a chain and a
/// context. Called by [`DeviceOperation::run`] (added in `op.rs`).
pub(crate) fn run_chain<Op>(chain: Op, context: &Context) -> ChainFuture<Op::Output>
where
    Op: DeviceOperation,
{
    // 1. Pick the per-device default OOO queue.
    let device = context.device().clone();
    let queue = match context.default_outoforder_queue(&device) {
        Ok(q) => q,
        Err(e) => return ChainFuture::Errored(Some(e)),
    };
    // 2. Build the ExecutionContext and submit the chain. `execute`
    //    may enqueue many CL commands; it returns the host-side output
    //    value immediately.
    let ec = ExecutionContext::new(context, device, queue.raw());
    let output = match chain.execute(&ec) {
        Ok(o) => o,
        Err(e) => return ChainFuture::Errored(Some(e)),
    };
    // 3. Enqueue a marker that completes after everything submitted
    //    above. Empty wait-list means "wait for all prior commands on
    //    this queue" (CL §5.13).
    //
    // SAFETY: `enqueue_marker_with_wait_list` is unsafe only because
    // it takes raw `cl_event` slices; we pass an empty slice, so no
    // validation is needed on our side.
    let marker = match unsafe { queue.raw().enqueue_marker_with_wait_list(&[]) } {
        Ok(ev) => ev,
        Err(code) => return ChainFuture::Errored(Some(Error::OpenCl(code))),
    };
    // 4. Wrap in a Future that polls the marker via
    //    clSetEventCallback (the EventFuture machinery from claspr).
    ChainFuture::Running {
        output: Some(output),
        event_future: marker.into_future(),
    }
}
