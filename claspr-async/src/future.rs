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
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskCx, Poll};

/// Future returned by [`DeviceOperation::run`]. Resolves to
/// `Result<T>` when the chain's commands have all completed on the
/// device (or immediately, with an error, if the chain failed to
/// submit).
pub enum ChainFuture<T> {
    /// Chain failed during setup or `execute`. The error surfaces on
    /// the first `poll`. No workers ran (failure was before spawn),
    /// so there's no host-error slot to drain — the carried `Error`
    /// is already the canonical one.
    Errored(Option<Error>),
    /// Chain submitted successfully; waiting for the trailing marker
    /// event to complete. Carries an Arc clone of the chain's
    /// host-error slot so that on poll-time Err we surface the rich
    /// variant stashed by any `and_then_host` worker, mirroring the
    /// sync terminal's contract.
    Running {
        output: Option<T>,
        event_future: EventFuture,
        host_error: Arc<Mutex<Option<Error>>>,
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
                host_error,
            } => match Pin::new(event_future).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Err(e)) => {
                    // Prefer the stashed host error (from an
                    // `and_then_host` worker) over the CL cascade
                    // — same shape as the sync terminal.
                    let resolved = host_error.lock().unwrap().take().unwrap_or(e);
                    Poll::Ready(Err(resolved))
                }
                Poll::Ready(Ok(())) => {
                    // Even on a "successful" marker, a worker may
                    // have stashed an error the marker didn't
                    // propagate. pocl's `clEnqueueMarkerWithWaitList`
                    // does NOT cascade negative status from a user
                    // event in its wait list — the marker reports
                    // `CL_COMPLETE` while the chain has genuinely
                    // failed. A non-empty slot is itself the
                    // failure signal; surface it.
                    if let Some(rust_err) = host_error.lock().unwrap().take() {
                        return Poll::Ready(Err(rust_err));
                    }
                    Poll::Ready(Ok(output
                        .take()
                        .expect("ChainFuture polled after Ready (Running)")))
                }
            },
        }
    }
}

/// Crate-internal worker: build a [`ChainFuture`] from a chain and a
/// context. Called by [`DeviceOperation::run`] (added in `op.rs`).
///
/// Synchronous-error paths (everything that returns
/// `ChainFuture::Errored` from this function) invalidate the
/// context's cached out-of-order queue, mirroring the sync
/// terminal's contract.
///
/// TODO: poll-time errors (the awaited marker completes with a
/// negative status) don't yet invalidate. That requires plumbing
/// the `Context` + `Device` handles into [`ChainFuture::Running`]
/// so the future can call the invalidator on its own error branch.
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
    //    value immediately along with the events the chain produced.
    let ec = ExecutionContext::new(context, device.clone(), queue.raw());
    // Grab an Arc clone of the host-error slot before the EC drops.
    // Workers spawned by `and_then_host` populate it from their own
    // threads; this clone lets the future read it after the marker
    // resolves.
    let host_error = ec.host_error_slot();
    let (output, chain_evts) = match chain.execute(&ec, Vec::new()) {
        Ok(p) => p,
        Err(e) => {
            // A previously-spawned and_then_host worker may have
            // stashed before execute returned (see the parallel
            // sibling case in `run_chain_sync`). Prefer the rich
            // variant.
            let actual = host_error.lock().unwrap().take().unwrap_or(e);
            drop(queue);
            context.invalidate_default_outoforder_queue(&device);
            return ChainFuture::Errored(Some(actual));
        }
    };
    // 3. Enqueue a marker that completes after every event the chain
    //    produced. Precise wait-list — we don't penalise other work
    //    that may be sharing this OOO queue.
    //
    // SAFETY: each `cl_event` in `chain_evts` is held alive by the
    // `Arc<Event>` wrappers for the duration of this call; the marker
    // enqueue retains them internally before we drop the wrappers.
    let wait_list: Vec<opencl3::types::cl_event> =
        chain_evts.iter().map(|d| d.as_ref().get()).collect();
    let marker = match unsafe { queue.raw().enqueue_marker_with_wait_list(&wait_list) } {
        Ok(ev) => ev,
        Err(code) => {
            drop(chain_evts);
            drop(queue);
            context.invalidate_default_outoforder_queue(&device);
            return ChainFuture::Errored(Some(Error::OpenCl(code)));
        }
    };
    drop(chain_evts);
    // 3a. clFlush — push the queue to the device without blocking.
    //     The async terminal otherwise has no sync point: pocl
    //     happens to push commands eagerly, but rusticl is spec-strict
    //     and keeps the marker in `CL_QUEUED` forever without an
    //     explicit flush, so the CL_COMPLETE callback never fires
    //     and the future deadlocks. clFlush returns immediately;
    //     completion still happens asynchronously via the callback.
    if let Err(e) = queue.raw().flush() {
        drop(queue);
        context.invalidate_default_outoforder_queue(&device);
        return ChainFuture::Errored(Some(Error::OpenCl(e)));
    }
    // 4. Wrap in a Future that polls the marker via
    //    clSetEventCallback (the EventFuture machinery from claspr).
    ChainFuture::Running {
        output: Some(output),
        event_future: marker.into_future(),
        host_error,
    }
}
