//! Async bridge: the [`EventFuture`] standalone wrapper for any raw
//! [`Event`], plus [`LaunchOp`]'s [`IntoFuture`] impl (which resolves to
//! [`LaunchFuture`]). The eager device-graph terminal future
//! (`DeviceChainFuture`) lives in [`eager`](crate::eager).
//!
//! Both go through `clSetEventCallback(CL_COMPLETE, ...)` to flip a
//! `done` flag and wake the future's registered waker — the same
//! pattern as `NVlabs/cuda-oxide`'s `cuda-async` over `cuLaunchHostFunc`.
//!
//! Gated on the `async-events` cargo feature so users who don't want
//! an async runtime don't pull `futures` as a transitive dep. Without
//! the feature, [`LaunchOp::wait`](crate::op::LaunchOp::wait) and
//! [`LaunchOp::submit`](crate::op::LaunchOp::submit) still work — only
//! `.await` is gated.
//!
//! # Examples
//!
//! Await a kernel directly:
//!
//! ```ignore
//! kernels.collatz_kernel(&ctx, [N], &buf).await?;
//! ```
//!
//! Wait on a raw event (e.g. one returned from [`submit`](crate::op::LaunchOp::submit)):
//!
//! ```ignore
//! use claspr::EventFutureExt;
//!
//! let event = kernels.foo(&q_a, ..., &buf).submit()?;
//! event.into_future().await?;
//! ```

use crate::error::{Error, Result};
use crate::launch::KernelArgs;
use crate::op::LaunchOp;
use futures::task::AtomicWaker;
use opencl3::event::{CL_COMPLETE, Event, set_event_callback};
use opencl3::types::{cl_event, cl_int};
use std::ffi::c_void;
use std::future::{Future, IntoFuture};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

// ── Shared state + completion callback ──────────────────────────────

/// Shared between the future and the OpenCL completion callback.
struct State {
    done: AtomicBool,
    waker: AtomicWaker,
}

extern "C" fn completion_thunk(_event: cl_event, _status: cl_int, user_data: *mut c_void) {
    // Unwinding through FFI is UB. `catch_unwind` guards against a
    // panic in the waker (unlikely, but possible if the user's
    // executor panics from `wake_by_ref`).
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `user_data` was produced by `Arc::into_raw` when the
        // future first polled and registered itself. The OpenCL spec
        // guarantees this callback fires at most once per registration;
        // we take ownership of the Arc back here and drop on scope exit.
        let state = unsafe { Arc::from_raw(user_data as *const State) };
        state.done.store(true, Ordering::Release);
        state.waker.wake();
    }));
}

/// Register the completion callback for `event` against the shared
/// `state`. On success the runtime owns an `Arc::clone(&state)` until
/// the callback fires; on failure the would-be-leaked Arc is reclaimed
/// and the OpenCL error is returned.
fn register_completion(event: cl_event, state: &Arc<State>) -> Result<()> {
    let user_data = Arc::into_raw(Arc::clone(state)) as *mut c_void;
    let res = set_event_callback(event, CL_COMPLETE, completion_thunk, user_data);
    if let Err(code) = res {
        // SAFETY: registration failed, so OpenCL never took ownership
        // of `user_data` — reclaim and drop here so the Arc count
        // matches.
        unsafe {
            let _ = Arc::from_raw(user_data as *const State);
        }
        return Err(Error::OpenCl(opencl3::error_codes::ClError(code)));
    }
    Ok(())
}

// ── EventFuture ─────────────────────────────────────────────────────

/// A `Future` that resolves when an OpenCL [`Event`] completes.
/// Constructed via [`EventFutureExt::into_future`] on any event —
/// most commonly one returned from [`LaunchOp::submit`].
///
/// On first poll, registers a `CL_COMPLETE` callback that flips the
/// future's `done` flag and wakes the waker. Subsequent polls either
/// return `Ready` immediately (if done) or re-register the waker.
///
/// [`LaunchOp::submit`]: crate::op::LaunchOp::submit
pub struct EventFuture {
    _event: Event, // kept alive for the duration of the future
    state: Arc<State>,
    registered: bool,
}

impl EventFuture {
    pub(crate) fn new(event: Event) -> Self {
        EventFuture {
            _event: event,
            state: Arc::new(State {
                done: AtomicBool::new(false),
                waker: AtomicWaker::new(),
            }),
            registered: false,
        }
    }
}

impl Future for EventFuture {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Fast path: already done.
        if self.state.done.load(Ordering::Acquire) {
            return Poll::Ready(Ok(()));
        }
        // Register / refresh waker first so we don't miss a callback
        // that fires between our `done` check and the registration.
        self.state.waker.register(cx.waker());
        if !self.registered {
            if let Err(e) = register_completion(self._event.get(), &self.state) {
                return Poll::Ready(Err(e));
            }
            self.registered = true;
        }
        // Re-check `done` in case the callback fired between the
        // register/atomic-set and this point.
        if self.state.done.load(Ordering::Acquire) {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }
}

/// Extension trait that adds `into_future()` to opencl3's [`Event`].
/// Requires the `async-events` cargo feature.
pub trait EventFutureExt {
    /// Convert this event into a [`Future`] that completes when the
    /// event's underlying command finishes.
    fn into_future(self) -> EventFuture;
}

impl EventFutureExt for Event {
    fn into_future(self) -> EventFuture {
        EventFuture::new(self)
    }
}

// ── LaunchOp IntoFuture ─────────────────────────────────────────────

/// Future returned by `.await` on a [`LaunchOp`]. Enqueues the kernel
/// eagerly at `into_future()` time; an enqueue error surfaces on the
/// first `poll`.
pub enum LaunchFuture {
    /// Enqueue failed — return the error on first poll.
    Errored(Option<Error>),
    /// Enqueued successfully — delegate to [`EventFuture`].
    Running(EventFuture),
}

impl Future for LaunchFuture {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match &mut *self {
            LaunchFuture::Errored(slot) => {
                Poll::Ready(Err(slot.take().expect("LaunchFuture polled after Ready")))
            }
            // `EventFuture: Unpin` (no self-referential state), so
            // pin-projection through the enum variant is just `Pin::new`.
            LaunchFuture::Running(ef) => Pin::new(ef).poll(cx),
        }
    }
}

// ── Buffer-op IntoFuture impls removed ──────────────────────────────
//
// `WriteOp` / `ReadOp` / `CopyOp` (and friends) used to implement
// `IntoFuture`, letting users write `buf.write(&ctx, &data).await`.
// With the launcher-at-terminal API (`buf.write(&data).wait(&ctx)?`),
// `IntoFuture` no longer has a place to thread `&launcher` —
// `into_future(self) -> Self::IntoFuture` takes nothing extra. The
// equivalent path is now `op.submit(&ctx)?.await` (`Event: IntoFuture`
// is unchanged). The impls were never exercised in tests or examples.

// ── LaunchOp IntoFuture ─────────────────────────────────────────────

impl<'l, A: KernelArgs> IntoFuture for LaunchOp<'l, A> {
    type Output = Result<()>;
    type IntoFuture = LaunchFuture;

    fn into_future(self) -> LaunchFuture {
        let queue = self.queue;
        match self.into_event() {
            Ok(event) => {
                // Flush before parking on the callback: registering a
                // CL_COMPLETE callback does NOT submit the command, and
                // a lazy runtime (rusticl) never executes an unflushed
                // enqueue — the future would deadlock. Same reasoning
                // as the tier2 run() terminal's
                // flush_all_outoforder_queues; pocl flushes eagerly so
                // this is a no-op there.
                if let Err(e) = queue.flush() {
                    return LaunchFuture::Errored(Some(e.into()));
                }
                LaunchFuture::Running(EventFuture::new(event))
            }
            Err(e) => LaunchFuture::Errored(Some(e)),
        }
    }
}
