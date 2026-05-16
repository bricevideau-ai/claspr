//! Async bridge: [`EventFuture`] turns an OpenCL [`Event`] into a
//! `Future` that completes when the event's command finishes.
//!
//! Gated on the `async-events` cargo feature so users who don't
//! want an async runtime don't pull `futures` as a transitive dep.
//!
//! The bridge uses `clSetEventCallback(CL_COMPLETE, ...)` to set
//! the future's `done` flag and wake the registered waker —
//! mirrors the pattern in `NVlabs/cuda-oxide`'s `cuda-async` crate
//! (which uses `cuLaunchHostFunc` + AtomicWaker the same way).
//!
//! # Example
//!
//! ```ignore
//! use claspr::{Queue, OutOfOrder, EventFutureExt};
//!
//! let q = Queue::<OutOfOrder>::new(&ctx)?;
//! let event = q.launch_with_deps((), &kernel, [n], (&buf,))?;
//! event.into_future().await?;  // requires async-events feature + an executor
//! ```

use crate::error::{Error, Result};
use futures::task::AtomicWaker;
use opencl3::event::{CL_COMPLETE, Event, set_event_callback};
use opencl3::types::{cl_event, cl_int};
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

/// Internal state shared between the future and the OpenCL callback.
struct State {
    done: AtomicBool,
    waker: AtomicWaker,
}

/// A `Future` that resolves when the underlying OpenCL event
/// completes. Constructed via [`EventFutureExt::into_future`].
///
/// On first poll, registers a `CL_COMPLETE` callback on the event
/// that flips the future's `done` flag and wakes the registered
/// waker. Subsequent polls either return `Ready` immediately (if
/// done) or re-register the new waker and return `Pending`.
pub struct EventFuture {
    _event: Event, // kept alive for the duration of the future
    state: Arc<State>,
    registered: bool,
}

impl EventFuture {
    fn new(event: Event) -> Self {
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
            // Hand a strong Arc to the callback via `user_data`.
            // The callback consumes it back via `Arc::from_raw`.
            let user_data = Arc::into_raw(Arc::clone(&self.state)) as *mut c_void;
            // `event` is alive (we hold it as `_event`); `callback`
            // matches the expected `extern "C"` signature;
            // `user_data` is a valid `Arc<State>` pointer.
            let res = set_event_callback(self._event.get(), CL_COMPLETE, callback, user_data);
            if let Err(code) = res {
                // Reclaim the leaked Arc we just put into user_data.
                // SAFETY: callback wasn't registered, so user_data
                // is still uniquely ours.
                unsafe {
                    let _ = Arc::from_raw(user_data as *const State);
                };
                return Poll::Ready(Err(Error::OpenCl(opencl3::error_codes::ClError(code))));
            }
            self.registered = true;
        }
        // Re-check done in case the callback fired between the
        // register/atomic-set and our check above.
        if self.state.done.load(Ordering::Acquire) {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }
}

/// OpenCL completion callback. Stores `done = true` and wakes the
/// registered waker. The Arc<State> is reclaimed and dropped here.
extern "C" fn callback(_event: cl_event, _status: cl_int, user_data: *mut c_void) {
    // SAFETY: user_data was produced by `Arc::into_raw` in
    // `poll`. The OpenCL spec guarantees this callback fires at
    // most once per registration; we take ownership back and drop
    // the Arc when this function returns.
    let state = unsafe { Arc::from_raw(user_data as *const State) };
    state.done.store(true, Ordering::Release);
    state.waker.wake();
}

/// Extension trait that adds `into_future()` to opencl3's
/// `Event`. Requires the `async-events` cargo feature.
pub trait EventFutureExt {
    /// Convert this event into a [`Future`] that completes when
    /// the event's command finishes.
    fn into_future(self) -> EventFuture;
}

impl EventFutureExt for Event {
    fn into_future(self) -> EventFuture {
        EventFuture::new(self)
    }
}
