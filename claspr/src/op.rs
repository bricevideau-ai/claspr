//! Tier 1 launch builder — [`LaunchOp`] and its terminals.
//!
//! Returned by the proc-macro / build-script-emitted launch methods
//! per `#[claspr::kernel]`. Captures everything needed to enqueue the
//! kernel; defers the actual `clEnqueueNDRangeKernel` until the user
//! picks a terminal:
//!
//! - [`LaunchOp::wait`] — sync, blocks on completion.
//! - [`LaunchOp::submit`] — non-blocking; returns the [`Event`] so the
//!   user can chain across queues via [`LaunchOp::after`].
//! - `.await` (via [`IntoFuture`]) — async, completes via the OpenCL
//!   `CL_COMPLETE` event callback. Requires the `async-events` cargo
//!   feature.
//!
//! Modifiers (chain before the terminal):
//!
//! - [`LaunchOp::after`] — wait for a previously [`submit`](LaunchOp::submit)ted
//!   event on a different queue before this kernel starts.
//! - [`LaunchOp::profiled`] — register a completion callback that
//!   receives the kernel's timestamp set. Requires the queue to have
//!   been built with profiling enabled.

use crate::error::{Error, Result};
use crate::launch::{IntoLaunchSpec, KernelArgs, LaunchSpec};
use crate::queue::Launcher;
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::event::{CL_COMPLETE, Event, retain_event, set_event_callback};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::types::{cl_event, cl_int};
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

// ── Cross-context event check ───────────────────────────────────────

/// Panic if `event` was produced on a different `cl_context` than
/// `queue`'s. OpenCL rejects cross-context event deps with
/// `CL_INVALID_CONTEXT` at enqueue time; checking up-front turns that
/// cryptic CL error into a clear Rust panic at the call site that
/// passed the foreign event.
///
/// `call_site` is included in the panic message (e.g. `"LaunchOp::after"`
/// or `"FillU32Op::submit"`) so users can find the offending
/// `.after(event)` call quickly.
///
/// Cheap — two `clGet*Info` calls (one per side) plus a pointer
/// compare. The check skips silently when either query returns an
/// error (e.g. event already released); the CL runtime then surfaces
/// the underlying error.
pub fn assert_same_context(event: &Event, queue: &CommandQueue, call_site: &str) {
    if let (Ok(event_ctx), Ok(queue_ctx)) = (event.context(), queue.context())
        && event_ctx != queue_ctx
    {
        panic!(
            "{call_site}: event was produced on a different Context than the launcher's queue — \
             OpenCL rejects cross-context event deps (CL_INVALID_CONTEXT). \
             Both sides must use the same Context.",
        );
    }
}

// ── ProfilingInfo ────────────────────────────────────────────────────

/// The four OpenCL command-event timestamps, in device nanoseconds.
///
/// The reference epoch is implementation-defined per CL §5.14, so only
/// deltas are meaningful. The four points correspond to
/// `CL_PROFILING_COMMAND_QUEUED`, `_SUBMIT`, `_START`, `_END`.
///
/// Pass a closure to [`LaunchOp::profiled`] to receive this when the
/// kernel completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfilingInfo {
    /// Device timestamp when the command was queued on the host.
    pub queued: u64,
    /// Device timestamp when the runtime submitted the command to the device.
    pub submit: u64,
    /// Device timestamp when the device began executing the command.
    pub start: u64,
    /// Device timestamp when the device finished executing the command.
    pub end: u64,
}

impl ProfilingInfo {
    /// Wall-clock kernel runtime — `end - start` as a [`Duration`].
    pub fn duration(&self) -> std::time::Duration {
        std::time::Duration::from_nanos(self.end.saturating_sub(self.start))
    }
}

// ── LaunchOp ─────────────────────────────────────────────────────────

/// Type alias for the boxed profiling closure. Public so
/// [`register_profiling_callback`] callers (e.g. claspr-async's Tier
/// 2 profile combinator) can name the same shape.
pub type ProfileCb = Box<dyn FnOnce(Result<ProfilingInfo>) + Send + 'static>;

/// Lazy builder for one kernel launch. Constructed by the
/// proc-macro-generated launch methods; consumed by [`wait`](Self::wait),
/// [`submit`](Self::submit), or `.await`.
///
/// Modifiers — [`after`](Self::after), [`profiled`](Self::profiled) —
/// chain by-value; combine them in any order before the terminal.
pub struct LaunchOp<'l, A: KernelArgs> {
    queue: &'l CommandQueue,
    kernel: &'l Kernel,
    spec: LaunchSpec,
    args: A,
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'l, A: KernelArgs> LaunchOp<'l, A> {
    /// Construct a launch builder. Called by proc-macro-generated
    /// wrappers; user code uses `kernels.foo(...)` directly.
    pub fn new<L, S>(launcher: &'l L, kernel: &'l Kernel, spec: S, args: A) -> Self
    where
        L: Launcher + ?Sized,
        S: IntoLaunchSpec,
    {
        LaunchOp {
            queue: launcher.cl_queue(),
            kernel,
            spec: spec.into_launch_spec(),
            args,
            deps: Vec::new(),
            profile_cb: None,
        }
    }

    /// Add a cross-queue dependency: this kernel won't start until
    /// `event` (typically from another queue's [`submit`](Self::submit))
    /// completes. Chainable.
    ///
    /// **Panics** if `event` was produced on a different `Context`
    /// than `self`'s launcher's queue. OpenCL rejects cross-context
    /// event deps with `CL_INVALID_CONTEXT`; we surface the mismatch
    /// at `.after()` time with a clear message instead of as a
    /// cryptic CL error at enqueue.
    pub fn after(mut self, event: &Event) -> Self {
        assert_same_context(event, self.queue, "LaunchOp::after");
        self.deps.push(event.get());
        self
    }

    /// Add multiple wait-list events at once. Equivalent to calling
    /// [`after`](Self::after) for each in turn. Used by claspr-async
    /// Tier 2 to thread per-op dependency chains.
    ///
    /// Same cross-context panic as [`after`](Self::after).
    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        for event in events {
            assert_same_context(event, self.queue, "LaunchOp::after_all");
            self.deps.push(event.get());
        }
        self
    }

    /// Merge pre-collected dependency events and a pre-boxed profile
    /// callback into the builder. Used by proc-macro-generated typed
    /// `Op` types that hold their own builder state (so the user can
    /// call `.after(&ev)` / `.profiled(cb)` on the typed Op before
    /// it eventually delegates to LaunchOp at terminal time).
    ///
    /// Append-only for deps; replace for `profile_cb` only when one
    /// is supplied (None leaves the existing callback alone).
    pub fn with_state(mut self, deps: Vec<cl_event>, profile_cb: Option<ProfileCb>) -> Self {
        self.deps.extend(deps);
        if let Some(cb) = profile_cb {
            self.profile_cb = Some(cb);
        }
        self
    }

    /// Register a completion callback that receives the kernel's
    /// [`ProfilingInfo`] when execution finishes. The callback runs
    /// on the OpenCL runtime's callback thread — keep it short and
    /// avoid blocking calls.
    ///
    /// Panics inside the callback are caught and dropped (unwinding
    /// across the FFI boundary is UB); the `Result` the closure
    /// receives reflects only OpenCL-side failures querying the
    /// timestamps.
    ///
    /// Requires the queue to have `CL_QUEUE_PROFILING_ENABLE` — build
    /// the [`Context`](crate::Context) with
    /// [`.profiling(true)`](crate::context::ContextBuilder::profiling)
    /// to flip it on for the per-device defaults and every
    /// [`Queue`](crate::queue::Queue) built afterwards. The check
    /// fires at terminal time ([`wait`](Self::wait) /
    /// [`submit`](Self::submit) / `.await`), surfacing as
    /// [`Error::ProfilingDisabled`] before any enqueue happens.
    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    /// Sync terminal — enqueue the kernel and block on its completion.
    pub fn wait(self) -> Result<()> {
        let event = self.into_event()?;
        event.wait()?;
        Ok(())
    }

    /// Non-blocking terminal — enqueue the kernel and return its
    /// [`Event`]. The intended use is cross-queue chaining: pass the
    /// returned event to a later [`after`](Self::after) call on a
    /// different queue.
    ///
    /// For single-queue use, prefer [`wait`](Self::wait) or `.await`.
    pub fn submit(self) -> Result<Event> {
        self.into_event()
    }

    /// Crate-internal enqueue used by both [`submit`](Self::submit)
    /// and the `IntoFuture` impl in [`crate::future`].
    pub(crate) fn into_event(self) -> Result<Event> {
        let LaunchOp {
            queue,
            kernel,
            spec,
            args,
            deps,
            profile_cb,
        } = self;
        // If the caller asked for profiling, the target queue must
        // have `CL_QUEUE_PROFILING_ENABLE`. Check before enqueueing so
        // we surface a clean error instead of silently registering a
        // callback whose timestamp queries would fail later.
        // `CommandQueue::properties()` is `clGetCommandQueueInfo(...,
        // CL_QUEUE_PROPERTIES)` — one syscall per profiled launch.
        if profile_cb.is_some() && (queue.properties()? & CL_QUEUE_PROFILING_ENABLE) == 0 {
            return Err(Error::ProfilingDisabled);
        }
        let mut exec = ExecuteKernel::new(kernel);
        args.set_all(&mut exec);
        exec.set_global_work_sizes(spec.global());
        if let Some(local) = spec.local() {
            exec.set_local_work_sizes(local);
        }
        if !deps.is_empty() {
            exec.set_event_wait_list(&deps);
        }
        // SAFETY: opencl3's `enqueue_nd_range` is `unsafe` because it
        // doesn't validate argument types against the kernel signature.
        // claspr's typed wrappers (the proc-macro emits matched
        // `&DeviceSlice<T>` / `&Image2D<...>` arg types) are what make
        // this call safe in practice.
        let event = unsafe { exec.enqueue_nd_range(queue)? };
        // Let arg types that need post-enqueue bookkeeping (e.g.
        // `SharedBuffer<T>` recording the event for its Drop's
        // `clEnqueueSVMFree` wait-list) see the completion event.
        // Default impl on `KernelArg` is a no-op, so this is free for
        // every other arg type.
        args.register_all(&event);
        if let Some(cb) = profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(event)
    }
}

// ── Profiling callback FFI shim ─────────────────────────────────────

/// Box passed to OpenCL via `user_data`. The retained event keeps
/// the cl_event alive long enough for the callback to query its
/// timestamps even if the user's `Event` handle has been dropped.
struct ProfileData {
    event: Event,
    cb: ProfileCb,
}

/// Register a profiling callback on `event`. Wraps the
/// `clSetEventCallback(CL_COMPLETE, ...)` FFI dance — bumps the
/// event refcount, boxes the closure + event into `user_data`, and
/// hands it to OpenCL. The callback thunk (private to this module)
/// reclaims the box on completion, queries the four CL profiling
/// timestamps, and invokes `cb` with the result.
///
/// Shared by [`LaunchOp::profiled`] and by claspr-async's Tier 2
/// `.profiled()` combinator (which registers the same shim on a
/// marker event after a sub-chain completes).
pub fn register_profiling_callback(event: &Event, cb: ProfileCb) -> Result<()> {
    // Bump the cl_event refcount so we own a second handle inside the
    // callback's user_data. The user's Event handle in `submit()` /
    // the internal handle used by `wait()` may be released before the
    // callback fires; we need a dedicated reference.
    //
    // SAFETY: `event.get()` returns a live cl_event; `retain_event`
    // matches with the `Event::drop` inside `ProfileData` when the
    // callback reclaims the box.
    unsafe {
        retain_event(event.get())
            .map_err(|code| Error::OpenCl(opencl3::error_codes::ClError(code)))?
    };
    let owned_event = Event::new(event.get());
    let data = Box::new(ProfileData {
        event: owned_event,
        cb,
    });
    let user_data = Box::into_raw(data) as *mut c_void;
    let res = set_event_callback(event.get(), CL_COMPLETE, profile_callback_thunk, user_data);
    if let Err(code) = res {
        // Reclaim the leaked box so the retained event drops.
        // SAFETY: registration failed, so OpenCL never took ownership
        // of `user_data` — it's still uniquely ours.
        unsafe {
            drop(Box::from_raw(user_data as *mut ProfileData));
        }
        return Err(Error::OpenCl(opencl3::error_codes::ClError(code)));
    }
    Ok(())
}

extern "C" fn profile_callback_thunk(_event: cl_event, _status: cl_int, user_data: *mut c_void) {
    // Unwinding across the FFI boundary is UB; `catch_unwind` here is
    // load-bearing. The user closure runs inside it — if it panics,
    // we drop the panic and let the OpenCL runtime continue.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `user_data` is exactly the box we leaked in
        // `register_profiling_callback`. The OpenCL spec guarantees the
        // CL_COMPLETE callback fires at most once per registration; we
        // reclaim ownership here and drop on scope exit.
        let data = unsafe { Box::from_raw(user_data as *mut ProfileData) };
        let info = collect_profiling(&data.event);
        (data.cb)(info);
    }));
}

fn collect_profiling(event: &Event) -> Result<ProfilingInfo> {
    Ok(ProfilingInfo {
        queued: event.profiling_command_queued()?,
        submit: event.profiling_command_submit()?,
        start: event.profiling_command_start()?,
        end: event.profiling_command_end()?,
    })
}

// ── User events ─────────────────────────────────────────────────────

/// Create a `cl_event` of execution status `CL_SUBMITTED` whose
/// completion is signalled from the host via
/// [`complete_user_event`]. Wraps `clCreateUserEvent`.
///
/// Building block for emulating `clEnqueueNativeKernel` on devices
/// that don't expose `CL_EXEC_NATIVE_KERNEL`: stage an async host
/// computation, hand its completion to the queue via a user event,
/// chain other commands on it as if it were a normal queue command.
///
/// The returned [`Event`] owns its reference (refcount 1 from
/// `clCreateUserEvent`); its `Drop` calls `clReleaseEvent` per the
/// usual opencl3 invariant. Status defaults to `CL_SUBMITTED` until
/// the first call to [`complete_user_event`].
pub fn create_user_event(ctx: &crate::Context) -> Result<Event> {
    let cl_ctx = ctx.raw_context().get();
    let raw = opencl3::event::create_user_event(cl_ctx)
        .map_err(|code| Error::OpenCl(opencl3::error_codes::ClError(code)))?;
    Ok(Event::new(raw))
}

/// Set a user event's execution status. Wraps `clSetUserEventStatus`.
///
/// `status` must be `CL_COMPLETE` (0) or a negative value (treated
/// as an error code). Per the CL spec, may be called at most once
/// per user event — a second call returns `CL_INVALID_OPERATION`.
///
/// A negative status causes every command waiting on this user
/// event (and transitively, anything waiting on those commands) to
/// fail with the same negative code, propagating the abort through
/// the in-queue dependency graph.
pub fn complete_user_event(event: &Event, status: cl_int) -> Result<()> {
    opencl3::event::set_user_event_status(event.get(), status)
        .map_err(|code| Error::OpenCl(opencl3::error_codes::ClError(code)))
}

// ── Keep-alive callback FFI shim ────────────────────────────────────

/// Register a `CL_COMPLETE` callback on `event` whose sole job is to
/// drop `holder` when the event fires.
///
/// Use case: non-blocking host-to-device writes. `clEnqueueWriteBuffer(CL_FALSE)`
/// requires the source host buffer to stay valid until the write event
/// completes (CL §5.2.1). The caller hands us the heap holder of the
/// source (a `Box<Vec<T>>`, a `Box<Arc<[T]>>`, etc.); we keep it alive
/// by stashing the box pointer in OpenCL's `user_data`, and drop it
/// from a `CL_COMPLETE` callback once the write is done.
///
/// `T` is monomorphised per holder type — the thunk reclaims the box
/// via `Box::from_raw` with the same `T` it was allocated as.
///
/// Panics inside the drop are caught via `catch_unwind` (FFI safety).
pub fn register_drop_callback<T>(event: &Event, holder: Box<T>) -> Result<()>
where
    T: Send + 'static,
{
    // Bump the cl_event refcount so the callback still fires even if
    // every other Event handle for this cl_event has been released.
    // SAFETY: `event.get()` is a live cl_event; retain matches the
    // release in `drop_callback_thunk` when it reclaims the box.
    unsafe {
        opencl3::event::retain_event(event.get())
            .map_err(|code| Error::OpenCl(opencl3::error_codes::ClError(code)))?
    };
    let owned_event = Event::new(event.get());
    let data = Box::new(KeepAlive {
        event: owned_event,
        holder,
    });
    let user_data = Box::into_raw(data) as *mut c_void;
    let res = set_event_callback(
        event.get(),
        CL_COMPLETE,
        drop_callback_thunk::<T>,
        user_data,
    );
    if let Err(code) = res {
        // SAFETY: registration failed, OpenCL never took ownership;
        // reclaim and drop.
        unsafe {
            drop(Box::from_raw(user_data as *mut KeepAlive<T>));
        }
        return Err(Error::OpenCl(opencl3::error_codes::ClError(code)));
    }
    Ok(())
}

/// `user_data` payload for [`register_drop_callback`]. Both fields
/// drop when the thunk reclaims the box — the `event` releases the
/// retained cl_event refcount we added, and `holder` releases
/// whatever the caller wanted us to keep alive.
#[allow(dead_code)] // fields are read only via their Drop impls
struct KeepAlive<T> {
    event: Event,
    holder: Box<T>,
}

extern "C" fn drop_callback_thunk<T>(_event: cl_event, _status: cl_int, user_data: *mut c_void)
where
    T: Send + 'static,
{
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `user_data` was leaked from `register_drop_callback`
        // with the same `T`. CL_COMPLETE fires at most once. Reclaim
        // and drop the holder + event on scope exit.
        let _ = unsafe { Box::from_raw(user_data as *mut KeepAlive<T>) };
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencl3::event::CL_COMPLETE;
    use std::sync::Arc;

    /// Host thread signals a user event; another host thread blocked on
    /// `event.wait()` returns. Exercises both [`create_user_event`] and
    /// [`complete_user_event`] end-to-end.
    #[test]
    fn user_event_signals_across_threads() {
        let Ok(ctx) = crate::Context::any() else {
            eprintln!("skipping: no OpenCL device");
            return;
        };
        let ev = Arc::new(create_user_event(&ctx).expect("create"));
        let ev2 = Arc::clone(&ev);
        let t0 = std::time::Instant::now();
        let signal = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            complete_user_event(&ev2, CL_COMPLETE as cl_int).expect("signal");
        });
        ev.wait().expect("wait");
        signal.join().expect("signal thread");
        let elapsed = t0.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_millis(40),
            "wait returned too early: {elapsed:?}"
        );
    }
}
