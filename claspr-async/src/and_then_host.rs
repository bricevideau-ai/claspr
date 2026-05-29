//! [`AndThenHost`] — async in-queue host work between two device ops.
//!
//! Emulates `clEnqueueNativeKernel` on devices that don't expose
//! `CL_EXEC_NATIVE_KERNEL` (i.e. essentially every GPU OpenCL
//! driver). Built on top of `clEnqueueMapBuffer` + `clEnqueueUnmapMemObject`
//! + `clCreateUserEvent` + a per-call spawned worker thread.
//!
//! ## What's wrong with sync host work in a chain
//!
//! The previous synchronous shape (`F: FnOnce(Output) -> Result<U>`,
//! drain upstream events with `ev.wait()`, run closure inline,
//! produce no event) collapsed the chain back into "host code with
//! extra type machinery." A host stage couldn't sit in-queue with
//! pipelined device work — every chain serialised through the
//! submitting thread.
//!
//! ## What this version does
//!
//! For each `and_then_host(|view| { ... })`:
//!
//! 1. The submitting thread (inside [`DeviceOperation::execute`])
//!    enqueues non-blocking maps for every buffer in the upstream
//!    output, with upstream events as the map's wait-list.
//! 2. Creates a user event (`clCreateUserEvent`) that downstream
//!    queue commands will wait on.
//! 3. Enqueues the matching unmaps with the user event as their
//!    wait-list.
//! 4. Spawns a worker thread holding (map handle, map events,
//!    closure, user event).
//! 5. Returns the unchanged input plus the unmap events as `Deps`.
//!
//! The worker thread:
//!
//! 1. Waits for the map events to signal (host-side wait — fine,
//!    we're on our own thread).
//! 2. Builds a borrowed view from the handle.
//! 3. Runs the closure inside `catch_unwind`.
//! 4. Sets the user event to `CL_COMPLETE` on success, a negative
//!    status on closure `Err` / panic / wait failure. The negative
//!    status propagates through the in-queue dependency graph and
//!    surfaces as the chain's failure at the next `.wait()`.
//!
//! ## Closure signature
//!
//! `F: for<'a> FnOnce(<Self::Output as Mappable>::View<'a>) -> Result<()> + Send + 'static`.
//!
//! Output type passes through (`Output = Self::Output`) so downstream
//! chains naturally. Anything the closure "produces" goes through
//! in-place mutation of the borrowed view, or side-effects (e.g.
//! `Arc<Mutex<_>>`).
//!
//! ## Error model
//!
//! Workers stash the original Rust [`Error`] into a per-chain slot on
//! the [`ExecutionContext`] before signalling the user event with
//! negative status. Terminals (`sync` / `run`) check the slot after
//! the marker event resolves with Err and prefer the stashed rich
//! variant — so `Err(Error::Build { log })` from the closure surfaces
//! at the terminal as `Error::Build { log }` rather than collapsing to
//! `Error::OpenCl(-1)`. Cases:
//!
//! - **Closure returns `Err(rust_err)`** → `rust_err` stashed.
//! - **Closure panics** → `Error::HostPanic(msg)` stashed (`msg` is
//!   the panic payload extracted via `downcast_ref`).
//! - **Map-event `wait()` fails** → `Error::OpenCl(cl_err)` stashed
//!   (genuine CL-side cause).
//! - **Upstream source-event `wait()` fails** → no stash; the upstream
//!   worker has already populated the slot (or there's no host-side
//!   cause, just a CL cascade).
//!
//! Multiple concurrent failures in `bundle!` / `fan_out`: first-writer-
//! wins. Acceptable — the others are typically cascades of the first.
//!
//! [`Error`]: claspr::Error
//! [`ExecutionContext`]: crate::ExecutionContext

use crate::exec_ctx::ExecutionContext;
use crate::mappable::Mappable;
use crate::op::{Deps, DeviceOperation, wrap_event};
use claspr::{Context, Error, Launcher, Result, complete_user_event, create_user_event};
use opencl3::event::CL_COMPLETE;
use opencl3::types::{cl_event, cl_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

/// Combinator built by [`DeviceOperationHostExt::and_then_host`].
pub struct AndThenHost<S, F> {
    source: S,
    f: Option<F>,
}

/// Combinator built by
/// [`DeviceOperationHostExt::and_then_host_with_context`]. Same
/// shape as [`AndThenHost`] but the closure also receives `&Context`
/// so it can read device / context properties or build per-context
/// host state.
pub struct AndThenHostWithContext<S, F> {
    source: S,
    f: Option<F>,
}

/// Extension trait adding [`and_then_host`](Self::and_then_host) to
/// every [`DeviceOperation`] whose output implements [`Mappable`].
pub trait DeviceOperationHostExt: DeviceOperation
where
    Self::Output: Mappable,
{
    /// Run a host closure on a borrowed view of this op's output, in
    /// queue order. The closure runs asynchronously on a per-call
    /// worker thread; the chain continues as soon as the
    /// `clCreateUserEvent` signal is queued, not when the closure
    /// returns. See module docs for the full sequencing.
    ///
    /// The closure may mutate the view in place; the mutations are
    /// committed back to the device via the matching unmap (gated
    /// on the user event). Anything else the closure wants to
    /// communicate goes through side-effects (`Arc<Mutex<_>>`, etc.)
    /// — the closure itself returns only a status.
    ///
    /// Errors from the closure (and panics, and map/unmap failures)
    /// propagate downstream as a negative `cl_event` status.
    fn and_then_host<F>(self, f: F) -> AndThenHost<Self, F>
    where
        F: for<'a> FnOnce(<Self::Output as Mappable>::View<'a>) -> Result<()> + Send + 'static,
    {
        AndThenHost {
            source: self,
            f: Some(f),
        }
    }

    /// Like [`and_then_host`](Self::and_then_host) but the closure
    /// also receives a [`&Context`](claspr::Context) — the chain's
    /// running context, for host-side use (read device props,
    /// iterate `context.devices()`, etc.).
    ///
    /// Same async / worker-thread semantics as `and_then_host`:
    /// the closure runs on a per-call worker thread; the chain
    /// continues at execute time; mutations to the view commit
    /// via the matching unmap; errors propagate through the
    /// user-event signal + the host-error slot.
    ///
    /// `Context` is `Arc`-backed so the clone the worker holds is
    /// cheap. The closure body shouldn't outlive the chain run
    /// — the `Context` clone keeps it alive for the worker's
    /// lifetime regardless.
    fn and_then_host_with_context<F>(self, f: F) -> AndThenHostWithContext<Self, F>
    where
        F: for<'a> FnOnce(&Context, <Self::Output as Mappable>::View<'a>) -> Result<()>
            + Send
            + 'static,
    {
        AndThenHostWithContext {
            source: self,
            f: Some(f),
        }
    }
}

impl<S> DeviceOperationHostExt for S
where
    S: DeviceOperation,
    S::Output: Mappable,
{
}

impl<S, F> DeviceOperation for AndThenHost<S, F>
where
    S: DeviceOperation,
    S::Output: Mappable,
    F: for<'a> FnOnce(<S::Output as Mappable>::View<'a>) -> Result<()> + Send + 'static,
{
    type Output = S::Output;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (input, source_evts) = self.source.execute(ctx, deps)?;
        let q = ctx.cl_queue();

        // Enqueue maps with upstream events as wait-list. The maps
        // are non-blocking — they queue immediately and produce
        // events the worker will wait on before reading the mapped
        // memory. The borrow of `input` ends with the call.
        let source_cl: Vec<cl_event> = source_evts.iter().map(|d| d.as_ref().get()).collect();
        let (mut handle, map_events) = input.map(q, &source_cl)?;

        // Create the user event downstream waits on (via the
        // unmaps). Status: CL_SUBMITTED until the worker calls
        // `complete_user_event`.
        let user_event = Arc::new(create_user_event(ctx.context())?);

        // Enqueue unmaps with the user event as their wait-list —
        // they fire automatically once the worker signals completion.
        // After this call, if anything below errors out, the queue
        // would be stuck forever (unmaps waiting on a never-signalled
        // user event). The defensive Drop on `handle` doesn't cover
        // that. Mitigation: from this point on we MUST signal the
        // user event before returning Err.
        let unmap_events =
            match <S::Output as Mappable>::enqueue_unmap(&mut handle, q, &[user_event.get()]) {
                Ok(evs) => evs,
                Err(e) => {
                    // Map succeeded, unmap-enqueue failed. The user
                    // event is unsignalled, but there's nothing waiting
                    // on it yet (unmaps weren't enqueued). Set it to a
                    // negative status anyway in case future code in this
                    // execute() body added a waiter.
                    let _ = complete_user_event(&user_event, -1);
                    return Err(e);
                }
            };

        let f = self
            .f
            .take()
            .expect("AndThenHost::execute called twice — internal claspr-async bug");

        // Spawn the worker. It owns the handle, the map events, the
        // upstream events (for error short-circuiting), the user-event
        // Arc clone, the host-error slot, and the closure.
        //
        // Why source_evts in addition to map_events: when the upstream
        // user event resolves to a negative status (chain error), the
        // map command transitively fails too — except for `Mappable`
        // impls that don't enqueue maps (scalars, unit). For those,
        // `map_events` is empty and the worker would otherwise run the
        // closure regardless. Including source_evts directly is the
        // uniform fix.
        let worker_user_event = Arc::clone(&user_event);
        let worker_map_events = map_events;
        let worker_source_evts = source_evts;
        let worker_host_error = ctx.host_error_slot();
        std::thread::spawn(move || {
            // Worker is the only thread allowed to touch `handle`
            // and to call `complete_user_event`. Any path through
            // this body must signal the user event exactly once.
            let (status, mut handle, rust_err) =
                run_worker::<S::Output, F>(handle, worker_map_events, worker_source_evts, f);
            // Stash the original Rust error variant before signalling
            // negative status, so the terminal can prefer it over the
            // cascade. First-writer-wins — leave the slot alone if
            // someone else already populated it (a concurrent failing
            // host worker in the same bundle/fan-out).
            if let Some(err) = rust_err {
                let mut slot = worker_host_error.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(err);
                }
            }
            if status < 0 {
                // On the error path the queued unmap (which waits on
                // the user event) is "terminated" by the OpenCL
                // runtime when we signal a negative status — it
                // never actually unmaps the buffer. Per the spec,
                // queue / context state after such a termination is
                // implementation-defined, so we can't rely on the
                // unmap "eventually completing" or on the runtime
                // gracefully recovering. Force the defensive sync
                // unmap NOW, before signalling failure, so that by
                // the time the chain's `.sync()` observes the error
                // and returns, the buffer is in a clean (unmapped)
                // state regardless of how the implementation handles
                // the rest of the queue.
                <S::Output as Mappable>::mark_unmap_not_done(&mut handle);
                drop(handle);
            }
            let _ = complete_user_event(&worker_user_event, status);
            // In the success path, handle drops here (no-op — the
            // queued unmap is firing on its own via the user event).
        });

        // Downstream waits on the unmap events when there are any
        // (transitively this implies user-event completion). When
        // the output has no buffers (scalar / unit / nested empty
        // tuple), unmaps are empty — fall back to the user event
        // directly so downstream still has a gate.
        let deps_out: Deps = if unmap_events.is_empty() {
            vec![user_event]
        } else {
            unmap_events.into_iter().map(wrap_event).collect()
        };
        Ok((input, deps_out))
    }
}

impl<S, F> DeviceOperation for AndThenHostWithContext<S, F>
where
    S: DeviceOperation,
    S::Output: Mappable,
    F: for<'a> FnOnce(&Context, <S::Output as Mappable>::View<'a>) -> Result<()> + Send + 'static,
{
    type Output = S::Output;

    /// Mirrors [`AndThenHost::execute`] — same worker / user-event /
    /// host-error-slot plumbing. Only difference: captures a
    /// `Context` clone (cheap, Arc-backed) for the worker so the
    /// closure can be invoked as `f(&context, view)`.
    ///
    /// Closure-HRTB inference doesn't go through GAT-based view
    /// types, so this can't cleanly delegate to `AndThenHost` by
    /// wrapping the user closure — the wrapped form fails to match
    /// the `for<'a> FnOnce(View<'a>) -> Result<()>` bound at the
    /// type-checker level. Duplicated body is the simpler answer.
    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (input, source_evts) = self.source.execute(ctx, deps)?;
        let q = ctx.cl_queue();

        let source_cl: Vec<cl_event> = source_evts.iter().map(|d| d.as_ref().get()).collect();
        let (mut handle, map_events) = input.map(q, &source_cl)?;

        let user_event = Arc::new(create_user_event(ctx.context())?);

        let unmap_events =
            match <S::Output as Mappable>::enqueue_unmap(&mut handle, q, &[user_event.get()]) {
                Ok(evs) => evs,
                Err(e) => {
                    let _ = complete_user_event(&user_event, -1);
                    return Err(e);
                }
            };

        let f = self
            .f
            .take()
            .expect("AndThenHostWithContext::execute called twice — internal claspr-async bug");
        // Cheap Arc-backed clone — gives the worker thread its own
        // 'static handle on the running context.
        let worker_context: Context = ctx.context().clone();
        let worker_user_event = Arc::clone(&user_event);
        let worker_map_events = map_events;
        let worker_source_evts = source_evts;
        let worker_host_error = ctx.host_error_slot();
        std::thread::spawn(move || {
            let (status, mut handle, rust_err) = run_worker_with_context::<S::Output, F>(
                handle,
                worker_map_events,
                worker_source_evts,
                worker_context,
                f,
            );
            if let Some(err) = rust_err {
                let mut slot = worker_host_error.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(err);
                }
            }
            if status < 0 {
                <S::Output as Mappable>::mark_unmap_not_done(&mut handle);
                drop(handle);
            }
            let _ = complete_user_event(&worker_user_event, status);
        });

        let deps_out: Deps = if unmap_events.is_empty() {
            vec![user_event]
        } else {
            unmap_events.into_iter().map(wrap_event).collect()
        };
        Ok((input, deps_out))
    }
}

/// Twin of [`run_worker`] for [`AndThenHostWithContext`]. Only
/// difference: the closure receives `&Context` (passed by reference
/// from the owned `context` held on the worker's stack frame) along
/// with the view.
fn run_worker_with_context<O, F>(
    handle: O::MapHandle,
    map_events: Vec<claspr::Event>,
    source_evts: Deps,
    context: Context,
    f: F,
) -> (cl_int, O::MapHandle, Option<Error>)
where
    O: Mappable,
    F: for<'a> FnOnce(&Context, <O as Mappable>::View<'a>) -> Result<()> + Send + 'static,
{
    let mut handle = handle;
    for ev in &source_evts {
        if ev.as_ref().wait().is_err() {
            return (-1, handle, None);
        }
    }
    for ev in &map_events {
        if let Err(e) = ev.wait() {
            return (-1, handle, Some(Error::OpenCl(e)));
        }
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let view = <O as Mappable>::view(&mut handle);
        f(&context, view)
    }));
    match result {
        Ok(Ok(())) => (CL_COMPLETE as cl_int, handle, None),
        Ok(Err(rust_err)) => (-1, handle, Some(rust_err)),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            (-1, handle, Some(Error::HostPanic(msg)))
        }
    }
}

/// Worker body. Wait on upstream + map events, then run the closure
/// inside `catch_unwind`. Returns the (status, handle, optional rich
/// error) so the caller can stash the rich error before signalling
/// and decide whether to trigger the defensive sync unmap.
///
/// Why we return the handle: in the error path the queued unmap (with
/// the user event in its wait-list) gets terminated by the OpenCL
/// runtime instead of executing, which would leave the buffer mapped
/// forever. The caller uses [`crate::mappable::Mappable`]'s defensive
/// Drop path to issue a synchronous unmap when status is negative.
///
/// The third return slot is `Some(err)` when this worker has a
/// host-side cause for the failure (closure `Err`, closure panic,
/// or map-event wait failure). It's `None` for upstream-cascade
/// short-circuits — the upstream worker (or a CL command without a
/// host cause) is responsible for that signal.
fn run_worker<O, F>(
    handle: O::MapHandle,
    map_events: Vec<claspr::Event>,
    source_evts: Deps,
    f: F,
) -> (cl_int, O::MapHandle, Option<Error>)
where
    O: Mappable,
    F: for<'a> FnOnce(<O as Mappable>::View<'a>) -> Result<()> + Send + 'static,
{
    let mut handle = handle;
    // Short-circuit on upstream chain error (negative source-event
    // status, e.g. from a previous and_then_host whose closure failed).
    // Do NOT stash — the upstream worker already populated the slot
    // (or there was no host-side cause).
    for ev in &source_evts {
        if ev.as_ref().wait().is_err() {
            return (-1, handle, None);
        }
    }
    // Map-event failure is a host-observable CL error — stash so the
    // terminal sees the actual ClError rather than the cascade.
    for ev in &map_events {
        if let Err(e) = ev.wait() {
            return (-1, handle, Some(Error::OpenCl(e)));
        }
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let view = <O as Mappable>::view(&mut handle);
        f(view)
    }));
    match result {
        Ok(Ok(())) => (CL_COMPLETE as cl_int, handle, None),
        Ok(Err(rust_err)) => (-1, handle, Some(rust_err)),
        Err(panic) => {
            // `catch_unwind` returns `Box<dyn Any + Send>`. The
            // payload is typically `&'static str` (from `panic!("lit")`)
            // or `String` (from `panic!("{}", x)`). Anything else
            // gets a generic placeholder — the panic stack isn't
            // available anyway once it's crossed the boundary.
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            (-1, handle, Some(Error::HostPanic(msg)))
        }
    }
}
