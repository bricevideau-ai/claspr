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
//! v1 propagates "something went wrong" — no detailed error info
//! survives the user-event boundary. Closure `Err` / panic / map
//! failure / unmap failure all produce a negative user-event status.
//! Downstream's `event.wait()` returns that status as an [`Error`].
//! Capturing detailed errors into the chain via side-effects is the
//! caller's responsibility for now.
//!
//! [`Error`]: claspr::Error

use crate::exec_ctx::ExecutionContext;
use crate::mappable::Mappable;
use crate::op::{Deps, DeviceOperation, wrap_event};
use claspr::{Launcher, Result, complete_user_event, create_user_event};
use opencl3::event::CL_COMPLETE;
use opencl3::types::{cl_event, cl_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

/// Combinator built by [`DeviceOperationHostExt::and_then_host`].
pub struct AndThenHost<S, F> {
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

    fn execute(
        mut self,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(Self::Output, Deps)> {
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
        let unmap_events = match <S::Output as Mappable>::enqueue_unmap(
            &mut handle,
            q,
            &[user_event.get()],
        ) {
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
        // Arc clone, and the closure.
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
        std::thread::spawn(move || {
            // Worker is the only thread allowed to touch `handle`
            // and to call `complete_user_event`. Any path through
            // this body must signal the user event exactly once.
            let (status, mut handle) =
                run_worker::<S::Output, F>(handle, worker_map_events, worker_source_evts, f);
            if status < 0 {
                // On the error path the queued unmap (which waits on
                // the user event) is "terminated" by the OpenCL runtime
                // when we signal a negative status (CL spec §5.11) —
                // it never actually unmaps the buffer. Force the
                // defensive sync unmap NOW, before signalling failure,
                // so that by the time the chain's `.sync()` observes
                // the error and returns, the buffer is in a clean
                // (unmapped) state. Otherwise subsequent commands on
                // the same context can see a still-mapped buffer
                // that confuses strict implementations like rusticl.
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

/// Worker body. Wait on upstream + map events, then run the closure
/// inside `catch_unwind`. Returns the (status, handle) so the caller
/// can decide whether to trigger the defensive sync unmap.
///
/// Why we return the handle: in the error path the queued unmap (with
/// the user event in its wait-list) gets terminated by the OpenCL
/// runtime instead of executing, which would leave the buffer mapped
/// forever. The caller uses [`crate::mappable::Mappable`]'s defensive
/// Drop path to issue a synchronous unmap when status is negative.
fn run_worker<O, F>(
    handle: O::MapHandle,
    map_events: Vec<claspr::Event>,
    source_evts: Deps,
    f: F,
) -> (cl_int, O::MapHandle)
where
    O: Mappable,
    F: for<'a> FnOnce(<O as Mappable>::View<'a>) -> Result<()> + Send + 'static,
{
    let mut handle = handle;
    // Short-circuit on upstream chain error (negative source-event
    // status, e.g. from a previous and_then_host whose closure failed).
    for ev in &source_evts {
        if ev.as_ref().wait().is_err() {
            return (-1, handle);
        }
    }
    for ev in &map_events {
        if ev.wait().is_err() {
            return (-1, handle);
        }
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let view = <O as Mappable>::view(&mut handle);
        f(view)
    }));
    let status = match result {
        Ok(Ok(())) => CL_COMPLETE as cl_int,
        Ok(Err(_)) | Err(_) => -1,
    };
    (status, handle)
}
