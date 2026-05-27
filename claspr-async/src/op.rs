//! [`DeviceOperation`] — the core trait. Anything that describes
//! lazy GPU work implements it.
//!
//! ## Execute signature: event-tracking through the chain
//!
//! ```ignore
//! fn execute(self, ctx, deps: Deps) -> Result<(Self::Output, Deps)>;
//! ```
//!
//! Each op takes the dependency events from its predecessor as the
//! wait-list for whatever it enqueues, and returns the events its
//! own commands produced so the next op can use them in turn.
//! Combinators thread this through automatically: `and_then` forwards
//! source's events to next, `bundle!` / `fan_out` clone the input
//! deps to each child and join the children's events via
//! `clEnqueueMarkerWithWaitList`. Host-only ops ([`Value`],
//! [`WithContext`]) wait on `deps` synchronously before running their
//! closure and return an empty event list.
//!
//! The terminals ([`sync`](DeviceOperation::sync) /
//! [`run`](DeviceOperation::run)) wait on the final events before
//! handing the output back to the user.
//!
//! ## Combinators (this module)
//!
//! - [`value(v)`](value) — lift a host value into the chain.
//! - [`with_context(|ctx| ...)`](with_context) — defer construction
//!   of an op until the [`ExecutionContext`] is available.
//! - [`.and_then(|out| next_op)`](DeviceOperation::and_then) —
//!   sequential dependency.
//! - [`.arc()`](DeviceOperation::arc) — wrap output in `Arc<T>`.

use crate::exec_ctx::ExecutionContext;
use claspr::{Event, Result};
use std::sync::Arc;

// ── Deps ────────────────────────────────────────────────────────────

/// A single tracked event in a [`Deps`] chain. Arc-wrapped so it can
/// be cheaply shared across parallel branches in
/// [`bundle!`](crate::bundle!) / [`fan_out`](crate::fan_out) without
/// extra `clRetainEvent` calls.
pub type Dep = Arc<Event>;

/// The wait-list / produced-event list threaded through every
/// [`DeviceOperation::execute`] call. Empty at chain start; one
/// element per device op the previous step enqueued; multi-element
/// after a parallel join (Bundle/FanOut) collapses children's events
/// into the marker that joins them.
pub type Deps = Vec<Dep>;

/// Convenience: borrow each `Dep` as `&Event` for an
/// `after_all(...)` call on a Tier 1 op builder.
pub fn deps_as_events(deps: &Deps) -> impl Iterator<Item = &Event> {
    deps.iter().map(|d| d.as_ref())
}

/// Wrap an opencl3 [`Event`] in a [`Dep`].
pub fn wrap_event(event: Event) -> Dep {
    Arc::new(event)
}

// ── DeviceOperation ─────────────────────────────────────────────────

/// Anything that describes lazy device work.
///
/// Built up via free constructors ([`value`], [`with_context`], plus
/// the proc-macro-emitted `kernels.foo_op(...)`) and the combinator
/// methods on this trait. Executes when the user picks a terminal —
/// [`sync`](Self::sync) (blocking) or [`run`](Self::run) (async
/// `.await`).
pub trait DeviceOperation: Send + Sized {
    /// The host value this op produces when it finishes.
    type Output: Send;

    /// Run the op against `ctx` with `deps` as the wait-list.
    /// Returns the op's output **and** the events the op produced
    /// (so the next op in the chain can wait on them).
    ///
    /// Op authors implement this; user code uses a terminal.
    fn execute(self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)>;

    /// Synchronous terminal — execute the chain on `context`'s default
    /// device using its out-of-order default queue, wait for every
    /// event the chain produced, return the output.
    fn sync(self, context: &claspr::Context) -> Result<Self::Output> {
        let device = context.device().clone();
        let queue = context.default_outoforder_queue(&device)?;
        let ctx = ExecutionContext::new(context, device, queue.raw());
        let (out, events) = self.execute(&ctx, Vec::new())?;
        // Wait on every final event. Most chains have at most one
        // (after a join marker); fan-outs may have several.
        for ev in &events {
            ev.wait()?;
        }
        // Defensive: also drain the queue in case any commands were
        // enqueued without being tracked in the events list (e.g.
        // a `with_context` closure that called `.submit()` and
        // discarded the event — bad style, but we don't want to
        // strand commands).
        queue.finish()?;
        Ok(out)
    }

    /// Async terminal — submit the chain and return a [`ChainFuture`](crate::ChainFuture)
    /// that resolves when the chain's events fire.
    fn run(self, context: &claspr::Context) -> crate::future::ChainFuture<Self::Output> {
        crate::future::run_chain(self, context)
    }

    /// Sequential dependency: when `self` produces its output, hand
    /// it to `f` to build the next op in the chain. The next op
    /// receives `self`'s events as *its* `deps` — so subsequent device
    /// work is queue-ordered after `self`'s commands without a
    /// host-side wait.
    ///
    /// **Use [`and_then_host`][ath] (not `and_then`) if the closure
    /// reads or drops data produced by a non-blocking source op**
    /// (e.g. summing a `Vec<u32>` from `download`, or otherwise
    /// touching the host side of an in-flight transfer's destination).
    /// `and_then` does NOT drain source events before invoking the
    /// closure — that's deliberate, so kernel→kernel chains can
    /// pipeline queue-side without host stalls. If the closure
    /// consumes prior by value and drops it, the heap is freed
    /// while OpenCL may still be writing into it (UB). `and_then_host`
    /// drains source events synchronously before invoking the closure,
    /// which is the right behavior for host-touching work.
    ///
    /// [ath]: crate::DeviceOperationHostExt::and_then_host
    fn and_then<F, U>(self, f: F) -> AndThen<Self, F>
    where
        F: FnOnce(Self::Output) -> U + Send,
        U: DeviceOperation,
    {
        AndThen {
            source: self,
            f: Some(f),
        }
    }

    /// Wrap this op's output in [`Arc<T>`](std::sync::Arc) for sharing
    /// across downstream branches.
    fn arc(self) -> Arced<Self>
    where
        Self::Output: Sync,
    {
        Arced { source: self }
    }
}

// ── Value: lift a host value into the chain ─────────────────────────

/// Lazy wrapper around a host value. `execute` passes `deps` through
/// unchanged — no enqueue, no new events.
pub struct Value<T: Send> {
    v: Option<T>,
}

pub fn value<T: Send>(v: T) -> Value<T> {
    Value { v: Some(v) }
}

impl<T: Send> DeviceOperation for Value<T> {
    type Output = T;

    fn execute(mut self, _ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(T, Deps)> {
        Ok((
            self.v
                .take()
                .expect("Value::execute called twice — internal claspr-async bug"),
            deps,
        ))
    }
}

// ── WithContext: defer construction until the ctx is known ──────────

/// Lazy wrapper around a closure that produces a host value given the
/// running [`ExecutionContext`].
///
/// **Note on dependencies:** the closure can't naturally accept
/// per-call `deps`, so `execute` waits on `deps` synchronously
/// before running the closure (ensuring any device state the closure
/// might read is consistent). The closure's own enqueues are
/// expected to use `.wait()` terminals — if they use `.submit()`,
/// those events are untracked by the chain and may strand commands
/// the terminal's defensive `queue.finish()` will catch (but not
/// chain into downstream ops).
pub struct WithContext<F> {
    f: Option<F>,
}

pub fn with_context<F, O>(f: F) -> WithContext<F>
where
    F: FnOnce(&ExecutionContext<'_>) -> Result<O> + Send,
    O: Send,
{
    WithContext { f: Some(f) }
}

impl<F, O> DeviceOperation for WithContext<F>
where
    F: FnOnce(&ExecutionContext<'_>) -> Result<O> + Send,
    O: Send,
{
    type Output = O;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(O, Deps)> {
        // Drain deps before running the closure — anything the
        // closure might read from device state needs to be ready.
        for ev in &deps {
            ev.wait()?;
        }
        let out = (self
            .f
            .take()
            .expect("WithContext::execute called twice — internal claspr-async bug"))(
            ctx
        )?;
        // Closure assumed to use .wait() on its internal Tier 1 ops;
        // no events to forward.
        Ok((out, Vec::new()))
    }
}

// ── AndThen ─────────────────────────────────────────────────────────

/// Sequential dependency combinator. `source` runs with the chain's
/// deps; its events become `next`'s deps.
pub struct AndThen<S, F> {
    source: S,
    f: Option<F>,
}

impl<S, F, U> DeviceOperation for AndThen<S, F>
where
    S: DeviceOperation,
    F: FnOnce(S::Output) -> U + Send,
    U: DeviceOperation,
{
    type Output = U::Output;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(U::Output, Deps)> {
        let (prior, prior_evts) = self.source.execute(ctx, deps)?;
        let next = (self
            .f
            .take()
            .expect("AndThen::execute called twice — internal claspr-async bug"))(
            prior
        );
        next.execute(ctx, prior_evts)
    }
}

// ── Arced: wrap output in Arc<T> ────────────────────────────────────

/// Combinator built by [`DeviceOperation::arc`]. Pass-through for
/// events.
pub struct Arced<S> {
    source: S,
}

impl<S> DeviceOperation for Arced<S>
where
    S: DeviceOperation,
    S::Output: Sync,
{
    type Output = Arc<S::Output>;

    fn execute(self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Arc<S::Output>, Deps)> {
        let (out, evts) = self.source.execute(ctx, deps)?;
        Ok((Arc::new(out), evts))
    }
}
