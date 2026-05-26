//! [`DeviceOperation`] — the core trait. Anything that describes
//! lazy GPU work implements it, and a terminal ([`sync`](DeviceOperation::sync)
//! today; `.await` in Phase 3.4) actually runs the chain.
//!
//! ## Combinators (this module)
//!
//! - [`value(v)`](value) — lift a host value into the chain.
//! - [`with_context(|ctx| ...)`](with_context) — defer construction
//!   of an op until the [`ExecutionContext`] is available (used to
//!   compose Tier 1 ops into a Tier 2 chain).
//! - [`.and_then(|out| next_op)`](DeviceOperation::and_then) —
//!   sequential dependency; `next_op` runs after `self` produces its
//!   output.
//! - [`.arc()`](DeviceOperation::arc) — wrap the output in `Arc<T>`
//!   so it can be shared across downstream branches (needs
//!   `Self::Output: Sync`).
//!
//! Later phases add: `bundle!`/`fan_out` (parallel structure),
//! `ArcSplit::split::<N>()` (fan-out from a single Arc),
//! `and_then_host` (host work between two device ops),
//! `HostAccessible` (host views of device buffers), `IntoFuture`
//! (`.await`), and `.profiled(|info| ...)`.

use crate::exec_ctx::ExecutionContext;
use claspr::Result;

// ── DeviceOperation ─────────────────────────────────────────────────

/// Anything that describes lazy device work — a single kernel launch,
/// a buffer upload, a chain of those, a parallel bundle of branches,
/// etc.
///
/// Built up via free constructors ([`value`], [`with_context`], plus
/// the proc-macro-emitted `kernels.foo_op(...)` in Phase 4) and the
/// combinator methods on this trait. Executes when the user picks a
/// terminal — today only the synchronous [`sync`](Self::sync); `.await`
/// lands in Phase 3.4.
pub trait DeviceOperation: Send + Sized {
    /// The host value this op produces when it finishes. Bounded
    /// `Send` so the chain can later be polled across threads (the
    /// async terminal in Phase 3.4 needs this).
    type Output: Send;

    /// Run the op against `ctx`. Op authors implement this; user code
    /// should generally use a terminal ([`sync`](Self::sync)) instead.
    fn execute(self, ctx: &ExecutionContext<'_>) -> Result<Self::Output>;

    /// Synchronous terminal — execute the chain on `context`'s
    /// default device using its out-of-order default queue, block
    /// until every enqueued command has completed, return the chain's
    /// output. The OOO queue is used so independent sub-ops added
    /// by later combinators ([`bundle!`], `fan_out`, etc.) can overlap.
    fn sync(self, context: &claspr::Context) -> Result<Self::Output> {
        let device = context.device().clone();
        let queue = context.default_outoforder_queue(&device)?;
        let ctx = ExecutionContext::new(context, device, queue.raw());
        let out = self.execute(&ctx)?;
        // Drain the queue before returning — any commands the chain
        // enqueued (uploads, kernels, downloads) need to have finished
        // for the returned output to be meaningfully usable.
        queue.finish()?;
        Ok(out)
    }

    /// Async terminal — submit the chain and return a [`ChainFuture`]
    /// that resolves when the chain's commands have all completed on
    /// the device. The chain's `Output` value is materialised eagerly
    /// (handles, `Vec`s, etc.); the future just gates *when* the user
    /// gets to see it on the queue's marker firing.
    ///
    /// ```ignore
    /// let result = chain.run(&ctx).await?;
    /// ```
    fn run(self, context: &claspr::Context) -> crate::future::ChainFuture<Self::Output> {
        crate::future::run_chain(self, context)
    }

    /// Sequential dependency: when `self` produces its output, hand
    /// it to `f` to build the next op in the chain, then run that op
    /// against the same [`ExecutionContext`].
    ///
    /// The closure is [`FnOnce`] + [`Send`] so the resulting
    /// [`AndThen`] is also [`Send`] (and so the chain can later be
    /// polled across threads).
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

    /// Wrap this op's output in [`Arc<T>`](std::sync::Arc) so it can
    /// be shared across downstream branches that all want read-only
    /// access. Requires `Self::Output: Sync` because `Arc<T>: Send`
    /// only when `T: Send + Sync`.
    fn arc(self) -> Arced<Self>
    where
        Self::Output: Sync,
    {
        Arced { source: self }
    }
}

// ── Value: lift a host value into the chain ─────────────────────────

/// Lazy wrapper around a host value — the simplest [`DeviceOperation`].
///
/// `value(v)` returns a leaf op that, when executed, yields `v`. Useful
/// at the head of a chain when the first device step needs a host-
/// supplied input.
pub struct Value<T: Send> {
    v: Option<T>,
}

/// Construct a [`Value`] op. See the type docs.
pub fn value<T: Send>(v: T) -> Value<T> {
    Value { v: Some(v) }
}

impl<T: Send> DeviceOperation for Value<T> {
    type Output = T;

    fn execute(mut self, _ctx: &ExecutionContext<'_>) -> Result<T> {
        Ok(self
            .v
            .take()
            .expect("Value::execute called twice — internal claspr-async bug"))
    }
}

// ── WithContext: defer construction until the ctx is known ──────────

/// Lazy wrapper around a closure that produces a host value given the
/// running [`ExecutionContext`].
///
/// The escape hatch for code that needs the live `&Context` /
/// `&CommandQueue` to do its work — buffer allocations, kernel
/// launches via Tier 1 ops, etc. The closure runs when the surrounding
/// chain reaches this op.
pub struct WithContext<F> {
    f: Option<F>,
}

/// Construct a [`WithContext`] op. See the type docs.
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

    fn execute(mut self, ctx: &ExecutionContext<'_>) -> Result<O> {
        (self
            .f
            .take()
            .expect("WithContext::execute called twice — internal claspr-async bug"))(ctx)
    }
}

// ── AndThen ─────────────────────────────────────────────────────────

/// Sequential dependency combinator built by
/// [`DeviceOperation::and_then`]. Runs `source` first, then feeds its
/// output to `f` to build the next op and runs that.
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

    fn execute(mut self, ctx: &ExecutionContext<'_>) -> Result<U::Output> {
        let prior = self.source.execute(ctx)?;
        let next = (self
            .f
            .take()
            .expect("AndThen::execute called twice — internal claspr-async bug"))(
            prior
        );
        next.execute(ctx)
    }
}

// ── Arced: wrap output in Arc<T> ────────────────────────────────────

/// Combinator built by [`DeviceOperation::arc`]. Wraps the source op's
/// output in [`Arc<T>`](std::sync::Arc) so multiple downstream
/// consumers can share it without each needing to own it.
///
/// Phase 3.3 adds [`ArcSplit::split::<N>()`] which takes the Arc'd
/// output and produces N leaves each cloning the Arc — the typical
/// shape for fan-out from a single shared input.
pub struct Arced<S> {
    source: S,
}

impl<S> DeviceOperation for Arced<S>
where
    S: DeviceOperation,
    S::Output: Sync,
{
    type Output = std::sync::Arc<S::Output>;

    fn execute(self, ctx: &ExecutionContext<'_>) -> Result<std::sync::Arc<S::Output>> {
        Ok(std::sync::Arc::new(self.source.execute(ctx)?))
    }
}
