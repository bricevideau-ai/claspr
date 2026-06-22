//! Eager struct-graph core — the closure-free `DeviceOperation` replacement.
//!
//! A graph is a **closure-free nested struct** of [`EagerOp`]s. `.and_then(f)`
//! runs the builder `f` **once at construction**, handing it a [`Pipe<T>`]
//! handle for the upstream's future output, and stores the **returned op** —
//! never the closure. So `g = upload(v).and_then(|p| fill(p, 7))` is a plain
//! struct (`AndThen<Upload, Fill>`); it can be traversed/inspected without
//! executing.
//!
//! ## Edges carry `(value, Deps)`
//!
//! A [`Pipe`] carries the produced value AND the events its commands enqueued.
//! An op takes the upstream `Deps` from its input pipe, threads them as the
//! wait-list of its **non-blocking** enqueue, and deposits `(output,
//! vec![its_event])` into its output pipe. Nothing blocks mid-graph — the same
//! `execute(deps) -> (out, deps)` threading the old closure layer did, carried
//! through the pipe payload. Only [`sync`](EagerOpExt::sync) waits, on the
//! terminal pipe's `Deps`.
//!
//! ## `Input<T>`: concrete or piped
//!
//! A leaf's input is an [`Input<T>`] — `Concrete(T)` (bound at build) or
//! `Pipe(Pipe<T>)` (produced upstream). One type, two states: this is the edge
//! that unifies concrete args, intermediate values, and (later) slots.
//!
//! This module is being grown to replace the closure-based `op.rs` layer
//! (NOTES → "CONVERSION PLAN"); the old trait stays alive in parallel until
//! every leaf is ported.

use crate::copy::CopyTo;
use crate::device_op::{Deps, DeviceOperation, wrap_event};
use crate::exec_ctx::ExecutionContext;
use crate::host_view::{
    DeviceSliceHostView, HostReadableExt, HostWritableExt, MapAccess, MapReadOnly, MapReadWrite,
    MappedSliceHostView,
};
use crate::image::ImageHostTransfer;
use crate::transfer::UploadSource;
use crate::{
    Buffer, Context, DeviceSlice, DeviceSliceUninit, Error, Fillable, HostReadable, HostUploadable,
    HostWritable, MappedSlice, MappedSliceUninit, MemMode, ReadWrite, Result, USMSlice,
    USMSliceUninit, register_drop_callback,
};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

// ── Pipe<T> + Input<T>: the graph edge ─────────────────────────────────

/// A build-time handle to an op's future output, carrying `(value, Deps)`. The
/// producing op **moves** its value + the events its commands enqueued in at
/// execute; the consuming op moves them out as its own wait-list. Cheap-clone
/// (`Arc`); identity is the `Arc` cell, so independently-built subgraphs
/// compose with no global numbering.
pub struct Pipe<T> {
    cell: Arc<Mutex<Option<(T, Deps)>>>,
}

impl<T> Clone for Pipe<T> {
    fn clone(&self) -> Self {
        Pipe {
            cell: Arc::clone(&self.cell),
        }
    }
}

impl<T> Default for Pipe<T> {
    fn default() -> Self {
        Pipe {
            cell: Arc::new(Mutex::new(None)),
        }
    }
}

impl<T> Pipe<T> {
    /// A fresh, empty pipe.
    pub fn new() -> Self {
        Self::default()
    }
    /// Deposit the value and the events its commands produced.
    pub fn put(&self, v: T, deps: Deps) {
        *self.cell.lock().unwrap() = Some((v, deps));
    }
    /// Move out the value + its events (the downstream wait-list).
    pub fn take(&self) -> Option<(T, Deps)> {
        self.cell.lock().unwrap().take()
    }
}

/// An op argument: a concrete value known at build, or a [`Pipe`] filled by an
/// upstream op at execute time.
pub enum Input<T> {
    /// Bound at construction (e.g. a caller-owned buffer passed directly).
    Concrete(T),
    /// Deferred — produced by an upstream op, moved out of the shared cell.
    Pipe(Pipe<T>),
}

impl<T> Input<T> {
    /// Resolve to `(value, upstream Deps)` at execute time (consuming it). A
    /// concrete value carries no upstream events; a pipe carries whatever its
    /// producer enqueued (the downstream wait-list).
    pub fn resolve(self) -> Result<(T, Deps)> {
        match self {
            Input::Concrete(v) => Ok((v, Deps::new())),
            Input::Pipe(p) => p.take().ok_or(Error::NotSupported(
                "eager graph: upstream pipe was not filled before downstream ran \
                 — internal ordering bug",
            )),
        }
    }

    /// Resolve to a concrete value, erroring if this is a pipe. Used by the
    /// Tier-1 / `KernelOp` enqueue path, where a pipe is unreachable (a pipe
    /// only exists inside an eager `and_then` closure — a context that never
    /// calls the Tier-1 terminals). The `(value, Deps)` form is the eager path.
    pub fn resolve_concrete(self) -> Result<T> {
        match self {
            Input::Concrete(v) => Ok(v),
            Input::Pipe(_) => Err(Error::NotSupported(
                "kernel: a pipe input reached the Tier-1 path — use the eager \
                 `.and_then`/`.sync` terminals for piped (graph) inputs",
            )),
        }
    }
}

impl<T> From<T> for Input<T> {
    fn from(v: T) -> Self {
        Input::Concrete(v)
    }
}

impl<T> From<Pipe<T>> for Input<T> {
    fn from(p: Pipe<T>) -> Self {
        Input::Pipe(p)
    }
}

// ── ToInput: a kernel buffer arg, concrete-or-pipe, with Buf inferred ──
//
// The macro-emitted kernel method takes each buffer arg as `impl ToInput<elem>`
// and stores the resulting `Input<Buf>`. `Buf` is the concrete buffer type
// (`DeviceSlice`/`MappedSlice`/`USMSlice`) — inferred from the arg, so neither
// `kernels.foo(buf)` nor `kernels.foo(pipe)` needs a turbofish. Per-family
// impls (NOT a blanket over the buffer type) so the `Pipe<D>` impl doesn't
// overlap — coherence can't otherwise rule out a buffer type also being a pipe.
//
// `E` is the slice element type, fixed per kernel by its signature; the macro
// hardcodes it, so only `Buf` varies.
pub trait ToInput<E> {
    /// The concrete buffer type this arg resolves to. The macro pins it via
    /// `Buf = __D{n}` and bounds `__D{n}` with the right per-arg slice trait
    /// (`KernelSliceReadArg` / `…ReadWriteArg`), so no bound is needed here.
    type Buf;
    /// Wrap as a concrete or piped [`Input`].
    fn to_input(self) -> Input<Self::Buf>;
}

// A pipe of any buffer type → a deferred input. `E` is unconstrained on the
// pipe itself; the macro's `Buf = __D` + `__D: KernelSlice*Arg<E>` ties it.
impl<E, D> ToInput<E> for Pipe<D> {
    type Buf = D;
    fn to_input(self) -> Input<D> {
        Input::Pipe(self)
    }
}

/// Implement [`ToInput`] for a concrete buffer family. Per-family (not a
/// blanket) so it stays disjoint from the `Pipe<D>` impl.
macro_rules! impl_to_input_concrete {
    ($buf:ident) => {
        impl<E, M> ToInput<E> for $crate::$buf<E, M>
        where
            M: $crate::MemMode,
        {
            type Buf = $crate::$buf<E, M>;
            fn to_input(self) -> Input<$crate::$buf<E, M>> {
                Input::Concrete(self)
            }
        }
    };
}
impl_to_input_concrete!(DeviceSlice);
impl_to_input_concrete!(MappedSlice);
impl_to_input_concrete!(USMSlice);

// `Arc<DeviceSlice<E, M>>` — the shared-buffer kernel arg (read-only fan-out;
// impls `KernelSliceReadArg`). Separate impl since it's a distinct nominal type
// from the bare families above; still disjoint from `Pipe<D>`.
impl<E, M> ToInput<E> for std::sync::Arc<DeviceSlice<E, M>>
where
    M: MemMode,
{
    type Buf = std::sync::Arc<DeviceSlice<E, M>>;
    fn to_input(self) -> Input<std::sync::Arc<DeviceSlice<E, M>>> {
        Input::Concrete(self)
    }
}

// ── ExecMode: terminal-blocking opt-in ─────────────────────────────────

/// How an op should enqueue, threaded through [`execute`](EagerOp::execute).
///
/// Only the **terminal** op of a chain (the outermost one a `sync`/`wait`
/// terminal calls) ever sees [`Blocking`](ExecMode::Blocking); every upstream
/// op gets [`Pipelined`](ExecMode::Pipelined) (propagated by
/// [`AndThen::execute`]). A blocking-capable leaf (read/write/fill/copy) given
/// `Blocking` uses its native `CL_BLOCKING` enqueue — no event allocated, no
/// wait round-trip — exactly what Tier-1 `wait_on` does. Given `Pipelined` (or
/// for ops with no native blocking mode, e.g. kernels) it uses the non-blocking
/// `submit_on` + event path so downstream work can pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecMode {
    /// Non-blocking enqueue; carry a completion event forward in the pipe.
    Pipelined,
    /// This op is the chain terminal — use a native blocking enqueue if the op
    /// supports one (skips the event + wait).
    Blocking,
}

// ── EagerOp: the closure-free graph node ───────────────────────────────

/// A node in the eager graph. `execute` runs it against the context, moving its
/// output into its pipe; `describe` reports structure **without** executing.
/// Builder verbs ([`and_then`](EagerOpExt::and_then)) are on [`EagerOpExt`].
pub trait EagerOp: Send {
    /// What this op produces at run time.
    type Output: Send;

    /// The **build-time, downstream-facing handle** — what a downstream
    /// `and_then` closure receives. Defaults to a single [`Pipe<Output>`]
    /// (the common case: leaves, kernels). A multi-output combinator overrides
    /// it to expose its parts individually at build time — e.g. a 2-branch
    /// bundle sets `Handle = (A::Handle, B::Handle)` so the closure gets two
    /// pipes (`|(pa, pb)| …`) rather than one `Pipe<(A,B)>`. `Clone` so it can
    /// be handed to the closure while the op keeps its own copy.
    type Handle: Clone = Pipe<Self::Output>;

    /// The output value pipe — where `execute` deposits the result; what the
    /// terminal (`sync`) drains. Always a single `Pipe<Output>` regardless of
    /// [`Handle`](Self::Handle).
    fn output_pipe(&self) -> Pipe<Self::Output>;

    /// The downstream-facing [`Handle`](Self::Handle). Default: the output pipe
    /// (so a downstream closure gets `Pipe<Output>`). Combinators override.
    fn handle(&self) -> Self::Handle;

    /// Run the op: resolve inputs, enqueue, **move** the result + its events
    /// into the output pipe. Returns `()` — the value lives in the pipe.
    ///
    /// `mode` is [`ExecMode::Blocking`] only when this op is the chain terminal
    /// (see [`ExecMode`]); composite ops forward `Pipelined` to their upstream
    /// children and `mode` to the tail. A leaf with no native blocking enqueue
    /// ignores `mode`.
    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()>;

    /// Run this op as a **(sub)terminal** and yield `(Output, Deps)` **without
    /// waiting** on the completion events.
    ///
    /// This is the uniform gather seam. Default (single-output ops): `execute`
    /// deposits the value into [`output_pipe`](Self::output_pipe); this drains
    /// it and returns the value together with its carried [`Deps`]. Multi-output
    /// ops (whose storage is per-element pipes, not a single output pipe)
    /// override this to scatter-then-reconstruct the tuple by draining every
    /// element pipe and gathering their deps.
    ///
    /// Two callers depend on the non-blocking contract:
    /// - **Composites** (`bundle*`, `fan_out`) call `collect` on each *branch*
    ///   so a branch that is itself multi-output runs its own override instead
    ///   of being drained from an empty single pipe (the alternative —
    ///   `output_pipe().take()` — is exactly the nested-multi-output bug).
    /// - The terminals [`into_output`](Self::into_output) (blocking) and the
    ///   async `run` wrap `collect` and decide *when* to wait.
    fn collect(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
    where
        Self: Sized,
    {
        let out = self.output_pipe();
        self.execute(ec, mode)?;
        out.take()
            .ok_or(Error::NotSupported("eager graph: op produced no output"))
    }

    /// Run this op as the **chain terminal** and yield its [`Output`](Self::Output),
    /// having waited on its completion events per `mode`.
    ///
    /// Uniform across single- and multi-output ops: it [`collect`](Self::collect)s
    /// (which dispatches to the right per-op gather) then waits once on the
    /// returned deps. This is the seam that lets [`sync`](EagerOpExt::sync) be
    /// arity-agnostic. Ops never override this — they override `collect`.
    fn into_output(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<Self::Output>
    where
        Self: Sized,
    {
        let (value, deps) = self.collect(ec, mode)?;
        for d in &deps {
            d.as_ref().wait().map_err(Error::OpenCl)?;
        }
        Ok(value)
    }

    /// Structural description — node names in execution order, NO execution.
    fn describe(&self, out: &mut Vec<String>);
}

/// Builder verbs for composing [`EagerOp`]s. Blanket-implemented.
pub trait EagerOpExt: EagerOp + Sized {
    /// Sequential composition. **Eager**: runs `f` now with the upstream's
    /// build-time output [`Pipe`], stores the returned op. No closure is kept.
    fn and_then<U, F>(self, f: F) -> AndThen<Self, U>
    where
        U: EagerOp,
        F: FnOnce(Self::Handle) -> U,
    {
        let next = f(self.handle());
        AndThen { source: self, next }
    }

    /// Sequential composition whose builder runs at **execute** with the live
    /// [`ExecutionContext`] in scope (not at construction like
    /// [`and_then`](Self::and_then)). The closure receives `&ExecutionContext`
    /// together with the upstream's runtime value, so it can read `ec.device()`
    /// / `ec.context()` or route via [`on_device`](Self::on_device) while
    /// building the downstream op. See [`AndThenWithContext`].
    fn and_then_with_context<U, F>(self, f: F) -> AndThenWithContext<Self, U, F>
    where
        U: EagerOp,
        F: for<'a> FnOnce(&ExecutionContext<'a>, Self::Output) -> U + Send,
    {
        let src_pipe = self.output_pipe();
        AndThenWithContext {
            source: self,
            src_pipe,
            f: Some(f),
            out: Pipe::new(),
        }
    }

    /// Route this op's `execute` to `device`'s default out-of-order queue
    /// instead of the chain's primary queue. Downstream stages resume on the
    /// parent's queue; the routed op's events are valid across both via
    /// OpenCL's shared-context event semantics. See [`OnDevice`].
    fn on_device(self, device: &crate::Device) -> OnDevice<Self> {
        let src_pipe = self.output_pipe();
        OnDevice {
            source: self,
            device: device.clone(),
            src_pipe,
            out: Pipe::new(),
        }
    }

    /// Run a host closure on a borrowed [`Mappable::View`](crate::mappable::Mappable::View) of this op's output,
    /// in chain order. The seam drains the upstream events (so the data is
    /// host-valid), maps the value, runs the closure (mutations persist via the
    /// unmap), then forwards the same value downstream. Errors from the closure
    /// propagate directly. See [`AndThenHost`].
    fn and_then_host<F>(self, f: F) -> AndThenHost<Self, F>
    where
        Self::Output: crate::mappable::Mappable,
        F: for<'a> FnOnce(<Self::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
            + Send,
    {
        let src_pipe = self.output_pipe();
        AndThenHost {
            source: self,
            src_pipe,
            f: Some(f),
            out: Pipe::new(),
        }
    }

    /// Like [`and_then_host`](Self::and_then_host) but the closure also receives
    /// the running [`Context`] (e.g. to read device props). See
    /// [`AndThenHostWithContext`].
    fn and_then_host_with_context<F>(self, f: F) -> AndThenHostWithContext<Self, F>
    where
        Self::Output: crate::mappable::Mappable,
        F: for<'a> FnOnce(
                &Context,
                <Self::Output as crate::mappable::Mappable>::View<'a>,
            ) -> Result<()>
            + Send,
    {
        let src_pipe = self.output_pipe();
        AndThenHostWithContext {
            source: self,
            src_pipe,
            f: Some(f),
            out: Pipe::new(),
        }
    }

    /// Run `self` to completion on `context` (forward path; no replay). Blocks
    /// once, here, on the terminal op's events — the only wait in the graph.
    fn sync(self, context: &Context) -> Result<Self::Output> {
        let device = context.device().clone();
        let queue = context.default_outoforder_queue(&device)?;
        let ec = ExecutionContext::new(context, device, queue.raw());
        // The terminal op yields its Output and waits on its own completion
        // events (Blocking); upstream ops pipeline. Single-output ops use the
        // default `into_output` (drain output pipe + wait); multi-output ops
        // override it to scatter-then-reconstruct the tuple.
        self.into_output(&ec, ExecMode::Blocking)
    }

    /// Async terminal — run `self` on `context` and return a future that
    /// resolves to its [`Output`](EagerOp::Output) once every command the
    /// chain enqueued has completed on the device.
    ///
    /// The non-blocking analog of [`sync`](Self::sync): instead of draining
    /// the output pipe and *blocking* on its [`Deps`], `run` runs `execute`
    /// in [`ExecMode::Pipelined`], drains the output pipe, then enqueues an
    /// `clEnqueueMarkerWithWaitList` over the chain's deps on the same OOO
    /// queue and wraps it in an [`EventFuture`](crate::EventFuture) — the
    /// Tier-1 `clSetEventCallback` + `AtomicWaker` machinery wakes the
    /// future when the marker fires. Mirrors the structure of the old
    /// `chain_future::run_chain` terminal.
    ///
    /// **Host errors surface synchronously.** Unlike the old closure layer
    /// (where `and_then_host` workers ran on their own threads and stashed
    /// into an `Arc<Mutex<Option<Error>>>` slot read at poll time), the
    /// eager host seam runs its closure *inside* `execute` and returns the
    /// closure's `Err` directly (see `run_host_seam`). So a failing chain
    /// returns [`EagerChainFuture::Errored`] right here — there is no
    /// host-error slot to drain at poll time.
    ///
    /// Arity-agnostic: like [`sync`](Self::sync), `run` gathers via
    /// [`collect`](EagerOp::collect), so multi-output terminals (`arc_split`,
    /// `bundle*`, the `CopyTo` pair) reconstruct their tuple/array the same way
    /// the blocking terminal does — the future then resolves to that value.
    #[cfg(feature = "async-events")]
    fn run(self, context: &Context) -> EagerChainFuture<Self::Output>
    where
        Self::Output: Unpin,
    {
        run_eager_chain(self, context)
    }

    /// Describe the whole graph structurally without running it.
    fn description(&self) -> Vec<String> {
        let mut v = Vec::new();
        self.describe(&mut v);
        v
    }
}
impl<T: EagerOp> EagerOpExt for T {}

// ── AndThen: source then next; next eagerly built over source's pipe ───

/// Sequential composition node. Holds the source op and the **already-built**
/// downstream op (which reads the source's output via a [`Pipe`]). No `FnOnce`.
pub struct AndThen<S, U> {
    source: S,
    next: U,
}

impl<S, U> EagerOp for AndThen<S, U>
where
    S: EagerOp,
    U: EagerOp,
{
    type Output = U::Output;
    // The chain's downstream handle is the tail op's handle.
    type Handle = U::Handle;

    fn output_pipe(&self) -> Pipe<U::Output> {
        self.next.output_pipe()
    }

    fn handle(&self) -> U::Handle {
        self.next.handle()
    }

    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // Source is always upstream → must pipeline (its output feeds `next`).
        // Only the tail inherits the caller's `mode` (Blocking iff terminal).
        self.source.execute(ec, ExecMode::Pipelined)?;
        self.next.execute(ec, mode)
    }

    fn collect(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(U::Output, Deps)>
    where
        Self: Sized,
    {
        // Delegate the gather to the tail op so a multi-output `next` (bundle*,
        // arc_split, CopyTo pair) runs its *overridden* `collect`
        // (scatter-then-reconstruct over its per-element pipes) rather than the
        // default single-pipe drain — whose `output_pipe` it never fills. The
        // source pipelines; only the tail observes the terminal `mode`.
        self.source.execute(ec, ExecMode::Pipelined)?;
        self.next.collect(ec, mode)
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        self.next.describe(out);
    }
}

// ── Value: lift a host value into the graph ────────────────────────────

/// A host value lifted into the graph — produces it with no device work and no
/// events. Useful as a chain head or to thread a host value alongside buffers.
pub struct Value<T: Send> {
    v: Option<T>,
    out: Pipe<T>,
}

/// Lift `v` into the graph.
pub fn value<T: Send + 'static>(v: T) -> Value<T> {
    Value {
        v: Some(v),
        out: Pipe::new(),
    }
}

impl<T: Send + 'static> EagerOp for Value<T> {
    type Output = T;

    fn output_pipe(&self) -> Pipe<T> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, _ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let v = self
            .v
            .take()
            .expect("Value::execute called twice — internal eager bug");
        self.out.put(v, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("value".into());
    }
}

// ── Forward: select/identity — make one upstream Pipe a single-output op ──

/// Forward a single upstream value (a `Pipe<T>`) onward as a single-output
/// [`EagerOp`]. The identity op: it resolves its input and re-deposits it
/// (threading the deps), changing nothing. Its purpose is **shape**, not work —
/// it lets you pick ONE element out of a multi-output op's handle (e.g. a
/// kernel's `(Pipe<a>, Pipe<b>, Pipe<out>)`, or a bundle's per-branch pipes) and
/// continue on-device with that single value, instead of dropping to the host
/// or inserting a no-op kernel. The selected pipe becomes a normal
/// `EagerOp<Output = T>` that composes via `and_then` / `bundle` like any leaf.
///
/// ```ignore
/// // pick `out` from add_u32's 3-tuple handle and keep going on-device:
/// ks.add_u32([N], a, b, out).and_then(|(_a, _b, out)| forward(out))
/// ```
pub struct Forward<T: Send> {
    input: Input<T>,
    out: Pipe<T>,
}

/// Forward an upstream value onward unchanged (identity op; see [`Forward`]).
/// Takes a [`Pipe<T>`] directly (the selected element of a multi-output handle)
/// — `Pipe` rather than `impl Into<Input>` so `T` infers cleanly from the
/// selected element without an annotation.
pub fn forward<T: Send + 'static>(pipe: Pipe<T>) -> Forward<T> {
    Forward {
        input: Input::Pipe(pipe),
        out: Pipe::new(),
    }
}

impl<T: Send + 'static> EagerOp for Forward<T> {
    type Output = T;

    fn output_pipe(&self) -> Pipe<T> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, _ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Resolve the upstream value + its events and re-deposit unchanged — no
        // device work; deps threaded through so ordering/termination is intact.
        let (v, deps) = self.input.resolve()?;
        self.out.put(v, deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("forward".into());
    }
}

// ── EagerDynOp: type-erased single-output op for conditional graphs ─────

/// Object-safe erasure of [`EagerOp`], specialised to output `T`. Crate-internal
/// — users go through [`EagerDynOp`]. `EagerOp` itself is NOT object-safe (it has
/// an associated `Handle` type and `self`-consuming `collect`/`into_output`), so
/// this mirror trait restates the one operation a terminal/branch needs —
/// gather `(value, deps)` — as a `self: Box<Self>` method that *is*
/// dyn-dispatchable. It delegates to the concrete op's [`collect`](EagerOp::collect),
/// which already reconstructs any arity down to a single `Output`, so even a
/// multi-output inner op erases cleanly to a single-output `EagerDynOp`.
trait ErasedEagerOp<T>: Send {
    fn collect_erased(
        self: Box<Self>,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(T, Deps)>;

    fn describe_erased(&self, out: &mut Vec<String>);
}

impl<O> ErasedEagerOp<O::Output> for O
where
    O: EagerOp,
{
    fn collect_erased(
        self: Box<Self>,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(O::Output, Deps)> {
        (*self).collect(ec, mode)
    }

    fn describe_erased(&self, out: &mut Vec<String>) {
        self.describe(out);
    }
}

/// Type-erased single-output [`EagerOp`] yielding `T`. Lets `if` / `match` arms
/// produce DIFFERENT concrete op types as long as they agree on `Output` — the
/// eager analog of the legacy closure-layer `DynOp`.
///
/// Each combinator chain has its own deeply-nested concrete type
/// (`AndThen<Upload, AndThen<…>>` vs `Value<T>`), so an `if`/`else` that builds a
/// chain in each arm is a type-mismatch error. Wrapping each arm in
/// `EagerDynOp::new(...)` erases the concrete type to one nominal
/// `EagerDynOp<'op, T>`, which is itself an [`EagerOp`] and composes with
/// `and_then` / `bundle` / `fan_out` like any single-output leaf.
///
/// ```ignore
/// let chain: EagerDynOp<u32> = if use_kernel {
///     EagerDynOp::new(upload(v).and_then(|b| ks.fill_u32([N], b, 9)).and_then(|_| value(0u32)))
/// } else {
///     EagerDynOp::new(value(0u32))            // different concrete type, same Output
/// };
/// let r = chain.sync(&ctx)?;
/// ```
///
/// One heap allocation per erased op. The `'op` lifetime lets the boxed op borrow
/// from the surrounding scope (typically `&Kernels` for kernel launches); it
/// infers to `'static` for chains built from owned data only.
///
/// **Single-output.** `Handle = Pipe<T>` (the default). A multi-output op CAN be
/// erased — its tuple `Output` becomes the `T` of the `EagerDynOp` (reconstructed
/// via the inner op's `collect`), but the per-element build-time handle is gone;
/// downstream sees one `Pipe<tuple>`. For the conditional-graph use case (arms
/// agreeing on one `Output`) that is exactly right.
pub struct EagerDynOp<'op, T> {
    inner: Option<Box<dyn ErasedEagerOp<T> + 'op>>,
    out: Pipe<T>,
}

impl<'op, T: Send + 'static> EagerDynOp<'op, T> {
    /// Erase a concrete op into a single-output `EagerDynOp`. Both arms of an
    /// `if`/`match` can produce `EagerDynOp::new(...)` of the same `T` without
    /// their concrete types matching.
    pub fn new<O>(op: O) -> Self
    where
        O: EagerOp<Output = T> + 'op,
    {
        EagerDynOp {
            inner: Some(Box::new(op)),
            out: Pipe::new(),
        }
    }
}

impl<T: Send + 'static> EagerOp for EagerDynOp<'_, T> {
    type Output = T;

    fn output_pipe(&self) -> Pipe<T> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // Gather the erased inner op (any arity → one value + deps) and deposit
        // into our own pipe, so the default collect/into_output/handle path treats
        // this as an ordinary single-output leaf. The inner op observes `mode`
        // (it is the real terminal work when this EagerDynOp is the chain tail).
        let inner = self
            .inner
            .take()
            .expect("EagerDynOp::execute called twice — internal eager bug");
        let (v, deps) = inner.collect_erased(ec, mode)?;
        self.out.put(v, deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("dyn_op{".into());
        if let Some(inner) = &self.inner {
            inner.describe_erased(out);
        }
        out.push("}".into());
    }
}

// ── Arced: wrap the output in Arc<T> ───────────────────────────────────

/// Wrap an upstream op's output in [`Arc`] for shared fan-out. Passes events
/// through unchanged.
pub struct Arced<S: EagerOp> {
    source: S,
    src_pipe: Pipe<S::Output>,
    out: Pipe<Arc<S::Output>>,
}

/// Wrap `source`'s output in `Arc`.
pub fn arced<S: EagerOp>(source: S) -> Arced<S>
where
    S::Output: Sync,
{
    let src_pipe = source.output_pipe();
    Arced {
        source,
        src_pipe,
        out: Pipe::new(),
    }
}

impl<S> EagerOp for Arced<S>
where
    S: EagerOp,
    S::Output: Sync,
{
    type Output = Arc<S::Output>;

    fn output_pipe(&self) -> Pipe<Arc<S::Output>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // Source pipelines (its value feeds us); we add no device work, so the
        // terminal `mode` is irrelevant — pass it through for symmetry.
        self.source.execute(ec, mode)?;
        let (v, deps) = self.src_pipe.take().ok_or(Error::NotSupported(
            "eager arced: source produced no output",
        ))?;
        self.out.put(Arc::new(v), deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("arced".into());
    }
}

// ── ArcSplit: fan one Arc output to N read-only branches ───────────────

/// Fan a single `Arc<T>` upstream output out to `N` independent downstream
/// branches, each receiving its own cheap `Arc::clone`. Mirrors the closure
/// layer's `.arc().and_then(|arc| { let [a, b, c] = arc.split::<3>(); … })`:
/// one producer, `N` read-only consumers.
///
/// `Handle = [Pipe<S::Output>; N]` — the downstream `and_then` closure
/// destructures the array (`let [a, b, c] = handle`) and each element is a
/// `Pipe<Arc<T>>` that flows into a branch op (e.g. a read-only kernel, or a
/// download). `execute` runs the source once, then scatters `Arc::clone(&arc)`
/// (plus a clone of the producer's wait-list) into each of the `N` element
/// pipes — every branch sees the same value and waits on the same producer
/// event. `Output = [S::Output; N]` (the `N` clones) for the terminal case.
///
/// Use [`arc_split`] to build one — it follows an [`arced`] source.
pub struct ArcSplit<S: EagerOp, const N: usize>
where
    S::Output: Clone,
{
    source: S,
    src_pipe: Pipe<S::Output>,
    // One element pipe per branch (move-once storage); each gets an
    // `Arc::clone` of the source value in `execute`.
    outs: [Pipe<S::Output>; N],
}

/// Build an [`ArcSplit`]: fan `source`'s `Arc<T>` output to `N` read-only
/// branches. `source` is typically an [`arced`] op (`Output = Arc<T>`), so the
/// per-branch clone is a cheap refcount bump. Pick `N` via turbofish to match
/// the destructure arity: `arc_split::<3, _>(arced(upload(…)))`.
pub fn arc_split<const N: usize, S: EagerOp>(source: S) -> ArcSplit<S, N>
where
    S::Output: Clone,
{
    let src_pipe = source.output_pipe();
    ArcSplit {
        source,
        src_pipe,
        outs: std::array::from_fn(|_| Pipe::new()),
    }
}

impl<S, const N: usize> EagerOp for ArcSplit<S, N>
where
    S: EagerOp,
    S::Output: Clone,
{
    type Output = [S::Output; N];
    // An array of N element pipes; the downstream closure does
    // `let [a, b, c] = handle` and routes each pipe into its own branch.
    type Handle = [Pipe<S::Output>; N];

    fn output_pipe(&self) -> Pipe<Self::Output> {
        // Multi-output storage is the per-element pipes; this single pipe is
        // never filled or drained (the default `into_output` is overridden, and
        // `and_then` uses `handle()`). Return a fresh empty pipe — well-typed,
        // never read.
        Pipe::new()
    }

    fn handle(&self) -> Self::Handle {
        self.outs.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // The source feeds us; we add no device work. Pipeline it, then take its
        // value + completion events and scatter a clone of each into every
        // branch pipe (Arc::clone is a cheap refcount bump; Deps clone shares
        // the same producer events).
        self.source.execute(ec, ExecMode::Pipelined)?;
        let (v, deps) = self.src_pipe.take().ok_or(Error::NotSupported(
            "eager arc_split: source produced no output",
        ))?;
        for out in &self.outs {
            out.put(v.clone(), deps.clone());
        }
        Ok(())
    }

    fn collect(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
    where
        Self: Sized,
    {
        // Grab the element pipes before consuming `self`, scatter via `execute`,
        // then drain all N to reconstruct the `[clone; N]` array, gathering
        // every branch's deps (the terminal `into_output` waits on them once).
        let outs = self.outs.clone();
        self.execute(ec, mode)?;
        let mut all_deps: Deps = Deps::new();
        let mut vals: Vec<S::Output> = Vec::with_capacity(N);
        for p in &outs {
            let (v, d) = p.take().ok_or(Error::NotSupported(
                "eager arc_split: a branch produced no output",
            ))?;
            vals.push(v);
            all_deps.extend(d);
        }
        // `vals` has exactly N elements (one per element pipe) — the conversion
        // cannot fail.
        let arr = vals
            .try_into()
            .unwrap_or_else(|_| unreachable!("arc_split drained exactly N branch pipes"));
        Ok((arr, all_deps))
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push(format!("arc_split[{N}]"));
    }
}

// ── Bundle: N independent branches, joined by a marker ─────────────────

/// Join `n` branch wait-lists into one marker event on `ec`'s queue.
fn join_marker(ec: &ExecutionContext<'_>, branch_deps: &[Deps]) -> Result<Deps> {
    use crate::Launcher;
    let all: Vec<crate::cl_event> = branch_deps
        .iter()
        .flat_map(|d| d.iter().map(|e| e.as_ref().get()))
        .collect();
    // SAFETY: cl_event handles are valid — held by the branch `Deps` Arcs
    // until this call returns.
    let marker =
        unsafe { ec.cl_queue().enqueue_marker_with_wait_list(&all) }.map_err(Error::OpenCl)?;
    Ok(vec![wrap_event(marker)])
}

macro_rules! impl_eager_bundle {
    ($name:ident, $ctor:ident, $($field:ident : $ty:ident : $pf:ident),+) => {
        #[doc = concat!("Eager bundle of independent branches (arity ",
            stringify!($name), "). Built by [`", stringify!($ctor),
            "`]; branches run with no inter-ordering, joined by a marker.")]
        pub struct $name<$($ty: EagerOp),+> {
            $($field: $ty,)+
            // Each branch's output pipe, captured at build. These are the
            // move-once storage (like `CopyTo2`'s element pipes): the branch
            // fills its own pipe at `execute`; `handle()` exposes clones so a
            // downstream multi-arg op (e.g. a kernel) can pull each branch as a
            // separate `Pipe<buffer>` input; `into_output` drains them for the
            // terminal-tuple case.
            $($pf: Pipe<<$ty as EagerOp>::Output>,)+
        }

        #[doc = concat!("Construct an eager [`", stringify!($name), "`].")]
        #[allow(clippy::too_many_arguments)]
        pub fn $ctor<$($ty: EagerOp),+>($($field: $ty),+) -> $name<$($ty),+> {
            $(let $pf = $field.output_pipe();)+
            $name { $($field,)+ $($pf,)+ }
        }

        impl<$($ty: EagerOp),+> EagerOp for $name<$($ty),+> {
            type Output = ( $(<$ty as EagerOp>::Output,)+ );
            // A tuple of each branch's OWN output pipe. The downstream closure
            // gets `(pa, pb, …)` — one `Pipe<branch output>` per branch — so a
            // multi-arg op can consume them as separate inputs (each is a
            // `Pipe<buffer>`, i.e. `ToInput`). Mirrors `CopyTo2`/the
            // macro-emitted multi-output kernel.
            type Handle = ( $(Pipe<<$ty as EagerOp>::Output>,)+ );

            fn output_pipe(&self) -> Pipe<Self::Output> {
                // Multi-output storage is the per-branch pipes; this single pipe
                // is never filled or drained (the default `into_output` is
                // overridden, and `and_then` uses `handle()`). Return a fresh
                // empty pipe — well-typed, never read.
                Pipe::new()
            }

            fn handle(&self) -> Self::Handle {
                ( $(self.$pf.clone(),)+ )
            }

            fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
                // Each branch pipelines (independent). We `collect` the branch —
                // NOT `branch.execute` — so a branch that is *itself* multi-output
                // (a nested bundle, arc_split, the copy pair) runs its own gather
                // and yields a single reconstructed value; we then deposit that
                // value into the branch's `$pf` pipe. This keeps `$pf` filled
                // uniformly for both consumers of a bundle: the mid-graph
                // `handle()` (a downstream `and_then` reading `$pf`) and the
                // terminal `collect` below. (For a single-output branch, `$pf`
                // is the branch's own output pipe; `collect` drains it and we put
                // it straight back — a cheap round-trip.)
                $(
                    let (v, d) = self.$field.collect(ec, ExecMode::Pipelined)?;
                    self.$pf.put(v, d);
                )+
                Ok(())
            }

            fn collect(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
            where
                Self: Sized,
            {
                // Grab the branch pipes before consuming `self`, scatter via
                // `execute` (which fills each `$pf` via the branch's own gather),
                // then drain each to reconstruct the tuple, joining the branch
                // wait-lists into one marker. The terminal `into_output` waits.
                $(let $pf = self.$pf.clone();)+
                self.execute(ec, mode)?;
                let mut branch_deps: Vec<Deps> = Vec::new();
                let outputs = ( $({
                    let (v, d) = $pf.take().ok_or(Error::NotSupported(
                        "eager bundle: a branch produced no output"))?;
                    branch_deps.push(d);
                    v
                },)+ );
                let joined = join_marker(ec, &branch_deps)?;
                Ok((outputs, joined))
            }

            fn describe(&self, out: &mut Vec<String>) {
                out.push(concat!(stringify!($name), "{").into());
                $(self.$field.describe(out);)+
                out.push("}".into());
            }
        }
    };
}

impl_eager_bundle!(Bundle2, bundle2, a: A: pa, b: B: pb);
impl_eager_bundle!(Bundle3, bundle3, a: A: pa, b: B: pb, c: C: pc);
impl_eager_bundle!(Bundle4, bundle4, a: A: pa, b: B: pb, c: C: pc, d: D: pd);
impl_eager_bundle!(Bundle5, bundle5, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe);
impl_eager_bundle!(Bundle6, bundle6, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf);
impl_eager_bundle!(Bundle7, bundle7, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg);
impl_eager_bundle!(Bundle8, bundle8, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph);
impl_eager_bundle!(Bundle9, bundle9, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi);
impl_eager_bundle!(Bundle10, bundle10, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj);
impl_eager_bundle!(Bundle11, bundle11, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk);
impl_eager_bundle!(Bundle12, bundle12, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl);
impl_eager_bundle!(Bundle13, bundle13, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl, m: M: pm);
impl_eager_bundle!(Bundle14, bundle14, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl, m: M: pm, n: N: pn);
impl_eager_bundle!(Bundle15, bundle15, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl, m: M: pm, n: N: pn, o: O: po);
impl_eager_bundle!(Bundle16, bundle16, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl, m: M: pm, n: N: pn, o: O: po, p: P: pp);

/// Variadic constructor for the eager [`Bundle2`] through [`Bundle16`] — picks
/// the right `bundleN` based on the number of arguments.
///
/// The eager analog of the legacy [`bundle!`](crate::bundle!) macro (which still
/// targets the closure layer during the cutover). Renamed to `bundle!` once the
/// old layer is removed.
///
/// ```ignore
/// let (a, b) = eager_bundle!(op_a, op_b).sync(&ctx)?;
/// let (a, b, c) = eager_bundle!(op_a, op_b, op_c).sync(&ctx)?;
/// // ... up to 16 children
/// ```
#[macro_export]
macro_rules! eager_bundle {
    ($a:expr, $b:expr $(,)?) => {
        $crate::eager::bundle2($a, $b)
    };
    ($a:expr, $b:expr, $c:expr $(,)?) => {
        $crate::eager::bundle3($a, $b, $c)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr $(,)?) => {
        $crate::eager::bundle4($a, $b, $c, $d)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr $(,)?) => {
        $crate::eager::bundle5($a, $b, $c, $d, $e)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr $(,)?) => {
        $crate::eager::bundle6($a, $b, $c, $d, $e, $f)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr $(,)?) => {
        $crate::eager::bundle7($a, $b, $c, $d, $e, $f, $g)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr $(,)?) => {
        $crate::eager::bundle8($a, $b, $c, $d, $e, $f, $g, $h)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr $(,)?) => {
        $crate::eager::bundle9($a, $b, $c, $d, $e, $f, $g, $h, $i)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr $(,)?) => {
        $crate::eager::bundle10($a, $b, $c, $d, $e, $f, $g, $h, $i, $j)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr $(,)?) => {
        $crate::eager::bundle11($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr $(,)?) => {
        $crate::eager::bundle12($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr, $m:expr $(,)?) => {
        $crate::eager::bundle13($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr, $m:expr, $n:expr $(,)?) => {
        $crate::eager::bundle14($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr, $m:expr, $n:expr, $o:expr $(,)?) => {
        $crate::eager::bundle15($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr, $m:expr, $n:expr, $o:expr, $p:expr $(,)?) => {
        $crate::eager::bundle16(
            $a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p,
        )
    };
}

// ── FanOut: a homogeneous Vec of branches, joined by a marker ──────────

/// Eager fan-out: build one op per input (the builder `f` runs at construction
/// — eager — over the known input list), run them independently, join via a
/// marker. Output is `Vec<U::Output>`.
pub struct FanOut<U: EagerOp> {
    ops: Vec<U>,
    pipes: Vec<Pipe<U::Output>>,
    out: Pipe<Vec<U::Output>>,
}

/// Build a fan-out: `f` is called now for each input, producing the branch ops.
pub fn fan_out<I, F, U>(inputs: Vec<I>, mut f: F) -> FanOut<U>
where
    F: FnMut(I) -> U,
    U: EagerOp,
{
    let ops: Vec<U> = inputs.into_iter().map(&mut f).collect();
    let pipes: Vec<Pipe<U::Output>> = ops.iter().map(|o| o.output_pipe()).collect();
    FanOut {
        ops,
        pipes,
        out: Pipe::new(),
    }
}

/// Method-call form of [`fan_out`]: `vec![a, b].fan_out(|i| value(i))`.
///
/// Mirrors the old closure-layer `FanOutExt`. Reads as data → operation and
/// composes cleanly with downstream `.and_then`; the free-fn form stays
/// available — use whichever fits the call site. Named `EagerFanOutExt` to
/// avoid clashing with the old [`FanOutExt`](crate::FanOutExt) (both are
/// re-exported at the crate root).
pub trait EagerFanOutExt<I>: Sized {
    /// See [`fan_out`] — this delegates to it.
    fn fan_out<F, U>(self, f: F) -> FanOut<U>
    where
        F: FnMut(I) -> U,
        U: EagerOp;
}

impl<I> EagerFanOutExt<I> for Vec<I> {
    fn fan_out<F, U>(self, f: F) -> FanOut<U>
    where
        F: FnMut(I) -> U,
        U: EagerOp,
    {
        fan_out(self, f)
    }
}

impl<U: EagerOp> EagerOp for FanOut<U> {
    type Output = Vec<U::Output>;

    fn output_pipe(&self) -> Pipe<Vec<U::Output>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // `collect` each branch op (not `execute`) so a multi-output branch runs
        // its own gather and yields one reconstructed value + deps — `self.pipes`
        // (captured single output pipes) are empty for such branches. The pipes
        // field is now unused for gathering; we read values straight from
        // `collect`.
        let n = self.ops.len();
        let mut branch_deps: Vec<Deps> = Vec::with_capacity(n);
        let mut outputs: Vec<U::Output> = Vec::with_capacity(n);
        for op in self.ops {
            let (v, d) = op.collect(ec, ExecMode::Pipelined)?;
            outputs.push(v);
            branch_deps.push(d);
        }
        let joined = join_marker(ec, &branch_deps)?;
        self.out.put(outputs, joined);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("fan_out[{}]", self.pipes.len()));
    }
}

/// Allocate a zero-initialised `DeviceSlice<T, M>` of `len` elements. Eager
/// leaf: produces a usable buffer, no upstream input. (`alloc_zero` is
/// synchronous internally, so it carries no in-flight events.)
pub struct AllocZero<T, M: MemMode = ReadWrite> {
    len: usize,
    out: Pipe<DeviceSlice<T, M>>,
    _t: PhantomData<fn() -> T>,
}

/// Build a zero-init alloc leaf.
pub fn alloc_zero<T, M>(len: usize) -> AllocZero<T, M>
where
    T: Copy + Default + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    AllocZero {
        len,
        out: Pipe::new(),
        _t: PhantomData,
    }
}

impl<T, M> EagerOp for AllocZero<T, M>
where
    T: Copy + Default + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // alloc_zero is synchronous internally; no in-flight event, mode N/A.
        let buf = DeviceSlice::<T, M>::alloc_zero(ec.context(), self.len)?;
        self.out.put(buf, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("alloc_zero(len={})", self.len));
    }
}

// ── Leaf: in-place fill (eager port of DeviceSliceFillOp) ──────────────

/// Fill a buffer (upstream pipe or concrete) with `value` via a non-blocking
/// `clEnqueueFillBuffer`, threading the upstream events as the wait-list.
pub struct Fill<T: Copy, M: MemMode> {
    buf: Input<DeviceSlice<T, M>>,
    value: T,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build a fill leaf over an upstream buffer.
pub fn fill<T, M>(buf: impl Into<Input<DeviceSlice<T, M>>>, value: T) -> Fill<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    Fill {
        buf: buf.into(),
        value,
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for Fill<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (mut buf, deps) = self.buf.resolve()?;
        match mode {
            // Terminal: native blocking fill (CL_BLOCKING) — no event, the
            // driver waits. Empty deps forward (nothing left to await).
            ExecMode::Blocking => {
                buf.fill(self.value)
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .wait_on(ec)?;
                self.out.put(buf, Deps::new());
            }
            // Pipelined: non-blocking; carry the event for downstream ordering.
            ExecMode::Pipelined => {
                let event = buf
                    .fill(self.value)
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .submit_on(ec)?;
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill".into());
    }
}

// ── Leaf: upload (host → device, alloc + CL_MEM_COPY_HOST_PTR) ──────────

/// Allocate a `DeviceSlice<T, M>` and bake `src` into it at creation
/// (`CL_MEM_COPY_HOST_PTR`). A chain-entry leaf — no upstream input. (Uses the
/// from_slice path: works for any marker, one synchronous create, no in-flight
/// event.)
pub struct Upload<T: Copy, M: MemMode = ReadWrite> {
    src: Option<UploadSource<T>>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an upload leaf from any `Vec<T>` / `Box<[T]>` / `Arc<[T]>`.
pub fn upload<T, M, S>(src: S) -> Upload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
    S: Into<UploadSource<T>>,
{
    Upload {
        src: Some(src.into()),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for Upload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // from_slice (CL_MEM_COPY_HOST_PTR) is a synchronous create — no
        // in-flight event, mode N/A.
        let src = self
            .src
            .take()
            .expect("Upload::execute called twice — internal eager bug");
        let buf = DeviceSlice::<T, M>::from_slice(ec.context(), src.as_slice())?;
        self.out.put(buf, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("upload".into());
    }
}

// ── Leaf: download (device → host Vec, non-blocking read) ──────────────

/// Consume an upstream buffer, alloc a host `Vec<T>`, non-blocking-read into it
/// threading the upstream events. Output is the `Vec<T>`.
pub struct Download<T, M: MemMode = ReadWrite> {
    buf: Input<DeviceSlice<T, M>>,
    out: Pipe<Vec<T>>,
}

/// Build a download leaf over an upstream buffer.
pub fn download<T, M>(buf: impl Into<Input<DeviceSlice<T, M>>>) -> Download<T, M>
where
    T: Clone + Default + Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    Download {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for Download<T, M>
where
    T: Clone + Default + Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    type Output = Vec<T>;

    fn output_pipe(&self) -> Pipe<Vec<T>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve()?;
        let mut host = vec![T::default(); buf.len()];
        match mode {
            // Terminal: native blocking read (CL_BLOCKING) — the driver waits,
            // the host Vec is valid on return, no event. Matches Tier-1
            // `ReadOp::wait_on`; restores parity for `…download().sync()`.
            ExecMode::Blocking => {
                buf.read(&mut host)
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .wait_on(ec)?;
                self.out.put(host, Deps::new());
            }
            // Pipelined: non-blocking; the event gates the Vec being valid.
            ExecMode::Pipelined => {
                let event = buf
                    .read(&mut host)
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .submit_on(ec)?;
                self.out.put(host, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("download".into());
    }
}

// ── Leaf: migrate a DeviceSlice to another device (eager TransferToDevice) ──

/// Eager port of the closure-layer `transfer_to_device(buf, &dev)`. Enqueues a
/// `clEnqueueMigrateMemObjects` for the buffer on `device`'s default OOO queue,
/// yielding the (now-migrated) buffer. The matching per-op routing combinator
/// kernels need after the buffer is migrated is [`on_device`](EagerOpExt::on_device).
///
/// ## Shape: a leaf, not a wrapping method
///
/// Unlike [`on_device`](EagerOpExt::on_device) (which *routes* an upstream op's
/// own enqueue to another queue without touching its value), `transfer_to_device`
/// is a buffer-*consuming* leaf: it resolves the upstream `DeviceSlice` value,
/// reads its `cl_mem`, and enqueues a migrate. That puts it in the same family as
/// [`download`] / [`fill`] / [`copy_to`](crate::eager::eager_copy_to) — every member
/// takes `impl Into<Input<DeviceSlice<…>>>` as its dataflow input — and mirrors
/// the old free-fn signature `transfer_to_device(buf, dev)` 1:1. A method form
/// would have to pin `S::Output = DeviceSlice<T>` (like [`OnDevice`]) yet still
/// resolve the value (unlike `OnDevice`), fighting both patterns; the leaf form
/// composes cleanly via `.and_then(|p| transfer_to_device(p, dev))`.
///
/// ## What the migrate actually does
///
/// For two devices sharing one `cl_context`, the runtime may or may not move
/// bytes (shared-memory topologies / sub-devices: typically a no-op; two dGPUs:
/// real migration). Either way the migrate is a queue command (non-blocking) so
/// the graph stays pipelined; downstream stages wait on the migrate event via
/// the carried [`Deps`]. Cross-*context* transfer is **not** this op — that goes
/// through host bounce ([`download`] → [`upload`]).
pub struct TransferToDevice<T, M: MemMode = ReadWrite> {
    buf: Input<DeviceSlice<T, M>>,
    device: crate::Device,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build a transfer-to-device leaf: migrate `buf` onto `device`'s default OOO
/// queue, yielding the migrated buffer. See [`TransferToDevice`] for semantics
/// and the rationale for the leaf (free-fn) shape over a wrapping method.
pub fn transfer_to_device<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
    device: &crate::Device,
) -> TransferToDevice<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    TransferToDevice {
        buf: buf.into(),
        device: device.clone(),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for TransferToDevice<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve()?;
        // Resolve the target device's default OOO queue (cached on the Context,
        // so the terminal's flush_all_outoforder_queues pushes it). Same path
        // OnDevice uses to reach a non-primary device's queue.
        let target_q = ec.context().default_outoforder_queue(&self.device)?;
        // Enqueue the migrate with the upstream events as the wait-list, on the
        // target queue (`&*target_q` is the `Queue: Launcher`). Non-blocking —
        // mode is ignored; the chain terminal's `into_output` does the final
        // wait. The migrate body mirrors the closure layer's
        // `transfer_to_device.rs` exactly.
        let event = buf
            .migrate()
            .after_all(deps.iter().map(|d| d.as_ref()))
            .submit_on(&*target_q)?;
        self.out.put(buf, vec![wrap_event(event)]);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("transfer_to_device".into());
    }
}

// ════════════════════════════════════════════════════════════════════════
// uninit_ext.rs ports — fill / write an alloc-uninit buffer → initialised
// ════════════════════════════════════════════════════════════════════════

// ── Leaf: fill a DeviceSliceUninit → DeviceSlice (eager FillFromUninitOp) ──

/// Consume an uninit `DeviceSlice` (upstream pipe or concrete) and fill it
/// with `value`, yielding the initialised buffer. Mirrors [`Fill`] (transform
/// shape, ExecMode branch on the Tier-1 `fill` builder's `wait_on`/`submit_on`).
pub struct FillDeviceUninit<T: Copy, M: MemMode> {
    uninit: Input<DeviceSliceUninit<T, M>>,
    value: T,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an eager fill-from-uninit leaf over a `DeviceSliceUninit`.
pub fn fill_device_uninit<T, M>(
    uninit: impl Into<Input<DeviceSliceUninit<T, M>>>,
    value: T,
) -> FillDeviceUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    FillDeviceUninit {
        uninit: uninit.into(),
        value,
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for FillDeviceUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve()?;
        // SAFETY: the fill below writes every byte; downstream gates on the
        // returned fill event (Pipelined) or the driver waits (Blocking).
        let mut buf = unsafe { uninit.assume_init() };
        match mode {
            ExecMode::Blocking => {
                buf.fill(self.value)
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .wait_on(ec)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                let event = buf
                    .fill(self.value)
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .submit_on(ec)?;
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill_device_uninit".into());
    }
}

// ── Leaf: fill a MappedSliceUninit → MappedSlice ───────────────────────────

/// Eager analog of `FillFromUninitOp<MappedSliceUninit, _>`: fill an uninit
/// SVM slice with `value`. Mirrors [`Fill`].
pub struct FillMappedUninit<T: Copy, M: MemMode> {
    uninit: Input<MappedSliceUninit<T, M>>,
    value: T,
    out: Pipe<MappedSlice<T, M>>,
}

/// Build an eager fill-from-uninit leaf over a `MappedSliceUninit`.
pub fn fill_mapped_uninit<T, M>(
    uninit: impl Into<Input<MappedSliceUninit<T, M>>>,
    value: T,
) -> FillMappedUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    FillMappedUninit {
        uninit: uninit.into(),
        value,
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for FillMappedUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn output_pipe(&self) -> Pipe<MappedSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve()?;
        // SAFETY: the SVM fill below writes every byte.
        let buf = unsafe { uninit.assume_init() };
        match mode {
            ExecMode::Blocking => {
                buf.fill(self.value)
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .wait_on(ec)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                let event = buf
                    .fill(self.value)
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .submit_on(ec)?;
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill_mapped_uninit".into());
    }
}

// ── Leaf: fill a USMSliceUninit → USMSlice (pure host op) ───────────────────

/// Eager analog of `FillFromUninitOp<USMSliceUninit, _>`. USM is host memory,
/// so this is a pure host op: no enqueue, no event, deps pass through (mode
/// N/A) — mirrors [`Upload`]'s synchronous-create shape.
pub struct FillUsmUninit<T: Copy, M: MemMode> {
    uninit: Input<USMSliceUninit<T, M>>,
    value: T,
    out: Pipe<USMSlice<T, M>>,
}

/// Build an eager fill-from-uninit leaf over a `USMSliceUninit`.
pub fn fill_usm_uninit<T, M>(
    uninit: impl Into<Input<USMSliceUninit<T, M>>>,
    value: T,
) -> FillUsmUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    FillUsmUninit {
        uninit: uninit.into(),
        value,
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for FillUsmUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSlice<T, M>;

    fn output_pipe(&self) -> Pipe<USMSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, _ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Pure host op — no event; forward the upstream deps unchanged.
        let (uninit, deps) = self.uninit.resolve()?;
        let buf = uninit.fill_into(self.value);
        self.out.put(buf, deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill_usm_uninit".into());
    }
}

// ── Leaf: write host data into a DeviceSliceUninit → DeviceSlice ────────────

/// Consume an uninit `DeviceSlice` and write host `src` into it, yielding the
/// initialised buffer. Mirrors [`Fill`] (ExecMode branch). For the non-blocking
/// path the host `src` is kept alive until the write event fires via
/// `register_drop_callback`; for the blocking path the write completes before
/// return, so `src` drops normally at end of `execute`.
pub struct WriteDeviceUninit<T, M: MemMode> {
    uninit: Input<DeviceSliceUninit<T, M>>,
    src: Option<UploadSource<T>>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an eager write-from-uninit leaf over a `DeviceSliceUninit`.
pub fn write_device_uninit<T, M, S>(
    uninit: impl Into<Input<DeviceSliceUninit<T, M>>>,
    src: S,
) -> WriteDeviceUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostUploadable + HostWritable + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteDeviceUninit {
        uninit: uninit.into(),
        src: Some(src.into()),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for WriteDeviceUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostUploadable + HostWritable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve()?;
        let src = self
            .src
            .take()
            .expect("WriteDeviceUninit::execute called twice — internal eager bug");
        // SAFETY: the write below covers every byte; downstream gates on the
        // returned write event (Pipelined) or the driver waits (Blocking).
        let mut buf = unsafe { uninit.assume_init() };
        match mode {
            ExecMode::Blocking => {
                buf.write(src.as_slice())
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .wait_on(ec)?;
                // Blocking write completed — `src` drops at end of execute.
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                let event = buf
                    .write(src.as_slice())
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .submit_on(ec)?;
                // Keep-alive: drop the host source when CL_COMPLETE fires.
                register_drop_callback(&event, Box::new(src))?;
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_device_uninit".into());
    }
}

// ── Leaf: write host data into a MappedSliceUninit → MappedSlice ────────────

/// Eager analog of `WriteFromUninitOp<MappedSliceUninit, _>`. Mirrors
/// [`WriteDeviceUninit`].
pub struct WriteMappedUninit<T, M: MemMode> {
    uninit: Input<MappedSliceUninit<T, M>>,
    src: Option<UploadSource<T>>,
    out: Pipe<MappedSlice<T, M>>,
}

/// Build an eager write-from-uninit leaf over a `MappedSliceUninit`.
pub fn write_mapped_uninit<T, M, S>(
    uninit: impl Into<Input<MappedSliceUninit<T, M>>>,
    src: S,
) -> WriteMappedUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteMappedUninit {
        uninit: uninit.into(),
        src: Some(src.into()),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for WriteMappedUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn output_pipe(&self) -> Pipe<MappedSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve()?;
        let src = self
            .src
            .take()
            .expect("WriteMappedUninit::execute called twice — internal eager bug");
        // SAFETY: the SVM write below covers every byte.
        let buf = unsafe { uninit.assume_init() };
        match mode {
            ExecMode::Blocking => {
                buf.write(src.as_slice())
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .wait_on(ec)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                let event = buf
                    .write(src.as_slice())
                    .after_all(deps.iter().map(|d| d.as_ref()))
                    .submit_on(ec)?;
                register_drop_callback(&event, Box::new(src))?;
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_mapped_uninit".into());
    }
}

// ── Leaf: write host data into a USMSliceUninit → USMSlice (pure host op) ───

/// Eager analog of `WriteFromUninitOp<USMSliceUninit, _>`. Pure host memcpy via
/// the Tier-1 `write_from` helper — surfaces `LengthMismatch` at execute. No
/// enqueue, deps pass through (mode N/A) — mirrors [`Upload`].
pub struct WriteUsmUninit<T: Copy, M: MemMode> {
    uninit: Input<USMSliceUninit<T, M>>,
    src: Option<UploadSource<T>>,
    out: Pipe<USMSlice<T, M>>,
}

/// Build an eager write-from-uninit leaf over a `USMSliceUninit`.
pub fn write_usm_uninit<T, M, S>(
    uninit: impl Into<Input<USMSliceUninit<T, M>>>,
    src: S,
) -> WriteUsmUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteUsmUninit {
        uninit: uninit.into(),
        src: Some(src.into()),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for WriteUsmUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSlice<T, M>;

    fn output_pipe(&self) -> Pipe<USMSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, _ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Host memcpy via Tier-1 helper; Err on length mismatch propagates.
        let (uninit, deps) = self.uninit.resolve()?;
        let src = self
            .src
            .take()
            .expect("WriteUsmUninit::execute called twice — internal eager bug");
        let buf = uninit.write_from(src.as_slice())?;
        // src drops at end of execute — memcpy is done, no async keep-alive.
        self.out.put(buf, deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_usm_uninit".into());
    }
}

// ════════════════════════════════════════════════════════════════════════
// usm_op.rs ports — USM alloc / wrap (pure host, synchronous)
// ════════════════════════════════════════════════════════════════════════

// ── Leaf: wrap a host Vec<T> as a USMSlice (eager UsmSliceOp) ───────────────

/// Wrap a host `Vec<T>` as a [`USMSlice<T, M>`]. Source leaf (no upstream
/// input); construction is pure host code (`USMSlice::new`) — no enqueue, no
/// event (mode N/A). Mirrors [`Upload`]'s synchronous-create shape.
pub struct UsmSlice<T, M: MemMode = ReadWrite> {
    data: Option<Vec<T>>,
    out: Pipe<USMSlice<T, M>>,
}

/// Build an eager USM-wrap leaf from a host `Vec<T>`.
pub fn usm_slice<T, M>(data: Vec<T>) -> UsmSlice<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    UsmSlice {
        data: Some(data),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for UsmSlice<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSlice<T, M>;

    fn output_pipe(&self) -> Pipe<USMSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // USMSlice::new is pure host code — no in-flight event, mode N/A.
        let data = self
            .data
            .take()
            .expect("UsmSlice::execute called twice — internal eager bug");
        let slice = USMSlice::new(ec.context(), data)?;
        self.out.put(slice, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("usm_slice".into());
    }
}

// ── Leaf: alloc an uninit USMSlice (eager UsmSliceAllocUninit) ──────────────

/// Allocate a [`USMSliceUninit<T, M>`]. Source leaf; allocation is pure host
/// code (`USMSlice::alloc_uninit`) — no enqueue, no event (mode N/A). Mirrors
/// [`Upload`]'s synchronous-create shape.
pub struct UsmAllocUninit<T, M: MemMode = ReadWrite> {
    len: usize,
    out: Pipe<USMSliceUninit<T, M>>,
    _t: PhantomData<fn() -> (T, M)>,
}

/// Build an eager uninit-USM alloc leaf.
pub fn usm_alloc_uninit<T, M>(len: usize) -> UsmAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    UsmAllocUninit {
        len,
        out: Pipe::new(),
        _t: PhantomData,
    }
}

impl<T, M> EagerOp for UsmAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSliceUninit<T, M>;

    fn output_pipe(&self) -> Pipe<USMSliceUninit<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // alloc_uninit is pure host code — no in-flight event, mode N/A.
        let uninit = USMSlice::<T, M>::alloc_uninit(ec.context(), self.len)?;
        self.out.put(uninit, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("usm_alloc_uninit(len={})", self.len));
    }
}

// ════════════════════════════════════════════════════════════════════════
// image_transfer.rs ports — image upload / download
// ════════════════════════════════════════════════════════════════════════

// ── Leaf: image upload (host pixels → image I) ──────────────────────────────

/// Allocate an image of type `I` with `dims` and write `pixels` into it.
/// Source-ish leaf (no upstream image input). The underlying image `write_op`
/// has **only a non-blocking enqueue** (no native `wait_on`), so this op always
/// uses `submit_on` and ignores `mode`; the source `pixels` is kept alive until
/// the write event fires via `register_drop_callback`. Mirrors [`Upload`]
/// (chain-entry) but carries a write event because the enqueue is non-blocking.
pub struct ImageUploadEager<I: ImageHostTransfer> {
    pixels: Option<Vec<I::Pixel>>,
    dims: I::Dims,
    out: Pipe<I>,
    _ty: PhantomData<fn() -> I>,
}

/// Build an eager image-upload leaf.
pub fn image_upload<I>(pixels: Vec<I::Pixel>, dims: I::Dims) -> ImageUploadEager<I>
where
    I: ImageHostTransfer + Send + 'static,
    I::Pixel: Send + 'static,
{
    ImageUploadEager {
        pixels: Some(pixels),
        dims,
        out: Pipe::new(),
        _ty: PhantomData,
    }
}

impl<I> EagerOp for ImageUploadEager<I>
where
    I: ImageHostTransfer + Send + 'static,
    I::Pixel: Send + 'static,
{
    type Output = I;

    fn output_pipe(&self) -> Pipe<I> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // `write_op` is non-blocking only — always submit_on, mode ignored.
        let pixels = self
            .pixels
            .take()
            .expect("ImageUploadEager::execute called twice — internal eager bug");
        let mut img = I::alloc(ec.context(), self.dims)?;
        // Source leaf: no upstream Input, so no wait-list to thread.
        let event = img.write_op(&pixels).submit_on(ec)?;
        // Keep-alive: the runtime reads from `pixels` until the write fires.
        register_drop_callback(&event, Box::new(pixels))?;
        self.out.put(img, vec![wrap_event(event)]);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("image_upload".into());
    }
}

// ── Leaf: image download (image I → host Vec<Pixel>) ────────────────────────

/// Consume an upstream image of type `I`, alloc a host `Vec<I::Pixel>`, and
/// read the image into it. The underlying image `read_op` has **only a
/// non-blocking enqueue** (no native `wait_on`), so this op always uses
/// `submit_on` and ignores `mode`. Mirrors [`Download`] (output leaf) but
/// without the blocking branch the buffer read has.
pub struct ImageDownloadEager<I: ImageHostTransfer> {
    img: Input<I>,
    out: Pipe<Vec<I::Pixel>>,
}

/// Build an eager image-download leaf over an upstream image.
pub fn image_download<I>(img: impl Into<Input<I>>) -> ImageDownloadEager<I>
where
    I: ImageHostTransfer + Send + 'static,
    I::Pixel: Default + Copy + Send + 'static,
{
    ImageDownloadEager {
        img: img.into(),
        out: Pipe::new(),
    }
}

impl<I> EagerOp for ImageDownloadEager<I>
where
    I: ImageHostTransfer + Send + 'static,
    I::Pixel: Default + Copy + Send + 'static,
{
    type Output = Vec<I::Pixel>;

    fn output_pipe(&self) -> Pipe<Vec<I::Pixel>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // `read_op` is non-blocking only — always submit_on, mode ignored.
        let (img, deps) = self.img.resolve()?;
        let pixel_count = img.pixel_count();
        let mut pixels = vec![<I::Pixel as Default>::default(); pixel_count];
        let event = img
            .read_op(&mut pixels)?
            .after_all(deps.iter().map(|d| d.as_ref()))
            .submit_on(ec)?;
        self.out.put(pixels, vec![wrap_event(event)]);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("image_download".into());
    }
}

// ════════════════════════════════════════════════════════════════════════
// host_view.rs ports — acquire / release host views (map / unmap)
// ════════════════════════════════════════════════════════════════════════
//
// The host-view types (`DeviceSliceHostView` / `MappedSliceHostView`) have
// private fields, so the eager ops cannot reconstruct them directly. Instead
// each eager op holds the buffer/view and delegates the exact enqueue body by
// constructing the OLD `DeviceOperation` (via its public trait method
// `acquire_host_view{,_read}` / `release_to_device`) and calling its
// `execute(ec, deps)`. This reuses the old map/unmap body verbatim without
// modifying the old module. None of these primitives has a native blocking
// enqueue (the map/unmap is always non-blocking `false`), so `mode` is ignored.

// ── Leaf: acquire a read/write DeviceSlice host view ────────────────────────

/// Acquire a read/write host view of an upstream `DeviceSlice` via a
/// non-blocking `clEnqueueMapBuffer`. Output is the owned
/// [`DeviceSliceHostView`]. No native blocking enqueue — `mode` ignored.
/// Delegates to the old `AcquireDeviceSliceOp` body via `acquire_host_view`.
pub struct AcquireDeviceView<T, M: MemMode> {
    buf: Input<DeviceSlice<T, M>>,
    out: Pipe<DeviceSliceHostView<T, M, MapReadWrite>>,
}

/// Build an eager acquire-read/write-view leaf over an upstream `DeviceSlice`.
pub fn acquire_device_view<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
) -> AcquireDeviceView<T, M>
where
    T: Send + 'static,
    M: MemMode + HostWritable + HostReadable,
{
    AcquireDeviceView {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for AcquireDeviceView<T, M>
where
    T: Send + 'static,
    M: MemMode + HostWritable + HostReadable,
{
    type Output = DeviceSliceHostView<T, M, MapReadWrite>;

    fn output_pipe(&self) -> Pipe<Self::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve()?;
        // Delegate to the old op's verbatim map body (map/unmap is always
        // non-blocking — mode ignored).
        let (view, out_deps) = buf.acquire_host_view().execute(ec, deps)?;
        self.out.put(view, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("acquire_device_view".into());
    }
}

// ── Leaf: acquire a read-only DeviceSlice host view ─────────────────────────

/// Acquire a read-only host view of an upstream `DeviceSlice`
/// (`clEnqueueMapBuffer(CL_MAP_READ)`). Output is the owned
/// [`DeviceSliceHostView`]. No native blocking enqueue — `mode` ignored.
pub struct AcquireDeviceViewRead<T, M: MemMode> {
    buf: Input<DeviceSlice<T, M>>,
    out: Pipe<DeviceSliceHostView<T, M, MapReadOnly>>,
}

/// Build an eager acquire-read-only-view leaf over an upstream `DeviceSlice`.
pub fn acquire_device_view_read<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
) -> AcquireDeviceViewRead<T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable,
{
    AcquireDeviceViewRead {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for AcquireDeviceViewRead<T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable,
{
    type Output = DeviceSliceHostView<T, M, MapReadOnly>;

    fn output_pipe(&self) -> Pipe<Self::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve()?;
        let (view, out_deps) = buf.acquire_host_view_read().execute(ec, deps)?;
        self.out.put(view, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("acquire_device_view_read".into());
    }
}

// ── Leaf: release a DeviceSlice host view back to the device ─────────────────

/// Enqueue `clEnqueueUnmapMemObject` for an upstream
/// [`DeviceSliceHostView`] and yield the [`DeviceSlice`] back. No native
/// blocking enqueue — `mode` ignored. Generic over the view's map-access mode.
pub struct ReleaseDeviceView<T, M: MemMode, A: MapAccess> {
    view: Input<DeviceSliceHostView<T, M, A>>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an eager release-view leaf over an upstream `DeviceSliceHostView`.
pub fn release_device_view<T, M, A>(
    view: impl Into<Input<DeviceSliceHostView<T, M, A>>>,
) -> ReleaseDeviceView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    ReleaseDeviceView {
        view: view.into(),
        out: Pipe::new(),
    }
}

impl<T, M, A> EagerOp for ReleaseDeviceView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (view, deps) = self.view.resolve()?;
        let (buf, out_deps) = view.release_to_device().execute(ec, deps)?;
        self.out.put(buf, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("release_device_view".into());
    }
}

// ── Leaf: acquire a read/write MappedSlice (SVM) host view ──────────────────

/// Acquire a read/write SVM host view of an upstream `MappedSlice` via a
/// non-blocking `clEnqueueSVMMap`. No native blocking enqueue — `mode` ignored.
pub struct AcquireMappedView<T, M: MemMode> {
    buf: Input<MappedSlice<T, M>>,
    out: Pipe<MappedSliceHostView<T, M, MapReadWrite>>,
}

/// Build an eager acquire-read/write-SVM-view leaf over a `MappedSlice`.
pub fn acquire_mapped_view<T, M>(
    buf: impl Into<Input<MappedSlice<T, M>>>,
) -> AcquireMappedView<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + HostReadable,
{
    AcquireMappedView {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for AcquireMappedView<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + HostReadable,
{
    type Output = MappedSliceHostView<T, M, MapReadWrite>;

    fn output_pipe(&self) -> Pipe<Self::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve()?;
        let (view, out_deps) = buf.acquire_host_view().execute(ec, deps)?;
        self.out.put(view, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("acquire_mapped_view".into());
    }
}

// ── Leaf: acquire a read-only MappedSlice (SVM) host view ───────────────────

/// Acquire a read-only SVM host view of an upstream `MappedSlice`
/// (`clEnqueueSVMMap(CL_MAP_READ)`). No native blocking enqueue — `mode`
/// ignored.
pub struct AcquireMappedViewRead<T, M: MemMode> {
    buf: Input<MappedSlice<T, M>>,
    out: Pipe<MappedSliceHostView<T, M, MapReadOnly>>,
}

/// Build an eager acquire-read-only-SVM-view leaf over a `MappedSlice`.
pub fn acquire_mapped_view_read<T, M>(
    buf: impl Into<Input<MappedSlice<T, M>>>,
) -> AcquireMappedViewRead<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostReadable,
{
    AcquireMappedViewRead {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> EagerOp for AcquireMappedViewRead<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostReadable,
{
    type Output = MappedSliceHostView<T, M, MapReadOnly>;

    fn output_pipe(&self) -> Pipe<Self::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve()?;
        let (view, out_deps) = buf.acquire_host_view_read().execute(ec, deps)?;
        self.out.put(view, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("acquire_mapped_view_read".into());
    }
}

// ── Leaf: release a MappedSlice (SVM) host view back to the device ───────────

/// Enqueue `clEnqueueSVMUnmap` for an upstream [`MappedSliceHostView`] and
/// yield the [`MappedSlice`] back. No native blocking enqueue — `mode` ignored.
/// Generic over the view's map-access mode.
pub struct ReleaseMappedView<T, M: MemMode, A: MapAccess> {
    view: Input<MappedSliceHostView<T, M, A>>,
    out: Pipe<MappedSlice<T, M>>,
}

/// Build an eager release-SVM-view leaf over an upstream `MappedSliceHostView`.
pub fn release_mapped_view<T, M, A>(
    view: impl Into<Input<MappedSliceHostView<T, M, A>>>,
) -> ReleaseMappedView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    ReleaseMappedView {
        view: view.into(),
        out: Pipe::new(),
    }
}

impl<T, M, A> EagerOp for ReleaseMappedView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    type Output = MappedSlice<T, M>;

    fn output_pipe(&self) -> Pipe<MappedSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (view, deps) = self.view.resolve()?;
        let (buf, out_deps) = view.release_to_device().execute(ec, deps)?;
        self.out.put(buf, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("release_mapped_view".into());
    }
}

// ── Multi-output leaf: copy_to (src, dst) → (src, dst) ──────────────────────
//
// The eager analog of the closure-layer `CopyToOp` family in `copy.rs`. A copy
// is a **two-output** op: it returns BOTH the source and destination buffers so
// the chain can thread either onward. It mirrors the macro-emitted multi-output
// kernel shape (commit 0f7083d): two element pipes (`Handle = (Pipe<OS>,
// Pipe<OD>)`), `execute` enqueues once and scatters each output into its element
// pipe (cloning the single completion `Dep` onto both), and `into_output` drains
// both pipes to reconstruct the `(src, dst)` tuple.
//
// Rather than re-deriving the ten (src, dst) family bodies (incl. the unsafe
// cross-type SVM-memcpy machinery in `copy.rs`), this op **reuses** the existing
// `CopyTo` / `DeviceOperation` `CopyToOp` impls: resolve the two inputs, build
// the old op via `src.copy_to(dst)`, run its `DeviceOperation::execute` (which
// owns every per-family primitive + Uninit→Init transition + buffer-use
// registration), then scatter its `(out_src, out_dst)` Output across the two
// pipes. All ten families come along for free — no `copy.rs` change.
//
// Copy ops have no native blocking enqueue (the closure layer always uses
// `submit_on` + event); `mode` is therefore ignored — `submit_on`+event is the
// only path, and copy is rarely terminal anyway (it returns buffers onward).

/// Split a copy op's 2-tuple `Output` into its source + destination halves so
/// the eager [`CopyTo2`] op can hold one typed element pipe per side and
/// reconstruct the tuple in `into_output`. Implemented once for every `(A, B)`.
pub trait CopyOutputs {
    /// The post-copy source buffer (element 0 of the copy Output).
    type Src: Send;
    /// The post-copy destination buffer (element 1 of the copy Output).
    type Dst: Send;
    /// Decompose into `(src, dst)`.
    fn into_parts(self) -> (Self::Src, Self::Dst);
}

impl<A: Send, B: Send> CopyOutputs for (A, B) {
    type Src = A;
    type Dst = B;
    fn into_parts(self) -> (A, B) {
        self
    }
}

/// Eager multi-output copy: `eager_copy_to(src, dst)` enqueues a copy and yields
/// `(src, dst)`. `Handle = (Pipe<OutSrc>, Pipe<OutDst>)` — two element pipes, so
/// a downstream `.and_then(|(src, dst)| …)` selects either side. Polymorphic
/// over every supported `(src, dst)` family via the `Src: CopyTo<Dst>` bound.
pub struct CopyTo2<Src, Dst>
where
    Src: CopyTo<Dst>,
    <Src::Op as DeviceOperation>::Output: CopyOutputs,
{
    src: Input<Src>,
    dst: Input<Dst>,
    // One element pipe per copy output (move-once storage), mirroring the
    // macro-emitted multi-output kernel. The output tuple is reconstructed from
    // both in `into_output`.
    src_pipe: Pipe<<<Src::Op as DeviceOperation>::Output as CopyOutputs>::Src>,
    dst_pipe: Pipe<<<Src::Op as DeviceOperation>::Output as CopyOutputs>::Dst>,
}

/// Build an eager copy leaf. `src` / `dst` may each be a concrete buffer or an
/// upstream [`Pipe`]. Output is `(src, dst)` (an `Uninit` dst comes back `Init`
/// — the copy wrote every byte). See [`CopyTo2`].
pub fn eager_copy_to<Src, Dst>(
    src: impl Into<Input<Src>>,
    dst: impl Into<Input<Dst>>,
) -> CopyTo2<Src, Dst>
where
    Src: CopyTo<Dst>,
    <Src::Op as DeviceOperation>::Output: CopyOutputs,
{
    CopyTo2 {
        src: src.into(),
        dst: dst.into(),
        src_pipe: Pipe::new(),
        dst_pipe: Pipe::new(),
    }
}

impl<Src, Dst> EagerOp for CopyTo2<Src, Dst>
where
    Src: CopyTo<Dst> + Send,
    Dst: Send,
    Src::Op: Send,
    <Src::Op as DeviceOperation>::Output: CopyOutputs,
{
    type Output = (
        <<Src::Op as DeviceOperation>::Output as CopyOutputs>::Src,
        <<Src::Op as DeviceOperation>::Output as CopyOutputs>::Dst,
    );
    // Two element pipes, like the multi-output kernel: the downstream closure
    // gets `(pa, pb)` and selects either buffer.
    type Handle = (
        Pipe<<<Src::Op as DeviceOperation>::Output as CopyOutputs>::Src>,
        Pipe<<<Src::Op as DeviceOperation>::Output as CopyOutputs>::Dst>,
    );

    fn output_pipe(&self) -> Pipe<Self::Output> {
        // Multi-output storage is the per-element pipes; this single pipe is
        // never filled or drained (the default `into_output` is overridden, and
        // `and_then` uses `handle()`). Return a fresh empty pipe — well-typed,
        // never read.
        Pipe::new()
    }

    fn handle(&self) -> Self::Handle {
        (self.src_pipe.clone(), self.dst_pipe.clone())
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Resolve both inputs → (buffer, upstream Deps). Either may be a pipe
        // (upstream output) or concrete. Combine their wait-lists.
        let (src, src_deps) = self.src.resolve()?;
        let (dst, dst_deps) = self.dst.resolve()?;
        let mut deps = src_deps;
        deps.extend(dst_deps);
        // Reuse the closure-layer copy op: it owns the right per-family
        // primitive (CopyBuffer / SVMMemcpy), the Uninit→Init transition, and
        // buffer-use registration. ONE enqueue → its returned Deps hold one
        // completion event.
        let op = src.copy_to(dst);
        let (out, out_deps) = op.execute(ec, deps)?;
        let (out_src, out_dst) = out.into_parts();
        // Clone the completion Dep onto BOTH element pipes so whichever side
        // flows downstream carries the wait-list (and the terminal reconstruct
        // gathers from both).
        self.src_pipe.put(out_src, out_deps.clone());
        self.dst_pipe.put(out_dst, out_deps);
        Ok(())
    }

    fn collect(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
    where
        Self: Sized,
    {
        // Grab the element pipes before consuming `self`, then scatter via
        // `execute`, then drain + reconstruct the `(src, dst)` tuple, gathering
        // both pipes' deps (the terminal `into_output` waits on them once).
        let src_pipe = self.src_pipe.clone();
        let dst_pipe = self.dst_pipe.clone();
        self.execute(ec, mode)?;
        let (out_src, mut deps) = src_pipe.take().ok_or(Error::NotSupported(
            "eager graph: terminal copy produced no output",
        ))?;
        let (out_dst, dst_deps) = dst_pipe.take().ok_or(Error::NotSupported(
            "eager graph: terminal copy produced no output",
        ))?;
        deps.extend(dst_deps);
        Ok(((out_src, out_dst), deps))
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("copy_to".into());
    }
}

// ════════════════════════════════════════════════════════════════════════
// Execute-time closure nodes — the ONE place closures legitimately survive in
// the eager model (NOTES → "EXECUTE-TIME CLOSURE NODES"). Unlike eager
// `and_then` (its builder runs at BUILD with a `Pipe` handle), these three run
// their closure at EXECUTE because it needs the live `ec` / mapped host data,
// neither of which exists at build. Shape: hold `source` + `src_pipe` (captured
// at build) + `f: Option<F>` + `out`; `execute` runs the source, takes the
// upstream runtime value, runs the closure NOW to build the downstream op,
// grabs the downstream's out-pipe BEFORE consuming it (move-once), runs it, and
// moves the result into `out` (merging the source's deps so the terminal waits
// on the whole chain).
// ════════════════════════════════════════════════════════════════════════

// ── AndThenWithContext: closure gets the live ExecutionContext at execute ──

/// Sequential composition whose builder runs at **execute** with the live
/// [`ExecutionContext`] in scope — built by
/// [`and_then_with_context`](EagerOpExt::and_then_with_context).
///
/// Unlike [`AndThen`] (builder at construction, `Pipe` handle), the closure
/// here receives `&ExecutionContext` + the upstream's **runtime value**, so it
/// can read `ec.device()` / `ec.context()` / route via [`on_device`](EagerOpExt::on_device) while
/// building the downstream op. The downstream op is therefore built — and run —
/// at execute time.
pub struct AndThenWithContext<S: EagerOp, U: EagerOp, F> {
    source: S,
    src_pipe: Pipe<S::Output>,
    f: Option<F>,
    out: Pipe<U::Output>,
}

impl<S, U, F> EagerOp for AndThenWithContext<S, U, F>
where
    S: EagerOp,
    U: EagerOp,
    F: for<'a> FnOnce(&ExecutionContext<'a>, S::Output) -> U + Send,
{
    type Output = U::Output;

    fn output_pipe(&self) -> Pipe<U::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // Source is upstream — always pipeline it; its value + events feed the
        // downstream op the closure builds.
        self.source.execute(ec, ExecMode::Pipelined)?;
        let (value, src_deps) = self.src_pipe.take().ok_or(Error::NotSupported(
            "eager and_then_with_context: source produced no output",
        ))?;
        let f = self
            .f
            .take()
            .expect("AndThenWithContext::execute called twice — internal eager bug");
        // Closure runs NOW, at execute, with the live ec + runtime value.
        let downstream = f(ec, value);
        // Grab the downstream's out-pipe BEFORE consuming it (move-once).
        let down_pipe = downstream.output_pipe();
        downstream.execute(ec, mode)?;
        let (out_value, mut out_deps) = down_pipe.take().ok_or(Error::NotSupported(
            "eager and_then_with_context: downstream produced no output",
        ))?;
        // Thread the source's events through: merge them with the downstream's
        // so the terminal waits on the whole chain (the downstream consumed the
        // value concretely, so its enqueue carries no upstream wait-list —
        // forwarding the source deps keeps the chain's events live).
        out_deps.extend(src_deps);
        self.out.put(out_value, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("and_then_with_context".into());
    }
}

// ── OnDevice: re-point the op at a different device's queue at execute ──

/// Route `source`'s `execute` to a **different** device's default
/// out-of-order queue — built by [`on_device`](EagerOpExt::on_device).
///
/// No user closure: at execute it resolves the target device's queue from the
/// running context, builds a sibling [`ExecutionContext`] (same context + same
/// host-error slot, different device + queue), and runs `source` against it.
/// The source's events are valid across queues of the same context, so
/// downstream stages on the parent's queue can wait on them cross-device.
pub struct OnDevice<S: EagerOp> {
    source: S,
    device: crate::Device,
    src_pipe: Pipe<S::Output>,
    out: Pipe<S::Output>,
}

impl<S, S2> EagerOp for OnDevice<S>
where
    S: EagerOp<Output = S2>,
    S2: Send,
{
    type Output = S::Output;

    fn output_pipe(&self) -> Pipe<S::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, parent: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // Resolve the target queue from the running context (cached, so the
        // terminal's flush_all_outoforder_queues picks it up).
        let target_q = parent.context().default_outoforder_queue(&self.device)?;
        // Sibling EC: same context + same host-error slot, different device +
        // queue. `target_q` lives on this frame; its `.raw()` borrows for the
        // inner execute().
        let child = ExecutionContext::with_host_error_slot(
            parent.context(),
            self.device.clone(),
            target_q.raw(),
            parent.host_error_slot(),
        );
        // Run the source against the child EC — it deposits into `src_pipe`.
        self.source.execute(&child, mode)?;
        let (value, deps) = self.src_pipe.take().ok_or(Error::NotSupported(
            "eager on_device: source produced no output",
        ))?;
        self.out.put(value, deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("on_device".into());
    }
}

// ── AndThenHost / AndThenHostWithContext: the host seam ──

/// Run a host closure on a borrowed [`Mappable::View`](crate::mappable::Mappable::View) of the upstream output,
/// in chain order — built by [`and_then_host`](EagerOpExt::and_then_host).
///
/// At execute: run the source, take its `(value, deps)`, **drain the deps
/// (blocking wait)** so the data is host-valid, map the value, run the closure
/// on its view (mutations persist via the unmap), then forward the **same**
/// value downstream (`Output = S::Output`).
///
/// ## 🚨 REGRESSION: currently synchronous, MUST become worker-thread-spawned
///
/// This node currently runs the host call **synchronously at execute** (drain
/// upstream deps with a blocking wait, map, wait the map event, run the closure,
/// unmap, wait, forward). **That is a regression, not an acceptable
/// simplification.** The old closure-layer `and_then_host`
/// (`claspr/src/and_then_host.rs`) deliberately engineered map →
/// `clCreateUserEvent` → unmap(gated on the user event) → **spawned worker
/// thread** precisely so a host stage sits *in-queue* and overlaps pipelined
/// device work — the chain continues at submit time, not when the closure
/// returns. The whole map/user-event apparatus exists for exactly that. Running
/// it on the submitting thread throws that away and serializes the chain.
///
/// TODO (tracked in NOTES "SERIOUS REGRESSION"): port the old async machinery —
/// worker thread waits the all-data-mapped events, runs the closure under
/// `catch_unwind`, signals a user event (CL_COMPLETE / negative); `execute`
/// returns the unmap events as deps so downstream gates on the user event
/// WITHOUT the submit thread blocking. Reinstate the `ExecutionContext`
/// host-error slot (rich Rust error survives the negative-status cascade) and
/// the defensive sync-unmap-on-error. The `catch_unwind` → `Error::HostPanic`
/// behaviour stays as the worker's panic handling. Do NOT conflate this with
/// the genuinely-synchronous host-VALUE seam (`and_then_host_value`).
pub struct AndThenHost<S: EagerOp, F>
where
    S::Output: crate::mappable::Mappable,
{
    source: S,
    src_pipe: Pipe<S::Output>,
    f: Option<F>,
    out: Pipe<S::Output>,
}

/// Like [`AndThenHost`] but the closure also receives `&Context` — built by
/// [`and_then_host_with_context`](EagerOpExt::and_then_host_with_context).
pub struct AndThenHostWithContext<S: EagerOp, F>
where
    S::Output: crate::mappable::Mappable,
{
    source: S,
    src_pipe: Pipe<S::Output>,
    f: Option<F>,
    out: Pipe<S::Output>,
}

/// Shared body: run source, drain its deps host-side, map, run `host_call` on
/// the view, unmap, and forward the value with the unmap event as deps.
///
/// 🚨 REGRESSION: this is the synchronous (submit-thread) implementation. It MUST
/// be replaced by the worker-thread + user-event machinery from the old
/// `and_then_host.rs` so the host stage overlaps device work — see the
/// `AndThenHost` doc above and NOTES "SERIOUS REGRESSION".
fn run_host_seam<O>(
    source_value: O,
    source_deps: Deps,
    ec: &ExecutionContext<'_>,
    host_call: impl FnOnce(<O as crate::mappable::Mappable>::View<'_>) -> Result<()>,
) -> Result<(O, Deps)>
where
    O: crate::mappable::Mappable,
{
    use crate::Launcher;
    use crate::mappable::Mappable;
    // Drain upstream events so the mapped memory is host-valid before the
    // closure reads it (the host seam's defining wait).
    for d in &source_deps {
        d.as_ref().wait().map_err(Error::OpenCl)?;
    }
    let q = ec.cl_queue();
    // Non-blocking map (deps already drained → empty wait-list), then wait its
    // event synchronously so the view is coherent.
    let (mut handle, map_events) = source_value.map(q, &[])?;
    for ev in &map_events {
        ev.wait().map_err(Error::OpenCl)?;
    }
    // Run the host closure on the borrowed view, inside `catch_unwind` so a
    // panic becomes `Error::HostPanic` rather than unwinding the caller
    // (mirrors the old closure-layer `and_then_host`). On Err or panic, the
    // handle's `Drop` issues the defensive blocking unmap (unmap_enqueued still
    // false), so the buffer is left clean.
    {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let view = <O as Mappable>::view(&mut handle);
        match catch_unwind(AssertUnwindSafe(|| host_call(view))) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(panic) => {
                // `catch_unwind` yields `Box<dyn Any + Send>`; the payload is
                // typically `&'static str` (`panic!("lit")`) or `String`
                // (`panic!("{}", x)`). Anything else gets a placeholder.
                let msg = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                return Err(Error::HostPanic(msg));
            }
        }
    }
    // Commit mutations: enqueue the unmap (no waiter) and wait its event.
    let unmap_events = <O as Mappable>::enqueue_unmap(&mut handle, q, &[])?;
    let unmap_deps: Deps = unmap_events.into_iter().map(wrap_event).collect();
    for d in &unmap_deps {
        d.as_ref().wait().map_err(Error::OpenCl)?;
    }
    // `handle` drops here — `unmap_enqueued` is true, so no second unmap.
    drop(handle);
    Ok((source_value, unmap_deps))
}

impl<S, F> EagerOp for AndThenHost<S, F>
where
    S: EagerOp,
    S::Output: crate::mappable::Mappable,
    F: for<'a> FnOnce(<S::Output as crate::mappable::Mappable>::View<'a>) -> Result<()> + Send,
{
    type Output = S::Output;

    fn output_pipe(&self) -> Pipe<S::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        self.source.execute(ec, ExecMode::Pipelined)?;
        let (value, deps) = self.src_pipe.take().ok_or(Error::NotSupported(
            "eager and_then_host: source produced no output",
        ))?;
        let f = self
            .f
            .take()
            .expect("AndThenHost::execute called twice — internal eager bug");
        let (out_value, out_deps) = run_host_seam::<S::Output>(value, deps, ec, f)?;
        self.out.put(out_value, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("and_then_host".into());
    }
}

impl<S, F> EagerOp for AndThenHostWithContext<S, F>
where
    S: EagerOp,
    S::Output: crate::mappable::Mappable,
    F: for<'a> FnOnce(&Context, <S::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
        + Send,
{
    type Output = S::Output;

    fn output_pipe(&self) -> Pipe<S::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        self.source.execute(ec, ExecMode::Pipelined)?;
        let (value, deps) = self.src_pipe.take().ok_or(Error::NotSupported(
            "eager and_then_host_with_context: source produced no output",
        ))?;
        let f = self
            .f
            .take()
            .expect("AndThenHostWithContext::execute called twice — internal eager bug");
        // Bind the context up front so the `host_call` closure borrows it
        // (cheap Arc-backed handle); the view borrow is supplied by the seam.
        let context = ec.context().clone();
        let (out_value, out_deps) =
            run_host_seam::<S::Output>(value, deps, ec, move |view| f(&context, view))?;
        self.out.put(out_value, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("and_then_host_with_context".into());
    }
}

// ── Profiled: wall-clock timing for a sub-chain ────────────────────────

/// Times whatever the source op enqueued, registering a completion callback —
/// built by [`profiled`](EagerProfileExt::profiled). Mirrors the old
/// closure-layer `Profiled`: at execute it runs the source, enqueues an
/// `clEnqueueMarkerWithWaitList` over the source's events, registers the user
/// callback on the marker via [`register_profiling_callback`](crate::register_profiling_callback), and forwards the
/// **same** value downstream with the marker as its deps (so anything after the
/// `.profiled()` waits on the marker, which subsumes the source's events).
///
/// Requires the chain's OOO queue to have `CL_QUEUE_PROFILING_ENABLE` (build
/// the [`Context`] with [`.profiling(true)`](crate::context::ContextBuilder::profiling));
/// otherwise `execute` returns [`Error::ProfilingDisabled`] up front (the
/// source op still ran — profiling is a host side-effect, not data flow).
pub struct Profiled<S: EagerOp, F> {
    source: S,
    src_pipe: Pipe<S::Output>,
    cb: Option<F>,
    out: Pipe<S::Output>,
}

/// Extension trait adding [`profiled`](Self::profiled) to every [`EagerOp`].
/// Separate from [`EagerOpExt`] to mirror the old layer's
/// `DeviceOperationProfileExt`. Blanket-implemented.
pub trait EagerProfileExt: EagerOp + Sized {
    /// Register `cb` to receive the wall-clock [`ProfilingInfo`](crate::ProfilingInfo) for everything
    /// `self` enqueued onto the chain's queue. The closure fires on an OpenCL
    /// callback thread when the marker event completes. See [`Profiled`].
    fn profiled<F>(self, cb: F) -> Profiled<Self, F>
    where
        F: FnOnce(Result<crate::ProfilingInfo>) + Send + 'static,
    {
        let src_pipe = self.output_pipe();
        Profiled {
            source: self,
            src_pipe,
            cb: Some(cb),
            out: Pipe::new(),
        }
    }
}
impl<T: EagerOp> EagerProfileExt for T {}

impl<S, F> EagerOp for Profiled<S, F>
where
    S: EagerOp,
    F: FnOnce(Result<crate::ProfilingInfo>) + Send + 'static,
{
    // Profiling is a host side-effect; the chain's data flow is unchanged.
    type Output = S::Output;

    fn output_pipe(&self) -> Pipe<S::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        use crate::Launcher;
        self.source.execute(ec, ExecMode::Pipelined)?;
        let (value, source_deps) = self.src_pipe.take().ok_or(Error::NotSupported(
            "eager profiled: source produced no output",
        ))?;
        // Same up-front check as the old layer / Tier 1: the queue needs
        // profiling enabled before we waste a marker + callback registration.
        if (ec.cl_queue().properties()? & crate::CL_QUEUE_PROFILING_ENABLE) == 0 {
            return Err(Error::ProfilingDisabled);
        }
        // The marker waits for the source op's events, so the timestamps
        // reflect the source's wall-clock duration (first command queued to
        // last command finished).
        let wait_list: Vec<crate::cl_event> =
            source_deps.iter().map(|d| d.as_ref().get()).collect();
        // SAFETY: cl_event handles are valid — held by `source_deps` Arcs until
        // this call returns.
        let marker = unsafe { ec.cl_queue().enqueue_marker_with_wait_list(&wait_list) }
            .map_err(Error::OpenCl)?;
        // `source_deps` keeps the underlying cl_events alive across the
        // enqueue; safe to drop after.
        drop(source_deps);
        crate::register_profiling_callback(
            &marker,
            Box::new(
                self.cb
                    .take()
                    .expect("Profiled::execute called twice — internal eager bug"),
            ),
        )?;
        // The marker becomes this op's completion event for downstream
        // chaining (it subsumes the source's events).
        self.out.put(value, vec![wrap_event(marker)]);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("profiled".into());
    }
}

// ── EagerChainFuture: the async `.run().await` terminal ────────────────

/// Future returned by [`EagerOpExt::run`]. Resolves to `Result<T>` once the
/// chain's commands have all completed on the device (or immediately, with an
/// error, if the chain failed to submit or any host seam returned `Err`).
///
/// The eager analog of `chain_future::ChainFuture`. The key simplification
/// versus the old layer: the eager host seam (`run_host_seam`) runs its closure
/// synchronously inside `execute` and returns `Err` directly, rather than
/// stashing into a shared `Arc<Mutex<Option<Error>>>` from a worker thread. So a
/// host-side failure is already captured as a synchronous `Err` from `execute`
/// and becomes [`Errored`](Self::Errored) — there is no poll-time host-error
/// slot to reconcile.
#[cfg(feature = "async-events")]
pub enum EagerChainFuture<T> {
    /// Chain failed during setup, `execute` (including a host-seam closure
    /// `Err`/panic), or marker enqueue. The error surfaces on the first `poll`.
    Errored(Option<Error>),
    /// Chain submitted successfully; waiting for the trailing marker event to
    /// complete. The host-side `Output` is already materialised (drained from
    /// the output pipe at `run` time); the future just gates *when* the caller
    /// sees it on whether the queue work is done.
    Running {
        output: Option<T>,
        event_future: crate::EventFuture,
    },
}

// `T: Unpin` covers every realistic chain output (`Vec<u8>`, `DeviceSlice<T>`,
// `Arc<T>`, tuples of those, ...) and lets us pin-project via the cheap
// `Pin::get_mut`. Mirrors `ChainFuture`'s bound.
#[cfg(feature = "async-events")]
impl<T: Unpin> std::future::Future for EagerChainFuture<T> {
    type Output = Result<T>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        use std::task::Poll;
        let this = self.as_mut().get_mut();
        match this {
            EagerChainFuture::Errored(slot) => Poll::Ready(Err(slot
                .take()
                .expect("EagerChainFuture polled after Ready (Errored)"))),
            EagerChainFuture::Running {
                output,
                event_future,
            } => match std::pin::Pin::new(event_future).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Ready(Ok(())) => Poll::Ready(Ok(output
                    .take()
                    .expect("EagerChainFuture polled after Ready (Running)"))),
            },
        }
    }
}

/// Crate-internal worker behind [`EagerOpExt::run`]: build the
/// [`ExecutionContext`] (default OOO queue, like `sync`), run `execute` in
/// [`ExecMode::Pipelined`], drain the single output pipe, enqueue a marker over
/// the chain's deps, and wrap it in an [`EventFuture`](crate::EventFuture).
///
/// Synchronous-error paths invalidate the context's cached OOO queue, mirroring
/// `chain_future::run_chain`'s contract.
#[cfg(feature = "async-events")]
fn run_eager_chain<Op>(chain: Op, context: &Context) -> EagerChainFuture<Op::Output>
where
    Op: EagerOp,
    Op::Output: Unpin,
{
    use crate::EventFutureExt;

    // 1. Pick the per-device default OOO queue (same as `sync`).
    let device = context.device().clone();
    let queue = match context.default_outoforder_queue(&device) {
        Ok(q) => q,
        Err(e) => return EagerChainFuture::Errored(Some(e)),
    };
    let ec = ExecutionContext::new(context, device.clone(), queue.raw());

    // 2-3. Run the chain non-blocking and gather its result via `collect` —
    //    the uniform gather seam. `collect` dispatches to the right per-op
    //    reconstruction (single OR multi-output: bundle*, arc_split, the copy
    //    pair all yield their reconstructed value + joined deps), so the async
    //    terminal supports every arity the blocking `sync` does. The eager host
    //    seam executes its closure here and returns any `Err` synchronously (no
    //    worker-thread stash), so a host failure becomes `Errored` directly.
    let (output, deps) = match chain.collect(&ec, ExecMode::Pipelined) {
        Ok(pair) => pair,
        Err(e) => {
            drop(queue);
            context.invalidate_default_outoforder_queue(&device);
            return EagerChainFuture::Errored(Some(e));
        }
    };

    // 4. Enqueue a marker over every event the chain produced. Precise
    //    wait-list — we don't penalise other work sharing this OOO queue.
    //    SAFETY: each `cl_event` is held alive by the `deps` Arc wrappers for
    //    the duration of this call; the marker enqueue retains them internally.
    let wait_list: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
    let marker = match unsafe { queue.raw().enqueue_marker_with_wait_list(&wait_list) } {
        Ok(ev) => ev,
        Err(code) => {
            drop(deps);
            drop(queue);
            context.invalidate_default_outoforder_queue(&device);
            return EagerChainFuture::Errored(Some(Error::OpenCl(code)));
        }
    };
    drop(deps);

    // 4a. clFlush — push every queue the chain touched without blocking.
    //     rusticl is spec-strict and keeps commands `CL_QUEUED` until an
    //     explicit flush, so the marker's `CL_COMPLETE` callback would never
    //     fire and the future would deadlock. flush_all also covers
    //     `.on_device(&dev_b)` chains whose commands land on non-primary queues.
    if let Err(e) = context.flush_all_outoforder_queues() {
        drop(queue);
        context.invalidate_default_outoforder_queue(&device);
        return EagerChainFuture::Errored(Some(e));
    }

    // 5. Wrap the marker in the EventFuture machinery (clSetEventCallback).
    EagerChainFuture::Running {
        output: Some(output),
        event_future: marker.into_future(),
    }
}
