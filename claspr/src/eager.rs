//! Eager struct-graph core — the closure-free `DeviceOperation` replacement.
//!
//! A graph is a **closure-free nested struct** of [`DeviceOp`]s. `.and_then(f)`
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
//! through the pipe payload. Only [`sync`](DeviceOpExt::sync) waits, on the
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

// A bare `Pipe<T>` IS a single-output op — the identity node. This lets a pipe
// (e.g. one branch of a multi-output `handle()`) be passed directly into a
// `bundle!` / `and_then` source position WITHOUT a `forward(..)` wrapper: the
// type system already knows which `handle` slots are pipes, so the coercion is
// implicit. It is the wrapper-free form of [`Forward`] — `output_pipe()` aliases
// the pipe's own storage, so `collect`'s default (`execute` then `take`) pulls
// whatever the upstream producer already deposited.
//
// CONTRACT (same as `forward`): the pipe's producer must have run upstream in the
// same sub-chain before this node is gathered (`AndThen`/composites run the
// source first, so this holds). If not, `collect` finds an empty cell and returns
// the standard "op produced no output" error — loud, never silent.
impl<T: Send + 'static> DeviceOp for Pipe<T> {
    type Output = T;

    fn output_pipe(&self) -> Pipe<T> {
        // The pipe is its OWN output storage — no separate `out`. The producer
        // already (or will) deposit here; we alias it.
        self.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.clone()
    }

    fn execute(self, _ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // No work: the value is deposited by the upstream producer, and
        // `output_pipe()` already aliases this same cell, so `collect` reads it
        // directly. (Unlike `Forward`, there is no resolve-and-re-deposit step —
        // there is nowhere else to move it to.)
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("pipe".into());
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

/// How an op should enqueue, threaded through [`execute`](DeviceOp::execute).
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

// ── DeviceOp: the closure-free graph node ───────────────────────────────

/// A node in the eager graph. `execute` runs it against the context, moving its
/// output into its pipe; `describe` reports structure **without** executing.
/// Builder verbs ([`and_then`](DeviceOpExt::and_then)) are on [`DeviceOpExt`].
pub trait DeviceOp: Send {
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
    /// returned deps. This is the seam that lets [`sync`](DeviceOpExt::sync) be
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

/// Builder verbs for composing [`DeviceOp`]s. Blanket-implemented.
pub trait DeviceOpExt: DeviceOp + Sized {
    /// Sequential composition. **Eager**: runs `f` now with the upstream's
    /// build-time output [`Pipe`], stores the returned op. No closure is kept.
    fn and_then<U, F>(self, f: F) -> AndThen<Self, U>
    where
        U: DeviceOp,
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
        U: DeviceOp,
        F: for<'a> FnOnce(&ExecutionContext<'a>, Self::Output) -> U + Send,
    {
        AndThenWithContext {
            source: self,
            f: Some(f),
            out: Pipe::new(),
        }
    }

    /// Route this op's `execute` to `device`'s default out-of-order queue
    /// instead of the chain's primary queue. Downstream stages resume on the
    /// parent's queue; the routed op's events are valid across both via
    /// OpenCL's shared-context event semantics. See [`OnDevice`].
    fn on_device(self, device: &crate::Device) -> OnDevice<Self> {
        OnDevice {
            source: self,
            device: device.clone(),
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
            + Send
            + 'static,
    {
        AndThenHost {
            source: self,
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
            + Send
            + 'static,
    {
        AndThenHostWithContext {
            source: self,
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
        let result = self.into_output(&ec, ExecMode::Blocking);
        match result {
            // A failing `and_then_host` worker stashed its rich error and signalled
            // its user event negative; the blocking wait may return the cl_event
            // cascade (`Error::OpenCl(-1)`). Prefer the stashed variant.
            Err(cascade) => Err(ec.take_host_error().unwrap_or(cascade)),
            // Even on a "successful" wait, a worker may have stashed an error the
            // wait did NOT surface: pocl does not cascade negative user-event
            // status to commands downstream of it (and a discarded-handle chain
            // may not even wait on the failing op's events). A non-empty slot is
            // itself the failure signal — check it. (Same as the async terminal.)
            Ok(v) => match ec.take_host_error() {
                Some(rust_err) => Err(rust_err),
                None => Ok(v),
            },
        }
    }

    /// Async terminal — run `self` on `context` and return a future that
    /// resolves to its [`Output`](DeviceOp::Output) once every command the
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
    /// returns [`DeviceChainFuture::Errored`] right here — there is no
    /// host-error slot to drain at poll time.
    ///
    /// Arity-agnostic: like [`sync`](Self::sync), `run` gathers via
    /// [`collect`](DeviceOp::collect), so multi-output terminals (`arc_split`,
    /// `bundle*`, the `CopyTo` pair) reconstruct their tuple/array the same way
    /// the blocking terminal does — the future then resolves to that value.
    #[cfg(feature = "async-events")]
    fn run(self, context: &Context) -> DeviceChainFuture<Self::Output>
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
impl<T: DeviceOp> DeviceOpExt for T {}

// ── AndThen: source then next; next eagerly built over source's pipe ───

/// If `src_pipe` still holds a value (the `and_then` closure discarded the
/// source's handle), merge its stranded events into `out_pipe`'s deps so the
/// whole chain's events still gate the terminal. See `AndThen::collect`. No-op
/// when the source pipe was consumed by `next` (the normal case). The discarded
/// source value drops.
fn thread_orphaned_source_deps<A, B>(src_pipe: &Pipe<A>, out_pipe: &Pipe<B>) {
    // Source pipe consumed by `next` (the normal case) → nothing to thread.
    let Some((_discarded, src_deps)) = src_pipe.take() else {
        return;
    };
    // Merge the stranded source events into the out pipe's deps. If `out_pipe` is
    // empty (a multi-output `next` whose storage is its element pipes, not
    // `output_pipe`), `execute` isn't the gather path — `collect` handles
    // orphaned deps for that case directly, so this is a no-op here.
    if let Some((v, mut deps)) = out_pipe.take() {
        deps.extend(src_deps);
        out_pipe.put(v, deps);
    }
}

/// Sequential composition node. Holds the source op and the **already-built**
/// downstream op (which reads the source's output via a [`Pipe`]). No `FnOnce`.
pub struct AndThen<S, U> {
    source: S,
    next: U,
}

impl<S, U> DeviceOp for AndThen<S, U>
where
    S: DeviceOp,
    U: DeviceOp,
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
        // Capture the source's output pipe BEFORE moving it, so we can thread any
        // events the `next` op discarded (see `collect`'s note on orphaned deps).
        let src_pipe = self.source.output_pipe();
        let out_pipe = self.next.output_pipe();
        self.source.execute(ec, ExecMode::Pipelined)?;
        self.next.execute(ec, mode)?;
        thread_orphaned_source_deps(&src_pipe, &out_pipe);
        Ok(())
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
        let src_pipe = self.source.output_pipe();
        self.source.execute(ec, ExecMode::Pipelined)?;
        let (value, mut deps) = self.next.collect(ec, mode)?;
        // ORPHANED DEPS: if the `and_then` closure discarded the source's handle
        // (e.g. `.and_then(|_buf| value(0))`), the source's value + events are
        // still sitting un-taken in its output pipe. Those events MUST still gate
        // the terminal — most critically when the source is an `and_then_host`
        // whose worker thread signals completion via a user event; without this,
        // `sync` would return before the worker ran. Thread them in. (The old
        // closure layer got this for free by passing `prior_evts` into
        // `next.execute`.) The discarded value drops here — the closure didn't
        // want it; its `cl_mem` is retained by any in-flight unmap until done.
        if let Some((_discarded, src_deps)) = src_pipe.take() {
            deps.extend(src_deps);
        }
        Ok((value, deps))
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        self.next.describe(out);
    }
}

// ── Value: lift a host value into the graph ────────────────────────────

/// A host value lifted into the graph — produces it with no device work and no
/// events. Useful as a chain head, or to thread a host value alongside buffers.
///
/// ## By-VALUE handle (the whole point)
///
/// Unlike most ops (whose `Handle` defaults to `Pipe<Output>`), `Value` exposes
/// its **value itself** as the build-time handle (`Handle = T`, requires
/// `T: Clone`). So a downstream `and_then` / bundle closure receives the actual
/// `T`, NOT a `Pipe<T>` — letting host-side computation happen at build:
///
/// ```ignore
/// value(1u32).and_then(|n| value(n + 1))            // n is u32, n+1 works
/// bundle!(kernel, value(1u32)).and_then(|(buf, step)| {  // step is u32...
///     bundle!(kernel2(buf), value(step + 1))        // ...so step+1 computes here
/// })
/// ```
///
/// This is why `value` requires `T: Clone`: `handle(&self)` clones the value out
/// while the op keeps its own copy for `execute` (which still deposits into the
/// pipe for the terminal/bundle reconstruction path). For a non-`Clone` owned
/// resource (a `DeviceSlice` etc.), use [`lift`] instead — it keeps the default
/// `Pipe` handle (a buffer can't and shouldn't be computed on at build).
pub struct Value<T: Send> {
    v: Option<T>,
    out: Pipe<T>,
}

/// Lift a `Clone` host value into the graph with a **by-value** handle (so
/// downstream closures get the value, not a pipe — see [`Value`]). For a
/// non-`Clone` owned resource use [`lift`].
pub fn value<T: Send + Clone + 'static>(v: T) -> Value<T> {
    Value {
        v: Some(v),
        out: Pipe::new(),
    }
}

impl<T: Send + Clone + 'static> DeviceOp for Value<T> {
    type Output = T;
    // By-value handle: downstream gets `T`, enabling build-time host compute.
    type Handle = T;

    fn output_pipe(&self) -> Pipe<T> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        // Clone the value out for the downstream closure; `execute` still has its
        // own copy (in `self.v`) to deposit into the pipe for the terminal path.
        self.v
            .clone()
            .expect("Value::handle after execute — internal eager bug")
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

// ── Lift: an owned (non-Clone) resource as a leaf, default Pipe handle ──

/// An owned resource lifted into the graph as a leaf — like [`Value`] but with
/// the **default `Pipe` handle** instead of by-value, so it works for non-`Clone`
/// types (a [`DeviceSlice`] etc. owns a `cl_mem` and cannot be cloned). Use it
/// to make a caller-owned buffer a chain head or a bundle branch:
///
/// ```ignore
/// lift(buf).and_then(acquire_mapped_view)                  // buffer chain head
/// bundle!(lift(buf), upload(v), alloc_zero(N))             // buffer bundle branch
/// ```
///
/// A buffer can't be computed on at build time anyway, so its downstream handle
/// is a `Pipe` (the value flows; you don't read it until execute). For a `Clone`
/// host value you want to compute on downstream, use [`value`] (by-value handle).
pub struct Lift<T: Send> {
    v: Option<T>,
    out: Pipe<T>,
}

/// Lift an owned resource into the graph (default `Pipe` handle — see [`Lift`]).
pub fn lift<T: Send + 'static>(v: T) -> Lift<T> {
    Lift {
        v: Some(v),
        out: Pipe::new(),
    }
}

impl<T: Send + 'static> DeviceOp for Lift<T> {
    type Output = T;
    // Default `Handle = Pipe<T>` — a resource flows, it isn't read at build.

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
            .expect("Lift::execute called twice — internal eager bug");
        self.out.put(v, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("lift".into());
    }
}

// ── Forward: select/identity — make one upstream Pipe a single-output op ──

/// Forward a single upstream value (a `Pipe<T>`) onward as a single-output
/// [`DeviceOp`]. The identity op: it resolves its input and re-deposits it
/// (threading the deps), changing nothing. Its purpose is **shape**, not work —
/// it lets you pick ONE element out of a multi-output op's handle (e.g. a
/// kernel's `(Pipe<a>, Pipe<b>, Pipe<out>)`, or a bundle's per-branch pipes) and
/// continue on-device with that single value, instead of dropping to the host
/// or inserting a no-op kernel. The selected pipe becomes a normal
/// `DeviceOp<Output = T>` that composes via `and_then` / `bundle` like any leaf.
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

impl<T: Send + 'static> DeviceOp for Forward<T> {
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

// ── DeviceDynOp: type-erased single-output op for conditional graphs ─────

/// Object-safe erasure of [`DeviceOp`], specialised to output `T`. Crate-internal
/// — users go through [`DeviceDynOp`]. `DeviceOp` itself is NOT object-safe (it has
/// an associated `Handle` type and `self`-consuming `collect`/`into_output`), so
/// this mirror trait restates the one operation a terminal/branch needs —
/// gather `(value, deps)` — as a `self: Box<Self>` method that *is*
/// dyn-dispatchable. It delegates to the concrete op's [`collect`](DeviceOp::collect),
/// which already reconstructs any arity down to a single `Output`, so even a
/// multi-output inner op erases cleanly to a single-output `DeviceDynOp`.
trait ErasedDeviceOp<T>: Send {
    fn collect_erased(
        self: Box<Self>,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(T, Deps)>;

    fn describe_erased(&self, out: &mut Vec<String>);
}

impl<O> ErasedDeviceOp<O::Output> for O
where
    O: DeviceOp,
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

/// Type-erased single-output [`DeviceOp`] yielding `T`. Lets `if` / `match` arms
/// produce DIFFERENT concrete op types as long as they agree on `Output` — the
/// eager analog of the legacy closure-layer `DynOp`.
///
/// Each combinator chain has its own deeply-nested concrete type
/// (`AndThen<Upload, AndThen<…>>` vs `Value<T>`), so an `if`/`else` that builds a
/// chain in each arm is a type-mismatch error. Wrapping each arm in
/// `DeviceDynOp::new(...)` erases the concrete type to one nominal
/// `DeviceDynOp<'op, T>`, which is itself an [`DeviceOp`] and composes with
/// `and_then` / `bundle` / `fan_out` like any single-output leaf.
///
/// ```ignore
/// let chain: DeviceDynOp<u32> = if use_kernel {
///     DeviceDynOp::new(upload(v).and_then(|b| ks.fill_u32([N], b, 9)).and_then(|_| value(0u32)))
/// } else {
///     DeviceDynOp::new(value(0u32))            // different concrete type, same Output
/// };
/// let r = chain.sync(&ctx)?;
/// ```
///
/// One heap allocation per erased op. The `'op` lifetime lets the boxed op borrow
/// from the surrounding scope (typically `&Kernels` for kernel launches); it
/// infers to `'static` for chains built from owned data only.
///
/// **Single-output.** `Handle = Pipe<T>` (the default). A multi-output op CAN be
/// erased — its tuple `Output` becomes the `T` of the `DeviceDynOp` (reconstructed
/// via the inner op's `collect`), but the per-element build-time handle is gone;
/// downstream sees one `Pipe<tuple>`. For the conditional-graph use case (arms
/// agreeing on one `Output`) that is exactly right.
pub struct DeviceDynOp<'op, T> {
    inner: Option<Box<dyn ErasedDeviceOp<T> + 'op>>,
    out: Pipe<T>,
}

impl<'op, T: Send + 'static> DeviceDynOp<'op, T> {
    /// Erase a concrete op into a single-output `DeviceDynOp`. Both arms of an
    /// `if`/`match` can produce `DeviceDynOp::new(...)` of the same `T` without
    /// their concrete types matching.
    pub fn new<O>(op: O) -> Self
    where
        O: DeviceOp<Output = T> + 'op,
    {
        DeviceDynOp {
            inner: Some(Box::new(op)),
            out: Pipe::new(),
        }
    }
}

impl<T: Send + 'static> DeviceOp for DeviceDynOp<'_, T> {
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
        // (it is the real terminal work when this DeviceDynOp is the chain tail).
        let inner = self
            .inner
            .take()
            .expect("DeviceDynOp::execute called twice — internal eager bug");
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
pub struct Arced<S: DeviceOp> {
    source: S,
    out: Pipe<Arc<S::Output>>,
}

/// Wrap `source`'s output in `Arc`.
pub fn arced<S: DeviceOp>(source: S) -> Arced<S>
where
    S::Output: Sync,
{
    Arced {
        source,
        out: Pipe::new(),
    }
}

impl<S> DeviceOp for Arced<S>
where
    S: DeviceOp,
    S::Output: Sync,
{
    type Output = Arc<S::Output>;

    fn output_pipe(&self) -> Pipe<Arc<S::Output>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Gather the source via `collect` (reconstructs any arity — a bundle
        // source fills element pipes, not output_pipe), then wrap in Arc.
        let (v, deps) = self.source.collect(ec, ExecMode::Pipelined)?;
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
pub struct ArcSplit<S: DeviceOp, const N: usize>
where
    S::Output: Clone,
{
    source: S,
    // One element pipe per branch (move-once storage); each gets an
    // `Arc::clone` of the source value in `execute`.
    outs: [Pipe<S::Output>; N],
}

/// Build an [`ArcSplit`]: fan `source`'s `Arc<T>` output to `N` read-only
/// branches. `source` is typically an [`arced`] op (`Output = Arc<T>`), so the
/// per-branch clone is a cheap refcount bump. Pick `N` via turbofish to match
/// the destructure arity: `arc_split::<3, _>(arced(upload(…)))`.
pub fn arc_split<const N: usize, S: DeviceOp>(source: S) -> ArcSplit<S, N>
where
    S::Output: Clone,
{
    ArcSplit {
        source,
        outs: std::array::from_fn(|_| Pipe::new()),
    }
}

impl<S, const N: usize> DeviceOp for ArcSplit<S, N>
where
    S: DeviceOp,
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
        // Gather the source via `collect` (any arity), then scatter a clone of
        // its value + events into every branch pipe (Arc::clone is a cheap
        // refcount bump; Deps clone shares the same producer events).
        let (v, deps) = self.source.collect(ec, ExecMode::Pipelined)?;
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
        pub struct $name<$($ty: DeviceOp),+> {
            $($field: $ty,)+
            // Each branch's output pipe, captured at build. These are the
            // move-once storage (like `CopyTo2`'s element pipes): the branch
            // fills its own pipe at `execute`; `handle()` exposes clones so a
            // downstream multi-arg op (e.g. a kernel) can pull each branch as a
            // separate `Pipe<buffer>` input; `into_output` drains them for the
            // terminal-tuple case.
            $($pf: Pipe<<$ty as DeviceOp>::Output>,)+
        }

        #[doc = concat!("Construct an eager [`", stringify!($name), "`].")]
        #[allow(clippy::too_many_arguments)]
        pub fn $ctor<$($ty: DeviceOp),+>($($field: $ty),+) -> $name<$($ty),+> {
            $(let $pf = $field.output_pipe();)+
            $name { $($field,)+ $($pf,)+ }
        }

        impl<$($ty: DeviceOp),+> DeviceOp for $name<$($ty),+> {
            type Output = ( $(<$ty as DeviceOp>::Output,)+ );
            // A tuple of each branch's OWN build-time handle (NOT forced to a
            // pipe). For a buffer-producing branch that handle defaults to
            // `Pipe<buffer>` (so a multi-arg op consumes it via `ToInput`, as
            // before); for a `value(scalar)` branch it is the scalar BY VALUE, so
            // the downstream closure can compute on it at build (e.g. `step + 1`);
            // for a nested bundle / multi-output branch it is that branch's own
            // composite handle. Composing per-branch handles (rather than
            // flattening to pipes) is what carries computable host values down.
            type Handle = ( $(<$ty as DeviceOp>::Handle,)+ );

            fn output_pipe(&self) -> Pipe<Self::Output> {
                // Multi-output storage is the per-branch pipes; this single pipe
                // is never filled or drained (the default `into_output` is
                // overridden, and `and_then` uses `handle()`). Return a fresh
                // empty pipe — well-typed, never read.
                Pipe::new()
            }

            fn handle(&self) -> Self::Handle {
                // Delegate to each branch's own `handle()` — preserves by-value
                // for `value`, pipe for buffers, composite for nested bundles.
                ( $(self.$field.handle(),)+ )
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
pub struct FanOut<U: DeviceOp> {
    ops: Vec<U>,
    pipes: Vec<Pipe<U::Output>>,
    out: Pipe<Vec<U::Output>>,
}

/// Build a fan-out: `f` is called now for each input, producing the branch ops.
pub fn fan_out<I, F, U>(inputs: Vec<I>, mut f: F) -> FanOut<U>
where
    F: FnMut(I) -> U,
    U: DeviceOp,
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
/// available — use whichever fits the call site. Named `DeviceFanOutExt` to
/// avoid clashing with the old [`FanOutExt`](crate::FanOutExt) (both are
/// re-exported at the crate root).
pub trait DeviceFanOutExt<I>: Sized {
    /// See [`fan_out`] — this delegates to it.
    fn fan_out<F, U>(self, f: F) -> FanOut<U>
    where
        F: FnMut(I) -> U,
        U: DeviceOp;
}

impl<I> DeviceFanOutExt<I> for Vec<I> {
    fn fan_out<F, U>(self, f: F) -> FanOut<U>
    where
        F: FnMut(I) -> U,
        U: DeviceOp,
    {
        fan_out(self, f)
    }
}

impl<U: DeviceOp> DeviceOp for FanOut<U> {
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

impl<T, M> DeviceOp for AllocZero<T, M>
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

impl<T, M> DeviceOp for Fill<T, M>
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

impl<T, M> DeviceOp for Upload<T, M>
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

impl<T, M> DeviceOp for Download<T, M>
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
/// kernels need after the buffer is migrated is [`on_device`](DeviceOpExt::on_device).
///
/// ## Shape: a leaf, not a wrapping method
///
/// Unlike [`on_device`](DeviceOpExt::on_device) (which *routes* an upstream op's
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

impl<T, M> DeviceOp for TransferToDevice<T, M>
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

impl<T, M> DeviceOp for FillDeviceUninit<T, M>
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

impl<T, M> DeviceOp for FillMappedUninit<T, M>
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

impl<T, M> DeviceOp for FillUsmUninit<T, M>
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

impl<T, M> DeviceOp for WriteDeviceUninit<T, M>
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

// ── Leaf: write host data into an existing (init) DeviceSlice ───────────────

/// Write host `src` into an already-initialised `DeviceSlice`, in place, via a
/// non-blocking `clEnqueueWriteBuffer` — the eager analog of the closure layer's
/// `device_slice_write(buf, src)`. The buffer passes through as the op's output.
///
/// This is a real **async host→device transfer** (a queue command), NOT a
/// map/host-memcpy/unmap host seam: `submit_on` enqueues `CL_FALSE` and returns
/// the write event as the op's deps, so the write overlaps downstream device
/// work; `register_drop_callback` keeps the host `src` alive until the DMA
/// completes (`CL_COMPLETE`). The `Blocking` terminal path uses `wait_on`
/// (`CL_BLOCKING`) instead, mirroring [`WriteDeviceUninit`].
pub struct WriteDevice<T, M: MemMode = ReadWrite> {
    buf: Input<DeviceSlice<T, M>>,
    src: Option<UploadSource<T>>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an eager in-place write leaf over an existing `DeviceSlice` (concrete or
/// piped). `M: HostWritable` — same gate as the closure layer's
/// `device_slice_write`.
pub fn write<T, M, S>(buf: impl Into<Input<DeviceSlice<T, M>>>, src: S) -> WriteDevice<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteDevice {
        buf: buf.into(),
        src: Some(src.into()),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for WriteDevice<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (mut buf, deps) = self.buf.resolve()?;
        let src = self
            .src
            .take()
            .expect("WriteDevice::execute called twice — internal eager bug");
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
                // Keep-alive: drop the host source when CL_COMPLETE fires (the
                // runtime is done reading the host heap exactly then).
                register_drop_callback(&event, Box::new(src))?;
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write".into());
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

impl<T, M> DeviceOp for WriteMappedUninit<T, M>
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

impl<T, M> DeviceOp for WriteUsmUninit<T, M>
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

impl<T, M> DeviceOp for UsmSlice<T, M>
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

impl<T, M> DeviceOp for UsmAllocUninit<T, M>
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

// ── Leaf: alloc an uninit DeviceSlice (eager DeviceSliceAllocUninit) ─────────

/// Allocate a [`DeviceSliceUninit<T, M>`] inside the graph. Producing source
/// leaf; allocation happens at execute (`DeviceSlice::alloc_uninit`), so the
/// uninit is a graph-produced value a downstream `fill_device_uninit` /
/// `write_device_uninit` consumes — the eager analog of the old layer's
/// `DeviceSliceAllocUninit`. Mirrors [`UsmAllocUninit`].
pub struct DeviceAllocUninit<T, M: MemMode = ReadWrite> {
    len: usize,
    out: Pipe<DeviceSliceUninit<T, M>>,
    _t: PhantomData<fn() -> (T, M)>,
}

/// Build an eager uninit-`DeviceSlice` alloc leaf.
pub fn device_alloc_uninit<T, M>(len: usize) -> DeviceAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    DeviceAllocUninit {
        len,
        out: Pipe::new(),
        _t: PhantomData,
    }
}

impl<T, M> DeviceOp for DeviceAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = DeviceSliceUninit<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSliceUninit<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // alloc_uninit is pure host code — no in-flight event, mode N/A.
        let uninit = DeviceSlice::<T, M>::alloc_uninit(ec.context(), self.len)?;
        self.out.put(uninit, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("device_alloc_uninit(len={})", self.len));
    }
}

// ── Leaf: alloc an uninit MappedSlice (eager MappedSliceAllocUninit) ─────────

/// Allocate a [`MappedSliceUninit<T, M>`] inside the graph. Producing source
/// leaf; allocation happens at execute (`MappedSlice::alloc_uninit`), which on a
/// no-SVM device surfaces [`Error::SvmNotAvailable`] **at the graph terminal**
/// (not eagerly) — the eager analog of the old layer's `MappedSliceAllocUninit`.
/// A downstream `fill_mapped_uninit` / `write_mapped_uninit` consumes the result.
pub struct MappedAllocUninit<T, M: MemMode = ReadWrite> {
    len: usize,
    out: Pipe<MappedSliceUninit<T, M>>,
    _t: PhantomData<fn() -> (T, M)>,
}

/// Build an eager uninit-`MappedSlice` alloc leaf.
pub fn mapped_alloc_uninit<T, M>(len: usize) -> MappedAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    MappedAllocUninit {
        len,
        out: Pipe::new(),
        _t: PhantomData,
    }
}

impl<T, M> DeviceOp for MappedAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = MappedSliceUninit<T, M>;

    fn output_pipe(&self) -> Pipe<MappedSliceUninit<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // alloc_uninit is pure host code; on a no-SVM device it returns
        // `SvmNotAvailable` here (at execute → surfaces at the terminal).
        let uninit = MappedSlice::<T, M>::alloc_uninit(ec.context(), self.len)?;
        self.out.put(uninit, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("mapped_alloc_uninit(len={})", self.len));
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

impl<I> DeviceOp for ImageUploadEager<I>
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

impl<I> DeviceOp for ImageDownloadEager<I>
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

impl<T, M> DeviceOp for AcquireDeviceView<T, M>
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

impl<T, M> DeviceOp for AcquireDeviceViewRead<T, M>
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

impl<T, M, A> DeviceOp for ReleaseDeviceView<T, M, A>
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

impl<T, M> DeviceOp for AcquireMappedView<T, M>
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

impl<T, M> DeviceOp for AcquireMappedViewRead<T, M>
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

impl<T, M, A> DeviceOp for ReleaseMappedView<T, M, A>
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

impl<Src, Dst> DeviceOp for CopyTo2<Src, Dst>
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
/// [`and_then_with_context`](DeviceOpExt::and_then_with_context).
///
/// Unlike [`AndThen`] (builder at construction, `Pipe` handle), the closure
/// here receives `&ExecutionContext` + the upstream's **runtime value**, so it
/// can read `ec.device()` / `ec.context()` / route via [`on_device`](DeviceOpExt::on_device) while
/// building the downstream op. The downstream op is therefore built — and run —
/// at execute time.
pub struct AndThenWithContext<S: DeviceOp, U: DeviceOp, F> {
    source: S,
    f: Option<F>,
    out: Pipe<U::Output>,
}

impl<S, U, F> DeviceOp for AndThenWithContext<S, U, F>
where
    S: DeviceOp,
    U: DeviceOp,
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
        // Gather the source via `collect` (any arity); its value + events feed
        // the downstream op the closure builds.
        let (value, src_deps) = self.source.collect(ec, ExecMode::Pipelined)?;
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
/// out-of-order queue — built by [`on_device`](DeviceOpExt::on_device).
///
/// No user closure: at execute it resolves the target device's queue from the
/// running context, builds a sibling [`ExecutionContext`] (same context + same
/// host-error slot, different device + queue), and runs `source` against it.
/// The source's events are valid across queues of the same context, so
/// downstream stages on the parent's queue can wait on them cross-device.
pub struct OnDevice<S: DeviceOp> {
    source: S,
    device: crate::Device,
    out: Pipe<S::Output>,
}

impl<S, S2> DeviceOp for OnDevice<S>
where
    S: DeviceOp<Output = S2>,
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
        // Gather the source against the child EC via `collect` (any arity).
        let (value, deps) = self.source.collect(&child, mode)?;
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
/// in chain order — built by [`and_then_host`](DeviceOpExt::and_then_host).
///
/// ## In-queue, worker-thread (NOT submit-thread)
///
/// At execute (on the submitting thread, all non-blocking): gather the source,
/// enqueue maps for its value (wait-list = its upstream events), create a user
/// event downstream gates on, enqueue the unmaps gated on that user event, then
/// **spawn a worker thread** and return immediately — the chain continues at
/// submit time, the host stage overlaps pipelined device work.
///
/// The worker waits the map events (host-side, on its own thread), runs the
/// closure on the view inside `catch_unwind` (mutations persist via the unmap),
/// and signals the user event: `CL_COMPLETE` on success, negative on closure
/// `Err` / panic (→ `Error::HostPanic`) / map failure. `Output = S::Output`
/// (the value passes through unchanged).
///
/// **Errors** are stashed in the chain-wide host-error slot on the
/// `ExecutionContext` and the user event is signalled
/// negative; the status cascades through the in-queue dependency graph and the
/// terminal (`sync` / `run`) prefers the stashed rich variant over the
/// `Error::OpenCl(-1)` cascade. On error the worker forces a defensive
/// synchronous unmap so the buffer is left clean. This mirrors the old
/// closure-layer `and_then_host.rs` exactly. (Distinct from any future
/// host-VALUE seam, which would be pure host compute with no map.)
pub struct AndThenHost<S: DeviceOp, F>
where
    S::Output: crate::mappable::Mappable,
{
    source: S,
    f: Option<F>,
    out: Pipe<S::Output>,
}

/// Like [`AndThenHost`] but the closure also receives `&Context` — built by
/// [`and_then_host_with_context`](DeviceOpExt::and_then_host_with_context).
pub struct AndThenHostWithContext<S: DeviceOp, F>
where
    S::Output: crate::mappable::Mappable,
{
    source: S,
    f: Option<F>,
    out: Pipe<S::Output>,
}

/// Shared body for the host seam: enqueue maps for the source value (wait-list =
/// its upstream events), create a user event downstream gates on, enqueue the
/// unmaps (gated on the user event), then **spawn a worker thread** that waits
/// the map events, runs `host_call` on the view, and signals the user event.
/// Returns the (unchanged) value + the unmap events as deps **without blocking
/// the submitting thread** — so the host stage sits *in-queue* and overlaps
/// pipelined device work. This is the async machinery ported from the old
/// closure-layer `and_then_host.rs`; running it synchronously was a regression.
///
/// Errors (closure `Err`, panic → `HostPanic`, map-wait failure) are stashed in
/// the chain-wide host-error slot (first-writer-wins) and the user event is
/// signalled with a negative status; the negative status cascades through the
/// in-queue dependency graph and the terminal (`sync`/`run`) prefers the stashed
/// rich variant over the `Error::OpenCl(-1)` cascade.
fn run_host_seam<O, F>(
    source_value: O,
    source_deps: Deps,
    ec: &ExecutionContext<'_>,
    host_call: F,
) -> Result<(O, Deps)>
where
    O: crate::mappable::Mappable,
    F: for<'a> FnOnce(<O as crate::mappable::Mappable>::View<'a>) -> Result<()> + Send + 'static,
{
    use crate::mappable::Mappable;
    use crate::{Launcher, complete_user_event, create_user_event};
    use std::sync::Arc;

    let q = ec.cl_queue();
    // Non-blocking maps with the upstream events as wait-list — the worker waits
    // these (host-side, on its own thread) before reading the mapped memory.
    let source_cl: Vec<crate::cl_event> = source_deps.iter().map(|d| d.as_ref().get()).collect();
    let (mut handle, map_events) = source_value.map(q, &source_cl)?;

    // The user event downstream waits on (via the unmaps). Worker signals it.
    let user_event = Arc::new(create_user_event(ec.context())?);

    // Enqueue the unmaps gated on the user event — they fire once the worker
    // signals completion. After this point we MUST signal the user event before
    // any early return, or the queue would wait on it forever.
    let unmap_events = match <O as Mappable>::enqueue_unmap(&mut handle, q, &[user_event.get()]) {
        Ok(evs) => evs,
        Err(e) => {
            let _ = complete_user_event(&user_event, -1);
            return Err(e);
        }
    };

    // Spawn the worker. It owns the handle, the map events, the source events
    // (for upstream-error short-circuit), the user-event Arc clone, the chain's
    // host-error slot, and the closure.
    let worker_user_event = Arc::clone(&user_event);
    let worker_host_error = ec.host_error_slot();
    std::thread::spawn(move || {
        let (status, mut handle, rust_err) =
            run_host_worker::<O, F>(handle, map_events, source_deps, host_call);
        // Stash the rich Rust error before signalling, so the terminal can prefer
        // it over the cl_event cascade. First-writer-wins (a concurrent failing
        // host worker in the same bundle/fan-out may already have written).
        if let Some(err) = rust_err {
            let mut slot = worker_host_error.lock().unwrap();
            if slot.is_none() {
                *slot = Some(err);
            }
        }
        if status < 0 {
            // On error the queued unmap (waiting on the user event) is terminated
            // by the runtime rather than executing — leaving the buffer mapped.
            // Force the defensive sync unmap NOW so the buffer is clean by the
            // time the terminal observes the error.
            <O as Mappable>::mark_unmap_not_done(&mut handle);
            drop(handle);
        }
        let _ = complete_user_event(&worker_user_event, status);
        // Success path: handle drops here (no-op — the queued unmap fires via the
        // user event on its own).
    });

    // Downstream gates on the unmap events (transitively the user event). When
    // the output has no buffers (scalar / unit), unmaps are empty — fall back to
    // the user event directly so downstream still has a gate.
    let deps_out: Deps = if unmap_events.is_empty() {
        vec![user_event]
    } else {
        unmap_events.into_iter().map(wrap_event).collect()
    };
    Ok((source_value, deps_out))
}

/// Worker body for [`run_host_seam`]. Waits the source + map events, runs the
/// closure under `catch_unwind`, and returns `(status, handle, optional rich
/// error)` so the caller can stash the error + trigger the defensive unmap on
/// failure. `status` is `CL_COMPLETE` on success, negative otherwise.
fn run_host_worker<O, F>(
    mut handle: O::MapHandle,
    map_events: Vec<crate::Event>,
    source_deps: Deps,
    host_call: F,
) -> (i32, O::MapHandle, Option<Error>)
where
    O: crate::mappable::Mappable,
    F: for<'a> FnOnce(<O as crate::mappable::Mappable>::View<'a>) -> Result<()> + Send + 'static,
{
    use crate::mappable::Mappable;
    use opencl3::event::CL_COMPLETE;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    // Short-circuit on an upstream chain error (negative source-event status,
    // e.g. a previous failing host seam). Don't stash — the upstream already did.
    for ev in &source_deps {
        if ev.as_ref().wait().is_err() {
            return (-1, handle, None);
        }
    }
    // Map-event failure is a host-observable CL error — stash the real cause.
    for ev in &map_events {
        if let Err(e) = ev.wait() {
            return (-1, handle, Some(Error::OpenCl(e)));
        }
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let view = <O as Mappable>::view(&mut handle);
        host_call(view)
    }));
    match result {
        Ok(Ok(())) => (CL_COMPLETE, handle, None),
        Ok(Err(rust_err)) => (-1, handle, Some(rust_err)),
        Err(panic) => {
            // `catch_unwind` yields `Box<dyn Any + Send>`; the payload is
            // typically `&'static str` (`panic!("lit")`) or `String`
            // (`panic!("{}", x)`). Anything else gets a placeholder.
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            (-1, handle, Some(Error::HostPanic(msg)))
        }
    }
}

impl<S, F> DeviceOp for AndThenHost<S, F>
where
    S: DeviceOp,
    S::Output: crate::mappable::Mappable,
    F: for<'a> FnOnce(<S::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
        + Send
        + 'static,
{
    type Output = S::Output;

    fn output_pipe(&self) -> Pipe<S::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Gather the source via `collect` (any arity — a bundle source fills
        // element pipes, not output_pipe).
        let (value, deps) = self.source.collect(ec, ExecMode::Pipelined)?;
        let f = self
            .f
            .take()
            .expect("AndThenHost::execute called twice — internal eager bug");
        let (out_value, out_deps) = run_host_seam::<S::Output, F>(value, deps, ec, f)?;
        self.out.put(out_value, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("and_then_host".into());
    }
}

impl<S, F> DeviceOp for AndThenHostWithContext<S, F>
where
    S: DeviceOp,
    S::Output: crate::mappable::Mappable,
    F: for<'a> FnOnce(&Context, <S::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
        + Send
        + 'static,
{
    type Output = S::Output;

    fn output_pipe(&self) -> Pipe<S::Output> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(mut self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Gather the source via `collect` (any arity).
        let (value, deps) = self.source.collect(ec, ExecMode::Pipelined)?;
        let f = self
            .f
            .take()
            .expect("AndThenHostWithContext::execute called twice — internal eager bug");
        // Move a `Context` clone (cheap, Arc-backed, 'static) into the worker
        // closure so it can call `f(&context, view)`; the view borrow is supplied
        // by the seam. The closure is Send + 'static (context + f both are).
        let context = ec.context().clone();
        let (out_value, out_deps) =
            run_host_seam::<S::Output, _>(value, deps, ec, move |view| f(&context, view))?;
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
/// built by [`profiled`](DeviceProfileExt::profiled). Mirrors the old
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
pub struct Profiled<S: DeviceOp, F> {
    source: S,
    cb: Option<F>,
    out: Pipe<S::Output>,
}

/// Extension trait adding [`profiled`](Self::profiled) to every [`DeviceOp`].
/// Separate from [`DeviceOpExt`] to mirror the old layer's
/// `DeviceOperationProfileExt`. Blanket-implemented.
pub trait DeviceProfileExt: DeviceOp + Sized {
    /// Register `cb` to receive the wall-clock [`ProfilingInfo`](crate::ProfilingInfo) for everything
    /// `self` enqueued onto the chain's queue. The closure fires on an OpenCL
    /// callback thread when the marker event completes. See [`Profiled`].
    fn profiled<F>(self, cb: F) -> Profiled<Self, F>
    where
        F: FnOnce(Result<crate::ProfilingInfo>) + Send + 'static,
    {
        Profiled {
            source: self,
            cb: Some(cb),
            out: Pipe::new(),
        }
    }
}
impl<T: DeviceOp> DeviceProfileExt for T {}

impl<S, F> DeviceOp for Profiled<S, F>
where
    S: DeviceOp,
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
        // Gather the source via `collect` (any arity).
        let (value, source_deps) = self.source.collect(ec, ExecMode::Pipelined)?;
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

// ── DeviceChainFuture: the async `.run().await` terminal ────────────────

/// Future returned by [`DeviceOpExt::run`]. Resolves to `Result<T>` once the
/// chain's commands have all completed on the device (or immediately, with an
/// error, if the chain failed to submit or any host seam returned `Err`).
///
/// The eager analog of `chain_future::ChainFuture`. The host seam
/// (`run_host_seam`) runs its closure on a worker thread and stashes any failure
/// into the chain's host-error slot before signalling its user event with a
/// negative status. That status cascades into the trailing marker, so the
/// future's marker poll resolves with `Err`; [`Running`](Self::Running) then
/// prefers the stashed rich variant (closure `Err`, `HostPanic`) over the
/// `Error::OpenCl(-1)` cascade — mirroring the `sync` terminal.
#[cfg(feature = "async-events")]
pub enum DeviceChainFuture<T> {
    /// Chain failed during setup, `execute`, or marker enqueue. The error
    /// surfaces on the first `poll`.
    Errored(Option<Error>),
    /// Chain submitted successfully; waiting for the trailing marker event to
    /// complete. The host-side `Output` is already materialised (drained from
    /// the output pipe at `run` time); the future just gates *when* the caller
    /// sees it on whether the queue work is done. `host_error` is the chain's
    /// stash, preferred over the cl_event cascade if the marker resolves `Err`.
    Running {
        output: Option<T>,
        event_future: crate::EventFuture,
        host_error: std::sync::Arc<std::sync::Mutex<Option<Error>>>,
    },
}

// `T: Unpin` covers every realistic chain output (`Vec<u8>`, `DeviceSlice<T>`,
// `Arc<T>`, tuples of those, ...) and lets us pin-project via the cheap
// `Pin::get_mut`. Mirrors `ChainFuture`'s bound.
#[cfg(feature = "async-events")]
impl<T: Unpin> std::future::Future for DeviceChainFuture<T> {
    type Output = Result<T>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        use std::task::Poll;
        let this = self.as_mut().get_mut();
        match this {
            DeviceChainFuture::Errored(slot) => Poll::Ready(Err(slot
                .take()
                .expect("DeviceChainFuture polled after Ready (Errored)"))),
            DeviceChainFuture::Running {
                output,
                event_future,
                host_error,
            } => match std::pin::Pin::new(event_future).poll(cx) {
                Poll::Pending => Poll::Pending,
                // Marker resolved Err: a host worker (or a CL command) failed.
                // Prefer the rich Rust variant the worker stashed over the
                // cl_event cascade, mirroring `sync`.
                Poll::Ready(Err(e)) => {
                    Poll::Ready(Err(host_error.lock().unwrap().take().unwrap_or(e)))
                }
                // Even on a "successful" marker, a host worker may have stashed
                // an error the marker did NOT propagate: pocl's
                // `clEnqueueMarkerWithWaitList` does not cascade negative status
                // from a user event in its wait-list (it reports CL_COMPLETE while
                // the chain genuinely failed). A non-empty slot is itself the
                // failure signal. (Same handling as the old `ChainFuture`.)
                Poll::Ready(Ok(())) => {
                    if let Some(rust_err) = host_error.lock().unwrap().take() {
                        return Poll::Ready(Err(rust_err));
                    }
                    Poll::Ready(Ok(output
                        .take()
                        .expect("DeviceChainFuture polled after Ready (Running)")))
                }
            },
        }
    }
}

/// Crate-internal worker behind [`DeviceOpExt::run`]: build the
/// [`ExecutionContext`] (default OOO queue, like `sync`), run `execute` in
/// [`ExecMode::Pipelined`], drain the single output pipe, enqueue a marker over
/// the chain's deps, and wrap it in an [`EventFuture`](crate::EventFuture).
///
/// Synchronous-error paths invalidate the context's cached OOO queue, mirroring
/// `chain_future::run_chain`'s contract.
#[cfg(feature = "async-events")]
fn run_eager_chain<Op>(chain: Op, context: &Context) -> DeviceChainFuture<Op::Output>
where
    Op: DeviceOp,
    Op::Output: Unpin,
{
    use crate::EventFutureExt;

    // 1. Pick the per-device default OOO queue (same as `sync`).
    let device = context.device().clone();
    let queue = match context.default_outoforder_queue(&device) {
        Ok(q) => q,
        Err(e) => return DeviceChainFuture::Errored(Some(e)),
    };
    let ec = ExecutionContext::new(context, device.clone(), queue.raw());
    // Clone the chain's host-error slot out before `ec` drops — host-seam workers
    // stash failures here, and the future reconciles them at poll time.
    let host_error = ec.host_error_slot();

    // 2-3. Run the chain non-blocking and gather its result via `collect` —
    //    the uniform gather seam. `collect` dispatches to the right per-op
    //    reconstruction (single OR multi-output: bundle*, arc_split, the copy
    //    pair all yield their reconstructed value + joined deps), so the async
    //    terminal supports every arity the blocking `sync` does. A host-seam
    //    setup error (map/unmap enqueue) still surfaces synchronously here; a
    //    host-CLOSURE failure surfaces at poll time via the host-error slot.
    let (output, deps) = match chain.collect(&ec, ExecMode::Pipelined) {
        Ok(pair) => pair,
        Err(e) => {
            drop(queue);
            context.invalidate_default_outoforder_queue(&device);
            return DeviceChainFuture::Errored(Some(e));
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
            return DeviceChainFuture::Errored(Some(Error::OpenCl(code)));
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
        return DeviceChainFuture::Errored(Some(e));
    }

    // 5. Wrap the marker in the EventFuture machinery (clSetEventCallback).
    DeviceChainFuture::Running {
        output: Some(output),
        event_future: marker.into_future(),
        host_error,
    }
}
