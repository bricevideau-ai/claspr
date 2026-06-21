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

use crate::device_op::{Deps, DeviceOperation, wrap_event};
use crate::exec_ctx::ExecutionContext;
use crate::host_view::{
    DeviceSliceHostView, HostReadableExt, HostWritableExt, MapAccess, MapReadOnly, MapReadWrite,
    MappedSliceHostView,
};
use crate::image::ImageHostTransfer;
use crate::transfer::UploadSource;
use crate::{
    Buffer, Context, DeviceSlice, DeviceSliceUninit, Error, Fillable, HostReadable,
    HostUploadable, HostWritable, MappedSlice, MappedSliceUninit, MemMode, ReadWrite, Result,
    USMSlice, USMSliceUninit, register_drop_callback,
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

    /// Run this op as the **chain terminal** and yield its [`Output`](Self::Output),
    /// having waited on its completion events per `mode`.
    ///
    /// Default (single-output ops): `execute` deposits the value into
    /// [`output_pipe`](Self::output_pipe); this drains it and waits on the
    /// carried [`Deps`]. Multi-output ops (whose storage is per-element pipes,
    /// not a single output pipe) override this to scatter-then-reconstruct the
    /// tuple by draining every element pipe and gathering their deps.
    ///
    /// This is the trait seam that lets [`sync`](EagerOpExt::sync) be uniform
    /// across single- and multi-output ops: it always calls `into_output`.
    fn into_output(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<Self::Output>
    where
        Self: Sized,
    {
        let out = self.output_pipe();
        self.execute(ec, mode)?;
        let (value, deps) = out
            .take()
            .ok_or(Error::NotSupported("eager graph: terminal op produced no output"))?;
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
        let (v, deps) = self
            .src_pipe
            .take()
            .ok_or(Error::NotSupported("eager arced: source produced no output"))?;
        self.out.put(Arc::new(v), deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("arced".into());
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
    let marker = unsafe { ec.cl_queue().enqueue_marker_with_wait_list(&all) }
        .map_err(Error::OpenCl)?;
    Ok(vec![wrap_event(marker)])
}

macro_rules! impl_eager_bundle {
    ($name:ident, $ctor:ident, $($field:ident : $ty:ident : $pf:ident),+) => {
        #[doc = concat!("Eager bundle of independent branches (arity ",
            stringify!($name), "). Built by [`", stringify!($ctor),
            "`]; branches run with no inter-ordering, joined by a marker.")]
        pub struct $name<$($ty: EagerOp),+> {
            $($field: $ty,)+
            // Each branch's output pipe, captured at build so `execute` can
            // drain the branch values + their event deps.
            $($pf: Pipe<<$ty as EagerOp>::Output>,)+
            out: Pipe<( $(<$ty as EagerOp>::Output,)+ )>,
        }

        #[doc = concat!("Construct an eager [`", stringify!($name), "`].")]
        #[allow(clippy::too_many_arguments)]
        pub fn $ctor<$($ty: EagerOp),+>($($field: $ty),+) -> $name<$($ty),+> {
            $(let $pf = $field.output_pipe();)+
            $name { $($field,)+ $($pf,)+ out: Pipe::new() }
        }

        impl<$($ty: EagerOp),+> EagerOp for $name<$($ty),+> {
            type Output = ( $(<$ty as EagerOp>::Output,)+ );

            fn output_pipe(&self) -> Pipe<Self::Output> {
                self.out.clone()
            }

            fn handle(&self) -> Self::Handle {
                self.out.clone()
            }

            fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
                // Each branch pipelines (independent; the join marker is the
                // terminal event). Run all, then drain values + deps.
                $(self.$field.execute(ec, ExecMode::Pipelined)?;)+
                let mut branch_deps: Vec<Deps> = Vec::new();
                let outputs = ( $({
                    let (v, d) = self.$pf.take().ok_or(Error::NotSupported(
                        "eager bundle: a branch produced no output"))?;
                    branch_deps.push(d);
                    v
                },)+ );
                let joined = join_marker(ec, &branch_deps)?;
                self.out.put(outputs, joined);
                Ok(())
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

impl<U: EagerOp> EagerOp for FanOut<U> {
    type Output = Vec<U::Output>;

    fn output_pipe(&self) -> Pipe<Vec<U::Output>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        for op in self.ops {
            op.execute(ec, ExecMode::Pipelined)?;
        }
        let mut branch_deps: Vec<Deps> = Vec::with_capacity(self.pipes.len());
        let mut outputs: Vec<U::Output> = Vec::with_capacity(self.pipes.len());
        for p in &self.pipes {
            let (v, d) = p.take().ok_or(Error::NotSupported(
                "eager fan_out: a branch produced no output",
            ))?;
            outputs.push(v);
            branch_deps.push(d);
        }
        let joined = join_marker(ec, &branch_deps)?;
        self.out.put(outputs, joined);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("fan_out[{}]", self.ops.len()));
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
