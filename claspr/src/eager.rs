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

use crate::device_op::{Deps, wrap_event};
use crate::exec_ctx::ExecutionContext;
use crate::transfer::UploadSource;
use crate::{
    Buffer, Context, DeviceSlice, Error, Fillable, HostReadable, MemMode, ReadWrite, Result,
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

    /// The build-time output handle other ops wire to.
    fn output_pipe(&self) -> Pipe<Self::Output>;

    /// Run the op: resolve inputs, enqueue, **move** the result + its events
    /// into the output pipe. Returns `()` — the value lives in the pipe.
    ///
    /// `mode` is [`ExecMode::Blocking`] only when this op is the chain terminal
    /// (see [`ExecMode`]); composite ops forward `Pipelined` to their upstream
    /// children and `mode` to the tail. A leaf with no native blocking enqueue
    /// ignores `mode`.
    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()>;

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
        F: FnOnce(Pipe<Self::Output>) -> U,
    {
        let next = f(self.output_pipe());
        AndThen { source: self, next }
    }

    /// Run `self` to completion on `context` (forward path; no replay). Blocks
    /// once, here, on the terminal op's events — the only wait in the graph.
    fn sync(self, context: &Context) -> Result<Self::Output> {
        let out = self.output_pipe();
        let device = context.device().clone();
        let queue = context.default_outoforder_queue(&device)?;
        let ec = ExecutionContext::new(context, device, queue.raw());
        // Terminal op may use a native blocking enqueue (no event); upstream
        // ops pipeline. If the terminal blocked, `deps` is empty and the wait
        // loop is a no-op; otherwise we block on its event here.
        self.execute(&ec, ExecMode::Blocking)?;
        let (value, deps) = out
            .take()
            .ok_or(Error::NotSupported("eager graph: terminal op produced no output"))?;
        for d in &deps {
            d.as_ref().wait().map_err(Error::OpenCl)?;
        }
        Ok(value)
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

    fn output_pipe(&self) -> Pipe<U::Output> {
        self.next.output_pipe()
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

// ── Leaf: zero-init alloc (eager port of DeviceSliceAllocUninit+fill) ───

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
