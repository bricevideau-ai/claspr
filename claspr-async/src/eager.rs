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

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, wrap_event};
use claspr::{Context, DeviceSlice, Error, Fillable, MemMode, ReadWrite, Result};
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

// ── EagerOp: the closure-free graph node ───────────────────────────────

/// A node in the eager graph. `execute` runs it against the context, moving its
/// output into its pipe; `describe` reports structure **without** executing.
/// Builder verbs ([`and_then`](EagerOpExt::and_then)) are on [`EagerOpExt`].
pub trait EagerOp: Send {
    /// What this op produces at run time.
    type Output: Send;

    /// The build-time output handle other ops wire to.
    fn output_pipe(&self) -> Pipe<Self::Output>;

    /// Run the op: resolve inputs, enqueue (non-blocking), **move** the result
    /// + its events into the output pipe. Returns `()` — the value lives in the
    /// pipe.
    fn execute(self, ec: &ExecutionContext<'_>) -> Result<()>;

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
        self.execute(&ec)?;
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

    fn execute(self, ec: &ExecutionContext<'_>) -> Result<()> {
        self.source.execute(ec)?;
        self.next.execute(ec)
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

    fn execute(self, ec: &ExecutionContext<'_>) -> Result<()> {
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

    fn execute(self, ec: &ExecutionContext<'_>) -> Result<()> {
        let (mut buf, deps) = self.buf.resolve()?;
        let event = buf
            .fill(self.value)
            .after_all(deps.iter().map(|d| d.as_ref()))
            .submit_on(ec)?;
        self.out.put(buf, vec![wrap_event(event)]);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill".into());
    }
}
