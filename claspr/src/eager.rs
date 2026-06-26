//! Eager struct-graph core — the closure-free device-graph layer (`DeviceOp`).
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
//! This module IS the Tier 2 device-graph layer. The former closure-based
//! `DeviceOperation` layer it replaced has been removed; the only residue is the
//! tiny [`DeviceEnqueue`] contract a few primitive leaves (host-view map/unmap,
//! the polymorphic `copy_to` family) delegate their raw enqueue body to.

use crate::copy::CopyTo;
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
    USMSliceUninit,
};
use std::any::{Any, TypeId, type_name};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

// ── Deps: the event wait-list threaded through the graph ────────────────

/// A single tracked event in a [`Deps`] chain. `Arc`-wrapped so it can be
/// cheaply shared across parallel branches in `bundle!` / `fan_out` without
/// extra `clRetainEvent` calls.
pub type Dep = Arc<crate::Event>;

/// The wait-list / produced-event list threaded through every op's
/// [`execute`](DeviceOp::execute). Empty at chain start; one element per device
/// op the previous step enqueued; multi-element after a parallel join
/// (`bundle`/`fan_out`) collapses children's events into the marker that joins
/// them.
pub type Deps = Vec<Dep>;

/// Borrow each [`Dep`] as `&Event` for an `after_all(...)` call on a Tier 1 op
/// builder.
pub fn deps_as_events(deps: &Deps) -> impl Iterator<Item = &crate::Event> {
    deps.iter().map(|d| d.as_ref())
}

/// Wrap an opencl3 [`Event`](crate::Event) in a [`Dep`].
pub fn wrap_event(event: crate::Event) -> Dep {
    Arc::new(event)
}

// ── DeviceEnqueue: minimal raw-enqueue contract for delegated primitives ──
//
// A handful of eager leaves (the host-view acquire/release ops in `host_view.rs`
// and the polymorphic `copy_to` family in `copy.rs`) can't be re-derived inline:
// they reach into private fields and own per-family `clEnqueue*` bodies. Rather
// than duplicate those bodies, the eager wrapper holds the buffer/view and
// delegates to a small op type whose only job is one non-blocking enqueue
// returning `(Output, Deps)`. This trait is that contract — the residue of the
// old `DeviceOperation` trait, pared down to the single `run` method the eager
// graph actually needs (no terminals, no combinators, no blanket).

/// One non-blocking enqueue: take the upstream `deps` as the wait-list, enqueue,
/// and return the produced value plus the events the enqueue created. Implemented
/// by the few primitive ops the eager graph delegates to (host-view map/unmap,
/// the `copy_to` family).
pub trait DeviceEnqueue: Send + Sized {
    /// The host value the enqueue produces.
    type Output: Send;
    /// Enqueue against `ec` with `deps` as the wait-list; return `(value, Deps)`.
    fn run(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)>;
}

// ── Cell<T>: interior-mutable resource slot (the reusable-graph primitive) ──

/// An interior-mutable slot holding (or temporarily not holding) a resource.
/// The unifying primitive of the reusable graph: a [`Pipe`] is a cell that
/// also carries [`Deps`]; a [`Concrete`](Input::Concrete) input is a cell that
/// is *lent* during a run and *returned* on `Checkout` drop. `Arc` so a run can
/// hold a clone to deposit the value back home.
pub type Cell<T> = Arc<Mutex<Option<T>>>;

// ── Typed slots: per-tag value type compile-time, presence runtime ─────────

/// A compile-time **tag** naming a typed hole in a reusable graph. The tag *type*
/// is the identity key (matched by [`TypeId`] at bind time); its [`Value`](Tag::Value)
/// is the one buffer type that tag carries — fixed at compile time, so a `slot!(Buf)`
/// and a `Buf(value)` binding are checked against the same type without any
/// turbofish.
///
/// Declared via the [`slots!`](crate::slots) macro, which emits a `pub struct
/// Tag(pub Value)` tuple struct (binding is plain tuple-struct construction —
/// `Buf(value)`, no `Fn`/`fn_traits` games) and this trait impl. The struct value
/// is never inspected; only `TypeId::of::<Tag>()` is, so the wrapper is a pure
/// move-the-value carrier.
///
/// Tag *presence* (was every slot bound?) is a **runtime** property, checked at
/// [`sync`](DeviceOpExt::sync) by walking the graph's slot cells — deliberately
/// NOT compile-time set-algebra (the abandoned HList approach). Only the per-tag
/// *value type* is compile-time.
pub trait Tag: Sized + 'static {
    /// The buffer type this tag carries. `Send + 'static` so it can flow through
    /// the same [`Cell`]/[`Checkout`] lend-and-return machinery as a concrete
    /// input.
    type Value: Send + 'static;

    /// Unwrap the tuple-struct binding `Tag(value)` to its value (moved). The
    /// [`slots!`](crate::slots) macro emits this as `self.0`; it is the only way
    /// [`call`](DeviceOpExt::call) can pull the value out of a generic `Tg` wrapper
    /// (a generic tuple-struct field is not nameable).
    fn into_value(self) -> Self::Value;
}

/// A type-erased carrier for one `call(Tag(value))` binding, folded into a graph's
/// slot cells by [`bind_slots`](DeviceOp::bind_slots).
///
/// Carries the tag's [`TypeId`] and the boxed value (`Box<dyn Any>` over the tag's
/// `Value`). The binding **MOVES**: each [`Input::Slot`] whose `id` matches takes
/// the value (downcast back to its concrete type) into its cell, then clears the
/// binder so a second matching cell sees nothing — a single buffer is single-owner,
/// so a tag fills at most one slot occurrence per `call`.
pub struct SlotBinder {
    id: TypeId,
    /// `None` once a matching slot consumed it. `Box<dyn Any + Send>` holds the
    /// tag's `Value` by value.
    value: Option<Box<dyn Any + Send>>,
}

impl SlotBinder {
    /// Build a binder for tag `Tg` carrying `value` (moved). Use via
    /// [`DeviceOpExt::call`].
    pub fn new<Tg: Tag>(value: Tg::Value) -> Self {
        SlotBinder {
            id: TypeId::of::<Tg>(),
            value: Some(Box::new(value)),
        }
    }

    /// Whether the binding has already been deposited into a matching slot cell.
    /// `bind_slots` walks short-circuit on this (one `call` binds one cell), and
    /// the kernel-op codegen checks it before each arg.
    pub fn is_consumed(&self) -> bool {
        self.value.is_none()
    }
}

// ── Rehome: type-erased "deposit an output back into its origin cell" ───

/// How a run's output value is returned to the (possibly weaker-typed) cell it
/// came from, re-arming the graph on [`Checkout`] drop. The home channel carries
/// a `Box<dyn Rehome<Out>>` instead of a bare `Cell<Out>` so the origin cell may
/// hold a DIFFERENT (weaker) type than the output: a copy with a
/// [`DeviceSliceUninit`] dst lends a
/// `Cell<DeviceSliceUninit<T, M>>` but produces an `Init` `DeviceSlice<T, M>`;
/// the rehome DOWNGRADES the Init buffer back into the uninit cell (a sound,
/// `unsafe`-free re-wrap — `Init` is the stronger capability).
///
/// Keyed by the pipe's OWN output type `Out`, so generalizing `home` does not
/// ripple into `Pipe<T>`'s type parameter. The common case (an in-place op whose
/// output type equals its input cell type) is the identity rehome:
/// [`Cell<Out>`] implements `Rehome<Out>` directly.
///
/// `: Send` so the boxed home can travel through the graph alongside the value
/// (every buffer that reaches the home channel is already `Send + 'static`).
pub trait Rehome<Out>: Send {
    /// Deposit `value` into the origin cell (consuming the boxed home).
    fn rehome(self: Box<Self>, value: Out);
}

/// A boxed, type-erased return home for an output of type `Out` — the home
/// channel's payload (`None` = nothing to return). Aliased so the `Pipe` /
/// `Input` / `Checkout` signatures stay readable.
pub type BoxedHome<Out> = Box<dyn Rehome<Out>>;

/// Identity rehome: an output returns to a cell of its own type (the in-place
/// case — fill/scale/kernel-buffer-arg/copy's same-typed sides). This is the
/// behaviour the old `Option<Cell<T>>` home had, now expressed through the trait.
impl<T: Send> Rehome<T> for Cell<T> {
    fn rehome(self: Box<Self>, value: T) {
        *self.lock().unwrap() = Some(value);
    }
}

// ── Pipe<T> + Input<T>: the graph edge ─────────────────────────────────

/// A build-time handle to an op's future output, carrying `(value, Deps)`. The
/// producing op **moves** its value + the events its commands enqueued in at
/// execute; the consuming op moves them out as its own wait-list. Cheap-clone
/// (`Arc`); identity is the `Arc` cell, so independently-built subgraphs
/// compose with no global numbering.
pub struct Pipe<T> {
    cell: Arc<Mutex<Option<PipePayload<T>>>>,
}

/// What a [`Pipe`] cell carries for one in-flight value: the value, the events
/// its commands enqueued (the downstream wait-list), and an OPTIONAL **home** —
/// the [`Cell`] the value must be returned to on [`Checkout`] drop.
///
/// ## The home channel (reusable-graph provenance)
///
/// The home is how an output knows which lent concrete cell to re-arm, with NO
/// type-matching heuristic. It travels WITH the value through the graph:
/// [`resolve`](Input::resolve) lends a `Concrete` cell and yields it as the home;
/// an **in-place** op (fill/scale/copy-dst/the kernel writing its buffer arg)
/// passes that home THROUGH to its output pipe; a **mint** op (upload/alloc/value)
/// or a **transform/consume** op (download's host Vec, uninit→init, host-view) sets
/// home `None`. At the terminal, each output pipe yields `(value, home)` and the
/// [`Checkout`] is built with that exact home.
///
/// Most producers mint fresh values, so home defaults to `None`: [`Pipe::put`]
/// stores `None`, and the home-carrying [`Pipe::put_home`] /
/// [`Pipe::take_home`] are used only on the in-place paths and at the terminal.
struct PipePayload<T> {
    value: T,
    deps: Deps,
    home: Option<BoxedHome<T>>,
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
    /// Deposit the value and the events its commands produced, with **no home**
    /// (the common case: a producer that mints a fresh value or transforms/consumes
    /// — nothing to return on `Checkout` drop). In-place ops that must return a lent
    /// buffer use [`put_home`](Self::put_home) instead.
    pub fn put(&self, v: T, deps: Deps) {
        self.put_home(v, deps, None);
    }
    /// Deposit value + events + the **home** cell the value should be returned to
    /// on `Checkout` drop (re-arming the graph). Used by in-place ops, which pass
    /// their input's home THROUGH to the output, and by the terminal builders.
    pub fn put_home(&self, v: T, deps: Deps, home: Option<BoxedHome<T>>) {
        *self.cell.lock().unwrap() = Some(PipePayload {
            value: v,
            deps,
            home,
        });
    }
    /// Move out the value + its events (the downstream wait-list), **dropping the
    /// home**. For callers that don't propagate provenance (most mid-graph gathers
    /// and the structural/record walks). Use [`take_home`](Self::take_home) on the
    /// in-place + terminal paths.
    pub fn take(&self) -> Option<(T, Deps)> {
        self.take_home().map(|(v, deps, _home)| (v, deps))
    }
    /// Move out value + events + **home** — the provenance-preserving drain. An
    /// in-place op uses it (via [`resolve_home`](Input::resolve_home)) to thread
    /// the home; the terminal uses it to build the `Checkout` with the right home.
    pub fn take_home(&self) -> Option<(T, Deps, Option<BoxedHome<T>>)> {
        self.cell
            .lock()
            .unwrap()
            .take()
            .map(|p| (p.value, p.deps, p.home))
    }

    /// Stable identity of this pipe's storage cell — the graph-edge key. Two
    /// clones of the same pipe share it; independently-built pipes differ. Used
    /// by the record walk to thread a producer's output handle to the consumer
    /// that holds a clone of the same pipe (the recording twin of how `execute`
    /// moves the value through the cell).
    pub fn cell_id(&self) -> usize {
        std::sync::Arc::as_ptr(&self.cell) as *const () as usize
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

    fn execute(&self, _ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
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
///
/// ## Reusable-graph model: the `Concrete` arm is a **cell**
///
/// For the op-tree to be a *reusable* graph (`g.sync()` callable repeatedly), an
/// op's [`execute`](DeviceOp::execute) takes `&self` and must NOT move its inputs
/// out — a second `sync` would find them gone. So `Concrete` holds its value in
/// an interior-mutable cell (`Arc<Mutex<Option<T>>>`, the same shape as a
/// [`Pipe`]). [`resolve`](Input::resolve) **lends** the value out of the cell for
/// the duration of one run; the run's [`Checkout`] returns it to
/// the cell on drop, re-arming `g`. A caller-owned buffer is thus lent (not
/// copied — [`DeviceSlice`] is deliberately not `Clone`) and comes back.
pub enum Input<T> {
    /// Bound at construction (e.g. a caller-owned buffer passed directly). Held
    /// in an interior-mutable cell so `resolve(&self)` can lend it and the run's
    /// `Checkout` can return it (re-arming the graph).
    Concrete(Cell<T>),
    /// Deferred — produced by an upstream op, moved out of the shared cell.
    Pipe(Pipe<T>),
    /// An **unbound typed slot** — a hole built with [`slot!`](crate::slot)`(Tag)`.
    /// The `cell` starts EMPTY; it is filled by a later [`call`](DeviceOpExt::call)`(Tag(value))`
    /// that walks the graph and deposits a matching value (see
    /// [`bind_slots`](DeviceOp::bind_slots)). Once filled, it lends + re-arms
    /// exactly like a [`Concrete`](Input::Concrete) cell (the run's `Checkout`
    /// returns the value to it on drop, so a bound graph is re-runnable). Resolved
    /// while still empty, it is the runtime "slot unbound" error.
    ///
    /// `id` is `TypeId::of::<Tag>()` (matched against a [`SlotBinder`]); `name` is
    /// `type_name::<Tag>()`, carried solely for the unbound-slot diagnostic.
    Slot {
        /// `TypeId::of::<Tag>()` — the bind-matching key.
        id: TypeId,
        /// `type_name::<Tag>()` — for the "slot `<name>` unbound" error only.
        name: &'static str,
        /// Empty until a matching `call(Tag(value))` deposits the value; then it
        /// behaves as a `Concrete` cell (lend + return-on-`Checkout`-drop).
        cell: Cell<T>,
    },
}

impl<T> Input<T> {
    /// Resolve to `(value, upstream Deps)` for ONE run, **lending** (not
    /// consuming) — `&self`, so the op survives for a re-run. A concrete value is
    /// an **entry leaf** — a chain head with no upstream — so it normally carries
    /// no events. The ONE exception is the host-seam start gate: when `ec` has a
    /// `start` event set
    /// (only for chains that [`contains_host_seam`](DeviceOp::contains_host_seam)),
    /// the entry leaf threads that one event into its wait-list so its enqueue
    /// waits on the gate — the whole graph is committed before any of it runs.
    /// (`scratch/start_threaded.c`: `mk()` merges `start` into every entry
    /// command's wait-list.) A pipe is NOT an entry leaf — its producer is
    /// already transitively gated — so the [`Pipe`](Input::Pipe) arm is
    /// unchanged.
    ///
    /// The lent value is taken OUT of the `Concrete` cell here; the cell stays
    /// empty for the rest of the run. It is returned by the run's `Checkout` on
    /// drop (see [`Checkout`]).
    ///
    /// `T: Send + 'static` so the lent cell can be recorded type-erased in the
    /// run's ledger for return — every buffer type that flows here satisfies it.
    pub fn resolve(&self, ec: &ExecutionContext<'_>) -> Result<(T, Deps)>
    where
        T: Send + 'static,
    {
        let (v, deps, _home) = self.resolve_home(ec)?;
        Ok((v, deps))
    }

    /// Lend the value out of a [`Cell`] (the shared lending body of the
    /// `Concrete` and `Slot` arms): take it (the cell stays empty for the run,
    /// re-armed on `Checkout` drop), build its identity home (`Cell<T>: Rehome<T>`),
    /// and thread the host-seam start gate if `ec` has one. `empty_err` is the
    /// error to return when the cell is already empty (busy concrete vs. unbound
    /// slot have distinct messages).
    fn lend_from_cell(
        cell: &Cell<T>,
        ec: &ExecutionContext<'_>,
        empty_err: Error,
    ) -> Result<(T, Deps, Option<BoxedHome<T>>)>
    where
        T: Send + 'static,
    {
        let v = cell.lock().unwrap().take().ok_or(empty_err)?;
        // The home is this very cell: the lent buffer (possibly transformed
        // in place) is returned here on `Checkout` drop, re-arming `g`. An
        // in-place op's output type equals the cell type → identity rehome
        // (`Cell<T>: Rehome<T>`).
        let home: Option<BoxedHome<T>> = Some(Box::new(Arc::clone(cell)));
        match ec.start_dep() {
            // Retain the start event independently: `ec` owns the original
            // handle for the whole enqueue; this `Dep`'s `Event::drop`
            // will `clReleaseEvent` its own ref, balanced by a retain.
            Some(raw) => {
                // SAFETY: `raw` is a live cl_event owned by the terminal
                // for the duration of the enqueue; retaining bumps its
                // refcount, balanced by the wrapped `Event`'s Drop.
                unsafe { opencl3::event::retain_event(raw) }
                    .map_err(|code| Error::OpenCl(opencl3::error_codes::ClError(code)))?;
                Ok((v, vec![wrap_event(crate::Event::new(raw))], home))
            }
            None => Ok((v, Deps::new(), home)),
        }
    }

    /// Like [`resolve`](Self::resolve), but also yields the value's **home** — the
    /// [`Cell`] it must be returned to on the run's [`Checkout`] drop (re-arming
    /// the graph). This is the provenance-preserving form, used by **in-place** ops
    /// (fill/scale/copy-dst/the kernel writing its buffer arg) which pass the home
    /// THROUGH to their output pipe via [`Pipe::put_home`].
    ///
    /// - A [`Concrete`](Input::Concrete) cell: the home IS that cell (the lent
    ///   buffer must come back to it).
    /// - A [`Pipe`](Input::Pipe): the home is whatever the upstream deposited
    ///   (propagated from a concrete head through every in-place stage; `None` if
    ///   the upstream minted/transformed the value).
    pub fn resolve_home(&self, ec: &ExecutionContext<'_>) -> Result<(T, Deps, Option<BoxedHome<T>>)>
    where
        T: Send + 'static,
    {
        match self {
            // A concrete cell: lend the value (it is returned on `Checkout` drop).
            Input::Concrete(cell) => Self::lend_from_cell(
                cell,
                ec,
                Error::NotSupported(
                    "eager graph: a concrete input was already lent and not \
                     returned — a graph is `sync`'d while a previous `Checkout` is \
                     still alive (the graph is busy)",
                ),
            ),
            // A typed slot: once bound (cell full) it lends EXACTLY like a concrete
            // cell — the run's `Checkout` returns the value to it on drop, so a
            // bound graph re-runs. An EMPTY cell is the runtime "slot unbound"
            // error (completeness check), carrying the tag's type name. (A slot
            // that WAS bound but is currently lent out — a still-alive `Checkout` —
            // also reads empty here; the error covers "unbound or graph busy".)
            Input::Slot { name, cell, .. } => {
                Self::lend_from_cell(cell, ec, Error::SlotUnbound(name))
            }
            Input::Pipe(p) => p.take_home().ok_or(Error::NotSupported(
                "eager graph: upstream pipe was not filled before downstream ran \
                 — internal ordering bug",
            )),
        }
    }

    /// Resolve to `(value, upstream Deps)` against a bare [`Launcher`](crate::Launcher)
    /// (building a transient [`ExecutionContext`] internally), for callers OUTSIDE this crate
    /// that have a launcher but cannot construct an `ExecutionContext` (which is
    /// crate-private). Used by the `#[kernel]` proc-macro's **image (consuming)**
    /// terminal: an image kernel is single-shot and not a [`DeviceOp`], so its
    /// buffer args resolve here directly rather than through `execute(&self)`.
    ///
    /// Like [`resolve`](Self::resolve), this **lends** the value out of the cell;
    /// the image terminal consumes the Op and hands the value back by value, so
    /// nothing is returned to the cell — that's expected (single-shot).
    pub fn resolve_on<L>(&self, launcher: &L) -> Result<(T, Deps)>
    where
        T: Send + 'static,
        L: crate::Launcher + ?Sized,
    {
        let device = launcher.context().device().clone();
        let ec = ExecutionContext::new(launcher.context(), device, launcher.cl_queue());
        self.resolve(&ec)
    }

    /// If this input has a lending cell (a `Concrete` head, or a bound `Slot`),
    /// return it so a run's `Checkout` can deposit the (possibly transformed-in-place)
    /// value back into it on drop, re-arming the graph. A `Pipe` input has no home
    /// cell (its producer re-mints the value each run) — returns `None`.
    pub fn return_cell(&self) -> Option<Cell<T>> {
        match self {
            Input::Concrete(cell) | Input::Slot { cell, .. } => Some(Arc::clone(cell)),
            Input::Pipe(_) => None,
        }
    }

    /// Borrow the concrete value via a clone of its cell, or `None` if this is a
    /// pipe. Used by the concrete-head no-launcher terminals (`wait`/`submit`) to
    /// recover the owning context from the buffer before running — a pipe-fed op
    /// has no concrete buffer, so those terminals error clearly. A bound `Slot`
    /// also has a cell (its value flows the same way).
    ///
    /// Returns a [`Cell`] handle (not a borrow) because the value lives behind a
    /// `Mutex`; callers `.lock()` it to read the buffer. `None` ⇒ pipe-fed.
    pub fn concrete_cell(&self) -> Option<Cell<T>> {
        match self {
            Input::Concrete(cell) | Input::Slot { cell, .. } => Some(Arc::clone(cell)),
            Input::Pipe(_) => None,
        }
    }

    /// Read a `Concrete`/bound-`Slot` input's value by reference (the value is
    /// parked in its cell — locked, not lent), mapping it via `f`. `None` if this
    /// is a pipe input or the value is currently lent / the slot is unbound (cell
    /// empty). Used by the record walk and the concrete-head context-recovery
    /// helpers, which only need to inspect the buffer's handle/byte-len.
    pub fn with_concrete<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        match self {
            Input::Concrete(cell) | Input::Slot { cell, .. } => {
                cell.lock().unwrap().as_ref().map(f)
            }
            Input::Pipe(_) => None,
        }
    }

    /// The upstream pipe's [`cell_id`](Pipe::cell_id) if this input is pipe-fed,
    /// else `None` (a concrete entry-leaf buffer or a slot).
    pub fn pipe_cell_id(&self) -> Option<usize> {
        match self {
            Input::Concrete(_) | Input::Slot { .. } => None,
            Input::Pipe(p) => Some(p.cell_id()),
        }
    }

    /// Try to deposit a [`SlotBinder`]'s value into this input, IFF it is an unbound
    /// (or rebindable) [`Slot`](Input::Slot) whose `id` matches the binder's tag.
    ///
    /// Used by [`bind_slots`](DeviceOp::bind_slots) as the graph is walked by
    /// [`call`](DeviceOpExt::call). On a match the binder's boxed value is downcast
    /// back to `T` and **moved** into the slot's cell (overwriting any prior bind —
    /// a second `call(Tag(other))` rebinds), then the binder is marked consumed so a
    /// later same-tag slot in the same walk is left alone (a single buffer is
    /// single-owner). Non-matching arms / tags are a no-op.
    pub fn try_bind_slot(&self, binder: &mut SlotBinder)
    where
        T: Send + 'static,
    {
        let Input::Slot { id, cell, .. } = self else {
            return;
        };
        if *id != binder.id {
            return;
        }
        let Some(boxed) = binder.value.take() else {
            return; // already consumed by an earlier matching slot in this walk
        };
        match boxed.downcast::<T>() {
            Ok(v) => {
                // Bind / rebind: MOVE the value into the cell. A prior value (an
                // earlier `call`'s buffer) is dropped here.
                *cell.lock().unwrap() = Some(*v);
            }
            // Type mismatch is impossible: a tag's `TypeId` (the matched `id`) pins
            // its `Value` type at compile time, and the slot's `T == Tag::Value`. If
            // it ever fired, put the value back so a correctly-typed slot can still
            // see it rather than silently swallowing the bind.
            Err(boxed) => binder.value = Some(boxed),
        }
    }
}

impl<T> From<T> for Input<T> {
    fn from(v: T) -> Self {
        Input::Concrete(Arc::new(Mutex::new(Some(v))))
    }
}

impl<T> From<Pipe<T>> for Input<T> {
    fn from(p: Pipe<T>) -> Self {
        Input::Pipe(p)
    }
}

// ── SlotHandle: the value a `slot!(Tag)` produces ──────────────────────────

/// The build-time handle produced by [`slot!`](crate::slot)`(Tag)` — an UNBOUND
/// typed hole that plugs into the same positions a concrete buffer does (kernel
/// args, `download`/`fill`/`write`/copy sources, …). It carries the tag's
/// [`TypeId`] + `type_name` and a fresh empty [`Cell`]; converting it (via
/// [`From`] / [`ToInput`]) yields an [`Input::Slot`] sharing that cell, which a
/// later [`call`](DeviceOpExt::call)`(Tag(value))` fills.
///
/// `PhantomData<fn() -> Tg>` keeps the handle `Send`/`Sync` regardless of `Tg`
/// (the tag type is a pure marker — never stored).
pub struct SlotHandle<Tg: Tag> {
    id: TypeId,
    name: &'static str,
    cell: Cell<Tg::Value>,
    _tag: PhantomData<fn() -> Tg>,
}

impl<Tg: Tag> SlotHandle<Tg> {
    /// Mint a fresh unbound slot handle for tag `Tg`. Prefer the
    /// [`slot!`](crate::slot) macro spelling (`slot!(Buf)`).
    pub fn new() -> Self {
        SlotHandle {
            id: TypeId::of::<Tg>(),
            name: type_name::<Tg>(),
            cell: Arc::new(Mutex::new(None)),
            _tag: PhantomData,
        }
    }

    /// Consume the handle into its [`Input::Slot`] (shares the empty cell).
    fn into_input(self) -> Input<Tg::Value> {
        Input::Slot {
            id: self.id,
            name: self.name,
            cell: self.cell,
        }
    }
}

impl<Tg: Tag> Default for SlotHandle<Tg> {
    fn default() -> Self {
        Self::new()
    }
}

// NOTE: a `From<SlotHandle<Tg>> for Input<Tg::Value>` impl would let `slot!(Tag)`
// flow into the `impl Into<Input<_>>` positions (`download`/`fill`/`write`/copy),
// but it collides with the blanket `From<T> for Input<T>` (the compiler can't rule
// out `Tg::Value == SlotHandle<Tg>`). So a slot reaches those positions via the
// `into_slot_input` helper / an explicit `Input::from`-free path. The KERNEL-ARG
// position — the primary slot site — works through the [`ToInput`] impl below,
// which is a distinct nominal trait with no such clash.

impl<Tg: Tag> SlotHandle<Tg> {
    /// Convert into the deferred [`Input::Slot`] for an `Into<Input<_>>`-typed
    /// position (e.g. `download(slot.into_slot_input())`). The blanket
    /// `From<T> for Input<T>` blocks a direct `From<SlotHandle>` impl (coherence),
    /// so this named conversion is the explicit bridge for non-kernel positions.
    pub fn into_slot_input(self) -> Input<Tg::Value> {
        self.into_input()
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
                Input::from(self)
            }
        }
    };
}
impl_to_input_concrete!(DeviceSlice);
impl_to_input_concrete!(MappedSlice);
impl_to_input_concrete!(USMSlice);

// A `slot!(Tag)` plugs straight into a kernel arg position: it resolves to
// `Input<Tag::Value>`, with `Buf = Tag::Value` (the macro's `__D`) — so
// `kernels.scale([N], slot!(Buf), 2u32)` infers `__D = Tag::Value` and applies
// the right `KernelSlice*Arg<E>` bound to it, exactly as a concrete buffer would.
// `SlotHandle<Tg>` is a distinct nominal type from the bare families / `Pipe<D>` /
// `Checkout<_>`, so it stays disjoint under coherence.
impl<E, Tg> ToInput<E> for SlotHandle<Tg>
where
    Tg: Tag,
{
    type Buf = Tg::Value;
    fn to_input(self) -> Input<Tg::Value> {
        self.into_input()
    }
}

// ── Transparency: a `Checkout<buffer>` is usable wherever the bare buffer is ──
//
// So a reused-graph output flows straight into the next op WITHOUT an explicit
// `.into_inner()`:
//   let b = x.fill(7).wait()?;          // b: Checkout<DeviceSlice>
//   ks.scale([N], b, 3) …               // fed directly as a kernel arg
// Consuming the `Checkout` extracts the inner buffer (severing the return — the
// same effect as `into_inner`) and feeds it as a concrete `Input`. Distinct
// nominal type from the bare families and `Pipe<D>`, so it stays disjoint under
// coherence.
macro_rules! impl_to_input_checkout {
    ($buf:ident) => {
        impl<E, M> ToInput<E> for Checkout<$crate::$buf<E, M>>
        where
            M: $crate::MemMode,
            E: Send,
        {
            type Buf = $crate::$buf<E, M>;
            fn to_input(self) -> Input<$crate::$buf<E, M>> {
                // Sever the return and feed the inner buffer as a concrete input.
                Input::from(self.into_inner())
            }
        }

        // `From<Checkout<buf>> for Input<buf>` for the `.into()` arg paths. The
        // blanket `From<T> for Input<T>` doesn't cover it (source is the Checkout,
        // not the buffer), and the source type differs, so no coherence clash.
        impl<E, M> From<Checkout<$crate::$buf<E, M>>> for Input<$crate::$buf<E, M>>
        where
            M: $crate::MemMode,
            E: Send,
        {
            fn from(co: Checkout<$crate::$buf<E, M>>) -> Self {
                Input::from(co.into_inner())
            }
        }
    };
}
impl_to_input_checkout!(DeviceSlice);
impl_to_input_checkout!(MappedSlice);
impl_to_input_checkout!(USMSlice);

// `Checkout<Arc<DeviceSlice<E, M>>>` — the shared-buffer arg, severed to its Arc.
impl<E, M> ToInput<E> for Checkout<std::sync::Arc<DeviceSlice<E, M>>>
where
    M: MemMode,
    std::sync::Arc<DeviceSlice<E, M>>: Send,
{
    type Buf = std::sync::Arc<DeviceSlice<E, M>>;
    fn to_input(self) -> Input<std::sync::Arc<DeviceSlice<E, M>>> {
        Input::from(self.into_inner())
    }
}

// `Arc<DeviceSlice<E, M>>` — the shared-buffer kernel arg (read-only fan-out;
// impls `KernelSliceReadArg`). Separate impl since it's a distinct nominal type
// from the bare families above; still disjoint from `Pipe<D>`.
impl<E, M> ToInput<E> for std::sync::Arc<DeviceSlice<E, M>>
where
    M: MemMode,
{
    type Buf = std::sync::Arc<DeviceSlice<E, M>>;
    fn to_input(self) -> Input<std::sync::Arc<DeviceSlice<E, M>>> {
        Input::from(self)
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
///
/// **Inspectable without running.** Because the graph is a closure-free struct
/// (builders ran eagerly at construction; no `FnOnce` is retained), it can be
/// walked structurally before — or instead of — execution:
/// [`description()`](DeviceOpExt::description) returns the node names in
/// execution order without enqueueing a single command. This is the flagship
/// capability cuda-oxide's lazy, closure-composed `DeviceOperation` cannot
/// offer: there the composition lives inside opaque closures, so the only way
/// to learn what a graph does is to run it. claspr's vocabulary is shared
/// heritage; the eager, inspectable model is the divergence.
///
/// **Must be terminated.** Builder verbs run eagerly but enqueue *nothing* on
/// the device until a terminal — [`.sync()`](DeviceOpExt::sync) /
/// [`.wait_on()`](DeviceOpExt::wait_on) / [`.run()`](DeviceOpExt::run) /
/// [`.submit_on()`](DeviceOpExt::submit_on), or the concrete-head `.wait()` —
/// is called. Dropping a built op silently discards all the work it describes,
/// so the trait is `#[must_use]`.
#[must_use = "device ops do nothing until a terminal like .sync()/.wait()/.run() is called"]
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

    /// The **terminal result** — what [`sync`](DeviceOpExt::sync) /
    /// [`wait_on`](DeviceOpExt::wait_on) hand back. Defaults to a single
    /// [`Checkout<Output>`] (the common case). A **multi-output** op overrides it
    /// to the per-element tuple `(Checkout<A>, Checkout<B>, …)` so each output is
    /// independently readable / `into_inner`'d / returned-on-drop — built by
    /// [`gather_checkouts`](Self::gather_checkouts) draining each element pipe with
    /// its own [`home`](Cell). Mirrors how [`Handle`](Self::Handle) exposes the
    /// per-element build-time pipes.
    type Checkouts = Checkout<Self::Output>;

    /// The output value pipe — where `execute` deposits the result; what the
    /// terminal (`sync`) drains. Always a single `Pipe<Output>` regardless of
    /// [`Handle`](Self::Handle).
    fn output_pipe(&self) -> Pipe<Self::Output>;

    /// The downstream-facing [`Handle`](Self::Handle). Default: the output pipe
    /// (so a downstream closure gets `Pipe<Output>`). Combinators override.
    fn handle(&self) -> Self::Handle;

    /// Run the op for ONE run: **lend** inputs (take from their cells / upstream
    /// pipes for the duration of the run), enqueue, deposit the result + its
    /// events into the output pipe. Returns `()` — the value lives in the pipe.
    ///
    /// **Borrows `&self`** — the op is NOT consumed, so the graph it belongs to is
    /// reusable: a terminal can `execute` it again after the previous run's
    /// [`Checkout`] has returned the lent resources to their
    /// cells. Leaves that mint fresh values each run (`upload`/`alloc_zero`/
    /// `value`) re-seed from a retained source; leaves over a caller-owned
    /// [`Concrete`](Input::Concrete) input lend the buffer (returned on `Checkout`
    /// drop). One-shot leaves (host seams) run once and error on a second run.
    ///
    /// `mode` is [`ExecMode::Blocking`] only when this op is the chain terminal
    /// (see [`ExecMode`]); composite ops forward `Pipelined` to their upstream
    /// children and `mode` to the tail. A leaf with no native blocking enqueue
    /// ignores `mode`.
    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()>;

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
    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
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
    ///
    /// Takes `&self` (not `self`): in the reusable-graph model the op is borrowed,
    /// not consumed (the name predates the model — it "produces the output", it no
    /// longer consumes the op).
    #[allow(clippy::wrong_self_convention)]
    fn into_output(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<Self::Output>
    where
        Self: Sized,
    {
        let (value, deps) = self.collect(ec, mode)?;
        for d in &deps {
            d.as_ref().wait().map_err(Error::OpenCl)?;
        }
        Ok(value)
    }

    /// Run this op as the **chain terminal** and build its
    /// [`Checkouts`](Self::Checkouts) — the per-output [`Checkout`] guard(s),
    /// each carrying its own typed return [`home`](Cell) — **without** waiting on
    /// the completion events (the caller waits, per `mode`, on the returned
    /// [`Deps`]).
    ///
    /// This is the home-aware analog of [`collect`](Self::collect): where
    /// `collect` drains values+deps and discards provenance, this drains
    /// value **+ home** ([`Pipe::take_home`]) and wraps each in a `Checkout` so
    /// the run's drop re-arms exactly the right cell.
    ///
    /// Default (single-output ops): `execute`, drain the [`output_pipe`](Self::output_pipe)
    /// with its home, build one `Checkout`. Multi-output ops override this to drain
    /// each element pipe (value+home) into a tuple of `Checkout`s, gathering every
    /// pipe's deps.
    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)>
    where
        Self: Sized,
        Self::Output: Send + 'static,
        Self::Checkouts: FromCheckout<Self::Output>,
    {
        let out = self.output_pipe();
        self.execute(ec, mode)?;
        let (value, deps, home) = out
            .take_home()
            .ok_or(Error::NotSupported("eager graph: op produced no output"))?;
        Ok((
            Self::Checkouts::from_single(Checkout::new(value, home)),
            deps,
        ))
    }

    /// Structural description — node names in execution order, NO execution.
    fn describe(&self, out: &mut Vec<String>);

    /// Fold one [`SlotBinder`] into this op's [`slot!`](crate::slot) cells —
    /// the per-op half of [`call`](DeviceOpExt::call)`(Tag(value))`.
    ///
    /// Walks the op's own [`Input`] fields, calling
    /// [`try_bind_slot`](Input::try_bind_slot) on each so a matching unbound slot
    /// takes the (moved) value; combinators recurse into their children
    /// (mirroring [`describe`](Self::describe)). The default is a **no-op** — most
    /// leaves hold no slot, and `call` simply finds nothing to bind. Ops that
    /// accept buffer args (kernels, `download`/`fill`/`write`/copy, the bundles)
    /// override this to visit their inputs.
    ///
    /// Order-free + curryable falls out of this being one binder per `call`: each
    /// `call` deposits ONE tag's value into the first matching cell, independent of
    /// other tags / call order; completeness is only enforced later at
    /// [`sync`](DeviceOpExt::sync). A short-circuit on
    /// [`is_consumed`](SlotBinder::is_consumed) lets a walk stop early once the
    /// value has landed.
    fn bind_slots(&self, binder: &mut SlotBinder) {
        let _ = binder;
    }

    /// Whether this op (transitively) contains an `and_then_host` /
    /// `and_then_host_with_context` host seam — a node whose worker thread can
    /// complete a downstream-gating user event with a **negative** status on
    /// error. Default `false`; the two host-seam leaves override to `true`, and
    /// every combinator ORs its owned children (mirroring [`describe`](Self::describe)).
    ///
    /// **Why it matters.** The host seam's negative `proceed` can race a
    /// downstream blocking transfer's wait-commit on legacy Intel NEO
    /// (lost-wakeup deadlock). The waiting terminals (`wait_on`/`sync`, `run`)
    /// fix this by gating the WHOLE graph on a `start` user event (carried on the
    /// `ExecutionContext`) released only after the entire graph — including the
    /// terminal marker — is enqueued, so `proceed` negative can never land
    /// in the enqueue/wait-commit window. That start-gate carries a small cost
    /// (one extra user event + a deferred terminal wait), so it is paid **only
    /// when this returns `true`**; pure device graphs keep the zero-overhead
    /// fast path. Validated in `scratch/start_threaded.c` (NEO 40/40, 0 hung).
    ///
    /// NOTE: the former execute-time `and_then_with_context` combinator built its
    /// downstream op at execute (invisible here), leaving any host seam nested in
    /// its closure un-gated — the one documented gap. That combinator is gone:
    /// its sole use (device-by-index routing) is now structural via
    /// [`on_device_at`](DeviceOpExt::on_device_at) /
    /// [`transfer_to_device_at`], so the whole graph is
    /// build-time inspectable and the gap is CLOSED.
    fn contains_host_seam(&self) -> bool {
        false
    }
}

/// Builder verbs for composing [`DeviceOp`]s. Blanket-implemented.
pub trait DeviceOpExt: DeviceOp + Sized {
    /// Sequential composition. **Eager**: runs `f` now with the upstream's
    /// build-time output [`Pipe`], stores the returned op. No closure is kept.
    ///
    /// **False friend with cuda-oxide.** Unlike cuda-oxide's `and_then` (whose
    /// closure runs at *execute* time over the runtime value), claspr's runs the
    /// builder **now, at construction**, over a build-time [`Handle`](DeviceOp::Handle)
    /// — a [`Pipe`] or, for [`value`], the by-value `T` — and retains no closure.
    /// Same name, opposite timing: the eager build is what makes the resulting
    /// graph a closure-free, [`describe`](DeviceOp::describe)-able struct.
    fn and_then<U, F>(self, f: F) -> AndThen<Self, U>
    where
        U: DeviceOp,
        F: FnOnce(Self::Handle) -> U,
    {
        let next = f(self.handle());
        AndThen { source: self, next }
    }

    /// Route this op's `execute` to `device`'s default out-of-order queue
    /// instead of the chain's primary queue. Downstream stages resume on the
    /// parent's queue; the routed op's events are valid across both via
    /// OpenCL's shared-context event semantics. See [`OnDevice`].
    fn on_device(self, device: &crate::Device) -> OnDevice<Self> {
        OnDevice {
            source: self,
            target: DeviceTarget::Concrete(device.clone()),
            out: Pipe::new(),
        }
    }

    /// Like [`on_device`](Self::on_device) but names the target device by
    /// `index` into the running context's device list, resolved at execute.
    /// Lets device-by-index routing be expressed structurally (no execute-time
    /// closure). See [`OnDevice`].
    ///
    /// **Panics** at execute if `index` is out of range for `context().devices()`
    /// (same timing/semantics as resolving `ec.device_at(index)` did).
    fn on_device_at(self, index: usize) -> OnDevice<Self> {
        OnDevice {
            source: self,
            target: DeviceTarget::Index(index),
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
            f: Mutex::new(Some(f)),
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
            f: Mutex::new(Some(f)),
            out: Pipe::new(),
        }
    }

    /// Bind one typed slot of this reusable graph: deposit `tag`'s value into the
    /// graph's matching [`slot!`](crate::slot) cell. Returns `&self` so binds
    /// **chain** and the graph is then `sync`'d:
    /// `g.call(Buf(b)).call(W(w)).sync(&ctx)?`.
    ///
    /// **Order-free + curryable + partial.** Each `call` carries exactly one tag,
    /// folded independently of the others, so `g.call(Buf(b)).call(W(w))` and
    /// `g.call(W(w)).call(Buf(b))` are equivalent, and a subset is allowed —
    /// completeness (every slot bound) is enforced only at
    /// [`sync`](Self::sync)/[`wait_on`](Self::wait_on) (runtime), where an unbound
    /// slot is [`Error::SlotUnbound`]. Binding **MOVES**
    /// the value into the cell; a second `call(Tag(other))` rebinds (the previous
    /// buffer drops). After a run, the run's [`Checkout`] returns the buffer to the
    /// slot cell on drop (same machinery as a concrete head), so a bound graph is
    /// re-runnable.
    ///
    /// The binding is dispatched via [`bind_slots`](DeviceOp::bind_slots), which
    /// walks the op-tree's [`Input`] fields. Ops that route buffer args through
    /// [`Input`] (kernels, `download`/`fill`/`write`/copy, the bundles) propagate
    /// it; a slot placed in an op that does not yet override `bind_slots` simply
    /// stays unbound (caught at `sync`).
    ///
    /// TODO(step b→c): `call` returns `&self`, which serves the `g.call(...).sync()`
    /// path and chained binds. The fully-composable form — a single-output
    /// `g.call()` usable as a kernel arg / `bundle2(b, g.call())` nesting (NOTES
    /// "Closure-free graph model", §3) — is deferred to the segment-plan step; it
    /// needs a node that re-exposes the bound graph's output pipe.
    fn call<Tg: Tag>(&self, tag: Tg) -> &Self {
        // Unwrap the tuple-struct binding `Tag(value)` to its value (the wrapper is
        // a pure move-carrier — only `TypeId::of::<Tg>()` matters for matching),
        // box it into a `Tg`-keyed binder, and fold it into the graph's slot cells.
        let mut binder = SlotBinder::new::<Tg>(tag.into_value());
        self.bind_slots(&mut binder);
        self
    }

    /// Run `self` to completion on `launcher`'s queue and hand back a
    /// [`Checkout`] guard over its output (forward path; no replay). Blocks once,
    /// here, on the terminal op's events — the only wait in the graph.
    ///
    /// **Reusable.** `&self` (not `self`): the graph stays intact. The returned
    /// `Checkout` `Deref`s to the output; on **drop** it returns any LENT concrete
    /// buffers to their cells (re-arming `g`) and releases the run, so `g` can be
    /// `wait_on`/`sync`'d again. While a `Checkout` is live, a second run that
    /// needs a still-lent buffer errors "graph busy". [`Checkout::into_inner`]
    /// permanently extracts the output (severs the return).
    ///
    /// A [`Launcher`](crate::Launcher) is a specific queue (`Context` → its
    /// default OOO queue; a `Queue`/`ExecutionContext` → that exact queue), so
    /// `wait_on` is also the cross-queue / cross-device control (Tier-1 heritage).
    /// [`sync`](Self::sync) is `wait_on` over a `Context`.
    fn wait_on<L: crate::Launcher + ?Sized>(&self, launcher: &L) -> Result<Self::Checkouts>
    where
        Self::Output: Send + 'static,
        Self::Checkouts: FromCheckout<Self::Output>,
    {
        let device = launcher.context().device().clone();
        let mut ec = ExecutionContext::new(launcher.context(), device, launcher.cl_queue());

        // FAST PATH — no host seam: the current zero-overhead terminal. The
        // terminal builds its per-output `Checkout`(s) (each with its own home
        // for return-on-drop) via `gather_checkouts`, then waits on the chain's
        // completion events here. Single-output ops use the default
        // `gather_checkouts`; multi-output ops override it to build a tuple.
        if !self.contains_host_seam() {
            let result = self.gather_checkouts(&ec, ExecMode::Blocking);
            return match result {
                // A failing `and_then_host` worker stashed its rich error and
                // signalled its user event negative; the blocking wait may return
                // the cl_event cascade (`Error::OpenCl(-1)`). Prefer the stash.
                Err(cascade) => Err(ec.take_host_error().unwrap_or(cascade)),
                Ok((checkouts, deps)) => {
                    // Blocking-mode leaves already waited inline, but pipelined
                    // upstream stages (and kernels, which have no native blocking
                    // enqueue) carry events here — wait on them so every command is
                    // complete before the Checkout(s) are observed.
                    let mut wait_err: Option<Error> = None;
                    for d in &deps {
                        if let Err(code) = d.as_ref().wait() {
                            wait_err.get_or_insert(Error::OpenCl(code));
                        }
                    }
                    // Even on a "successful" wait, a worker may have stashed an
                    // error the wait did NOT surface (pocl does not cascade
                    // negative user-event status). A non-empty slot is itself the
                    // failure signal — check it. (Same as the async terminal.)
                    match ec.take_host_error() {
                        Some(rust_err) => Err(rust_err),
                        None => match wait_err {
                            Some(cascade) => Err(cascade),
                            None => Ok(checkouts),
                        },
                    }
                }
            };
        }

        // START-GATE PATH — the chain contains a host seam. Enqueue the WHOLE
        // graph FIRST (gated on a `start` user event so nothing runs yet), then
        // release `start`, then wait. This closes the legacy NEO lost-wakeup
        // window: a host seam's negative `proceed` can never land while a
        // downstream blocking transfer is committing to its wait, because the
        // whole graph is already enqueued before any of it executes. Validated in
        // `scratch/start_threaded.c` (enqueue all → release start → wait last).
        let start = crate::create_user_event(ec.context())?;
        ec.set_start(start.get());

        // Enqueue non-blocking (Pipelined) — a Blocking leaf would wait inline on
        // a command gated on the unreleased `start` and deadlock. The terminal
        // wait happens below, after `start` is released. `gather_checkouts` builds
        // the per-output Checkout(s) with their homes (same as the fast path).
        let collected = self.gather_checkouts(&ec, ExecMode::Pipelined);

        // Release the graph regardless of a setup error: any commands already
        // enqueued are gated on `start`; completing it lets them drain (or abort
        // via their own `proceed`) instead of the queue waiting forever. After
        // this, `start` may be dropped (its retained dep refs outlive it).
        let _ = crate::complete_user_event(&start, opencl3::event::CL_COMPLETE);

        let (checkouts, deps) = match collected {
            Ok(pair) => pair,
            Err(setup_err) => {
                // Setup failed before/while enqueueing. Join any workers that did
                // spawn so their CL calls finish before we return, then surface
                // the rich error if the seam stashed one.
                ec.join_workers();
                return Err(ec.take_host_error().unwrap_or(setup_err));
            }
        };

        // Wait on the chain's completion events (NOT clFinish — clFinish on a
        // terminated command is the pocl hang). A negative `proceed` from a
        // failing seam surfaces here as a cl_event cascade; we reconcile it with
        // the stashed rich error below.
        let mut wait_err: Option<Error> = None;
        for d in &deps {
            if let Err(code) = d.as_ref().wait() {
                wait_err.get_or_insert(Error::OpenCl(code));
            }
        }
        drop(deps);

        // Join host-seam workers AFTER the device wait, so no worker's late CL
        // calls (signalling its user events, then dropping its retained queue)
        // race the caller dropping the Context.
        ec.join_workers();

        // The host-error slot is the authoritative caller-facing error (a worker
        // may have failed without the cl_event cascade reaching us — pocl). Prefer
        // it over the cascade, mirroring the fast path.
        match ec.take_host_error() {
            Some(rust_err) => Err(rust_err),
            None => match wait_err {
                Some(cascade) => Err(cascade),
                None => Ok(checkouts),
            },
        }
    }

    /// Run `self` to completion on `context`'s default out-of-order queue and
    /// hand back its [`Checkouts`](DeviceOp::Checkouts) — a [`Checkout`] over the
    /// output (or a tuple of `Checkout`s for a multi-output graph). The named
    /// graph terminal (cuda-oxide / Rust-CUDA heritage spelling); equal to
    /// [`wait_on`](Self::wait_on) over the `context`. Use `wait_on` with an
    /// explicit `Queue` for cross-queue ordering.
    ///
    /// Reusable: `&self`. See [`wait_on`](Self::wait_on) and [`Checkout`].
    fn sync(&self, context: &Context) -> Result<Self::Checkouts>
    where
        Self::Output: Send + 'static,
        Self::Checkouts: FromCheckout<Self::Output>,
    {
        let device = context.device().clone();
        let queue = context.default_outoforder_queue(&device)?;
        self.wait_on(&*queue)
    }

    /// Non-blocking terminal — enqueue the whole graph on `launcher`'s queue and
    /// return a single completion [`Event`](crate::Event) (a marker over every
    /// command the graph produced) WITHOUT waiting. The caller can `.wait()` the
    /// event or thread it into other Tier-1 ordering. The graph's `Output` value
    /// is materialised here (handles/Vecs); the event just gates *when* the
    /// device work is done. Tier-1 heritage spelling; `submit` is the
    /// concrete-head no-launcher form (see the buffer ops).
    ///
    /// ≈ cuda-oxide's `unsafe async_on`, but safe here and event-returning
    /// (you get a completion [`Event`](crate::Event) to chain, not a raw stream).
    fn submit_on<L: crate::Launcher + ?Sized>(self, launcher: &L) -> Result<crate::Event> {
        use crate::Launcher;
        let device = launcher.context().device().clone();
        let ec = ExecutionContext::new(launcher.context(), device, launcher.cl_queue());
        // Gather non-blocking (Pipelined); a host-seam setup error surfaces here.
        let (_output, deps) = self.collect(&ec, ExecMode::Pipelined)?;
        let wait_list: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SAFETY: the cl_events are held alive by `deps` across this call.
        let marker = unsafe { ec.cl_queue().enqueue_marker_with_wait_list(&wait_list) }
            .map_err(Error::OpenCl)?;
        Ok(marker)
    }

    /// Non-blocking terminal returning BOTH the output value AND a single
    /// completion event — the Tier-1 `(Output, Event)` contract used by the
    /// kernel-op `submit`/`submit_on`. Runs `collect(Pipelined)` on `launcher`'s
    /// queue, then reduces the op's deps to one chainable marker event. (The
    /// plain [`submit_on`](Self::submit_on) drops the value; this keeps it so the
    /// caller can keep using the buffers and chain via `.after(event)`.)
    fn submit_value_on<L: crate::Launcher + ?Sized>(
        self,
        launcher: &L,
    ) -> Result<(Self::Output, crate::Event)> {
        let device = launcher.context().device().clone();
        let ec = ExecutionContext::new(launcher.context(), device, launcher.cl_queue());
        let (output, deps) = self.collect(&ec, ExecMode::Pipelined)?;
        let event = deps_into_single_event(&ec, deps)?;
        Ok((output, event))
    }

    /// Async terminal — run `self` on `context` and return a future that
    /// resolves to its [`Output`](DeviceOp::Output) once every command the
    /// chain enqueued has completed on the device.
    ///
    /// The non-blocking analog of [`sync`](Self::sync): instead of gathering
    /// and *blocking* on the chain's [`Deps`], `run` gathers via
    /// [`collect`](DeviceOp::collect) in [`ExecMode::Pipelined`], then enqueues
    /// an `clEnqueueMarkerWithWaitList` over the chain's deps on the same OOO
    /// queue and wraps it in an [`EventFuture`](crate::EventFuture) — the
    /// Tier-1 `clSetEventCallback` + `AtomicWaker` machinery wakes the
    /// future when the marker fires. (When the chain contains a host seam, the
    /// marker is enqueued *before* the `start` gate is released, so it is gated
    /// like the rest of the graph.)
    ///
    /// **Host errors surface at poll time, via a worker thread.** The eager
    /// host seam (`run_host_seam`) does *not* run its closure inside `execute`:
    /// it spawns a worker thread that waits the upstream/map events, runs the
    /// closure against the borrowed view, stashes any failure into the chain's
    /// `Arc<Mutex<Option<Error>>>` host-error slot, then signals its `proceed`
    /// user event with a negative status to abort downstream device work (the
    /// unmap is gated by a separate, always-`CL_COMPLETE` `fire` event, never by
    /// a negative status). That status cascades into the trailing marker, so the
    /// future's marker poll resolves `Err`; the `Running` variant's `poll` then
    /// prefers the stashed rich error over the `Error::OpenCl(-1)` cascade. The
    /// slot is also drained on a *successful* marker, because some drivers (pocl)
    /// don't propagate a user-event's negative status through
    /// `clEnqueueMarkerWithWaitList` — a non-empty slot is itself the failure
    /// signal. Only an error *submitting* the chain (before any worker spawns)
    /// returns [`DeviceChainFuture::Errored`] eagerly here.
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

    /// Wrap this op's output in [`Arc`] for shared fan-out — the cuda-oxide-style
    /// method spelling of the free fn [`arced(self)`](arced). Equivalent in every
    /// way; pick whichever reads better at the call site
    /// (`upload(v).arc()` vs `arced(upload(v))`). Feeds [`arc_split`].
    fn arc(self) -> Arced<Self>
    where
        Self::Output: Sync,
    {
        arced(self)
    }
}
impl<T: DeviceOp> DeviceOpExt for T {}

// ── Checkout: runtime guard over one run's output ───────────────────────

/// A runtime guard over the output of one reusable-graph run, returned by
/// [`sync`](DeviceOpExt::sync) / [`wait_on`](DeviceOpExt::wait_on).
///
/// ## Lend-and-return
///
/// A graph (`g`) is reusable: `g.sync(&ctx)` enqueues its commands, waits, and
/// returns a `Checkout` holding the output. While the `Checkout` is alive you can
/// read (and mutate) the output via [`Deref`](std::ops::Deref) /
/// [`DerefMut`](std::ops::DerefMut). Any caller-owned
/// buffer the run **lent** (a [`Concrete`](Input::Concrete) input) is held by the
/// `Checkout` for return: on **drop** it deposits the output back into the lending
/// cell, **re-arming** `g` for another `sync`. A second `sync` that needs a
/// still-lent buffer (its cell empty) errors "graph busy" — so a `Checkout` is
/// also the no-parallel-use guard.
///
/// Pure mint-and-consume graphs (`upload…download`) lend nothing, so the
/// `Checkout` drop is a no-op for them and they are reusable purely by re-seeding
/// (the `upload` op re-creates its buffer each run).
///
/// [`into_inner`](Checkout::into_inner) **permanently** extracts the output and
/// severs the return (the lent cell stays empty / the graph re-allocates next
/// run).
///
/// ## Per-output, typed home (no heuristic, no `Any`)
///
/// A `Checkout` carries ONE typed [`home`](Cell) — the exact cell this output
/// must be returned to, learned via the [`Pipe`]'s home channel (not
/// a type-match heuristic, not a type-erased ledger). A multi-output graph yields
/// a **tuple of** `Checkout`s (`(Checkout<A>, Checkout<B>, …)`), each with its own
/// home — so same-typed multi-buffer ops (`add(a, b, out)`) re-arm every cell
/// correctly, which the old single-guard / type-match design could not.
#[must_use = "a Checkout holds the graph's output; reading it requires keeping it alive"]
pub struct Checkout<O> {
    // `Option` so `into_inner`/drop can move the output out.
    value: Option<O>,
    // How this output is returned to its origin cell on drop (re-arming `g`).
    // `None` for a minted/transformed/consumed output (nothing to return).
    // Learned from the output pipe's home channel — no matching, no `Any`. A
    // type-erased rehome (not a bare `Cell<O>`) so the origin cell may hold a
    // weaker type than `O` (the copy's Uninit→Init downgrade).
    home: Option<BoxedHome<O>>,
}

impl<O: Send> Checkout<O> {
    /// Build a checkout over `value`, carrying its typed return `home` (the cell
    /// to re-arm on drop), or `None` if nothing should be returned.
    ///
    /// `pub` (not `pub(crate)`): the `#[kernel]` proc-macro emits
    /// `::claspr::Checkout::new(...)` inside the *user's* crate for multi-output
    /// kernels' `gather_checkouts`, so this must be reachable cross-crate. Not
    /// part of the stable surface — prefer the terminals (`sync`/`wait_on`).
    #[doc(hidden)]
    pub fn new(value: O, home: Option<BoxedHome<O>>) -> Self {
        Checkout {
            value: Some(value),
            home,
        }
    }

    /// Permanently extract the output, **severing** the lend-return: the home
    /// cell stays empty, so the graph re-allocates (or errors "busy") for that
    /// input next run. Use when you want to keep the buffer rather than hand it
    /// back to `g`.
    pub fn into_inner(mut self) -> O {
        // Drop the home WITHOUT returning (sever); take the value out.
        self.home = None;
        self.value
            .take()
            .expect("Checkout::into_inner after value already taken — internal bug")
    }
}

impl<O> Drop for Checkout<O> {
    fn drop(&mut self) {
        // Return the output to its home cell, re-arming `g`. The home is the
        // EXACT cell the value flowed from (concrete head → through every in-place
        // stage), carried by the pipe — no type-matching, no ambiguity. A
        // minted/transformed/consumed output has `home == None`: nothing to return.
        let Some(value) = self.value.take() else {
            return; // already taken by into_inner
        };
        if let Some(home) = self.home.take() {
            home.rehome(value);
        }
        // else: `value` drops here — nothing re-armed.
    }
}

// ── Transparency: consuming verbs forwarded from `Checkout<DeviceSlice>` ──
//
// `Deref`/`DerefMut` already give the `&self`/`&mut self` methods (`.iter()`,
// indexing, `.len()`). The CONSUMING buffer verbs (`read(self)`, `copy_to(self)`,
// `map(&self)`) take the buffer by value, which Deref-coercion can't reach — so
// forward them here: each severs the return (`into_inner`) and calls through, so
// `checkout.read(&mut v).wait()?` / `checkout.copy_to(dst)` compile WITHOUT an
// explicit `.into_inner()`. (`map` only needs `&self`, but the underlying buffer
// must outlive the map builder, so it too consumes the Checkout.)
impl<T, M> Checkout<DeviceSlice<T, M>>
where
    M: MemMode,
{
    /// Read into a caller slice (severs the return, then `DeviceSlice::read`).
    pub fn read<'d>(self, dst: &'d mut [T]) -> ReadInto<'d, T, M>
    where
        T: Send + 'static,
        M: crate::HostReadable + Send + 'static,
    {
        self.into_inner().read(dst)
    }

    /// Device-to-device copy (severs the return, then `DeviceSlice::copy_to`).
    pub fn copy_to<M2>(
        self,
        dst: DeviceSlice<T, M2>,
    ) -> CopyTo2<DeviceSlice<T, M>, DeviceSlice<T, M2>>
    where
        T: Send + 'static,
        M: Send + 'static,
        M2: MemMode + Send + 'static,
    {
        self.into_inner().copy_to(dst)
    }
}

/// Bridge for the default [`gather_checkouts`](DeviceOp::gather_checkouts): lets a
/// single-output op build its `Self::Checkouts` (which defaults to
/// `Checkout<Output>`) from one [`Checkout`] without the trait method statically
/// knowing `Self::Checkouts == Checkout<Output>`. Implemented ONLY for
/// `Checkout<O>`; multi-output ops override `gather_checkouts` and never use it.
pub trait FromCheckout<O> {
    /// Wrap a single output's checkout as the terminal result.
    fn from_single(co: Checkout<O>) -> Self;
}

impl<O> FromCheckout<O> for Checkout<O> {
    fn from_single(co: Checkout<O>) -> Self {
        co
    }
}

// Multi-output terminal results — a tuple / array of `Checkout`s — also satisfy
// `FromCheckout<Output>` so the `sync`/`wait_on` bound holds uniformly. They
// NEVER reach `from_single`: a multi-output op overrides
// [`gather_checkouts`](DeviceOp::gather_checkouts) and builds the tuple directly.
// The body is therefore unreachable (a defensive panic, not a real code path).
macro_rules! impl_from_checkout_tuple {
    ( $( $ty:ident ),+ ) => {
        impl<$($ty),+> FromCheckout<( $($ty,)+ )> for ( $(Checkout<$ty>,)+ ) {
            fn from_single(_co: Checkout<( $($ty,)+ )>) -> Self {
                unreachable!(
                    "multi-output graphs build their Checkouts tuple in \
                     gather_checkouts; from_single is never called"
                )
            }
        }
    };
}
impl_from_checkout_tuple!(A, B);
impl_from_checkout_tuple!(A, B, C);
impl_from_checkout_tuple!(A, B, C, D);
impl_from_checkout_tuple!(A, B, C, D, E);
impl_from_checkout_tuple!(A, B, C, D, E, F);
impl_from_checkout_tuple!(A, B, C, D, E, F, G);
impl_from_checkout_tuple!(A, B, C, D, E, F, G, H);
impl_from_checkout_tuple!(A, B, C, D, E, F, G, H, I);
impl_from_checkout_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_from_checkout_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_from_checkout_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_from_checkout_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_from_checkout_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_from_checkout_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_from_checkout_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

// Same for the homogeneous `[Checkout<O>; N]` shape produced by `arc_split`.
impl<O, const N: usize> FromCheckout<[O; N]> for [Checkout<O>; N] {
    fn from_single(_co: Checkout<[O; N]>) -> Self {
        unreachable!(
            "arc_split builds its [Checkout; N] in gather_checkouts; \
             from_single is never called"
        )
    }
}

impl<O> std::ops::Deref for Checkout<O> {
    type Target = O;
    fn deref(&self) -> &O {
        self.value
            .as_ref()
            .expect("Checkout dereferenced after into_inner — internal bug")
    }
}

impl<O> std::ops::DerefMut for Checkout<O> {
    fn deref_mut(&mut self) -> &mut O {
        self.value
            .as_mut()
            .expect("Checkout dereferenced after into_inner — internal bug")
    }
}

// Pass-through `Debug`/`PartialEq` over the held output, so a `Checkout<O>` reads
// like its `O` in `assert_eq!`/`{:?}` (tests + user diagnostics) without an
// explicit deref. Mirrors how `Deref` makes read methods transparent.
impl<O: std::fmt::Debug> std::fmt::Debug for Checkout<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}

impl<O: PartialEq> PartialEq for Checkout<O> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

// Compare a `Checkout<O>` directly against an `O` (the common `assert_eq!(co, v)`).
impl<O: PartialEq> PartialEq<O> for Checkout<O> {
    fn eq(&self, other: &O) -> bool {
        **self == *other
    }
}

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

impl<S, U> crate::record::RecordableOp for AndThen<S, U>
where
    S: crate::record::RecordableOp,
    U: crate::record::RecordableOp,
{
    fn record(&self, ctx: &mut crate::record::RecordContext) -> Result<()> {
        // Record source first (registers its outputs), then next (which resolves
        // its inputs from those edges) — the same source-before-next order as
        // `execute`.
        self.source.record(ctx)?;
        self.next.record(ctx)
    }
}

impl<S, U> DeviceOp for AndThen<S, U>
where
    S: DeviceOp,
    U: DeviceOp,
{
    type Output = U::Output;
    // The chain's downstream handle is the tail op's handle.
    type Handle = U::Handle;
    // The chain's terminal checkout shape is the tail op's: a multi-output tail
    // (bundle*, arc_split, CopyTo pair) yields its tuple/array of `Checkout`s.
    type Checkouts = U::Checkouts;

    fn output_pipe(&self) -> Pipe<U::Output> {
        self.next.output_pipe()
    }

    fn handle(&self) -> U::Handle {
        self.next.handle()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
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

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(U::Output, Deps)>
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

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)>
    where
        Self: Sized,
        Self::Output: Send + 'static,
        Self::Checkouts: FromCheckout<Self::Output>,
    {
        // Mirror `collect`: delegate to the tail so a multi-output `next` builds
        // its per-element `Checkout` tuple via its OWN `gather_checkouts` override
        // (the default single-pipe drain reads `output_pipe`, which a multi-output
        // op never fills → "op produced no output"). Source pipelines; tail takes
        // the terminal `mode`. Same orphaned-source-deps threading as `collect`.
        let src_pipe = self.source.output_pipe();
        self.source.execute(ec, ExecMode::Pipelined)?;
        let (checkouts, mut deps) = self.next.gather_checkouts(ec, mode)?;
        if let Some((_discarded, src_deps)) = src_pipe.take() {
            deps.extend(src_deps);
        }
        Ok((checkouts, deps))
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        self.next.describe(out);
    }

    fn bind_slots(&self, binder: &mut SlotBinder) {
        // Walk source then next (execution order). Stop early once the value has
        // landed in a matching slot — a single `call` binds one cell.
        self.source.bind_slots(binder);
        if binder.is_consumed() {
            return;
        }
        self.next.bind_slots(binder);
    }

    fn contains_host_seam(&self) -> bool {
        // `next` is the downstream op built eagerly inside the `and_then` closure
        // at construction (a real owned field, not a deferred closure), so a host
        // seam built in the closure IS visible here.
        self.source.contains_host_seam() || self.next.contains_host_seam()
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
    // Held by value (not `Option`): `T: Clone`, so each run re-emits a fresh
    // clone and the source is never drained — the graph is reusable + idempotent.
    v: T,
    out: Pipe<T>,
}

/// Lift a `Clone` host value into the graph with a **by-value** handle (so
/// downstream closures get the value, not a pipe — see [`Value`]). For a
/// non-`Clone` owned resource use [`lift`].
///
/// `value` + [`lift`] together ≈ cuda-oxide's `value` (one host value into the
/// graph), split here by whether the handle is by-value (`value`) or by-pipe
/// (`lift`).
pub fn value<T: Send + Clone + 'static>(v: T) -> Value<T> {
    Value {
        v,
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
        // Clone the value out for the downstream closure; `self.v` keeps its own
        // copy for `execute` (which clones again each run — idempotent).
        self.v.clone()
    }

    fn execute(&self, _ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Re-emit a fresh clone every run — never drains the source.
        self.out.put(self.v.clone(), Deps::new());
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
    // One-shot: a non-`Clone` owned resource can't be re-emitted, so a `Lift`
    // chain head runs once; a second `sync` errors clearly. (For a reusable
    // chain head over a caller-owned buffer, pass it as a `Concrete` input to a
    // buffer verb — that lends-and-returns. `Lift` is the move-in-once form.)
    v: Mutex<Option<T>>,
    out: Pipe<T>,
}

/// Lift an owned resource into the graph (default `Pipe` handle — see [`Lift`]).
/// With [`value`], together ≈ cuda-oxide's `value` (the by-pipe half, for
/// non-`Clone` owned resources).
pub fn lift<T: Send + 'static>(v: T) -> Lift<T> {
    Lift {
        v: Mutex::new(Some(v)),
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

    fn execute(&self, _ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let v = self.v.lock().unwrap().take().ok_or(Error::NotSupported(
            "eager graph: a `lift`ed resource was already consumed — `lift` is a \
             move-in-once chain head and can't drive a reused graph; pass a \
             caller-owned buffer as a concrete input instead",
        ))?;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Resolve the upstream value + its events and re-deposit unchanged — no
        // device work; deps threaded through so ordering/termination is intact.
        // In-place identity: the home flows straight through.
        let (v, deps, home) = self.input.resolve_home(ec)?;
        self.out.put_home(v, deps, home);
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
    fn collect_erased(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(T, Deps)>;

    fn describe_erased(&self, out: &mut Vec<String>);

    /// Forwards [`DeviceOp::contains_host_seam`] through the erasure so a
    /// [`DeviceDynOp`] reports the right gate bit for whichever concrete arm it
    /// wraps.
    fn contains_host_seam_erased(&self) -> bool;
}

impl<O> ErasedDeviceOp<O::Output> for O
where
    O: DeviceOp,
{
    fn collect_erased(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(O::Output, Deps)> {
        self.collect(ec, mode)
    }

    fn describe_erased(&self, out: &mut Vec<String>) {
        self.describe(out);
    }

    fn contains_host_seam_erased(&self) -> bool {
        self.contains_host_seam()
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
    inner: Box<dyn ErasedDeviceOp<T> + 'op>,
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
            inner: Box::new(op),
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // Gather the erased inner op (any arity → one value + deps) and deposit
        // into our own pipe, so the default collect/into_output/handle path treats
        // this as an ordinary single-output leaf. The inner op observes `mode`
        // (it is the real terminal work when this DeviceDynOp is the chain tail).
        // `&self`: the inner op runs by reference, so an erased arm is reusable.
        let (v, deps) = self.inner.collect_erased(ec, mode)?;
        self.out.put(v, deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("dyn_op{".into());
        self.inner.describe_erased(out);
        out.push("}".into());
    }

    fn contains_host_seam(&self) -> bool {
        self.inner.contains_host_seam_erased()
    }
}

// ── Arced: wrap the output in Arc<T> ───────────────────────────────────

/// Wrap an upstream op's output in [`Arc`] for shared fan-out. Passes events
/// through unchanged.
pub struct Arced<S: DeviceOp> {
    source: S,
    out: Pipe<Arc<S::Output>>,
}

/// Wrap `source`'s output in `Arc`. ≈ cuda-oxide's `.arc()` (also available here
/// as the [`arc`](DeviceOpExt::arc) method).
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
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

    fn contains_host_seam(&self) -> bool {
        self.source.contains_host_seam()
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
///
/// ≈ a homogeneous N-ary `unzip!` over a shared [`arc`](DeviceOpExt::arc)ed input
/// (one producer, `N` read-only consumers).
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
    // Per-branch Checkouts (homes are all `None` — these are read-only Arc clones).
    type Checkouts = [Checkout<S::Output>; N];

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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Gather the source via `collect` (any arity), then scatter a clone of
        // its value + events into every branch pipe (Arc::clone is a cheap
        // refcount bump; Deps clone shares the same producer events).
        let (v, deps) = self.source.collect(ec, ExecMode::Pipelined)?;
        for out in &self.outs {
            out.put(v.clone(), deps.clone());
        }
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
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

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // Drain each branch pipe with its home → a `[Checkout; N]`. Arc-clone
        // branches carry no home (read-only fan-out), so each Checkout's home is
        // `None`; the per-branch shape is still correct.
        let outs = self.outs.clone();
        self.execute(ec, mode)?;
        let mut all_deps: Deps = Deps::new();
        let mut cos: Vec<Checkout<S::Output>> = Vec::with_capacity(N);
        for p in &outs {
            let (v, d, home) = p.take_home().ok_or(Error::NotSupported(
                "eager arc_split: a branch produced no output",
            ))?;
            cos.push(Checkout::new(v, home));
            all_deps.extend(d);
        }
        let arr = cos
            .try_into()
            .unwrap_or_else(|_| unreachable!("arc_split drained exactly N branch pipes"));
        Ok((arr, all_deps))
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push(format!("arc_split[{N}]"));
    }

    fn contains_host_seam(&self) -> bool {
        self.source.contains_host_seam()
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

/// Reduce a gathered op's [`Deps`] to a single owned [`Event`](crate::Event) — used by the
/// Tier-1 `submit`/`submit_on` terminals (which hand the caller a chainable
/// event). Always enqueues a marker over the deps, so the result is one owned
/// `Event` that completes exactly when the op's work does, regardless of how
/// many events the op produced (one for a single-output kernel, several for a
/// multi-output launch). Empty deps → a bare marker (fires immediately).
pub fn deps_into_single_event(ec: &ExecutionContext<'_>, deps: Deps) -> Result<crate::Event> {
    use crate::Launcher;
    let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
    // SAFETY: the cl_event handles are held alive by `deps` across this call.
    let marker =
        unsafe { ec.cl_queue().enqueue_marker_with_wait_list(&raw) }.map_err(Error::OpenCl)?;
    drop(deps);
    Ok(marker)
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

        #[doc = concat!("Construct an eager [`", stringify!($name),
            "`]. \u{2248} cuda-oxide's `zip!` at this fixed arity.")]
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
            // Per-branch Checkouts: one independent `Checkout` per branch output.
            // (Branch values arrive via each branch's `collect`, which collapses
            // any nested arity to a single value with no home, so a bundle's
            // Checkouts carry `None` homes in step (a) — the per-branch SHAPE is
            // what this fixes; home re-arm through a bundle is deferred, the same
            // boundary as copy/OnDevice.)
            type Checkouts = ( $(Checkout<<$ty as DeviceOp>::Output>,)+ );

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

            fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
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

            fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
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

            fn gather_checkouts(
                &self,
                ec: &ExecutionContext<'_>,
                mode: ExecMode,
            ) -> Result<(Self::Checkouts, Deps)> {
                // Scatter via `execute` (fills each `$pf`), then drain each branch
                // pipe (value + home) into its own `Checkout`, joining the branch
                // wait-lists into one marker.
                $(let $pf = self.$pf.clone();)+
                self.execute(ec, mode)?;
                let mut branch_deps: Vec<Deps> = Vec::new();
                let checkouts = ( $({
                    let (v, d, home) = $pf.take_home().ok_or(Error::NotSupported(
                        "eager bundle: a branch produced no output"))?;
                    branch_deps.push(d);
                    Checkout::new(v, home)
                },)+ );
                let joined = join_marker(ec, &branch_deps)?;
                Ok((checkouts, joined))
            }

            fn describe(&self, out: &mut Vec<String>) {
                out.push(concat!(stringify!($name), "{").into());
                $(self.$field.describe(out);)+
                out.push("}".into());
            }

            fn contains_host_seam(&self) -> bool {
                // OR every branch: a host seam in any one of them must gate.
                false $(|| self.$field.contains_host_seam())+
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

/// Variadic constructor for [`Bundle2`] through [`Bundle16`] — picks the right
/// `bundleN` based on the number of arguments. Each arm runs its branches
/// independently and joins them with a single marker event.
///
/// ≈ cuda-oxide's `zip!` (heterogeneous parallel join into a tuple).
///
/// ```ignore
/// let (a, b) = bundle!(op_a, op_b).sync(&ctx)?;
/// let (a, b, c) = bundle!(op_a, op_b, op_c).sync(&ctx)?;
/// // ... up to 16 children
/// ```
#[macro_export]
macro_rules! bundle {
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
/// Reads as data → operation and composes cleanly with downstream `.and_then`;
/// the free-fn form stays available — use whichever fits the call site.
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // `collect` each branch op (not `execute`) so a multi-output branch runs
        // its own gather and yields one reconstructed value + deps — `self.pipes`
        // (captured single output pipes) are empty for such branches. The pipes
        // field is now unused for gathering; we read values straight from
        // `collect`.
        let n = self.ops.len();
        let mut branch_deps: Vec<Deps> = Vec::with_capacity(n);
        let mut outputs: Vec<U::Output> = Vec::with_capacity(n);
        for op in &self.ops {
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

    fn contains_host_seam(&self) -> bool {
        self.ops.iter().any(|op| op.contains_host_seam())
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

/// Build a zero-init alloc leaf with the **default [`ReadWrite`] marker** — no
/// turbofish: `alloc_zero(N)`. For a non-default marker use [`alloc_zero_as`]
/// with a marker witness (`alloc_zero_as(N, HostReadOnly)`).
pub fn alloc_zero<T>(len: usize) -> AllocZero<T, ReadWrite>
where
    T: Copy + Default + Send + Sync + 'static,
{
    alloc_zero_as(len, ReadWrite)
}

/// Build a zero-init alloc leaf with an **explicit access marker**, inferred
/// from the `marker` witness — no turbofish: `alloc_zero_as(N, HostReadOnly)`.
/// The default-marker shorthand is [`alloc_zero`].
pub fn alloc_zero_as<T, M>(len: usize, marker: M) -> AllocZero<T, M>
where
    T: Copy + Default + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    let _ = marker;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // alloc_zero is synchronous internally; no in-flight event, mode N/A.
        let buf = DeviceSlice::<T, M>::alloc_zero(ec.context(), self.len)?;
        self.out.put(buf, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("alloc_zero(len={})", self.len));
    }
}

// ── Concrete-head terminal helper ──────────────────────────────────────
//
// The buffer-verb ops (`Fill`/`Download`/`ReadInto`/`WriteDevice`/
// `TransferToDevice`) are **concrete-head**: their input is a caller-owned
// `DeviceSlice`, whose `.ctx()` supplies the queue. That lets them offer the
// no-launcher Tier-1 terminals `wait()`/`submit()` — the context is recovered
// from the owned buffer rather than passed in. A pipe-fed op (only reachable
// inside an eager `and_then` closure) has no concrete buffer, so these terminals
// error clearly, steering the caller to `wait_on(&ctx)` / `sync(&ctx)`.

/// Recover the owning [`Context`] from a concrete-head [`Input<DeviceSlice>`],
/// or a clear "pipe-fed" error for the no-launcher concrete-head terminals.
fn concrete_buf_ctx<T, M: MemMode>(buf: &Input<DeviceSlice<T, M>>) -> Result<Context> {
    use crate::Buffer;
    buf.with_concrete(|b| b.ctx().clone())
        .ok_or(Error::NotSupported(
            "concrete-head terminal (wait/submit) on a pipe-fed buffer op — use \
         wait_on(&ctx) / sync(&ctx) for piped (graph) inputs",
        ))
}

/// SVM analog of [`concrete_buf_ctx`]: recover the owning [`Context`] from a
/// concrete-head [`Input<MappedSlice>`], or a clear "pipe-fed" error.
fn concrete_svm_ctx<T, M: MemMode>(buf: &Input<MappedSlice<T, M>>) -> Result<Context> {
    use crate::Buffer;
    buf.with_concrete(|b| b.ctx().clone())
        .ok_or(Error::NotSupported(
            "concrete-head terminal (wait/submit) on a pipe-fed SVM op — use \
         wait_on(&ctx) / sync(&ctx) for piped (graph) inputs",
        ))
}

// ── Leaf: in-place fill (eager port of DeviceSliceFillOp) ──────────────

/// Fill a buffer (upstream pipe or concrete) with `value` via a non-blocking
/// `clEnqueueFillBuffer`, threading the upstream events as the wait-list.
pub struct Fill<T: Copy, M: MemMode> {
    buf: Input<DeviceSlice<T, M>>,
    value: T,
    out: Pipe<DeviceSlice<T, M>>,
}

impl<T, M> crate::record::RecordableOp for Fill<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    fn record(&self, ctx: &mut crate::record::RecordContext) -> Result<()> {
        use crate::record::{BufHandle, MemRef};
        use opencl3::memory::ClMem;
        // Resolve the buffer: this op's concrete input (chain head) or the
        // upstream producer's output (mid-chain, in-place fill).
        let concrete = self.buf.with_concrete(|b| BufHandle {
            mem: MemRef::Buffer(b.buffer().get()),
            byte_len: b.byte_len(),
        });
        let (handle, waits) = ctx.resolve_input(concrete, self.buf.pipe_cell_id())?;
        // Byte pattern of the fill value.
        // SAFETY: `T: Copy`; read its `size_of::<T>()` bytes.
        let pattern = unsafe {
            std::slice::from_raw_parts(
                (&self.value as *const T) as *const u8,
                std::mem::size_of::<T>(),
            )
        }
        .to_vec();
        let sp = ctx.fill_buffer(handle.mem, pattern, 0, handle.byte_len, waits);
        // Fill is in-place: its output is the same buffer, gated on this command.
        ctx.register_output(self.out.cell_id(), handle, vec![sp]);
        Ok(())
    }
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (mut buf, deps, home) = self.buf.resolve_home(ec)?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // Fill has no native CL_BLOCKING flag (it's always enqueue + optional
        // wait — exactly what the old `FillOp::wait_on` did internally), so both
        // modes enqueue non-blocking; Blocking then waits on the event here.
        // In-place: the filled buffer is the lent buffer → home threads through.
        let event = crate::buffer::fill_buffer_enqueue(&mut buf, ec, self.value, &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill".into());
    }
}

impl<T, M> Fill<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    /// Concrete-head blocking terminal: fill on the buffer's own context default
    /// queue and return the (filled) buffer. The no-launcher Tier-1 spelling
    /// (`buf.fill(v).wait()?`); use [`wait_on`](DeviceOpExt::wait_on) for a
    /// specific queue, or `sync`/`wait_on` for a pipe-fed op.
    pub fn wait(self) -> Result<DeviceSlice<T, M>> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal: enqueue the fill on the buffer's own
    /// context default queue and return a completion [`Event`](crate::Event).
    pub fn submit(self) -> Result<crate::Event> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_on(&*queue)
    }
}

// ── Leaf: upload (host → device, alloc + CL_MEM_COPY_HOST_PTR) ──────────

/// Allocate a `DeviceSlice<T, M>` and bake `src` into it at creation
/// (`CL_MEM_COPY_HOST_PTR`). A chain-entry leaf — no upstream input. (Uses the
/// from_slice path: works for any marker, one synchronous create, no in-flight
/// event.)
pub struct Upload<T: Copy, M: MemMode = ReadWrite> {
    // Held by value (not `Option`): the host source is RETAINED so every run
    // re-creates a fresh buffer from it (`CL_MEM_COPY_HOST_PTR`). This is the
    // "mutable buffers re-seed each run" rule — `upload(v)…download` is
    // idempotent by construction (the upload op re-writes its source each run).
    src: UploadSource<T>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an upload leaf from any `Vec<T>` / `Box<[T]>` / `Arc<[T]>`, with the
/// **default [`ReadWrite`] marker** — the overwhelming common case, so no
/// turbofish: `upload(vec![1u32, 2, 3])`. For a non-default marker use
/// [`upload_as`] with a marker witness (`upload_as(src, Frozen)`); both paths
/// go through `from_slice` (`CL_MEM_COPY_HOST_PTR`), the only constructor that
/// can build an immutable `Frozen`/`ReadOnly` buffer.
pub fn upload<T, S>(src: S) -> Upload<T, ReadWrite>
where
    T: Copy + Send + Sync + 'static,
    S: Into<UploadSource<T>>,
{
    upload_as(src, ReadWrite)
}

/// Build an upload leaf with an **explicit access marker**, inferred from the
/// `marker` witness — no turbofish: `upload_as(src, Frozen)` /
/// `upload_as(src, ReadOnly)`. `T`/`S` infer from `src`, `M` from the witness.
/// The default-marker shorthand is [`upload`]. Like `upload`, backed by
/// `from_slice` (`CL_MEM_COPY_HOST_PTR`).
pub fn upload_as<T, M, S>(src: S, marker: M) -> Upload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
    S: Into<UploadSource<T>>,
{
    let _ = marker; // witness only — fixes M, zero-sized, no runtime use.
    Upload {
        src: src.into(),
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // from_slice (CL_MEM_COPY_HOST_PTR) is a synchronous create — no
        // in-flight event, mode N/A. Reads `self.src` BY REFERENCE so the source
        // is retained and the buffer re-seeds on every run (idempotent reuse).
        let buf = DeviceSlice::<T, M>::from_slice(ec.context(), self.src.as_slice())?;
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve(ec)?;
        let mut host = vec![T::default(); buf.len()];
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        match mode {
            // Terminal: native blocking read (CL_BLOCKING) — the driver waits,
            // the host Vec is valid on return, no event. Matches Tier-1
            // `ReadOp::wait_on`; restores parity for `…download().sync()`.
            ExecMode::Blocking => {
                crate::buffer::read_buffer_enqueue(&buf, ec, &mut host, true, &raw)?;
                self.out.put(host, Deps::new());
            }
            // Pipelined: non-blocking; the event gates the Vec being valid.
            ExecMode::Pipelined => {
                let event = crate::buffer::read_buffer_enqueue(&buf, ec, &mut host, false, &raw)?;
                self.out.put(host, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("download".into());
    }
}

impl<T, M> Download<T, M>
where
    T: Clone + Default + Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    /// Concrete-head blocking terminal: read on the buffer's own context default
    /// queue and return the host `Vec<T>`.
    pub fn wait(self) -> Result<Vec<T>> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal returning the (not-yet-valid) host
    /// `Vec<T>` plus a completion [`Event`](crate::Event) — mirrors the Tier-1
    /// `(Output, Event)` submit contract.
    pub fn submit(self) -> Result<(Vec<T>, crate::Event)> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}

// ── Leaf: read a DeviceSlice into a caller-supplied slice (read-into) ───────

/// Read a buffer into a **caller-supplied** `&mut [T]` (rather than allocating a
/// fresh `Vec` like [`Download`]), yielding the buffer back so it can be reused.
/// The eager analog of the old Tier-1 `buf.read(&mut dst)` builder: a
/// concrete-head op (it borrows the destination slice for `'d`, so it never
/// flows through a pipe — a pipe-fed read uses [`Download`]).
///
/// `Output = DeviceSlice<T, M>`: the buffer moves in and rebinds out
/// (`let buf = buf.read(&mut dst).wait()?;`), so a caller can read into the same
/// destination repeatedly.
pub struct ReadInto<'d, T, M: MemMode = ReadWrite> {
    buf: Input<DeviceSlice<T, M>>,
    // Behind a `Mutex` so `execute(&self)` can get the `&mut [T]` it needs to
    // read into. The caller slice is borrowed for `'d`; re-runs read into the
    // same destination (overwriting it).
    dst: Mutex<&'d mut [T]>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build a read-into leaf: read `buf` into the caller slice `dst`. See
/// [`ReadInto`].
pub fn read_into<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
    dst: &mut [T],
) -> ReadInto<'_, T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    ReadInto {
        buf: buf.into(),
        dst: Mutex::new(dst),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for ReadInto<'_, T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Pipe<DeviceSlice<T, M>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (buf, deps, home) = self.buf.resolve_home(ec)?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let mut dst = self.dst.lock().unwrap();
        // In-place: the buffer is read and handed back unchanged → home threads.
        match mode {
            // Terminal: native blocking read — `dst` is valid on return, no event.
            ExecMode::Blocking => {
                crate::buffer::read_buffer_enqueue(&buf, ec, &mut dst, true, &raw)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            // Pipelined: non-blocking; the event gates `dst` being valid.
            ExecMode::Pipelined => {
                let event = crate::buffer::read_buffer_enqueue(&buf, ec, &mut dst, false, &raw)?;
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("read_into".into());
    }
}

impl<T, M> ReadInto<'_, T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    /// Concrete-head blocking terminal: read into the caller slice on the
    /// buffer's own context default queue; return the buffer for reuse.
    pub fn wait(self) -> Result<DeviceSlice<T, M>> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal: enqueue the read on the buffer's own
    /// context default queue and return a completion [`Event`](crate::Event).
    /// (The `dst` slice must outlive the event.)
    pub fn submit(self) -> Result<crate::Event> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_on(&*queue)
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
    target: DeviceTarget,
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
        target: DeviceTarget::Concrete(device.clone()),
        out: Pipe::new(),
    }
}

/// Build a transfer-to-device leaf targeting the device at `index` in the
/// running context's device list, resolved at execute. See [`transfer_to_device`]
/// for migrate semantics.
///
/// **Panics** at execute if `index` is out of range for `context().devices()`
/// (same timing/semantics as resolving `ec.device_at(index)` did).
pub fn transfer_to_device_at<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
    index: usize,
) -> TransferToDevice<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    TransferToDevice {
        buf: buf.into(),
        target: DeviceTarget::Index(index),
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // In-place: the migrated buffer is the same buffer → home threads through.
        let (buf, deps, home) = self.buf.resolve_home(ec)?;
        // Resolve the target device (concrete, or by index into the running
        // context's device list) before resolving its queue.
        let device = match &self.target {
            DeviceTarget::Concrete(d) => d.clone(),
            DeviceTarget::Index(i) => ec.context().devices()[*i].clone(),
        };
        // Resolve the target device's default OOO queue (cached on the Context,
        // so the terminal's flush_all_outoforder_queues pushes it). Same path
        // OnDevice uses to reach a non-primary device's queue.
        let target_q = ec.context().default_outoforder_queue(&device)?;
        // Enqueue the migrate with the upstream events as the wait-list, on the
        // target queue (`&*target_q` is the `Queue: Launcher`). Non-blocking —
        // mode is ignored; the chain terminal's `into_output` does the final
        // wait. The migrate body mirrors the closure layer's
        // `transfer_to_device.rs` exactly.
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let event = crate::buffer::migrate_buffer_enqueue(&buf, &*target_q, &raw)?;
        self.out.put_home(buf, vec![wrap_event(event)], home);
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve(ec)?;
        // SAFETY: the fill below writes every byte; downstream gates on the
        // returned fill event (Pipelined) or the driver waits (Blocking).
        let mut buf = unsafe { uninit.assume_init() };
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // Fill has no native CL_BLOCKING flag — enqueue, then wait on Blocking.
        let event = crate::buffer::fill_buffer_enqueue(&mut buf, ec, self.value, &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve(ec)?;
        // SAFETY: the SVM fill below writes every byte.
        let buf = unsafe { uninit.assume_init() };
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SVM fill is always a non-blocking enqueue; Blocking waits here.
        let event = crate::mapped::svm_fill_enqueue(&buf, ec, self.value, &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Pure host op — no event; forward the upstream deps unchanged.
        let (uninit, deps) = self.uninit.resolve(ec)?;
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
    // Retained by value; read by reference each run (see `WriteDevice::src`).
    src: UploadSource<T>,
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
        src: src.into(),
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve(ec)?;
        // SAFETY: the write below covers every byte; downstream gates on the
        // returned write event (Pipelined) or the driver waits (Blocking).
        let mut buf = unsafe { uninit.assume_init() };
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        match mode {
            ExecMode::Blocking => {
                crate::buffer::write_buffer_enqueue(&mut buf, ec, self.src.as_slice(), true, &raw)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                // `self.src` is valid for the whole `sync` — no keep-alive needed.
                let event = crate::buffer::write_buffer_enqueue(
                    &mut buf,
                    ec,
                    self.src.as_slice(),
                    false,
                    &raw,
                )?;
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
    // Retained by value (not `Option`): the host source is read BY REFERENCE each
    // run (re-seed). `&self` outlives the whole `sync` (the terminal waits before
    // returning), so the source stays valid across the async write window — the
    // former per-run `register_drop_callback` keep-alive is unnecessary now that
    // the op lives in the reusable graph rather than being moved into an executor.
    src: UploadSource<T>,
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
        src: src.into(),
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (mut buf, deps, home) = self.buf.resolve_home(ec)?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // In-place: the written buffer is the lent buffer → home threads through.
        match mode {
            ExecMode::Blocking => {
                crate::buffer::write_buffer_enqueue(&mut buf, ec, self.src.as_slice(), true, &raw)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                // Non-blocking write; `self.src` stays valid for the whole `sync`
                // (the op lives in the graph; the terminal waits before returning),
                // so no per-run keep-alive callback is needed.
                let event = crate::buffer::write_buffer_enqueue(
                    &mut buf,
                    ec,
                    self.src.as_slice(),
                    false,
                    &raw,
                )?;
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write".into());
    }
}

impl<T, M> WriteDevice<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    /// Concrete-head blocking terminal: write on the buffer's own context default
    /// queue and return the buffer for reuse (`let buf = buf.write(d).wait()?;`).
    pub fn wait(self) -> Result<DeviceSlice<T, M>> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal returning the buffer plus a completion
    /// [`Event`](crate::Event) — mirrors the Tier-1 `(Output, Event)` contract so
    /// the caller can keep using the buffer and chain via `.after(event)`.
    pub fn submit(self) -> Result<(DeviceSlice<T, M>, crate::Event)> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}

// ── Leaf: write host data into a MappedSliceUninit → MappedSlice ────────────

/// Eager analog of `WriteFromUninitOp<MappedSliceUninit, _>`. Mirrors
/// [`WriteDeviceUninit`].
pub struct WriteMappedUninit<T, M: MemMode> {
    uninit: Input<MappedSliceUninit<T, M>>,
    // Retained by value; read by reference each run (see `WriteDevice::src`).
    src: UploadSource<T>,
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
        src: src.into(),
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve(ec)?;
        // SAFETY: the SVM write below covers every byte.
        let buf = unsafe { uninit.assume_init() };
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SVM write is always a non-blocking enqueue (no native CL_BLOCKING flag);
        // Blocking waits on the returned event here, Pipelined threads it
        // downstream. `self.src` is valid for the whole `sync` — no keep-alive.
        let event = crate::mapped::svm_write_enqueue(&buf, ec, self.src.as_slice(), &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_mapped_uninit".into());
    }
}

// ── Leaf: in-place SVM fill (eager port of SvmFillOp) ──────────────────────

/// Fill an existing (init) [`MappedSlice`] with `value` via a non-blocking
/// `clEnqueueSVMMemFill` (or kernel fill for kernel-RO markers), threading the
/// upstream events as the wait-list. SVM analog of [`Fill`]. The buffer passes
/// through as the op's output (concrete-head reusable). The fill event is
/// auto-registered on the buffer's last-use list (inside the raw helper) so
/// Drop's `clEnqueueSVMFree` waits for it.
pub struct FillMapped<T: Copy, M: MemMode> {
    buf: Input<MappedSlice<T, M>>,
    value: T,
    out: Pipe<MappedSlice<T, M>>,
}

/// Build an SVM fill leaf over an existing `MappedSlice` (concrete or piped).
pub fn fill_mapped<T, M>(buf: impl Into<Input<MappedSlice<T, M>>>, value: T) -> FillMapped<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    FillMapped {
        buf: buf.into(),
        value,
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for FillMapped<T, M>
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (buf, deps, home) = self.buf.resolve_home(ec)?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SVM fill is always a non-blocking enqueue (no native CL_BLOCKING flag);
        // Blocking waits on the returned event here. In-place → home threads.
        let event = crate::mapped::svm_fill_enqueue(&buf, ec, self.value, &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill_mapped".into());
    }
}

impl<T, M> FillMapped<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    /// Concrete-head blocking terminal: fill on the buffer's own context default
    /// queue and return the (filled) buffer (`let buf = buf.fill(v).wait()?;`).
    pub fn wait(self) -> Result<MappedSlice<T, M>> {
        let ctx = concrete_svm_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal: enqueue the fill on the buffer's own
    /// context default queue and return a completion [`Event`](crate::Event).
    pub fn submit(self) -> Result<crate::Event> {
        let ctx = concrete_svm_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_on(&*queue)
    }
}

// ── Leaf: write host data into an existing (init) MappedSlice ───────────────

/// Write host `src` into an already-initialised [`MappedSlice`], in place, via a
/// non-blocking `clEnqueueSVMMemcpy` (host-pointer source). SVM analog of
/// [`WriteDevice`]. The buffer passes through as the op's output.
///
/// SVM write stays **non-blocking** regardless of terminal: `submit_on` returns
/// the write event so the copy overlaps downstream work, and
/// `register_drop_callback` keeps the host `src` alive until the memcpy completes
/// (`CL_COMPLETE`). The `Blocking` terminal waits on that same event.
pub struct WriteMapped<T, M: MemMode = ReadWrite> {
    buf: Input<MappedSlice<T, M>>,
    // Retained by value; read by reference each run (see `WriteDevice::src`).
    src: UploadSource<T>,
    out: Pipe<MappedSlice<T, M>>,
}

/// Build an in-place SVM write leaf over an existing `MappedSlice` (concrete or
/// piped). `M: HostWritable` — same gate as [`MappedSlice::write`].
pub fn write_mapped<T, M, S>(buf: impl Into<Input<MappedSlice<T, M>>>, src: S) -> WriteMapped<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteMapped {
        buf: buf.into(),
        src: src.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for WriteMapped<T, M>
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

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (buf, deps, home) = self.buf.resolve_home(ec)?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SVM write is always a non-blocking enqueue; Blocking waits on the event
        // here, Pipelined threads it downstream. `self.src` is valid for the whole
        // `sync` — no keep-alive callback needed. In-place → home threads through.
        let event = crate::mapped::svm_write_enqueue(&buf, ec, self.src.as_slice(), &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_mapped".into());
    }
}

impl<T, M> WriteMapped<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    /// Concrete-head blocking terminal: write on the buffer's own context default
    /// queue and return the buffer for reuse (`let buf = buf.write(d).wait()?;`).
    pub fn wait(self) -> Result<MappedSlice<T, M>> {
        let ctx = concrete_svm_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal returning the buffer plus a completion
    /// [`Event`](crate::Event) — mirrors the Tier-1 `(Output, Event)` contract.
    pub fn submit(self) -> Result<(MappedSlice<T, M>, crate::Event)> {
        let ctx = concrete_svm_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}

// ── Leaf: write host data into a USMSliceUninit → USMSlice (pure host op) ───

/// Eager analog of `WriteFromUninitOp<USMSliceUninit, _>`. Pure host memcpy via
/// the Tier-1 `write_from` helper — surfaces `LengthMismatch` at execute. No
/// enqueue, deps pass through (mode N/A) — mirrors [`Upload`].
pub struct WriteUsmUninit<T: Copy, M: MemMode> {
    uninit: Input<USMSliceUninit<T, M>>,
    // Retained by value; read by reference each run (see `WriteDevice::src`).
    src: UploadSource<T>,
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
        src: src.into(),
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Host memcpy via Tier-1 helper; Err on length mismatch propagates.
        let (uninit, deps) = self.uninit.resolve(ec)?;
        let buf = uninit.write_from(self.src.as_slice())?;
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
    // One-shot: the host `Vec` is moved into the `USMSlice` (USM IS host memory),
    // so a `usm_slice(data)` chain head runs once; a second `sync` errors.
    data: Mutex<Option<Vec<T>>>,
    out: Pipe<USMSlice<T, M>>,
}

/// Build an eager USM-wrap leaf from a host `Vec<T>` with the **default
/// [`ReadWrite`] marker** — no turbofish: `usm_slice(data)`. For a non-default
/// marker use [`usm_slice_as`] with a marker witness.
pub fn usm_slice<T>(data: Vec<T>) -> UsmSlice<T, ReadWrite>
where
    T: Send + 'static,
{
    usm_slice_as(data, ReadWrite)
}

/// Build an eager USM-wrap leaf with an **explicit access marker**, inferred
/// from the `marker` witness — no turbofish: `usm_slice_as(data, HostReadOnly)`.
/// The default-marker shorthand is [`usm_slice`].
pub fn usm_slice_as<T, M>(data: Vec<T>, marker: M) -> UsmSlice<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    let _ = marker;
    UsmSlice {
        data: Mutex::new(Some(data)),
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // USMSlice::new is pure host code — no in-flight event, mode N/A.
        let data = self.data.lock().unwrap().take().ok_or(Error::NotSupported(
            "eager graph: a `usm_slice` host Vec was already consumed — \
             `usm_slice` is a move-in-once chain head and can't drive a reused graph",
        ))?;
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

/// Build an eager uninit-USM alloc leaf with the **default [`ReadWrite`]
/// marker** — no turbofish: `usm_alloc_uninit(N)`. For a non-default marker use
/// [`usm_alloc_uninit_as`] with a marker witness.
pub fn usm_alloc_uninit<T>(len: usize) -> UsmAllocUninit<T, ReadWrite>
where
    T: Send + 'static,
{
    usm_alloc_uninit_as(len, ReadWrite)
}

/// Build an eager uninit-USM alloc leaf with an **explicit access marker**,
/// inferred from the `marker` witness. The default-marker shorthand is
/// [`usm_alloc_uninit`].
pub fn usm_alloc_uninit_as<T, M>(len: usize, marker: M) -> UsmAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    let _ = marker;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
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

/// Build an eager uninit-`DeviceSlice` alloc leaf with the **default
/// [`ReadWrite`] marker** — no turbofish: `device_alloc_uninit(N)`. For a
/// non-default marker use [`device_alloc_uninit_as`] with a marker witness.
pub fn device_alloc_uninit<T>(len: usize) -> DeviceAllocUninit<T, ReadWrite>
where
    T: Send + 'static,
{
    device_alloc_uninit_as(len, ReadWrite)
}

/// Build an eager uninit-`DeviceSlice` alloc leaf with an **explicit access
/// marker**, inferred from the `marker` witness. The default-marker shorthand
/// is [`device_alloc_uninit`].
pub fn device_alloc_uninit_as<T, M>(len: usize, marker: M) -> DeviceAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    let _ = marker;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
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

/// Build an eager uninit-`MappedSlice` alloc leaf with the **default
/// [`ReadWrite`] marker** — no turbofish: `mapped_alloc_uninit(N)`. For a
/// non-default marker use [`mapped_alloc_uninit_as`] with a marker witness.
pub fn mapped_alloc_uninit<T>(len: usize) -> MappedAllocUninit<T, ReadWrite>
where
    T: Send + 'static,
{
    mapped_alloc_uninit_as(len, ReadWrite)
}

/// Build an eager uninit-`MappedSlice` alloc leaf with an **explicit access
/// marker**, inferred from the `marker` witness. The default-marker shorthand
/// is [`mapped_alloc_uninit`].
pub fn mapped_alloc_uninit_as<T, M>(len: usize, marker: M) -> MappedAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    let _ = marker;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
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
    // Retained by value; read by reference each run (re-seed). `&self` outlives
    // the whole `sync`, so the host pixels stay valid across the async write —
    // no per-run keep-alive callback needed.
    pixels: Vec<I::Pixel>,
    dims: I::Dims,
    out: Pipe<I>,
    _ty: PhantomData<fn() -> I>,
}

/// Build an eager image-upload leaf.
pub fn image_upload<I>(pixels: Vec<I::Pixel>, dims: I::Dims) -> ImageUploadEager<I>
where
    I: ImageHostTransfer + crate::image::ImageEnqueue + Send + 'static,
    I::Pixel: Send + 'static,
{
    ImageUploadEager {
        pixels,
        dims,
        out: Pipe::new(),
        _ty: PhantomData,
    }
}

impl<I> DeviceOp for ImageUploadEager<I>
where
    I: ImageHostTransfer + crate::image::ImageEnqueue + Send + 'static,
    I::Pixel: Send + 'static,
{
    type Output = I;

    fn output_pipe(&self) -> Pipe<I> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Image write has no native CL_BLOCKING flag we want here — always a
        // non-blocking enqueue, mode ignored; the chain terminal waits.
        let mut img = I::alloc(ec.context(), self.dims)?;
        let region = img.enqueue_region();
        // Source leaf: no upstream Input, so no wait-list to thread. `self.pixels`
        // stays valid for the whole `sync`, so no keep-alive callback is needed.
        let event = crate::image::write_image_enqueue(
            img.image_mut(),
            ec,
            region,
            self.pixels.as_ptr() as *const std::ffi::c_void,
            false,
            &[],
        )?;
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
    I: ImageHostTransfer + crate::image::ImageEnqueue + Send + 'static,
    I::Pixel: Default + Copy + Send + 'static,
{
    ImageDownloadEager {
        img: img.into(),
        out: Pipe::new(),
    }
}

impl<I> DeviceOp for ImageDownloadEager<I>
where
    I: ImageHostTransfer + crate::image::ImageEnqueue + Send + 'static,
    I::Pixel: Default + Copy + Send + 'static,
{
    type Output = Vec<I::Pixel>;

    fn output_pipe(&self) -> Pipe<Vec<I::Pixel>> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Image read enqueued non-blocking; the chain terminal waits, mode ignored.
        let (img, deps) = self.img.resolve(ec)?;
        let pixel_count = img.pixel_count();
        let region = img.enqueue_region();
        let mut pixels = vec![<I::Pixel as Default>::default(); pixel_count];
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let event = crate::image::read_image_enqueue(
            img.image_ref(),
            ec,
            region,
            pixels.as_mut_ptr() as *mut std::ffi::c_void,
            false,
            &raw,
        )?;
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
// each eager op holds the buffer/view and delegates the exact enqueue body to
// the host-view layer's `acquire_host_view{,_read}` / `release_to_device`
// builders and their inherent `run(ec, deps) -> (Output, Deps)` method (the
// map/unmap primitive that survives in `host_view.rs`). None of these has a
// native blocking enqueue (the map/unmap is always non-blocking `false`), so
// `mode` is ignored.

// ── Leaf: acquire a read/write DeviceSlice host view ────────────────────────

/// Acquire a read/write host view of an upstream `DeviceSlice` via a
/// non-blocking `clEnqueueMapBuffer`. Output is the owned
/// [`DeviceSliceHostView`]. No native blocking enqueue — `mode` ignored.
/// Delegates to the `AcquireDeviceSliceOp` body via `acquire_host_view`.
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve(ec)?;
        // Delegate to the old op's verbatim map body (map/unmap is always
        // non-blocking — mode ignored).
        let (view, out_deps) = buf.acquire_host_view().run(ec, deps)?;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve(ec)?;
        let (view, out_deps) = buf.acquire_host_view_read().run(ec, deps)?;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (view, deps) = self.view.resolve(ec)?;
        let (buf, out_deps) = view.release_to_device().run(ec, deps)?;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve(ec)?;
        let (view, out_deps) = buf.acquire_host_view().run(ec, deps)?;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve(ec)?;
        let (view, out_deps) = buf.acquire_host_view_read().run(ec, deps)?;
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (view, deps) = self.view.resolve(ec)?;
        let (buf, out_deps) = view.release_to_device().run(ec, deps)?;
        self.out.put(buf, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("release_mapped_view".into());
    }
}

// ── Multi-output leaf: copy_to (src, dst) → (src, dst) ──────────────────────
//
// The `copy_to` graph leaf. A copy is a **two-output** op: it returns BOTH the
// source and destination buffers so the chain can thread either onward. It
// mirrors the macro-emitted multi-output kernel shape (commit 0f7083d): two
// element pipes (`Handle = (Pipe<OS>, Pipe<OD>)`), `execute` enqueues once and
// scatters each output into its element pipe (cloning the single completion
// `Dep` onto both), and `into_output` drains both pipes to reconstruct the
// `(src, dst)` tuple.
//
// Rather than re-deriving the ten (src, dst) family bodies (incl. the unsafe
// cross-type SVM-memcpy machinery in `copy.rs`), this op **reuses** the
// `CopyTo` / [`DeviceEnqueue`] `CopyToOp` impls: resolve the two inputs, build
// the op via `src.copy_to(dst)`, run its `DeviceEnqueue::run` (which owns every
// per-family primitive + Uninit→Init transition + buffer-use registration), then
// scatter its `(out_src, out_dst)` Output across the two pipes. All ten families
// come along for free — no `copy.rs` change.
//
// Copy ops have no native blocking enqueue (`submit_on` + event is the only
// path); `mode` is therefore ignored, and copy is rarely terminal anyway (it
// returns buffers onward).

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

// ── CopyHome: build a copy output's return home from its input cell ─────
//
// A copy lends concrete `src`/`dst` cells but may RE-TYPE the dst (`Uninit →
// Init`). `CopyHome<Out>` lets the (input-typed) cell rehome the (output-typed)
// value: identity when the input type already equals `Out`, or a downgrade when
// the input is the `Uninit` wrapper of `Out`. Implemented per buffer family;
// the [`CopyTo2`] `DeviceOp` impl bounds `Src`/`Dst` by it so every supported
// `(src, dst)` pair threads homes. A family that can't express the downgrade
// returns `None` (still safe — that side just doesn't re-arm).

/// Build the typed return [`Rehome`] for a copy output of type `Out` from the
/// (possibly weaker-typed) input cell of type `Self`. `None` when this family
/// can't soundly express the return.
pub trait CopyHome<Out>: Sized {
    /// The home that returns an `Out` into a `Cell<Self>` on `Checkout` drop.
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<Out>>;
}

/// Rehome that DOWNGRADES an `Init` buffer back into a `Cell<Uninit-wrapper>`.
/// Re-wraps via the family's `from_init` (a safe private-field re-wrap) before
/// storing — `Init` is the stronger capability, so forgetting it is sound.
struct DowngradeRehome<U, Init> {
    cell: Cell<U>,
    wrap: fn(Init) -> U,
}

impl<U: Send, Init: Send> Rehome<Init> for DowngradeRehome<U, Init> {
    fn rehome(self: Box<Self>, value: Init) {
        *self.cell.lock().unwrap() = Some((self.wrap)(value));
    }
}

// Identity homes: src is never retyped, and an Init→Init dst is identity too.
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<DeviceSlice<T, M>>
    for DeviceSlice<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<DeviceSlice<T, M>>> {
        Some(Box::new(cell))
    }
}
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<MappedSlice<T, M>>
    for MappedSlice<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<MappedSlice<T, M>>> {
        Some(Box::new(cell))
    }
}
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<USMSlice<T, M>> for USMSlice<T, M> {
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<USMSlice<T, M>>> {
        Some(Box::new(cell))
    }
}

// Downgrade homes: an Uninit dst comes back Init; re-wrap into the uninit cell.
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<DeviceSlice<T, M>>
    for DeviceSliceUninit<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<DeviceSlice<T, M>>> {
        Some(Box::new(DowngradeRehome {
            cell,
            wrap: DeviceSliceUninit::from_init,
        }))
    }
}
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<MappedSlice<T, M>>
    for MappedSliceUninit<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<MappedSlice<T, M>>> {
        Some(Box::new(DowngradeRehome {
            cell,
            wrap: MappedSliceUninit::from_init,
        }))
    }
}
// USM uninit's backing is a `Vec<MaybeUninit<T>>`, so its `from_init` is a
// same-layout `Vec` reinterpret (Init→Uninit, the SAFE downgrade direction —
// the inverse of `assume_init`, with no init assertion). It preserves the heap
// address so the SVM pointer stays valid. Re-arms like the other two families.
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<USMSlice<T, M>>
    for USMSliceUninit<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<USMSlice<T, M>>> {
        Some(Box::new(DowngradeRehome {
            cell,
            wrap: USMSliceUninit::from_init,
        }))
    }
}

/// Eager multi-output copy: `eager_copy_to(src, dst)` enqueues a copy and yields
/// `(src, dst)`. `Handle = (Pipe<OutSrc>, Pipe<OutDst>)` — two element pipes, so
/// a downstream `.and_then(|(src, dst)| …)` selects either side. Polymorphic
/// over every supported `(src, dst)` family via the `Src: CopyTo<Dst>` bound.
pub struct CopyTo2<Src, Dst>
where
    Src: CopyTo<Dst>,
    <Src::Op as crate::eager::DeviceEnqueue>::Output: CopyOutputs,
{
    src: Input<Src>,
    dst: Input<Dst>,
    // One element pipe per copy output (move-once storage), mirroring the
    // macro-emitted multi-output kernel. The output tuple is reconstructed from
    // both in `into_output`.
    src_pipe: Pipe<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src>,
    dst_pipe: Pipe<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst>,
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
    <Src::Op as crate::eager::DeviceEnqueue>::Output: CopyOutputs,
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
    Src: CopyTo<Dst> + Send + 'static,
    Dst: Send + 'static,
    Src::Op: Send,
    <Src::Op as crate::eager::DeviceEnqueue>::Output: CopyOutputs,
    // Each input cell knows how to rehome its (possibly retyped) output: src is
    // identity (never retyped), dst is identity or the Uninit→Init downgrade.
    Src: CopyHome<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src>,
    Dst: CopyHome<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst>,
{
    type Output = (
        <<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src,
        <<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst,
    );
    // Two element pipes, like the multi-output kernel: the downstream closure
    // gets `(pa, pb)` and selects either buffer.
    type Handle = (
        Pipe<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src>,
        Pipe<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst>,
    );
    // Per-output Checkouts: each side independently readable / into_inner'd.
    type Checkouts = (
        Checkout<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src>,
        Checkout<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst>,
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Grab each CONCRETE input's lending cell BEFORE resolving so we can
        // thread it as the output's return home (re-arming `g` on Checkout drop).
        // A pipe-fed input has no concrete cell → `None` (its producer re-mints
        // the value each run; propagating a retyped pipe home is a later step).
        let src_home = self
            .src
            .return_cell()
            .and_then(<Src as CopyHome<_>>::copy_home);
        let dst_home = self
            .dst
            .return_cell()
            .and_then(<Dst as CopyHome<_>>::copy_home);
        // Resolve both inputs → (buffer, upstream Deps). Either may be a pipe
        // (upstream output) or concrete. Combine their wait-lists.
        let (src, src_deps) = self.src.resolve(ec)?;
        let (dst, dst_deps) = self.dst.resolve(ec)?;
        let mut deps = src_deps;
        deps.extend(dst_deps);
        // Reuse the closure-layer copy op: it owns the right per-family
        // primitive (CopyBuffer / SVMMemcpy), the Uninit→Init transition, and
        // buffer-use registration. ONE enqueue → its returned Deps hold one
        // completion event.
        let op = src.copy_to(dst);
        let (out, out_deps) = op.run(ec, deps)?;
        let (out_src, out_dst) = out.into_parts();
        // Clone the completion Dep onto BOTH element pipes so whichever side
        // flows downstream carries the wait-list (and the terminal reconstruct
        // gathers from both). Each output carries its return home: SRC is an
        // identity rehome (the copy never retypes the source); DST is identity
        // (Init→Init) or a sound DOWNGRADE (Uninit dst comes back Init, re-wrapped
        // into its `Cell<…Uninit>` by `CopyHome`). So a concrete-buffer copy in a
        // reused graph re-arms both cells on `Checkout` drop.
        self.src_pipe.put_home(out_src, out_deps.clone(), src_home);
        self.dst_pipe.put_home(out_dst, out_deps, dst_home);
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
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

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // Drain each element pipe with its own home → a tuple of independent
        // Checkouts. (Copy threads no home, so both are `None`; the per-output
        // shape is still correct — each side is its own guard.)
        let src_pipe = self.src_pipe.clone();
        let dst_pipe = self.dst_pipe.clone();
        self.execute(ec, mode)?;
        let (out_src, mut deps, src_home) = src_pipe.take_home().ok_or(Error::NotSupported(
            "eager graph: terminal copy produced no output",
        ))?;
        let (out_dst, dst_deps, dst_home) = dst_pipe.take_home().ok_or(Error::NotSupported(
            "eager graph: terminal copy produced no output",
        ))?;
        deps.extend(dst_deps);
        Ok((
            (
                Checkout::new(out_src, src_home),
                Checkout::new(out_dst, dst_home),
            ),
            deps,
        ))
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("copy_to".into());
    }
}

impl<Src, Dst> crate::record::RecordableOp for CopyTo2<Src, Dst>
where
    Src: CopyTo<Dst> + Send + crate::record::RecordableBuffer + 'static,
    Dst: Send + crate::record::RecordableBuffer + 'static,
    Src::Op: Send,
    <Src::Op as crate::eager::DeviceEnqueue>::Output: CopyOutputs,
    // Mirror the `DeviceOp` impl's home bounds (RecordableOp: DeviceOp).
    Src: CopyHome<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src>,
    Dst: CopyHome<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst>,
{
    fn record(&self, ctx: &mut crate::record::RecordContext) -> Result<()> {
        // `record_handle` comes from the `RecordableBuffer` bound on Src/Dst.
        // Resolve src + dst (own concrete buffer, or upstream producer edge).
        let src_concrete = self.src.with_concrete(|b| b.record_handle());
        let (src_h, src_w) = ctx.resolve_input(src_concrete, self.src.pipe_cell_id())?;
        let dst_concrete = self.dst.with_concrete(|b| b.record_handle());
        let (dst_h, dst_w) = ctx.resolve_input(dst_concrete, self.dst.pipe_cell_id())?;
        // The copy moves `min(src,dst)` bytes (both equal in practice).
        let size = src_h.byte_len.min(dst_h.byte_len);
        let mut waits = src_w;
        waits.extend(dst_w);
        let sp = ctx.copy_buffer(src_h.mem, dst_h.mem, 0, 0, size, waits);
        // Output is `(src, dst)`; both element pipes carry their handle, gated on
        // the copy. (The dst is now initialised — its bytes were written.)
        ctx.register_output(self.src_pipe.cell_id(), src_h, vec![sp]);
        ctx.register_output(self.dst_pipe.cell_id(), dst_h, vec![sp]);
        Ok(())
    }
}

// ── Piped-buffer verb methods: a piped buffer behaves as a buffer ───────────
//
// The concrete `DeviceSlice::{write,read,fill,copy_to}` verbs (buffer.rs) return
// eager ops. A buffer that is *produced upstream* in a graph — a `Pipe<buffer>`,
// the build-time handle of an alloc/upload/kernel op — should read the same way:
// `device_alloc_uninit(n).and_then(|u| u.write(data))`, `bundle!(buf.write(vec),
// other)`. These inherent impls give the pipe types the same verbs, each
// delegating to the eager free fn (which takes `impl Into<Input<_>>`, and a
// `Pipe<T>` converts to `Input::Pipe`). Inherent impls on the concrete owned
// `Pipe<...>` type — no coherence wall. The marker bounds match the concrete
// `DeviceSlice` methods exactly (`HostWritable` / `HostReadable` / `Fillable`).

impl<T, M: MemMode> Pipe<DeviceSlice<T, M>> {
    /// Write `src` into this piped buffer — delegates to [`write`](fn@write). Same
    /// `M: HostWritable` bound as [`DeviceSlice::write`](crate::DeviceSlice::write).
    pub fn write<S>(self, src: S) -> WriteDevice<T, M>
    where
        T: Send + Sync + 'static,
        M: HostWritable + Send + 'static,
        S: Into<UploadSource<T>>,
    {
        write(self, src)
    }

    /// Read this piped buffer into a fresh `Vec<T>` — delegates to [`download`].
    /// Same `M: HostReadable` bound as [`download`].
    pub fn read(self) -> Download<T, M>
    where
        T: Clone + Default + Send + 'static,
        M: HostReadable + Send + 'static,
    {
        download(self)
    }

    /// Fill this piped buffer with `value` — delegates to [`fill`]. Same
    /// `M: Fillable` bound as [`DeviceSlice::fill`](crate::DeviceSlice::fill).
    pub fn fill(self, value: T) -> Fill<T, M>
    where
        T: Copy + Send + Sync + 'static,
        M: Fillable + Send + 'static,
    {
        fill(self, value)
    }

    /// Device-to-device copy this piped buffer into `dst` — delegates to
    /// [`eager_copy_to`]. Yields `(src, dst)`. Same shape as
    /// [`DeviceSlice::copy_to`](crate::DeviceSlice::copy_to).
    pub fn copy_to<M2>(
        self,
        dst: DeviceSlice<T, M2>,
    ) -> CopyTo2<DeviceSlice<T, M>, DeviceSlice<T, M2>>
    where
        T: Send + 'static,
        M: Send + 'static,
        M2: MemMode + Send + 'static,
    {
        eager_copy_to(self, dst)
    }
}

impl<T, M: MemMode> Pipe<DeviceSliceUninit<T, M>> {
    /// Write `src` into this piped uninit buffer (transitioning it to init) —
    /// delegates to [`write_device_uninit`].
    pub fn write<S>(self, src: S) -> WriteDeviceUninit<T, M>
    where
        T: Send + Sync + 'static,
        M: HostUploadable + HostWritable + Send + 'static,
        S: Into<UploadSource<T>>,
    {
        write_device_uninit(self, src)
    }

    /// Fill this piped uninit buffer with `value` (transitioning it to init) —
    /// delegates to [`fill_device_uninit`].
    pub fn fill(self, value: T) -> FillDeviceUninit<T, M>
    where
        T: Copy + Send + Sync + 'static,
        M: Fillable + Send + 'static,
    {
        fill_device_uninit(self, value)
    }
}

impl<T, M: MemMode> Pipe<MappedSlice<T, M>> {
    /// Write `src` into this piped SVM buffer — delegates to
    /// [`write_mapped`](fn@write_mapped). Same `M: HostWritable` bound as
    /// [`MappedSlice::write`](crate::MappedSlice::write).
    pub fn write<S>(self, src: S) -> WriteMapped<T, M>
    where
        T: Send + Sync + 'static,
        M: HostWritable + Send + 'static,
        S: Into<UploadSource<T>>,
    {
        write_mapped(self, src)
    }

    /// Fill this piped SVM buffer with `value` — delegates to
    /// [`fill_mapped`](fn@fill_mapped). Same `M: Fillable` bound as
    /// [`MappedSlice::fill`](crate::MappedSlice::fill).
    pub fn fill(self, value: T) -> FillMapped<T, M>
    where
        T: Copy + Send + Sync + 'static,
        M: Fillable + Send + 'static,
    {
        fill_mapped(self, value)
    }

    /// SVM→SVM copy this piped buffer into `dst` — delegates to
    /// [`eager_copy_to`]. Yields `(src, dst)`. Same shape as
    /// [`MappedSlice::copy_to`](crate::MappedSlice::copy_to).
    pub fn copy_to<Dst>(self, dst: Dst) -> CopyTo2<MappedSlice<T, M>, Dst>
    where
        MappedSlice<T, M>: crate::CopyTo<Dst>,
        <<MappedSlice<T, M> as crate::CopyTo<Dst>>::Op as DeviceEnqueue>::Output: CopyOutputs,
    {
        eager_copy_to(self, dst)
    }
}

impl<T, M: MemMode> Pipe<MappedSliceUninit<T, M>> {
    /// Write `src` into this piped uninit mapped buffer — delegates to
    /// [`write_mapped_uninit`].
    pub fn write<S>(self, src: S) -> WriteMappedUninit<T, M>
    where
        T: Send + Sync + 'static,
        M: HostWritable + Send + 'static,
        S: Into<UploadSource<T>>,
    {
        write_mapped_uninit(self, src)
    }

    /// Fill this piped uninit mapped buffer with `value` — delegates to
    /// [`fill_mapped_uninit`].
    pub fn fill(self, value: T) -> FillMappedUninit<T, M>
    where
        T: Copy + Send + Sync + 'static,
        M: Fillable + Send + 'static,
    {
        fill_mapped_uninit(self, value)
    }
}

impl<T, M: MemMode> Pipe<USMSliceUninit<T, M>> {
    /// Write `src` into this piped uninit USM buffer — delegates to
    /// [`write_usm_uninit`].
    pub fn write<S>(self, src: S) -> WriteUsmUninit<T, M>
    where
        T: Copy + Send + Sync + 'static,
        M: Send + 'static,
        S: Into<UploadSource<T>>,
    {
        write_usm_uninit(self, src)
    }

    /// Fill this piped uninit USM buffer with `value` — delegates to
    /// [`fill_usm_uninit`].
    pub fn fill(self, value: T) -> FillUsmUninit<T, M>
    where
        T: Copy + Send + Sync + 'static,
        M: Send + 'static,
    {
        fill_usm_uninit(self, value)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Execute-time closure nodes — the ONE place closures legitimately survive in
// the eager model (NOTES → "EXECUTE-TIME CLOSURE NODES"). Unlike eager
// `and_then` (its builder runs at BUILD with a `Pipe` handle), the host-seam
// nodes below run their closure at EXECUTE because it needs the mapped host
// data, which does not exist at build. (Device-by-index routing used to live
// here too via `and_then_with_context`; it is now structural — see
// [`OnDevice`] / [`TransferToDevice`] + `DeviceTarget` below.)
// ════════════════════════════════════════════════════════════════════════

// ── DeviceTarget: a device picked either concretely or by context index ──

/// How a routing op (`OnDevice` / `TransferToDevice`) names its target device:
/// either a concrete [`Device`](crate::Device) resolved at build, or an index
/// into the running [`ExecutionContext`]'s `context().devices()`, resolved at
/// execute. The index form is what lets device-by-index routing be expressed
/// structurally (no execute-time closure), so the host-seam gate sees through it.
pub(crate) enum DeviceTarget {
    Concrete(crate::Device),
    Index(usize),
}

// ── OnDevice: re-point the op at a different device's queue at execute ──

/// Route `source`'s `execute` to a **different** device's default
/// out-of-order queue — built by [`on_device`](DeviceOpExt::on_device) (concrete
/// device) or [`on_device_at`](DeviceOpExt::on_device_at) (device-by-index).
///
/// No user closure: at execute it resolves the target device (concrete, or by
/// index into the running context's device list), then its default queue, builds
/// a sibling [`ExecutionContext`] (same context + same host-error slot, different
/// device + queue), and runs `source` against it. The source's events are valid
/// across queues of the same context, so downstream stages on the parent's queue
/// can wait on them cross-device.
pub struct OnDevice<S: DeviceOp> {
    source: S,
    target: DeviceTarget,
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

    fn execute(&self, parent: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // Resolve the target device (concrete, or by index into the running
        // context's device list) before anything else; the rest is unchanged.
        let device = match &self.target {
            DeviceTarget::Concrete(d) => d.clone(),
            DeviceTarget::Index(i) => parent.context().devices()[*i].clone(),
        };
        // Resolve the target queue from the running context (cached, so the
        // terminal's flush_all_outoforder_queues picks it up).
        let target_q = parent.context().default_outoforder_queue(&device)?;
        // Sibling EC: same context + same host-error slot, different device +
        // queue. `target_q` lives on this frame; its `.raw()` borrows for the
        // inner execute().
        let child = ExecutionContext::with_host_error_slot(
            parent.context(),
            device.clone(),
            target_q.raw(),
            parent.host_error_slot(),
            parent.start_dep(),
            parent.workers_handle(),
        );
        // Gather the source against the child EC via `collect` (any arity). The
        // routed sub-chain collapses to OnDevice's single output pipe, so any
        // home a concrete head carried is not threaded across the routing boundary
        // in step (a) (a routed concrete buffer is read via `into_inner`, not
        // auto-re-armed) — the same boundary as the copy/transform ops.
        let (value, deps) = self.source.collect(&child, mode)?;
        self.out.put(value, deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("on_device".into());
    }

    fn contains_host_seam(&self) -> bool {
        self.source.contains_host_seam()
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
    f: Mutex<Option<F>>,
    out: Pipe<S::Output>,
}

/// Like [`AndThenHost`] but the closure also receives `&Context` — built by
/// [`and_then_host_with_context`](DeviceOpExt::and_then_host_with_context).
pub struct AndThenHostWithContext<S: DeviceOp, F>
where
    S::Output: crate::mappable::Mappable,
{
    source: S,
    f: Mutex<Option<F>>,
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
/// ## Error / abort path — TWO user events
///
/// Errors (closure `Err`, panic → `HostPanic`, map-wait failure) are stashed in
/// the chain-wide host-error slot (first-writer-wins). The slot — **not** any
/// cl_event status nor the terminal's blocking return code — is the
/// authoritative caller-facing error channel: the terminal (`sync`/`run`/async
/// poll) checks it even on a "successful" wait and surfaces the rich error.
///
/// The seam is gated by **two** user events, because one event cannot do both
/// jobs at once:
/// - **`fire`** gates the unmaps and is **always** completed `CL_COMPLETE`
///   (success *and* error). Each buffer is unmapped exactly once, cleanly. We do
///   NOT additionally issue a host-side defensive unmap — a *second* unmap on an
///   already-unmapped buffer (legacy Intel NEO decrements map-count at enqueue,
///   not execution) returns `CL_INVALID_VALUE` and corrupts queue state. Folding
///   the defensive unmap away is the concrete bug this two-event split fixes vs.
///   the previous single-event design.
/// - **`proceed`** gates downstream and, on success, is `CL_COMPLETE`. On error
///   it is completed with a negative status to abort downstream device work.
///
/// ### Error path: the start-gate makes the negative `proceed` driver-safe
///
/// Completing a *wait-list* user event with a negative status to abort the
/// downstream was historically unreliable: on legacy Intel NEO a blocking
/// transfer parked on the event could lose its wakeup if the negative status
/// landed in the wait-commit window → deadlock. The fix (landed, see the
/// [`contains_host_seam`](DeviceOp::contains_host_seam) and terminal docs) is the
/// **start-gate**: the waiting terminals (`wait_on`/`sync`, `run`) gate the WHOLE
/// graph on a `start` user event and release it only after everything — including
/// the terminal marker — is enqueued, so a negative `proceed` can never race a
/// downstream command's wait-commit. NEO + rusticl are clean; the error-path
/// tests run normally (no `#[ignore]`).
///
/// Driver note: on pocl the concurrent error cascade exercised three distinct
/// pocl-internal event-handling races (an already-failed dependency running
/// anyway, a `clSetEventCallback` inline-callback use-after-free, and a
/// concurrent double-finish of a multi-dependency event). Those are pocl bugs,
/// fixed in `bricevideau-ai/pocl` (PRs #2214/#2215/#2216) — not claspr
/// correctness issues. See NOTES → Concerns for the root causes.
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

    // `fire` gates the unmaps (always completed CL_COMPLETE — one clean unmap per
    // buffer). `proceed` gates downstream and carries the abort. See the doc
    // comment above for why these must be two distinct events.
    let fire = Arc::new(create_user_event(ec.context())?);
    let proceed = Arc::new(create_user_event(ec.context())?);

    // Enqueue the unmaps gated on `fire`. After this point we MUST complete BOTH
    // user events before any early return, or the queue would wait forever.
    let unmap_events = match <O as Mappable>::enqueue_unmap(&mut handle, q, &[fire.get()]) {
        Ok(evs) => evs,
        Err(e) => {
            let _ = complete_user_event(&fire, -1);
            let _ = complete_user_event(&proceed, -1);
            return Err(e);
        }
    };

    // Spawn the worker. It owns the handle, the map events, the source events
    // (for upstream-error short-circuit), both user-event Arc clones, the chain's
    // host-error slot, and the closure.
    let worker_fire = Arc::clone(&fire);
    let worker_proceed = Arc::clone(&proceed);
    let worker_host_error = ec.host_error_slot();
    let handle = std::thread::spawn(move || {
        let (status, handle, rust_err) =
            run_host_worker::<O, F>(handle, map_events, source_deps, host_call);
        // Stash the rich Rust error first (first-writer-wins — a concurrent
        // failing worker in the same bundle/fan-out may already have written).
        if let Some(err) = rust_err {
            let mut slot = worker_host_error.lock().unwrap();
            if slot.is_none() {
                *slot = Some(err);
            }
        }
        // ALWAYS fire the unmaps cleanly (single unmap per buffer). No defensive
        // unmap — that double-unmap is what corrupts legacy NEO.
        let _ = complete_user_event(&worker_fire, opencl3::event::CL_COMPLETE);
        // Allow / abort downstream device work via `proceed` (negative on error).
        // With the start gate (the whole graph is enqueued before `start` is
        // released), a negative `proceed` can no longer race a downstream blocking
        // transfer's wait-commit on legacy NEO.
        let _ = complete_user_event(&worker_proceed, status);
        // `handle` drops here: `enqueue_unmap` succeeded so its `unmap_enqueued`
        // flag is set and Drop is a no-op — the gated unmap fired via `fire`.
        drop(handle);
    });
    // Hand the worker to the EC so the terminal joins it AFTER the device wait —
    // no detached worker whose CL calls (signal events, drop retained queue) race
    // the caller dropping the Context.
    ec.push_worker(handle);

    // Downstream gates on ALL unmap events PLUS `proceed`. When the output has no
    // buffers (scalar / unit), there are no unmaps and `fire` gates nothing — but
    // `proceed` still gives downstream its proceed/abort gate.
    let mut deps_out: Deps = unmap_events.into_iter().map(wrap_event).collect();
    deps_out.push(proceed);
    Ok((source_value, deps_out))
}

/// Has `ev` been COMMITTED to a terminal error/cancellation state? Reads the
/// event's `command_execution_status` (the authoritative committed value, unlike
/// the `clWaitForEvents` return code which can lose the abort to a legacy-NEO
/// wakeup race) and reports `true` when it is negative — i.e. the command (a
/// device command, or an upstream host seam's `proceed` user event) terminated
/// abnormally. A query failure is treated as "not cancelled" (conservative: we
/// then fall through to the normal closure path rather than spuriously aborting).
fn event_is_cancelled(ev: &crate::Event) -> bool {
    ev.command_execution_status().map(|s| s.0).unwrap_or(0) < 0
}

/// Worker body for [`run_host_seam`]. Waits the source + map events, runs the
/// closure under `catch_unwind`, and returns `(status, handle, optional rich
/// error)`. The caller stashes the error in the chain host-error slot, always
/// completes the `fire` event (so the unmaps run cleanly), and forwards `status`
/// to the `proceed` event (negative aborts downstream). `status` is
/// `CL_COMPLETE` on success, negative otherwise. The returned `handle` is
/// dropped by the caller as a no-op (its unmap already fired via `fire`).
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

    // CHAINED-CANCEL via the EVENT DEPENDENCY (not a host-side slot — that races
    // the upstream worker's stash). An upstream host seam that fails completes its
    // `proceed` user event with a NEGATIVE status; that `proceed` is in THIS
    // seam's `source_deps` (it is the upstream op's output dep). So once the
    // upstream cancels, the cancellation is carried by the EVENT, and this seam
    // short-circuits — DO NOT run its closure — then completes its own `proceed`
    // negative so the rest of the chain cancels too.
    //
    // Robustness on legacy NEO: we do NOT trust `clWaitForEvents`' RETURN code to
    // report the abort (a lost-wakeup-style race can let the wait return
    // CL_SUCCESS even when the event was set negative — the same NEO race the
    // start-gate fixes for device commands, but here on a host-side wait). After
    // waiting, we re-read the event's COMMITTED `command_execution_status`: a
    // negative value is the authoritative "this command terminated in error"
    // signal and is not subject to the wakeup race. That negative status is what
    // we treat as cancellation.
    //
    // Source events first: a negative committed status here is an upstream chain
    // abort, not a fault of this seam — short-circuit WITHOUT stashing (the
    // upstream already stashed the first/authoritative error).
    for ev in &source_deps {
        let _ = ev.as_ref().wait();
        if event_is_cancelled(ev.as_ref()) {
            return (-1, handle, None);
        }
    }
    // Map events: enqueued over the source events, so an upstream cancel also
    // errors these. The source check above already caught the upstream-cancel
    // case, so a negative status here means a real map failure — stash the cause.
    for ev in &map_events {
        let _ = ev.wait();
        if event_is_cancelled(ev) {
            // Best-effort fetch of the concrete code; fall back to a generic
            // exec-status error if the query itself fails.
            let code = ev
                .command_execution_status()
                .map(|s| s.0)
                .unwrap_or(opencl3::error_codes::CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
            return (
                -1,
                handle,
                Some(Error::OpenCl(opencl3::error_codes::ClError(code))),
            );
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Gather the source via `collect` (any arity — a bundle source fills
        // element pipes, not output_pipe).
        let (value, deps) = self.source.collect(ec, ExecMode::Pipelined)?;
        let f = self.f.lock().unwrap().take().ok_or(Error::NotSupported(
            "eager graph: an `and_then_host` closure was already consumed — a host \
             seam is a one-shot `FnOnce`, so a graph containing one can't be reused \
             (step-(a) limitation; reuse covers pure-device graphs)",
        ))?;
        let (out_value, out_deps) = run_host_seam::<S::Output, F>(value, deps, ec, f)?;
        self.out.put(out_value, out_deps);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("and_then_host".into());
    }

    fn contains_host_seam(&self) -> bool {
        true
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Gather the source via `collect` (any arity).
        let (value, deps) = self.source.collect(ec, ExecMode::Pipelined)?;
        let f = self.f.lock().unwrap().take().ok_or(Error::NotSupported(
            "eager graph: an `and_then_host_with_context` closure was already \
             consumed — a host seam is a one-shot `FnOnce`, so a graph containing \
             one can't be reused (step-(a) limitation)",
        ))?;
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

    fn contains_host_seam(&self) -> bool {
        true
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
    cb: Mutex<Option<F>>,
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
            cb: Mutex::new(Some(cb)),
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

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
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
        let cb = self.cb.lock().unwrap().take().ok_or(Error::NotSupported(
            "eager graph: a `.profiled()` callback was already consumed — the \
             profiling callback is a one-shot `FnOnce` and can't drive a reused graph",
        ))?;
        crate::register_profiling_callback(&marker, Box::new(cb))?;
        // The marker becomes this op's completion event for downstream
        // chaining (it subsumes the source's events).
        self.out.put(value, vec![wrap_event(marker)]);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("profiled".into());
    }

    fn contains_host_seam(&self) -> bool {
        self.source.contains_host_seam()
    }
}

// ── DeviceChainFuture: the async `.run().await` terminal ────────────────

/// Future returned by [`DeviceOpExt::run`]. Resolves to `Result<T>` once the
/// chain's commands have all completed on the device (or immediately, with an
/// error, if the chain failed to submit or any host seam returned `Err`).
///
/// The future returned by the async terminal [`run`](DeviceOpExt::run). The host
/// seam (`run_host_seam`) runs its closure on a worker thread and stashes any
/// failure into the chain's host-error slot before signalling its user event with a
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
        /// Host-seam worker handles, joined once the marker resolves (so a
        /// worker's late CL calls finish before the caller can drop the
        /// `Context`). Empty for pure device graphs. Shared `Arc` with the EC
        /// that spawned them (including routed `on_device` sub-chains).
        workers: std::sync::Arc<std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>>,
    },
}

/// Drain and join every host-seam worker handle (no-op when empty). Mirrors
/// [`ExecutionContext::join_workers`] for the async terminal, which only has the
/// shared `Arc` (the EC dropped at `run` time).
#[cfg(feature = "async-events")]
fn join_chain_workers(
    workers: &std::sync::Arc<std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>>,
) {
    let handles: Vec<std::thread::JoinHandle<()>> = std::mem::take(&mut *workers.lock().unwrap());
    for h in handles {
        let _ = h.join();
    }
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
                workers,
            } => match std::pin::Pin::new(event_future).poll(cx) {
                Poll::Pending => Poll::Pending,
                // Marker resolved Err: a host worker (or a CL command) failed.
                // Prefer the rich Rust variant the worker stashed over the
                // cl_event cascade, mirroring `sync`.
                Poll::Ready(Err(e)) => {
                    // Marker done → all device work (and the seam's signals) are
                    // settled; join workers before returning.
                    join_chain_workers(workers);
                    Poll::Ready(Err(host_error.lock().unwrap().take().unwrap_or(e)))
                }
                // Even on a "successful" marker, a host worker may have stashed
                // an error the marker did NOT propagate: pocl's
                // `clEnqueueMarkerWithWaitList` does not cascade negative status
                // from a user event in its wait-list (it reports CL_COMPLETE while
                // the chain genuinely failed). A non-empty slot is itself the
                // failure signal. (Same handling as the old `ChainFuture`.)
                Poll::Ready(Ok(())) => {
                    join_chain_workers(workers);
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
/// [`ExecutionContext`] (default OOO queue, like `sync`), gather via `collect` in
/// [`ExecMode::Pipelined`], enqueue a marker over the chain's deps (before
/// releasing the `start` gate, when present), and wrap it in an
/// [`EventFuture`](crate::EventFuture).
///
/// Synchronous-error paths invalidate the context's cached OOO queue.
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
    let mut ec = ExecutionContext::new(context, device.clone(), queue.raw());
    // Clone the chain's host-error slot + worker-join list out before `ec` drops
    // — host-seam workers stash failures in the slot, and the future joins the
    // workers + reconciles errors at poll time.
    let host_error = ec.host_error_slot();
    let workers = ec.workers_handle();

    // START-GATE (only when the chain contains a host seam): create a `start`
    // user event, gate the whole graph on it, enqueue everything, then release
    // `start`. Mirrors the blocking terminal — closes the legacy NEO lost-wakeup
    // window. Pure device graphs skip this entirely (start = None, zero cost).
    let start = if chain.contains_host_seam() {
        match crate::create_user_event(context) {
            Ok(ev) => {
                ec.set_start(ev.get());
                Some(ev)
            }
            Err(e) => {
                drop(queue);
                context.invalidate_default_outoforder_queue(&device);
                return DeviceChainFuture::Errored(Some(e));
            }
        }
    } else {
        None
    };

    // 2-3. Run the chain non-blocking and gather its result via `collect` —
    //    the uniform gather seam. `collect` dispatches to the right per-op
    //    reconstruction (single OR multi-output: bundle*, arc_split, the copy
    //    pair all yield their reconstructed value + joined deps), so the async
    //    terminal supports every arity the blocking `sync` does. A host-seam
    //    setup error (map/unmap enqueue) still surfaces synchronously here; a
    //    host-CLOSURE failure surfaces at poll time via the host-error slot.
    let collected = chain.collect(&ec, ExecMode::Pipelined);

    // Helper: release the start-gate (if any) so gated commands can drain/abort
    // instead of waiting on `start` forever. Used on every exit path below.
    let release_start = |start: &Option<crate::Event>| {
        if let Some(start) = start {
            let _ = crate::complete_user_event(start, opencl3::event::CL_COMPLETE);
        }
    };

    let (output, deps) = match collected {
        Ok(pair) => pair,
        Err(e) => {
            // Setup failed before/while enqueueing — release the gate so any
            // already-enqueued commands drain, then join workers and surface the
            // error (prefer the seam's rich stash).
            release_start(&start);
            join_chain_workers(&workers);
            drop(queue);
            context.invalidate_default_outoforder_queue(&device);
            return DeviceChainFuture::Errored(Some(
                host_error.lock().unwrap().take().unwrap_or(e),
            ));
        }
    };

    // 4. Enqueue a marker over every event the chain produced. Precise
    //    wait-list — we don't penalise other work sharing this OOO queue.
    //    SAFETY: each `cl_event` is held alive by the `deps` Arc wrappers for
    //    the duration of this call; the marker enqueue retains them internally.
    //
    //    ORDER: enqueue the marker BEFORE releasing `start`. The marker is part
    //    of the chain's terminal join, so it must be inside the start-gate like
    //    every other command — otherwise its wait-list edges get wired against
    //    deps that may already be resolving/failing concurrently (the gate's
    //    whole point is that the *entire* graph is committed before anything
    //    runs). The marker enqueue is non-blocking, so wiring it while `start`
    //    is still held does not deadlock.
    let wait_list: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
    let marker = match unsafe { queue.raw().enqueue_marker_with_wait_list(&wait_list) } {
        Ok(ev) => ev,
        Err(code) => {
            release_start(&start);
            drop(deps);
            join_chain_workers(&workers);
            drop(queue);
            context.invalidate_default_outoforder_queue(&device);
            return DeviceChainFuture::Errored(Some(Error::OpenCl(code)));
        }
    };
    drop(deps);

    // The whole graph — including the terminal marker — is now enqueued and
    // gated on `start`. Release it so everything drains (or aborts via its own
    // `proceed`).
    release_start(&start);

    // 4a. clFlush — push every queue the chain touched without blocking.
    //     rusticl is spec-strict and keeps commands `CL_QUEUED` until an
    //     explicit flush, so the marker's `CL_COMPLETE` callback would never
    //     fire and the future would deadlock. flush_all also covers
    //     `.on_device(&dev_b)` chains whose commands land on non-primary queues.
    if let Err(e) = context.flush_all_outoforder_queues() {
        join_chain_workers(&workers);
        drop(queue);
        context.invalidate_default_outoforder_queue(&device);
        return DeviceChainFuture::Errored(Some(e));
    }
    // `start` (if any) has been completed; its retained dep refs keep the cl_event
    // alive in the wait-lists until those commands run. Safe to drop here.
    drop(start);

    // 5. Wrap the marker in the EventFuture machinery (clSetEventCallback).
    //    The future joins `workers` when the marker resolves.
    DeviceChainFuture::Running {
        output: Some(output),
        event_future: marker.into_future(),
        host_error,
        workers,
    }
}
