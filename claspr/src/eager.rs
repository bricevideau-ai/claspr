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
//!
//! # Writing reusable-graph host code
//!
//! The sections above describe the engine internals; this one is the *usage*
//! guide — how to write host code against a graph. Worked examples:
//! `examples/gray-scott` (`run_immutable`, a curried meta-kernel replayed by
//! period) and `examples/cg` (a self-closing Conjugate-Gradient loop).
//!
//! ## Combinator cheat-sheet (concept → the function to reach for)
//!
//! Start here — most "how do I express X" questions map to one existing name:
//!
//! | You want to… | Use | Notes |
//! |---|---|---|
//! | sequence: run B on A's output | [`and_then`](DeviceOpExt::and_then) | builder runs at construction over A's [`Handle`](DeviceOp::Handle) |
//! | **carry a value PAST an intervening step** (keep it to hand to a later op / the terminal) | **[`forward`]** | `forward(x)` re-exposes an already-produced value as a one-node op; capture its handle in the `move` closures. This is the "thread a buffer forward" primitive. |
//! | run device ops in PARALLEL | [`bundle2`] / [`bundle!`](crate::bundle) | independent branches; per-branch structure-preserving `Checkouts` |
//! | N-way homogeneous parallel | [`fan_out`] | one op per item |
//! | hold a host VALUE as a graph input | [`value`] (by-value, `Clone`) or [`lift`] (owned / non-`Clone`, self-rehoming — replays) | |
//! | present an OWNED buffer/scalar as a re-homing branch (no device work) | [`lift`] | lends-and-returns from its own [`Cell`]; a `lift`ed value re-arms across `sync`s like a concrete input, so `bundle!(lift(a), lift(b), …).and_then_host(…)` is a replayable multi-home seam |
//! | get data ONTO / OFF the device | [`upload`] / [`download`] | non-recordable (host transfer) |
//! | a rebindable typed hole | [`slot!`](crate::slot) + [`bind`](DeviceOpExt::bind)/[`call`](DeviceOpExt::call) | see "Slots" below |
//! | a device-resident scalar | [`crate::DeviceScalar`] via [`crate::device_scalar`] | binds a `&T`/`&mut T` kernel arg; see "Device scalars" below |
//! | run host code mid-graph | [`and_then_host`](DeviceOpExt::and_then_host) | writable [`&mut View`](crate::mappable::Mappable::View); reusable — the `Fn` closure re-runs on every replay (borrow / `Arc` / clone captures, don't move-consume) |
//!
//! There is intentionally NO `present`/`hold`/`carry`/`thread`/`identity` — the
//! "keep a value around" verb is [`forward`] (for an already-produced [`Pipe`]);
//! the "inject / present an owned host value or buffer" verbs are [`value`] (by
//! value, `Clone`) and [`lift`] (owned / non-`Clone`, self-rehoming so it presents
//! a concrete buffer/scalar as a re-arming branch that replays).
//!
//! ## Build once, run many — the [`Checkout`] lend/rehome cycle
//!
//! A graph `g` is a **reusable** value: [`sync`](DeviceOpExt::sync) takes `&self`,
//! so you build `g` **once** and call `g.sync(ctx)?` in a loop. Each run **lends**
//! the graph's buffers (a concrete cell hands its buffer out for the duration of
//! the run) and returns a [`Checkout`] over the output. When that `Checkout` is
//! **dropped**, the buffer is **rehomed** — returned to its origin cell — so the
//! *next* `sync` re-lends the SAME `cl_mem` with zero rebinding (a stable handle
//! across replays). This is the whole basis of graph reuse:
//!
//! ```ignore
//! let g = ks.scale([N], slot!(Buf), 2.0).bind(Buf(b));  // build + bind ONCE
//! loop {
//!     let co = g.sync(ctx)?;   // lends Buf's buffer, runs, returns a Checkout
//!     // ... read co if needed ...
//!     drop(co);                // rehomes the buffer → g is armed for the next sync
//! }
//! ```
//!
//! ## Reading a result: deref + `map` (borrow, don't consume)
//!
//! A [`Checkout<O>`] **derefs to `O`**, and buffer reads (`.map()`, `.read()`)
//! take `&self` — so you can read a result **without consuming** the Checkout,
//! leaving it free to rehome on drop:
//!
//! ```ignore
//! let co = g.sync(ctx)?;
//! let v = (*co).map().wait()?;   // borrows — co still drops/rehomes afterward
//! ```
//!
//! ## [`into_inner`](Checkout::into_inner) vs `drop` — sever vs rehome
//!
//! - **`drop(co)`** → the buffer REHOMES to its cell; `g` re-arms and re-runs.
//! - **`co.into_inner()`** → SEVERS: you take ownership of the buffer, and its
//!   origin cell is left empty (a plain `bind`/`sync` won't re-arm it; that needs
//!   `mutate_bind`). Use `into_inner` only when you genuinely want to *extract* a
//!   buffer out of the graph, not to read it. For read-and-reuse, prefer
//!   deref+`map` then `drop`.
//!
//! ## `reclaim_undelivered` — you needn't thread every buffer to the terminal
//!
//! Only the outputs your terminal *names* come back as Checkouts. A mid-graph
//! buffer that is produced but **not consumed downstream and not returned** is
//! **reclaimed** on the run's drop — rehomed to its cell like any lent buffer (see
//! [`reclaim_undelivered`](DeviceOp::reclaim_undelivered)). So a self-closing loop
//! body only has to thread to the terminal the handful of values the host actually
//! reads; every other buffer just needs to be *used* (lent from its cell) and its
//! unconsumed tail reclaims. (`examples/cg` threads only the solution `x` and the
//! residual scalar `rsnew`; its other 8 buffers reclaim.)
//!
//! ## What [`sync`](DeviceOpExt::sync) hands back
//!
//! The [`Checkouts`](DeviceOp::Checkouts) shape mirrors the graph's output shape:
//! - a single-output op → one `Checkout<Output>`;
//! - a **multi-output kernel** → per-buffer Checkouts, `(Checkout<a>, Checkout<b>,
//!   …)` — each droppable/readable independently;
//! - a **[`bundle`](bundle2)** → per-branch, structure-preserving:
//!   `(A::Checkouts, B::Checkouts)` (a single-output branch contributes one
//!   `Checkout`, a multi-output branch its own tuple, a nested bundle its nested
//!   shape). Grouped by branch, per-buffer within each branch.
//!
//! ## Self-closing loops (in-graph iteration)
//!
//! A graph whose outputs feed back to its own inputs across `sync`s is a
//! *self-closing* loop: build the whole iteration as one `and_then` chain that
//! ends producing the values the next iteration reads (each an internal
//! [`Pipe`] edge, threaded through the builder closures — carry a value forward by
//! capturing its handle in the `move` closures, not by naming the concrete cell
//! twice, which would double-lend). `sync` in a loop; drop the Checkout to re-arm.
//! Keeping loop control on the host (a convergence test) while all compute is one
//! device graph is the ideal shape for the future command-buffer backend (one
//! recordable region, replayed, with a small host readback between replays).
//! `examples/cg` is this pattern end-to-end.
//!
//! ## Slots, scalars, and other authoring notes
//!
//! - **Slots** ([`slot!`](crate::slot) / [`slots!`](crate::slots)) make a graph
//!   rebindable: a `slot!(Tag)` hole is filled by the consuming set-once
//!   [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call) (or re-set by the
//!   fluent [`mutate_bind`](DeviceOpExt::mutate_bind) /
//!   [`mutate_call`](DeviceOpExt::mutate_call) for replay loops). Errors on the
//!   consuming verbs are deferred to `sync` and are STICKY (rebuild to recover);
//!   the fluent `mutate_*` verbs error eagerly and never poison the graph.
//! - **Device scalars**: a kernel arg `#[spirv(cross_workgroup)] s: &f32` /
//!   `&mut f32` binds a [`DeviceScalar<f32>`](crate::DeviceScalar) (built with
//!   [`device_scalar`](crate::device_scalar); also [`MappedScalar`](crate::MappedScalar)
//!   / [`USMScalar`](crate::USMScalar) for the SVM tiers), read/written in-kernel as
//!   `*s`. A scalar lives on-device and pipes through the graph like any buffer — the
//!   way `examples/cg` keeps α/β/residual device-resident and avoids host round-trips
//!   inside the loop. A plain len-1 `DeviceSlice` does NOT bind to a `&T` arg (and a
//!   `DeviceScalar` does not bind to a `&[T]` arg) — the mismatch is a compile error.
//!   `and_then_host` over a `DeviceScalar` maps it as a scalar [`&mut T`](crate::mappable::Mappable::View).
//! - **`Kernels` is cheap to share** across builder closures: a launcher clones
//!   the context internally rather than borrowing the `Kernels`, so the built op
//!   owns what it needs — pass `&ks` freely into nested `and_then` closures.

use crate::buffer::{DeviceScalar, Scalar};
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
use std::any::{Any, TypeId};
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

// ── SlotState<T>: the five-state slot cell ─────────────────────────────────

/// The cell behind an [`Input::Slot`] — a **five-state** resource holder, the
/// distinction a bare `Option<T>` cannot make.
///
/// The first four states form the eager value-bind machine (`Unbound` / `Bound` /
/// `Lent` / `Severed`); the fifth, [`FedByPipe`](SlotState::FedByPipe), wires the
/// slot to an upstream pipe instead of a value (installed by a `Tag(pipe)` source
/// through [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call)) and behaves
/// like [`Input::Pipe`] at run time.
///
/// A concrete head's [`Cell<T>`] only ever needs `Some`/`None` (full / lent),
/// because "the cell is empty" unambiguously means "a run lent it and a still-live
/// `Checkout` owes it back" (the graph is busy). A SLOT, though, can be empty for
/// **three different reasons**, and the verb 2×2 ([`bind`](DeviceOpExt::bind) /
/// [`mutate_bind`](DeviceOpExt::mutate_bind)) must tell them apart at bind time:
///
/// - [`Unbound`](SlotState::Unbound) — **virgin**: never filled. A `bind` here is
///   the slot's first declaration, so both verbs simply fill it; resolving it is
///   [`Error::SlotUnbound`].
/// - [`Bound`](SlotState::Bound) — holds a buffer, ready to lend. `bind` is
///   idempotent if the new value is the *same* buffer ([`SlotEq`]) and
///   [`Error::SlotConflict`] otherwise; `mutate_bind` overwrites.
/// - [`Lent`](SlotState::Lent) — its buffer is currently checked out to a live
///   [`Checkout`] from an in-flight run. Re-binding here is
///   [`Error::SlotCheckedOut`] for BOTH verbs: the value is in the caller's hands,
///   and the Checkout's drop will rehome the OLD buffer over any new one — a
///   silent clobber. The caller must drop the Checkout (re-arm → `Bound`) or
///   `into_inner` it (sever → `Severed`) first.
/// - [`Severed`](SlotState::Severed) — **was bound, then its value was taken** via
///   [`into_inner`](Checkout::into_inner). The slot is empty like `Unbound`, but
///   NOT virgin: re-providing a buffer is a *change*, not a first declaration. So
///   the set-once `bind` is [`Error::SlotSevered`] (it must not silently re-fill a
///   slot whose value the caller deliberately extracted); only `mutate_bind` may
///   re-arm it (→ `Bound`). Resolving it is [`Error::SlotUnbound`] (nothing to
///   lend, same as virgin). This is the state that fixes the old `into_inner`
///   bug, where `Lent → Unbound` let a plain `bind` wrongly succeed after sever.
///
/// The four transitions live at: lend (`Bound → Lent`, `Input::lend_slot`),
/// rehome on Checkout drop (`Lent → Bound`, `SlotHome::rehome`), sever on
/// `into_inner` (`Lent → Severed`, `SlotHome::sever`), and `mutate_bind` re-arming
/// a severed slot (`Severed → Bound`, `Input::try_bind_slot`).
pub enum SlotState<T> {
    /// **Virgin** — never bound. The slot is genuinely empty and a `bind` is its
    /// first declaration (both verbs fill it).
    Unbound,
    /// Holds a buffer ready to lend on the next run.
    Bound(T),
    /// The buffer is lent to a live `Checkout`; the slot is empty *because a run
    /// is in flight*, NOT because it was never bound.
    Lent,
    /// **Was bound, value taken** — the caller extracted the buffer via
    /// [`into_inner`](Checkout::into_inner). Empty like `Unbound`, but a set-once
    /// `bind` rejects it ([`Error::SlotSevered`]); only `mutate_bind` re-arms it.
    Severed,
    /// **Fed by an upstream pipe** — the slot's value is produced by an UPSTREAM op
    /// each run and delivered through this [`Pipe`], NOT bound eagerly to a buffer.
    ///
    /// This is the fifth slot state and the engine half of the pipe-feed source
    /// (`Tag(pipe)` through [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call)):
    /// it lets a `slot!(Tag)` declared in one
    /// subgraph be wired to an upstream [`Handle`](DeviceOp::Handle) (a build-time
    /// [`Pipe`]) at compose time, so a downstream stage READS whatever an upstream
    /// stage produced without a concrete buffer ever being named. It is how the
    /// crossed double-buffer rotation is expressed as *data* in the arg list.
    ///
    /// At run time it behaves exactly like [`Input::Pipe`], NOT like a bound slot:
    ///
    /// - `Input::lend_slot` DRAINS the pipe (which the producer filled
    ///   earlier THIS run) via [`Pipe::take_home`], carrying the upstream deps + home
    ///   onward — and LEAVES the `FedByPipe` variant in place so the slot RE-ARMS on
    ///   the next replay (the upstream refills the pipe each run). This mirrors the
    ///   [`Input::Pipe`] arm of [`resolve_home`](Input::resolve_home) exactly.
    /// - [`check_ready`](Input::check_ready) treats it as *satisfied-by-upstream*
    ///   (deferred, like [`Input::Pipe`]) — NEVER [`Error::SlotUnbound`], even though
    ///   it is not [`Bound`](SlotState::Bound).
    ///
    /// A `FedByPipe` slot is never `Bound`/`Lent`/`Severed` under normal use, so the
    /// bind-time verb-2×2 arms for it are inert-but-spelled (a value bind onto a
    /// pipe-fed slot is a misuse the spike never performs; see
    /// [`try_bind_slot`](Input::try_bind_slot)).
    FedByPipe(Pipe<T>),
}

/// The four-state cell shared by a [`SlotHandle`] and its [`Input::Slot`].
pub type SlotCell<T> = Arc<Mutex<SlotState<T>>>;

// ── DeferredErrors: the graph-reachable sink for the INFALLIBLE apply path ──

/// A **graph-reachable deferred-error sink** — the record-don't-drop channel for
/// the infallible, consuming [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call)
/// path (via [`CallArg::apply`]).
///
/// ## Why it exists (the silent-swallow hole it closes)
///
/// The consuming `bind`/`call` verbs are INFALLIBLE (they return the owned graph so
/// they fit inside an [`and_then`](DeviceOpExt::and_then) closure). Their per-element
/// [`apply`](CallArg::apply) used to `let _ = g.bind(..)` — DROPPING every bind
/// error and trusting [`check_ready`](DeviceOp::check_ready) to re-catch it at
/// `sync`. But `check_ready` only re-catches slots left *unbound*: a
/// [`SlotConflict`](Error::SlotConflict) leaves the cell `Bound` to the OLD value,
/// a [`SlotNoSuchTag`](Error::SlotNoSuchTag) has no cell at all, and
/// [`SlotCheckedOut`](Error::SlotCheckedOut) / [`SlotSevered`](Error::SlotSevered)
/// leave states `check_ready` may accept — so those errors VANISHED and the graph
/// RAN with wrong / stale data.
///
/// ## The mechanism
///
/// Every [`slot!`](crate::slot) hole carries its own (empty) sink. The infallible
/// apply path RECORDS a bind error into a sink reachable from the graph (the sink
/// of the first slot cell the `bind_slots` walk visits — captured by the
/// [`SlotBinder`] as it walks, which works even for an ABSENT tag, whose walk
/// matches no cell but still visits the graph's real slots). At `sync`,
/// [`check_ready`](DeviceOp::check_ready) DRAINS every slot's sink FIRST — before
/// any enqueue — and returns the first recorded error, preserving the atomicity
/// guarantee (nothing ran). A graph on which no deferred bind ever erred keeps
/// EMPTY sinks, so `check_ready` drains nothing and the reuse / re-sync path is
/// byte-for-byte unchanged. An errored graph reports at `sync` instead of running;
/// it need not stay reusable (the error is terminal for that graph value).
///
/// Only the infallible apply path writes here; the fluent
/// [`mutate_bind`](DeviceOpExt::mutate_bind) / [`mutate_call`](DeviceOpExt::mutate_call)
/// verbs still surface their errors EAGERLY and never touch a sink.
///
/// ## Sticky / poison recovery contract
///
/// A recorded deferred error **poisons the graph**: [`check_ready`](DeviceOp::check_ready)
/// **peeks** the sink (see `peek_deferred`) rather than draining it, so EVERY
/// subsequent `sync` re-reports the same error. There is no in-place clearing — the
/// legitimate recovery is to **rebuild the graph** (the factory idiom; graphs are
/// cheap, and reuse was always factory-shaped). This is deliberate: an errored graph
/// stays errored, so a caller cannot accidentally ignore the error and re-`sync` into
/// a run. Contrast the fluent [`mutate_bind`](DeviceOpExt::mutate_bind) /
/// [`mutate_call`](DeviceOpExt::mutate_call): they fail EAGERLY at the call site and
/// never touch the sink, so a failed mutate leaves the graph unpoisoned and usable.
pub type DeferredErrors = Arc<Mutex<Vec<Error>>>;

/// **Peek** the first recorded error in a [`DeferredErrors`] sink WITHOUT removing it
/// (sticky/poison — see the type doc), reconstructing an owned [`Error`] from the
/// borrowed one. Returns `None` for an empty sink (the common, never-erred case).
///
/// [`Error`] is not `Clone` (it wraps a foreign `ClError`), but every error the
/// deferred apply path records is one of the slot variants carrying a `Copy`
/// `&'static str` tag name — so a peek reconstructs the exact variant. Any other
/// variant (never produced here) is reported as an internal-inconsistency
/// [`Error::NotSupported`] rather than silently dropped.
fn peek_deferred(sink: &Mutex<Vec<Error>>) -> Option<Error> {
    sink.lock().unwrap().first().map(|e| match e {
        Error::SlotConflict(n) => Error::SlotConflict(n),
        Error::SlotNoSuchTag(n) => Error::SlotNoSuchTag(n),
        Error::SlotCheckedOut(n) => Error::SlotCheckedOut(n),
        Error::SlotSevered(n) => Error::SlotSevered(n),
        Error::SlotUnbound(n) => Error::SlotUnbound(n),
        _ => Error::NotSupported(
            "eager graph: a non-slot error was recorded in a deferred-error sink \
             (internal inconsistency)",
        ),
    })
}

// ── ScalarSlotState: the TWO-state cell for non-resource (scalar/launch) slots ─

/// The simple, **two-state** cell for a NON-resource slot — a scalar kernel arg
/// (`slot!(Factor)` in a `factor: u32` position) or a launch geometry
/// (`slot!(Grid)` in the grid position).
///
/// A non-resource slot is fundamentally different from the four-state
/// [`SlotState`] resource machine. Its value is `Copy`/`Clone`, has no `cl_mem`,
/// and is **never handed back to the user** — there is no [`Checkout`], no
/// lend/rehome/sever. At execute it is just **read** (cloned) into the launch, NOT
/// taken/emptied, so it naturally persists across replays and is trivially
/// re-readable. That removes the entire `Lent`/`Severed` half of the resource
/// machine: a non-resource slot is only ever `Unbound` (completeness error at
/// [`sync`](DeviceOpExt::sync)) or `Bound(value)` (read on every run).
///
/// Bind idempotency is by **value equality** ([`SlotEq`] over the scalar value /
/// [`LaunchSpec`](crate::LaunchSpec) geometry), not handle identity: `bind(Factor(2))` twice is a
/// no-op, `bind(Factor(9))` over a bound `Factor(2)` is a
/// [`SlotConflict`](Error::SlotConflict) under `Set`, and `mutate_bind` overwrites.
pub enum ScalarSlotState<V> {
    /// **Virgin** — never bound. Resolved (read at execute) it is
    /// [`Error::SlotUnbound`]; both verbs fill it.
    Unbound,
    /// Holds the value, **read (cloned) on every run** — never emptied. A bound
    /// scalar/launch slot persists across replays for free.
    Bound(V),
}

/// The two-state cell shared by a non-resource [`SlotHandle`] (built from
/// `slot!(Tag)` in a scalar / launch position) and its [`ScalarInput::Slot`].
pub type ScalarSlotCell<V> = Arc<Mutex<ScalarSlotState<V>>>;

// ── SlotEq: "same buffer object" for the bind idempotency check ─────────────

/// Pointer-identity equality for the [`bind`](DeviceOpExt::bind) idempotency
/// check: does a new binding name the **same buffer object** as the one already
/// bound?
///
/// This is deliberately NOT byte-equality of contents — `bind` is a set-*once*
/// verb whose "no-op on equal" leg means "you re-handed me the buffer I already
/// have", identified by its OpenCL handle (`cl_mem` for a [`DeviceSlice`], the SVM
/// pointer for a [`MappedSlice`]/[`USMSlice`]). Two distinct buffers that happen to
/// hold equal bytes still *conflict* under `bind` (use
/// [`mutate_bind`](DeviceOpExt::mutate_bind) to swap one for the other).
///
/// Implemented for the buffer families via their
/// [`RecordableBuffer`](crate::record::RecordableBuffer) handle (the same handle the
/// record/replay walk keys on). Required by `bind`/`mutate_bind` on `Tg::Value`;
/// slots are buffer-typed in practice, so only the buffer-family impls exist. A tag
/// whose value type is not a buffer simply can't be `bind`'d (a compile error at the
/// call site, not a silent fallback).
pub trait SlotEq {
    /// `true` iff `self` and `other` are the same underlying buffer object
    /// (handle / pointer identity).
    fn slot_eq(&self, other: &Self) -> bool;
}

/// Implement [`SlotEq`] for a buffer family by comparing its
/// [`RecordableBuffer`](crate::record::RecordableBuffer) handle (`cl_mem` /
/// SVM-pointer identity). Per-family (not a blanket over `RecordableBuffer`) so the
/// trait stays a small, explicit surface tied to the families slots actually carry.
macro_rules! impl_slot_eq {
    ($buf:ident) => {
        impl<E, M> SlotEq for $crate::$buf<E, M>
        where
            M: $crate::MemMode,
        {
            fn slot_eq(&self, other: &Self) -> bool {
                use crate::record::{MemRef, RecordableBuffer};
                match (self.record_handle().mem, other.record_handle().mem) {
                    (MemRef::Buffer(a), MemRef::Buffer(b)) => a == b,
                    (MemRef::Svm(a), MemRef::Svm(b)) => std::ptr::eq(a, b),
                    // Different memory classes can never be the same object.
                    _ => false,
                }
            }
        }
    };
}
impl_slot_eq!(DeviceSlice);
impl_slot_eq!(MappedSlice);
impl_slot_eq!(USMSlice);

// A device SCALAR ([`Scalar<B>`], any memory tier) is a length-1 buffer — its
// `SlotEq` is the backing buffer's handle identity, so it can be a `slot!(Tag)`
// value bound/rebound by buffer identity across all three tiers.
impl<B: SlotEq> SlotEq for Scalar<B> {
    fn slot_eq(&self, other: &Self) -> bool {
        self.inner.slot_eq(other.as_buffer())
    }
}

// A shared (read-only fan-out) `Arc<DeviceSlice>` slot compares its inner buffer.
impl<E, M> SlotEq for std::sync::Arc<DeviceSlice<E, M>>
where
    M: MemMode,
{
    fn slot_eq(&self, other: &Self) -> bool {
        (**self).slot_eq(&**other)
    }
}

// ── SlotEq for non-resource (scalar / launch) slot values ───────────────────
//
// A scalar or launch slot is NOT a buffer — it has no `cl_mem` handle, so its
// `bind` idempotency leg is by **value equality** (`PartialEq`), not handle
// identity. Two equal scalar bindings (`Factor(2)` twice) are an idempotent
// no-op; a different one (`Factor(9)` over a bound `Factor(2)`) is a
// `SlotConflict` under `Set`. This is the value-equality contract the
// non-resource slot path needs — the same `SlotEq` trait, a different (and for a
// `Copy` POD, cheaper) notion of "same".

/// Value-equality [`SlotEq`] for the built-in scalar kernel-arg types: a scalar
/// slot's "same binding" is byte/value equality (it has no buffer handle).
macro_rules! impl_slot_eq_scalar {
    ($($t:ty),* $(,)?) => {
        $(
            impl SlotEq for $t {
                fn slot_eq(&self, other: &Self) -> bool {
                    self == other
                }
            }
        )*
    };
}
impl_slot_eq_scalar!(i8, u8, i16, u16, i32, u32, i64, u64);
// Floats compare bitwise so `NaN`-vs-`NaN` and `-0.0`-vs-`0.0` are decided
// deterministically (a slot binding is "the same bytes I gave you", not IEEE
// numeric equality) — and it sidesteps the `clippy::float_cmp` lint.
impl SlotEq for f32 {
    fn slot_eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}
impl SlotEq for f64 {
    fn slot_eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

/// Value-equality [`SlotEq`] for a launch-geometry slot: two specs are "the same"
/// iff they dispatch identically (same global size, same local size, same
/// dimensionality). [`LaunchSpec`](crate::LaunchSpec) is not `PartialEq`, so this
/// compares its observable geometry.
impl SlotEq for crate::LaunchSpec {
    fn slot_eq(&self, other: &Self) -> bool {
        self.dims() == other.dims()
            && self.global() == other.global()
            && self.local() == other.local()
    }
}

// ── SlotValue: the shared-fill (clone-into-every-cell) capability ───────────

/// Whether a tag's value can be **fanned out** — cloned into *every* matching
/// slot cell by a single [`bind`](DeviceOpExt::bind) — and, if so, how.
///
/// This is the trait that resolves the shared-slot design fork (one tag, many
/// positions, one bind fills ALL) WITHOUT forcing `Clone` on every value type:
///
/// - A **clone-able** value (a `Copy` scalar, a [`LaunchSpec`](crate::LaunchSpec),
///   an `Arc<DeviceSlice>` read-only buffer) reports
///   [`fill_clone`](SlotValue::fill_clone)`= Some(..)`: the [`SlotBinder`] fan-out
///   leg clones it into each matching cell, so `slot!(W)` used at N sites is all
///   filled by one `bind(W(v))`.
/// - A **move-only** value (a bare [`DeviceSlice`] / [`MappedSlice`] /
///   [`USMSlice`]) — which can't be in two cells at once — reports
///   `fill_clone = None`. The binder then keeps its TAKE-ONCE move path: the first
///   matching cell takes the value (single owner), so move-only single-site buffer
///   slots behave **exactly as before** (no `Clone` bound forced on them, no
///   regression). A move-only tag used at >1 site simply fills its first site and
///   leaves the rest unbound — caught at `sync` as [`Error::SlotUnbound`], the
///   honest "you can't share a move-only buffer" diagnostic.
///
/// `bind`/`mutate_bind` require `Tg::Value: SlotValue`. The trait is a small,
/// **explicit** surface (NOT a `Clone` blanket — that would coherence-conflict
/// with the move-only buffer impls, since Rust can't prove a buffer family is
/// `!Clone`). Two helper macros populate it:
/// - `impl_slot_value_clone!` for the fan-out (clone-able) value types — every
///   `Copy` scalar, [`LaunchSpec`](crate::LaunchSpec), and `Arc<DeviceSlice>`.
/// - `impl_slot_value_move!` for the move-only buffer families.
pub trait SlotValue: Sized + Send + 'static {
    /// Produce a boxed clone for filling **one** matching cell, or `None` if this
    /// value is move-only (the binder then moves the single value into the first
    /// matching cell — take-once). `Some(..)` enables the fill-all fan-out.
    fn fill_clone(&self) -> Option<Box<dyn Any + Send>>;
}

/// Implement [`SlotValue`] for a **clone-able** value type — the fan-out path: one
/// `bind` clones the value into every matching cell. Used for `Copy` scalars,
/// [`LaunchSpec`](crate::LaunchSpec), and the read-only shared `Arc<DeviceSlice>`.
macro_rules! impl_slot_value_clone {
    ($($t:ty),* $(,)?) => {
        $(
            impl SlotValue for $t {
                fn fill_clone(&self) -> Option<Box<dyn Any + Send>> {
                    Some(Box::new(::core::clone::Clone::clone(self)))
                }
            }
        )*
    };
}

// The built-in scalar kernel-arg types — clone-able (`Copy`), so a scalar slot
// fans out across sites. A user `#[repr(C)] Copy` scalar opts in the same way it
// opts into `ScalarArg`: a `SlotValue`-aware extension of `scalar_arg!` is the
// eventual sugar; for now a one-line `impl SlotValue` suffices.
impl_slot_value_clone!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);
// The launch geometry as a single fan-out-able value.
impl_slot_value_clone!(crate::LaunchSpec);

/// `Arc<DeviceSlice>` is the read-only shared-buffer value: `Arc::clone` fans the
/// SAME `cl_mem` out to every matching cell (the move-only-buffer-sharing path).
impl<E, M> SlotValue for std::sync::Arc<DeviceSlice<E, M>>
where
    E: Send + Sync + 'static,
    M: MemMode,
{
    fn fill_clone(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(std::sync::Arc::clone(self)))
    }
}

/// Implement [`SlotValue`] for a **move-only** buffer family: it can't be cloned
/// into two cells, so it reports `None` (take-once move into the first matching
/// cell). This is the single-site buffer-slot path, unchanged from before the
/// shared-slot generalisation.
macro_rules! impl_slot_value_move {
    ($buf:ident) => {
        impl<E, M> SlotValue for $crate::$buf<E, M>
        where
            E: Send + 'static,
            M: $crate::MemMode,
        {
            fn fill_clone(&self) -> Option<Box<dyn Any + Send>> {
                // Move-only: no clone. The binder takes the single value once.
                None
            }
        }
    };
}
impl_slot_value_move!(DeviceSlice);
impl_slot_value_move!(MappedSlice);
impl_slot_value_move!(USMSlice);

// A device SCALAR ([`Scalar<B>`], any tier) is a move-only buffer (single
// owner), exactly like a slice.
impl<B: Send + 'static> SlotValue for Scalar<B> {
    fn fill_clone(&self) -> Option<Box<dyn Any + Send>> {
        None
    }
}

// ── Typed slots: per-tag value type compile-time, presence runtime ─────────

/// A compile-time **tag** naming a typed hole in a reusable graph. The tag's
/// [`Key`](Tag::Key) is the identity (matched by [`TypeId`] at bind time); its
/// [`Value`](Tag::Value) is the one buffer type that tag carries — fixed at compile
/// time, so a `slot!(Buf)` and a `Buf(value)` binding are checked against the same
/// type without any turbofish.
///
/// Declared via the [`slots!`](crate::slots) macro, which emits a **source-generic**
/// `pub struct Tag<S = Value>(pub S)` tuple struct (binding is plain tuple-struct
/// construction — `Buf(value)` for a raw value, `Buf(checkout)` for a sever-and-adopt
/// [`Checkout`], no `Fn`/`fn_traits` games) and this trait impl. The struct value is
/// inspected only through [`into_value`](Tag::into_value) (which runs [`IntoBound`]:
/// identity for a raw value, `into_inner`/sever for a `Checkout`); matching keys on
/// `TypeId::of::<Key>()`, which is the SAME for every source `S`.
///
/// Tag *presence* (was every slot bound?) is a **runtime** property, checked at
/// [`sync`](DeviceOpExt::sync) by walking the graph's slot cells — deliberately
/// NOT compile-time set-algebra (the abandoned HList approach). Only the per-tag
/// *value type* is compile-time.
pub trait Tag: Sized + 'static {
    /// The clean, human-readable identifier of this tag for slot-error diagnostics
    /// — just the tag ident (e.g. `"Buf"`), NOT `type_name::<Key>()` (which would
    /// leak the internal `<KeyMarker>` source suffix into user-facing text). The
    /// [`slots!`](crate::slots) macro sets this to `stringify!($name)`. It is a
    /// display string ONLY; tag *matching* is by [`TypeId::of::<Key>()`](Tag::Key),
    /// which is wholly independent of this name.
    const NAME: &'static str;

    /// The buffer type this tag carries. `Send + 'static` so it can flow through
    /// the same [`Cell`]/[`Checkout`] lend-and-return machinery as a concrete
    /// input.
    type Value: Send + 'static;

    /// The **stable matching identity** of this tag, independent of the *source*
    /// the binding was built from. A tag is now a tuple struct that is GENERIC over
    /// its source (`Tag<S>` — `S = Value` for the raw `Tag(buf)` form, `S =
    /// Checkout<Value>` for the new sever-and-adopt `Tag(co)` form), so
    /// `TypeId::of::<Tag<S>>()` would differ between the two source variants and a
    /// `Checkout`-built binding would fail to match a `slot!(Tag)` (built from the
    /// default `Tag<Value>`). `Key` is the per-tag NON-generic marker (the same for
    /// every `S`), so `slot!` and every `bind` form key on `TypeId::of::<Key>()` and
    /// match regardless of source. The [`slots!`](crate::slots) macro emits one
    /// zero-size `Key` marker per tag; it is used ONLY for `TypeId`-based matching.
    /// The human-readable slot diagnostics use [`NAME`](Tag::NAME) instead, so no
    /// internal `<KeyMarker>` suffix ever leaks into user-facing error text.
    type Key: 'static;

    /// Unwrap the tag binding `Tag(source)` to its [`Value`](Tag::Value) (moved),
    /// converting the source: a raw value passes through; a [`Checkout`] is
    /// `into_inner`'d — which **severs** its source home (`Lent → Severed`) and
    /// adopts the buffer into the target slot. The [`slots!`](crate::slots) macro
    /// emits this as `self.0.into_bound()` ([`IntoBound`]); it is the only way
    /// [`bind`](DeviceOpExt::bind) can pull the value out of a generic `Tg` wrapper
    /// (a generic tuple-struct field is not nameable).
    fn into_value(self) -> Self::Value;

    /// The **slot-cell id this tag's source will sever** when [`into_value`](Self::into_value)
    /// runs ([`Arc::as_ptr`] as `usize`), or `None` if it severs nothing. Read-only —
    /// does NOT consume the tag. A raw-value tag returns `None`; a `Checkout`-sourced
    /// tag returns the id of the slot cell its `into_inner` will sever. The
    /// [`slots!`](crate::slots) macro emits this as `self.0.source_cell_id()`
    /// ([`IntoBound::source_cell_id`]).
    ///
    /// It feeds the `call`/`mutate_call` phase-0 probe: the tuple's Checkout-sourced
    /// elements contribute their ids so the probe can recognise a crossed swap (a
    /// `Lent` target the tuple itself will sever) versus an external checkout.
    fn source_cell_id(&self) -> Option<usize>;
}

/// Conversion from a slot **binding source** to the tag's
/// [`Value`](Tag::Value) — the trait that lets a tag constructor accept EITHER a
/// raw buffer/scalar OR a [`Checkout`] over it, with no `.into()` at the call site.
///
/// Two impls:
/// - identity (`impl IntoBound<V> for V`): a raw value binds as before — no source
///   home, nothing to sever.
/// - [`Checkout<V>`](Checkout): `into_bound` calls [`into_inner`](Checkout::into_inner),
///   which **severs** the Checkout's source slot home (`Lent → Severed`) and hands
///   the raw buffer to the target slot to adopt. This is the EXPLICIT-sever path —
///   binding a finished run's output into a (usually different) slot, e.g. the
///   double-buffer swap `g.mutate_call((In(out_co), Out(in_co)))`.
///
/// The two impls do not overlap: `Checkout<X>` matches the blanket as
/// `IntoBound<Checkout<X>>` and the Checkout impl as `IntoBound<X>` — different
/// trait type-params, so coherence is satisfied. No `Clone` is required (the
/// identity impl moves; the Checkout impl moves out of `into_inner`), so move-only
/// buffers are unaffected.
pub trait IntoBound<V> {
    /// Resolve this binding source to the tag's value (severing a `Checkout`).
    fn into_bound(self) -> V;

    /// The **slot-cell id this source will sever** ([`Arc::as_ptr`] as `usize`) when
    /// `into_bound` runs, or `None` if it severs nothing. Read-only — does NOT
    /// consume. A raw value severs nothing (`None`); a [`Checkout`] returns the id of
    /// the slot cell its `into_inner` will sever. Feeds the `call`/`mutate_call`
    /// phase-0 probe's severable-cells set (the crossed-swap recogniser).
    ///
    /// Default `None` covers the identity (raw-value) impl.
    fn source_cell_id(&self) -> Option<usize> {
        None
    }
}

impl<V> IntoBound<V> for V {
    fn into_bound(self) -> V {
        self
    }
}

impl<V: Send> IntoBound<V> for Checkout<V> {
    fn into_bound(self) -> V {
        // Severs the Checkout's source home (Lent → Severed) and returns the buffer
        // for the target slot to adopt — the explicit sever-and-adopt path.
        self.into_inner()
    }

    fn source_cell_id(&self) -> Option<usize> {
        // The slot cell this Checkout's `into_inner` will sever in phase 1.
        self.home_cell_id()
    }
}

/// The shared zero-size marker the [`slots!`](crate::slots) macro plugs into a
/// tag's source slot to form its [`Key`](Tag::Key): `Tag<KeyMarker>` is a distinct
/// `'static` type per tag (the tag ident differs) yet independent of the *binding
/// source* `S`. Keying the bind-match on `TypeId::of::<Tag<KeyMarker>>()` (rather
/// than `TypeId::of::<Tag<S>>()`) is what lets a `Checkout`-built binding
/// (`Tag<Checkout<Value>>`) match a `slot!(Tag)` (`Tag<Value>`). Never constructed.
pub struct KeyMarker;

/// Which verb of the set-binding 2×2 a [`SlotBinder`] carries — the leg that
/// decides what happens when a matching slot is already [`Bound`](SlotState::Bound).
///
/// |               | `Set` ([`bind`](DeviceOpExt::bind))     | `Mutate` ([`mutate_bind`](DeviceOpExt::mutate_bind)) |
/// |---------------|-----------------------------------------|------------------------------------------------------|
/// | `Unbound`     | fill                                    | fill                                                 |
/// | `Bound`, `==` | no-op (idempotent)                      | overwrite                                            |
/// | `Bound`, `!=` | [`Error::SlotConflict`]                 | overwrite                                            |
/// | `Lent`        | [`Error::SlotCheckedOut`]               | [`Error::SlotCheckedOut`]                            |
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BindMode {
    /// `bind` — set-once: idempotent on an equal binding, conflict on a different
    /// one.
    Set,
    /// `mutate_bind` — set/change: overwrite a bound slot (or fill an unbound one).
    Mutate,
}

/// The type-erased [`SlotEq`] comparator a [`SlotBinder`] captures at construction:
/// given the currently-`Bound` value and the new value (both `&dyn Any` over the
/// tag's `Value`), report whether they name the same buffer (handle identity).
type SlotEqFn = Box<dyn Fn(&dyn Any, &dyn Any) -> bool + Send>;

/// The type-erased clone-for-fill a [`SlotBinder`] captures when its value is a
/// **fan-out** ([`SlotValue::fill_clone`]`= Some`) type: given the binder's boxed
/// value (`&dyn Any` over the tag's `Value`), produce a fresh boxed clone to
/// deposit into one matching cell. `None` on the binder means the value is
/// move-only (take-once).
type SlotCloneFn = Box<dyn Fn(&dyn Any) -> Option<Box<dyn Any + Send>> + Send>;

/// A type-erased carrier for one `bind`/`mutate_bind` binding, folded into a
/// graph's slot cells by [`bind_slots`](DeviceOp::bind_slots).
///
/// Carries the tag's [`TypeId`], the [`BindMode`], the boxed value (`Box<dyn Any>`
/// over the tag's `Value`), a type-erased [`SlotEq`] comparator (captured at `bind`
/// time where `Tg::Value: SlotEq` is known), a type-erased [`SlotValue`] clone
/// hook (captured the same way), and an `outcome` slot threaded out of the fold
/// (the verb-2×2 verdict). The tag's [`NAME`](Tag::NAME) for diagnostics lives on
/// the matched [`Input::Slot`] itself (the error sites read it there).
///
/// ## Fill-all vs take-once — the shared-slot mechanism
///
/// How the binding lands depends on whether the value can be cloned
/// ([`SlotValue::fill_clone`], captured into `clone`):
///
/// - **Fan-out (clone-able value):** a `Copy` scalar, [`LaunchSpec`](crate::LaunchSpec), or
///   `Arc<DeviceSlice>`. The binder clones the value into **every** matching cell
///   and is NEVER consumed — so `slot!(W)` used at N positions is all set by one
///   `bind(W(v))`. The walk runs to completion (no early stop on
///   [`is_consumed`](SlotBinder::is_consumed)).
/// - **Take-once (move-only value):** a bare [`DeviceSlice`] / [`MappedSlice`] /
///   [`USMSlice`]. A buffer is single-owner, so the FIRST matching cell MOVES the
///   value out and the binder is consumed — identical to the pre-generalisation
///   behaviour, so single-site move-only buffer slots are unchanged. (A move-only
///   tag at >1 site fills only its first; the rest stay unbound → `SlotUnbound` at
///   sync — the honest "you can't share a move-only buffer" diagnostic.)
pub struct SlotBinder {
    id: TypeId,
    mode: BindMode,
    /// The boxed value (`Box<dyn Any + Send>` over the tag's `Value`). For a
    /// **fan-out** binding it stays `Some` for the whole walk (each cell gets a
    /// clone via [`clone`](Self::clone)); for a **move-only** binding it is
    /// `take()`n into the first matching cell (then `None` = consumed).
    value: Option<Box<dyn Any + Send>>,
    /// Pointer-identity (buffer) OR value (scalar/launch) comparison for the `bind`
    /// idempotency leg: given the currently-`Bound` value and the new value (both
    /// as `&dyn Any` over `Tg::Value`), reports whether they are the "same" binding
    /// ([`SlotEq`]). Captured at construction so the generic, `SlotEq`-free
    /// [`try_bind_slot`](Input::try_bind_slot) can invoke it.
    eq: SlotEqFn,
    /// The fan-out clone hook ([`SlotValue::fill_clone`]). Returns `Some(box)` to
    /// fill a cell WITHOUT consuming the binder (clone-able → fill-all), or `None`
    /// to signal move-only (the binder then `take`s its single value once).
    clone: SlotCloneFn,
    /// The verdict of applying this binder, threaded out of the type-erased fold.
    /// `Ok(())` until a matching slot records a conflict / checked-out error.
    outcome: Result<()>,
    /// How many cells with this tag's `id` the walk encountered — the "is this tag
    /// PRESENT in the graph?" counter. Incremented by both
    /// [`try_bind_slot`](Input::try_bind_slot) impls whenever a cell's `id` matches
    /// the binder's, REGARDLESS of the per-state outcome (fill, idempotent no-op,
    /// conflict, sever-reject, or a same-tag cell visited after a move-only binder
    /// was already consumed). A zero count after the fold therefore means "no such
    /// tag here" — a typo'd / unused tag — which [`fold_bind`](DeviceOpExt::fold_bind)
    /// turns into [`Error::SlotNoSuchTag`] (the AT-LEAST-ONE rule; fan-out
    /// legitimately matches N>=1 sites, so only zero is the error). A
    /// conflict/sever still produces its own error via [`outcome`](Self::outcome),
    /// so counting those as matches never masks them.
    matched: usize,
    /// **Probe (read-only) mode.** When `true`, [`try_bind_slot`](Input::try_bind_slot)
    /// does NOT fill / take / replace any cell — it only INSPECTS state and records
    /// [`matched`](Self::matched) + [`outcome`](Self::outcome), so the whole
    /// `bind_slots` walk becomes a dry run. This is the phase-0 pre-check that makes
    /// [`call`](DeviceOpExt::call) / [`mutate_call`](DeviceOpExt::mutate_call)
    /// all-or-nothing: it proves every tuple element CAN bind (present + a state the
    /// verb handles) BEFORE any `into_value` severs a `Checkout` source. A probe
    /// binder carries NO value (its `value`/`eq`/`clone` are inert dummies).
    ///
    /// A probe binder reports [`is_fanout`](Self::is_fanout)`== true` so the
    /// [`AndThen`]/bundle walks visit EVERY matching cell (a valueless binder would
    /// otherwise read "consumed" and stop after the first subtree, missing a later
    /// checked-out cell).
    probe: bool,
    /// The set of slot-cell identities ([`Arc::as_ptr`] as `usize`) that phase 1 WILL
    /// sever — one per [`Checkout`]-sourced element in the SAME `call`/`mutate_call`
    /// tuple (a raw-value element contributes nothing). Consulted ONLY in
    /// [`probe`](Self::probe) mode: a `Lent` target whose cell id is in this set is
    /// held by a tuple Checkout, so phase 1 turns it `Lent → Severed` before the
    /// fold — the crossed double-buffer swap. A `Lent` target NOT in this set is
    /// held by an EXTERNAL live Checkout and stays `Lent` at fold time
    /// ([`Error::SlotCheckedOut`]). Empty for every non-probe binder.
    severable_cells: Vec<usize>,
    /// **The pipe to install for a pipe-fed bind** ([`feed`](SlotBinder::feed)).
    /// When `Some`, this binder installs [`SlotState::FedByPipe`] into every matching
    /// cell (draining the boxed `Pipe<Tg::Value>` via `downcast_ref` +
    /// [`Pipe::clone`]) INSTEAD of a value — the bind-slot-to-pipe path. It is a
    /// fan-out install ([`is_fanout`](Self::is_fanout) reports `true`): the pipe is
    /// cloned into EVERY site the tag appears (e.g. a `UIn` used at both a laplacian
    /// and a combine position). `None` for every ordinary value / probe binder. Its
    /// presence is mutually exclusive with a `value` (a feed binder carries no
    /// value; its `eq`/`clone` are inert dummies).
    feed_pipe: Option<Box<dyn Any + Send>>,
    /// **Infallible-apply marker.** `true` ONLY for a binder built by the consuming,
    /// infallible [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call) path (via
    /// the deferred `bind`/`feed` helpers behind [`CallArg::apply`]). When set, the
    /// [`try_bind_slot`](Input::try_bind_slot) walk CAPTURES a
    /// [`captured_sink`](Self::captured_sink) so a bind error can be RECORDED into a
    /// graph-reachable [`DeferredErrors`] sink instead of dropped — see the sink type
    /// docs. `false` for every fluent-verb / probe binder (they surface errors
    /// eagerly and never touch a sink).
    deferred: bool,
    /// The [`DeferredErrors`] sink captured from the FIRST slot cell the walk visits
    /// (only when [`deferred`](Self::deferred) is set). After the walk, the infallible
    /// apply path pushes any recorded error here (`SlotConflict`/`SlotSevered`/
    /// `SlotCheckedOut` from [`outcome`](Self::outcome), or `SlotNoSuchTag` when
    /// [`matched`](Self::matched)`== 0`). `None` for a graph with no slots (nothing to
    /// bind — no error is possible) and for every non-deferred binder.
    captured_sink: Option<DeferredErrors>,
}

impl SlotBinder {
    /// Build a binder for tag `Tg` carrying `value` (moved), in [`BindMode`]
    /// `mode`. Use via [`DeviceOpExt::bind`] / [`DeviceOpExt::mutate_bind`].
    ///
    /// `Tg::Value: SlotEq` so the idempotency comparator can be captured here, and
    /// `Tg::Value: SlotValue` so the fan-out clone hook can be too (both are
    /// erased; the generic [`try_bind_slot`](Input::try_bind_slot) the macro calls
    /// carries neither bound).
    pub fn new<Tg: Tag>(value: Tg::Value, mode: BindMode) -> Self
    where
        Tg::Value: SlotEq + SlotValue,
    {
        SlotBinder {
            id: TypeId::of::<Tg::Key>(),
            mode,
            value: Some(Box::new(value)),
            eq: Box::new(|bound: &dyn Any, new: &dyn Any| {
                match (
                    bound.downcast_ref::<Tg::Value>(),
                    new.downcast_ref::<Tg::Value>(),
                ) {
                    (Some(a), Some(b)) => a.slot_eq(b),
                    // Downcast can't fail (TypeId already matched), but if it ever
                    // did, treat as "different" → a Set verb conflicts rather than
                    // silently no-ops.
                    _ => false,
                }
            }),
            clone: Box::new(|v: &dyn Any| {
                // `v` is the binder's `Box<dyn Any>` over `Tg::Value`. Clone-able
                // values (scalars/launch/Arc) yield `Some(box)` (fan-out fill);
                // move-only buffers yield `None` (take-once).
                v.downcast_ref::<Tg::Value>()
                    .and_then(|val| <Tg::Value as SlotValue>::fill_clone(val))
            }),
            outcome: Ok(()),
            matched: 0,
            probe: false,
            severable_cells: Vec::new(),
            feed_pipe: None,
            deferred: false,
            captured_sink: None,
        }
    }

    /// **Build a pipe-feeding binder** for tag `Tg`, carrying `pipe` (boxed,
    /// type-erased). The [`bind_slots`](DeviceOp::bind_slots) walk deposits
    /// [`SlotState::FedByPipe`] into EVERY matching cell (fan-out: the pipe is
    /// [`Pipe::clone`]d per cell). No `SlotEq`/`SlotValue` bound is needed — the
    /// install is unconditional and needs neither a value comparator nor a fan-out
    /// clone hook (the `eq`/`clone` fields are inert dummies, exactly as for a
    /// [`probe`](Self::probe)). Used by [`DeviceOpExt::feed_deferred`].
    fn feed<Tg: Tag>(pipe: Pipe<Tg::Value>) -> Self {
        SlotBinder {
            id: TypeId::of::<Tg::Key>(),
            // `Mutate` so a re-`feed` (a graph fed a second time) re-installs cleanly;
            // the actual install ignores `mode` (it is unconditional over state), but
            // `Mutate` is the honest label for "overwrite the slot's source".
            mode: BindMode::Mutate,
            value: None,
            eq: Box::new(|_, _| false),
            clone: Box::new(|_| None),
            outcome: Ok(()),
            matched: 0,
            probe: false,
            severable_cells: Vec::new(),
            feed_pipe: Some(Box::new(pipe)),
            deferred: false,
            captured_sink: None,
        }
    }

    /// Build a **read-only probe** binder for tag `Tg` in [`BindMode`] `mode`,
    /// carrying no value — the phase-0 dry run of [`call`](DeviceOpExt::call) /
    /// [`mutate_call`](DeviceOpExt::mutate_call). `severable_cells` is the set of
    /// slot-cell ids ([`Arc::as_ptr`] as `usize`) that phase 1 will sever (one per
    /// `Checkout`-sourced tuple element); a `Lent` target in that set is a crossed
    /// swap (OK), one outside it is an external checkout ([`Error::SlotCheckedOut`]).
    ///
    /// The `value`/`eq`/`clone` fields are inert dummies: probe mode never calls
    /// `provide`/`fill_clone`/`eq` (it inspects state only), so they are never read.
    /// No `SlotEq`/`SlotValue` bound is needed here — the read-only probe compares
    /// nothing and clones nothing.
    fn probe<Tg: Tag>(mode: BindMode, severable_cells: Vec<usize>) -> Self {
        SlotBinder {
            id: TypeId::of::<Tg::Key>(),
            mode,
            // Inert: probe mode inspects state only; these are never invoked.
            value: None,
            eq: Box::new(|_, _| false),
            clone: Box::new(|_| None),
            outcome: Ok(()),
            matched: 0,
            probe: true,
            severable_cells,
            feed_pipe: None,
            deferred: false,
            captured_sink: None,
        }
    }

    /// Whether this binder is a read-only [`probe`](Self::probe) (phase-0 dry run).
    fn is_probe(&self) -> bool {
        self.probe
    }

    /// Predict the outcome of the verb 2×2 on a target that is currently
    /// [`Lent`](SlotState::Lent), for a probe over cell `cell_id`. A `Lent` cell held
    /// by a tuple `Checkout` (`cell_id ∈ severable_cells`) becomes
    /// [`Severed`](SlotState::Severed) in phase 1, so the verb then sees `Severed`
    /// ([`Set`](BindMode::Set) → [`SlotSevered`](Error::SlotSevered);
    /// [`Mutate`](BindMode::Mutate) → re-arm, OK). A `Lent` cell held by an EXTERNAL
    /// checkout stays `Lent` at fold time → [`SlotCheckedOut`](Error::SlotCheckedOut)
    /// for both verbs. Returns the [`Error`] to record, or `Ok(())` if the verb will
    /// succeed post-sever. `name` is the tag's display name for the error.
    fn probe_lent(&self, cell_id: usize, name: &'static str) -> Result<()> {
        if self.severable_cells.contains(&cell_id) {
            // Phase 1 will sever this cell (Lent → Severed) — a crossed swap.
            match self.mode {
                // Mutate re-arms a Severed slot: OK.
                BindMode::Mutate => Ok(()),
                // Set rejects a Severed slot (re-arming is a change, not a first
                // declaration) — the honest post-sever verdict.
                BindMode::Set => Err(Error::SlotSevered(name)),
            }
        } else {
            // Held by an external live Checkout — stays Lent at fold; both verbs
            // hard-error rather than clobber a value in the caller's hands.
            Err(Error::SlotCheckedOut(name))
        }
    }

    /// Produce a boxed clone of this binder's value for filling **one** cell, if the
    /// value is a fan-out (clone-able) type. `None` ⇒ move-only (take-once path).
    /// The binder stays armed either way; the caller decides whether to also
    /// [`take`](Self::take_value) the value (move path).
    fn fill_clone(&self) -> Option<Box<dyn Any + Send>> {
        self.value.as_deref().and_then(|v| (self.clone)(v))
    }

    /// Take the binder's value out (the move-only, take-once path), marking it
    /// consumed. Returns `None` if already taken.
    fn take_value(&mut self) -> Option<Box<dyn Any + Send>> {
        self.value.take()
    }

    /// Whether this binder fans out (clone into every matching cell, never
    /// consumed) rather than moving once. Drives the `bind_slots` walk: a fan-out
    /// binder must visit ALL cells (no early `is_consumed` stop).
    pub fn is_fanout(&self) -> bool {
        // A probe visits EVERY matching cell (it must catch a checked-out cell that
        // appears AFTER an OK one), so it never lets the `AndThen`/bundle walk stop
        // early — it behaves like a fan-out even though it carries no value.
        //
        // A pipe-feed binder also fans out — one `Tag(pipe)` installs
        // `FedByPipe` at EVERY site the tag appears (e.g. a `UIn` at both its
        // laplacian and combine positions), cloning the pipe per cell.
        self.probe || self.feed_pipe.is_some() || self.fill_clone().is_some()
    }

    /// Whether the binding has already been deposited into (or rejected by) a
    /// matching slot cell — only ever `true` for a **move-only** (take-once)
    /// binding once its single value has been moved out. A **fan-out** binding is
    /// never consumed (it clones into every cell), so this stays `false`; walks
    /// must gate early-stop on `!is_fanout() && is_consumed()`.
    pub fn is_consumed(&self) -> bool {
        self.value.is_none()
    }

    /// Take the verb-2×2 verdict out of the binder after the fold. `Ok(())` if the
    /// binding landed (or was a clean idempotent no-op), else the
    /// [`SlotConflict`](Error::SlotConflict) / [`SlotCheckedOut`](Error::SlotCheckedOut)
    /// / [`SlotSevered`](Error::SlotSevered) that a matching slot recorded.
    pub fn outcome(&self) -> Result<()> {
        match &self.outcome {
            Ok(()) => Ok(()),
            // Every error arm is `&'static str`-carrying, so a cheap copy.
            Err(Error::SlotConflict(n)) => Err(Error::SlotConflict(n)),
            Err(Error::SlotCheckedOut(n)) => Err(Error::SlotCheckedOut(n)),
            Err(Error::SlotSevered(n)) => Err(Error::SlotSevered(n)),
            // No other error is ever recorded into a binder.
            Err(_) => unreachable!("SlotBinder only records slot-bind errors"),
        }
    }

    /// How many cells carrying this binder's tag the fold encountered (see
    /// [`matched`](Self::matched)). Zero ⇒ the tag is not present in this graph;
    /// [`fold_bind`](DeviceOpExt::fold_bind) maps that to
    /// [`Error::SlotNoSuchTag`]. Any positive count satisfies the AT-LEAST-ONE
    /// rule (a fan-out tag legitimately matches every site it appears at).
    pub fn matched(&self) -> usize {
        self.matched
    }

    /// Mark this binder as belonging to the INFALLIBLE, consuming
    /// [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call) apply path, so the
    /// [`bind_slots`](DeviceOp::bind_slots) walk captures a
    /// [`captured_sink`](Self::captured_sink) (see [`DeferredErrors`]). Only the
    /// deferred `bind`/`feed` helpers behind [`CallArg::apply`] call this.
    fn mark_deferred(&mut self) {
        self.deferred = true;
    }

    /// After a deferred-apply walk, RECORD any bind error into the graph-reachable
    /// [`captured_sink`](Self::captured_sink) (record-don't-drop) rather than dropping
    /// it. Pushes the [`outcome`](Self::outcome) error (`SlotConflict`/`SlotSevered`/
    /// `SlotCheckedOut`), OR — when the tag matched NO cell — a
    /// [`SlotNoSuchTag`](Error::SlotNoSuchTag). A clean bind (matched ≥ 1, `Ok`
    /// outcome) records nothing, so the sink stays empty and the reuse path is
    /// unchanged. `name` is the tag's display name for the `SlotNoSuchTag` case (which
    /// has no matching cell to read a name from). No-op if no sink was captured (a
    /// graph with no slots — nothing could have been bound).
    fn record_deferred(&self, name: &'static str) {
        let err = match self.outcome() {
            Err(e) => Some(e),
            Ok(()) if self.matched == 0 => Some(Error::SlotNoSuchTag(name)),
            Ok(()) => None,
        };
        if let (Some(err), Some(sink)) = (err, &self.captured_sink) {
            sink.lock().unwrap().push(err);
        }
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
    /// Deposit `value` into the origin cell (consuming the boxed home) — the
    /// re-arm path, fired by [`Checkout`] **drop**. For a concrete cell this fills
    /// it; for a [`slot`](SlotState) cell this is the `Lent → Bound(value)`
    /// transition.
    fn rehome(self: Box<Self>, value: Out);

    /// **Sever** the return without depositing — fired by
    /// [`Checkout::into_inner`], where the caller KEEPS the value. For a concrete
    /// [`Cell`] this is a no-op (its cell is already empty, which correctly reads
    /// "busy / re-allocate next run"). For a [`slot`](SlotState) cell this is the
    /// `Lent → Severed` transition: the value is gone for good, so the slot must
    /// read empty (not stuck in `Lent`) — otherwise a later `bind`/`mutate_bind`
    /// would wrongly see [`Error::SlotCheckedOut`]. It lands in
    /// [`Severed`](SlotState::Severed), NOT [`Unbound`](SlotState::Unbound): a
    /// severed slot was once bound, so a set-once `bind` must reject it
    /// ([`Error::SlotSevered`]) rather than silently re-fill — only `mutate_bind`
    /// may re-arm it.
    ///
    /// Default: no-op (the concrete-cell behaviour). Slots override it.
    fn sever(self: Box<Self>) {}

    /// The **stable identity of the slot cell** this home would sever, as
    /// [`Arc::as_ptr`] cast to `usize` — or `None` if severing this home does not
    /// empty a slot cell (a concrete [`Cell`] home, whose `sever` is a no-op).
    ///
    /// Used by [`Checkout`]'s home-id accessor to feed the `call`/`mutate_call`
    /// phase-0 probe's severable-cells set: the probe recognises a crossed
    /// double-buffer swap by matching a `Lent` target slot's cell id against the ids
    /// of the tuple Checkouts that will sever it in phase 1. Only the slot-cell home
    /// returns `Some`.
    ///
    /// Default: `None` (concrete-cell homes never sever a slot).
    fn home_cell_id(&self) -> Option<usize> {
        None
    }
}

/// A boxed, type-erased return home for an output of type `Out` — the home
/// channel's payload (`None` = nothing to return). Aliased so the `Pipe` /
/// `Input` / `Checkout` signatures stay readable.
pub type BoxedHome<Out> = Box<dyn Rehome<Out>>;

/// Return a buffer CONSUMED by an op (read into a host `Vec`, never carried onward
/// in a pipe) to its home, or release it if homeless. The general
/// [`PipePayload`] drop handles values that flow THROUGH a pipe; a consuming op
/// (e.g. [`download`]) instead splits the buffer from its output value, so it
/// rehomes the buffer directly here. `home == None` ⇒ a minted buffer with no
/// origin cell, which simply drops (releasing its `cl_mem`).
///
/// `pub` + `#[doc(hidden)]`: the `#[kernel]` proc-macro emits
/// `::claspr::rehome_consumed(...)` inside the *user's* crate for multi-output
/// kernels' `reclaim_undelivered`, so it must be reachable cross-crate. Not part
/// of the stable surface.
#[doc(hidden)]
pub fn rehome_consumed<T>(buf: T, home: Option<BoxedHome<T>>) {
    if let Some(home) = home {
        home.rehome(buf);
    }
    // else: homeless → `buf` drops here, releasing it.
}

/// Identity rehome: an output returns to a cell of its own type (the in-place
/// case — fill/scale/kernel-buffer-arg/copy's same-typed sides). This is the
/// behaviour the old `Option<Cell<T>>` home had, now expressed through the trait.
impl<T: Send> Rehome<T> for Cell<T> {
    fn rehome(self: Box<Self>, value: T) {
        *self.lock().unwrap() = Some(value);
    }
    // `sever` keeps the default no-op: a concrete cell stays empty after
    // `into_inner`, which already reads correctly (re-allocate / busy next run).
}

/// The return home for a **slot** input — distinct from the concrete
/// [`Cell<T>: Rehome`] home because the slot's four-state cell needs two distinct
/// exits.
///
/// A slot cell is the four-state [`SlotCell<T>`]; while a run is in flight it sits
/// in [`Lent`](SlotState::Lent). This home owns a clone of that cell and resolves
/// the two terminal transitions:
/// - [`rehome`](Rehome::rehome) (Checkout drop OR undelivered/consumed buffer,
///   re-arm): `Lent → Bound(value)`. Under "homeless is never legitimate", a slot
///   buffer that is produced and not handed out — including the
///   [`download`]-consumed case (the device buffer is returned even though the
///   output is a host `Vec`) — is RETURNED to its slot with a stable handle, NOT
///   severed. This rehome fires from the general [`PipePayload`] drop, from
///   [`Checkout`] drop, or directly from a consuming op via
///   [`rehome_consumed`] — one and only one, because [`BoxedHome`] is not `Clone`.
/// - [`sever`](Rehome::sever) (`into_inner`, keep the value): `Lent → Severed`.
///   The ONLY path that empties a slot cell. It lands in
///   [`Severed`](SlotState::Severed) (NOT [`Unbound`](SlotState::Unbound)): the
///   slot was once bound and the caller took its value, so a set-once `bind` must
///   reject it ([`Error::SlotSevered`]) rather than silently re-fill — only
///   `mutate_bind` re-arms it.
///
/// (A concrete `Cell` needs no `sever` override: its empty state is overloaded, so
/// its `sever` is a no-op. The slot's third/fourth states are what distinguish
/// "checked out / re-armable" (`Lent`) from "severed / value taken" (`Severed`)
/// from "virgin / never bound" (`Unbound`) — and there is now no
/// drop-without-firing fallback: the general payload-drop rule rehomes any
/// undelivered slot buffer, so a slot is never left stuck in `Lent`.)
struct SlotHome<T> {
    cell: SlotCell<T>,
}

impl<T: Send> Rehome<T> for SlotHome<T> {
    fn rehome(self: Box<Self>, value: T) {
        // Re-arm: the lent buffer (possibly transformed in place, or consumed into
        // a host Vec by a download whose home channel still carries it back) comes
        // back to its slot with a STABLE handle.
        *self.cell.lock().unwrap() = SlotState::Bound(value);
    }
    fn sever(self: Box<Self>) {
        // The caller kept the value (`into_inner`); the slot is empty but NOT
        // virgin — it lands in `Severed`, so a set-once `bind` rejects it
        // (`Error::SlotSevered`) and only `mutate_bind` may re-arm it.
        *self.cell.lock().unwrap() = SlotState::Severed;
    }
    fn home_cell_id(&self) -> Option<usize> {
        // The identity of THIS slot's cell — matched by the `call`/`mutate_call`
        // probe against a `Lent` target to recognise the crossed swap.
        Some(Arc::as_ptr(&self.cell) as usize)
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
///
/// ## "Homeless is never legitimate": rehome on undelivered drop
///
/// A payload OWNS its home, and its [`Drop`] is the general enforcement of the
/// invariant: if a payload still holds BOTH a value AND a home when it drops —
/// i.e. the value was produced mid-graph but never handed onward (no downstream
/// op moved it out, no terminal built a [`Checkout`] from it) — the value is a
/// homed buffer about to be released. That is exactly what the invariant
/// forbids: the `Drop` instead [`rehome`](Rehome::rehome)s it to its origin
/// cell, so a reused graph re-runs with the SAME backing handle.
///
/// Both `value` and `home` are `Option` so the move-out drains
/// ([`take_home`](Pipe::take_home) for the value-AND-home transfer; an upstream
/// in-place op forwarding via [`put_home`](Pipe::put_home)) can pull them out
/// in place, leaving the emptied payload to drop as a harmless no-op. The
/// **disarm signal is "home moved out"**: once a payload's home is `None` (it
/// was forwarded into the next payload, or into a `Checkout`), its `Drop` does
/// nothing — the new owner is now responsible for the eventual rehome. Because
/// [`BoxedHome`] is not `Clone`, the home lives in exactly one place at a time,
/// so the rehome fires from exactly one drop — never double.
struct PipePayload<T> {
    value: Option<T>,
    deps: Deps,
    home: Option<BoxedHome<T>>,
}

impl<T> Drop for PipePayload<T> {
    fn drop(&mut self) {
        // The invariant's catch-all: a value produced mid-graph but never
        // delivered (no downstream `take_home`, no terminal `Checkout`) and still
        // carrying a home must be RETURNED to its origin cell, not released. If the
        // home was already moved out (forwarded downstream / into a Checkout), this
        // is a no-op — the new owner now owns the rehome obligation.
        if let (Some(value), Some(home)) = (self.value.take(), self.home.take()) {
            home.rehome(value);
        }
        // else: a homeless payload (minted, nothing to return) or one whose value
        // and/or home were already drained — nothing to rehome.
    }
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
        // Overwriting the cell drops any previous payload first. That previous
        // payload's `Drop` fires the rehome IF it still held an undelivered
        // value+home — the correct behaviour for a pipe re-deposited without its
        // prior value having been drained (it shouldn't happen on the live paths,
        // but the invariant holds regardless). The freshly stored payload arms the
        // new (value, home) pair for the next drain or its own drop.
        *self.cell.lock().unwrap() = Some(PipePayload {
            value: Some(v),
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
        // Move the whole payload out of the cell, then drain its `value` + `home`
        // in place. `PipePayload` has a `Drop`, so it cannot be destructured
        // by-move; `.take()` on the two `Option` fields leaves the husk to drop as
        // a no-op (both now `None`). The home is MOVED to the caller — the payload
        // no longer owns it, so its `Drop` won't re-fire the rehome (single owner,
        // `BoxedHome: !Clone`). `deps` is `Default`, swapped out cheaply.
        self.cell.lock().unwrap().take().map(|mut p| {
            let value = p
                .value
                .take()
                .expect("PipePayload drained twice — internal bug");
            (value, std::mem::take(&mut p.deps), p.home.take())
        })
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

    fn output_pipe(&self) -> Option<Pipe<T>> {
        // The pipe is its OWN output storage — no separate `out`. The producer
        // already (or will) deposit here; we alias it.
        Some(self.clone())
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
    /// The `cell` starts [`Unbound`](SlotState::Unbound); a later
    /// [`bind`](DeviceOpExt::bind)`(Tag(value))` / [`mutate_bind`](DeviceOpExt::mutate_bind)
    /// walks the graph and deposits a matching value (see
    /// [`bind_slots`](DeviceOp::bind_slots)), moving it to [`Bound`](SlotState::Bound).
    ///
    /// Unlike a [`Concrete`](Input::Concrete) cell (a bare `Option`, full/lent), a
    /// slot is the **four-state** [`SlotCell`] so the verb 2×2 can tell
    /// [`Unbound`](SlotState::Unbound) ("virgin / never filled") and
    /// [`Severed`](SlotState::Severed) ("value taken via `into_inner`") from
    /// [`Lent`](SlotState::Lent) ("checked out, run in flight"). It still lends +
    /// re-arms like a concrete cell on the happy path: lend takes
    /// `Bound → Lent`, the run's `Checkout` drop returns `Lent → Bound`, so a
    /// bound graph re-runs. Resolved while [`Unbound`](SlotState::Unbound) it is
    /// [`Error::SlotUnbound`]; resolved while [`Lent`](SlotState::Lent) the graph
    /// is busy on that slot (also `SlotUnbound`, message covers both).
    ///
    /// `id` is `TypeId::of::<Tag::Key>()` (matched against a [`SlotBinder`]); `name`
    /// is [`Tag::NAME`] (the clean tag ident), carried for the unbound-slot
    /// diagnostic AND the conflict / checked-out bind errors.
    Slot {
        /// `TypeId::of::<Tag::Key>()` — the bind-matching key.
        id: TypeId,
        /// [`Tag::NAME`] (the clean tag ident) — for the slot diagnostics (unbound /
        /// conflict / checked-out).
        name: &'static str,
        /// Tri-state: [`Unbound`](SlotState::Unbound) until a matching bind
        /// deposits the value ([`Bound`](SlotState::Bound)); [`Lent`](SlotState::Lent)
        /// while a run holds it.
        cell: SlotCell<T>,
        /// The [`DeferredErrors`] sink for the INFALLIBLE, consuming
        /// [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call) apply path: a bind
        /// error the consuming, infallible path cannot return is RECORDED here (record-don't-drop)
        /// and DRAINED by [`check_ready`](Input::check_ready) at `sync`, FIRST, before
        /// any enqueue. Starts empty; stays empty for a graph the deferred path never
        /// errs on (so the reuse path is unchanged). See [`DeferredErrors`].
        sink: DeferredErrors,
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

    /// Lend the value out of a **concrete** [`Cell`]: take it (the cell stays
    /// empty for the run, re-armed on `Checkout` drop), build its identity home
    /// (`Cell<T>: Rehome<T>`), and thread the host-seam start gate if `ec` has one.
    /// `empty_err` is the error to return when the cell is already empty (a graph
    /// `sync`'d while a previous `Checkout` is still alive — busy).
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
        Self::thread_start_gate(v, ec, home)
    }

    /// Lend the value out of a **slot** [`SlotCell`]: the multi-state analogue of
    /// [`lend_from_cell`](Self::lend_from_cell). Transitions
    /// [`Bound(v) → Lent`](SlotState::Lent) and hands `v` to the run; any empty
    /// state ([`Unbound`](SlotState::Unbound) "never bound" or
    /// [`Severed`](SlotState::Severed) "value taken via `into_inner`") is
    /// [`Error::SlotUnbound`] (nothing to lend); a [`Lent`](SlotState::Lent) slot is
    /// the graph-busy case — a previous run's `Checkout` still holds the buffer —
    /// and also surfaces as `SlotUnbound` (whose message covers all three). The
    /// home is a [`SlotHome`] so the run's `Checkout` drop re-arms `Lent → Bound`
    /// and `into_inner` severs `Lent → Severed`.
    fn lend_slot(
        cell: &SlotCell<T>,
        ec: &ExecutionContext<'_>,
        name: &'static str,
    ) -> Result<(T, Deps, Option<BoxedHome<T>>)>
    where
        T: Send + 'static,
    {
        // A pipe-fed slot behaves like `Input::Pipe` — DRAIN the upstream pipe
        // (which the producer filled earlier THIS run) via `take_home`, LEAVING the
        // `FedByPipe` variant in place so the slot re-arms for the next replay (the
        // upstream refills the pipe each run). The value + deps + home come straight
        // from the pipe payload, exactly as the `Input::Pipe` arm of `resolve_home`
        // — no start gate is threaded here (the upstream producer already gated its
        // own enqueue, and its events flow onward as the drained `Deps`).
        {
            let guard = cell.lock().unwrap();
            if let SlotState::FedByPipe(pipe) = &*guard {
                let pipe = pipe.clone();
                drop(guard);
                return pipe.take_home().ok_or(Error::NotSupported(
                    "eager graph: upstream pipe for a FedByPipe slot was not filled \
                     before downstream ran — internal ordering bug",
                ));
            }
        }
        // `Bound(v) → Lent`, take `v`; anything else (Unbound / Severed / already
        // Lent) is the "nothing to lend" error.
        let v = {
            let mut guard = cell.lock().unwrap();
            match std::mem::replace(&mut *guard, SlotState::Lent) {
                SlotState::Bound(v) => v,
                // Restore the prior state before erroring (we tentatively wrote
                // `Lent` above; put it back so the slot is unchanged on failure).
                // `Unbound`/`Severed`/`Lent`/`FedByPipe` (the last handled above)
                // all surface as `SlotUnbound`.
                other => {
                    *guard = other;
                    return Err(Error::SlotUnbound(name));
                }
            }
        };
        let home: Option<BoxedHome<T>> = Some(Box::new(SlotHome {
            cell: Arc::clone(cell),
        }));
        Self::thread_start_gate(v, ec, home)
    }

    /// Attach the host-seam **start gate** to a freshly-lent value's wait-list (if
    /// `ec` carries one), shared by the concrete and slot lend paths. Without a
    /// gate the value is an entry leaf with an empty wait-list; with one its enqueue
    /// waits on the gate so the whole graph commits before any of it runs.
    fn thread_start_gate(
        v: T,
        ec: &ExecutionContext<'_>,
        home: Option<BoxedHome<T>>,
    ) -> Result<(T, Deps, Option<BoxedHome<T>>)>
    where
        T: Send + 'static,
    {
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
            // A typed slot: once `Bound` it lends like a concrete cell — `Bound →
            // Lent`, and the run's `Checkout` drop returns `Lent → Bound`, so a
            // bound graph re-runs. `Unbound` is the completeness error; `Lent`
            // means a previous run's `Checkout` still holds the buffer (busy).
            // Both surface as `SlotUnbound` (its message covers the two).
            Input::Slot { name, cell, .. } => Self::lend_slot(cell, ec, name),
            Input::Pipe(p) => p.take_home().ok_or(Error::NotSupported(
                "eager graph: upstream pipe was not filled before downstream ran \
                 — internal ordering bug",
            )),
        }
    }

    /// **Read-only** pre-flight check: would [`resolve_home`](Self::resolve_home)
    /// succeed on this input *right now*, WITHOUT lending / taking / mutating
    /// anything? Returns the SAME [`Error`] variant + message `resolve_home` would
    /// produce for an unsatisfiable input, or `Ok(())` if it is ready.
    ///
    /// This is the per-input half of the [`check_ready`](DeviceOp::check_ready)
    /// atomicity guarantee: a graph terminal walks EVERY input's `check_ready`
    /// before enqueuing any device work, so a failed `sync` leaves the graph
    /// unchanged + re-runnable (no buffer left `Lent`, no command enqueued). It
    /// must check the SAME conditions `resolve_home` errors on — and ONLY those —
    /// so the early catch is identical to the (retained) execute-time backstop.
    ///
    /// - [`Concrete`](Input::Concrete): error iff the cell is empty (already lent
    ///   to a live `Checkout` — the graph is busy). Inspect via `is_none()`; do NOT
    ///   `take()`.
    /// - [`Slot`](Input::Slot): OK iff [`Bound`](SlotState::Bound); any empty state
    ///   ([`Unbound`](SlotState::Unbound)/[`Severed`](SlotState::Severed)/[`Lent`](SlotState::Lent))
    ///   is [`Error::SlotUnbound`], exactly as `lend_slot` reports. Inspect via
    ///   `matches!`; do NOT replace the state.
    /// - [`Pipe`](Input::Pipe): **always OK** (deferred). A pipe is NEVER a pre-run
    ///   completeness failure — it is either (a) an internal edge whose producer
    ///   runs earlier in THIS graph and fills it at run time (empty now is normal),
    ///   or (b) a pre-loaded LENT pipe (`Input::lent`) whose payload is ALREADY
    ///   present (satisfiable). `resolve_home`'s Pipe arm CAN error ("upstream pipe
    ///   was not filled") — but ONLY if an upstream producer FAILED mid-run; with
    ///   `check_ready` having proved every LEAF input ready, no producer fails for a
    ///   readiness reason, so that arm cannot fire as a pre-run completeness error.
    ///   It stays as the execute-time internal-ordering backstop.
    pub fn check_ready(&self) -> Result<()> {
        match self {
            Input::Concrete(cell) => {
                if cell.lock().unwrap().is_some() {
                    Ok(())
                } else {
                    // Same condition + message `lend_from_cell`'s `empty_err`
                    // produces (a concrete input lent to a still-alive Checkout).
                    Err(Error::NotSupported(
                        "eager graph: a concrete input was already lent and not \
                         returned — a graph is `sync`'d while a previous `Checkout` is \
                         still alive (the graph is busy)",
                    ))
                }
            }
            // A typed slot: check its CELL STATE first, then DRAIN the deferred-error
            // sink. State-first gives a genuine missing bind (`SlotUnbound`) priority
            // over a recorded deferred error — so a graph left partly-unbound reports
            // the honest completeness failure, exactly as before this fix. Only once a
            // slot's own state is satisfiable do we surface a RECORDED bind error the
            // infallible `bind`/`call` apply path could not return (record-don't-drop;
            // see `DeferredErrors`): a `SlotConflict` (cell left `Bound` to the OLD
            // value), a `SlotNoSuchTag` (no cell of its own — recorded onto the first
            // real slot), or a `SlotCheckedOut`/`SlotSevered`. This fires BEFORE any
            // enqueue, so an errored deferred bind fails closed at sync instead of
            // silently running; the sink is empty for any graph the deferred path never
            // erred on, so the happy / reuse path is byte-for-byte untouched.
            //
            // `Bound` lends; `FedByPipe` is satisfied-by-upstream (deferred, like
            // `Input::Pipe` — never a pre-run failure); `Unbound`/`Severed`/`Lent` are
            // all `SlotUnbound` (its message covers all three) — mirrors `lend_slot`.
            Input::Slot {
                name, cell, sink, ..
            } => {
                match &*cell.lock().unwrap() {
                    SlotState::Bound(_) | SlotState::FedByPipe(_) => {}
                    SlotState::Unbound | SlotState::Severed | SlotState::Lent => {
                        return Err(Error::SlotUnbound(name));
                    }
                }
                // PEEK (not pop): a recorded deferred error is STICKY — it poisons
                // the graph so every subsequent `sync` re-reports it (recovery =
                // rebuild). See `DeferredErrors`.
                match peek_deferred(sink) {
                    Some(err) => Err(err),
                    None => Ok(()),
                }
            }
            // Deferred — see the doc above: never a pre-run failure.
            Input::Pipe(_) => Ok(()),
        }
    }

    // NOTE: `Input::resolve_on(&launcher)` — which built a transient
    // `ExecutionContext` so an image kernel's args could resolve outside an
    // `execute(&self)` — was removed when image kernels became reusable
    // `DeviceOp`s. Image args now lend through `resolve_home` from inside
    // `execute` exactly like slice args, so the standalone launcher-resolve seam
    // (the last piece of the image one-shot fork) has no remaining caller.

    /// If this input is a [`Concrete`](Input::Concrete) head, return its lending
    /// [`Cell`] so a run's `Checkout` can deposit the (possibly transformed-in-place /
    /// Uninit→Init-downgraded) value back into it on drop, re-arming the graph. A
    /// `Pipe` has no home cell, and a `Slot`'s home is the four-state
    /// `SlotHome` (not a plain `Cell`) — both return `None` here; a slot threads
    /// its home via [`slot_home`](Self::slot_home).
    pub fn return_cell(&self) -> Option<Cell<T>> {
        match self {
            Input::Concrete(cell) => Some(Arc::clone(cell)),
            Input::Slot { .. } | Input::Pipe(_) => None,
        }
    }

    /// If this input is a bound [`Slot`](Input::Slot), build its four-state return
    /// [`BoxedHome`] (a `SlotHome`) so an in-place op (e.g. a copy with a slot
    /// src/dst) re-arms `Lent → Bound` on `Checkout` drop and severs `Lent →
    /// Severed` on `into_inner`. `None` for concrete (use
    /// [`return_cell`](Self::return_cell) + `CopyHome`, which also handles the
    /// Uninit downgrade) and pipes.
    pub fn slot_home(&self) -> Option<BoxedHome<T>>
    where
        T: Send + 'static,
    {
        match self {
            Input::Slot { cell, .. } => Some(Box::new(SlotHome {
                cell: Arc::clone(cell),
            })),
            Input::Concrete(_) | Input::Pipe(_) => None,
        }
    }

    /// Build a copy output's return [`BoxedHome`] from THIS input, output-typed to
    /// `Out` (the post-copy buffer type, which may differ from the input `T` for an
    /// `Uninit → Init` dst). Unifies the concrete and slot copy-operand paths under
    /// the home invariant:
    /// - [`Concrete`](Input::Concrete): `T`'s [`CopyHome::copy_home`] (identity, or
    ///   the `Uninit → Init` downgrade re-wrap).
    /// - [`Slot`](Input::Slot): `T`'s [`CopyHome::copy_slot_home`] (a four-state
    ///   [`SlotHome`] — re-arms `Lent → Bound`, severs on `into_inner`). This is the
    ///   wiring of the formerly-dead [`slot_home`](Self::slot_home), now generalised
    ///   through `CopyHome` so it threads even when the copy retypes the output.
    /// - [`Pipe`](Input::Pipe): `None` — the upstream producer owns the value's
    ///   provenance; a copy doesn't re-mint it.
    ///
    /// The threaded home rides the output element pipe; the general
    /// [`PipePayload`] drop (or the terminal [`Checkout`] drop) fires the rehome,
    /// so a copy-positioned slot/concrete cell re-arms with a stable handle across
    /// `g.sync()` replays.
    fn copy_input_home<Out>(&self) -> Option<BoxedHome<Out>>
    where
        T: CopyHome<Out> + Send + 'static,
    {
        match self {
            Input::Concrete(cell) => <T as CopyHome<Out>>::copy_home(Arc::clone(cell)),
            Input::Slot { cell, .. } => <T as CopyHome<Out>>::copy_slot_home(Arc::clone(cell)),
            Input::Pipe(_) => None,
        }
    }

    /// Resolve a copy operand → `(value, deps, output-typed return home)`, the
    /// home-preserving form of [`resolve`](Self::resolve) for the copy path.
    /// Unlike [`copy_input_home`](Self::copy_input_home) (built BEFORE resolving,
    /// so its `Pipe` arm is blind to the payload), this reads the home while
    /// draining the input, so a **lent pipe** operand (a cross-graph `Checkout`
    /// fed to the copy — see [`Input::lent`]) forwards the ORIGIN graph's home via
    /// [`CopyHome::pipe_home`], giving copy operands the same LEND-and-return
    /// semantics as the kernel-arg path. Concrete / slot arms take their
    /// retype-aware home from `copy_input_home` (identity or `Uninit → Init`
    /// downgrade); a homeless (minted-upstream) pipe stays `None`.
    fn resolve_copy<Out>(
        &self,
        ec: &ExecutionContext<'_>,
    ) -> Result<(T, Deps, Option<BoxedHome<Out>>)>
    where
        T: CopyHome<Out> + Send + 'static,
    {
        let (v, deps, in_home) = self.resolve_home(ec)?;
        let out_home = match self {
            // The pipe payload's home is the origin's; forward it output-typed.
            Input::Pipe(_) => in_home.and_then(<T as CopyHome<Out>>::pipe_home),
            // Concrete / slot: rebuild the retype-aware home from the cell.
            Input::Concrete(_) | Input::Slot { .. } => self.copy_input_home(),
        };
        Ok((v, deps, out_home))
    }

    /// Borrow the concrete value via a clone of its cell, or `None` if this is a
    /// pipe or a slot. Used by the concrete-head no-launcher terminals
    /// (`wait`/`submit`) to recover the owning context from the buffer before
    /// running — a pipe-fed op has no concrete buffer, so those terminals error
    /// clearly.
    ///
    /// Returns a [`Cell`] handle (not a borrow) because the value lives behind a
    /// `Mutex`; callers `.lock()` it to read the buffer. `None` ⇒ pipe-fed or slot.
    /// (A slot's four-state cell isn't a plain `Cell`; slot-headed graphs run through
    /// the launcher terminals, not the concrete-head `wait`/`submit`.)
    pub fn concrete_cell(&self) -> Option<Cell<T>> {
        match self {
            Input::Concrete(cell) => Some(Arc::clone(cell)),
            Input::Slot { .. } | Input::Pipe(_) => None,
        }
    }

    /// Read a `Concrete`/bound-`Slot` input's value by reference (the value is
    /// parked in its cell — locked, not lent), mapping it via `f`. `None` if this
    /// is a pipe input or the value is currently lent / the slot is unbound (cell
    /// empty). Used by the record walk and the concrete-head context-recovery
    /// helpers, which only need to inspect the buffer's handle/byte-len.
    pub fn with_concrete<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        match self {
            Input::Concrete(cell) => cell.lock().unwrap().as_ref().map(f),
            // A slot is readable by-ref only while `Bound`; every empty state
            // (`Unbound`/`Severed`/`Lent`) maps to `None` (the same "no concrete
            // value to inspect" the empty concrete cell gives).
            Input::Slot { cell, .. } => match &*cell.lock().unwrap() {
                SlotState::Bound(v) => Some(f(v)),
                // `FedByPipe` has no concrete value to inspect (deferred) — like
                // `Pipe`.
                SlotState::Unbound
                | SlotState::Severed
                | SlotState::Lent
                | SlotState::FedByPipe(_) => None,
            },
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

    /// Apply a [`SlotBinder`]'s binding to this input, IFF it is a [`Slot`](Input::Slot)
    /// whose `id` matches the binder's tag — running the verb 2×2 against the slot's
    /// [`SlotState`].
    ///
    /// Used by [`bind_slots`](DeviceOp::bind_slots) as the graph is walked by
    /// [`bind`](DeviceOpExt::bind) / [`mutate_bind`](DeviceOpExt::mutate_bind). On a
    /// matching tag this resolves the state matrix:
    ///
    /// | state            | [`Set`](BindMode::Set)             | [`Mutate`](BindMode::Mutate) |
    /// |------------------|------------------------------------|------------------------------|
    /// | `Unbound`        | fill → `Bound`                     | fill → `Bound`               |
    /// | `Bound`, `==`    | no-op (idempotent)                 | overwrite                    |
    /// | `Bound`, `!=`    | record [`SlotConflict`](Error::SlotConflict) | overwrite          |
    /// | `Lent`           | record [`SlotCheckedOut`](Error::SlotCheckedOut) | (same)         |
    /// | `Severed`        | record [`SlotSevered`](Error::SlotSevered) | fill → `Bound`        |
    ///
    /// Equality is **buffer-handle identity** ([`SlotEq`], via the binder's captured
    /// comparator) — "the same buffer object", not byte-equal contents. After
    /// applying (or rejecting) a **move-only** binder is marked consumed so a later
    /// same-tag slot in the same walk is left alone (a single buffer is
    /// single-owner). A **fan-out** binder (clone-able value — scalar / launch /
    /// `Arc<DeviceSlice>`) is NOT consumed: it CLONES into this cell and stays armed
    /// so every matching cell is filled by the one `bind` (the shared-slot path).
    /// Errors are threaded out via [`SlotBinder::outcome`]. Non-matching arms / tags
    /// are a no-op.
    pub fn try_bind_slot(&self, binder: &mut SlotBinder)
    where
        T: Send + 'static,
    {
        let Input::Slot {
            id,
            name,
            cell,
            sink,
        } = self
        else {
            return;
        };
        // Capture a sink handle from the FIRST slot cell this walk visits — BEFORE
        // the tag-id gate below. This is what lets the infallible apply path land an
        // ABSENT-tag error (which matches no cell) into a graph-reachable sink that
        // `check_ready` also drains: even an unrelated `slot!(Tag)` is a visited
        // real slot of the same graph. Only the deferred apply path sets
        // `binder.deferred` (the fluent verbs leave it `None` → no capture, no cost).
        if binder.deferred && binder.captured_sink.is_none() {
            binder.captured_sink = Some(Arc::clone(sink));
        }
        if *id != binder.id {
            return;
        }
        // The tag IS present at this site — record a match regardless of what the
        // per-state arm below does (fill / idempotent no-op / conflict / sever /
        // a same-tag cell reached after a move-only binder was already consumed).
        // This is the "tag exists in the graph" counter that `fold_bind` reads to
        // turn a ZERO-match `bind` into `SlotNoSuchTag`; a conflict/sever still
        // surfaces via `binder.outcome`, so counting it here cannot mask it.
        binder.matched += 1;

        // PROBE (read-only) — the phase-0 dry run of `call`/`mutate_call`. Inspect
        // this cell's state WITHOUT filling / taking / replacing, recording the
        // verdict the phase-2 fold WOULD produce on the POST-sever state (a `Lent`
        // cell that a tuple `Checkout` will sever is predicted as `Severed`; see
        // `probe_lent`). It records into `binder.outcome` exactly like the real fold,
        // so `fold_probe` surfaces the first error having severed / mutated NOTHING.
        // The value-equality leg of `Set` on a `Bound` cell is the ONE case a probe
        // cannot decide (the value lives in an unsevered `Checkout`) — a probe treats
        // `Bound` as OK and lets phase 2 catch a genuine `SlotConflict`; that is the
        // documented residual (see `call`/`mutate_call` docs).
        if binder.is_probe() {
            let cell_id = Arc::as_ptr(cell) as usize;
            match &*cell.lock().unwrap() {
                // Both verbs fill a virgin / re-arm a bound cell → OK. `Set` on
                // `Bound` is the value-dependent residual (treated OK here). A
                // `FedByPipe` cell is treated like `Bound` for a value-bind probe: a
                // value bind over it would overwrite under `Mutate` / conflict-ish
                // under `Set`, but the spike never value-binds a pipe-fed slot, so it
                // is inert here (a `feed` binder, which is the only writer of this
                // state, is never a probe).
                SlotState::Unbound | SlotState::Bound(_) | SlotState::FedByPipe(_) => {}
                // `Set` rejects a severed slot; `Mutate` re-arms it.
                SlotState::Severed => {
                    if binder.mode == BindMode::Set {
                        binder.outcome = Err(Error::SlotSevered(name));
                    }
                }
                // Post-sever prediction: tuple-held → Severed (Set fails / Mutate
                // OK); external-held → stays Lent (both fail SlotCheckedOut).
                SlotState::Lent => {
                    if let Err(e) = binder.probe_lent(cell_id, name) {
                        binder.outcome = Err(e);
                    }
                }
            }
            // A probe never consumes: return so the walk visits the next cell too.
            return;
        }

        // PIPE-FEED install (the `feed` verb). Deposit `FedByPipe(pipe.clone())` into
        // THIS cell — a fan-out, so every matching site is fed. Unconditional over
        // the current state: the common case installs onto a virgin (`Unbound`) slot
        // freshly built by the subgraph; re-feeding an already-`FedByPipe` cell (a
        // graph fed a second time) just re-installs the same-or-new pipe. `Lent`
        // should not occur (a pipe-fed slot is never lent to a `Checkout`), but
        // overwriting it is still sound — the pipe is drained fresh next run. Handled
        // BEFORE the `value.is_none()` early-bail below (a feed binder carries no
        // value, so it would otherwise be misread as "consumed" and skip every cell).
        if let Some(boxed) = &binder.feed_pipe {
            if let Some(pipe) = boxed.downcast_ref::<Pipe<T>>() {
                *cell.lock().unwrap() = SlotState::FedByPipe(pipe.clone());
            }
            return;
        }

        // A move-only binder is consumed after its single value lands; bail. A
        // fan-out binder never sets `value = None`, so it keeps filling cells.
        if binder.value.is_none() {
            return;
        }

        // Produce the value to deposit (only when we will actually fill): a fan-out
        // binder CLONES (so it can fill the next cell too); a move-only binder TAKES
        // its single value. `provide` is called at most once per cell, lazily, so an
        // idempotent no-op / conflict / sever-reject path costs no clone or move.
        // Returns `None` only on the impossible downcast mismatch (TypeId already
        // pinned `T == Tag::Value`).
        let fanout = binder.is_fanout();
        let provide = |binder: &mut SlotBinder| -> Option<T> {
            let boxed = if fanout {
                // Clone into THIS cell; the binder stays armed for the rest.
                binder.fill_clone()?
            } else {
                // Move the single value out; the binder is now consumed.
                binder.take_value()?
            };
            match boxed.downcast::<T>() {
                Ok(v) => Some(*v),
                // Downcast can't fail (TypeId matched). If it ever did and we had
                // TAKEN the value, put it back so a correctly-typed slot can see it.
                Err(boxed) => {
                    if !fanout {
                        binder.value = Some(boxed);
                    }
                    None
                }
            }
        };

        let mut guard = cell.lock().unwrap();
        match &*guard {
            // Virgin — never bound. Both verbs fill it (a `bind` is the slot's
            // first declaration).
            SlotState::Unbound => {
                if let Some(new) = provide(binder) {
                    *guard = SlotState::Bound(new);
                }
            }
            // Severed — was bound, the caller took its value via `into_inner`.
            // Re-providing a value is a *change*, not a first declaration: the
            // set-once `bind` rejects it (it must not silently re-fill a slot whose
            // value the caller deliberately extracted); only `mutate_bind` re-arms.
            // (A non-resource scalar/launch slot never reaches `Severed` — it is
            // never lent/checked-out — so this arm is buffer-only in practice.)
            SlotState::Severed => match binder.mode {
                BindMode::Set => {
                    binder.outcome = Err(Error::SlotSevered(name));
                }
                BindMode::Mutate => {
                    if let Some(new) = provide(binder) {
                        *guard = SlotState::Bound(new);
                    }
                }
            },
            SlotState::Bound(cur) => match binder.mode {
                BindMode::Set => {
                    // Idempotent on the SAME value (buffer handle identity, or scalar
                    // value equality); conflict on a different one. Equality is the
                    // binder's captured `SlotEq` comparator, applied to `cur` vs the
                    // binder's value as `&dyn Any` — WITHOUT consuming the value, so
                    // an idempotent no-op neither clones nor moves.
                    let same = binder
                        .value
                        .as_deref()
                        .map(|v| (binder.eq)(cur as &dyn Any, v))
                        .unwrap_or(false);
                    if same {
                        // no-op: the caller re-handed us the value we already hold.
                    } else {
                        binder.outcome = Err(Error::SlotConflict(name));
                    }
                }
                BindMode::Mutate => {
                    // Overwrite: the prior value drops here.
                    if let Some(new) = provide(binder) {
                        *guard = SlotState::Bound(new);
                    }
                }
            },
            // The value is in the caller's hands (a live `Checkout`). Re-binding
            // would let the Checkout's drop rehome the OLD buffer over the NEW one
            // — a silent clobber — so BOTH verbs hard-error. (Buffer slots only;
            // non-resource slots are never `Lent`.)
            SlotState::Lent => {
                binder.outcome = Err(Error::SlotCheckedOut(name));
            }
            // A value bind onto a pipe-fed slot. `Set` conflicts (the slot is
            // already sourced by an upstream pipe); `Mutate` overwrites the pipe
            // source with the value. The spike never value-binds a pipe-fed slot, so
            // this arm is inert in practice — present only for exhaustiveness +
            // correctness (a value bind should not silently no-op over a live feed).
            SlotState::FedByPipe(_) => match binder.mode {
                BindMode::Set => {
                    binder.outcome = Err(Error::SlotConflict(name));
                }
                BindMode::Mutate => {
                    if let Some(new) = provide(binder) {
                        *guard = SlotState::Bound(new);
                    }
                }
            },
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

impl<T> Input<T> {
    /// Build an input that **lends** an already-produced value while carrying its
    /// existing return `home` ONWARD — the vehicle for feeding a [`Checkout`] from
    /// graph A forward as a borrow input to a second graph B.
    ///
    /// A plain [`From<T>`](Input::from) input is a fresh `Concrete` cell with NO
    /// home: when B drops it, the value is released and A is left broken. Instead
    /// this pre-loads a [`Pipe`] with `(value, deps, home)` via
    /// [`put_home`](Pipe::put_home), then wraps it as [`Input::Pipe`]. The home —
    /// A's still-`Lent` cell — thus rides B's graph exactly like any internal edge:
    /// [`resolve_home`](Self::resolve_home)'s `Pipe` arm hands the home through to
    /// B's ops, and B's terminal `Checkout` (or an undelivered
    /// [`PipePayload`] drop) fires the rehome, RETURNING the value to A and
    /// re-arming it for a plain `g_a.sync()`.
    ///
    /// A pre-filled pipe resolves correctly even as a LEAF input (no producer runs
    /// before it): its payload is already present, so the first
    /// [`take_home`](Pipe::take_home) just drains it. And because the home threads
    /// pipe → pipe through B (and onward to C, …) by the same machinery, the lend
    /// composes transitively — the value returns to A only at the FINAL drop of the
    /// whole chain.
    ///
    /// `home == None` (the source Checkout carried no home — minted/consumed) is
    /// fine: the value simply rides the pipe with nothing to return, identical to a
    /// homeless mint.
    fn lent(value: T, home: Option<BoxedHome<T>>) -> Self {
        let pipe = Pipe::new();
        pipe.put_home(value, Deps::new(), home);
        Input::Pipe(pipe)
    }
}

// ── ScalarInput<V>: a non-resource (scalar / launch) kernel input ───────────

/// A by-value kernel input that can be a plain literal OR an unbound non-resource
/// slot — the scalar/launch analogue of [`Input<T>`] (which is for move-only
/// resources: buffers / images).
///
/// Two states, mirroring how a scalar reaches a kernel:
/// - [`Concrete`](ScalarInput::Concrete): a plain value passed at build (the
///   common case — `scale_u32([N], buf, 2u32)` stores `Concrete(2)`).
/// - [`Slot`](ScalarInput::Slot): an unbound `slot!(Tag)` hole, filled by a later
///   [`bind`](DeviceOpExt::bind)`(Tag(value))`. Its cell is the **two-state**
///   [`ScalarSlotCell`] — `Unbound`/`Bound` only, NO resource machine.
///
/// At execute the value is **read (cloned)**, never lent: [`read`](Self::read)
/// returns `V` by clone for both arms, so a bound slot is re-read on every replay
/// (no re-bind needed) and a `Concrete` literal is reusable too. `V: Clone` is the
/// one bound the by-value path needs (every scalar / [`LaunchSpec`](crate::LaunchSpec) satisfies it).
pub enum ScalarInput<V> {
    /// A plain value bound at construction.
    Concrete(V),
    /// An unbound (or later-bound) non-resource slot — `slot!(Tag)` in a scalar /
    /// launch position. `id`/`name` are the tag's `TypeId::of::<Tag::Key>()` /
    /// [`Tag::NAME`] for matching and diagnostics; `cell` is the two-state
    /// [`ScalarSlotCell`].
    Slot {
        /// `TypeId::of::<Tag::Key>()` — the bind-matching key.
        id: TypeId,
        /// [`Tag::NAME`] (the clean tag ident) — for the unbound-slot / conflict
        /// diagnostics.
        name: &'static str,
        /// Two-state cell: [`Unbound`](ScalarSlotState::Unbound) until a matching
        /// bind deposits the value ([`Bound`](ScalarSlotState::Bound)).
        cell: ScalarSlotCell<V>,
        /// The [`DeferredErrors`] sink for the infallible `bind`/`call` apply path —
        /// same role as [`Input::Slot`]'s `sink`. A scalar slot fed a conflicting
        /// value-bind records here; `check_ready` drains it. Starts empty.
        sink: DeferredErrors,
    },
}

impl<V: Clone> ScalarInput<V> {
    /// **Read** the value for one run (clone — NOT lend, NOT take): a `Concrete`
    /// clones its stored value; a `Slot` clones its [`Bound`](ScalarSlotState::Bound)
    /// value or, while [`Unbound`](ScalarSlotState::Unbound), is
    /// [`Error::SlotUnbound`] (the completeness check, fired at execute). The cell
    /// is left intact, so the next replay re-reads the same value for free.
    pub fn read(&self) -> Result<V> {
        match self {
            ScalarInput::Concrete(v) => Ok(v.clone()),
            ScalarInput::Slot { name, cell, .. } => match &*cell.lock().unwrap() {
                ScalarSlotState::Bound(v) => Ok(v.clone()),
                ScalarSlotState::Unbound => Err(Error::SlotUnbound(name)),
            },
        }
    }

    /// **Read-only** pre-flight check: would [`read`](Self::read) succeed right now,
    /// WITHOUT cloning anything out? The non-resource half of the
    /// [`check_ready`](DeviceOp::check_ready) atomicity walk. A `Concrete` value is
    /// always ready; a `Slot` is ready iff [`Bound`](ScalarSlotState::Bound),
    /// otherwise [`Error::SlotUnbound`] — the SAME error `read` returns for an
    /// unbound scalar/launch slot (there is no `Lent`/`Severed` state for a scalar).
    pub fn check_ready(&self) -> Result<()> {
        match self {
            ScalarInput::Concrete(_) => Ok(()),
            ScalarInput::Slot {
                name, cell, sink, ..
            } => {
                // State-first, then drain the deferred-error sink (see
                // `Input::check_ready` for the ordering rationale).
                match &*cell.lock().unwrap() {
                    ScalarSlotState::Bound(_) => {}
                    ScalarSlotState::Unbound => return Err(Error::SlotUnbound(name)),
                }
                // PEEK (not pop): sticky/poison — see `Input::check_ready` and
                // `DeferredErrors`.
                match peek_deferred(sink) {
                    Some(err) => Err(err),
                    None => Ok(()),
                }
            }
        }
    }
}

impl<V: Send + 'static> ScalarInput<V> {
    /// Apply a [`SlotBinder`] to this non-resource input, IFF it is a
    /// [`Slot`](ScalarInput::Slot) whose `id` matches the binder's tag — the
    /// two-state analogue of [`Input::try_bind_slot`].
    ///
    /// | state       | [`Set`](BindMode::Set)                       | [`Mutate`](BindMode::Mutate) |
    /// |-------------|----------------------------------------------|------------------------------|
    /// | `Unbound`   | fill → `Bound`                               | fill → `Bound`               |
    /// | `Bound`, `==` | no-op (idempotent, value equality)         | overwrite                    |
    /// | `Bound`, `!=` | record [`SlotConflict`](Error::SlotConflict) | overwrite                  |
    ///
    /// There is no `Lent`/`Severed` row: a scalar/launch value is never lent or
    /// severed (it is read by clone, never handed out), so the resource-machine
    /// errors ([`SlotCheckedOut`](Error::SlotCheckedOut) /
    /// [`SlotSevered`](Error::SlotSevered)) cannot arise here. Equality is by
    /// **value** ([`SlotEq`] over the scalar / [`LaunchSpec`](crate::LaunchSpec)).
    ///
    /// Fan-out is automatic: a scalar/launch value is clone-able
    /// ([`SlotValue::fill_clone`]`= Some`), so the binder clones into this cell and
    /// stays armed — one `bind(Grid(g))` fills EVERY `slot!(Grid)` site.
    pub fn try_bind_slot(&self, binder: &mut SlotBinder) {
        let ScalarInput::Slot {
            id,
            name,
            cell,
            sink,
        } = self
        else {
            return;
        };
        // Capture a sink handle from the first slot this walk visits (before the
        // tag-id gate) so an ABSENT-tag deferred error has a graph-reachable home —
        // the scalar mirror of `Input::try_bind_slot`. Only the deferred apply path
        // sets `binder.deferred`.
        if binder.deferred && binder.captured_sink.is_none() {
            binder.captured_sink = Some(Arc::clone(sink));
        }
        if *id != binder.id {
            return;
        }
        // The tag IS present at this scalar site — record the match before the
        // consumed-binder guard, mirroring `Input::try_bind_slot`, so `fold_bind`
        // sees a nonzero count even for a same-tag cell reached after consumption.
        binder.matched += 1;

        // PROBE (read-only): a scalar/launch slot is only ever `Unbound` or `Bound`
        // (never `Lent`/`Severed` — it is read by clone, never lent or severed), so
        // presence (`matched`, recorded above) is the whole probe verdict here. `Set`
        // on a `Bound` scalar is the same value-dependent residual as a resource slot
        // (treated OK; phase 2 catches a genuine `SlotConflict`).
        if binder.is_probe() {
            return;
        }

        if binder.value.is_none() {
            return; // a consumed move-only binder (never happens for a clone-able
            // scalar value, but keeps the guard uniform with the resource path).
        }

        // Same fan-out-vs-move discipline as `Input::try_bind_slot`, but a scalar
        // value is always clone-able, so this is effectively always the clone path.
        let fanout = binder.is_fanout();
        let provide = |binder: &mut SlotBinder| -> Option<V> {
            let boxed = if fanout {
                binder.fill_clone()?
            } else {
                binder.take_value()?
            };
            match boxed.downcast::<V>() {
                Ok(v) => Some(*v),
                Err(boxed) => {
                    if !fanout {
                        binder.value = Some(boxed);
                    }
                    None
                }
            }
        };

        let mut guard = cell.lock().unwrap();
        match &*guard {
            ScalarSlotState::Unbound => {
                if let Some(new) = provide(binder) {
                    *guard = ScalarSlotState::Bound(new);
                }
            }
            ScalarSlotState::Bound(cur) => match binder.mode {
                BindMode::Set => {
                    // Idempotent on an equal value; conflict on a different one.
                    let same = binder
                        .value
                        .as_deref()
                        .map(|v| (binder.eq)(cur as &dyn Any, v))
                        .unwrap_or(false);
                    if same {
                        // no-op: the caller re-handed us the value we already hold.
                    } else {
                        binder.outcome = Err(Error::SlotConflict(name));
                    }
                }
                BindMode::Mutate => {
                    if let Some(new) = provide(binder) {
                        *guard = ScalarSlotState::Bound(new);
                    }
                }
            },
        }
    }
}

// `ScalarInput<V>` is built from a plain value / grid literal / `slot!(Tag)` via
// the per-type `From` impls under "ToScalarInput" below (NOT a blanket `From<V>`,
// which would coherence-clash with the `SlotHandle` conversion). The macro routes
// scalar / grid positions through `Into<ScalarInput<#ty>>`, which preserves
// integer-literal inference (`fill_u32([N], buf, 5)` infers `5: u32`).

// ── SlotHandle: the value a `slot!(Tag)` produces ──────────────────────────

/// The build-time handle produced by [`slot!`](crate::slot)`(Tag)` — an UNBOUND
/// typed hole that plugs into the same positions a concrete buffer does (kernel
/// args, `download`/`fill`/`write`/copy sources, …). It carries the tag's
/// [`TypeId`] + [`NAME`](Tag::NAME) and a fresh [`Unbound`](SlotState::Unbound)
/// [`SlotCell`]; converting it (via [`From`] / [`ToInput`]) yields an
/// [`Input::Slot`] sharing that cell, which a later
/// [`bind`](DeviceOpExt::bind)`(Tag(value))` fills.
///
/// `PhantomData<fn() -> Tg>` keeps the handle `Send`/`Sync` regardless of `Tg`
/// (the tag type is a pure marker — never stored).
pub struct SlotHandle<Tg: Tag> {
    id: TypeId,
    name: &'static str,
    cell: SlotCell<Tg::Value>,
    _tag: PhantomData<fn() -> Tg>,
}

impl<Tg: Tag> SlotHandle<Tg> {
    /// Mint a fresh unbound slot handle for tag `Tg`. Prefer the
    /// [`slot!`](crate::slot) macro spelling (`slot!(Buf)`).
    pub fn new() -> Self {
        SlotHandle {
            id: TypeId::of::<Tg::Key>(),
            // Match-key TypeId is `Tg::Key`; the display name is the clean tag ident
            // (`Tg::NAME`) — NOT `type_name::<Tg::Key>()`, which would leak the
            // internal `<KeyMarker>` source suffix into user-facing slot errors.
            name: Tg::NAME,
            cell: Arc::new(Mutex::new(SlotState::Unbound)),
            _tag: PhantomData,
        }
    }

    /// Consume the handle into its [`Input::Slot`] (shares the `Unbound` cell).
    fn into_input(self) -> Input<Tg::Value> {
        Input::Slot {
            id: self.id,
            name: self.name,
            cell: self.cell,
            // Fresh empty deferred-error sink (see [`DeferredErrors`]): written only
            // by the infallible `bind`/`call` apply path, drained by `check_ready`.
            sink: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Consume the handle into a non-resource [`ScalarInput::Slot`] (a scalar /
    /// launch position). The handle's four-state [`SlotCell`] is the resource
    /// machinery a scalar slot does NOT need, so it is dropped here and a fresh
    /// **two-state** [`ScalarSlotCell`] is minted in its place; only the tag's
    /// `id`/`name` carry over. (The wasted `Arc` alloc keeps `slot!(Tag)` uniform
    /// across buffer and scalar positions — a single spelling, dispatched by the
    /// position's `ToInput` vs `ToScalarInput` trait.)
    fn into_scalar_input(self) -> ScalarInput<Tg::Value> {
        ScalarInput::Slot {
            id: self.id,
            name: self.name,
            cell: Arc::new(Mutex::new(ScalarSlotState::Unbound)),
            // Fresh empty deferred-error sink (see [`DeferredErrors`]).
            sink: Arc::new(Mutex::new(Vec::new())),
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

// A device SCALAR ([`Scalar<B>`], any memory tier) plugs into a (scalar-ref)
// kernel-arg position exactly like a slice does into a slice position — the
// macro's `__D: KernelScalarRefArg<E>` bound (not the slice trait) is what pins
// it to `&T` args only. `E` is unconstrained here (the macro's `Buf = __D` bound
// ties it); `Scalar<B>` is a distinct nominal type from the bare slice families
// / `Pipe<D>` / `Checkout<_>`, so it stays disjoint under coherence.
impl<E, B> ToInput<E> for Scalar<B> {
    type Buf = Scalar<B>;
    fn to_input(self) -> Input<Scalar<B>> {
        Input::from(self)
    }
}

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

// ── ToScalarInput: a non-resource (scalar / launch) arg, value-or-slot ──────
//
// The scalar / launch-arg analogue of [`ToInput`]. The macro-emitted kernel
// method takes each SCALAR arg (and the GRID) as `impl Into<ScalarInput<#ty>>`
// and stores the resulting `ScalarInput<#ty>`. Two source shapes convert in:
//   - a plain value (`u32`, a grid literal `[N]`, …) → `Concrete`.
//   - a `slot!(Tag)` (`SlotHandle<Tg>`) → an unbound two-state `Slot`.
// so BOTH `scale_u32([N], buf, 2u32)` and `scale_u32([N], buf, slot!(Factor))`
// type-check in the scalar position, and both `kernels.k([N], …)` and
// `kernels.k(slot!(Grid), …)` in the grid position.
//
// The bound is `Into<ScalarInput<#ty>>` (NOT a custom trait with a projected
// `Val`): an `Into<Target>` bound drives integer-literal inference — a bare `5` in
// `fill_u32([N], buf, 5)` resolves to `u32` because `From<u32> for
// ScalarInput<u32>` is the only `From` whose source unifies with `{integer}`. A
// projected-associated-type trait (`ToScalarInput<Val = u32>`) would NOT — the
// literal defaults to `i32` first and then fails the `Val` check.
//
// Per-type `From` impls (NOT a blanket `From<V>`) keep them disjoint from the
// `SlotHandle<Tg>` conversion under coherence (a `SlotHandle` is never a plain
// scalar value / grid literal).

/// A `slot!(Tag)` in a scalar / launch position → an unbound two-state
/// [`ScalarInput::Slot`]. Distinct nominal source type from the per-value `From`
/// impls below, so the two stay disjoint under coherence.
impl<Tg> From<SlotHandle<Tg>> for ScalarInput<Tg::Value>
where
    Tg: Tag,
{
    fn from(handle: SlotHandle<Tg>) -> Self {
        handle.into_scalar_input()
    }
}

/// `From<scalar> for ScalarInput<scalar>` — a plain value in a scalar position →
/// [`ScalarInput::Concrete`]. Per-type (not a `impl<V>` blanket) so it stays
/// disjoint from the `SlotHandle<Tg>` conversion. A user `#[repr(C)] Copy` scalar
/// opts in with the same one-liner (alongside its [`ScalarArg`](crate::ScalarArg) /
/// [`SlotValue`] / [`SlotEq`] impls).
macro_rules! impl_from_scalar_input_value {
    ($($t:ty),* $(,)?) => {
        $(
            impl From<$t> for ScalarInput<$t> {
                fn from(v: $t) -> Self {
                    ScalarInput::Concrete(v)
                }
            }
        )*
    };
}
impl_from_scalar_input_value!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);

/// `From<grid-literal> for ScalarInput<LaunchSpec>` — a launch-geometry literal
/// (`[N]`, `([W], [L])`, a [`LaunchSpec`](crate::LaunchSpec)) → a
/// [`ScalarInput::Concrete`] over the built [`LaunchSpec`](crate::LaunchSpec). Converts the geometry
/// to its canonical form so a `slot!(Grid)` (carrying a `LaunchSpec`) and a literal
/// grid share one resolved type. Per-type over the
/// [`IntoLaunchSpec`](crate::IntoLaunchSpec) literal shapes, disjoint from
/// `SlotHandle`.
macro_rules! impl_from_scalar_input_grid {
    ($($t:ty),* $(,)?) => {
        $(
            impl From<$t> for ScalarInput<crate::LaunchSpec> {
                fn from(g: $t) -> Self {
                    ScalarInput::Concrete(crate::IntoLaunchSpec::into_launch_spec(g))
                }
            }
        )*
    };
}
impl_from_scalar_input_grid!(
    [usize; 1],
    [usize; 2],
    [usize; 3],
    ([usize; 1], [usize; 1]),
    ([usize; 2], [usize; 2]),
    ([usize; 3], [usize; 3]),
);
// `LaunchSpec` itself: the reflexive value path (and what a bound `slot!(Grid)`
// carries). Kept out of the macro above so it doesn't collide with the std
// reflexive `From<T> for T` (here source = target = `ScalarInput<LaunchSpec>`? no —
// source is `LaunchSpec`, target `ScalarInput<LaunchSpec>`, so it's a normal impl).
impl From<crate::LaunchSpec> for ScalarInput<crate::LaunchSpec> {
    fn from(spec: crate::LaunchSpec) -> Self {
        ScalarInput::Concrete(spec)
    }
}

// ── Transparency: a `Checkout<buffer>` is usable wherever the bare buffer is ──
//
// So a reused-graph output flows straight into the next op WITHOUT an explicit
// `.into_inner()`:
//   let b = x.fill(7).wait()?;          // b: Checkout<DeviceSlice>
//   ks.scale([N], b, 3) …               // fed directly as a kernel arg
//
// Feeding a `Checkout` from graph A forward as a borrow input to a SECOND graph B
// **LENDS** it — it does NOT sever A. The Checkout's home (pointing into A's
// still-`Lent` cell) rides INTO B via a pre-loaded pipe
// ([`Input::lent`]/[`Checkout::into_value_and_home`]): A stays BUSY for as long as
// B holds the buffer (a second `g_a.sync()` errors busy, never silently re-runs),
// and when B's terminal `Checkout` (or an undelivered drop) releases the value it
// RETURNS to A, re-arming it for a plain `g_a.sync()` — no `mutate_bind`.
// `.into_inner()` remains the explicit "take it out for good" verb (severs); only
// this implicit feed-as-input path lends. Distinct nominal type from the bare
// families and `Pipe<D>`, so it stays disjoint under coherence.
//
// CONTRAST: binding a Checkout INTO A SLOT (`IntoBound`) correctly SEVERS+ADOPTS —
// the buffer changes role there; here it is only borrowed and returned.
macro_rules! impl_to_input_checkout {
    ($buf:ident) => {
        impl<E, M> ToInput<E> for Checkout<$crate::$buf<E, M>>
        where
            M: $crate::MemMode,
            E: Send,
        {
            type Buf = $crate::$buf<E, M>;
            fn to_input(self) -> Input<$crate::$buf<E, M>> {
                // LEND: relocate the value + its home onto a pre-loaded pipe so the
                // home rides into the consuming graph and returns to A on drop.
                let (value, home) = self.into_value_and_home();
                Input::lent(value, home)
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
                // LEND (see the `ToInput` twin above) — relocate value + home.
                let (value, home) = co.into_value_and_home();
                Input::lent(value, home)
            }
        }
    };
}
impl_to_input_checkout!(DeviceSlice);
impl_to_input_checkout!(MappedSlice);
impl_to_input_checkout!(USMSlice);

// A `Checkout<Scalar<B>>` (any memory tier) LENDS forward exactly like a slice
// checkout. Distinct nominal type from the bare families / `Pipe` / slice
// checkouts, so it stays disjoint under coherence.
impl<E, B> ToInput<E> for Checkout<Scalar<B>>
where
    B: Send,
{
    type Buf = Scalar<B>;
    fn to_input(self) -> Input<Scalar<B>> {
        let (value, home) = self.into_value_and_home();
        Input::lent(value, home)
    }
}
impl<B> From<Checkout<Scalar<B>>> for Input<Scalar<B>>
where
    B: Send,
{
    fn from(co: Checkout<Scalar<B>>) -> Self {
        let (value, home) = co.into_value_and_home();
        Input::lent(value, home)
    }
}

// `Checkout<Arc<DeviceSlice<E, M>>>` — the shared-buffer arg, LENT via its home
// (see the macro above): A stays busy until B drops the Arc, then it rehomes.
impl<E, M> ToInput<E> for Checkout<std::sync::Arc<DeviceSlice<E, M>>>
where
    M: MemMode,
    std::sync::Arc<DeviceSlice<E, M>>: Send,
{
    type Buf = std::sync::Arc<DeviceSlice<E, M>>;
    fn to_input(self) -> Input<std::sync::Arc<DeviceSlice<E, M>>> {
        let (value, home) = self.into_value_and_home();
        Input::lent(value, home)
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

    /// The **runtime value-storage** pipe — where `execute` deposits the result;
    /// what the single-output terminal (`sync`) drains. `Some` for a single-output
    /// op (its one storage pipe, independent of the build-time [`Handle`](Self::Handle)),
    /// `None` for a **multi-output** op (Bundle2..16, [`FanOut`], [`CopyTo2`],
    /// `arc_split`, the `and_then_host` seam over a multi-output source, …), whose
    /// storage is per-branch/element pipes with no single collapsed pipe. The
    /// generic default gather methods ([`collect`](Self::collect) /
    /// [`collect_home`](Self::collect_home) / [`gather_checkouts`](Self::gather_checkouts)
    /// / [`reclaim_undelivered`](Self::reclaim_undelivered)) unwrap the `Some` — they
    /// are single-output-only (every multi-output op overrides all of them), so the
    /// `None` case is never reached through them.
    fn output_pipe(&self) -> Option<Pipe<Self::Output>>;

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
        let out = self
            .output_pipe()
            .expect("single-output op must have an output pipe (multi-output overrides collect)");
        self.execute(ec, mode)?;
        out.take()
            .ok_or(Error::NotSupported("eager graph: op produced no output"))
    }

    /// Home-preserving analog of [`collect`](Self::collect): run this op as a
    /// (sub)terminal and yield `(Output, Deps, home)` — the SAME value+deps as
    /// `collect`, plus the return [`home`](BoxedHome) so a caller-owned buffer
    /// re-arms its origin cell on `Checkout` drop.
    ///
    /// Used by [`AndThen::collect_home`](Self::collect_home) to preserve the
    /// tail's return home so an `and_then`-terminated chain nested as a bundle
    /// branch (e.g. `upload(buf).and_then(|b| fill(b, v))`) re-arms its caller
    /// buffer. (A `bundle!` itself no longer routes re-arm through `collect_home`:
    /// its terminal [`gather_checkouts`](Self::gather_checkouts) DELEGATES to each
    /// branch's own `gather_checkouts`, so every branch — single- OR multi-output,
    /// at any nesting depth — threads its OWN per-buffer homes and re-arms.)
    ///
    /// Default (single-output ops): `execute`, drain
    /// [`output_pipe`](Self::output_pipe) WITH its home ([`Pipe::take_home`]) —
    /// the home an in-place op ([`fill`]/`scale`/copy-dst/…) threaded through.
    ///
    /// **Multi-output ops** (whose storage is per-element pipes, so their single
    /// `output_pipe` is never filled) override this to delegate to their own
    /// [`collect`](Self::collect) and return `home == None`: a `Vec`/tuple output
    /// is ONE value with ONE home slot but N per-buffer homes, so those homes
    /// can't ride a single collapsed slot (the same boundary [`FanOut`]
    /// documents). This `None` only affects the BY-VALUE path (`collect` / async
    /// `run`), which never builds `Checkout`s so has no homes to thread anyway;
    /// the Checkout terminal ([`gather_checkouts`](Self::gather_checkouts))
    /// delegates per-branch and re-arms every buffer regardless of multiplicity.
    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Output, Deps, Option<BoxedHome<Self::Output>>)>
    where
        Self: Sized,
        Self::Output: Send + 'static,
    {
        let out = self.output_pipe().expect(
            "single-output op must have an output pipe (multi-output overrides collect_home)",
        );
        self.execute(ec, mode)?;
        out.take_home()
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
        let out = self.output_pipe().expect(
            "single-output op must have an output pipe (multi-output overrides gather_checkouts)",
        );
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
    /// the per-op half of [`bind`](DeviceOpExt::bind)`(Tag(value))`.
    ///
    /// Walks the op's own [`Input`] fields, calling
    /// [`try_bind_slot`](Input::try_bind_slot) on each so a matching unbound slot
    /// takes the (moved) value; combinators recurse into their children
    /// (mirroring [`describe`](Self::describe)). The default is a **no-op** — most
    /// leaves hold no slot, and `bind` simply finds nothing to bind. Ops that
    /// accept buffer args (kernels, `download`/`fill`/`write`/copy, the bundles)
    /// override this to visit their inputs.
    ///
    /// Order-free + curryable falls out of this being one binder per `bind`: each
    /// `bind` deposits ONE tag's value into the first matching cell, independent of
    /// other tags / bind order; completeness is only enforced later at
    /// [`sync`](DeviceOpExt::sync). A short-circuit on
    /// [`is_consumed`](SlotBinder::is_consumed) lets a walk stop early once the
    /// value has landed.
    fn bind_slots(&self, binder: &mut SlotBinder) {
        let _ = binder;
    }

    /// **Atomicity pre-pass** — validate that EVERY input cell of the WHOLE
    /// (sub)graph is satisfiable RIGHT NOW, WITHOUT executing / enqueuing / lending
    /// anything. Called once by the terminals ([`wait_on`](DeviceOpExt::wait_on) →
    /// [`sync`](DeviceOpExt::sync)) BEFORE the first
    /// [`gather_checkouts`](Self::gather_checkouts)/[`execute`](Self::execute), so a
    /// failed `sync` leaves the graph UNCHANGED + re-runnable.
    ///
    /// ## Why it exists (the atomicity hole it closes)
    ///
    /// [`execute`](Self::execute) interleaves resolution with enqueue: each leaf
    /// [`resolve_home`](Input::resolve_home)s its input (which LENDS the buffer —
    /// `Bound → Lent`) AND enqueues its device work in the same call, and the graph
    /// is walked depth-first ([`AndThen::execute`](Self::execute) runs `source` then
    /// `next`). So if a LATER node has an unsatisfiable input, EARLIER nodes have
    /// already lent their buffers and enqueued — and the failing `sync` strands
    /// those earlier cells `Lent` with no `Checkout` to re-arm them, so a retry
    /// spuriously reports them busy. This walk proves all leaves ready FIRST; only
    /// then does `execute` proceed, so nothing is touched on the failure path.
    ///
    /// ## Contract
    ///
    /// - **Read-only.** Inspect cells via `is_some()` / `matches!`; NEVER `take`,
    ///   `replace`, lend, or enqueue. Coverage MUST match what
    ///   [`execute`](Self::execute) actually resolves: every input a node's
    ///   `execute` calls [`resolve_home`](Input::resolve_home)/[`resolve`](Input::resolve)/[`read`](ScalarInput::read)
    ///   on, its `check_ready` must inspect (else the hole reopens).
    /// - **Same error.** Produce the identical [`Error`] variant + message
    ///   `resolve_home` would for the unsatisfiable input
    ///   ([`Input::check_ready`]/[`ScalarInput::check_ready`] do this), so the early
    ///   catch is indistinguishable from the retained execute-time backstop.
    /// - **Pipes are deferred-OK** — see [`Input::check_ready`]: an internal-edge
    ///   pipe is filled at run by an upstream producer (whose own leaves this walk
    ///   already proved ready), and a pre-loaded lent pipe already carries its
    ///   payload; neither is a pre-run completeness failure.
    ///
    /// Default: `Ok(())` — a leaf with no checkable input. Leaves that resolve
    /// inputs override to check their [`Input`]/[`ScalarInput`] cells read-only;
    /// combinators ([`AndThen`], `bundle*`, [`CopyTo2`]) recurse into their children
    /// (mirroring [`describe`](Self::describe)/[`bind_slots`](Self::bind_slots)) so
    /// coverage matches `execute`'s traversal.
    fn check_ready(&self) -> Result<()> {
        Ok(())
    }

    /// Return any of this op's **own output values that were produced but never
    /// delivered** to their home cells — the "undelivered drop" half of the home
    /// invariant for MID-graph producers.
    ///
    /// ## Why this exists
    ///
    /// When an [`and_then`](DeviceOpExt::and_then) closure discards some of a
    /// multi-output source's handles (e.g. `kernel(a, b, out).and_then(|(_a, _b,
    /// out)| download(out))` keeps only `out`), the kernel still DEPOSITS `a`/`b`
    /// into their element pipes at execute. Nothing downstream drains them, so —
    /// without this — those homed buffers would sit in the pipe cells until the
    /// NEXT run's `put_home` overwrites them. That is too late: the next run's
    /// UPSTREAM producer (e.g. the [`upload`] that minted `a`) re-lends from its
    /// home cell at the START of the run, before the kernel re-executes, and finds
    /// it still empty → spurious graph-busy.
    ///
    /// So at the END of each run [`AndThen`] calls `reclaim_undelivered` on its
    /// source, which drains each of the op's output pipes ([`Pipe::take_home`]);
    /// dropping the drained `(value, home)` fires the rehome (the value returns to
    /// its origin cell with a stable handle, or drops if homeless). Idempotent and
    /// cheap when the pipes are already empty (the consumed / single-output case).
    ///
    /// Default: drain the single [`output_pipe`](Self::output_pipe). Multi-output
    /// ops (the macro-emitted kernels, [`CopyTo2`]) override to drain each element
    /// pipe. A consuming/transforming leaf whose output has no home (download's Vec,
    /// host-views) is a no-op in effect — `take_home` yields `home == None`.
    fn reclaim_undelivered(&self) {
        // Drain the output pipe and fire the rehome explicitly: `take_home` moves
        // `value` + `home` OUT of the `PipePayload` (so its `Drop` no longer
        // rehomes), so we must call `rehome_consumed` ourselves. `None` (already
        // drained downstream, or never produced) is a no-op; `home == None` (a
        // minted/transformed output) just releases the value.
        // Single-output only (multi-output ops override this to drain each element
        // pipe); `output_pipe()` is therefore always `Some` here. Use `and_then`
        // so a defensive `None` is simply a no-op rather than a panic.
        if let Some((value, _deps, home)) = self.output_pipe().and_then(|p| p.take_home()) {
            rehome_consumed(value, home);
        }
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

    /// Record this op's device commands into `ctx` — the **non-consuming twin of
    /// [`execute`](Self::execute)** used by the command-buffer path (see
    /// [`crate::record`]). Walk `&self`, resolve inputs from `ctx`'s edge map (a
    /// concrete buffer, or an upstream producer's output keyed by
    /// [`cell_id`](Pipe::cell_id)), emit the op's device commands, and register its
    /// output handle(s) under its output pipe(s) — threading
    /// [`SyncPoints`](crate::record::SyncPoints) where `execute` threads
    /// [`Deps`].
    ///
    /// **Default: not recordable.** Host-touching leaves (`Upload`/`Download`/the
    /// host seams) and any op with no device-command lowering inherit this error,
    /// so a graph that reaches one on the CB path falls back to the per-op execute
    /// path. Device ops (kernels, `fill`, copy, and the structural combinators)
    /// override it. This replaces the former `RecordableOp` sub-trait: recordability
    /// is now a **run-time** property (the default errors) rather than a
    /// compile-time bound, which is what lets the segmenter walk a mixed graph by
    /// `&dyn DeviceOp` and record only its seam-free subtrees.
    fn record(&self, ctx: &mut crate::record::RecordContext) -> Result<()> {
        let _ = ctx;
        Err(Error::NotSupported("op is not device-recordable"))
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
    ///
    /// **Reusable / replayable.** The closure is `Fn` (not `FnOnce`), so a graph
    /// containing a host seam can be `sync`'d repeatedly — the seam re-runs the
    /// closure on every replay. The trade: a replayed closure borrows / `Arc`s /
    /// clones its captures rather than move-consuming them (the right constraint
    /// for something that runs more than once). The engine keeps the closure in an
    /// `Arc` and hands the per-run worker thread its own owned handle.
    fn and_then_host<F>(self, f: F) -> AndThenHost<Self, F>
    where
        Self::Output: crate::mappable::Mappable,
        Self::Checkouts: SeamScatter<Value = Self::Output>,
        F: for<'a> Fn(<Self::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
            + Send
            + Sync
            + 'static,
    {
        AndThenHost {
            source: self,
            f: Arc::new(f),
            handle: <Self::Checkouts as SeamScatter>::empty_handle(),
        }
    }

    /// Like [`and_then_host`](Self::and_then_host) but the closure also receives
    /// the running [`Context`] (e.g. to read device props). See
    /// [`AndThenHostWithContext`].
    ///
    /// **Reusable / replayable** — same as [`and_then_host`](Self::and_then_host):
    /// the closure is `Fn`, the graph replays, and the closure re-runs each
    /// `sync` (borrow / `Arc` / clone captures, don't move-consume them).
    fn and_then_host_with_context<F>(self, f: F) -> AndThenHostWithContext<Self, F>
    where
        Self::Output: crate::mappable::Mappable,
        Self::Checkouts: SeamScatter<Value = Self::Output>,
        F: for<'a> Fn(
                &Context,
                <Self::Output as crate::mappable::Mappable>::View<'a>,
            ) -> Result<()>
            + Send
            + Sync
            + 'static,
    {
        AndThenHostWithContext {
            source: self,
            f: Arc::new(f),
            handle: <Self::Checkouts as SeamScatter>::empty_handle(),
        }
    }

    /// **Set-once** bind of ONE typed slot — **consuming + infallible**. Deposit
    /// `arg`'s binding into the graph's matching [`slot!`](crate::slot) cell(s) and
    /// return the OWNED graph, so binds **chain by move** and the graph is then
    /// `sync`'d: `g.bind(Buf(b)).bind(W(w)).sync(&ctx)?` (the `?` is only at the
    /// terminal — `bind` itself never returns a `Result`).
    ///
    /// **One verb, two sources.** `arg` is any [`CallArg`], so `bind(Tag(value))`
    /// binds by value (set-once) and `bind(Tag(pipe))` installs
    /// [`FedByPipe`](SlotState::FedByPipe) — wiring the slot to an upstream pipe.
    /// This is the single-element form of [`call`](Self::call): `bind(arg)` ==
    /// `call((arg,))`, so it inherits `call`'s consuming, infallible, deferred-error
    /// contract exactly (below).
    ///
    /// **The set-once contract (verb 2×2 — see [`try_bind_slot`](Input::try_bind_slot)):**
    /// a value bind is idempotent on an *equal* binding and an error on a
    /// *conflicting* one.
    ///
    /// - slot [`Unbound`](SlotState::Unbound) (virgin) → fill it.
    /// - slot [`Bound`](SlotState::Bound) to the SAME buffer object
    ///   ([`SlotEq`] handle identity) → no-op (re-handing the buffer you already
    ///   gave is fine).
    /// - slot [`Bound`](SlotState::Bound) to a DIFFERENT buffer →
    ///   [`Error::SlotConflict`]. Use [`mutate_bind`](Self::mutate_bind) to change it.
    /// - slot [`Lent`](SlotState::Lent) (its buffer is checked out to a live
    ///   [`Checkout`]) → [`Error::SlotCheckedOut`] (re-binding would clobber the
    ///   value in the caller's hands).
    /// - slot [`Severed`](SlotState::Severed) (its value was taken via
    ///   [`into_inner`](Checkout::into_inner)) → [`Error::SlotSevered`]:
    ///   re-providing a buffer is a *change*, not a first declaration, so the
    ///   set-once verb rejects it. Use [`mutate_bind`](Self::mutate_bind) to re-arm.
    ///
    /// ## DEFERRED bind errors (record-don't-drop)
    ///
    /// `bind` is INFALLIBLE: it returns owned `Self` (so it fits inside an
    /// [`and_then`](Self::and_then) closure as the bare `U: DeviceOp` and chains by
    /// move) and therefore cannot return the set-once errors above. Instead each is
    /// RECORDED into the graph's [`DeferredErrors`] sink (via [`CallArg::apply`]) and
    /// surfaces at [`sync`](Self::sync) — [`check_ready`](DeviceOp::check_ready)
    /// drains the sink FIRST, before any enqueue, returning the recorded error with
    /// NOTHING run (the atomicity guarantee holds; sound via the state-first drain).
    /// A missing or typo'd bind cannot silently run wrong data — it fails closed at
    /// sync. For the fluent, EAGER-error, `&Self` set/change path use
    /// [`mutate_bind`](Self::mutate_bind) / [`mutate_call`](Self::mutate_call)
    /// (the reuse-loop verbs).
    ///
    /// **Known residual wart (not fixed).** The sink is drained by `pop()` —
    /// REPORT-ONCE. An errored graph fails its FIRST `sync` closed (correct); a
    /// caller that IGNORES that `Err` and re-`sync`s WITHOUT rebinding finds an empty
    /// sink and could then run. That is an abuse path (ignoring a returned error and
    /// re-running) and is benign, but is documented here rather than papered over.
    ///
    /// **Order-free + curryable + partial.** Each `bind` carries exactly one tag,
    /// folded independently, so `g.bind(Buf(b)).bind(W(w))` and the reverse are
    /// equivalent, and a subset is allowed — *completeness* (every slot bound) is
    /// enforced only at [`sync`](Self::sync)/[`wait_on`](Self::wait_on) (runtime),
    /// where an unbound slot is [`Error::SlotUnbound`]. After a run, the run's
    /// [`Checkout`] returns the buffer to the slot cell on drop (same machinery as a
    /// concrete head), so a bound graph is re-runnable.
    ///
    /// The binding is dispatched via [`bind_slots`](DeviceOp::bind_slots), which
    /// walks the op-tree's [`Input`] fields. Ops that route buffer args through
    /// [`Input`] (kernels, `download`/`fill`/`write`/copy, the bundles) propagate
    /// it; a slot placed in an op that does not yet override `bind_slots` simply
    /// stays unbound (caught at `sync`).
    fn bind<A: CallArg>(self, arg: A) -> Self
    where
        Self: Sized,
    {
        self.call((arg,))
    }

    /// **Set/change** bind of one typed slot — the mutating sibling of
    /// [`bind`](Self::bind). Overwrites a [`Bound`](SlotState::Bound) slot (to the
    /// same OR a different buffer), fills an [`Unbound`](SlotState::Unbound) (virgin)
    /// one, AND re-arms a [`Severed`](SlotState::Severed) one (a slot whose value was
    /// taken via [`into_inner`](Checkout::into_inner)); returns `Result<&Self>` so it
    /// chains. Unlike `bind`, it does NOT require a prior bind and never reports
    /// [`SlotConflict`](Error::SlotConflict) or [`SlotSevered`](Error::SlotSevered) —
    /// so the loop case `for x in xs { g.mutate_bind(Buf(x))?.sync(&ctx)?; }` works
    /// without peeling the first iteration, and re-arming after `into_inner` is the
    /// `mutate_bind`-only path.
    ///
    /// The one case it STILL rejects is [`Lent`](SlotState::Lent) — a slot whose
    /// buffer is currently checked out — with [`Error::SlotCheckedOut`]: changing a
    /// value the caller is holding would let the `Checkout`'s drop rehome the old
    /// buffer over the new (a silent clobber). Drop the `Checkout` (re-arm) or
    /// `into_inner` it (sever) first.
    ///
    /// NOTE (step c/d): `mutate_bind` lands here as "overwrite the slot cell"
    /// (`Bound → Bound`, or fill if `Unbound`). The `clUpdateMutableCommandsKHR`
    /// in-place mutable-dispatch routing — reusing the same command buffer and only
    /// re-pointing the arg — attaches at the **segment-plan** step; it is NOT
    /// implemented here.
    fn mutate_bind<Tg: Tag>(&self, tag: Tg) -> Result<&Self>
    where
        Tg::Value: SlotEq + SlotValue,
    {
        self.fold_bind::<Tg>(tag.into_value(), BindMode::Mutate)
    }

    /// Shared body of [`bind`](Self::bind) / [`mutate_bind`](Self::mutate_bind):
    /// resolve the tag binding `Tag(source)` to its value via [`into_value`](Tag::into_value)
    /// (raw passes through; a [`Checkout`] is `into_inner`'d — severing its source
    /// home), box it into a binder keyed on `TypeId::of::<Tg::Key>()` in `mode`, fold
    /// it into the graph's slot cells, and surface the verb-2×2 verdict
    /// ([`SlotBinder::outcome`]). Callers pass the already-resolved value (the tag's
    /// `into_value` runs at the `bind`/`call` site).
    fn fold_bind<Tg: Tag>(&self, value: Tg::Value, mode: BindMode) -> Result<&Self>
    where
        Tg::Value: SlotEq + SlotValue,
    {
        let mut binder = SlotBinder::new::<Tg>(value, mode);
        self.bind_slots(&mut binder);
        binder.outcome()?;
        // AT-LEAST-ONE: a `bind` that matched ZERO cells is a hard error (typo'd
        // tag, or a tag not used in THIS graph). Without this it would silently
        // succeed here and only surface — if at all — as `SlotUnbound` at `sync`.
        // Checked only when `outcome` is Ok: a conflict/sever already counts as a
        // match (the tag IS present), so it errors above, not here.
        if binder.matched() == 0 {
            // Clean tag ident (`Tg::NAME`), not `type_name::<Tg::Key>()` — the
            // latter would leak the internal `<KeyMarker>` suffix into the message.
            return Err(Error::SlotNoSuchTag(Tg::NAME));
        }
        Ok(self)
    }

    /// **Read-only probe** — would a [`fold_bind`](Self::fold_bind) of tag `Tg` in
    /// `mode` SUCCEED right now, WITHOUT filling / severing / mutating ANYTHING?
    /// The phase-0 dry run behind [`call`](Self::call) / [`mutate_call`](Self::mutate_call)'s
    /// all-or-nothing guarantee: it drives the SAME [`bind_slots`](DeviceOp::bind_slots)
    /// walk with a value-less probe [`SlotBinder`], so it covers exactly the cells the
    /// real fold would touch — but only INSPECTS state.
    ///
    /// `severable_cells` is the set of slot-cell ids ([`Arc::as_ptr`] as `usize`)
    /// that phase 1 WILL sever — one per [`Checkout`]-sourced element in the same
    /// tuple. It lets the probe predict a crossed swap: a `Lent` target in that set
    /// becomes [`Severed`](SlotState::Severed) before the fold ([`Mutate`](BindMode::Mutate)
    /// re-arms it → OK), while a `Lent` target held by an EXTERNAL live `Checkout`
    /// stays `Lent` → [`Error::SlotCheckedOut`].
    ///
    /// Returns the SAME error the fold would ([`SlotNoSuchTag`](Error::SlotNoSuchTag)
    /// on absent, [`SlotCheckedOut`](Error::SlotCheckedOut) on external-lent,
    /// [`SlotSevered`](Error::SlotSevered) on `Set` of a severed / to-be-severed
    /// slot), with ONE documented exception: the value-dependent
    /// [`SlotConflict`](Error::SlotConflict) of a `Set` onto an already-`Bound`
    /// (different) slot is NOT pre-caught — the value is inside an unsevered
    /// `Checkout` and the probe never severs to read it. Only `Tg: Tag` is required
    /// (no `SlotEq`/`SlotValue`, no `'static` value bound), so it does not hit the
    /// value-comparison lifetime constraints the real fold carries.
    fn probe_bind<Tg: Tag>(&self, mode: BindMode, severable_cells: &[usize]) -> Result<()> {
        let mut binder = SlotBinder::probe::<Tg>(mode, severable_cells.to_vec());
        self.bind_slots(&mut binder);
        binder.outcome()?;
        // AT-LEAST-ONE, checked read-only up front — this is the "sever everything
        // then SlotNoSuchTag" bug's pre-catch: an absent tag errors here, having
        // touched nothing.
        if binder.matched() == 0 {
            return Err(Error::SlotNoSuchTag(Tg::NAME));
        }
        Ok(())
    }

    /// **Fill several slots at once**, turbofish-free — **consuming + infallible**.
    /// Each element self-tags via its tuple struct, is applied left-to-right, and the
    /// OWNED graph is returned: `let g = g.call((A(a), B(b), Out(o))); g.sync(&ctx)?`
    /// (the `?` is only at the terminal — `call` itself never returns a `Result`).
    ///
    /// **Mixed value-or-feed.** Each element is INDEPENDENTLY a **value tag**
    /// (`Buf(b)`, `Factor(2)` — set-once value bind) or a **pipe feed** (`Buf(pipe)`
    /// — the same tag constructor fed a [`Pipe`], installing
    /// [`FedByPipe`](SlotState::FedByPipe)), so one tuple freely MIXES concrete binds
    /// and upstream-pipe wiring (the crossed rotation visible in the arg list). This
    /// is dispatched through [`CallArgs`] / [`CallArg`] (arity 1..=8), NOT the fluent
    /// [`BindAll`] path (which now serves only [`mutate_bind`](Self::mutate_bind) /
    /// [`mutate_call`](Self::mutate_call)).
    ///
    /// ## Why consuming + infallible
    ///
    /// It returns owned `Self`, so it (a) chains further (`.call(..).and_then(..)`)
    /// AND (b) is usable INSIDE an [`and_then`](Self::and_then) closure as the bare
    /// `U: DeviceOp` that `FnOnce(Handle) -> U` requires. A *fallible* `-> Result<Self>`
    /// would force the closure to return `Result<U>`, which `and_then` does NOT accept.
    ///
    /// ## DEFERRED bind errors — the trade
    ///
    /// Because it is infallible, an absent tag ([`SlotNoSuchTag`](Error::SlotNoSuchTag)),
    /// a [`SlotConflict`](Error::SlotConflict), a [`SlotCheckedOut`](Error::SlotCheckedOut),
    /// or a [`SlotSevered`](Error::SlotSevered) is not returned at the call site — it is
    /// RECORDED (via [`CallArg::apply`] → [`bind_deferred`](Self::bind_deferred) /
    /// [`feed_deferred`](Self::feed_deferred)) into the graph's [`DeferredErrors`] sink
    /// and surfaces at [`sync`](Self::sync): [`check_ready`](DeviceOp::check_ready)
    /// reads the sink FIRST, before any enqueue, returning the recorded error with
    /// NOTHING run (the atomicity guarantee holds). A typo'd or missing bind cannot
    /// silently run wrong data — it fails closed at sync.
    ///
    /// **Sticky / poison + recovery.** A recorded deferred error POISONS the graph:
    /// `check_ready` PEEKS the sink (does not drain it), so EVERY subsequent `sync`
    /// re-reports the same error — a caller cannot ignore it and re-`sync` into a run.
    /// The recovery is to REBUILD the graph (the factory idiom; graphs are cheap). See
    /// [`DeferredErrors`]. Contrast the fluent [`mutate_bind`](Self::mutate_bind) /
    /// [`mutate_call`](Self::mutate_call): they fail EAGERLY at the call site and never
    /// touch the sink, so a failed mutate leaves the graph unpoisoned and reusable.
    ///
    /// ## Partial binds (currying)
    ///
    /// `call` binds ONLY the tags in `args` and leaves the rest of the graph's slots
    /// as they are (`Unbound` / `FedByPipe`-able for a later `call`). So it is
    /// naturally a PARTIAL bind — bind a subset now (e.g. the invariants), the rest
    /// later. Any slot still unbound at `sync` errors there (deferred, as above).
    ///
    /// Tuple ORDER does not matter for success (each binds its own tag).
    fn call<T: CallArgs>(self, args: T) -> Self
    where
        Self: Sized,
    {
        args.apply_all(&self);
        self
    }

    /// **Mutate several slots at once** — the mutating sibling of [`call`](Self::call).
    /// Each element folds through the [`mutate_bind`](Self::mutate_bind) path, so it
    /// inherits the set/change semantics (overwrite or fill; still
    /// [`SlotCheckedOut`](Error::SlotCheckedOut) on a slot lent to an EXTERNAL
    /// checkout).
    ///
    /// ## Fully all-or-nothing (probe before sever)
    ///
    /// Like [`call`](Self::call), a phase-0 [`probe_bind`](Self::probe_bind) vets
    /// every element BEFORE any source is severed — but `mutate` has no
    /// [`SlotConflict`](Error::SlotConflict) leg (it overwrites), so `mutate_call` has
    /// NO residual: an absent tag ([`SlotNoSuchTag`](Error::SlotNoSuchTag)) or an
    /// external-checkout target ([`SlotCheckedOut`](Error::SlotCheckedOut)) errors
    /// with the graph AND every source left untouched.
    ///
    /// The crossed double-buffer swap `mutate_call((In(out_co), Out(in_co)))` passes
    /// the probe: after a sync `In`/`Out` are `Lent`, but their leases are held by the
    /// tuple's own Checkouts (`out_co`/`in_co`), whose cell ids the probe collects —
    /// so it predicts phase 1 will sever both (`Lent → Severed`) and that the two
    /// `mutate` rebinds then re-arm cleanly, rather than misreading `Lent` as an
    /// external [`SlotCheckedOut`](Error::SlotCheckedOut).
    ///
    /// NOTE (step c/d): like [`mutate_bind`](Self::mutate_bind), the in-place
    /// mutable-dispatch (`clUpdateMutableCommandsKHR`) routing attaches at the
    /// segment-plan step and is not implemented here.
    fn mutate_call<Tags: BindAll>(&self, tags: Tags) -> Result<&Self> {
        tags.mutate_all(self)?;
        Ok(self)
    }

    /// **Deferred (record-don't-drop) value-bind** — the INFALLIBLE
    /// [`call`](Self::call) sibling of a fluent value bind, used ONLY by
    /// [`CallArg::apply`]. It folds the tag exactly like a set-once bind (same
    /// verb-2×2), but instead of RETURNING the error it RECORDS it into the graph's
    /// [`DeferredErrors`] sink so [`check_ready`](DeviceOp::check_ready) surfaces it at
    /// `sync` (FIRST, nothing enqueued). This closes the silent-swallow hole the old
    /// `let _ = g.bind(..)` left: a `SlotConflict` (cell left `Bound` to the OLD
    /// value), a `SlotNoSuchTag` (no cell at all), or a `SlotCheckedOut`/`SlotSevered`
    /// no longer vanishes — `check_ready` sees the recorded error and fails closed. A
    /// clean bind records nothing (sink stays empty; reuse path unchanged).
    fn bind_deferred<Tg: Tag>(&self, tag: Tg)
    where
        Tg::Value: SlotEq + SlotValue,
    {
        let mut binder = SlotBinder::new::<Tg>(tag.into_value(), BindMode::Set);
        binder.mark_deferred();
        self.bind_slots(&mut binder);
        binder.record_deferred(Tg::NAME);
    }

    /// **Deferred (record-don't-drop) pipe-feed** — the INFALLIBLE
    /// [`call`](Self::call) sibling of a value bind for the `Tag(pipe)` source, used
    /// ONLY by [`CallArg::apply`] for a `Tag(pipe)` element. Installs `FedByPipe` at
    /// every matching site, but an absent tag (`matched == 0`) is
    /// RECORDED as [`SlotNoSuchTag`](Error::SlotNoSuchTag) into the graph's
    /// [`DeferredErrors`] sink (drained by `check_ready`) rather than dropped. (A feed
    /// install never conflicts, so `SlotNoSuchTag` is its only failure.)
    fn feed_deferred<Tg: Tag>(&self, pipe: Pipe<Tg::Value>) {
        let mut binder = SlotBinder::feed::<Tg>(pipe);
        binder.mark_deferred();
        self.bind_slots(&mut binder);
        binder.record_deferred(Tg::NAME);
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
        // ATOMICITY PRE-PASS — validate that EVERY input cell of the whole graph is
        // satisfiable BEFORE touching anything. This is the read-only mirror of the
        // resolve_home checks `execute` does inline; running it FIRST means a graph
        // with an unsatisfiable input (empty concrete cell / unbound-or-lent slot /
        // unbound scalar) errors having LENT no buffer and ENQUEUED no command, so
        // the graph is left unchanged + re-runnable. `execute`'s own resolve_home
        // checks stay as the safety backstop. (Done before building `ExecutionContext`
        // and the start gate — neither has run yet, so there is nothing to unwind.)
        self.check_ready()?;

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
                    // Mop up the run's UNDELIVERED intermediates (multi-output
                    // values an `and_then` closure discarded): return each homed
                    // buffer to its origin cell now, so the NEXT run's upstream
                    // re-lend finds it. The terminal's OWN output was already drained
                    // into `checkouts`, so this only touches intermediates.
                    self.reclaim_undelivered();
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
        // Mop up undelivered intermediates (see the fast-path note) so a reused
        // graph re-arms its upstream cells.
        self.reclaim_undelivered();

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
        // Atomicity pre-pass: validate all inputs before any enqueue, mirroring wait_on.
        self.check_ready()?;
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
        // Atomicity pre-pass: validate all inputs before any enqueue, mirroring wait_on.
        self.check_ready()?;
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

// ── BindAll: fill a tuple of slots in one call (`g.call((A(a), B(b)))`) ──────

/// A tuple of [`Tag`]s bindable in one [`call`](DeviceOpExt::call) /
/// [`mutate_call`](DeviceOpExt::mutate_call). Each element self-tags via its tuple
/// struct (`A(a)`, `B(b)`, …) — turbofish-free — and folds through the SAME
/// single-slot [`bind`](DeviceOpExt::bind) / [`mutate_bind`](DeviceOpExt::mutate_bind)
/// path, so the multi-fill inherits the verb 2×2 (set-once / set-change, conflict,
/// checked-out) element-by-element.
///
/// **All-or-nothing (probe before sever).** A phase-0 read-only
/// [`probe_bind`](DeviceOpExt::probe_bind) vets EVERY element BEFORE any
/// [`into_value`](Tag::into_value) severs a `Checkout` source — so an absent tag
/// ([`SlotNoSuchTag`](Error::SlotNoSuchTag)), an externally-checked-out target
/// ([`SlotCheckedOut`](Error::SlotCheckedOut)), or a `Set`-onto-severed slot
/// ([`SlotSevered`](Error::SlotSevered)) leaves the graph AND every source
/// UNTOUCHED. This is stronger than the old "sever every source up front, THEN
/// `?`-chained fold", which could sever all Checkouts and then error.
///
/// **One residual, `bind_all` only.** The value-dependent
/// [`SlotConflict`](Error::SlotConflict) of a `Set` onto an already-`Bound`
/// (different) slot cannot be pre-caught (the value is inside an unsevered
/// `Checkout`); it fires in phase 2 after phase 1 may have severed OTHER sources.
/// It is a set-once misuse case. `mutate_all` has no `SlotConflict` leg, so it is
/// FULLY all-or-nothing. Order does not matter for *success*: every element binds
/// its own tag independently. Implemented for tuples of arity 1..=8 (mirroring
/// [`KernelArgs`](crate::KernelArgs)). See the `bind_all_body!` macro for the three
/// phases (probe → sever → fold).
pub trait BindAll {
    /// Fold each element through [`bind`](DeviceOpExt::bind) (set-once).
    fn bind_all<Op: DeviceOp>(self, g: &Op) -> Result<()>;
    /// Fold each element through [`mutate_bind`](DeviceOpExt::mutate_bind) (set/change).
    fn mutate_all<Op: DeviceOp>(self, g: &Op) -> Result<()>;
}

/// The shared three-phase body of [`BindAll::bind_all`] / [`BindAll::mutate_all`],
/// parameterised on the [`BindMode`]. Splitting it out keeps the two verbs a single
/// source of truth for the all-or-nothing sequence.
///
/// - **PHASE 0 — probe (read-only, severs NOTHING).** First gather the
///   `severable_cells` set: the slot-cell id every `Checkout`-sourced element will
///   sever in phase 1 ([`Tag::source_cell_id`]; a raw value contributes `None`).
///   Then [`probe_bind`](DeviceOpExt::probe_bind) EVERY element against `g` with that
///   set. A probe drives the real `bind_slots` walk but only INSPECTS state, so if
///   ANY element is absent ([`SlotNoSuchTag`](Error::SlotNoSuchTag)), held by an
///   external checkout ([`SlotCheckedOut`](Error::SlotCheckedOut)), or `Set`-onto-a-
///   severed slot ([`SlotSevered`](Error::SlotSevered)), we return that error HERE —
///   having severed / mutated NOTHING. This closes the "`into_value` severs every
///   source, THEN the fold errors" hole. (A crossed swap's `Lent` targets are in
///   `severable_cells`, so the probe passes them; see [`SlotBinder::probe_lent`].)
///
///   The ONE residual the probe cannot pre-catch is a `Set` (bind) onto an already-
///   `Bound`-DIFFERENT slot ([`SlotConflict`](Error::SlotConflict)): the conflicting
///   value lives inside an unsevered `Checkout` and the probe never severs to read
///   it. Such a conflict still fires in phase 2 — after phase 1 may have severed
///   OTHER elements' sources. This is a misuse case (`bind` is set-once; re-binding
///   an already-bound slot to a different buffer is the error), and it never affects
///   the motivating paths (the crossed swap uses `mutate`, absent-tag / checked-out
///   are fully covered). `mutate` has no `SlotConflict` leg, so `mutate_call` is
///   fully all-or-nothing.
///
/// - **PHASE 1 — sever all sources (`into_value`).** With the probe having proved
///   every element bindable, resolve each source: a `Checkout` severs its home
///   HERE (`Lent → Severed`), before any fold. This is what makes the crossed swap
///   `mutate_call((In(out_co), Out(in_co)))` work — both source slots are `Severed`
///   BEFORE either target is rebound, so neither rebind hits a still-`Lent` slot.
///
/// - **PHASE 2 — fold each resolved value.** `?` keeps first-error-stops semantics;
///   with phase 0 already vetted, the only error phase 2 can now surface is the
///   documented `Set`/`SlotConflict` residual.
macro_rules! bind_all_body {
    ($g:ident, $mode:expr, $($name:ident),+) => {{
        // PHASE 0a — the crossed-swap recogniser: which slot cells phase 1 will
        // sever (Checkout sources contribute their home cell id; raw values `None`).
        // Read-only — `source_cell_id` borrows, never consumes.
        let severable: Vec<usize> =
            [ $( $name.source_cell_id() ),+ ].into_iter().flatten().collect();
        // PHASE 0b — probe EVERY element (read-only). Any failure returns here
        // having severed / mutated NOTHING (the all-or-nothing guarantee).
        $( $g.probe_bind::<$name>($mode, &severable)?; )+
        // PHASE 1 — sever all Checkout sources first (see macro doc).
        $( let $name = $name.into_value(); )+
        // PHASE 2 — fold each resolved value; `?` stops at the first (now only the
        // residual Set/SlotConflict) error.
        $( $g.fold_bind::<$name>($name, $mode)?; )+
    }};
}

/// Implement [`BindAll`] for one tuple arity. Each element is a `Tag` whose `Value`
/// is `SlotEq` (the buffer-handle equality the set-once leg needs).
macro_rules! impl_bind_all_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name),+> BindAll for ($($name,)+)
        where
            $( $name: Tag, $name::Value: SlotEq + SlotValue, )+
        {
            #[allow(non_snake_case)]
            fn bind_all<Op: DeviceOp>(self, g: &Op) -> Result<()> {
                let ($($name,)+) = self;
                bind_all_body!(g, BindMode::Set, $($name),+);
                Ok(())
            }
            #[allow(non_snake_case)]
            fn mutate_all<Op: DeviceOp>(self, g: &Op) -> Result<()> {
                let ($($name,)+) = self;
                bind_all_body!(g, BindMode::Mutate, $($name),+);
                Ok(())
            }
        }
    };
}
impl_bind_all_tuple!(A);
impl_bind_all_tuple!(A, B);
impl_bind_all_tuple!(A, B, C);
impl_bind_all_tuple!(A, B, C, D);
impl_bind_all_tuple!(A, B, C, D, E);
impl_bind_all_tuple!(A, B, C, D, E, F);
impl_bind_all_tuple!(A, B, C, D, E, F, G);
impl_bind_all_tuple!(A, B, C, D, E, F, G, H);

// ── CallArg / CallArgs: the mixed value-or-feed tuple for `bind` / `call` ────

/// One element of a [`call`](DeviceOpExt::call) (or single-element
/// [`bind`](DeviceOpExt::bind)) tuple — EITHER a **value tag** (`Buf(b)`, `Factor(2)`)
/// that binds by value, OR a **pipe feed** (`Buf(pipe)` — the SAME tag constructor fed
/// a [`Pipe`] instead of a value) that wires the tag's slot(s) to an upstream pipe.
///
/// The two spellings share one surface: `Buf(x)` is a value-bind when `x` is a value
/// (or [`Checkout`]) and a pipe-feed when `x` is a `Pipe<Buf::Value>`. All three arms
/// are emitted PER-TAG by the [`slots!`](crate::slots) macro on the CONCRETE source
/// (`$name<$val>` and `$name<Checkout<$val>>` value-bind, `$name<Pipe<V>>` pipe-feed)
/// — deliberately NOT an open `impl<Tg: Tag> CallArg for Tg` blanket, which would
/// break cross-crate coherence against the pipe impl (see the module comment below).
/// The pipe-feed arm is gated to buffer-valued tags via
/// [`RecordableBuffer`](crate::record::RecordableBuffer), so a scalar `F(pipe)` does
/// not compile.
///
/// Applied **infallibly**: [`apply`](CallArg::apply) RECORDS any bind error (an absent
/// tag, a conflict, a checked-out slot) into the graph's [`DeferredErrors`] sink — the
/// error surfaces later at [`sync`](DeviceOpExt::sync) via
/// [`check_ready`](DeviceOp::check_ready) instead. This is the trade
/// [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call) make to be infallible +
/// consuming (usable as the bare `U` an [`and_then`](DeviceOpExt::and_then) closure
/// requires); see those methods' docs for the full rationale.
pub trait CallArg {
    /// Apply this element's binding to graph `g`, INFALLIBLY (errors deferred to
    /// `sync`). A value tag folds through [`bind_deferred`](DeviceOpExt::bind_deferred)
    /// (set-once); a pipe-source tag folds through
    /// [`feed_deferred`](DeviceOpExt::feed_deferred).
    fn apply<Op: DeviceOp>(self, g: &Op);
}

// The concrete `CallArg` impls (value-bind for the raw-value and `Checkout` sources,
// pipe-feed for the `Pipe` source) are emitted PER-TAG by the [`slots!`](crate::slots)
// macro — NOT as an open `impl<Tg: Tag> CallArg for Tg` blanket. This is a
// CROSS-CRATE COHERENCE requirement: an open blanket over `Tg: Tag` would collide with
// the per-tag pipe impl `CallArg for $name<Pipe<V>>` for SCALAR-valued tags, because
// the compiler must assume this (upstream) crate could later add both
// `IntoBound<f32> for Pipe<f32>` (making `F<Pipe<f32>>: Tag`, matching the blanket) and
// `RecordableBuffer for f32` (inhabiting the pipe impl) — a potential future overlap it
// rejects TODAY. Emitting VALUE-bind on the two CONCRETE non-pipe sources
// (`$name<$val>`, `$name<Checkout<$val>>`) instead makes the three source shapes
// (`$val` / `Checkout` / `Pipe`) structurally disjoint type constructors that NO
// upstream impl can ever unify — so the coherence holds unconditionally. The two
// concrete value-bind arms exactly cover the two existing `IntoBound` sources (identity
// `V` and `Checkout<V>`); there are no others.

/// A tuple of [`CallArg`]s — the argument to [`call`](DeviceOpExt::call) (and, as a
/// 1-tuple, [`bind`](DeviceOpExt::bind)). Each element is INDEPENDENTLY a value tag
/// (`Buf(v)`) or a pipe feed (`Buf(pipe)`), applied left-to-right, infallibly.
/// Implemented for arities 1..=8 (mirroring [`BindAll`]).
pub trait CallArgs {
    /// Apply every element to `g` (infallibly, in tuple order).
    fn apply_all<Op: DeviceOp>(self, g: &Op);
}

/// Implement [`CallArgs`] for one tuple arity. Each element is any [`CallArg`]
/// (value tag OR feed), so the tuple is freely MIXED.
macro_rules! impl_call_args_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name: CallArg),+> CallArgs for ($($name,)+) {
            #[allow(non_snake_case)]
            fn apply_all<Op: DeviceOp>(self, g: &Op) {
                let ($($name,)+) = self;
                $( $name.apply(g); )+
            }
        }
    };
}
impl_call_args_tuple!(A);
impl_call_args_tuple!(A, B);
impl_call_args_tuple!(A, B, C);
impl_call_args_tuple!(A, B, C, D);
impl_call_args_tuple!(A, B, C, D, E);
impl_call_args_tuple!(A, B, C, D, E, F);
impl_call_args_tuple!(A, B, C, D, E, F, G);
impl_call_args_tuple!(A, B, C, D, E, F, G, H);

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
    /// back to `g`. For a SLOT this lands in [`Severed`](SlotState::Severed): a
    /// later set-once `bind` is [`Error::SlotSevered`] (re-providing a buffer is a
    /// change, not a first declaration), and only `mutate_bind` re-arms it.
    /// The **identity of the slot cell this checkout's home will sever**
    /// ([`Arc::as_ptr`] as `usize`), or `None` if the home is a concrete cell / there
    /// is nothing to return. Read-only: it does NOT consume, sever, or peek the
    /// value.
    ///
    /// Feeds the `call`/`mutate_call` phase-0 probe: an element built from
    /// `Tag(checkout)` contributes this id to the probe's `severable_cells`, so a
    /// `Lent` target slot the probe finds can be recognised as "held by a tuple
    /// Checkout that phase 1 will sever" (a crossed swap) versus "held by an external
    /// live Checkout" (a real [`Error::SlotCheckedOut`]).
    pub(crate) fn home_cell_id(&self) -> Option<usize> {
        self.home.as_ref().and_then(|h| h.home_cell_id())
    }

    pub fn into_inner(mut self) -> O {
        // Sever the home (does NOT deposit the value): a concrete cell stays
        // empty (no-op), a SLOT cell transitions `Lent → Severed` so a later
        // set-once `bind` sees a severed (not virgin, not stuck-`Lent`) slot and
        // rejects it (`Error::SlotSevered`); `mutate_bind` re-arms it. Then take
        // the value out for the caller.
        if let Some(home) = self.home.take() {
            home.sever();
        }
        self.value
            .take()
            .expect("Checkout::into_inner after value already taken — internal bug")
    }

    /// **Lend** the output onward WITHOUT severing — the implicit
    /// "feed a `Checkout` forward as a borrow input to a SECOND graph" path.
    /// Unlike [`into_inner`](Self::into_inner) (which fires
    /// [`sever`](Rehome::sever) → a slot goes `Lent → Severed`, a concrete cell is
    /// left empty for good), this MOVES the `(value, home)` pair out intact: the
    /// home is RELOCATED, not fired, so the origin cell stays `Lent` (busy) and the
    /// value rides into the next graph carrying its return obligation. When that
    /// value finally drops (the second graph's terminal `Checkout`, or an
    /// undelivered [`PipePayload`] drop), the SAME home re-arms the origin cell
    /// transparently — `g.sync()` again, NO `mutate_bind`.
    ///
    /// Suppressing this `Checkout`'s own `Drop` is automatic: we drain BOTH
    /// `Option`s, so the subsequent drop sees `value == None` and returns early
    /// (it never reaches `home.rehome` / never fires `sever`). The single home now
    /// lives in the caller's pipe (`BoxedHome: !Clone`), so it fires exactly once.
    pub(crate) fn into_value_and_home(mut self) -> (O, Option<BoxedHome<O>>) {
        let value = self
            .value
            .take()
            .expect("Checkout::into_value_and_home after value already taken — internal bug");
        let home = self.home.take();
        (value, home)
        // `self` drops here: both `value` and `home` are now `None`, so the
        // `Drop` impl's `let Some(value) = self.value.take() else { return }`
        // short-circuits — no rehome, no sever. The home was MOVED, not fired.
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
//
// STRUCTURAL / RECURSIVE: each element `$ck` is only required to be
// `FromCheckout<$ty>` — NOT literally `Checkout<$ty>`. For a single-output
// branch that element IS `Checkout<$ty>` (via the identity impl above); for a
// **multi-output** branch it is that branch's OWN `Checkouts` (itself a tuple,
// satisfying `FromCheckout` via this same family one level down). So a
// `bundle!`'s `Checkouts = (A::Checkouts, B::Checkouts, …)` satisfies
// `FromCheckout<(A::Output, B::Output, …)>` at ANY branch multiplicity and
// nesting depth — the delegation composes transitively. No overlap with the
// identity impl: a `Checkout<O>` is never a tuple type.
macro_rules! impl_from_checkout_tuple {
    ( $( $ck:ident : $ty:ident ),+ ) => {
        impl<$($ck, $ty,)+> FromCheckout<( $($ty,)+ )> for ( $($ck,)+ )
        where
            $( $ck: FromCheckout<$ty>, )+
        {
            fn from_single(_co: Checkout<( $($ty,)+ )>) -> Self {
                unreachable!(
                    "multi-output graphs build their Checkouts tuple in \
                     gather_checkouts; from_single is never called"
                )
            }
        }
    };
}
impl_from_checkout_tuple!(CA: A, CB: B);
impl_from_checkout_tuple!(CA: A, CB: B, CC: C);
impl_from_checkout_tuple!(CA: A, CB: B, CC: C, CD: D);
impl_from_checkout_tuple!(CA: A, CB: B, CC: C, CD: D, CE: E);
impl_from_checkout_tuple!(CA: A, CB: B, CC: C, CD: D, CE: E, CF: F);
impl_from_checkout_tuple!(CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G);
impl_from_checkout_tuple!(CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G, CH: H);
impl_from_checkout_tuple!(CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G, CH: H, CI: I);
impl_from_checkout_tuple!(CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G, CH: H, CI: I, CJ: J);
impl_from_checkout_tuple!(
    CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G, CH: H, CI: I, CJ: J, CK: K
);
impl_from_checkout_tuple!(
    CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G, CH: H, CI: I, CJ: J, CK: K, CL: L
);
impl_from_checkout_tuple!(
    CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G, CH: H, CI: I, CJ: J, CK: K, CL: L, CM: M
);
impl_from_checkout_tuple!(
    CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G, CH: H, CI: I, CJ: J, CK: K, CL: L, CM: M, CN: N
);
impl_from_checkout_tuple!(
    CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G, CH: H, CI: I, CJ: J, CK: K, CL: L, CM: M,
    CN: N, CO: O
);
impl_from_checkout_tuple!(
    CA: A, CB: B, CC: C, CD: D, CE: E, CF: F, CG: G, CH: H, CI: I, CJ: J, CK: K, CL: L, CM: M,
    CN: N, CO: O, CP: P
);

// Same for the homogeneous `[C; N]` shape produced by `arc_split` — `C` need
// only be `FromCheckout<O>` (an arc_split branch is single-output, so `C =
// Checkout<O>`, but keeping it structural mirrors the tuple family).
impl<C, O, const N: usize> FromCheckout<[O; N]> for [C; N]
where
    C: FromCheckout<O>,
{
    fn from_single(_co: Checkout<[O; N]>) -> Self {
        unreachable!(
            "arc_split builds its [Checkout; N] in gather_checkouts; \
             from_single is never called"
        )
    }
}

// The DYNAMIC-length homogeneous `Vec<C>` shape produced by [`FanOut`] — the
// dynamic-arity analog of the `[C; N]` impl above. `C` need only be
// `FromCheckout<O>` (a fan-out branch may itself be multi-output, so `C =
// U::Checkouts`, a tuple, satisfying `FromCheckout` via the recursive tuple
// family; a single-output branch's `C = Checkout<O>` via the identity impl).
// Never reaches `from_single`: `FanOut` overrides `gather_checkouts` to build the
// `Vec` per-branch. No overlap with the array/tuple/identity impls (`Vec` is a
// distinct nominal type).
impl<C, O> FromCheckout<Vec<O>> for Vec<C>
where
    C: FromCheckout<O>,
{
    fn from_single(_co: Checkout<Vec<O>>) -> Self {
        unreachable!(
            "fan_out builds its Vec<Checkouts> in gather_checkouts; \
             from_single is never called"
        )
    }
}

/// The **bidirectional companion** to [`FromCheckout`]: SPLIT a
/// [`Checkouts`](DeviceOp::Checkouts) value into its assembled output value plus
/// its per-branch return homes, and REASSEMBLE it from a (possibly
/// seam-mutated) value + those same homes.
///
/// [`FromCheckout`] only goes one way — assemble a `Checkouts` from a single
/// `Checkout`. The **host seam** ([`AndThenHost`]) needs the reverse too. To run
/// its closure over the whole `S::Output` (which for a bundle/multi-output source
/// is a TUPLE) it must:
/// - (a) pull the assembled tuple VALUE out of the source's per-branch
///   `Checkout`s WITHOUT dropping their homes (so the buffers can be mapped +
///   written in place), and
/// - (b) after the seam has run, rebuild the `Checkouts` re-threading each
///   branch's ORIGINAL home, so every branch re-arms its own origin cell on drop
///   — the multi-home replay the single collapsed [`collect_home`](DeviceOp::collect_home)
///   slot cannot carry.
///
/// Implemented for `Checkout<O>` (identity: one value + one home) and,
/// recursively, for tuples of `CheckoutSplit`s — so the seam is arity- and
/// nesting-general (a nested bundle's `Checkouts` is a tuple-of-tuples, split by
/// this same family one level down). No `[C; N]` impl: an array `Checkouts`
/// (only `arc_split`) has `Output = [O; N]`, which is not `Mappable`, so it can
/// never feed a seam — the impl would be dead code.
pub trait CheckoutSplit {
    /// The assembled output value these checkouts wrap — equals the producing
    /// op's [`Output`](DeviceOp::Output).
    type Value;
    /// The per-branch return homes, in a shape that reassembles 1:1 with
    /// [`Value`](Self::Value).
    type Homes;
    /// Move out the assembled value + the homes intact. The homes are
    /// **relocated, not fired** (the value+home pair is moved out of each leaf
    /// `Checkout` intact, NOT severed) so [`reassemble`](Self::reassemble) can
    /// re-thread them.
    fn split(self) -> (Self::Value, Self::Homes);
    /// Rebuild the checkouts from a (possibly seam-mutated) value + the ORIGINAL
    /// homes, so each element re-arms its origin cell on drop.
    fn reassemble(value: Self::Value, homes: Self::Homes) -> Self;
}

impl<O: Send> CheckoutSplit for Checkout<O> {
    type Value = O;
    type Homes = Option<BoxedHome<O>>;
    fn split(self) -> (O, Option<BoxedHome<O>>) {
        // Relocate value+home out intact (does NOT sever / fire the home).
        self.into_value_and_home()
    }
    fn reassemble(value: O, home: Option<BoxedHome<O>>) -> Self {
        Checkout::new(value, home)
    }
}

// Recursive tuple family: a bundle / multi-output branch's `Checkouts` is a
// tuple, and each element is itself a `CheckoutSplit` (a `Checkout` for a
// single-output branch, or another tuple for a nested bundle / multi-output
// branch). Split/reassemble descend structurally, so re-threading works at any
// nesting depth. Arity 2..=16 mirrors the `FromCheckout` tuple family.
macro_rules! impl_checkout_split_tuple {
    ( $( $ck:ident : $vn:ident : $hn:ident : $idx:tt ),+ ) => {
        impl<$($ck: CheckoutSplit,)+> CheckoutSplit for ( $($ck,)+ ) {
            type Value = ( $(<$ck as CheckoutSplit>::Value,)+ );
            type Homes = ( $(<$ck as CheckoutSplit>::Homes,)+ );
            fn split(self) -> (Self::Value, Self::Homes) {
                $( let ($vn, $hn) = self.$idx.split(); )+
                ( ($($vn,)+), ($($hn,)+) )
            }
            fn reassemble(value: Self::Value, homes: Self::Homes) -> Self {
                ( $( <$ck as CheckoutSplit>::reassemble(value.$idx, homes.$idx), )+ )
            }
        }
    };
}
impl_checkout_split_tuple!(CA: va: ha: 0, CB: vb: hb: 1);
impl_checkout_split_tuple!(CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2);
impl_checkout_split_tuple!(CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11,
    CM: vm: hm: 12
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11,
    CM: vm: hm: 12, CN: vn: hn: 13
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11,
    CM: vm: hm: 12, CN: vn: hn: 13, CO: vo: ho: 14
);
impl_checkout_split_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11,
    CM: vm: hm: 12, CN: vn: hn: 13, CO: vo: ho: 14, CP: vp: hp: 15
);

/// The **MID-GRAPH companion** to [`CheckoutSplit`]: give a host seam
/// ([`AndThenHost`]) its own per-branch **element pipes** — shaped like the
/// source's [`Checkouts`](DeviceOp::Checkouts) — and SCATTER a seam-mutated value
/// plus per-branch homes into them, so a bundle / multi-output source's branches
/// stay individually consumable **downstream** (a `Pipe` per branch, not one
/// `Pipe<tuple>`) AND re-home across replays.
///
/// [`CheckoutSplit`] completed #212 only at the **terminal** (`gather_checkouts`
/// reassembles a `Checkouts` tuple). A seam nested MID-graph (the source of a
/// downstream [`and_then`](DeviceOpExt::and_then)) runs via `execute`, whose old
/// path collapsed a bundle source to `home == None` (no re-home) and exposed a
/// single `Pipe<S::Output>` (`= Pipe<tuple>`, so the written α/−α couldn't be
/// routed to separate downstream kernels). `SeamScatter` closes that: it is
/// implemented on the SAME `Checkout<O>` + tuple structure `CheckoutSplit` uses
/// (so it is arity- and nesting-general), and mirrors what a `bundle!`'s `execute`
/// does for kernel branches — scatter each branch into its own element pipe with
/// its own home. A single-output source keeps `Handle = Pipe<O>` (byte-identical
/// to the pre-#212 default), so only the multi-output mid-graph case changes.
pub trait SeamScatter: CheckoutSplit {
    /// The pipe-shaped downstream handle: `Pipe<O>` for a single-output source, a
    /// tuple of these for a bundle / multi-output source (mirrors
    /// [`Handle`](DeviceOp::Handle)). Owned by the seam; `execute` fills it,
    /// downstream reads it. `Send` because the seam struct owns one and
    /// [`DeviceOp`] is `Send`.
    type Handle: Clone + Send;
    /// A fresh, empty handle (each element an empty [`Pipe`]) — built once at
    /// construction, refilled every `execute`.
    fn empty_handle() -> Self::Handle;
    /// The seam's [`output_pipe`](DeviceOp::output_pipe) view — an optional single
    /// `Pipe<Value>`. For a **single-output** source this is `Some` of the storage
    /// pipe (so [`AndThen`]'s orphaned-source-deps threading behaves EXACTLY as it
    /// did pre-#212 when a downstream closure discards the seam's output —
    /// byte-identity of the single-output mid-graph path). For a **multi-output**
    /// source, storage is the per-branch element pipes and there is no single
    /// storage pipe, so this returns `None` (the same convention a `bundle!` /
    /// [`CopyTo2`] `output_pipe` use).
    fn output_pipe_view(handle: &Self::Handle) -> Option<Pipe<Self::Value>>;
    /// Scatter the seam-mutated `value` + per-branch `homes` into `handle`'s
    /// element pipes, cloning `deps` (the seam's unmap + `proceed` gate) onto each
    /// so whichever branch flows downstream carries the wait-list.
    fn scatter(handle: &Self::Handle, value: Self::Value, homes: Self::Homes, deps: &Deps);
    /// Drain any element pipes a downstream closure discarded, rehoming each
    /// undelivered branch to its origin cell (the mid-graph half of the home
    /// invariant — see [`reclaim_undelivered`](DeviceOp::reclaim_undelivered)).
    fn reclaim(handle: &Self::Handle);
    /// Reconstruct the assembled value + joined deps by draining every element
    /// pipe — the by-value (`collect` / async `run`) path for a multi-output seam.
    fn reconstruct(handle: &Self::Handle) -> Result<(Self::Value, Deps)>;
    /// Like [`reconstruct`](Self::reconstruct) but also yield the collapsed
    /// return home — the `collect_home` path. A single-output source preserves its
    /// one home (the #211 nested-in-`and_then` re-arm); a multi-output source
    /// returns `None` (N per-branch homes can't ride one slot — the same boundary
    /// [`collect_home`](DeviceOp::collect_home) documents; the multi-home re-arm
    /// rides the Checkout path instead).
    #[allow(clippy::type_complexity)]
    fn reconstruct_home(
        handle: &Self::Handle,
    ) -> Result<(Self::Value, Deps, Option<BoxedHome<Self::Value>>)>;
}

impl<O: Send + 'static> SeamScatter for Checkout<O> {
    type Handle = Pipe<O>;
    fn empty_handle() -> Pipe<O> {
        Pipe::new()
    }
    fn output_pipe_view(handle: &Pipe<O>) -> Option<Pipe<O>> {
        // Single-output: the handle IS the storage pipe.
        Some(handle.clone())
    }
    fn scatter(handle: &Pipe<O>, value: O, homes: Option<BoxedHome<O>>, deps: &Deps) {
        handle.put_home(value, deps.clone(), homes);
    }
    fn reclaim(handle: &Pipe<O>) {
        if let Some((value, _deps, home)) = handle.take_home() {
            rehome_consumed(value, home);
        }
    }
    fn reconstruct(handle: &Pipe<O>) -> Result<(O, Deps)> {
        handle
            .take()
            .ok_or(Error::NotSupported("eager graph: seam produced no output"))
    }
    fn reconstruct_home(handle: &Pipe<O>) -> Result<(O, Deps, Option<BoxedHome<O>>)> {
        handle
            .take_home()
            .ok_or(Error::NotSupported("eager graph: seam produced no output"))
    }
}

// Recursive tuple family — a bundle / multi-output branch's `Checkouts` is a
// tuple, split/scattered structurally at any nesting depth. Arity 2..=16 mirrors
// the `CheckoutSplit` / `FromCheckout` families.
macro_rules! impl_seam_scatter_tuple {
    ( $( $ck:ident : $vn:ident : $hn:ident : $idx:tt ),+ ) => {
        impl<$($ck: SeamScatter,)+> SeamScatter for ( $($ck,)+ ) {
            type Handle = ( $(<$ck as SeamScatter>::Handle,)+ );
            fn empty_handle() -> Self::Handle {
                ( $(<$ck as SeamScatter>::empty_handle(),)+ )
            }
            fn output_pipe_view(_handle: &Self::Handle) -> Option<Pipe<Self::Value>> {
                // Multi-output: storage is the element pipes; there is no single
                // storage pipe (same convention as bundle/CopyTo2 `output_pipe`).
                None
            }
            fn scatter(handle: &Self::Handle, value: Self::Value, homes: Self::Homes, deps: &Deps) {
                $( <$ck as SeamScatter>::scatter(&handle.$idx, value.$idx, homes.$idx, deps); )+
            }
            fn reclaim(handle: &Self::Handle) {
                $( <$ck as SeamScatter>::reclaim(&handle.$idx); )+
            }
            fn reconstruct(handle: &Self::Handle) -> Result<(Self::Value, Deps)> {
                let mut deps = Deps::new();
                let value = ( $({
                    let (v, d) = <$ck as SeamScatter>::reconstruct(&handle.$idx)?;
                    deps.extend(d);
                    v
                },)+ );
                Ok((value, deps))
            }
            fn reconstruct_home(
                handle: &Self::Handle,
            ) -> Result<(Self::Value, Deps, Option<BoxedHome<Self::Value>>)> {
                // A tuple value has N per-branch homes that can't ride one
                // collapsed slot — return `None` (the multi-home re-arm rides the
                // Checkout path). Reconstruct value + deps via `reconstruct`.
                let (value, deps) = Self::reconstruct(handle)?;
                Ok((value, deps, None))
            }
        }
    };
}
impl_seam_scatter_tuple!(CA: va: ha: 0, CB: vb: hb: 1);
impl_seam_scatter_tuple!(CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2);
impl_seam_scatter_tuple!(CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11,
    CM: vm: hm: 12
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11,
    CM: vm: hm: 12, CN: vn: hn: 13
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11,
    CM: vm: hm: 12, CN: vn: hn: 13, CO: vo: ho: 14
);
impl_seam_scatter_tuple!(
    CA: va: ha: 0, CB: vb: hb: 1, CC: vc: hc: 2, CD: vd: hd: 3, CE: ve: he: 4, CF: vf: hf: 5,
    CG: vg: hg: 6, CH: vh: hh: 7, CI: vi: hi: 8, CJ: vj: hj: 9, CK: vk: hk: 10, CL: vl: hl: 11,
    CM: vm: hm: 12, CN: vn: hn: 13, CO: vo: ho: 14, CP: vp: hp: 15
);

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
///
/// Both pipes are `Option` (the [`output_pipe`](DeviceOp::output_pipe) shape): a
/// **multi-output** source or `next` has NO single storage pipe (`None`), so its
/// storage is per-branch element pipes and there is nothing to thread through
/// here — the multi-output source's completion events ride its own per-branch
/// gather (`collect`/`gather_checkouts`), and a multi-output `next`'s deps are
/// threaded by its own override. A `None` on either side is therefore a no-op —
/// byte-identical to the pre-Option behaviour, where a multi-output op returned a
/// fresh never-filled `Pipe::new()` that `take()` always drained as empty.
fn thread_orphaned_source_deps<A, B>(src_pipe: &Option<Pipe<A>>, out_pipe: &Option<Pipe<B>>) {
    // Multi-output source (`None`) → no single pipe to thread; or the source pipe
    // was consumed by `next` (the normal case) → nothing to thread.
    let Some((_discarded, src_deps)) = src_pipe.as_ref().and_then(|p| p.take()) else {
        return;
    };
    // Merge the stranded source events into the out pipe's deps. If `out_pipe` is
    // absent/empty (a multi-output `next` whose storage is its element pipes, not
    // `output_pipe`), `execute` isn't the gather path — `collect` handles
    // orphaned deps for that case directly, so this is a no-op here.
    if let Some(out_pipe) = out_pipe.as_ref()
        && let Some((v, mut deps)) = out_pipe.take()
    {
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
    // The chain's terminal checkout shape is the tail op's: a multi-output tail
    // (bundle*, arc_split, CopyTo pair) yields its tuple/array of `Checkout`s.
    type Checkouts = U::Checkouts;

    fn output_pipe(&self) -> Option<Pipe<U::Output>> {
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
        if let Some((_discarded, src_deps)) = src_pipe.and_then(|p| p.take()) {
            deps.extend(src_deps);
        }
        Ok((value, deps))
    }

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(U::Output, Deps, Option<BoxedHome<U::Output>>)>
    where
        Self: Sized,
        U::Output: Send + 'static,
    {
        // Mirror `collect`, but preserve the tail's return home so an
        // `and_then`-terminated chain nested as a bundle branch (e.g.
        // `upload(buf).and_then(|b| fill(b, v))`) re-arms its caller buffer. The
        // home is the tail's — the source pipelines and its orphaned deps thread
        // in exactly as in `collect`.
        let src_pipe = self.source.output_pipe();
        self.source.execute(ec, ExecMode::Pipelined)?;
        let (value, mut deps, home) = self.next.collect_home(ec, mode)?;
        if let Some((_discarded, src_deps)) = src_pipe.and_then(|p| p.take()) {
            deps.extend(src_deps);
        }
        Ok((value, deps, home))
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
        if let Some((_discarded, src_deps)) = src_pipe.and_then(|p| p.take()) {
            deps.extend(src_deps);
        }
        Ok((checkouts, deps))
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        self.next.describe(out);
    }

    fn bind_slots(&self, binder: &mut SlotBinder) {
        // Walk source then next (execution order). A move-only binder stops once its
        // single value has landed (one `bind` binds one cell); a fan-out binder
        // (clone-able value — scalar / launch / `Arc`) NEVER stops, so one `bind`
        // fills EVERY matching cell across both subtrees (the shared-slot path).
        self.source.bind_slots(binder);
        if !binder.is_fanout() && binder.is_consumed() {
            return;
        }
        self.next.bind_slots(binder);
    }

    fn check_ready(&self) -> Result<()> {
        // Recurse source then next — the SAME traversal `execute`/`describe`/
        // `bind_slots` use, so the pre-pass covers exactly the inputs `execute`
        // resolves. Fail fast on the first unsatisfiable input.
        self.source.check_ready()?;
        self.next.check_ready()
    }

    fn reclaim_undelivered(&self) {
        // Mop up undelivered intermediates across the WHOLE subtree: the source's
        // outputs that `next` discarded (its element pipes), plus any intermediates
        // nested in source or next. Draining an already-consumed pipe (the normal
        // case) is a no-op (`take_home` yields `None`). The chain's own delivered
        // output lives in `next`'s output pipe, already drained by the terminal's
        // `gather_checkouts` before this runs → its `reclaim` is also a no-op.
        self.source.reclaim_undelivered();
        self.next.reclaim_undelivered();
    }

    fn contains_host_seam(&self) -> bool {
        // `next` is the downstream op built eagerly inside the `and_then` closure
        // at construction (a real owned field, not a deferred closure), so a host
        // seam built in the closure IS visible here.
        self.source.contains_host_seam() || self.next.contains_host_seam()
    }

    fn record(&self, ctx: &mut crate::record::RecordContext) -> Result<()> {
        // Record source first (registers its outputs), then next (which resolves
        // its inputs from those edges) — the same source-before-next order as
        // `execute`. A non-recordable child (a host seam nested in this subtree)
        // errors via the `DeviceOp::record` default; the segmenter only records a
        // subtree once `contains_host_seam()` has proven it seam-free, so on the CB
        // path this recursion never actually hits the error.
        self.source.record(ctx)?;
        self.next.record(ctx)
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

    fn output_pipe(&self) -> Option<Pipe<T>> {
        Some(self.out.clone())
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

// ── Lift: an owned (non-Clone) resource as a SELF-REHOMING leaf ──────────

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
///
/// ## Self-rehoming: reusable across replays
///
/// `Lift` holds its value in a [`Cell`] (`Arc<Mutex<Option<T>>>`) — the SAME cell
/// an [`Input::Concrete`] uses — and **lends-and-returns** it: on each run the
/// value is taken out, threaded downstream with a home pointing back at this very
/// cell, and rehomed on the run's [`Checkout`] drop (the home invariant). So a
/// graph containing a `lift` **replays** across `sync`s over the SAME `cl_mem`
/// handle — a `lift`ed scalar / buffer is a re-arming bundle branch, exactly like
/// a concrete cell fed to an in-place verb. This is the "just present this owned
/// resource as a re-homing branch" primitive (no device work, unlike fill/scale).
///
/// A second `sync` while a previous [`Checkout`] is still alive (the value not yet
/// rehomed) errors "already lent — the graph is busy", like any concrete input.
/// [`Checkout::into_inner`] severs (takes the value for good); the cell then stays
/// empty and a subsequent `sync` reports busy — the concrete-cell sever semantics.
pub struct Lift<T: Send> {
    // The lifted value lives in a `Cell` (`Arc<Mutex<Option<T>>>`): lent on each
    // run and returned on `Checkout` drop, so the lift node is its OWN home and
    // the graph replays. Wrapping it as an `Input::Concrete` reuses the exact
    // lend/rehome/check_ready machinery a concrete kernel input uses.
    input: Input<T>,
    out: Pipe<T>,
}

/// Lift an owned resource into the graph (default `Pipe` handle — see [`Lift`]).
/// With [`value`], together ≈ cuda-oxide's `value` (the by-pipe half, for
/// non-`Clone` owned resources). Self-rehoming: the graph replays across `sync`s.
///
/// `T: RecordableBuffer` — a `lift` presents an owned **device** resource (a
/// buffer / scalar) as a re-homing graph edge, so it carries a recording handle
/// like every other device leaf; this lets a `lift`ed cell live inside a
/// command-buffer subtree. (A `Clone` host value you compute on downstream is
/// [`value`], not `lift`.)
pub fn lift<T: crate::record::RecordableBuffer + Send + 'static>(v: T) -> Lift<T> {
    Lift {
        input: Input::from(v),
        out: Pipe::new(),
    }
}

impl<T: crate::record::RecordableBuffer + Send + 'static> DeviceOp for Lift<T> {
    type Output = T;
    // Default `Handle = Pipe<T>` — a resource flows, it isn't read at build.

    fn output_pipe(&self) -> Option<Pipe<T>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Lend the value out of the concrete cell WITH its home (this very cell),
        // threading the host-seam start gate — exactly the concrete-input lend
        // path. The home flows downstream via `put_home`, so the value returns to
        // this cell on `Checkout` drop and the graph re-arms for the next run.
        let (v, deps, home) = self.input.resolve_home(ec)?;
        self.out.put_home(v, deps, home);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        // Pre-run atomicity: the cell is empty iff a previous run's Checkout still
        // holds the value (busy) or it was severed — the concrete-input check.
        self.input.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("lift".into());
    }

    fn record(&self, ctx: &mut crate::record::RecordContext) -> Result<()> {
        // Like `Forward`: a lifted concrete device resource is a chain head with no
        // command. Resolve its handle from the lift's own `Concrete` cell and
        // register it under the output pipe so downstream consumers find it.
        let concrete = self.input.with_concrete(|b| b.record_handle());
        let (handle, waits) = ctx.resolve_input(concrete, self.input.pipe_cell_id())?;
        ctx.register_output(self.out.cell_id(), handle, waits);
        Ok(())
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

    fn output_pipe(&self) -> Option<Pipe<T>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.input.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("forward".into());
    }

    fn record(&self, ctx: &mut crate::record::RecordContext) -> Result<()> {
        // Identity: no command. `forward` is ALWAYS pipe-fed (its constructor
        // wraps a `Pipe`), so there is no concrete buffer to read a handle from —
        // resolve the upstream producer's output and re-register it under our OWN
        // output pipe's cell, so a downstream consumer resolving from the forwarded
        // pipe finds the same buffer with the same pending sync points. Purely a
        // pipe-alias in the edge map — mirrors `execute`'s resolve+re-deposit with
        // no device work. Passing `None` for the concrete side keeps this override
        // free of the `RecordableBuffer` bound (which `execute` also doesn't need).
        let (handle, waits) = ctx.resolve_input(None, self.input.pipe_cell_id())?;
        ctx.register_output(self.out.cell_id(), handle, waits);
        Ok(())
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

    /// Forwards [`DeviceOp::check_ready`] through the erasure so a [`DeviceDynOp`]'s
    /// atomicity pre-pass covers whichever concrete arm it wraps (the arm's inputs
    /// are resolved by `collect_erased` at run, so they must be pre-checked too).
    fn check_ready_erased(&self) -> Result<()>;

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

    fn check_ready_erased(&self) -> Result<()> {
        self.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<T>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.inner.check_ready_erased()
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

    fn output_pipe(&self) -> Option<Pipe<Arc<S::Output>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("arced".into());
    }

    fn contains_host_seam(&self) -> bool {
        self.source.contains_host_seam()
    }

    fn record(&self, ctx: &mut crate::record::RecordContext) -> Result<()> {
        // `Arc` is a host-side wrap — no command. Record the source (registers
        // its output buffer under the source pipe's cell), then alias that same
        // handle under our own output pipe so a downstream reader of the
        // `Arc<buffer>` resolves to the underlying buffer + its sync points. The
        // source's own `record` errors (via the default) if it is not recordable;
        // resolving `None`/`Some(src_cell)` needs no `RecordableBuffer` bound.
        self.source.record(ctx)?;
        // `arced` wraps a SINGLE-output source (its `Output` is one buffer it
        // `Arc`s), so the source always has a single storage pipe to key its edge
        // by; a multi-output source has no such single cell to alias here.
        let src_cell = self
            .source
            .output_pipe()
            .ok_or(Error::NotSupported(
                "record: arced source has no single output pipe (multi-output arced is unsupported)",
            ))?
            .cell_id();
        let (handle, waits) = ctx.resolve_input(None, Some(src_cell))?;
        ctx.register_output(self.out.cell_id(), handle, waits);
        Ok(())
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

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        // Multi-output storage is the per-element pipes; there is no single
        // storage pipe (the default `into_output` is overridden, and `and_then`
        // uses `handle()`), so return `None`.
        None
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

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Output, Deps, Option<BoxedHome<Self::Output>>)> {
        // Multi-output (`[T; N]` fan-out of read-only Arc clones): no single
        // collapsed home. Delegate to `collect`; `home == None`.
        let (value, deps) = self.collect(ec, mode)?;
        Ok((value, deps, None))
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

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
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
        }

        #[doc = concat!("Construct an eager [`", stringify!($name),
            "`]. \u{2248} cuda-oxide's `zip!` at this fixed arity.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $ctor<$($ty: DeviceOp),+>($($field: $ty),+) -> $name<$($ty),+> {
            $name { $($field,)+ }
        }

        impl<$($ty: DeviceOp),+> DeviceOp for $name<$($ty),+>
        where
            // Each branch's output must be `Send + 'static` so its return home (a
            // `BoxedHome`, i.e. `Box<dyn Rehome + 'static>`) can ride the branch's
            // own pipe(s) in `execute`/`gather_checkouts` — the seam that re-arms a
            // bundle over caller-owned buffers. Buffer outputs are always `'static`.
            $(<$ty as DeviceOp>::Output: Send + 'static,)+
            // Each branch's terminal `Checkouts` must reconstruct from its own
            // `Output` — the branch's OWN `gather_checkouts` bound. Holds for every
            // real op: a single-output branch's `Checkout<O>` via the identity impl,
            // a multi-output branch's tuple via the recursive `FromCheckout` family.
            $(<$ty as DeviceOp>::Checkouts: FromCheckout<<$ty as DeviceOp>::Output>,)+
        {
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
            // STRUCTURE-PRESERVING per-branch Checkouts: each branch contributes
            // its OWN `Checkouts` — NOT a single `Checkout` over the whole branch
            // output. A single-output branch contributes `Checkout<buf>`; a
            // **multi-output** branch contributes ITS tuple `(Checkout, Checkout,
            // …)`; a nested bundle contributes its own nested-by-branch shape. So
            // the bundle's `Checkouts` is grouped-by-branch, per-buffer WITHIN each
            // branch (e.g. `((Checkout<a>, Checkout<b>), Checkout<c>)`), matching
            // `Output` (also nested-by-branch). Delegation, not flatten: each
            // branch buffer is individually droppable / lendable / mappable, and
            // every branch — at any output multiplicity or nesting depth — threads
            // its OWN per-buffer return homes via its OWN `gather_checkouts`, so a
            // bundle of multi-output branches RE-ARMS (the former limitation, now
            // fixed).
            type Checkouts = ( $(<$ty as DeviceOp>::Checkouts,)+ );

            fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
                // Multi-output storage is the per-branch pipes (owned by each
                // branch); there is no single storage pipe (the default
                // `into_output` is overridden, and `and_then` uses `handle()`), so
                // return `None`.
                None
            }

            fn handle(&self) -> Self::Handle {
                // Delegate to each branch's own `handle()` — preserves by-value
                // for `value`, pipe for buffers, composite for nested bundles.
                ( $(self.$field.handle(),)+ )
            }

            fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
                // Each branch runs independently (pipelined) and fills its OWN
                // handle storage — a single-output branch its output pipe, a
                // multi-output branch its element pipes. This is the MID-GRAPH
                // scatter: a downstream `and_then` reads each branch through
                // `handle()` (which delegates per-branch to the SAME pipes filled
                // here), so there is no reconstruction / round-trip. The TERMINAL
                // gather (`gather_checkouts` / `collect`) instead delegates to each
                // branch's own gather so per-buffer homes are threaded — see below.
                $( self.$field.execute(ec, ExecMode::Pipelined)?; )+
                Ok(())
            }

            fn collect(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<(Self::Output, Deps)>
            where
                Self: Sized,
            {
                // Terminal by-value gather (async `run` / `into_output`): delegate
                // to each branch's OWN `collect` so a multi-output branch runs its
                // override (scatter-then-reconstruct over its element pipes) rather
                // than a drain of a pipe it never fills. Branches pipeline; the
                // terminal waits on the joined marker.
                let mut branch_deps: Vec<Deps> = Vec::new();
                let outputs = ( $({
                    let (v, d) = self.$field.collect(ec, ExecMode::Pipelined)?;
                    branch_deps.push(d);
                    v
                },)+ );
                let joined = join_marker(ec, &branch_deps)?;
                Ok((outputs, joined))
            }

            #[allow(clippy::type_complexity)]
            fn collect_home(
                &self,
                ec: &ExecutionContext<'_>,
                mode: ExecMode,
            ) -> Result<(Self::Output, Deps, Option<BoxedHome<Self::Output>>)> {
                // A bundle's per-branch homes ride each branch's own `Checkout`
                // (built in `gather_checkouts` by delegating to the branch), NOT one
                // collapsed tuple home — so a single-slot `BoxedHome` over the whole
                // tuple would be wrong. Return `home == None` here; this path is now
                // only a fallback (the terminal gather delegates to branch
                // `gather_checkouts`, which threads every branch's real per-buffer
                // homes directly). Delegate the value to `collect`.
                let (value, deps) = self.collect(ec, mode)?;
                Ok((value, deps, None))
            }

            fn gather_checkouts(
                &self,
                ec: &ExecutionContext<'_>,
                _mode: ExecMode,
            ) -> Result<(Self::Checkouts, Deps)> {
                // TERMINAL gather — the structure-preserving delegate. Each branch
                // runs its OWN `gather_checkouts`, producing its OWN `Checkouts`
                // (a `Checkout` for a single-output branch, a tuple for a
                // multi-output branch, a nested shape for a nested bundle) with its
                // OWN per-buffer return homes. So EVERY branch buffer — at any
                // output multiplicity or nesting depth — re-arms its origin cell on
                // drop, and the bundle re-runs (the fix: no more "multi-output
                // branch collapses to home == None"). Branches pipeline; join their
                // wait-lists into one marker.
                let mut branch_deps: Vec<Deps> = Vec::new();
                let checkouts = ( $({
                    let (co, d) = self.$field.gather_checkouts(ec, ExecMode::Pipelined)?;
                    branch_deps.push(d);
                    co
                },)+ );
                let joined = join_marker(ec, &branch_deps)?;
                Ok((checkouts, joined))
            }

            fn reclaim_undelivered(&self) {
                // Mid-graph mop-up: a downstream `and_then` closure may discard some
                // branch handles (e.g. keep only one branch's output). Each branch
                // still filled its own pipe(s) at `execute`; delegate to each
                // branch's `reclaim_undelivered` so those undelivered homed buffers
                // return to their origin cells for the next run. Pipes already
                // drained by the terminal / a downstream consumer are no-ops.
                $( self.$field.reclaim_undelivered(); )+
            }

            fn describe(&self, out: &mut Vec<String>) {
                out.push(concat!(stringify!($name), "{").into());
                $(self.$field.describe(out);)+
                out.push("}".into());
            }

            fn bind_slots(&self, binder: &mut SlotBinder) {
                // Recurse into EVERY branch — a `slot!(Tag)` placed inside any
                // bundle branch must be reachable by `g.bind(Tag(v))`. Without
                // this override the bundle would inherit the no-op default and
                // silently skip its branches, leaving the slot `Unbound` until
                // `sync` errors `SlotUnbound`.
                //
                // Fan-out discipline mirrors `AndThen::bind_slots`: a move-only
                // binder stops once its single value has landed (one `bind` →
                // one cell); a fan-out binder (clone-able value — scalar / launch
                // / `Arc`) NEVER consumes, so one `bind` fills EVERY matching cell
                // across all branches. We call each branch unconditionally; a
                // consumed move-only binder no-ops at the next branch via its own
                // `value.is_none()` guard (so the early-return below is purely an
                // optimization, not a correctness requirement).
                $(
                    self.$field.bind_slots(binder);
                    if !binder.is_fanout() && binder.is_consumed() {
                        return;
                    }
                )+
            }

            fn check_ready(&self) -> Result<()> {
                // Recurse into EVERY branch — `execute` `collect`s each, so each
                // branch's inputs are resolved and must be pre-checked. Fail-fast
                // on the first unsatisfiable branch.
                $(self.$field.check_ready()?;)+
                Ok(())
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
///
/// ## Per-branch re-arm (the dynamic-`Vec` analog of `bundle!`)
///
/// Like [`bundle!`](crate::bundle) (whose per-branch [`Checkout`] tuple carries
/// each branch's return [`home`](BoxedHome) so a bundle over caller-owned buffers
/// replays), `FanOut`'s terminal [`Checkouts`](DeviceOp::Checkouts) is a
/// `Vec<U::Checkouts>` — ONE `Checkout` (or nested `Checkouts` tuple, for a
/// multi-output branch) per branch, each threading its OWN return home. So a
/// fan-out whose branches are IN-PLACE ops over caller-owned buffers (e.g.
/// `fan_out(bufs, |b| fill(b, v))`) returns every buffer to its cell on drop, and
/// the SAME `FanOut` graph **replays** with stable `cl_mem` handles — exactly like
/// `bundle!`, just at dynamic arity. This is the bundle-arity `gather_checkouts`
/// per-branch delegation ([#207] / [#212]) generalised from a fixed tuple to a
/// runtime `Vec`. Fan-out over **minted** buffers (`upload`/`alloc`) or read-only
/// inputs also replays: those branches carry `home == None` (nothing to return),
/// which is fine.
///
/// The by-value paths ([`collect`](DeviceOp::collect) / async `run`) collapse to a
/// `Vec<U::Output>` with no per-element home (a `Vec` value has one slot, `N` homes
/// can't ride it) — the same by-value boundary `bundle!` documents; the re-arm
/// rides the [`Checkout`] terminal ([`gather_checkouts`](DeviceOp::gather_checkouts)),
/// which every waiting terminal uses.
pub struct FanOut<U: DeviceOp> {
    ops: Vec<U>,
    out: Pipe<Vec<U::Output>>,
}

/// Build a fan-out: `f` is called now for each input, producing the branch ops.
pub fn fan_out<I, F, U>(inputs: Vec<I>, mut f: F) -> FanOut<U>
where
    F: FnMut(I) -> U,
    U: DeviceOp,
{
    let ops: Vec<U> = inputs.into_iter().map(&mut f).collect();
    FanOut {
        ops,
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

impl<U: DeviceOp> DeviceOp for FanOut<U>
where
    // Each branch's output must be `Send + 'static` so its return home (a
    // `BoxedHome`) can ride the branch's own `Checkout` — the seam that re-arms a
    // fan-out over caller-owned buffers. Buffer outputs are always `'static`.
    U::Output: Send + 'static,
    // Each branch's terminal `Checkouts` must reconstruct from its own `Output` —
    // the branch's OWN `gather_checkouts` bound (single-output via the identity
    // impl, multi-output via the recursive tuple family).
    U::Checkouts: FromCheckout<U::Output>,
{
    type Output = Vec<U::Output>;
    // STRUCTURE-PRESERVING per-branch Checkouts: a `Vec` of each branch's OWN
    // `Checkouts` (a `Checkout<O>` for a single-output branch; its tuple for a
    // multi-output branch), each threading its own per-buffer return home — so
    // every branch re-arms its origin cell on drop and the fan-out replays. The
    // dynamic-`Vec` analog of `bundle!`'s per-branch tuple `Checkouts`.
    type Checkouts = Vec<U::Checkouts>;

    fn output_pipe(&self) -> Option<Pipe<Vec<U::Output>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // MID-GRAPH / by-value scatter: `collect` each branch (not `execute`) so a
        // multi-output branch runs its own gather → one reconstructed value + deps.
        // Deposits the collapsed `Vec<Output>` into the single output pipe (home
        // `None` — the by-value boundary; per-branch re-arm rides `gather_checkouts`).
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

    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Output, Deps, Option<BoxedHome<Self::Output>>)> {
        // A `Vec<Output>` value has ONE home slot but `N` per-branch homes that
        // can't ride it — return `home == None` (the same by-value boundary
        // `bundle!`/`arc_split` document). The Checkout terminal re-arms per branch.
        let (value, deps) = self.collect(ec, mode)?;
        Ok((value, deps, None))
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        _mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // TERMINAL gather — the per-branch re-arm. Each branch runs its OWN
        // `gather_checkouts` (a single-output branch builds one `Checkout` with its
        // return home; a multi-output branch its tuple, every buffer homed), so
        // EVERY branch buffer re-arms its origin cell on drop and the fan-out
        // replays. Branches pipeline; join their wait-lists into one marker. The
        // dynamic-`Vec` analog of `bundle!`'s per-branch `gather_checkouts` delegation.
        let n = self.ops.len();
        let mut branch_deps: Vec<Deps> = Vec::with_capacity(n);
        let mut cos: Vec<U::Checkouts> = Vec::with_capacity(n);
        for op in &self.ops {
            let (co, d) = op.gather_checkouts(ec, ExecMode::Pipelined)?;
            cos.push(co);
            branch_deps.push(d);
        }
        let joined = join_marker(ec, &branch_deps)?;
        Ok((cos, joined))
    }

    fn reclaim_undelivered(&self) {
        // Mid-graph mop-up: a downstream consumer of the collapsed `Vec` handle may
        // discard branch outputs. Delegate to each branch's `reclaim_undelivered`
        // so any undelivered homed buffer returns to its cell for the next run
        // (already-drained pipes are no-ops).
        for op in &self.ops {
            op.reclaim_undelivered();
        }
    }

    fn check_ready(&self) -> Result<()> {
        // `execute` `collect`s every branch op — pre-check each, fail-fast.
        for op in &self.ops {
            op.check_ready()?;
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("fan_out[{}]", self.ops.len()));
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

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
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

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill".into());
    }

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

// ── Leaf: upload (host → device, alloc-once + persistent home) ──────────

/// Allocate a `DeviceSlice<T, M>` ONCE, seed it from `src`, and hand it a
/// **persistent home** so the SAME `cl_mem` is reused across `g.sync()` replays
/// (the home invariant: "homeless is never legitimate" — even an upload-minted
/// buffer carries a home). A chain-entry leaf — no upstream input.
///
/// ## Stable handle + access-mode reseed
///
/// The buffer is allocated on the FIRST run (`from_slice`, `CL_MEM_COPY_HOST_PTR`)
/// into a persistent [`Cell`] this op owns; that cell is the buffer's home, so a
/// run's `Checkout` / `PipePayload` drop returns the SAME buffer to it. On replay
/// the buffer is re-lent from the cell (not re-minted), and whether its contents
/// are refreshed is decided by the marker via
/// [`UploadReseed::RESEED_ON_REPLAY`](crate::UploadReseed):
/// - **kernel-writable** (`ReadWrite`, …): re-seed the host source into the SAME
///   buffer each run — `upload(RW) → scale → download` stays idempotent (no
///   compounding) over a stable handle.
/// - **kernel read-only** (`ReadOnly`, `Frozen`): seed once on run 1; skip the
///   host write on replays (the kernel never mutated it).
///
/// If a previous run's `Checkout` is still alive (the buffer is lent out), the
/// cell is empty AND it has already been seeded → a second `sync` is **graph-busy**
/// (same contract as a concrete-head cell).
pub struct Upload<T: Copy, M: MemMode = ReadWrite> {
    // The host source, RETAINED for the seed-once write and any reseed-on-replay.
    src: UploadSource<T>,
    // The persistent device buffer's home cell: allocated once (first run), then
    // re-lent + re-armed across replays so the `cl_mem` handle stays stable. Empty
    // while lent (busy if already seeded); `None`-on-take is the lend.
    buf: Cell<DeviceSlice<T, M>>,
    // Whether the buffer has ever been allocated/seeded. Distinguishes "first run
    // → alloc" (cell empty, not seeded) from "lent out → busy" (cell empty, seeded).
    seeded: Arc<Mutex<bool>>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an upload leaf from any `Vec<T>` / `Box<[T]>` / `Arc<[T]>`, with the
/// **default [`ReadWrite`] marker** — the overwhelming common case, so no
/// turbofish: `upload(vec![1u32, 2, 3])`. For a non-default marker use
/// [`upload_as`] with a marker witness (`upload_as(src, Frozen)`); both paths
/// allocate once via `from_slice` (`CL_MEM_COPY_HOST_PTR`), the only constructor
/// that can build an immutable `Frozen`/`ReadOnly` buffer.
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
/// `from_slice` (`CL_MEM_COPY_HOST_PTR`) on the first run.
pub fn upload_as<T, M, S>(src: S, marker: M) -> Upload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
    S: Into<UploadSource<T>>,
{
    let _ = marker; // witness only — fixes M, zero-sized, no runtime use.
    Upload {
        src: src.into(),
        buf: Arc::new(Mutex::new(None)),
        seeded: Arc::new(Mutex::new(false)),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for Upload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // The buffer is allocated ONCE and lives in `self.buf` across runs; its home
        // is that very cell, so a run's Checkout / PipePayload drop returns the SAME
        // `cl_mem` here. Three cases, decided by the cell + the `seeded` flag:
        let mut seeded = self.seeded.lock().unwrap();
        let lent = self.buf.lock().unwrap().take();
        let buf = match (lent, *seeded) {
            // First run: never seeded → alloc + seed via from_slice
            // (CL_MEM_COPY_HOST_PTR, synchronous create, no in-flight event).
            (None, false) => {
                let buf = DeviceSlice::<T, M>::from_slice(ec.context(), self.src.as_slice())?;
                *seeded = true;
                buf
            }
            // Replay: the buffer is back in the cell. Re-lend it; re-seed the host
            // source IF the marker is kernel-writable (it may have been mutated in
            // place last run) — keeping `upload(RW) → … → download` idempotent. A
            // kernel read-only marker (ReadOnly/Frozen) skips the write: its bytes
            // never changed device-side, seed-once suffices.
            (Some(mut buf), _) => {
                if M::RESEED_ON_REPLAY {
                    // Synchronous host write back into the SAME buffer (stable
                    // handle). No upstream deps — upload is a chain head.
                    crate::buffer::write_buffer_enqueue(
                        &mut buf,
                        ec,
                        self.src.as_slice(),
                        true,
                        &[],
                    )?;
                }
                buf
            }
            // Cell empty but already seeded: the buffer is lent out (a prior run's
            // Checkout is still alive) → graph-busy, the concrete-cell contract.
            (None, true) => {
                return Err(Error::NotSupported(
                    "eager graph: an upload buffer was already lent and not returned \
                     — a graph is `sync`'d while a previous `Checkout` is still alive \
                     (the graph is busy)",
                ));
            }
        };
        // The home is this op's persistent cell (identity rehome): the buffer is
        // returned here on Checkout / PipePayload drop, re-arming the upload with a
        // STABLE handle. So a downstream consume (download) rehomes it here, not the
        // releasing drop.
        let home: Option<BoxedHome<DeviceSlice<T, M>>> = Some(Box::new(Arc::clone(&self.buf)));
        self.out.put_home(buf, Deps::new(), home);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("upload".into());
    }
}

// ── Leaf: seeded device-scalar alloc (host value → DeviceScalar) ────────

/// Allocate a [`DeviceScalar<T, M>`] ONCE, seed it from `value`, and hand it a
/// **persistent home** so the SAME `cl_mem` is reused across `g.sync()` replays
/// — the scalar twin of [`Upload`]. A chain-entry leaf (no upstream input).
///
/// Same stable-handle + reseed-on-replay contract as [`Upload`]: on the first
/// run the scalar is allocated + seeded via [`DeviceScalar::new`]
/// (`CL_MEM_COPY_HOST_PTR`); on replay it is re-lent from this op's home cell,
/// and re-seeded IFF the marker is kernel-writable
/// ([`UploadReseed::RESEED_ON_REPLAY`](crate::UploadReseed)).
pub struct ScalarUpload<T: Copy, M: MemMode = ReadWrite> {
    value: T,
    buf: Cell<DeviceScalar<T, M>>,
    seeded: Arc<Mutex<bool>>,
    out: Pipe<DeviceScalar<T, M>>,
}

/// Build a seeded device-scalar alloc leaf with the **default [`ReadWrite`]
/// marker** — the scalar twin of [`upload`]: `scalar_value(0.0f32)`.
pub fn scalar_value<T>(value: T) -> ScalarUpload<T, ReadWrite>
where
    T: Copy + Send + Sync + 'static,
{
    scalar_value_as(value, ReadWrite)
}

/// Build a seeded device-scalar alloc leaf with an **explicit access marker**,
/// inferred from the `marker` witness — the scalar twin of [`upload_as`].
pub fn scalar_value_as<T, M>(value: T, marker: M) -> ScalarUpload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    let _ = marker;
    ScalarUpload {
        value,
        buf: Arc::new(Mutex::new(None)),
        seeded: Arc::new(Mutex::new(false)),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for ScalarUpload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    type Output = DeviceScalar<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceScalar<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Same three-case stable-handle logic as `Upload`, over a length-1 scalar.
        let mut seeded = self.seeded.lock().unwrap();
        let lent = self.buf.lock().unwrap().take();
        let buf = match (lent, *seeded) {
            (None, false) => {
                let buf = DeviceScalar::<T, M>::new(ec.context(), self.value)?;
                *seeded = true;
                buf
            }
            (Some(mut buf), _) => {
                if M::RESEED_ON_REPLAY {
                    crate::buffer::write_buffer_enqueue(
                        &mut buf.inner,
                        ec,
                        std::slice::from_ref(&self.value),
                        true,
                        &[],
                    )?;
                }
                buf
            }
            (None, true) => {
                return Err(Error::NotSupported(
                    "eager graph: a device-scalar upload buffer was already lent and \
                     not returned — a graph is `sync`'d while a previous `Checkout` is \
                     still alive (the graph is busy)",
                ));
            }
        };
        let home: Option<BoxedHome<DeviceScalar<T, M>>> = Some(Box::new(Arc::clone(&self.buf)));
        self.out.put_home(buf, Deps::new(), home);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("scalar_value".into());
    }
}

// ── Leaf: zero-init device-scalar alloc ────────────────────────────────

/// Allocate a [`DeviceScalar<T, M>`] zero-initialised (via a length-1
/// [`DeviceScalar::new`]`(T::default())`), with a persistent home — the scalar
/// twin of [`alloc_zero`]. A chain-entry leaf.
pub struct ScalarZero<T: Copy, M: MemMode = ReadWrite> {
    inner: ScalarUpload<T, M>,
}

/// Build a zero-init device-scalar alloc leaf with the **default [`ReadWrite`]
/// marker** — `scalar_zero::<f32>()`.
pub fn scalar_zero<T>() -> ScalarZero<T, ReadWrite>
where
    T: Copy + Default + Send + Sync + 'static,
{
    ScalarZero {
        inner: scalar_value_as(T::default(), ReadWrite),
    }
}

/// Build a zero-init device-scalar alloc leaf with an **explicit access marker**.
pub fn scalar_zero_as<T, M>(marker: M) -> ScalarZero<T, M>
where
    T: Copy + Default + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    ScalarZero {
        inner: scalar_value_as(T::default(), marker),
    }
}

impl<T, M> DeviceOp for ScalarZero<T, M>
where
    T: Copy + Default + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    type Output = DeviceScalar<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceScalar<T, M>>> {
        self.inner.output_pipe()
    }

    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        self.inner.execute(ec, mode)
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("scalar_zero".into());
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

    fn output_pipe(&self) -> Option<Pipe<Vec<T>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // "Homeless is never legitimate": download CONSUMES the device buffer into
        // a host `Vec`, but the buffer itself still has a home (a user-allocated
        // concrete cell, a slot, or an upload-minted persistent cell). Resolve WITH
        // the home so the device buffer is RETURNED to its origin — the same
        // `cl_mem` is reused on replay — rather than released. The OUTPUT pipe
        // carries the `Vec` with NO home: the Vec is the user's result, it has no
        // origin cell. (`ReadInto` is the in-place template; here the buffer's home
        // and the output value diverge, so the rehome happens here, not via the
        // output pipe.)
        let (buf, deps, home) = self.buf.resolve_home(ec)?;
        let mut host = vec![T::default(); buf.len()];
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        match mode {
            // Terminal: native blocking read (CL_BLOCKING) — the driver waits,
            // the host Vec is valid on return, no event. Matches Tier-1
            // `ReadOp::wait_on`; restores parity for `…download().sync()`.
            ExecMode::Blocking => {
                crate::buffer::read_buffer_enqueue(&buf, ec, &mut host, true, &raw)?;
                rehome_consumed(buf, home);
                self.out.put(host, Deps::new());
            }
            // Pipelined: non-blocking; the event gates the Vec being valid. The
            // read is enqueued before we rehome, but the rehome only re-arms the
            // origin CELL (deposits the buffer handle for the NEXT run); the
            // in-flight read still holds the live `cl_mem` via the OpenCL queue, so
            // returning the handle to its cell here does not race the read.
            ExecMode::Pipelined => {
                let event = crate::buffer::read_buffer_enqueue(&buf, ec, &mut host, false, &raw)?;
                rehome_consumed(buf, home);
                self.out.put(host, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<USMSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<USMSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_usm_uninit".into());
    }
}

// ════════════════════════════════════════════════════════════════════════
// usm_op.rs ports — USM alloc / wrap (pure host, synchronous)
// ════════════════════════════════════════════════════════════════════════

// ── Leaf: wrap a host Vec<T> as a USMSlice (eager UsmSliceOp) ───────────────

/// Wrap a host `Vec<T>` as a [`USMSlice<T, M>`], allocating ONCE and re-lending
/// the SAME USM allocation across `g.sync()` replays — the USM twin of
/// [`Upload`], whose reusable structure it mirrors exactly (source leaf, no
/// upstream input; construction is pure host code — `USMSlice::new` — with no
/// enqueue / event).
///
/// Same stable-handle + reseed-on-replay contract as [`Upload`]: on the first run
/// the `Vec` is moved into a `USMSlice` (USM IS that host allocation); on replay
/// the SAME slice is re-lent from this op's home cell, and re-seeded IFF the marker
/// is kernel-writable ([`UploadReseed::RESEED_ON_REPLAY`](crate::UploadReseed)) —
/// a plain host `copy_from_slice` into the same allocation (USM is host memory),
/// keeping a replayed USM chain head idempotent. A kernel read-only marker
/// (`ReadOnly`/`Frozen`) seeds once and skips the replay write.
pub struct UsmSlice<T: Copy, M: MemMode = ReadWrite> {
    // The host source, RETAINED for the seed-once move and any reseed-on-replay
    // (the reseed copies from here into the persistent USM allocation).
    src: UploadSource<T>,
    // The persistent USM slice's home cell: allocated once (first run), then
    // re-lent + re-armed across replays so the SVM pointer stays stable. Empty
    // while lent (busy if already seeded); `None`-on-take is the lend.
    buf: Cell<USMSlice<T, M>>,
    // Whether the slice has ever been allocated/seeded. Distinguishes "first run
    // → alloc" (cell empty, not seeded) from "lent out → busy" (cell empty, seeded).
    seeded: Arc<Mutex<bool>>,
    out: Pipe<USMSlice<T, M>>,
}

/// Build an eager USM-wrap leaf from any `Vec<T>` / `Box<[T]>` / `Arc<[T]>` with
/// the **default [`ReadWrite`] marker** — no turbofish: `usm_slice(data)`. For a
/// non-default marker use [`usm_slice_as`] with a marker witness. Reusable across
/// `sync`s (stable SVM pointer, reseed-on-replay) — the USM twin of [`upload`].
pub fn usm_slice<T, S>(data: S) -> UsmSlice<T, ReadWrite>
where
    T: Copy + Send + Sync + 'static,
    S: Into<UploadSource<T>>,
{
    usm_slice_as(data, ReadWrite)
}

/// Build an eager USM-wrap leaf with an **explicit access marker**, inferred
/// from the `marker` witness — no turbofish: `usm_slice_as(data, HostReadOnly)`.
/// The default-marker shorthand is [`usm_slice`].
pub fn usm_slice_as<T, M, S>(data: S, marker: M) -> UsmSlice<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
    S: Into<UploadSource<T>>,
{
    let _ = marker; // witness only — fixes M, zero-sized, no runtime use.
    UsmSlice {
        src: data.into(),
        buf: Arc::new(Mutex::new(None)),
        seeded: Arc::new(Mutex::new(false)),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for UsmSlice<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    type Output = USMSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<USMSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // The USM slice is allocated ONCE and lives in `self.buf` across runs; its
        // home is that very cell, so a run's Checkout / PipePayload drop returns the
        // SAME SVM allocation here. Three cases, decided by the cell + `seeded` flag
        // — the exact shape `Upload::execute` uses (USMSlice::new is the synchronous
        // host-create analog of DeviceSlice::from_slice; reseed is a host copy).
        let mut seeded = self.seeded.lock().unwrap();
        let lent = self.buf.lock().unwrap().take();
        let buf = match (lent, *seeded) {
            // First run: never seeded → move the host source into a fresh USMSlice
            // (pure host code — USM IS the host allocation, no enqueue/event).
            (None, false) => {
                let buf = USMSlice::<T, M>::new(ec.context(), self.src.as_slice().to_vec())?;
                *seeded = true;
                buf
            }
            // Replay: the slice is back in the cell. Re-lend it; re-seed the host
            // source IF the marker is kernel-writable (it may have been mutated in
            // place last run) — keeping `usm_slice(RW) → … → download` idempotent
            // over a stable SVM pointer. A kernel read-only marker (ReadOnly/Frozen)
            // skips the write: its bytes never changed device-side, seed-once suffices.
            (Some(mut buf), _) => {
                if M::RESEED_ON_REPLAY {
                    // Plain host copy back into the SAME allocation (stable pointer),
                    // after draining in-flight kernel-use events. No SVM map/memcpy —
                    // USM is host memory.
                    buf.reseed_sync(self.src.as_slice())?;
                }
                buf
            }
            // Cell empty but already seeded: the slice is lent out (a prior run's
            // Checkout is still alive) → graph-busy, the concrete-cell contract.
            (None, true) => {
                return Err(Error::NotSupported(
                    "eager graph: a `usm_slice` buffer was already lent and not \
                     returned — a graph is `sync`'d while a previous `Checkout` is \
                     still alive (the graph is busy)",
                ));
            }
        };
        // The home is this op's persistent cell (identity rehome): the slice is
        // returned here on Checkout / PipePayload drop, re-arming the leaf with a
        // STABLE SVM pointer. So a downstream consume rehomes it here, not the
        // releasing drop.
        let home: Option<BoxedHome<USMSlice<T, M>>> = Some(Box::new(Arc::clone(&self.buf)));
        self.out.put_home(buf, Deps::new(), home);
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

    fn output_pipe(&self) -> Option<Pipe<USMSliceUninit<T, M>>> {
        Some(self.out.clone())
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

    fn output_pipe(&self) -> Option<Pipe<DeviceSliceUninit<T, M>>> {
        Some(self.out.clone())
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

    fn output_pipe(&self) -> Option<Pipe<MappedSliceUninit<T, M>>> {
        Some(self.out.clone())
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

    fn output_pipe(&self) -> Option<Pipe<I>> {
        Some(self.out.clone())
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

    fn output_pipe(&self) -> Option<Pipe<Vec<I::Pixel>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.img.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.view.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
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

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.view.check_ready()
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
    /// The home that returns an `Out` into a concrete `Cell<Self>` on `Checkout`
    /// drop (or `PipePayload` drop).
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<Out>>;

    /// The home that returns an `Out` into a **slot** `SlotCell<Self>` — the
    /// four-state analogue used when a copy operand is a `slot!(Tag)` directly
    /// (scenario 6). Re-arms `Lent → Bound` on rehome and severs `Lent → Severed`
    /// on `into_inner`. Default `None`: a slot's value type is always an `Init`
    /// buffer (`Tag::Value`), so only the identity `CopyHome` impls (`Self == Out`)
    /// override this; the `Uninit → Init` downgrade impls keep the default (an
    /// uninit buffer is never a slot value, so the path is unreachable).
    fn copy_slot_home(cell: SlotCell<Self>) -> Option<BoxedHome<Out>> {
        let _ = cell;
        None
    }

    /// Forward a home threaded on a **lent pipe** input (a cross-graph `Checkout`
    /// fed as a copy operand — see `Input::lent`) as the copy output's return
    /// home. A lent-pipe operand carries the ORIGIN graph's home on its payload;
    /// the copy must pass it through so the borrowed buffer RETURNS to the origin
    /// on the copy `Checkout`'s drop (LEND semantics, matching the kernel-arg
    /// path). Only reachable for the identity impls (`Self == Out`) — a lent
    /// `Checkout` is always an `Init` buffer, so the copy never retypes it — hence
    /// the default `None` (the `Uninit → Init` downgrade impls never see one).
    fn pipe_home(home: BoxedHome<Self>) -> Option<BoxedHome<Out>> {
        let _ = home;
        None
    }
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
// `copy_slot_home` returns the four-state `SlotHome` so a `slot!()` copy operand
// re-arms its slot (`Lent → Bound`) on rehome / severs (`Lent → Severed`) on
// `into_inner` — exactly like a slot in a kernel-arg position.
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<DeviceSlice<T, M>>
    for DeviceSlice<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<DeviceSlice<T, M>>> {
        Some(Box::new(cell))
    }
    fn copy_slot_home(cell: SlotCell<Self>) -> Option<BoxedHome<DeviceSlice<T, M>>> {
        Some(Box::new(SlotHome { cell }))
    }
    fn pipe_home(home: BoxedHome<Self>) -> Option<BoxedHome<DeviceSlice<T, M>>> {
        Some(home)
    }
}
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<MappedSlice<T, M>>
    for MappedSlice<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<MappedSlice<T, M>>> {
        Some(Box::new(cell))
    }
    fn copy_slot_home(cell: SlotCell<Self>) -> Option<BoxedHome<MappedSlice<T, M>>> {
        Some(Box::new(SlotHome { cell }))
    }
    fn pipe_home(home: BoxedHome<Self>) -> Option<BoxedHome<MappedSlice<T, M>>> {
        Some(home)
    }
}
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<USMSlice<T, M>> for USMSlice<T, M> {
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<USMSlice<T, M>>> {
        Some(Box::new(cell))
    }
    fn copy_slot_home(cell: SlotCell<Self>) -> Option<BoxedHome<USMSlice<T, M>>> {
        Some(Box::new(SlotHome { cell }))
    }
    fn pipe_home(home: BoxedHome<Self>) -> Option<BoxedHome<USMSlice<T, M>>> {
        Some(home)
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

/// A value usable as a [`eager_copy_to`] **operand**: a concrete buffer, an
/// upstream [`Pipe`], a [`Checkout`], or a [`slot!`](crate::slot)`(Tag)` hole. It
/// resolves to an `Input<Buf>` over the concrete buffer family `Buf` the copy
/// then drives (via `Buf: CopyTo<…>`).
///
/// ## Why a dedicated trait (not `Into<Input<Buf>>`)
///
/// `SlotHandle<Tg>` cannot impl `Into<Input<Tg::Value>>` — the blanket
/// `From<T> for Input<T>` blocks it under coherence (the compiler can't rule out
/// `Tg::Value == SlotHandle<Tg>`). `CopyOperand` is a distinct nominal trait with
/// no such clash, so a slot plugs straight into a copy operand position
/// (`eager_copy_to(slot!(Src), slot!(Dst))`) exactly as it already does in a
/// kernel-arg position via [`ToInput`]. Concrete buffers / pipes / checkouts route
/// through their existing `Into<Input<_>>` conversions.
pub trait CopyOperand<Buf> {
    /// Resolve into the copy's input edge over the concrete buffer type `Buf`.
    fn into_copy_input(self) -> Input<Buf>;
}

// A slot plugs into a copy operand position, mirroring its kernel-arg `ToInput`.
impl<Tg: Tag> CopyOperand<Tg::Value> for SlotHandle<Tg> {
    fn into_copy_input(self) -> Input<Tg::Value> {
        self.into_input()
    }
}

// A `Pipe<Buf>` (upstream producer's output edge) → a deferred input. Per-type
// (not a blanket over `Into<Input<_>>`) so it stays disjoint from the `SlotHandle`
// impl — a blanket would collide because the compiler can't rule out
// `Tg::Value == SlotHandle<Tg>`.
impl<Buf> CopyOperand<Buf> for Pipe<Buf> {
    fn into_copy_input(self) -> Input<Buf> {
        Input::Pipe(self)
    }
}

/// Implement [`CopyOperand`] for a concrete buffer family + its `Checkout`
/// wrapper (each a distinct nominal type, disjoint from the slot/pipe impls).
macro_rules! impl_copy_operand_concrete {
    ($buf:ident) => {
        impl<E, M> CopyOperand<$crate::$buf<E, M>> for $crate::$buf<E, M>
        where
            M: $crate::MemMode,
        {
            fn into_copy_input(self) -> Input<$crate::$buf<E, M>> {
                Input::from(self)
            }
        }
        impl<E, M> CopyOperand<$crate::$buf<E, M>> for Checkout<$crate::$buf<E, M>>
        where
            M: $crate::MemMode,
            E: Send,
        {
            // LEND: relocate the value + its home onto a pre-loaded pipe so the
            // home rides into the copy's graph and returns to A on drop — A stays
            // BUSY while the borrow is held, then re-arms for a plain `sync()` (no
            // `mutate_bind`). Identical semantics to the `ToInput`/`From` Checkout
            // arg paths; `.into_inner()` remains the explicit sever verb.
            fn into_copy_input(self) -> Input<$crate::$buf<E, M>> {
                let (value, home) = self.into_value_and_home();
                Input::lent(value, home)
            }
        }
    };
}
impl_copy_operand_concrete!(DeviceSlice);
impl_copy_operand_concrete!(MappedSlice);
impl_copy_operand_concrete!(USMSlice);

// The Uninit dst families are valid copy *destinations* (never a slot value), so
// they need a concrete + checkout operand impl too.
macro_rules! impl_copy_operand_uninit {
    ($buf:ident) => {
        impl<E, M> CopyOperand<$crate::$buf<E, M>> for $crate::$buf<E, M>
        where
            M: $crate::MemMode,
        {
            fn into_copy_input(self) -> Input<$crate::$buf<E, M>> {
                Input::from(self)
            }
        }
    };
}
impl_copy_operand_uninit!(DeviceSliceUninit);
impl_copy_operand_uninit!(MappedSliceUninit);
impl_copy_operand_uninit!(USMSliceUninit);

/// Build an eager copy leaf. `src` / `dst` may each be a concrete buffer, an
/// upstream [`Pipe`], a [`Checkout`], or a [`slot!`](crate::slot)`(Tag)` hole (see
/// [`CopyOperand`]). Output is `(src, dst)` (an `Uninit` dst comes back `Init` —
/// the copy wrote every byte). See [`CopyTo2`].
pub fn eager_copy_to<Src, Dst, S, D>(src: S, dst: D) -> CopyTo2<Src, Dst>
where
    Src: CopyTo<Dst>,
    <Src::Op as crate::eager::DeviceEnqueue>::Output: CopyOutputs,
    S: CopyOperand<Src>,
    D: CopyOperand<Dst>,
{
    CopyTo2 {
        src: src.into_copy_input(),
        dst: dst.into_copy_input(),
        src_pipe: Pipe::new(),
        dst_pipe: Pipe::new(),
    }
}

impl<Src, Dst> DeviceOp for CopyTo2<Src, Dst>
where
    // `RecordableBuffer` on both operands lets the folded `record` override resolve
    // each concrete buffer's handle (a copy operand is always a device buffer that
    // satisfies it — `DeviceSlice`/`MappedSlice`/`USMSlice` + their `Uninit` dst
    // forms), so the CB path records a copy leaf; no observable narrowing.
    Src: CopyTo<Dst> + Send + crate::record::RecordableBuffer + 'static,
    Dst: Send + crate::record::RecordableBuffer + 'static,
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

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        // Multi-output storage is the per-element pipes; there is no single
        // storage pipe (the default `into_output` is overridden, and `and_then`
        // uses `handle()`), so return `None`.
        None
    }

    fn handle(&self) -> Self::Handle {
        (self.src_pipe.clone(), self.dst_pipe.clone())
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Resolve each input → (buffer, upstream Deps, output-typed return home),
        // threading the home onto the output element pipe (re-arming `g` on Checkout
        // / PipePayload drop). `resolve_copy` unifies all three arms under the home
        // invariant: a CONCRETE cell routes through `CopyHome::copy_home` (identity,
        // or the `Uninit → Init` downgrade re-wrap); a SLOT routes through
        // `CopyHome::copy_slot_home` (a four-state `SlotHome` — re-arms `Lent →
        // Bound`, severs on `into_inner`), closing the former copy-slot gap; a
        // LENT pipe (a cross-graph `Checkout` fed as a copy operand) forwards the
        // ORIGIN's home via `CopyHome::pipe_home` so the borrow RETURNS to it on
        // drop (LEND, matching the kernel-arg path); a minted-upstream pipe is
        // `None`. Either input may be a pipe or concrete — combine their wait-lists.
        let (src, src_deps, src_home) = self.src.resolve_copy(ec)?;
        let (dst, dst_deps, dst_home) = self.dst.resolve_copy(ec)?;
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

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Output, Deps, Option<BoxedHome<Self::Output>>)>
    where
        Self: Sized,
        Self::Output: Send + 'static,
    {
        // Multi-output (`(src, dst)`): each side's home rides its own `Checkout`
        // via `gather_checkouts`, not one collapsed tuple home. Nested as a bundle
        // branch it collapses to `home == None`. Delegate to `collect`.
        let (value, deps) = self.collect(ec, mode)?;
        Ok((value, deps, None))
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // Drain each element pipe with its own home → a tuple of independent
        // Checkouts. Each output carries the home `execute` threaded (concrete cell,
        // slot, or `None` for a pipe), so the two sides re-arm independently:
        // dropping one side's Checkout rehomes it while `into_inner` on the other
        // severs only that side (scenario 11).
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

    fn bind_slots(&self, binder: &mut SlotBinder) {
        // A copy's src/dst may each be a `slot!()` operand; offer the binder to
        // both (execution order: src then dst), short-circuiting once it lands.
        // Non-slot (concrete / pipe) inputs are a no-op in `try_bind_slot`.
        self.src.try_bind_slot(binder);
        if binder.is_consumed() {
            return;
        }
        self.dst.try_bind_slot(binder);
    }

    fn check_ready(&self) -> Result<()> {
        // Both operands are resolved in `execute` (src then dst) — check both,
        // read-only, fail-fast on the first unsatisfiable one.
        self.src.check_ready()?;
        self.dst.check_ready()
    }

    fn reclaim_undelivered(&self) {
        // Two element pipes (src, dst). Drain + rehome each undelivered side so a
        // copy whose output is partly discarded (e.g. `…and_then(|(src, _dst)| …)`)
        // returns the dropped side's buffer to its origin cell. Already-drained
        // pipes (delivered to a terminal Checkout / consumed downstream) are no-ops.
        if let Some((v, _d, home)) = self.src_pipe.take_home() {
            rehome_consumed(v, home);
        }
        if let Some((v, _d, home)) = self.dst_pipe.take_home() {
            rehome_consumed(v, home);
        }
    }

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
        Dst: CopyOperand<Dst>,
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

    fn output_pipe(&self) -> Option<Pipe<S::Output>> {
        Some(self.out.clone())
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

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
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
    S::Checkouts: SeamScatter<Value = S::Output>,
{
    source: S,
    f: Arc<F>,
    // The per-branch, pipe-shaped downstream handle — `Pipe<O>` for a
    // single-output source (the pre-#212 default), a tuple of pipes for a bundle /
    // multi-output source. `execute` scatters the seam-mutated value+homes into
    // these, so downstream can route each written branch to its own kernel AND
    // every branch re-homes across replays. Owned (not `Pipe::new()` per run) so
    // `handle()` hands out stable pipe identities.
    handle: <S::Checkouts as SeamScatter>::Handle,
}

/// Like [`AndThenHost`] but the closure also receives `&Context` — built by
/// [`and_then_host_with_context`](DeviceOpExt::and_then_host_with_context).
pub struct AndThenHostWithContext<S: DeviceOp, F>
where
    S::Output: crate::mappable::Mappable,
    S::Checkouts: SeamScatter<Value = S::Output>,
{
    source: S,
    f: Arc<F>,
    // See `AndThenHost::handle`.
    handle: <S::Checkouts as SeamScatter>::Handle,
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
    // The source's terminal `Checkouts` must SPLIT into the assembled tuple value
    // (`S::Output`, what the seam maps) plus its per-branch homes, and REASSEMBLE
    // from a (seam-mutated) value + those homes. For a single-output source this
    // is `Checkout<S::Output>` (identity split — the #211 path stays byte-for-byte
    // the same); for a bundle / multi-output source it is that source's per-branch
    // `Checkouts` tuple, split/reassembled recursively (any arity, any nesting).
    S::Checkouts: CheckoutSplit<Value = S::Output>,
    // MID-GRAPH re-scatter (#212 completion): the same per-branch structure, but
    // exposed as element PIPES downstream + re-homed via `execute`. Single-output
    // → `Handle = Pipe<O>` (pre-#212 default, byte-identical); multi-output → a
    // tuple of pipes, so written branches route to separate downstream kernels AND
    // re-home across replays.
    S::Checkouts: SeamScatter<Value = S::Output>,
    // The source's own `gather_checkouts` needs this bound too — a bundle/multi-
    // output source's `Checkouts` satisfies it via the recursive `FromCheckout`
    // family, a single-output source via the identity impl.
    S::Checkouts: FromCheckout<S::Output>,
    F: for<'a> Fn(<S::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
        + Send
        + Sync
        + 'static,
{
    type Output = S::Output;
    // The downstream-facing handle is the source's per-branch PIPE shape (via
    // `SeamScatter::Handle`): `Pipe<O>` for a single-output source (unchanged), a
    // tuple of pipes for a bundle/multi-output source — so `and_then(|(a, b)| …)`
    // can route each written branch to its own kernel.
    type Handle = <S::Checkouts as SeamScatter>::Handle;
    // The seam's terminal result IS the source's — one `Checkout` for a
    // single-output source, a per-branch tuple for a bundle/multi-output source.
    // The seam re-threads EACH branch's home (via `CheckoutSplit`) so every branch
    // re-arms its origin cell across `sync`s — the multi-home replay a single
    // collapsed `collect_home` slot cannot carry.
    type Checkouts = S::Checkouts;

    fn output_pipe(&self) -> Option<Pipe<S::Output>> {
        // Single-output: `Some` of the storage pipe; multi-output: `None`
        // (storage is the element pipes). See `SeamScatter::output_pipe_view`.
        <S::Checkouts as SeamScatter>::output_pipe_view(&self.handle)
    }

    fn handle(&self) -> Self::Handle {
        self.handle.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // MID-GRAPH path (the seam nested as an `and_then` source, or async `run`).
        // Gather the source PER-BRANCH via its own `gather_checkouts` (a
        // single-output source builds one `Checkout`; a bundle/multi-output source
        // delegates to each branch, threading every branch's per-buffer return
        // home) — the SAME per-branch gather the terminal uses. Then SPLIT into the
        // assembled value (mapped + handed to the closure) + per-branch homes,
        // run the seam, and RE-SCATTER each written-back branch (value+home) into
        // its OWN element pipe. So downstream reads each branch as its own pipe AND
        // every branch re-homes on drop — the mid-graph multi-home replay (#212
        // completion). A single-output source scatters into one pipe with its home
        // preserved (byte-identical to the pre-#212 `collect_home` + `put_home`).
        let (src_cos, deps) = self.source.gather_checkouts(ec, ExecMode::Pipelined)?;
        let (value, homes) = src_cos.split();
        // Reusable: `Arc::clone` the closure so the per-run worker thread gets
        // its OWN owned handle to move in (it runs off the submitting thread).
        // `run_host_seam` keeps its `FnOnce` param — the clone is a fresh
        // one-shot callable per replay; the closure itself (`Fn`) re-runs.
        let f = Arc::clone(&self.f);
        let (out_value, out_deps) =
            run_host_seam::<S::Output, _>(value, deps, ec, move |view| (*f)(view))?;
        <S::Checkouts as SeamScatter>::scatter(&self.handle, out_value, homes, &out_deps);
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(S::Output, Deps)>
    where
        Self: Sized,
    {
        // BY-VALUE gather (async `run` / `into_output`): scatter via `execute`,
        // then reconstruct the assembled value by draining the element pipe(s).
        // Single-output drains one pipe (unchanged); multi-output reconstructs the
        // tuple, joining deps.
        self.execute(ec, mode)?;
        <S::Checkouts as SeamScatter>::reconstruct(&self.handle)
    }

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(S::Output, Deps, Option<BoxedHome<S::Output>>)>
    where
        Self: Sized,
        S::Output: Send + 'static,
    {
        // Home-preserving by-value gather: single-output preserves its one home
        // (the #211 nested-in-`and_then` re-arm); multi-output returns `home ==
        // None` (per-branch homes ride the Checkout / element-pipe path).
        self.execute(ec, mode)?;
        <S::Checkouts as SeamScatter>::reconstruct_home(&self.handle)
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        _mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // TERMINAL / CHECKOUT gather — the #212 pass-1 path, UNCHANGED (byte-for-
        // byte). Gather the source per-branch, split into value + homes, run the
        // seam, REASSEMBLE the checkouts re-threading each ORIGINAL home so every
        // branch re-arms on drop. (Distinct from `execute`, which re-scatters into
        // the seam's own element pipes for a DOWNSTREAM consumer; here the terminal
        // takes the checkouts directly.)
        let (src_cos, deps) = self.source.gather_checkouts(ec, ExecMode::Pipelined)?;
        let (value, homes) = src_cos.split();
        let f = Arc::clone(&self.f);
        let (out_value, out_deps) =
            run_host_seam::<S::Output, _>(value, deps, ec, move |view| (*f)(view))?;
        let checkouts = <S::Checkouts as CheckoutSplit>::reassemble(out_value, homes);
        Ok((checkouts, out_deps))
    }

    fn reclaim_undelivered(&self) {
        // Mid-graph mop-up: a downstream `and_then` closure may discard some of the
        // seam's element pipes (e.g. keep only the written α, drop −α). Each was
        // filled by `execute`; drain + rehome any the consumer left, so those
        // branches re-arm their origin cells for the next run. Already-drained pipes
        // are no-ops. The source's own reclaim runs too (its outputs were consumed
        // into the seam's checkouts at `execute`, so this is a no-op there, but keep
        // the traversal complete).
        <S::Checkouts as SeamScatter>::reclaim(&self.handle);
        self.source.reclaim_undelivered();
    }

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
    }

    fn bind_slots(&self, binder: &mut SlotBinder) {
        // A host seam is a structural pass-through: recurse into the source so
        // `bind`/`call` reach any `slot!` cells the source op carries.
        self.source.bind_slots(binder);
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
    // Same `CheckoutSplit` + `SeamScatter` bounds as `AndThenHost` — see its impl
    // for the rationale (terminal split/reassemble + mid-graph re-scatter).
    // Single-output source keeps the #211 / pre-#212 paths identical.
    S::Checkouts: CheckoutSplit<Value = S::Output>,
    S::Checkouts: SeamScatter<Value = S::Output>,
    // The source's own `gather_checkouts` needs this bound too — a bundle/multi-
    // output source's `Checkouts` satisfies it via the recursive `FromCheckout`
    // family, a single-output source via the identity impl.
    S::Checkouts: FromCheckout<S::Output>,
    F: for<'a> Fn(&Context, <S::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
        + Send
        + Sync
        + 'static,
{
    type Output = S::Output;
    // See `AndThenHost` — per-branch pipe handle downstream, per-branch checkouts
    // at the terminal.
    type Handle = <S::Checkouts as SeamScatter>::Handle;
    type Checkouts = S::Checkouts;

    fn output_pipe(&self) -> Option<Pipe<S::Output>> {
        <S::Checkouts as SeamScatter>::output_pipe_view(&self.handle)
    }

    fn handle(&self) -> Self::Handle {
        self.handle.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // MID-GRAPH re-scatter — twin of `AndThenHost::execute` (see it for the full
        // rationale); the closure additionally gets `&Context`.
        let (src_cos, deps) = self.source.gather_checkouts(ec, ExecMode::Pipelined)?;
        let (value, homes) = src_cos.split();
        // Reusable: `Arc::clone` the closure and clone a fresh `Context` per run,
        // then move both into a fresh one-shot callable for the worker thread.
        // The closure (`Fn`) re-runs on every replay; captures are borrowed via
        // the Arc rather than move-consumed.
        let f = Arc::clone(&self.f);
        let context = ec.context().clone();
        let (out_value, out_deps) =
            run_host_seam::<S::Output, _>(value, deps, ec, move |view| (*f)(&context, view))?;
        <S::Checkouts as SeamScatter>::scatter(&self.handle, out_value, homes, &out_deps);
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(S::Output, Deps)>
    where
        Self: Sized,
    {
        self.execute(ec, mode)?;
        <S::Checkouts as SeamScatter>::reconstruct(&self.handle)
    }

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(S::Output, Deps, Option<BoxedHome<S::Output>>)>
    where
        Self: Sized,
        S::Output: Send + 'static,
    {
        self.execute(ec, mode)?;
        <S::Checkouts as SeamScatter>::reconstruct_home(&self.handle)
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        _mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // TERMINAL / CHECKOUT gather — the #212 pass-1 path, UNCHANGED (twin of
        // `AndThenHost::gather_checkouts`; closure also gets `&Context`).
        let (src_cos, deps) = self.source.gather_checkouts(ec, ExecMode::Pipelined)?;
        let (value, homes) = src_cos.split();
        let f = Arc::clone(&self.f);
        let context = ec.context().clone();
        let (out_value, out_deps) =
            run_host_seam::<S::Output, _>(value, deps, ec, move |view| (*f)(&context, view))?;
        let checkouts = <S::Checkouts as CheckoutSplit>::reassemble(out_value, homes);
        Ok((checkouts, out_deps))
    }

    fn reclaim_undelivered(&self) {
        // Mid-graph mop-up — twin of `AndThenHost::reclaim_undelivered`.
        <S::Checkouts as SeamScatter>::reclaim(&self.handle);
        self.source.reclaim_undelivered();
    }

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
    }

    fn bind_slots(&self, binder: &mut SlotBinder) {
        // Pass-through: recurse into the source so `bind`/`call` reach its slots.
        self.source.bind_slots(binder);
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
///
/// **Reusable / replayable.** The callback is `Fn` (not `FnOnce`) behind an
/// `Arc`, so a `.profiled()` graph can be `sync`'d repeatedly — each replay
/// `Arc::clone`s the callback and re-registers a fresh one-shot shim on that run's
/// marker, so the callback fires once per run with that run's timestamps. Profiling
/// is a pure host side-effect (no rehoming value), so unlike the host seam there is
/// no home/checkout threading — this is strictly a subset of the `and_then_host`
/// reusability change.
pub struct Profiled<S: DeviceOp, F> {
    source: S,
    // Reusable: the callback is kept in an `Arc` and re-invoked on every replay
    // (each run boxes a fresh `FnOnce` shim that calls the `Fn`). Was a
    // `Mutex<Option<F>>` drained once — a one-shot that broke a second `sync`.
    cb: Arc<F>,
    out: Pipe<S::Output>,
}

/// Extension trait adding [`profiled`](Self::profiled) to every [`DeviceOp`].
/// Separate from [`DeviceOpExt`] to mirror the old layer's
/// `DeviceOperationProfileExt`. Blanket-implemented.
pub trait DeviceProfileExt: DeviceOp + Sized {
    /// Register `cb` to receive the wall-clock [`ProfilingInfo`](crate::ProfilingInfo) for everything
    /// `self` enqueued onto the chain's queue. The closure fires on an OpenCL
    /// callback thread when the marker event completes. See [`Profiled`].
    ///
    /// **Reusable / replayable.** `cb` is `Fn` (not `FnOnce`), so a `.profiled()`
    /// graph can be `sync`'d repeatedly — the callback re-fires each run with that
    /// run's timestamps (borrow / `Arc` / clone captures, don't move-consume them).
    fn profiled<F>(self, cb: F) -> Profiled<Self, F>
    where
        F: Fn(Result<crate::ProfilingInfo>) + Send + Sync + 'static,
    {
        Profiled {
            source: self,
            cb: Arc::new(cb),
            out: Pipe::new(),
        }
    }
}
impl<T: DeviceOp> DeviceProfileExt for T {}

impl<S, F> DeviceOp for Profiled<S, F>
where
    S: DeviceOp,
    // `collect_home` (used in `execute` to thread the source's return home so a
    // profiled graph replays) requires the output be `Send + 'static` — every real
    // buffer/scalar output is.
    S::Output: Send + 'static,
    F: Fn(Result<crate::ProfilingInfo>) + Send + Sync + 'static,
{
    // Profiling is a host side-effect; the chain's data flow is unchanged.
    type Output = S::Output;

    fn output_pipe(&self) -> Option<Pipe<S::Output>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        use crate::Launcher;
        // Gather the source WITH its return home (any arity) — profiling is a
        // transparent passthrough, so the source's home must ride through to the
        // terminal `Checkout` for the graph to REPLAY (else a caller-owned / minted
        // source buffer never rehomes and the 2nd `sync` reports "already lent").
        // A multi-output source collapses to `home == None` (the documented
        // by-value boundary) — profiling wraps a single logical output in practice.
        let (value, source_deps, home) = self.source.collect_home(ec, ExecMode::Pipelined)?;
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
        // Reusable: `Arc::clone` the callback and wrap it in a fresh one-shot
        // `FnOnce` shim per run. `register_profiling_callback` takes a boxed
        // `FnOnce` (fired exactly once when THIS run's marker completes); the
        // clone lets the underlying `Fn` re-fire on the next replay's marker.
        let cb = Arc::clone(&self.cb);
        crate::register_profiling_callback(&marker, Box::new(move |info| (*cb)(info)))?;
        // The marker becomes this op's completion event for downstream
        // chaining (it subsumes the source's events); thread the source's home so
        // the terminal `Checkout` rehomes it and the graph re-arms.
        self.out.put_home(value, vec![wrap_event(marker)], home);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
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

    // Atomicity pre-pass: validate all inputs before any enqueue, mirroring wait_on.
    // Surfaced SYNCHRONOUSLY here (before the queue is acquired and any command is
    // enqueued), returned as an eager `Errored` future so the caller sees an
    // unsatisfiable-input error at `run()` time, not at poll/await.
    if let Err(e) = chain.check_ready() {
        return DeviceChainFuture::Errored(Some(e));
    }

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
