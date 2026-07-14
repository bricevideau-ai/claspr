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
//! This module IS the Tier 2 device-graph layer. [`DeviceEnqueue`] is the minimal
//! enqueue contract a few primitive leaves (host-view map/unmap, the polymorphic
//! `copy_to` family) delegate their raw enqueue body to.
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
    Buffer, Context, DeviceSlice, DeviceSliceUninit, Error, Fillable, HostReadable, HostWritable,
    MappedSlice, MappedSliceUninit, MemMode, ReadWrite, Result, USMSlice, USMSliceUninit,
};
use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

// ── Deps: the event wait-list threaded through the graph ────────────────

/// A single tracked event in a [`Deps`] set. `Arc`-wrapped so it can be cheaply
/// shared across parallel branches in `bundle!` / `fan_out` without extra
/// `clRetainEvent` calls.
///
/// A newtype (not a bare `Arc<Event>`) so it can be a [`BTreeSet`](std::collections::BTreeSet) key: `Ord`/`Eq`
/// are by the underlying `cl_event` POINTER identity, which is exactly the dedup a
/// wait-list wants (the same event depended on twice is one wait, not two). It
/// [`AsRef<Event>`] and derefs to the `Arc<Event>`, so existing `.as_ref().get()`
/// call sites are unchanged.
#[derive(Clone, Debug)]
pub struct Dep(Arc<crate::Event>);

impl Dep {
    /// The raw `cl_event` handle — the wait-list element handed to OpenCL.
    pub fn get(&self) -> crate::cl_event {
        self.0.get()
    }

    /// Wrap an already-`Arc`'d event as a [`Dep`] — for the paths that must ALSO
    /// retain the `Arc` elsewhere (e.g. `register_use` on an SVM buffer, or a
    /// user-event kept to signal later), so the event isn't cloned into a second
    /// `Arc`.
    pub fn from_arc(event: Arc<crate::Event>) -> Self {
        Dep(event)
    }
}

impl AsRef<crate::Event> for Dep {
    fn as_ref(&self) -> &crate::Event {
        self.0.as_ref()
    }
}

impl std::ops::Deref for Dep {
    type Target = Arc<crate::Event>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Identity by the backing `cl_event` pointer — two `Dep`s are "the same wait" iff
// they name the same OpenCL event. This is what makes [`Deps`] a dedup'ing set.
impl PartialEq for Dep {
    fn eq(&self, other: &Self) -> bool {
        self.0.get() == other.0.get()
    }
}
impl Eq for Dep {}
impl PartialOrd for Dep {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Dep {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0.get() as usize).cmp(&(other.0.get() as usize))
    }
}

/// The wait-list / produced-event **set** threaded through every op's
/// [`execute`](DeviceOp::execute). Empty at chain start; one element per device op
/// the previous step enqueued; multi-element after a parallel join
/// (`bundle`/`fan_out`) collapses children's events into the marker that joins them.
///
/// A [`BTreeSet`](std::collections::BTreeSet) (not a `Vec`): events are an unordered wait-list, and the same
/// event reaching a node via two paths must produce ONE wait, not a duplicate — the
/// set dedups by `cl_event` identity for free (mirrors how CB sync points are a
/// `BTreeSet<cl_sync_point_khr>`). Convert to the raw OpenCL wait-list ONLY at an
/// enqueue boundary, via [`deps_to_wait_list`].
pub type Deps = std::collections::BTreeSet<Dep>;

/// Borrow each [`Dep`] as `&Event` for an `after_all(...)` call on a Tier 1 op
/// builder.
pub fn deps_as_events(deps: &Deps) -> impl Iterator<Item = &crate::Event> {
    deps.iter().map(|d| d.as_ref())
}

/// Convert a [`Deps`] set into the raw `cl_event` wait-list an OpenCL enqueue takes.
/// This is the ONE place `Deps` crosses the FFI boundary into a `Vec` — every
/// `clEnqueue*` / `clCommand*` consumer calls this instead of re-spelling
/// `deps.iter().map(|d| d.as_ref().get()).collect()`.
pub fn deps_to_wait_list(deps: &Deps) -> Vec<crate::cl_event> {
    deps.iter().map(|d| d.get()).collect()
}

/// Wrap an opencl3 [`Event`](crate::Event) in a [`Dep`].
pub fn wrap_event(event: crate::Event) -> Dep {
    Dep(Arc::new(event))
}

/// A single-element [`Deps`] set from one freshly-produced event — the common
/// "this op enqueued one command, here is its completion event" result.
pub fn single_dep(event: crate::Event) -> Deps {
    let mut d = Deps::new();
    d.insert(wrap_event(event));
    d
}

/// The terminal wait + error reconciliation shared by both `wait_on` paths
/// (fast/no-seam and start-gated/seam). Blocks on every completion event in
/// `deps`, then decides the caller-facing result:
///
/// - a worker's stashed rich error (`ec.take_host_error()`) is authoritative and
///   wins over a raw cl_event cascade — a failing host seam may signal a negative
///   user event whose status does NOT cascade to us (pocl), so a non-empty stash
///   is itself the failure signal even when every `wait()` "succeeded";
/// - otherwise the first `wait()` failure (as `Error::OpenCl`) surfaces;
/// - otherwise `Ok(checkouts)`.
///
/// Blocking-mode leaves already waited inline, but pipelined upstream stages (and
/// kernels, which have no native blocking enqueue) carry events here, so the wait
/// is always needed. Factored out so the two enqueue strategies share ONE copy of
/// the stash-beats-cascade precedence (a subtle invariant that must not drift).
fn wait_deps_reconcile<C>(deps: &Deps, ec: &ExecutionContext<'_>, checkouts: C) -> Result<C> {
    let mut wait_err: Option<Error> = None;
    for d in deps {
        if let Err(code) = d.as_ref().wait() {
            wait_err.get_or_insert(Error::OpenCl(code));
        }
    }
    match ec.take_host_error() {
        Some(rust_err) => Err(rust_err),
        None => match wait_err {
            Some(cascade) => Err(cascade),
            None => Ok(checkouts),
        },
    }
}

// ── DeviceEnqueue: minimal raw-enqueue contract for delegated primitives ──
//
// A handful of eager leaves (the host-view acquire/release ops in `host_view.rs`
// and the polymorphic `copy_to` family in `copy.rs`) can't be re-derived inline:
// they reach into private fields and own per-family `clEnqueue*` bodies. Rather
// than duplicate those bodies, the eager wrapper holds the buffer/view and
// delegates to a small op type whose only job is one non-blocking enqueue
// returning `(Output, Deps)`. This trait is that contract — a single `run` method,
// no terminals, no combinators, no blanket.

/// One non-blocking enqueue: take the upstream `deps` as the wait-list, enqueue,
/// and return the produced value plus the events the enqueue created. Implemented
/// by the few primitive ops the eager graph delegates to (host-view map/unmap,
/// the `copy_to` family).
pub trait DeviceEnqueue: Send + Sized {
    /// The host value the enqueue produces.
    type Output: Send;
    /// Enqueue against `ec` with `deps` as the wait-list; return `(value, Deps)`.
    fn run(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)>;

    /// CB twin of [`run`](Self::run) for the copy leaf: perform the same type
    /// conversion (an `Uninit` dst returns `Init` — the CB writes every byte at
    /// enqueue, so the `assume_init` is sound exactly as in `run`) but do NOT enqueue.
    ///
    /// - `builder = Some(b)`: RECORD the copy command into `b` (waiting on `waits`);
    ///   the returned sync point is `Some` on success. Returns `None` overall if the
    ///   command is unavailable/ineligible (a mixed cl_mem+SVM pair, or the driver
    ///   lacks the SVM command) — the caller then falls back to the per-op path.
    /// - `builder = None` (replay): convert types only, no record; sync point `None`.
    ///
    /// Default `None` — only the copy ops override it.
    #[allow(clippy::type_complexity)]
    fn record_cb(
        self,
        _builder: Option<&crate::record::CbBuilder>,
        _waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        None
    }
}

// ── Cell<T>: interior-mutable resource slot (the reusable-graph primitive) ──

/// An interior-mutable slot holding (or temporarily not holding) a resource.
/// The unifying primitive of the reusable graph: a [`Pipe`] is a cell that
/// also carries [`Deps`]; a [`Concrete`](Input::Concrete) input is a cell that
/// is *lent* during a run and *returned* on `Checkout` drop. `Arc` so a run can
/// hold a clone to deposit the value back home.
pub type Cell<T> = Arc<Mutex<Option<T>>>;

mod cb;
// Glob re-export: each item keeps its OWN visibility — the `pub` CB surface the
// kernel macro references (`new_cb_cache`, `cb_collect_external`, `CbCache`,
// `CbWalk`) stays reachable at `::claspr::eager::<name>`; the `pub(crate)` helpers
// stay crate-internal.
pub use cb::*;

mod slots;
pub use slots::*;

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

/// RAII guard over a buffer lent out of its origin cell by
/// [`resolve_home`](Input::resolve_home), for the window between the lend and the
/// deposit ([`put_home`](Pipe::put_home) / [`rehome_consumed`]).
///
/// `resolve_home` moves a buffer OUT of its cell (`Bound`/`Concrete → Lent`); the
/// op then runs a **fallible** enqueue and only deposits the buffer back on
/// success. If the enqueue (or a blocking `wait`) errors via `?`, the raw
/// `(value, home)` tuple would just drop — and since neither the bare tuple nor a
/// `BoxedHome` has a rehome-on-drop, the origin cell stays permanently `Lent`:
/// the graph is silently poisoned (a later run reports "busy"/`SlotUnbound`
/// forever) instead of failing cleanly and staying retryable. This hits the
/// hottest path — every kernel launch and every fill/copy/download `execute`.
///
/// `LentGuard` closes that window: it owns the lent `(value, home)` and, on drop,
/// rehomes the value to its cell — EXACTLY like [`PipePayload`]'s drop and a
/// [`Checkout`] drop (re-arm `Bound`, no-op for a concrete `Cell`). The success
/// path calls [`disarm`](Self::disarm) to reclaim `(value, home)` for the deposit,
/// which defuses the guard so it does nothing on drop. A minted/homeless lend
/// (`home == None`) simply releases on drop, unchanged.
pub(crate) struct LentGuard<T> {
    value: Option<T>,
    home: Option<BoxedHome<T>>,
}

impl<T> LentGuard<T> {
    /// Arm a guard over a freshly-lent `(value, home)`.
    pub(crate) fn new(value: T, home: Option<BoxedHome<T>>) -> Self {
        Self {
            value: Some(value),
            home,
        }
    }

    /// Borrow the lent value mutably for the enqueue (which may fail — if it does,
    /// the guard is still armed and rehomes on the `?` unwind).
    pub(crate) fn value_mut(&mut self) -> &mut T {
        self.value
            .as_mut()
            .expect("LentGuard value already disarmed")
    }

    /// Borrow the lent value immutably (read-only enqueues, e.g. image/read).
    pub(crate) fn value(&self) -> &T {
        self.value
            .as_ref()
            .expect("LentGuard value already disarmed")
    }

    /// Success path: defuse the guard and reclaim `(value, home)` for the deposit
    /// ([`Pipe::put_home`] / [`rehome_consumed`]). After this the guard's drop is a
    /// no-op — the deposit now owns the rehome obligation.
    pub(crate) fn disarm(mut self) -> (T, Option<BoxedHome<T>>) {
        let value = self.value.take().expect("LentGuard disarmed twice");
        let home = self.home.take();
        (value, home)
    }
}

impl<T> Drop for LentGuard<T> {
    fn drop(&mut self) {
        // Armed at drop ⇒ the enqueue failed (or an early return skipped `disarm`):
        // return the lent buffer to its origin cell so the graph is left unchanged
        // and re-runnable, rather than stranded in `Lent`. Mirrors `PipePayload`
        // and `Checkout` drop. `disarm` clears `value`, making this a no-op.
        if let (Some(value), Some(home)) = (self.value.take(), self.home.take()) {
            home.rehome(value);
        }
        // else: disarmed (deposited) or homeless (nothing to return).
    }
}

#[cfg(test)]
mod lent_guard_tests {
    use super::*;

    // Model a lend: empty the cell into a guard homed on that same cell (what
    // `resolve_home` does for a concrete `Cell`).
    fn lend(cell: &Cell<u32>) -> LentGuard<u32> {
        let value = cell.lock().unwrap().take().expect("cell was empty");
        let home: BoxedHome<u32> = Box::new(Arc::clone(cell));
        LentGuard::new(value, Some(home))
    }

    // Dropping an ARMED guard (the enqueue-failed path) must return the value to
    // its origin cell — the anti-stranding property. Without the guard's Drop the
    // cell would stay empty ("Lent" forever).
    #[test]
    fn armed_drop_rehomes_to_origin_cell() {
        let cell: Cell<u32> = Arc::new(Mutex::new(Some(99)));
        {
            let _guard = lend(&cell);
            assert!(
                cell.lock().unwrap().is_none(),
                "lent: cell empty during run"
            );
            // guard dropped here WITHOUT disarm → simulates a failed enqueue
        }
        assert_eq!(
            *cell.lock().unwrap(),
            Some(99),
            "armed guard drop must rehome the lent value"
        );
    }

    // `disarm` (the success path) hands the value back to the caller and defuses
    // the guard, so its later drop is a no-op — the deposit owns the rehome now.
    #[test]
    fn disarm_defuses_and_returns_value() {
        let cell: Cell<u32> = Arc::new(Mutex::new(Some(7)));
        let (value, home) = {
            let guard = lend(&cell);
            guard.disarm()
        };
        assert_eq!(value, 7, "disarm returns the lent value");
        assert!(
            cell.lock().unwrap().is_none(),
            "disarmed guard must NOT rehome (deposit owns it now)"
        );
        // The caller's deposit would normally re-fill via `home`; do it explicitly
        // to confirm the home still points at the origin cell.
        home.expect("concrete cell has a home").rehome(value);
        assert_eq!(*cell.lock().unwrap(), Some(7));
    }

    // A homeless lend (minted buffer, `home == None`) just releases on drop —
    // no panic, nothing to rehome.
    #[test]
    fn homeless_guard_drop_is_noop() {
        let guard = LentGuard::new(42u32, None);
        drop(guard); // must not panic
    }
}

/// Identity rehome: an output returns to a cell of its own type (the in-place
/// case — fill/scale/kernel-buffer-arg/copy's same-typed sides), putting the value
/// back into the cell so the next replay reuses the same buffer.
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

    fn cb_addable(&self) -> bool {
        // A bare pipe-as-op is a pure structural passthrough — it aliases the
        // upstream producer's storage cell (no device command, no re-deposit), so it
        // is trivially CB-addable. CRUCIAL for real graphs: CG feeds raw `Pipe`
        // handles as `bundle*` branches (`bundle6(p, ap, …)`, `bundle2(x, rsnew)`);
        // without this override a `Pipe` branch reports the default `false`, which
        // ANDs the whole bundle — and thus the whole iteration graph — down to the
        // per-op fallback (no kernel ever recorded into a CB). Since a `Pipe` shares
        // the producer's `cell_id`, the producer's registered sync points are already
        // found by a downstream consumer's `sp_lookup` on the SAME cell — nothing to
        // re-register here.
        true
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
                Ok((v, single_dep(crate::Event::new(raw)), home))
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
    ///   wiring of [`slot_home`](Self::slot_home) through `CopyHome`, so it threads
    ///   even when the copy retypes the output.
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
            Input::Concrete(_) => None,
            Input::Pipe(p) => Some(p.cell_id()),
            // A slot FED BY an upstream pipe (`call((Tag(pipe),))` → `FedByPipe`)
            // IS a producer→consumer edge, just carried through the slot machinery
            // rather than a direct `Input::Pipe`. Its upstream pipe's `cell_id` is
            // the key a CB-mode consumer's `sp_lookup` needs to find the producer's
            // sync point — WITHOUT this the cross-sub-graph edge (e.g. gray-scott's
            // unroll-2 step-2 laplacians reading step-1's output) is dropped and the
            // recorded command gets an empty wait-list (a race). The per-op path
            // already drains this same pipe in `resolve_home`; this exposes the same
            // edge to the record path. A non-`FedByPipe` slot (Unbound/Bound/Lent/
            // Severed) has no upstream pipe → `None`, like a concrete cell.
            Input::Slot { cell, .. } => match &*cell.lock().unwrap() {
                SlotState::FedByPipe(pipe) => Some(pipe.cell_id()),
                _ => None,
            },
        }
    }

    /// The identity (`Arc::as_ptr`) of this input's SLOT cell, if it is an
    /// [`Slot`](Input::Slot) — the key precise per-slot CB invalidation matches a
    /// mutated tag against (`mutate_bind` re-binds a slot cell; a CB that baked a
    /// buffer/scalar from this cell is stale). `None` for a concrete or pipe input.
    /// Independent of the slot's STATE (Unbound/Bound/FedByPipe/…): the cell identity
    /// is stable across binds, which is exactly what a re-bind targets.
    pub fn slot_cell_id(&self) -> Option<usize> {
        match self {
            Input::Slot { cell, .. } => Some(Arc::as_ptr(cell) as *const () as usize),
            Input::Concrete(_) | Input::Pipe(_) => None,
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
        // Record the matched cell identity for precise per-slot CB invalidation (a
        // subsequent `Mutate` clears exactly the CBs that baked a buffer from it).
        binder
            .matched_cells
            .push(Arc::as_ptr(cell) as *const () as usize);

        // Three mutually-exclusive kinds of binder, each its own phase (all leave
        // the walk free to visit the next cell — none of these consumes):
        //   1. PROBE — read-only dry run; never touches the cell.
        //   2. FEED  — installs a `FedByPipe` source; fan-out, no value.
        //   3. VALUE — the ordinary set/mutate bind; the state × mode matrix.
        if binder.is_probe() {
            Self::probe_bind(binder, cell, name);
            return;
        }
        if binder.feed_pipe.is_some() {
            Self::feed_bind(binder, cell);
            return;
        }
        // A move-only binder is consumed after its single value lands; bail. A
        // fan-out binder never sets `value = None`, so it keeps filling cells. (Kept
        // out of `apply_value_bind` so a consumed move-only skips the cell lock.)
        if binder.value.is_none() {
            return;
        }
        Self::apply_value_bind(binder, cell, name);
    }

    /// Phase 1 — the read-only PROBE (phase-0 dry run of `call`/`mutate_call`).
    /// Inspect the cell's state WITHOUT filling / taking / replacing, recording the
    /// verdict the phase-2 fold WOULD produce on the POST-sever state (a `Lent` cell
    /// a tuple `Checkout` will sever is predicted as `Severed`; see `probe_lent`).
    /// Records into `binder.outcome` exactly like the real fold, so `fold_probe`
    /// surfaces the first error having severed / mutated NOTHING. The value-equality
    /// leg of `Set` on a `Bound` cell is the ONE case a probe cannot decide (the
    /// value lives in an unsevered `Checkout`) — treated OK here, leaving phase 2 to
    /// catch a genuine `SlotConflict`; that is the documented residual.
    fn probe_bind(binder: &mut SlotBinder, cell: &SlotCell<T>, name: &'static str) {
        let cell_id = Arc::as_ptr(cell) as usize;
        match &*cell.lock().unwrap() {
            // Both verbs fill a virgin / re-arm a bound cell → OK. `Set` on `Bound`
            // is the value-dependent residual (treated OK here). A `FedByPipe` cell
            // is treated like `Bound`: a value bind over it would overwrite (Mutate)
            // / conflict (Set), but the runtime never value-binds a pipe-fed slot, so
            // it is inert here (a `feed` binder — the only writer of this state — is
            // never a probe).
            SlotState::Unbound | SlotState::Bound(_) | SlotState::FedByPipe(_) => {}
            // `Set` rejects a severed slot; `Mutate` re-arms it.
            SlotState::Severed => {
                if binder.mode == BindMode::Set {
                    binder.outcome = Err(Error::SlotSevered(name));
                }
            }
            // Post-sever prediction: tuple-held → Severed (Set fails / Mutate OK);
            // external-held → stays Lent (both fail SlotCheckedOut).
            SlotState::Lent => {
                if let Err(e) = binder.probe_lent(cell_id, name) {
                    binder.outcome = Err(e);
                }
            }
        }
    }

    /// Phase 2 — PIPE-FEED install (the `feed` verb). Deposit
    /// `FedByPipe(pipe.clone())` into this cell — a fan-out, so every matching site
    /// is fed. Unconditional over the current state: the common case installs onto a
    /// virgin (`Unbound`) slot freshly built by the subgraph; re-feeding an
    /// already-`FedByPipe` cell just re-installs the same-or-new pipe. `Lent` should
    /// not occur (a pipe-fed slot is never lent to a `Checkout`), but overwriting it
    /// is still sound — the pipe is drained fresh next run. Runs BEFORE the
    /// `value.is_none()` bail (a feed binder carries no value, so it would otherwise
    /// be misread as "consumed" and skip every cell).
    fn feed_bind(binder: &mut SlotBinder, cell: &SlotCell<T>)
    where
        T: 'static,
    {
        if let Some(boxed) = &binder.feed_pipe
            && let Some(pipe) = boxed.downcast_ref::<Pipe<T>>()
        {
            *cell.lock().unwrap() = SlotState::FedByPipe(pipe.clone());
        }
    }

    /// Phase 3 — the ordinary VALUE bind: resolve the state × mode matrix, filling /
    /// overwriting / conflicting per the table in [`try_bind_slot`]'s docs. Called
    /// only for a non-probe, non-feed binder that still carries a value. A fan-out
    /// binder CLONES its value (so it can fill the next cell too); a move-only binder
    /// TAKES its single value. `provide` is called at most once per cell, lazily, so
    /// an idempotent no-op / conflict / sever-reject path costs no clone or move.
    fn apply_value_bind(binder: &mut SlotBinder, cell: &SlotCell<T>, name: &'static str)
    where
        T: Send + 'static,
    {
        let fanout = binder.is_fanout();
        // The shared take-or-clone-and-downcast step (see `SlotBinder::provide`).
        // Returns `None` only on the impossible downcast mismatch (TypeId already
        // pinned `T == Tag::Value`).
        let provide = |binder: &mut SlotBinder| -> Option<T> { binder.provide::<T>(fanout) };

        let mut guard = cell.lock().unwrap();
        match &*guard {
            // Virgin — never bound. Both verbs fill it (a `bind` is the slot's first
            // declaration).
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
            // would let the Checkout's drop rehome the OLD buffer over the NEW one —
            // a silent clobber — so BOTH verbs hard-error. (Buffer slots only;
            // non-resource slots are never `Lent`.)
            SlotState::Lent => {
                binder.outcome = Err(Error::SlotCheckedOut(name));
            }
            // A value bind onto a pipe-fed slot. `Set` conflicts (the slot is already
            // sourced by an upstream pipe); `Mutate` overwrites the pipe source with
            // the value. The runtime never value-binds a pipe-fed slot, so this arm is
            // inert in practice — present only for exhaustiveness + correctness (a
            // value bind should not silently no-op over a live feed).
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

/// One flattened node in a **read-only structural dump** of an eager graph,
/// produced by [`DeviceOp::dump_graph`]. This is a debug/introspection surface,
/// SEPARATE from [`describe`](DeviceOp::describe) (whose output is snapshotted as
/// a golden and must not change): `dump_graph` additionally surfaces the SHARED
/// pipe edges that the struct nesting alone does not reveal — an upstream
/// producer whose output pipe fans out to several consumers appears here as the
/// SAME `cell_id` in multiple nodes' [`in_cells`](GraphNode::in_cells), which is
/// exactly the DAG structure a fork-tree (`cg`) lacks and a pipe-DAG
/// (`gray-scott`) has. See [`graph_edge_table`].
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Nesting depth in the struct tree (0 = root). Purely presentational —
    /// indentation for the human-readable tree; edges are keyed by cell id.
    pub depth: usize,
    /// Best-effort label for the node (kernel name, `and_then`, or a leaf label).
    pub name: String,
    /// The [`cell_id`](Pipe::cell_id)s of this node's OUTPUT pipe(s) — the
    /// producer side of a pipe edge.
    pub out_cells: Vec<usize>,
    /// The [`cell_id`](Pipe::cell_id)s of this node's pipe-fed INPUTS (each
    /// resource arg whose [`Input::pipe_cell_id`] is `Some`) — the consumer side.
    /// A cell that also appears in some node's `out_cells` is a real graph edge.
    pub in_cells: Vec<usize>,
    /// Whether this (sub)graph is command-buffer-addable ([`DeviceOp::cb_addable`]).
    pub cb_addable: bool,
    /// Whether this (sub)graph contains a host seam ([`DeviceOp::contains_host_seam`]).
    pub seam: bool,
}

/// Build the flat pipe-edge table from a [`dump_graph`](DeviceOp::dump_graph)
/// node list: for every producer `cell_id` (any node's `out_cells`), the indices
/// of the nodes that CONSUME it (their `in_cells` contain that id).
///
/// A producer with **out-degree > 1** is a SHARED fan-out edge — one pipe feeding
/// several dispatch sites. That is the read-only signal distinguishing a genuine
/// pipe-DAG (`gray-scott`: `combine` consumes four upstream pipes; each
/// `laplacian`'s field-passthrough + scratch outputs both thread forward) from a
/// fork-tree (`cg`), where every producer feeds exactly one consumer.
pub fn graph_edge_table(nodes: &[GraphNode]) -> Vec<(usize, Vec<usize>)> {
    let mut table: Vec<(usize, Vec<usize>)> = Vec::new();
    for node in nodes {
        for &out in &node.out_cells {
            // Collect every node index whose in_cells contain this producer id.
            let consumers: Vec<usize> = nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.in_cells.contains(&out))
                .map(|(i, _)| i)
                .collect();
            table.push((out, consumers));
        }
    }
    table
}

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

    /// The downstream-facing [`Handle`](Self::Handle) — what a downstream `and_then`
    /// closure receives. A single-output op returns its output pipe (`self.out.clone()`
    /// — i.e. `self.output_pipe().unwrap()`); a multi-output combinator returns its
    /// per-element tuple. **No default body**: the single-output form isn't expressible
    /// generically (the trait can't construct `Self::Handle` without a `From<Pipe<
    /// Output>>` bound that would then break generic callers on tuple-`Handle` ops), so
    /// every op writes it explicitly — one line for a leaf.
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

    /// Best-effort short label for this node in a [`dump_graph`](Self::dump_graph)
    /// dump. Default: the FIRST line of a fresh [`describe`](Self::describe) vec
    /// (so a leaf/kernel labels itself), or `"op"` if `describe` pushed nothing.
    /// Combinators override with a fixed name (`"and_then"`, …). NOT snapshotted —
    /// this is a debug-only surface, distinct from `describe`'s golden output.
    fn node_label(&self) -> String {
        let mut v = Vec::new();
        self.describe(&mut v);
        v.into_iter().next().unwrap_or_else(|| "op".to_string())
    }

    /// **Read-only structural dump** — flatten this (sub)graph into
    /// [`GraphNode`]s (see there for why this is separate from `describe`), one
    /// per op, recording each node's output pipe cell id(s), pipe-fed input cell
    /// id(s), and CB/seam flags. `depth` is the current nesting level; children
    /// recurse at `depth + 1`.
    ///
    /// Default (single-output leaves / kernels the macro doesn't override): push
    /// ONE node whose `out_cells` is this op's [`output_pipe`](Self::output_pipe)
    /// cell (if any), with NO `in_cells` — correct for concrete leaves
    /// (`alloc_zero`, seeded fill) which have no upstream pipe edges. Combinators
    /// (`AndThen`) and multi-output / multi-input kernel ops override this to
    /// recurse and to emit their shared pipe edges — the whole point of the dump.
    fn dump_graph(&self, depth: usize, out: &mut Vec<GraphNode>) {
        let out_cells = self
            .output_pipe()
            .map(|p| p.cell_id())
            .into_iter()
            .collect();
        out.push(GraphNode {
            depth,
            name: self.node_label(),
            out_cells,
            in_cells: Vec::new(),
            cb_addable: self.cb_addable(),
            seam: self.contains_host_seam(),
        });
    }

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
    fn contains_host_seam(&self) -> bool {
        false
    }

    /// Whether this WHOLE (sub)graph can be added to a command buffer as-is
    /// (design v2 eligibility): every node is a device command (`fill`, `copy`,
    /// kernel), a structural passthrough (`forward`/`lift`/`arced`/`Pipe`), or a
    /// combinator over CB-addable children — and NONE is a host-touching leaf
    /// (upload/download/host-view/scalar host transfer) or a host seam.
    ///
    /// **Default `false`** — a node not KNOWN to be CB-addable disqualifies its
    /// subtree (fail-safe: an un-forked op would enqueue normally inside a CB walk
    /// and desynchronize the sync-point ordering). Device leaves + the structural
    /// combinators override to `true` (combinators AND their children). This is
    /// coarser than the spec's per-subtree transfer-bracketing (transfers should
    /// bracket a CB, not disqualify the whole graph) — that finer segmentation is
    /// a follow-up; today an upload→kernel→download graph runs the whole thing on
    /// the per-op path (correct, just not CB-accelerated), while a fully device-
    /// resident graph (CG all-device, gray-scott) takes the CB.
    fn cb_addable(&self) -> bool {
        false
    }

    /// How many command-buffer commands (`clCommand{NDRangeKernel,FillBuffer,
    /// CopyBuffer}KHR`) this (sub)graph would record if run as ONE command buffer —
    /// a STATIC weight computed once at construction. Command leaves (`fill`, `copy`,
    /// the macro kernel `Op`) are `1`; structural passthroughs (`Pipe`, `forward`,
    /// `lift`, `arced`) and host/transfer leaves are `0`; combinators SUM their
    /// children. `Arced` forwards its source's weight.
    ///
    /// **Static under `mutate`:** topology (node set + `and_then_host` seam
    /// placement) is fixed at build, and per-node record-ability is a compile-time
    /// predicate over node TYPE + the image-arg gate + construction-time
    /// `.profiled()`/`.after()` state — never slot state. A `mutate_bind` changes a
    /// slot's value or source (`Bound` ↔ `FedByPipe`), i.e. a dependency EDGE, but a
    /// kernel records exactly one command whether its input is value- or pipe-fed. So
    /// every subtree's weight is invariant under rebind → CB-capable nodes store it in
    /// a field, set once by the constructor from the (already-built) children.
    ///
    /// The boundary-open predicates require `>= 2`: a span of a single command must
    /// NOT open a command buffer — one `clCreateCommandBufferKHR` +
    /// `clFinalizeCommandBufferKHR` + a per-replay `clEnqueueCommandBufferKHR` is pure
    /// overhead versus enqueuing that one command directly, with zero batching
    /// benefit. **Default `0`.**
    fn cbable_weight(&self) -> usize {
        0
    }

    /// This node's OWN [`CbCache`] home, if it is a CB-capable node (design v2).
    /// Default `None` (a node that never creates/homes a command buffer — host
    /// seams, transfers, the identity `Pipe`). CB-capable nodes (`AndThen`,
    /// `Bundle*`, `FanOut`, the structural passthroughs, `Fill`, `CopyTo2`, the
    /// macro kernel `Op`) override to return `Some(&self.cb_cache)`.
    ///
    /// The per-node algorithm reads/writes THIS to "home a CB in yourself" and to
    /// take the replay fast-path. [`invalidate_cbs`](Self::invalidate_cbs) clears
    /// it on mutation.
    fn cb_cache(&self) -> Option<&CbCache> {
        None
    }

    /// **Invalidate the homed command buffers this (sub)graph holds that depend on a
    /// re-bound slot** — called by [`mutate_bind`](DeviceOpExt::mutate_bind) /
    /// [`mutate_call`](DeviceOpExt::mutate_call) with `mutated` = the slot cell ids
    /// just re-bound (from [`SlotBinder::matched_cells`]). PRECISE: clears this node's
    /// [`cb_cache`](Self::cb_cache) iff its homed `FinalizedCb` baked a buffer/scalar
    /// traceable to a mutated slot (`captured_slots ∩ mutated ≠ ∅`), then recurses.
    ///
    /// `captured_slots` is transitive (it includes slots a CB's buffer reached through
    /// pipes / across a host seam — the [`CbReach`](crate::exec_ctx::CbReach)
    /// substrate propagated them at record time and each leaf `note_slot`'d them), so
    /// this covers the FedByPipe-across-seam case a naive "subtree contains the tag"
    /// test would miss. An EMPTY `mutated` set (should not happen from a real mutate)
    /// clears nothing.
    ///
    /// Default: clear own cache if it intersects; combinators override to recurse.
    fn invalidate_cbs(&self, mutated: &std::collections::BTreeSet<usize>) {
        if let Some(cache) = self.cb_cache() {
            cb_cache_invalidate(cache, mutated);
        }
    }

    /// **Collect the stable identities of every homed [`FinalizedCb`] in this
    /// (sub)graph**, appending `(Arc::as_ptr as usize)` for each node that currently
    /// holds a CB. Test/introspection hook (`#[doc(hidden)]`): it walks the SAME
    /// node set [`invalidate_cbs`](Self::invalidate_cbs) does, so a test can assert
    /// that a `mutate_bind` cleared exactly the interior CBs it should (a region-A
    /// CB's id disappears / changes while region-B's id is untouched). Default:
    /// push own id if homed; combinators override to recurse into children.
    #[doc(hidden)]
    fn collect_cb_ids(&self, out: &mut Vec<usize>) {
        if let Some(cache) = self.cb_cache() {
            cb_cache_collect_id(cache, out);
        }
    }

    /// **Stamp the command-buffer completion event onto this node's output
    /// pipe(s)** — the execute-position half of the CB boundary (design v2).
    ///
    /// When a device span is a MID-GRAPH boundary (a maximal seam-free subtree
    /// under a host seam), it is run in [`Build`](CbWalk::Build)/[`LendOnly`](CbWalk::LendOnly)
    /// mode, which deposits its outputs with EMPTY `cl_event` deps (CB-internal
    /// ordering is the sync points). After the boundary enqueues the CB and gets ONE
    /// completion event, a DOWNSTREAM consumer resolving those pipes needs that
    /// event as its wait-list (the event↔sync-point boundary: OUTSIDE the CB,
    /// ordering is `cl_event`s again). This re-deposits each output value with
    /// `deps = evs`, so the seam / next span waits on the whole CB.
    ///
    /// `evs` is normally the CB's single completion event; on the EMPTY-CB path
    /// (a pure-passthrough span that recorded nothing) it is the upstream producers'
    /// events collected into the span's `ext`, so the downstream still waits on the
    /// real work that fed the passthroughs.
    ///
    /// Default: the single [`output_pipe`](Self::output_pipe). Multi-output ops
    /// (kernels, bundles, `CopyTo2`, `FanOut`) override to stamp each element pipe —
    /// mirroring [`reclaim_undelivered`](Self::reclaim_undelivered)'s traversal.
    fn cb_restamp(&self, evs: &Deps) {
        if let Some(pipe) = self.output_pipe()
            && let Some((v, _deps, home)) = pipe.take_home()
        {
            pipe.put_home(v, evs.clone(), home);
        }
    }

    /// Whether this node CONTINUES / HEADS a maximal seam-free command-buffer span
    /// (design v2, finalize-at-close). The span is the longest prefix of the spine
    /// whose leading device work is CB-addable; it CLOSES at the first node that
    /// cannot continue it (a host seam / transfer).
    ///
    /// Default = [`cb_addable`](Self::cb_addable): a leaf / fully-addable subtree
    /// continues the span iff it is entirely addable. An [`AndThen`] overrides it to
    /// `self.source.cb_spine_head_addable()` — the CHAIN continues the span as long
    /// as its *leading source* is addable, EVEN THOUGH the whole chain is
    /// `!cb_addable` (a seam lives further down `next`). That recursion is what lets
    /// the span span multiple spine `AndThen`s: `dot.and_then(bundle4.and_then(seam))`
    /// continues (its leading source `dot` is addable), and the span closes exactly
    /// at the `bundle4.and_then(seam)` level whose `next` is the seam. This is the
    /// stop rule that batches CG's `[xpby, spmv, dot]` (+ the 0-command `bundle4`
    /// passthrough) into ONE CB.
    fn cb_spine_head_addable(&self) -> bool {
        self.cb_addable()
    }
}

/// The canonical `Handle` / `Checkouts` SHAPE of a [`DeviceOp::Output`].
///
/// Every built-in op's [`Handle`](DeviceOp::Handle) and
/// [`Checkouts`](DeviceOp::Checkouts) are *mechanically determined* by its
/// [`Output`](DeviceOp::Output): a single-value output has `Handle = Pipe<O>` /
/// `Checkouts = Checkout<O>` (the trait defaults), and a tuple output has the
/// element-wise tuple of each. So for EVERY op the identity
/// `A::Output: OutputShape<Handle = A::Handle, Checkouts = A::Checkouts>` holds.
///
/// This trait exposes that shape keyed on the OUTPUT TYPE, so a generic subgraph
/// bound can pin the shape ONCE via `Output` and *project* `Handle`/`Checkouts`
/// from it — instead of re-spelling the parallel `Pipe<_>` and `Checkout<_>` tuples
/// (which for a 6-field subgraph is three copies of the same shape). See
/// `examples/cg`'s `solve_with`, whose two closure-result bounds shrink from
/// Output+Handle+Checkouts blocks to `Output` + a one-line `OutputShape` projection.
pub trait OutputShape {
    /// The `DeviceOp::Handle` an op producing `Self` has (`Pipe<Self>`, or the
    /// element-wise tuple of handles).
    type Handle;
    /// The `DeviceOp::Checkouts` an op producing `Self` has (`Checkout<Self>`, or the
    /// element-wise tuple of checkouts).
    type Checkouts;
}

/// A single-value output: `Handle = Pipe<Self>`, `Checkouts = Checkout<Self>` — the
/// [`DeviceOp`] associated-type defaults. Per-family (not a blanket `impl<V>`, which
/// would collide with the tuple impls under coherence).
macro_rules! impl_output_shape_leaf {
    ($($t:ty),+ $(,)?) => { $(
        impl<T: Send + 'static, M: MemMode> OutputShape for $t {
            type Handle = Pipe<$t>;
            type Checkouts = Checkout<$t>;
        }
    )+ };
}
impl_output_shape_leaf!(DeviceSlice<T, M>, MappedSlice<T, M>, USMSlice<T, M>);

/// A device/mapped/usm SCALAR ([`Scalar<B>`], e.g. `DeviceScalar<T,M>`) output.
impl<B: Send + 'static> OutputShape for Scalar<B> {
    type Handle = Pipe<Scalar<B>>;
    type Checkouts = Checkout<Scalar<B>>;
}

/// A host-`Vec` output (a `download` terminal's value).
impl<T: Send + 'static> OutputShape for Vec<T> {
    type Handle = Pipe<Vec<T>>;
    type Checkouts = Checkout<Vec<T>>;
}

/// A tuple output (bundle / multi-branch): the element-wise tuple of each field's
/// shape — exactly what the bundle/tuple `DeviceOp` impls compute for `Handle` /
/// `Checkouts`.
macro_rules! impl_output_shape_tuple {
    ($($ty:ident),+ $(,)?) => {
        impl<$($ty: OutputShape),+> OutputShape for ($($ty,)+) {
            type Handle = ($(<$ty as OutputShape>::Handle,)+);
            type Checkouts = ($(<$ty as OutputShape>::Checkouts,)+);
        }
    };
}
impl_output_shape_tuple!(A0, A1);
impl_output_shape_tuple!(A0, A1, A2);
impl_output_shape_tuple!(A0, A1, A2, A3);
impl_output_shape_tuple!(A0, A1, A2, A3, A4);
impl_output_shape_tuple!(A0, A1, A2, A3, A4, A5);
impl_output_shape_tuple!(A0, A1, A2, A3, A4, A5, A6);
impl_output_shape_tuple!(A0, A1, A2, A3, A4, A5, A6, A7);

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
        let cbable_weight = self.cbable_weight() + next.cbable_weight();
        AndThen {
            source: self,
            next,
            cb_cache: new_cb_cache(),
            cbable_weight,
        }
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
    fn and_then_host<F>(self, f: F) -> AndThenHost<Self>
    where
        Self::Output: crate::mappable::Mappable,
        Self::Checkouts: SeamScatter<Value = Self::Output>,
        F: for<'a> Fn(<Self::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
            + Send
            + Sync
            + 'static,
    {
        // Wrap the no-context closure into the canonical `Fn(&Context, View)` shape
        // (see `HostSeamFn`), so `AndThenHost` is ONE type for both builders.
        AndThenHost {
            source: self,
            f: Arc::new(move |_ctx: &Context, view| f(view)),
            handle: <Self::Checkouts as SeamScatter>::empty_handle(),
        }
    }

    /// Like [`and_then_host`](Self::and_then_host) but the closure also receives
    /// the running [`Context`] (e.g. to read device props). Builds the SAME
    /// [`AndThenHost`] node (its stored closure always takes `&Context`).
    ///
    /// **Reusable / replayable** — same as [`and_then_host`](Self::and_then_host):
    /// the closure is `Fn`, the graph replays, and the closure re-runs each
    /// `sync` (borrow / `Arc` / clone captures, don't move-consume them).
    fn and_then_host_with_context<F>(self, f: F) -> AndThenHost<Self>
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
        AndThenHost {
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
        // CB-mode (design v2): a MUTATE re-bound a slot, so any homed command buffer
        // that captured that slot's buffer/args is stale. PRECISE per-slot: clear only
        // the CBs whose `captured_slots` include a cell just re-bound (the binder
        // recorded them in `matched_cells`), covering FedByPipe-across-seam via the
        // CbReach substrate. Set-once `bind` runs at build BEFORE any CB exists, so it
        // needs no invalidation.
        if matches!(mode, BindMode::Mutate) {
            let mutated: std::collections::BTreeSet<usize> =
                binder.matched_cells().iter().copied().collect();
            self.invalidate_cbs(&mutated);
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
    /// is dispatched through [`CallArgs`] / [`CallArg`] (arity 1..=16), NOT the fluent
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
            // CB-as-EXECUTION-MODE (design v2): if the platform supports command
            // buffers, this whole seam-free graph is ONE maximal CB region — run
            // the boundary protocol (build-or-replay + home in the root's cb_cache
            // + enqueue), which returns the SAME (Checkouts, Deps) shape as the
            // plain gather (the deps are the CB's single completion event). A graph
            // whose root carries no cb_cache, or an SVM/ineligible one, transparently
            // falls back inside `cb_boundary_gather` to the normal per-op path.
            // >= 2: a whole-graph span of a single command (e.g. a bare
            // `fill(buf, v).sync()`) runs per-op — a one-command CB is pure overhead.
            // Exact here (the span IS the whole graph).
            let result = if cb_graph_eligible(self, &ec)
                && self.cb_cache().is_some()
                && self.cbable_weight() >= 2
            {
                cb_boundary_gather(self, &ec, ExecMode::Blocking)
            } else {
                self.gather_checkouts(&ec, ExecMode::Blocking)
            };
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
                    wait_deps_reconcile(&deps, &ec, checkouts)
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
        // terminated command is the pocl hang). A negative `proceed` from a failing
        // seam surfaces as a cl_event cascade; `wait_deps_reconcile` reconciles it
        // with the stashed rich error. Workers are joined AFTER the device wait but
        // BEFORE reading the host-error slot, so no worker's late CL calls
        // (signalling its user events, then dropping its retained queue) race the
        // caller dropping the Context — hence the wait/join/reconcile split here vs.
        // the fast path's single call.
        let mut wait_err: Option<Error> = None;
        for d in &deps {
            if let Err(code) = d.as_ref().wait() {
                wait_err.get_or_insert(Error::OpenCl(code));
            }
        }
        drop(deps);
        ec.join_workers();
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
        let wait_list = deps_to_wait_list(&deps);
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
/// `mutate_all` has no `SlotConflict` leg (mutate overwrites), so it is FULLY
/// all-or-nothing. Order does not matter for *success*: every element binds its own
/// tag independently. Implemented for tuples of arity 1..=16 (mirroring
/// [`KernelArgs`](crate::KernelArgs)). See the `mutate_all_body!` macro for the three
/// phases (probe → sever → fold).
///
/// (The set-once tuple path — [`call`](DeviceOpExt::call) — does NOT route through
/// here; it folds each element via [`CallArg`], see `CallArgs` below. This trait is
/// the `mutate_call` driver only.)
pub trait BindAll {
    /// Fold each element through [`mutate_bind`](DeviceOpExt::mutate_bind) (set/change).
    fn mutate_all<Op: DeviceOp>(self, g: &Op) -> Result<()>;
}

/// The three-phase body of [`BindAll::mutate_all`] (the `mutate_call` driver). Since
/// mutate overwrites, there is no `SlotConflict` residual — this path is FULLY
/// all-or-nothing.
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
/// - **PHASE 1 — sever all sources (`into_value`).** With the probe having proved
///   every element bindable, resolve each source: a `Checkout` severs its home
///   HERE (`Lent → Severed`), before any fold. This is what makes the crossed swap
///   `mutate_call((In(out_co), Out(in_co)))` work — both source slots are `Severed`
///   BEFORE either target is rebound, so neither rebind hits a still-`Lent` slot.
///
/// - **PHASE 2 — fold each resolved value.** With phase 0 already vetting every
///   element and mutate having no conflict leg, phase 2 cannot surface a new error.
macro_rules! mutate_all_body {
    ($g:ident, $($name:ident),+) => {{
        // PHASE 0a — the crossed-swap recogniser: which slot cells phase 1 will
        // sever (Checkout sources contribute their home cell id; raw values `None`).
        // Read-only — `source_cell_id` borrows, never consumes.
        let severable: Vec<usize> =
            [ $( $name.source_cell_id() ),+ ].into_iter().flatten().collect();
        // PHASE 0b — probe EVERY element (read-only). Any failure returns here
        // having severed / mutated NOTHING (the all-or-nothing guarantee).
        $( $g.probe_bind::<$name>(BindMode::Mutate, &severable)?; )+
        // PHASE 1 — sever all Checkout sources first (see macro doc).
        $( let $name = $name.into_value(); )+
        // PHASE 2 — fold each resolved value (mutate has no conflict → cannot error).
        $( $g.fold_bind::<$name>($name, BindMode::Mutate)?; )+
    }};
}

/// Implement [`BindAll`] for one tuple arity. Each element is a `Tag` whose `Value`
/// is `SlotEq` (buffer-handle equality) + `SlotValue`.
macro_rules! impl_bind_all_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name),+> BindAll for ($($name,)+)
        where
            $( $name: Tag, $name::Value: SlotEq + SlotValue, )+
        {
            #[allow(non_snake_case)]
            fn mutate_all<Op: DeviceOp>(self, g: &Op) -> Result<()> {
                let ($($name,)+) = self;
                mutate_all_body!(g, $($name),+);
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
impl_bind_all_tuple!(A, B, C, D, E, F, G, H, I);
impl_bind_all_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_bind_all_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_bind_all_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_bind_all_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_bind_all_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_bind_all_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_bind_all_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

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
/// Implemented for arities 1..=16 (mirroring [`BindAll`]).
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
impl_call_args_tuple!(A, B, C, D, E, F, G, H, I);
impl_call_args_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_call_args_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_call_args_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_call_args_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_call_args_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_call_args_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_call_args_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

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
/// A seam nested MID-graph (the source of a downstream
/// [`and_then`](DeviceOpExt::and_then)) runs via `execute`, and for a multi-output
/// source must expose each branch as its OWN element `Pipe` (not one
/// `Pipe<tuple>`) so the written branches route to separate downstream kernels AND
/// each re-homes across replays. `SeamScatter` provides that: implemented on the
/// SAME `Checkout<O>` + tuple structure as [`CheckoutSplit`] (so it is arity- and
/// nesting-general), it scatters each branch into its own element pipe with its own
/// home — mirroring what a `bundle!`'s `execute` does. A single-output source keeps
/// `Handle = Pipe<O>` (the trait default); only the multi-output case scatters.
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
    /// pipe (so [`AndThen`]'s orphaned-source-deps threading works when a downstream
    /// closure discards the seam's output). For a **multi-output** source, storage is
    /// the per-branch element pipes with no single storage pipe, so this returns
    /// `None` (the same convention `bundle!` / [`CopyTo2`] `output_pipe` use).
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
    /// one home (the nested-in-`and_then` re-arm); a multi-output source
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

mod combinators;
pub use combinators::*;

mod leaves;
pub use leaves::*;

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
        M: HostWritable + Send + 'static,
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
// data, which does not exist at build. (Device-by-index routing is structural —
// see [`OnDevice`] / [`TransferToDevice`] + `DeviceTarget` below.)
// ════════════════════════════════════════════════════════════════════════
