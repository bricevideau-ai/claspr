//! Typed-slot machinery for the eager graph — the reuse substrate: `SlotState`
//! (5-state) + `ScalarSlotState`, the `SlotBinder` apply walk, `SlotEq`/`SlotValue`
//! (idempotency + shared-fill), `DeferredErrors`, the per-tag `Tag`/`slots!` support,
//! and the arg adapters `ScalarInput`/`SlotHandle`/`ToInput`/`ToScalarInput`. The
//! `Input::Slot` variant + `Pipe` edge live in `eager.rs`; this is the state +
//! binding logic. `SlotBinder` internals are `pub(crate)` — its drivers
//! (`fold_bind`/`probe_bind`) are `DeviceOpExt` methods in the parent.

use super::*;

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
    /// pipe-fed slot is a misuse the runtime never performs; see
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
pub(crate) fn peek_deferred(sink: &Mutex<Vec<Error>>) -> Option<Error> {
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
    pub(crate) id: TypeId,
    pub(crate) mode: BindMode,
    /// The boxed value (`Box<dyn Any + Send>` over the tag's `Value`). For a
    /// **fan-out** binding it stays `Some` for the whole walk (each cell gets a
    /// clone via [`clone`](Self::clone)); for a **move-only** binding it is
    /// `take()`n into the first matching cell (then `None` = consumed).
    pub(crate) value: Option<Box<dyn Any + Send>>,
    /// Pointer-identity (buffer) OR value (scalar/launch) comparison for the `bind`
    /// idempotency leg: given the currently-`Bound` value and the new value (both
    /// as `&dyn Any` over `Tg::Value`), reports whether they are the "same" binding
    /// ([`SlotEq`]). Captured at construction so the generic, `SlotEq`-free
    /// [`try_bind_slot`](Input::try_bind_slot) can invoke it.
    pub(crate) eq: SlotEqFn,
    /// The fan-out clone hook ([`SlotValue::fill_clone`]). Returns `Some(box)` to
    /// fill a cell WITHOUT consuming the binder (clone-able → fill-all), or `None`
    /// to signal move-only (the binder then `take`s its single value once).
    pub(crate) clone: SlotCloneFn,
    /// The verdict of applying this binder, threaded out of the type-erased fold.
    /// `Ok(())` until a matching slot records a conflict / checked-out error.
    pub(crate) outcome: Result<()>,
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
    pub(crate) matched: usize,
    /// The slot cell ids (`Arc::as_ptr`) this bind MATCHED — one per matching cell
    /// (fan-out fills N). Precise per-slot CB invalidation reads this after a `Mutate`
    /// to clear exactly the CBs whose `captured_slots` intersect the re-bound cells.
    /// Populated in [`try_bind_slot`](Input::try_bind_slot) alongside `matched`
    /// (both probe and real passes — harmless in probe, but only the real Mutate pass
    /// feeds `invalidate_cbs`).
    pub(crate) matched_cells: Vec<usize>,
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
    pub(crate) feed_pipe: Option<Box<dyn Any + Send>>,
    /// **Infallible-apply marker.** `true` ONLY for a binder built by the consuming,
    /// infallible [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call) path (via
    /// the deferred `bind`/`feed` helpers behind [`CallArg::apply`]). When set, the
    /// [`try_bind_slot`](Input::try_bind_slot) walk CAPTURES a
    /// [`captured_sink`](Self::captured_sink) so a bind error can be RECORDED into a
    /// graph-reachable [`DeferredErrors`] sink instead of dropped — see the sink type
    /// docs. `false` for every fluent-verb / probe binder (they surface errors
    /// eagerly and never touch a sink).
    pub(crate) deferred: bool,
    /// The [`DeferredErrors`] sink captured from the FIRST slot cell the walk visits
    /// (only when [`deferred`](Self::deferred) is set). After the walk, the infallible
    /// apply path pushes any recorded error here (`SlotConflict`/`SlotSevered`/
    /// `SlotCheckedOut` from [`outcome`](Self::outcome), or `SlotNoSuchTag` when
    /// [`matched`](Self::matched)`== 0`). `None` for a graph with no slots (nothing to
    /// bind — no error is possible) and for every non-deferred binder.
    pub(crate) captured_sink: Option<DeferredErrors>,
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
            matched_cells: ::std::vec::Vec::new(),
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
    pub(crate) fn feed<Tg: Tag>(pipe: Pipe<Tg::Value>) -> Self {
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
            matched_cells: ::std::vec::Vec::new(),
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
    pub(crate) fn probe<Tg: Tag>(mode: BindMode, severable_cells: Vec<usize>) -> Self {
        SlotBinder {
            id: TypeId::of::<Tg::Key>(),
            mode,
            // Inert: probe mode inspects state only; these are never invoked.
            value: None,
            eq: Box::new(|_, _| false),
            clone: Box::new(|_| None),
            outcome: Ok(()),
            matched: 0,
            matched_cells: ::std::vec::Vec::new(),
            probe: true,
            severable_cells,
            feed_pipe: None,
            deferred: false,
            captured_sink: None,
        }
    }

    /// Whether this binder is a read-only [`probe`](Self::probe) (phase-0 dry run).
    pub(crate) fn is_probe(&self) -> bool {
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
    pub(crate) fn probe_lent(&self, cell_id: usize, name: &'static str) -> Result<()> {
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
    pub(crate) fn fill_clone(&self) -> Option<Box<dyn Any + Send>> {
        self.value.as_deref().and_then(|v| (self.clone)(v))
    }

    /// Take the binder's value out (the move-only, take-once path), marking it
    /// consumed. Returns `None` if already taken.
    pub(crate) fn take_value(&mut self) -> Option<Box<dyn Any + Send>> {
        self.value.take()
    }

    /// Produce the concrete value to deposit into ONE matching cell, downcast to `V`.
    /// The shared take-or-clone-and-downcast step used by BOTH the 5-state resource
    /// [`Input::try_bind_slot`](crate::Input::try_bind_slot) and the 2-state scalar
    /// [`ScalarInput::try_bind_slot`] — the value-extraction scaffolding is identical
    /// (only their state-machine arms differ). `fanout` (from [`is_fanout`](Self::is_fanout),
    /// passed in so its clone isn't recomputed) selects clone-into-this-cell vs
    /// move-out-once. Returns `None` on the impossible downcast mismatch (the tag's
    /// `Key` TypeId already pinned `V`); on a move-path mismatch the taken value is put
    /// back so a correctly-typed slot can still see it.
    pub(crate) fn provide<V: 'static>(&mut self, fanout: bool) -> Option<V> {
        let boxed = if fanout {
            self.fill_clone()?
        } else {
            self.take_value()?
        };
        match boxed.downcast::<V>() {
            Ok(v) => Some(*v),
            Err(boxed) => {
                if !fanout {
                    self.value = Some(boxed);
                }
                None
            }
        }
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

    /// The slot cell ids this bind matched — the input to precise per-slot CB
    /// invalidation (`fold_bind` passes these to `invalidate_cbs` on a `Mutate`).
    pub fn matched_cells(&self) -> &[usize] {
        &self.matched_cells
    }

    /// Mark this binder as belonging to the INFALLIBLE, consuming
    /// [`bind`](DeviceOpExt::bind) / [`call`](DeviceOpExt::call) apply path, so the
    /// [`bind_slots`](DeviceOp::bind_slots) walk captures a
    /// [`captured_sink`](Self::captured_sink) (see [`DeferredErrors`]). Only the
    /// deferred `bind`/`feed` helpers behind [`CallArg::apply`] call this.
    pub(crate) fn mark_deferred(&mut self) {
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
    pub(crate) fn record_deferred(&self, name: &'static str) {
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

// ═══ Slot arg adapters (ScalarInput / SlotHandle / ToInput / ToScalarInput) ═══

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

    /// Stable slot-cell identity for precise CB invalidation — `Some(ptr)` for a
    /// [`Slot`](ScalarInput::Slot) (the `Arc::as_ptr` of its cell), `None` for a
    /// `Concrete`. A scalar slot can never thread through a pipe (scalars are
    /// value-only), so this id IS the whole origin set: a CB that baked this
    /// scalar's bytes depends on exactly this one slot. The resource analogue is
    /// [`Input::slot_cell_id`].
    pub fn slot_cell_id(&self) -> Option<usize> {
        match self {
            ScalarInput::Slot { cell, .. } => Some(Arc::as_ptr(cell) as *const () as usize),
            ScalarInput::Concrete(_) => None,
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
        // Scalar-slot cell identity for precise invalidation (a mutated scalar tag
        // like gray-scott F/K clears the CBs that baked its value).
        binder
            .matched_cells
            .push(Arc::as_ptr(cell) as *const () as usize);

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
        // The shared take-or-clone-and-downcast step (see `SlotBinder::provide`).
        let provide = |binder: &mut SlotBinder| -> Option<V> { binder.provide::<V>(fanout) };

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
    pub(crate) fn into_input(self) -> Input<Tg::Value> {
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
