//! Compile-time slot sets: a graph's TYPE carries the tags still to bind, so
//! `sync` exists only on a fully curried graph.
//!
//! A set is a type-level list of tag KEYS ([`Tag::Key`]): [`HNil`], `HCons<K,
//! Rest>`, or [`Dynamic`] — "not tracked: checked at run time, as before". Every
//! [`DeviceOp`] names its set as [`DeviceOp::Slots`] (default [`HNil`]): kernel
//! launchers compute theirs from their argument types, combinators combine their
//! children's, and `bind` / `call` remove the tags they fill. The runtime is
//! unchanged — this layer only rejects at compile time what `sync` would have
//! rejected at run time.
//!
//! Writing generic code over it needs no plumbing: [`SlotList::Append`] is total
//! (a generic associated type on the bound of `DeviceOp::Slots` itself), so
//! composing any ops type-checks without extra bounds. A generic subgraph
//! function that wants to `sync` pins its parts to `Slots = HNil` ("returns a
//! fully curried op"); code that does not want static tracking erases to
//! [`Dynamic`] ([`erase_unchecked`](DeviceOpExt::erase_unchecked)).
//!
//! Repeated tags (one tag at several slot sites, the shared-slot fan-out) cannot
//! be removed one at a time without type-level tag equality, which stable Rust
//! cannot express. A graph that repeats a tag declares its set once
//! ([`erase_as`](DeviceOpExt::erase_as)): each repeat is proven a member of the
//! declared distinct list, which collapses them.
use super::*;

/// The empty set: nothing left to bind.
pub struct HNil;

/// The set `{K} ∪ Rest`, `K` a tag key ([`Tag::Key`]).
pub struct HCons<K, Rest>(PhantomData<fn() -> (K, Rest)>);

/// Not tracked statically — the run-time checks (as before typed slot sets)
/// are the only checks. Absorbs whatever it is combined with, and always allows
/// `sync`.
pub struct Dynamic;

/// A slot set — the bound on [`DeviceOp::Slots`].
pub trait SlotList: 'static {
    /// `Self ∪ R` (concatenated; repeats kept). [`Dynamic`] on either side
    /// absorbs. Total, so generic code composes without bounds.
    type Append<R: SlotList>: SlotList;
    /// `L ∪ Self` for a statically known `L` — the second half of the double
    /// dispatch behind [`Append`](Self::Append).
    #[doc(hidden)]
    type AfterKnown<L: Known>: SlotList;
}

/// A statically known set ([`HNil`] or `HCons`), as opposed to [`Dynamic`].
pub trait Known: SlotList {
    /// `Self` followed by the known set `R`.
    #[doc(hidden)]
    type Concat<R: Known>: Known;
}

impl SlotList for HNil {
    type Append<R: SlotList> = <R as SlotList>::AfterKnown<HNil>;
    type AfterKnown<L: Known> = L;
}
impl Known for HNil {
    type Concat<R: Known> = R;
}

impl<K: 'static, Rest: Known> SlotList for HCons<K, Rest> {
    type Append<R: SlotList> = <R as SlotList>::AfterKnown<Self>;
    type AfterKnown<L: Known> = <L as Known>::Concat<Self>;
}
impl<K: 'static, Rest: Known> Known for HCons<K, Rest> {
    type Concat<R: Known> = HCons<K, <Rest as Known>::Concat<R>>;
}

impl SlotList for Dynamic {
    type Append<R: SlotList> = Dynamic;
    type AfterKnown<L: Known> = Dynamic;
}

/// Build a slot set type from tag types: `SlotSet![Grid, F, K]`. Each entry is
/// mapped to its [`Tag::Key`], so a tag reads the same whatever binding source
/// it is later bound from.
#[macro_export]
macro_rules! SlotSet {
    () => { $crate::eager::HNil };
    ($t:ty $(, $rest:ty)* $(,)?) => {
        $crate::eager::HCons<<$t as $crate::Tag>::Key, $crate::SlotSet![$($rest),*]>
    };
}

// ── Removing a bound tag ───────────────────────────────────────────────────

/// Index of a tag in a known set: here.
pub struct Here;
/// Index of a tag in a known set: further down.
pub struct There<I>(PhantomData<fn() -> I>);
/// Index into [`Dynamic`]: any tag may be bound; the set stays [`Dynamic`].
pub struct Untracked;

/// `Self` with the tag key `K` removed, at the inferred position `I`. Missing
/// impl ⇒ the tag is not (or no longer) in the graph's set.
#[diagnostic::on_unimplemented(
    // Not `{Self}`: rustc reports the innermost step of the list search, whose
    // `Self` is only the tail of the graph's set.
    message = "this graph has no unbound slot `{K}`",
    label = "this tag is not in the graph, or was already bound",
    note = "if the tag appears at several slot sites, bind every slot at once with `.call_all((..))`, or declare the graph's slots first with `.erase_as::<SlotSet![..], _>()`"
)]
pub trait Remove<K, I>: SlotList {
    /// The set without `K`.
    type Out: SlotList;
}
impl<K: 'static, Rest: Known> Remove<K, Here> for HCons<K, Rest> {
    type Out = Rest;
}
impl<K: 'static, H: 'static, Rest: Known + Remove<K, I>, I> Remove<K, There<I>> for HCons<H, Rest>
where
    <Rest as Remove<K, I>>::Out: Known,
{
    type Out = HCons<H, <Rest as Remove<K, I>>::Out>;
}
impl<K> Remove<K, Untracked> for Dynamic {
    type Out = Dynamic;
}

/// Remove every key of the known set `Keys` (a `call` tuple's tags), each at its
/// own inferred position.
pub trait RemoveAll<Keys, Idxs>: SlotList {
    /// The set without any of `Keys`.
    type Out: SlotList;
}
impl<S: SlotList> RemoveAll<HNil, HNil> for S {
    type Out = S;
}
impl<S, K, Ks, I, Is> RemoveAll<HCons<K, Ks>, HCons<I, Is>> for S
where
    S: Remove<K, I>,
    <S as Remove<K, I>>::Out: RemoveAll<Ks, Is>,
{
    type Out = <<S as Remove<K, I>>::Out as RemoveAll<Ks, Is>>::Out;
}

// ── Declaring a set (collapsing repeats, choosing an order) ─────────────────

/// Every key of `Self` (repeats allowed) is a member of the DISTINCT declared set
/// `Target`, each found at an inferred position. Searching the distinct list
/// keeps each lookup unambiguous however often a tag repeats in `Self`.
#[diagnostic::on_unimplemented(
    message = "the graph's slots `{Self}` are not all in the declared set `{Target}`",
    label = "a slot tag of this graph is missing from the declaration"
)]
pub trait SubsetOf<Target, Idxs> {}
impl<Target> SubsetOf<Target, HNil> for HNil {}
impl<K, Rest, Target, I, Is> SubsetOf<Target, HCons<I, Is>> for HCons<K, Rest>
where
    Target: Remove<K, I>,
    Rest: SubsetOf<Target, Is>,
{
}

// ── Renaming ───────────────────────────────────────────────────────────────

/// `Self` with the key `From` replaced by `To` (at the inferred position `I`).
pub trait RenameKey<From, To, I>: SlotList {
    /// The renamed set.
    type Out: SlotList;
}
impl<K: 'static, Rest: Known, From, To: 'static, I> RenameKey<From, To, I> for HCons<K, Rest>
where
    HCons<K, Rest>: Remove<From, I>,
    <HCons<K, Rest> as Remove<From, I>>::Out: Known,
{
    type Out = HCons<To, <HCons<K, Rest> as Remove<From, I>>::Out>;
}
impl<From, To> RenameKey<From, To, Untracked> for Dynamic {
    type Out = Dynamic;
}

// ── The terminal gate ──────────────────────────────────────────────────────

/// Nothing statically known is left to bind — required by the terminals.
#[diagnostic::on_unimplemented(
    message = "this graph still has unbound slots: `{Self}`",
    label = "bind them first (`.bind(Tag(v))` / `.call((Tag(v), ..))`)",
    note = "`sync` / `wait_on` / `run` are only available on a fully curried graph"
)]
pub trait AllBound: SlotList {}
impl AllBound for HNil {}
impl AllBound for Dynamic {}

// ── What an argument contributes ───────────────────────────────────────────

/// The slots a SCALAR or GRID argument contributes to the kernel launch it is
/// passed to: none for a value, `{Tag}` for a `slot!(Tag)`. Buffer arguments
/// carry the same through [`ToInput::Slots`].
///
/// Every `Copy` type contributes none — every scalar, grid literal and
/// [`LaunchSpec`](crate::LaunchSpec), a custom `#[repr(C)] Copy` scalar, and a
/// generic `Real: Copy` in width-generic code — so nothing needs to opt in. A
/// [`SlotHandle`] (not `Copy`) contributes its tag.
pub trait SlotsOf {
    /// The contributed set.
    type Slots: SlotList = HNil;
}
impl<T: Copy + 'static> SlotsOf for T {}
impl<Tg: Tag> SlotsOf for SlotHandle<Tg> {
    type Slots = HCons<Tg::Key, HNil>;
}
