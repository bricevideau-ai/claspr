//! Tier 2 chain-entry macros. `#[macro_export]`; they expand to the eager
//! device-graph free functions re-exported at the `claspr` root (`upload`,
//! `download`, `alloc_zero`, `device_alloc_uninit`, `fill_device_uninit`,
//! `write_device_uninit`, the SVM / USM analogs, …).
//!
//! These are sugar for the common chain-entry shapes; the eager free fns are
//! the canonical surface — `foo(args)` defaults the marker to `ReadWrite`, and
//! `foo_as(args, M)` infers an explicit marker from a zero-sized witness value
//! (no turbofish). Two arms per macro:
//!   - default arm:   `foo!(args)`        — marker defaults to `ReadWrite`.
//!   - marker arm:    `foo!(args; M)`     — `M` is a marker *value* witness
//!     (e.g. `Frozen`), forwarded to the `_as` constructor.

/// Lazy zero-init `DeviceSlice<T, M>` — `alloc_zero(N)`.
/// `device_slice_alloc_zero!(T, N)` for the default marker,
/// `device_slice_alloc_zero!(T, N; M)` for an explicit one (`M` is a marker
/// value, e.g. `HostReadOnly`, passed as a witness).
#[macro_export]
macro_rules! device_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::eager::alloc_zero::<$t>($n)
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::eager::alloc_zero_as::<$t, _>($n, $m)
    };
}

/// Lazy `DeviceSliceUninit<T, M>` alloc-producing leaf — `device_alloc_uninit`.
/// Downstream chain stages transition via `fill_device_uninit` /
/// `write_device_uninit` or `unsafe { uninit.assume_init() }`.
#[macro_export]
macro_rules! device_slice_alloc_uninit {
    ($t:ty, $n:expr) => {
        $crate::eager::device_alloc_uninit::<$t>($n)
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::eager::device_alloc_uninit_as::<$t, _>($n, $m)
    };
}

/// Lazy alloc + fill — `device_alloc_uninit(N).and_then(|u| fill_device_uninit(u, v))`.
/// `device_slice_filled!(value, N)` / `device_slice_filled!(value, N; M)`.
#[macro_export]
macro_rules! device_slice_filled {
    ($v:expr, $n:expr) => {
        $crate::DeviceOpExt::and_then($crate::eager::device_alloc_uninit::<_>($n), move |u| {
            $crate::eager::fill_device_uninit(u, $v)
        })
    };
    ($v:expr, $n:expr; $m:expr) => {
        $crate::DeviceOpExt::and_then(
            $crate::eager::device_alloc_uninit_as::<_, _>($n, $m),
            move |u| $crate::eager::fill_device_uninit(u, $v),
        )
    };
}

/// Lazy alloc + `CL_MEM_COPY_HOST_PTR`. Works for **any marker** (including
/// `Frozen` / `ReadOnly`) — data baked in at creation. Backed by the `upload`
/// (`from_slice`) leaf. `src` must be `Vec<T>` / `Box<[T]>` / `Arc<[T]>`.
#[macro_export]
macro_rules! device_slice_from_slice {
    ($data:expr) => {
        $crate::eager::upload::<_, _>($data)
    };
    ($data:expr; $m:expr) => {
        $crate::eager::upload_as::<_, _, _>($data, $m)
    };
}

/// SVM analog of `device_slice_alloc_zero!` — `MappedSlice` zero-init via
/// `mapped_alloc_uninit(N).and_then(|u| fill_mapped_uninit(u, T::default()))`.
#[macro_export]
macro_rules! mapped_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::DeviceOpExt::and_then($crate::eager::mapped_alloc_uninit::<$t>($n), |u| {
            $crate::eager::fill_mapped_uninit(u, <$t as ::core::default::Default>::default())
        })
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::DeviceOpExt::and_then(
            $crate::eager::mapped_alloc_uninit_as::<$t, _>($n, $m),
            |u| $crate::eager::fill_mapped_uninit(u, <$t as ::core::default::Default>::default()),
        )
    };
}

/// SVM analog of `device_slice_alloc_uninit!`.
#[macro_export]
macro_rules! mapped_slice_alloc_uninit {
    ($t:ty, $n:expr) => {
        $crate::eager::mapped_alloc_uninit::<$t>($n)
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::eager::mapped_alloc_uninit_as::<$t, _>($n, $m)
    };
}

/// SVM analog of `device_slice_filled!`.
#[macro_export]
macro_rules! mapped_slice_filled {
    ($v:expr, $n:expr) => {
        $crate::DeviceOpExt::and_then($crate::eager::mapped_alloc_uninit::<_>($n), move |u| {
            $crate::eager::fill_mapped_uninit(u, $v)
        })
    };
    ($v:expr, $n:expr; $m:expr) => {
        $crate::DeviceOpExt::and_then(
            $crate::eager::mapped_alloc_uninit_as::<_, _>($n, $m),
            move |u| $crate::eager::fill_mapped_uninit(u, $v),
        )
    };
}

/// SVM analog of `device_slice_from_slice!` — alloc + SVM write of a host slice.
#[macro_export]
macro_rules! mapped_slice_from_slice {
    ($data:expr) => {{
        let data = $data;
        let n = data.len();
        $crate::DeviceOpExt::and_then($crate::eager::mapped_alloc_uninit::<_>(n), move |u| {
            $crate::eager::write_mapped_uninit(u, data)
        })
    }};
    ($data:expr; $m:expr) => {{
        let data = $data;
        let n = data.len();
        $crate::DeviceOpExt::and_then(
            $crate::eager::mapped_alloc_uninit_as::<_, _>(n, $m),
            move |u| $crate::eager::write_mapped_uninit(u, data),
        )
    }};
}

/// SVM analog of `upload!` — alloc + SVM write of a host slice.
#[macro_export]
macro_rules! mapped_slice_upload {
    ($src:expr) => {
        $crate::mapped_slice_from_slice!($src)
    };
    ($src:expr; $m:expr) => {
        $crate::mapped_slice_from_slice!($src; $m)
    };
}

/// Lazy [`USMSliceUninit<T, M>`](crate::USMSliceUninit) alloc.
#[macro_export]
macro_rules! usm_slice_alloc_uninit {
    ($t:ty, $n:expr) => {
        $crate::eager::usm_alloc_uninit::<$t>($n)
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::eager::usm_alloc_uninit_as::<$t, _>($n, $m)
    };
}

/// USM zero-init alloc — `usm_alloc_uninit(N).and_then(|u| fill_usm_uninit(u, T::default()))`.
#[macro_export]
macro_rules! usm_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::DeviceOpExt::and_then($crate::eager::usm_alloc_uninit::<$t>($n), |u| {
            $crate::eager::fill_usm_uninit(u, <$t as ::core::default::Default>::default())
        })
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::DeviceOpExt::and_then($crate::eager::usm_alloc_uninit_as::<$t, _>($n, $m), |u| {
            $crate::eager::fill_usm_uninit(u, <$t as ::core::default::Default>::default())
        })
    };
}

/// `vec!`-shaped sugar producing a [`DeviceSlice<T>`](crate::DeviceSlice) op.
///
/// Two arms mirror [`vec!`](std::vec!):
///
/// - `device_slice![value; count]` — alloc + on-device `clEnqueueFillBuffer`.
///   No host allocation. Expands to [`device_slice_filled!`].
/// - `device_slice![a, b, c]` — upload a host literal (alloc + write). Expands
///   to [`upload!`].
///
/// The two arms have radically different bandwidth profiles even though they
/// look almost identical; choose intentionally.
///
/// ```ignore
/// let buf_op = device_slice![0u32; N];        // on-device fill
/// let buf_op = device_slice![1u32, 2, 3, 4];  // host upload
/// ```
#[macro_export]
macro_rules! device_slice {
    [$value:expr; $count:expr] => {
        $crate::device_slice_filled!($value, $count)
    };
    [$($v:expr),* $(,)?] => {
        $crate::eager::upload!(::std::vec![$($v),*])
    };
}

/// `vec!`-shaped sugar producing a [`MappedSlice<T>`](crate::MappedSlice) op —
/// SVM analog of [`device_slice!`].
///
/// - `mapped_slice![value; count]` — alloc + `clEnqueueSVMMemFill`. Expands to
///   [`mapped_slice_filled!`].
/// - `mapped_slice![a, b, c]` — alloc + SVM write from a host literal. Expands
///   to [`mapped_slice_upload!`].
#[macro_export]
macro_rules! mapped_slice {
    [$value:expr; $count:expr] => {
        $crate::mapped_slice_filled!($value, $count)
    };
    [$($v:expr),* $(,)?] => {
        $crate::mapped_slice_upload!(::std::vec![$($v),*])
    };
}

/// `vec!`-shaped sugar producing a [`USMSlice<T>`](crate::USMSlice) op.
///
/// Both arms expand to the `usm_slice` leaf over a host `Vec<T>` —
/// USMSlice always wraps an existing host allocation, so there's no cheap
/// on-device fill path to distinguish from the literal arm. The macro exists
/// for syntactic symmetry across the tier family.
///
/// ```ignore
/// let buf_op = usm_slice![0u32; N];
/// let buf_op = usm_slice![1u32, 2, 3, 4];
/// ```
#[macro_export]
macro_rules! usm_slice {
    // `usm_slice![v; N]` — wrap host vec![v; N].
    [$value:expr; $count:expr] => {
        $crate::usm_slice::<_>(::std::vec![$value; $count])
    };
    // `usm_slice!(host_vec)` — wrap an existing Vec, default marker. Put this
    // BEFORE the bracket-list arm so single-expr paren calls don't get wrapped
    // in another Vec.
    ($vec:expr) => {
        $crate::usm_slice::<_>($vec)
    };
    // `usm_slice![a, b, c]` — wrap a host vec literal.
    [$($v:expr),* $(,)?] => {
        $crate::usm_slice::<_>(::std::vec![$($v),*])
    };
}

// ── Device-scalar chain-entry leaves ───────────────────────────────────────

/// Lazy seeded [`DeviceScalar<T, M>`](crate::DeviceScalar) alloc leaf — the
/// scalar twin of [`device_slice_alloc_zero!`]. Backed by the
/// [`scalar_value`](crate::scalar_value) leaf (alloc-once + persistent home +
/// reseed-on-replay), so it plugs straight into an `and_then` chain / `bundle!`.
///
/// - `device_scalar_alloc!(value)` — seed with `value`, default `ReadWrite`.
/// - `device_scalar_alloc!(value; M)` — explicit marker witness (e.g. `Frozen`).
///
/// ```ignore
/// let g = scalar_value(0.0f32).and_then(|a| ks.finish(a));  // free-fn form
/// let g = device_scalar_alloc!(0.0f32).and_then(|a| ks.finish(a)); // macro form
/// ```
#[macro_export]
macro_rules! device_scalar_alloc {
    ($v:expr) => {
        $crate::eager::scalar_value($v)
    };
    ($v:expr; $m:expr) => {
        $crate::eager::scalar_value_as($v, $m)
    };
}

/// Lazy zero-init [`DeviceScalar<T, M>`](crate::DeviceScalar) alloc leaf — the
/// scalar twin of [`device_slice_alloc_zero!`]. Backed by
/// [`scalar_zero`](crate::scalar_zero).
///
/// - `device_scalar_zero!(T)` — default `ReadWrite`.
/// - `device_scalar_zero!(T; M)` — explicit marker witness.
#[macro_export]
macro_rules! device_scalar_zero {
    ($t:ty) => {
        $crate::eager::scalar_zero::<$t>()
    };
    ($t:ty; $m:expr) => {
        $crate::eager::scalar_zero_as::<$t, _>($m)
    };
}

// ── Typed slots (reusable-graph holes) ─────────────────────────────────────

/// Declare one or more typed [`Tag`](crate::Tag)s for reusable-graph slots.
///
/// ```ignore
/// slots! { Buf: DeviceSlice<u32>, W: DeviceSlice<f32> }
/// ```
///
/// expands, per entry, to a **source-generic** public tuple struct plus its
/// [`Tag`](crate::Tag) impl (keyed on `Tag<KeyMarker>` — a per-tag type
/// independent of the source `S`):
///
/// ```ignore
/// pub struct Buf<S = DeviceSlice<u32>>(pub S);   // S = the binding SOURCE
/// impl<S> ::claspr::Tag for Buf<S>
/// where S: ::claspr::IntoBound<DeviceSlice<u32>> + 'static {
///     type Value = DeviceSlice<u32>;
///     type Key   = Buf<::claspr::KeyMarker>;   // stable matching identity
///     fn into_value(self) -> DeviceSlice<u32> { self.0.into_bound() }
///     fn source_cell_id(&self) -> Option<usize> { self.0.source_cell_id() }
/// }
/// ```
///
/// The tag is generic over its *source* so ONE constructor spelling accepts both
/// forms with no `.into()`:
/// - `Buf(b)` — a raw buffer/scalar (`S = Value`, identity
///   [`IntoBound`](crate::IntoBound)).
/// - `Buf(co)` — a [`Checkout`](crate::Checkout) over the value (`S =
///   Checkout<Value>`), which **severs** the Checkout's source home (`Lent →
///   Severed`) and the target slot **adopts** the buffer. This is the one-line
///   double-buffer swap `g.mutate_call((In(out_co), Out(in_co)))`.
///
/// The `Key` marker (not `Buf<S>` itself, whose `TypeId` would vary with `S`) is
/// the identity matched against a slot, so a `Checkout`-built binding matches a
/// `slot!(Buf)` (built from the default `Buf<Value>`). Its `Value` is the one
/// buffer/scalar type the tag carries (compile-time fixed). Build a hole with
/// [`slot!`](crate::slot)`(Buf)` and bind it with **plain tuple-struct
/// construction** — `g.bind(Buf(b))` — no `Fn`/`fn_traits`.
#[macro_export]
macro_rules! slots {
    // Trailing comma + at least one entry.
    ( $( $name:ident : $val:ty ),+ $(,)? ) => {
        $(
            #[doc = concat!("Reusable-graph slot tag carrying a `", stringify!($val), "`.")]
            #[doc = ""]
            #[doc = "Build a hole with [`slot!`](crate::slot)`(...)`; bind with"]
            #[doc = "`g.bind(Self(value))` (raw) or `g.bind(Self(checkout))`"]
            #[doc = "(sever-and-adopt). The type param `S` is the binding SOURCE"]
            #[doc = "and defaults to the carried value type."]
            pub struct $name<S = $val>(pub S);

            impl<S> $crate::Tag for $name<S>
            where
                S: $crate::IntoBound<$val> + 'static,
            {
                // Clean display name for slot-error diagnostics: exactly the tag
                // ident (`"Buf"`), with NO `<KeyMarker>` suffix. Matching is by
                // `Key`'s TypeId (below) and is fully independent of this string.
                const NAME: &'static str = stringify!($name);
                type Value = $val;
                // The matching key is `$name<KeyMarker>` — a distinct type per tag
                // (the ident differs) yet INDEPENDENT of the source `S`, so a
                // `Checkout`-built binding (`$name<Checkout<Value>>`) matches a
                // `slot!($name)` (`$name<Value>`). `KeyMarker` is a shared ZST used
                // ONLY for TypeId matching (the display name is `NAME`, above).
                type Key = $name<$crate::eager::KeyMarker>;
                fn into_value(self) -> $val {
                    $crate::IntoBound::into_bound(self.0)
                }
                fn source_cell_id(&self) -> ::core::option::Option<usize> {
                    // Read-only: a raw-value source returns `None`; a `Checkout`
                    // source returns the slot cell its `into_inner` will sever —
                    // the `call`/`mutate_call` phase-0 crossed-swap recogniser.
                    $crate::IntoBound::source_cell_id(&self.0)
                }
            }

            // ── Unified `$name(x)` CallArg surface (value OR pipe, one ctor) ──
            //
            // These per-tag `CallArg` impls are what let a tag be used as an element
            // of a `bind` / `call` tuple. `$name(value)`
            // and `$name(checkout)` BIND by value; `$name(pipe)` WIRES the slot to an
            // upstream pipe (`SlotState::FedByPipe`) — the pipe-feed and value-bind
            // spellings unified into ONE tag constructor (no separate `feed` verb).
            //
            // WHY per-tag CONCRETE sources (not an `impl<Tg: Tag> CallArg for Tg`
            // blanket): cross-crate coherence. A blanket would collide with the pipe
            // impl below for SCALAR-valued tags, since the compiler must assume the
            // `claspr` crate could later add `IntoBound<$val> for Pipe<$val>` and
            // `RecordableBuffer for $val` — a hypothetical future overlap it rejects
            // now. Keying value-bind on the two CONCRETE non-pipe sources makes
            // `$val` / `Checkout` / `Pipe` structurally disjoint type constructors
            // that no upstream impl can unify. See the note by `CallArgs` in eager.rs.

            // Value-bind, raw-value source (`$name(v)` => `$name<$val>`).
            impl $crate::eager::CallArg for $name<$val>
            where
                $val: $crate::eager::SlotEq + $crate::SlotValue,
            {
                fn apply<Op: $crate::DeviceOp>(self, g: &Op) {
                    // Infallible but RECORD-don't-drop: a bind error is recorded into
                    // the graph's deferred-error sink and surfaced FIRST at sync's
                    // check_ready (nothing enqueued), not silently swallowed.
                    $crate::DeviceOpExt::bind_deferred(g, self);
                }
            }

            // Value-bind, `Checkout` source (`$name(co)` => `$name<Checkout<$val>>`)
            // — the sever-and-adopt bind.
            impl $crate::eager::CallArg for $name<$crate::Checkout<$val>>
            where
                $val: $crate::eager::SlotEq + $crate::SlotValue + ::core::marker::Send,
            {
                fn apply<Op: $crate::DeviceOp>(self, g: &Op) {
                    // Infallible but RECORD-don't-drop (see the raw-value arm): the
                    // sever-and-adopt error is recorded into the sink, surfaced at sync.
                    $crate::DeviceOpExt::bind_deferred(g, self);
                }
            }

            // Pipe-feed, `Pipe` source (`$name(pipe)` => `$name<Pipe<V>>`) — installs
            // `SlotState::FedByPipe` at every site the tag appears.
            //
            // BUFFER-ONLY by construction: the `V: RecordableBuffer` bound sits on a
            // TYPE PARAM (conditional existence, not a concrete trivial bound), and
            // only device-buffer families impl `RecordableBuffer` — never scalars
            // (`f32`) or `LaunchSpec`. So for a scalar/launch tag this impl is
            // uninhabited: `F(pipe)` finds no `CallArg` and FAILS TO COMPILE, keeping
            // scalar/launch slots value-only. The `$name<V>: Tag<Value = V>` bound
            // pins `V` to the tag's value type (via the identity `IntoBound<V> for V`)
            // and hands us the concrete tag to feed via `DeviceOpExt::feed_deferred`.
            impl<V> $crate::eager::CallArg for $name<$crate::Pipe<V>>
            where
                V: $crate::record::RecordableBuffer + ::core::marker::Send + 'static,
                $name<V>: $crate::Tag<Value = V>,
            {
                fn apply<Op: $crate::DeviceOp>(self, g: &Op) {
                    // Infallible but RECORD-don't-drop: an absent-tag feed error is
                    // recorded into the graph's deferred-error sink and surfaced at
                    // sync's check_ready, not silently swallowed.
                    $crate::DeviceOpExt::feed_deferred::<$name<V>>(g, self.0);
                }
            }
        )+
    };
}

/// Build an unbound typed graph hole for a [`Tag`](crate::Tag) declared with
/// [`slots!`](crate::slots). `slot!(Buf)` plugs into any position a concrete
/// buffer does (kernel args, `download`/`fill`/`write`/copy sources):
///
/// ```ignore
/// slots! { Buf: DeviceSlice<u32> }
/// let g = ks.scale([N], slot!(Buf), 2u32).and_then(download);
/// let out = g.bind(Buf(b)).sync(&ctx)?;  // bind (consuming), then run; re-runnable
/// ```
///
/// Expands to [`SlotHandle::<Tag>::new()`](crate::SlotHandle::new) — a fresh empty
/// cell filled by a later [`bind`](crate::DeviceOpExt::bind)`(Tag(value))`.
#[macro_export]
macro_rules! slot {
    ( $tag:ty ) => {
        $crate::SlotHandle::<$tag>::new()
    };
}
