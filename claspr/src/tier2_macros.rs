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
        $crate::alloc_zero::<$t>($n)
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::alloc_zero_as::<$t, _>($n, $m)
    };
}

/// Lazy `DeviceSliceUninit<T, M>` alloc-producing leaf — `device_alloc_uninit`.
/// Downstream chain stages transition via `fill_device_uninit` /
/// `write_device_uninit` or `unsafe { uninit.assume_init() }`.
#[macro_export]
macro_rules! device_slice_alloc_uninit {
    ($t:ty, $n:expr) => {
        $crate::device_alloc_uninit::<$t>($n)
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::device_alloc_uninit_as::<$t, _>($n, $m)
    };
}

/// Lazy alloc + fill — `device_alloc_uninit(N).and_then(|u| fill_device_uninit(u, v))`.
/// `device_slice_filled!(value, N)` / `device_slice_filled!(value, N; M)`.
#[macro_export]
macro_rules! device_slice_filled {
    ($v:expr, $n:expr) => {
        $crate::DeviceOpExt::and_then($crate::device_alloc_uninit::<_>($n), move |u| {
            $crate::fill_device_uninit(u, $v)
        })
    };
    ($v:expr, $n:expr; $m:expr) => {
        $crate::DeviceOpExt::and_then($crate::device_alloc_uninit_as::<_, _>($n, $m), move |u| {
            $crate::fill_device_uninit(u, $v)
        })
    };
}

/// Lazy alloc + `CL_MEM_COPY_HOST_PTR`. Works for **any marker** (including
/// `Frozen` / `ReadOnly`) — data baked in at creation. Backed by the `upload`
/// (`from_slice`) leaf. `src` must be `Vec<T>` / `Box<[T]>` / `Arc<[T]>`.
#[macro_export]
macro_rules! device_slice_from_slice {
    ($data:expr) => {
        $crate::upload::<_, _>($data)
    };
    ($data:expr; $m:expr) => {
        $crate::upload_as::<_, _, _>($data, $m)
    };
}

/// Lazy host-to-device upload (`from_slice` create + copy). `upload!(src)` /
/// `upload!(src; M)`. `src` must be `Vec<T>` / `Box<[T]>` / `Arc<[T]>`.
#[macro_export]
macro_rules! upload {
    ($src:expr) => {
        $crate::upload::<_, _>($src)
    };
    ($src:expr; $m:expr) => {
        $crate::upload_as::<_, _, _>($src, $m)
    };
}

/// Lazy non-blocking device-to-host read. `download!(buf)`. Marker inferred from
/// the input buffer; bound `M: HostReadable`.
#[macro_export]
macro_rules! download {
    ($buf:expr) => {
        $crate::download($buf)
    };
}

/// SVM analog of `device_slice_alloc_zero!` — `MappedSlice` zero-init via
/// `mapped_alloc_uninit(N).and_then(|u| fill_mapped_uninit(u, T::default()))`.
#[macro_export]
macro_rules! mapped_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::DeviceOpExt::and_then($crate::mapped_alloc_uninit::<$t>($n), |u| {
            $crate::fill_mapped_uninit(u, <$t as ::core::default::Default>::default())
        })
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::DeviceOpExt::and_then($crate::mapped_alloc_uninit_as::<$t, _>($n, $m), |u| {
            $crate::fill_mapped_uninit(u, <$t as ::core::default::Default>::default())
        })
    };
}

/// SVM analog of `device_slice_alloc_uninit!`.
#[macro_export]
macro_rules! mapped_slice_alloc_uninit {
    ($t:ty, $n:expr) => {
        $crate::mapped_alloc_uninit::<$t>($n)
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::mapped_alloc_uninit_as::<$t, _>($n, $m)
    };
}

/// SVM analog of `device_slice_filled!`.
#[macro_export]
macro_rules! mapped_slice_filled {
    ($v:expr, $n:expr) => {
        $crate::DeviceOpExt::and_then($crate::mapped_alloc_uninit::<_>($n), move |u| {
            $crate::fill_mapped_uninit(u, $v)
        })
    };
    ($v:expr, $n:expr; $m:expr) => {
        $crate::DeviceOpExt::and_then($crate::mapped_alloc_uninit_as::<_, _>($n, $m), move |u| {
            $crate::fill_mapped_uninit(u, $v)
        })
    };
}

/// SVM analog of `device_slice_from_slice!` — alloc + SVM write of a host slice.
#[macro_export]
macro_rules! mapped_slice_from_slice {
    ($data:expr) => {{
        let data = $data;
        let n = data.len();
        $crate::DeviceOpExt::and_then($crate::mapped_alloc_uninit::<_>(n), move |u| {
            $crate::write_mapped_uninit(u, data)
        })
    }};
    ($data:expr; $m:expr) => {{
        let data = $data;
        let n = data.len();
        $crate::DeviceOpExt::and_then($crate::mapped_alloc_uninit_as::<_, _>(n, $m), move |u| {
            $crate::write_mapped_uninit(u, data)
        })
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
        $crate::usm_alloc_uninit::<$t>($n)
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::usm_alloc_uninit_as::<$t, _>($n, $m)
    };
}

/// USM zero-init alloc — `usm_alloc_uninit(N).and_then(|u| fill_usm_uninit(u, T::default()))`.
#[macro_export]
macro_rules! usm_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::DeviceOpExt::and_then($crate::usm_alloc_uninit::<$t>($n), |u| {
            $crate::fill_usm_uninit(u, <$t as ::core::default::Default>::default())
        })
    };
    ($t:ty, $n:expr; $m:expr) => {
        $crate::DeviceOpExt::and_then($crate::usm_alloc_uninit_as::<$t, _>($n, $m), |u| {
            $crate::fill_usm_uninit(u, <$t as ::core::default::Default>::default())
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
        $crate::upload!(::std::vec![$($v),*])
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
