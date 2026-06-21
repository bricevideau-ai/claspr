//! Tier 2 chain-entry macros (folded in from the former `claspr-async` crate).
//! `#[macro_export]`; reference `$crate::…` items re-exported at the `claspr` root.

// ── Tier 2 entry macros ────────────────────────────────────────────
//
// All Tier 2 buffer constructors are macros, not free fns. Two arms
// per macro:
//   - default arm:   foo!(args)                — uses struct's M = ReadWrite default
//   - marker arm:    foo!(args; M)             — turbofishes M explicitly
// Macros expand to `Foo::<T>::new(...)` / `Foo::<T, M>::new(...)`.
// The struct method form is the canonical constructor; macros are
// sugar to skip the type-position turbofish noise at chain entry.

/// Lazy zero-init `DeviceSlice<T, M>` alloc — sugar for
/// `device_slice_alloc_uninit!(T, N).and_then(|u| u.fill(T::default()))`.
/// `device_slice_alloc_zero!(T, N)` for default marker,
/// `device_slice_alloc_zero!(T, N; M)` for explicit.
#[macro_export]
macro_rules! device_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::DeviceOperation::and_then($crate::DeviceSliceAllocUninit::<$t>::new($n), |u| {
            $crate::FillUninit::fill(u, <$t as ::core::default::Default>::default())
        })
    };
    ($t:ty, $n:expr; $m:ty) => {
        $crate::DeviceOperation::and_then($crate::DeviceSliceAllocUninit::<$t, $m>::new($n), |u| {
            $crate::FillUninit::fill(u, <$t as ::core::default::Default>::default())
        })
    };
}

/// Lazy `DeviceSliceUninit<T, M>` alloc. Output is the type-stated
/// uninit wrapper; downstream chain stages transition via the
/// wrapper's methods or `unsafe { uninit.assume_init() }`.
#[macro_export]
macro_rules! device_slice_alloc_uninit {
    ($t:ty, $n:expr) => {
        $crate::DeviceSliceAllocUninit::<$t>::new($n)
    };
    ($t:ty, $n:expr; $m:ty) => {
        $crate::DeviceSliceAllocUninit::<$t, $m>::new($n)
    };
}

/// Lazy alloc + fill — sugar for `device_slice_alloc_uninit!(_, N).and_then(|u| u.fill(value))`.
/// `device_slice_filled!(value, N)` / `device_slice_filled!(value, N; M)`.
/// Dispatches Runtime vs DeviceKernel fill via the marker's `FillStrategy`.
#[macro_export]
macro_rules! device_slice_filled {
    ($v:expr, $n:expr) => {
        $crate::DeviceOperation::and_then($crate::DeviceSliceAllocUninit::<_>::new($n), move |u| {
            $crate::FillUninit::fill(u, $v)
        })
    };
    ($v:expr, $n:expr; $m:ty) => {
        $crate::DeviceOperation::and_then(
            $crate::DeviceSliceAllocUninit::<_, $m>::new($n),
            move |u| $crate::FillUninit::fill(u, $v),
        )
    };
}

/// Lazy alloc + `CL_MEM_COPY_HOST_PTR`. Works for **any marker**
/// (including Frozen / ReadOnly) — data baked in at creation, no
/// post-creation runtime write.
#[macro_export]
macro_rules! device_slice_from_slice {
    ($data:expr) => {
        $crate::DeviceSliceFromSlice::<_>::new($data)
    };
    ($data:expr; $m:ty) => {
        $crate::DeviceSliceFromSlice::<_, $m>::new($data)
    };
}

/// Lazy alloc + non-blocking host-to-device write — sugar for
/// `device_slice_alloc_uninit!(_, src.len()).and_then(|u| u.write(src))`.
/// `upload!(src)` / `upload!(src; M)`. Bound `M: HostUploadable`.
/// `src` must be `Vec<T>` / `Box<[T]>` / `Arc<[T]>` (anything with
/// a `.len()` method that converts via `Into<UploadSource<T>>`).
#[macro_export]
macro_rules! upload {
    ($src:expr) => {{
        let src = $src;
        let n = src.len();
        $crate::DeviceOperation::and_then($crate::DeviceSliceAllocUninit::<_>::new(n), move |u| {
            $crate::WriteUninit::write(u, src)
        })
    }};
    ($src:expr; $m:ty) => {{
        let src = $src;
        let n = src.len();
        $crate::DeviceOperation::and_then(
            $crate::DeviceSliceAllocUninit::<_, $m>::new(n),
            move |u| $crate::WriteUninit::write(u, src),
        )
    }};
}

/// Lazy non-blocking device-to-host read. `download!(buf)`. Marker
/// inferred from the input buffer; bound `M: HostReadable`.
#[macro_export]
macro_rules! download {
    ($buf:expr) => {
        $crate::Download::<_, _>::new($buf)
    };
}

/// SVM analog of `device_slice_alloc_zero!` — sugar over
/// `mapped_slice_alloc_uninit!(T, N).and_then(|u| u.fill(T::default()))`.
#[macro_export]
macro_rules! mapped_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::DeviceOperation::and_then($crate::MappedSliceAllocUninit::<$t>::new($n), |u| {
            $crate::FillUninit::fill(u, <$t as ::core::default::Default>::default())
        })
    };
    ($t:ty, $n:expr; $m:ty) => {
        $crate::DeviceOperation::and_then($crate::MappedSliceAllocUninit::<$t, $m>::new($n), |u| {
            $crate::FillUninit::fill(u, <$t as ::core::default::Default>::default())
        })
    };
}

/// SVM analog of `device_slice_alloc_uninit!`.
#[macro_export]
macro_rules! mapped_slice_alloc_uninit {
    ($t:ty, $n:expr) => {
        $crate::MappedSliceAllocUninit::<$t>::new($n)
    };
    ($t:ty, $n:expr; $m:ty) => {
        $crate::MappedSliceAllocUninit::<$t, $m>::new($n)
    };
}

/// SVM analog of `device_slice_filled!` — sugar over
/// `mapped_slice_alloc_uninit!(_, N).and_then(|u| u.fill(value))`.
#[macro_export]
macro_rules! mapped_slice_filled {
    ($v:expr, $n:expr) => {
        $crate::DeviceOperation::and_then($crate::MappedSliceAllocUninit::<_>::new($n), move |u| {
            $crate::FillUninit::fill(u, $v)
        })
    };
    ($v:expr, $n:expr; $m:ty) => {
        $crate::DeviceOperation::and_then(
            $crate::MappedSliceAllocUninit::<_, $m>::new($n),
            move |u| $crate::FillUninit::fill(u, $v),
        )
    };
}

/// SVM analog of `device_slice_from_slice!`.
#[macro_export]
macro_rules! mapped_slice_from_slice {
    ($data:expr) => {
        $crate::MappedSliceFromSlice::<_>::new($data)
    };
    ($data:expr; $m:ty) => {
        $crate::MappedSliceFromSlice::<_, $m>::new($data)
    };
}

/// SVM analog of `upload!` — sugar over
/// `mapped_slice_alloc_uninit!(_, src.len()).and_then(|u| u.write(src))`.
#[macro_export]
macro_rules! mapped_slice_upload {
    ($src:expr) => {{
        let src = $src;
        let n = src.len();
        $crate::DeviceOperation::and_then($crate::MappedSliceAllocUninit::<_>::new(n), move |u| {
            $crate::WriteUninit::write(u, src)
        })
    }};
    ($src:expr; $m:ty) => {{
        let src = $src;
        let n = src.len();
        $crate::DeviceOperation::and_then(
            $crate::MappedSliceAllocUninit::<_, $m>::new(n),
            move |u| $crate::WriteUninit::write(u, src),
        )
    }};
}

// Note: `usm_slice!` is defined further below to merge with the
// existing `vec!`-shape convenience arms (`usm_slice![v; N]` and
// `usm_slice![a, b, c]`).

/// Lazy [`USMSliceUninit<T, M>`](crate::USMSliceUninit) alloc.
/// No marker bound (USM is host memory).
#[macro_export]
macro_rules! usm_slice_alloc_uninit {
    ($t:ty, $n:expr) => {
        $crate::UsmSliceAllocUninit::<$t>::new($n)
    };
    ($t:ty, $n:expr; $m:ty) => {
        $crate::UsmSliceAllocUninit::<$t, $m>::new($n)
    };
}

/// USM zero-init alloc — sugar over
/// `usm_slice_alloc_uninit!(T, N).and_then(|u| u.fill(T::default()))`.
#[macro_export]
macro_rules! usm_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::DeviceOperation::and_then($crate::UsmSliceAllocUninit::<$t>::new($n), |u| {
            $crate::FillUninit::fill(u, <$t as ::core::default::Default>::default())
        })
    };
    ($t:ty, $n:expr; $m:ty) => {
        $crate::DeviceOperation::and_then($crate::UsmSliceAllocUninit::<$t, $m>::new($n), |u| {
            $crate::FillUninit::fill(u, <$t as ::core::default::Default>::default())
        })
    };
}

/// `vec!`-shaped sugar for producing a [`DeviceSlice<T>`](crate::DeviceSlice) op.
///
/// Two arms mirror [`vec!`](std::vec!):
///
/// - `device_slice![value; count]` — alloc + `clEnqueueFillBuffer`
///   on the chain's queue. No host allocation, no host→device
///   transfer; just the pattern repeated across the new buffer.
///   Expands to [`device_slice_filled(value, count)`](crate::device_slice_filled).
/// - `device_slice![a, b, c]` — upload a host literal. Allocates
///   a host `Vec<T>` and a fresh `cl_mem`, runs a non-blocking
///   `clEnqueueWriteBuffer`. Expands to [`upload(vec![a, b, c])`](crate::upload).
///
/// Choose intentionally: the two arms have radically different
/// bandwidth profiles even though they look almost identical. For
/// the explicit form prefer [`device_slice_alloc_zero!`](crate::device_slice_alloc_zero)
/// and [`DeviceSlice::fill`](crate::DeviceSlice::fill) directly
/// when the alloc + fill decomposition matters in the chain shape.
///
/// ```ignore
/// // Allocates one cl_mem, fills with 0u32 on-device.
/// let buf_op = device_slice![0u32; N];
///
/// // Allocates a Vec on the host, uploads it.
/// let buf_op = device_slice![1u32, 2, 3, 4];
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

/// `vec!`-shaped sugar for producing a [`MappedSlice<T>`](crate::MappedSlice) op — SVM
/// analog of [`device_slice!`](crate::device_slice!).
///
/// Two arms mirror [`vec!`](std::vec!):
///
/// - `mapped_slice![value; count]` — alloc + `clEnqueueSVMMemFill`.
///   Expands to [`mapped_slice_filled(value, count)`](crate::mapped_slice_filled).
/// - `mapped_slice![a, b, c]` — alloc + `clEnqueueSVMMemcpy` from a
///   host literal. Expands to [`mapped_slice_upload(vec![a, b, c])`](crate::mapped_slice_upload).
///
/// Both arms gate on SVM availability and surface
/// [`Error::SvmNotAvailable`](crate::Error::SvmNotAvailable) at
/// execute time on devices without SVM.
///
/// ```ignore
/// // SVM alloc + on-device fill with 0u32.
/// let buf_op = mapped_slice![0u32; N];
///
/// // SVM alloc + SVM memcpy from a host literal.
/// let buf_op = mapped_slice![1u32, 2, 3, 4];
/// ```
#[macro_export]
macro_rules! mapped_slice {
    [$value:expr; $count:expr] => {
        $crate::mapped_slice_filled!($value, $count)
    };
    [$($v:expr),* $(,)?] => {
        $crate::mapped_slice_upload!(::std::vec![$($v),*])
    };
}

/// `vec!`-shaped sugar for producing a [`USMSlice<T>`](crate::USMSlice)
/// op — symmetric with [`device_slice!`](crate::device_slice!) /
/// [`mapped_slice!`](crate::mapped_slice!).
///
/// Both arms expand to [`usm_slice`](crate::usm_slice!) over a host
/// `Vec<T>` — USMSlice always wraps an existing host allocation, so
/// there's no cheap on-device fill path to distinguish from the
/// literal arm. The macro exists for syntactic symmetry across the
/// tier family, not for cost-path sugar.
///
/// ```ignore
/// let buf_op = usm_slice![0u32; N];
/// let buf_op = usm_slice![1u32, 2, 3, 4];
/// ```
#[macro_export]
macro_rules! usm_slice {
    // `usm_slice![v; N]` — alloc-and-fill via host vec![v; N].
    [$value:expr; $count:expr] => {
        $crate::UsmSliceOp::<_>::new(::std::vec![$value; $count])
    };
    // `usm_slice!(host_vec)` — wrap an existing Vec, default marker.
    // Put this BEFORE the bracket-list arm so single-expr paren
    // calls don't get wrapped in another Vec.
    ($vec:expr) => {
        $crate::UsmSliceOp::<_>::new($vec)
    };
    // `usm_slice![a, b, c]` — alloc-and-fill via host vec literal.
    [$($v:expr),* $(,)?] => {
        $crate::UsmSliceOp::<_>::new(::std::vec![$($v),*])
    };
}
