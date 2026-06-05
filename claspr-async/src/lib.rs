//! claspr-async — the Tier 2 combinator layer on top of claspr.
//!
//! Where [claspr]'s [`LaunchOp`] surfaces one explicit queue per call
//! (Tier 1: `.wait()` / `.submit()` / `.await`), claspr-async lets you
//! compose lazy [`DeviceOperation`]s into a single chain that runs
//! end-to-end, with the per-device default out-of-order queue picked
//! automatically and event dependencies threaded through behind the
//! scenes.
//!
//! ## At a glance
//!
//! ```ignore
//! use claspr::Context;
//! use claspr_async::{DeviceOperation, download, upload};
//!
//! let ctx = Context::any()?;
//! let kernels = gpu::kernels(&ctx)?;
//!
//! // Lift a Vec to device, run a kernel, then download.
//! let result: Vec<u32> = upload(input_vec)
//!     .and_then(|buf| kernels.foo([N], buf, scalar))
//!     .and_then(|buf| download!(buf))
//!     .sync(&ctx)?;
//! ```
//!
//! ## Crate layout (mirrors `IMPLEMENTATION-PLAN.md`)
//!
//! - [`op`] — [`DeviceOperation`] trait + the core combinators
//!   ([`AndThen`], [`AndThenWithContext`], [`Arced`], [`Value`]).
//! - [`exec_ctx`] — [`ExecutionContext`] (passed to each op's
//!   `execute`; implements [`claspr::Launcher`] so existing Tier 1
//!   ops compose into the chain).
//!
//! Later phases add: `bundle` / `fan_out` / `arc` / `future` /
//! `and_then_host` / `host_view` / `profile`.
//!
//! [claspr]: https://docs.rs/claspr
//! [`LaunchOp`]: claspr::LaunchOp

pub mod alloc;
pub mod and_then_host;
pub mod arc;
pub mod buffer_ops;
pub mod bundle;
pub mod copy;
pub mod dyn_op;
pub mod exec_ctx;
pub mod fan_out;
pub mod future;
pub mod host_view;
pub mod image_transfer;
pub mod mappable;
pub mod on_device;
pub mod op;
pub mod profile;
pub mod transfer;
pub mod transfer_to_device;
pub mod usm;

pub use alloc::{
    DeviceSliceAllocUninit, DeviceSliceAllocZero, DeviceSliceFilled, DeviceSliceFromSlice,
    MappedSliceAllocUninit, MappedSliceAllocZero, MappedSliceFilled, MappedSliceFromSlice,
    MappedSliceUpload,
};
pub use and_then_host::{AndThenHost, AndThenHostWithContext, DeviceOperationHostExt};
pub use arc::ArcSplit;
pub use buffer_ops::{
    DeviceSliceFillOp, DeviceSliceWriteOp, MappedSliceFillOp, device_slice_fill,
    device_slice_write, mapped_slice_fill,
};
pub use bundle::{
    Bundle2, Bundle3, Bundle4, Bundle5, Bundle6, Bundle7, Bundle8, Bundle9, Bundle10, Bundle11,
    Bundle12, Bundle13, Bundle14, Bundle15, Bundle16,
};
pub use copy::{CopyTo, CopyToOp};
pub use dyn_op::DynOp;
pub use exec_ctx::ExecutionContext;
pub use fan_out::{FanOut, FanOutExt, fan_out};
pub use future::ChainFuture;
pub use host_view::{
    AcquireDeviceSliceOp, AcquireMappedSliceOp, DeviceSliceHostView, HostAccessibleExt,
    HostReadableExt, HostWritableExt, MapAccess, MapReadOnly, MapReadWrite, MappedSliceHostView,
    ReleaseDeviceSliceOp, ReleaseMappedSliceOp,
};
pub use image_transfer::{ImageDownload, ImageUpload, image_download, image_upload};
pub use mappable::{DeviceSliceMapHandle, Mappable};
pub use on_device::OnDevice;
pub use op::{
    AndThen, AndThenWithContext, Arced, Dep, Deps, DeviceOperation, Value, deps_as_events, value,
    wrap_event,
};
pub use profile::{DeviceOperationProfileExt, Profiled};
pub use transfer::{Download, Upload, UploadSource};
pub use transfer_to_device::{TransferToDevice, transfer_to_device};
pub use usm::{UsmSliceAllocZero, UsmSliceOp};

// ── Tier 2 entry macros ────────────────────────────────────────────
//
// All Tier 2 buffer constructors are macros, not free fns. Two arms
// per macro:
//   - default arm:   foo!(args)                — uses struct's M = ReadWrite default
//   - marker arm:    foo!(args; M)             — turbofishes M explicitly
// Macros expand to `Foo::<T>::new(...)` / `Foo::<T, M>::new(...)`.
// The struct method form is the canonical constructor; macros are
// sugar to skip the type-position turbofish noise at chain entry.

/// Lazy zero-init `DeviceSlice<T, M>` alloc. `device_slice_alloc_zero!(T, N)`
/// for default marker, `device_slice_alloc_zero!(T, N; M)` for explicit.
#[macro_export]
macro_rules! device_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::DeviceSliceAllocZero::<$t>::new($n)
    };
    ($t:ty, $n:expr; $m:ty) => {
        $crate::DeviceSliceAllocZero::<$t, $m>::new($n)
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

/// Lazy alloc + fill: `device_slice_filled!(value, N)` /
/// `device_slice_filled!(value, N; M)`. Dispatches Runtime vs
/// DeviceKernel fill via the marker's `FillStrategy`.
#[macro_export]
macro_rules! device_slice_filled {
    ($v:expr, $n:expr) => {
        $crate::DeviceSliceFilled::<_>::new($v, $n)
    };
    ($v:expr, $n:expr; $m:ty) => {
        $crate::DeviceSliceFilled::<_, $m>::new($v, $n)
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

/// Lazy alloc + non-blocking host-to-device write.
/// `upload!(src)` / `upload!(src; M)`. Bound `M: HostUploadable`.
#[macro_export]
macro_rules! upload {
    ($src:expr) => {
        $crate::Upload::<_>::new($src)
    };
    ($src:expr; $m:ty) => {
        $crate::Upload::<_, $m>::new($src)
    };
}

/// Lazy non-blocking device-to-host read. `download!(buf)`. Marker
/// inferred from the input buffer; bound `M: HostReadable`.
#[macro_export]
macro_rules! download {
    ($buf:expr) => {
        $crate::Download::<_, _>::new($buf)
    };
}

/// SVM analog of `device_slice_alloc_zero!`.
#[macro_export]
macro_rules! mapped_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::MappedSliceAllocZero::<$t>::new($n)
    };
    ($t:ty, $n:expr; $m:ty) => {
        $crate::MappedSliceAllocZero::<$t, $m>::new($n)
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

/// SVM analog of `device_slice_filled!`.
#[macro_export]
macro_rules! mapped_slice_filled {
    ($v:expr, $n:expr) => {
        $crate::MappedSliceFilled::<_>::new($v, $n)
    };
    ($v:expr, $n:expr; $m:ty) => {
        $crate::MappedSliceFilled::<_, $m>::new($v, $n)
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

/// SVM analog of `upload!`.
#[macro_export]
macro_rules! mapped_slice_upload {
    ($src:expr) => {
        $crate::MappedSliceUpload::<_>::new($src)
    };
    ($src:expr; $m:ty) => {
        $crate::MappedSliceUpload::<_, $m>::new($src)
    };
}

// Note: `usm_slice!` is defined further below to merge with the
// existing `vec!`-shape convenience arms (`usm_slice![v; N]` and
// `usm_slice![a, b, c]`).

/// USM zero-init alloc.
#[macro_export]
macro_rules! usm_slice_alloc_zero {
    ($t:ty, $n:expr) => {
        $crate::UsmSliceAllocZero::<$t>::new($n)
    };
    ($t:ty, $n:expr; $m:ty) => {
        $crate::UsmSliceAllocZero::<$t, $m>::new($n)
    };
}

/// `vec!`-shaped sugar for producing a [`DeviceSlice<T>`](claspr::DeviceSlice) op.
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
/// and [`DeviceSlice::fill`](claspr::DeviceSlice::fill) directly
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

/// `vec!`-shaped sugar for producing a [`MappedSlice<T>`](claspr::MappedSlice) op — SVM
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
/// [`Error::SvmNotAvailable`](claspr::Error::SvmNotAvailable) at
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

/// `vec!`-shaped sugar for producing a [`USMSlice<T>`](claspr::USMSlice)
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
