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
//!     .and_then(download)
//!     .sync(&ctx)?;
//! ```
//!
//! ## Crate layout (mirrors [`IMPLEMENTATION-PLAN.md`])
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
pub mod bundle;
pub mod dyn_op;
pub mod exec_ctx;
pub mod fan_out;
pub mod future;
pub mod host_view;
pub mod mappable;
pub mod on_device;
pub mod op;
pub mod profile;
pub mod transfer;
pub mod transfer_to_device;
pub mod usm;

pub use alloc::{
    DeviceSliceAlloc, DeviceSliceFilled, SharedBufferAlloc, SharedBufferFilled, SharedBufferUpload,
    device_slice_alloc, device_slice_filled, shared_buffer_alloc, shared_buffer_filled,
    shared_buffer_upload,
};
pub use and_then_host::{AndThenHost, AndThenHostWithContext, DeviceOperationHostExt};
pub use arc::ArcSplit;
pub use bundle::{
    Bundle2, Bundle3, Bundle4, Bundle5, Bundle6, Bundle7, Bundle8, Bundle9, Bundle10, Bundle11,
    Bundle12, Bundle13, Bundle14, Bundle15, Bundle16,
};
pub use dyn_op::DynOp;
pub use exec_ctx::ExecutionContext;
pub use fan_out::{FanOut, FanOutExt, fan_out};
pub use future::ChainFuture;
pub use host_view::{
    AcquireDeviceSliceOp, AcquireSharedBufferOp, DeviceSliceHostView, HostAccessibleExt,
    ReleaseDeviceSliceOp, ReleaseSharedBufferOp, SharedBufferHostView,
};
pub use mappable::{DeviceSliceMapHandle, Mappable};
pub use on_device::OnDevice;
pub use op::{
    AndThen, AndThenWithContext, Arced, Dep, Deps, DeviceOperation, Value, deps_as_events, value,
    wrap_event,
};
pub use profile::{DeviceOperationProfileExt, Profiled};
pub use transfer::{Download, Upload, UploadSource, download, upload};
pub use transfer_to_device::{TransferToDevice, transfer_to_device};
pub use usm::{UsmSliceOp, usm_slice};

/// `vec!`-shaped sugar for producing a [`DeviceSlice<T>`] op.
///
/// Two arms mirror [`vec!`](std::vec):
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
/// the explicit form prefer [`device_slice_alloc`](crate::device_slice_alloc)
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
        $crate::device_slice_filled($value, $count)
    };
    [$($v:expr),* $(,)?] => {
        $crate::upload(::std::vec![$($v),*])
    };
}

/// `vec!`-shaped sugar for producing a [`SharedBuffer<T>`] op — SVM
/// analog of [`device_slice!`](crate::device_slice!).
///
/// Two arms mirror [`vec!`](std::vec):
///
/// - `shared_buffer![value; count]` — alloc + `clEnqueueSVMMemFill`.
///   Expands to [`shared_buffer_filled(value, count)`](crate::shared_buffer_filled).
/// - `shared_buffer![a, b, c]` — alloc + `clEnqueueSVMMemcpy` from a
///   host literal. Expands to [`shared_buffer_upload(vec![a, b, c])`](crate::shared_buffer_upload).
///
/// Both arms gate on SVM availability and surface
/// [`Error::SvmNotAvailable`](claspr::Error::SvmNotAvailable) at
/// execute time on devices without SVM.
///
/// ```ignore
/// // SVM alloc + on-device fill with 0u32.
/// let buf_op = shared_buffer![0u32; N];
///
/// // SVM alloc + SVM memcpy from a host literal.
/// let buf_op = shared_buffer![1u32, 2, 3, 4];
/// ```
#[macro_export]
macro_rules! shared_buffer {
    [$value:expr; $count:expr] => {
        $crate::shared_buffer_filled($value, $count)
    };
    [$($v:expr),* $(,)?] => {
        $crate::shared_buffer_upload(::std::vec![$($v),*])
    };
}

// `host_buffer!` macro removed 2026-05-29 when HostBuffer was
// deleted (see commit log). Use `device_slice!` or `shared_buffer!`
// instead; for fine-grain-system SVM use `usm_slice(vec)`.
