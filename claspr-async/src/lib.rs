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

pub use alloc::{
    DeviceSliceAlloc, HostBufferAlloc, SharedBufferAlloc, device_slice_alloc, host_buffer_alloc,
    shared_buffer_alloc,
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
    AcquireDeviceSliceOp, AcquireHostBufferOp, AcquireSharedBufferOp, DeviceSliceHostView,
    HostAccessibleExt, HostBufferHostView, ReleaseDeviceSliceOp, ReleaseHostBufferOp,
    ReleaseSharedBufferOp, SharedBufferHostView,
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
