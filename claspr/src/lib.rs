//! claspr — single-source OpenCL with rust-gpu.
//!
//! This crate is the **runtime helper layer**: typed `Context`,
//! `DeviceSlice<T>`, kernel argument plumbing, image helpers, and the
//! `#[claspr::kernel]` / `#[claspr::device]` proc-macro re-exports.
//! The matching build-script library `claspr-build` handles
//! compiling the rust-gpu kernel sub-crates into SPIR-V at build time.
//!
//! ## Quickstart
//!
//! Single-source mode (the recommended path — see the workspace
//! `examples/`):
//!
//! ```ignore
//! use claspr::{Context, DeviceSlice};
//!
//! #[claspr::device]
//! mod gpu {
//!     #[claspr::kernel]
//!     pub fn collatz_kernel(
//!         #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
//!         #[spirv(cross_workgroup)] data: &mut [u32],
//!     ) { /* ... */ }
//! }
//!
//! fn main() -> claspr::Result<()> {
//!     let ctx = Context::any()?;
//!     let kernels = gpu::kernels(&ctx)?;
//!     let mut data: Vec<u32> = (1..=1024).collect();
//!     let mut buf = DeviceSlice::<u32>::alloc_zero(&ctx, data.len())?;
//!     buf.write(&data).wait(&ctx)?;
//!     let buf = kernels.collatz_kernel([data.len()], buf).wait(&ctx)?;
//!     buf.read(&mut data).wait(&ctx)?;
//!     Ok(())
//! }
//! ```
//!
//! Pair with a `build.rs` that calls `claspr_build::compile_from_host(...).write()`.
//!
//! ## Crate structure
//!
//! - [`Context`] — device, OpenCL context, and command queue
//!   (the default in-order / out-of-order queues are the launcher
//!   surface for Tier 1 ops).
//! - [`DeviceSlice<T>`] — typed device buffer + length, mirrors
//!   rust-gpu's slice decomposition.
//! - [`KernelArg`] / [`KernelArgs`] / [`ScalarArg`] — typed launch
//!   surface. User structs opt in via the [`scalar_arg!`] macro.
//! - [`Image2DRgba8`] + [`write_ppm_rgba8`] — render-to-image helpers.
//! - [`device`](macro@device) / [`kernel`] — proc-macros from `claspr-macros`.
//!
//! ## Error type
//!
//! Every fallible function returns [`Result<T>`] from [`mod@error`] —
//! a typed [`Error`] enum so callers can `match` on the failure mode
//! (OpenCL status, build failure, missing capability) instead of
//! string-sniffing a boxed trait object.
//!
//! [claspr-build]: https://docs.rs/claspr-build
//! [`Image2DRgba8`]: crate::image::Image2DRgba8

pub mod access;
pub mod buffer;
pub mod context;
pub mod device;
pub mod error;
pub(crate) mod fill_kernel;
#[cfg(feature = "async-events")]
pub mod future;
pub mod image;
pub mod kernel_op;
pub mod launch;
pub mod map_primitive;
pub mod mapped;
pub mod op;
pub mod ppm;
pub mod queue;
pub mod usm;
#[doc(hidden)]
pub mod util;

// ── Tier 2 combinator layer (folded in from the former claspr-async) ──
pub mod alloc;
pub mod and_then_host;
pub mod arc;
pub mod buffer_ops;
pub mod bundle;
#[cfg(feature = "async-events")]
pub mod chain_future;
pub mod copy;
pub mod device_op;
pub mod dyn_op;
pub mod eager;
pub mod exec_ctx;
pub mod fan_out;
pub mod host_view;
pub mod image_transfer;
pub mod mappable;
pub mod on_device;
pub mod profile;
pub mod transfer;
pub mod transfer_to_device;
mod tier2_macros;
pub mod uninit_ext;
pub mod usm_op;

// ── Public surface ────────────────────────────────────────────────────

pub use access::{
    DeviceScratch, FillStrategy, Fillable, Frozen, HostAccess, HostReadOnly, HostReadable,
    HostUploadable, HostWritable, KernelAccess, KernelReadable, KernelWritable, MemMode, ReadOnly,
    ReadWrite, RuntimeFillable, WriteOnly,
};
pub use buffer::{
    Buffer, CopyOp, DeviceMapMutOp, DeviceMapOp, DeviceMapReadPending, DeviceMapWritePending,
    DeviceMappedReadGuard, DeviceMappedWriteGuard, DeviceSlice, DeviceSliceUninit, FillOp,
    MigrateOp, ReadOp, WriteOp,
};
pub use context::{Context, SvmLevel};
pub use device::{Device, DeviceType, Platform};
pub use error::{Error, Result};
pub use image::{
    Image1D, Image1DArray, Image1DBuffer, Image1DBufferView, Image2D, Image2DArray, Image2DRgba8,
    Image3D, ImageAccess, ImageCopyOp, ImageFillOp, ImageHostTransfer, ImageReadAlloc,
    ImageReadBytesAlloc, ImageReadOp, ImageWriteOp, KernelImage1DArrayReadArg,
    KernelImage1DArrayReadWriteArg, KernelImage1DArrayWriteArg, KernelImage1DReadArg,
    KernelImage1DReadWriteArg, KernelImage1DWriteArg, KernelImage2DArrayReadArg,
    KernelImage2DArrayReadWriteArg, KernelImage2DArrayWriteArg, KernelImage2DReadArg,
    KernelImage2DReadWriteArg, KernelImage2DWriteArg, KernelImage3DReadArg,
    KernelImage3DReadWriteArg, KernelImage3DWriteArg, KernelImageBufferReadArg,
    KernelImageBufferReadWriteArg, KernelImageBufferWriteArg, format,
};
#[doc(hidden)]
pub use kernel_op::__seal;
pub use kernel_op::KernelOp;
pub use launch::{
    IntoLaunchSpec, KernelArg, KernelArgs, KernelSliceArg, KernelSliceReadArg,
    KernelSliceReadWriteArg, LaunchSpec, LocalBuffer, ScalarArg, profiling_duration,
};
pub use op::{
    LaunchOp, ProfileCb, ProfilingInfo, assert_same_context, complete_user_event,
    create_user_event, register_drop_callback, register_profiling_callback,
};
pub use ppm::write_ppm_rgba8;
pub use queue::{InOrder, Launcher, OutOfOrder, Queue, QueueOrder};

#[cfg(feature = "async-events")]
pub use future::{EventFuture, EventFutureExt, LaunchFuture};
pub use mapped::{
    MapMutOp, MapOp, MappedReadGuard, MappedReadPending, MappedSlice, MappedSliceUninit,
    MappedWriteGuard, MappedWritePending, SvmCopyOp, SvmFillOp, SvmWriteOp,
};
pub use usm::{USMSlice, USMSliceUninit};

// ── Tier 2 combinator re-exports (folded in from claspr-async) ──
pub use alloc::{
    DeviceSliceAllocUninit, DeviceSliceFromSlice, MappedSliceAllocUninit, MappedSliceFromSlice,
};
pub use and_then_host::{AndThenHost, AndThenHostWithContext, DeviceOperationHostExt};
pub use arc::ArcSplit;
pub use buffer_ops::{
    DeviceSliceFillOp, DeviceSliceWriteOp, MappedSliceFillOp, device_slice_fill, device_slice_write,
    mapped_slice_fill,
};
pub use bundle::{
    Bundle2, Bundle3, Bundle4, Bundle5, Bundle6, Bundle7, Bundle8, Bundle9, Bundle10, Bundle11,
    Bundle12, Bundle13, Bundle14, Bundle15, Bundle16,
};
#[cfg(feature = "async-events")]
pub use chain_future::ChainFuture;
pub use copy::{CopyTo, CopyToOp};
pub use device_op::{
    AndThen, AndThenWithContext, Arced, Dep, Deps, DeviceOperation, Value, deps_as_events, value,
    wrap_event,
};
pub use dyn_op::DynOp;
pub use eager::{AllocZero, EagerOp, EagerOpExt, Fill, Input, Pipe};
pub use exec_ctx::ExecutionContext;
pub use fan_out::{FanOut, FanOutExt, fan_out};
pub use host_view::{
    AcquireDeviceSliceOp, AcquireMappedSliceOp, DeviceSliceHostView, HostAccessibleExt,
    HostReadableExt, HostWritableExt, MapAccess, MapReadOnly, MapReadWrite, MappedSliceHostView,
    ReleaseDeviceSliceOp, ReleaseMappedSliceOp,
};
pub use image_transfer::{ImageDownload, ImageUpload, image_download, image_upload};
pub use mappable::{DeviceSliceMapHandle, Mappable};
pub use on_device::OnDevice;
pub use profile::{DeviceOperationProfileExt, Profiled};
pub use transfer::{Download, UploadSource};
pub use transfer_to_device::{TransferToDevice, transfer_to_device};
pub use uninit_ext::{FillFromUninitOp, FillUninit, WriteFromUninitOp, WriteUninit};
pub use usm_op::{UsmSliceAllocUninit, UsmSliceOp};

// Stage-3 proc-macro frontend.
pub use claspr_macros::{device, kernel, kernels};

// Re-exports from opencl3 — the types users actually touch through
// claspr's API.
pub use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
pub use opencl3::event::Event;
pub use opencl3::kernel::Kernel;
pub use opencl3::program::Program;
pub use opencl3::types::cl_event;
