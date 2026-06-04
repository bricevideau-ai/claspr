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
//!         #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
//!         #[spirv(cross_workgroup)] data: &mut [u32],
//!     ) { /* ... */ }
//! }
//!
//! fn main() -> claspr::Result<()> {
//!     let ctx = Context::any()?;
//!     let kernels = gpu::kernels(&ctx)?;
//!     let mut data: Vec<u32> = (1..=1024).collect();
//!     let mut buf = DeviceSlice::<u32>::alloc(&ctx, data.len())?;
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
#[cfg(feature = "async-events")]
pub mod future;
pub mod image;
pub mod kernel_op;
pub mod launch;
pub mod mapped;
pub mod op;
pub mod ppm;
pub mod queue;
pub mod usm;
#[doc(hidden)]
pub mod util;

// ── Public surface ────────────────────────────────────────────────────

pub use access::{
    DeviceScratch, Frozen, HostAccess, HostReadOnly, HostReadable, HostWritable, KernelAccess,
    KernelReadable, KernelWritable, MemMode, ReadOnly, ReadWrite, WriteOnly,
};
pub use buffer::{Buffer, CopyOp, DeviceSlice, FillOp, MigrateOp, ReadOp, WriteOp};
pub use context::{Context, SvmLevel};
pub use device::{Device, DeviceType, Platform};
pub use error::{Error, Result};
pub use image::{
    Image1D, Image1DArray, Image1DBuffer, Image1DBufferView, Image2D, Image2DArray, Image2DRgba8,
    Image3D, ImageAccess, ImageCopyOp, ImageFillOp, ImageReadAlloc, ImageReadBytesAlloc,
    ImageReadOp, ImageWriteOp, KernelImage1DArrayReadArg, KernelImage1DArrayReadWriteArg,
    KernelImage1DArrayWriteArg, KernelImage1DReadArg, KernelImage1DReadWriteArg,
    KernelImage1DWriteArg, KernelImage2DArrayReadArg, KernelImage2DArrayReadWriteArg,
    KernelImage2DArrayWriteArg, KernelImage2DReadArg, KernelImage2DReadWriteArg,
    KernelImage2DWriteArg, KernelImage3DReadArg, KernelImage3DReadWriteArg, KernelImage3DWriteArg,
    KernelImageBufferReadArg, KernelImageBufferReadWriteArg, KernelImageBufferWriteArg, format,
};
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
pub use mapped::{MappedReadGuard, MappedSlice, MappedWriteGuard, SvmCopyOp, SvmFillOp};
pub use usm::USMSlice;

// Stage-3 proc-macro frontend.
pub use claspr_macros::{device, kernel, kernels};

// Re-exports from opencl3 — the types users actually touch through
// claspr's API.
pub use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
pub use opencl3::event::Event;
pub use opencl3::kernel::Kernel;
pub use opencl3::program::Program;
pub use opencl3::types::cl_event;
