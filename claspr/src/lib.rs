//! claspr — single-source OpenCL with rust-gpu.
//!
//! This crate is the **runtime helper layer**: typed `Context`,
//! `DeviceSlice<T>`, kernel argument plumbing, image helpers, and the
//! `#[claspr::kernel]` / `#[claspr::device]` proc-macro re-exports.
//! The matching build-script library [`claspr-build`] handles
//! compiling the rust-gpu kernel sub-crates into SPIR-V at build time.
//!
//! ## Quickstart
//!
//! Single-source mode (the recommended path — see the workspace
//! `examples/`):
//!
//! ```ignore
//! use claspr::Context;
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
//!     let ctx = Context::new()?;
//!     let kernels = gpu::kernels(&ctx)?;
//!     let mut data: Vec<u32> = (1..=1024).collect();
//!     let buf = ctx.upload(&data)?;
//!     kernels.collatz_kernel(&ctx, [data.len()], &buf).wait()?;
//!     buf.download(&ctx, &mut data).wait()?;
//!     Ok(())
//! }
//! ```
//!
//! Pair with a `build.rs` that calls `claspr_build::compile_from_host(...).write()`.
//!
//! ## Crate structure
//!
//! - [`Context`] — device, OpenCL context, command queue, and the
//!   [`launch`](Context::launch) entry point.
//! - [`DeviceSlice<T>`] — typed device buffer + length, mirrors
//!   rust-gpu's slice decomposition.
//! - [`KernelArg`] / [`KernelArgs`] / [`ScalarArg`] — typed launch
//!   surface. User structs opt in via the [`scalar_arg!`] macro.
//! - [`Image2DRgba8`] + [`write_ppm_rgba8`] — render-to-image helpers.
//! - [`device`] / [`kernel`] — proc-macros from `claspr-macros`.
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

pub mod buffer;
pub mod context;
pub mod device;
pub mod error;
#[cfg(feature = "async-events")]
pub mod future;
pub mod image;
pub mod launch;
pub mod op;
pub mod ppm;
pub mod queue;
pub mod svm;
#[doc(hidden)]
pub mod util;

// ── Public surface ────────────────────────────────────────────────────

pub use buffer::{Buffer, CopyOp, DeviceSlice, FillOp, HostBuffer, MigrateOp, ReadOp, WriteOp};
pub use context::{Context, SvmLevel};
pub use device::{Device, DeviceType, Platform};
pub use error::{Error, Result};
pub use image::{Image2D, Image2DRgba8, ImageAccess, ReadOnly, ReadWrite, WriteOnly, format};
pub use launch::{
    IntoLaunchSpec, KernelArg, KernelArgs, KernelSliceArg, LaunchSpec, LocalBuffer, ScalarArg,
    profiling_duration,
};
pub use op::{
    LaunchOp, ProfileCb, ProfilingInfo, assert_same_context, complete_user_event,
    create_user_event, register_drop_callback, register_profiling_callback,
};
pub use ppm::write_ppm_rgba8;
pub use queue::{InOrder, Launcher, OutOfOrder, Queue, QueueOrder};

#[cfg(feature = "async-events")]
pub use future::{CopyFuture, EventFuture, EventFutureExt, LaunchFuture, ReadFuture, WriteFuture};
pub use svm::{SharedBuffer, SharedReadGuard, SharedWriteGuard, SvmCopyOp, SvmFillOp};

// Stage-3 proc-macro frontend.
pub use claspr_macros::{device, kernel};

// Re-exports from opencl3 — the types users actually touch through
// claspr's API.
pub use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
pub use opencl3::event::Event;
pub use opencl3::kernel::Kernel;
pub use opencl3::program::Program;
pub use opencl3::types::cl_event;
