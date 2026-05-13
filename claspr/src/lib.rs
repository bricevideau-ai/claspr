//! claspr — single-source OpenCL with rust-gpu.
//!
//! This crate is the **runtime helper layer** (stage 1 of three):
//! it wraps `opencl3` and `spirv-builder` so a host program can compile
//! and launch a rust-gpu kernel crate without the per-project boilerplate
//! that otherwise piles up around every sample. Stage 2 (build-time
//! codegen) and stage 3 (`#[claspr::kernel]` proc-macro single-source)
//! build on this layer.
//!
//! ## Quickstart
//!
//! ```ignore
//! use claspr::{Context, compile, profiling_duration};
//!
//! // 1. Compile the kernel crate to SPIR-V.
//! let module = compile("kernels/collatz").opencl12().build()?;
//!
//! // 2. Pick an OpenCL device and create a context + queue.
//! let ctx = Context::new()?;
//!
//! // 3. Load the kernel.
//! let kernel = ctx.kernel_from_spv(&module.spv_bytes, "collatz_kernel")?;
//!
//! // 4. Upload data, launch with a typed argument tuple, read back.
//! let mut data: Vec<u32> = (1..=1024).collect();
//! let buf = ctx.upload(&data)?;
//! let event = ctx.launch(&kernel, [data.len()], (&buf,))?;
//! ctx.download(&buf, &mut data)?;
//!
//! println!("kernel ran in {:?}", profiling_duration(&event));
//! ```
//!
//! ## Crate structure
//!
//! - [`Context`] — device, OpenCL context, command queue, and the
//!   [`launch`](Context::launch) entry point.
//! - [`DeviceSlice<T>`] — typed device buffer + length, mirrors
//!   rust-gpu's slice decomposition.
//! - [`KernelArg`] / [`KernelArgs`] / [`ScalarArg`] — typed launch
//!   surface. User structs opt in via the [`scalar_arg!`] macro.
//! - [`compile()`] / [`CompileBuilder`] — thin builder around
//!   [`spirv_builder::SpirvBuilder`] with named presets
//!   (`opencl12`, `opencl20_groups`, `image`, `with_f64`).
//! - [`Image2DRgba8`] + [`write_ppm_rgba8`] — render-to-image helpers.
//!
//! ## Error type
//!
//! Every fallible function returns [`Result<T>`], which is just
//! `std::result::Result<T, Box<dyn Error + Send + Sync + 'static>>`.
//! This will likely become a `thiserror` enum once we have real
//! patterns of error handling to enumerate.
//!
//! [`Image2DRgba8`]: crate::image::Image2DRgba8

pub mod buffer;
pub mod compile;
pub mod context;
pub mod image;
pub mod launch;
pub mod ppm;

// ── Public surface ────────────────────────────────────────────────────

pub use buffer::DeviceSlice;
pub use compile::{CompileBuilder, CompiledModule, compile};
pub use context::Context;
pub use image::Image2DRgba8;
pub use launch::{
    IntoLaunchSpec, KernelArg, KernelArgs, LaunchSpec, LocalBuffer, ScalarArg, profiling_duration,
};
pub use ppm::write_ppm_rgba8;

// Re-exports from spirv-builder so kernel-crate users don't need to
// add it as a separate dep.
pub use spirv_builder::{Capability, ShaderPanicStrategy};

// Stage-3 proc-macro frontend.
pub use claspr_macros::{device, kernel};

// Re-exports from opencl3 — the types users actually touch through
// claspr's API.
pub use opencl3::event::Event;
pub use opencl3::kernel::Kernel;
pub use opencl3::program::Program;

/// Boxed-error result alias. All claspr APIs return this.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;
