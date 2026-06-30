// `associated_type_defaults`: `DeviceOp::Handle` defaults to `Pipe<Output>`
// (the common case); multi-output combinators override it. See `eager`.
#![feature(associated_type_defaults)]
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
//!     let buf = DeviceSlice::<u32>::alloc_zero(&ctx, data.len())?;
//!     // Verbs return eager `DeviceOp` builders; `.wait()` is the terminal.
//!     // A concrete-head op's `wait()` takes no argument, and upload sources
//!     // are owned (`Vec` / `Box<[T]>` / `Arc<[T]>`), not borrowed slices.
//!     let buf = buf.write(data.clone()).wait()?;
//!     let buf = kernels.collatz_kernel([data.len()], buf).wait()?;
//!     buf.read(&mut data).wait()?;
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
pub mod launch;
pub mod map_primitive;
pub mod mapped;
pub mod op;
pub mod ppm;
pub mod queue;
pub mod usm;
#[doc(hidden)]
pub mod util;

// ── Tier 2 device-graph layer (the eager struct-graph core + its
// supporting primitives: the polymorphic copy verb, the host-view
// map/unmap layer, and the shared upload source) ──
pub mod copy;
pub mod eager;
pub mod exec_ctx;
pub mod host_view;
pub mod mappable;
pub mod record;
mod tier2_macros;
pub mod transfer;

// ── Public surface ────────────────────────────────────────────────────

pub use access::{
    DeviceScratch, FillStrategy, Fillable, Frozen, HostAccess, HostReadOnly, HostReadable,
    HostUploadable, HostWritable, KernelAccess, KernelReadable, KernelWritable, MemMode, ReadOnly,
    ReadWrite, RuntimeFillable, WriteOnly,
};
pub use buffer::{
    Buffer, DeviceMapMutOp, DeviceMapOp, DeviceMapReadPending, DeviceMapWritePending,
    DeviceMappedReadGuard, DeviceMappedWriteGuard, DeviceSlice, DeviceSliceUninit,
};
pub use context::{Context, SvmLevel};
pub use device::{Device, DeviceType, Platform};
pub use error::{Error, Result};
pub use image::{
    Image1D, Image1DArray, Image1DBuffer, Image1DBufferView, Image2D, Image2DArray, Image2DRgba8,
    Image3D, ImageAccess, ImageCopy, ImageEnqueue, ImageFill, ImageHostTransfer, ImageRead,
    ImageReadAlloc, ImageReadBytesAlloc, ImageWrite, KernelImage1DArrayReadArg,
    KernelImage1DArrayReadWriteArg, KernelImage1DArrayWriteArg, KernelImage1DReadArg,
    KernelImage1DReadWriteArg, KernelImage1DWriteArg, KernelImage2DArrayReadArg,
    KernelImage2DArrayReadWriteArg, KernelImage2DArrayWriteArg, KernelImage2DReadArg,
    KernelImage2DReadWriteArg, KernelImage2DWriteArg, KernelImage3DReadArg,
    KernelImage3DReadWriteArg, KernelImage3DWriteArg, KernelImageBufferReadArg,
    KernelImageBufferReadWriteArg, KernelImageBufferWriteArg, format,
};
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
    MappedWriteGuard, MappedWritePending,
};
pub use usm::{USMSlice, USMSliceUninit};

// ── Tier 2 device-graph re-exports ──
//
// The eager struct-graph IS the Tier 2 API. Its items live in `mod eager`
// (reachable as `claspr::eager::…`) and are re-exported here at the crate root
// with their plain names — there is no longer a separate closure layer to
// disambiguate against, so the former `eager_*` aliases are gone.
pub use copy::CopyTo;
pub use exec_ctx::ExecutionContext;
pub use host_view::{
    AcquireDeviceSliceOp, AcquireMappedSliceOp, DeviceSliceHostView, HostAccessibleExt,
    HostReadableExt, HostWritableExt, MapAccess, MapReadOnly, MapReadWrite, MappedSliceHostView,
    ReleaseDeviceSliceOp, ReleaseMappedSliceOp,
};
pub use mappable::{DeviceSliceMapHandle, Mappable};
pub use opencl3::types::cl_uint;
pub use record::{
    BufHandle, MemRef, RecordContext, RecordExt, RecordableBuffer, RecordableOp, RecordedGraph,
};
pub use transfer::UploadSource;

#[cfg(feature = "async-events")]
pub use eager::DeviceChainFuture;
pub use eager::{
    AndThenHost, AndThenHostWithContext, ArcSplit, BindAll, BindMode, Cell, Checkout, CopyTo2, Dep,
    Deps, DeviceDynOp, DeviceEnqueue, DeviceFanOutExt, DeviceOp, DeviceOpExt, DeviceProfileExt,
    Download, ExecMode, FanOut, Fill, FillMapped, FromCheckout, ImageDownloadEager,
    ImageUploadEager, Input, OnDevice, Pipe, ReadInto, SlotBinder, SlotCell, SlotEq, SlotHandle,
    SlotState, Tag, ToInput, TransferToDevice, Upload, WriteDevice, WriteMapped, arc_split, arced,
    bundle2, bundle3, bundle4, bundle5, bundle6, bundle7, bundle8, bundle9, bundle10, bundle11,
    bundle12, bundle13, bundle14, bundle15, bundle16, deps_as_events, deps_into_single_event,
    eager_copy_to, fan_out, fill_mapped, forward, image_download, image_upload, lift, read_into,
    transfer_to_device, transfer_to_device_at, value, wrap_event, write, write_mapped,
};

// Stage-3 proc-macro frontend.
pub use claspr_macros::{device, kernel, kernels};

// Re-exports from opencl3 — the types users actually touch through
// claspr's API.
pub use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
pub use opencl3::event::Event;
pub use opencl3::kernel::Kernel;
pub use opencl3::program::Program;
pub use opencl3::types::cl_event;

/// Convenience glob-import surface.
///
/// `use claspr::prelude::*;` brings claspr's graph verbs/terminals (the
/// [`DeviceOp`]/[`DeviceOpExt`]/[`DeviceFanOutExt`]/[`DeviceProfileExt`]
/// traits), the [`Launcher`] trait, the host-view ext traits
/// ([`HostReadableExt`]/[`HostWritableExt`]/[`HostAccessibleExt`]), the common
/// access markers ([`ReadWrite`]/[`ReadOnly`]/[`WriteOnly`]/[`Frozen`]), the
/// high-frequency Tier 2 constructors (`upload`/`download`/`fill`/`alloc_zero`/
/// `value`/`lift`/`bundle2..16`/`fan_out`/`arc_split`/…), the chain-entry
/// macros (`bundle!`/`upload!`/`download!`/`device_slice!`/…), and the core
/// types ([`Context`]/[`DeviceSlice`]/[`Device`]/[`Result`]/[`Error`]) into
/// scope — so callers can invoke the trait methods without hand-listing every
/// extension trait.
///
/// This is deliberately focused: it does **not** re-export the entire crate.
/// Anything not in the prelude (image kernel-arg types, `Buffer`, the SVM/USM
/// slice families, lower-level `op`/`launch` items, …) is still reachable at
/// the crate root (`claspr::…`) and through the module paths.
pub mod prelude {
    // ── Graph verbs / terminals + supporting traits ──
    pub use crate::eager::{DeviceFanOutExt, DeviceOp, DeviceOpExt, DeviceProfileExt};
    pub use crate::host_view::{HostAccessibleExt, HostReadableExt, HostWritableExt};
    pub use crate::queue::Launcher;

    // ── Access markers (the ones graphs are parameterised over) ──
    pub use crate::access::{Frozen, ReadOnly, ReadWrite, WriteOnly};

    // ── Core types ──
    pub use crate::buffer::DeviceSlice;
    pub use crate::context::Context;
    pub use crate::device::Device;
    pub use crate::error::{Error, Result};

    // ── High-frequency Tier 2 constructors ──
    pub use crate::eager::{
        alloc_zero, arc_split, arced, bundle2, bundle3, bundle4, bundle5, bundle6, bundle7,
        bundle8, bundle9, bundle10, bundle11, bundle12, bundle13, bundle14, bundle15, bundle16,
        download, fan_out, fill, forward, lift, transfer_to_device, transfer_to_device_at, upload,
        upload_as, value, write,
    };

    // ── Chain-entry macros (re-exported from the crate root, where
    // `#[macro_export]` places them) ──
    pub use crate::{
        bundle, device_slice, device_slice_alloc_uninit, device_slice_alloc_zero,
        device_slice_filled, device_slice_from_slice, download, slot, slots, upload,
    };

    // ── Typed-slot tags + multi-fill ──
    pub use crate::eager::{BindAll, SlotHandle, Tag};
}
