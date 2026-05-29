//! Access markers — typestate generics encoding `clCreateBuffer`'s
//! kernel and host access flags. Shared between [`DeviceSlice`] and
//! [`Image2D`].
//!
//! Each marker is a zero-sized struct that impls two traits:
//!
//! - [`KernelAccess`] — the `CL_MEM_{READ_WRITE,READ_ONLY,WRITE_ONLY}`
//!   bit. Determines whether kernels can read and/or write through the
//!   slice / image arg.
//! - [`HostAccess`] — the `CL_MEM_HOST_*` bits. Determines which
//!   `acquire_host_view_*` / `write` / `fill` methods are available
//!   on a buffer. (Images currently ignore this axis — all today's
//!   image markers set `HOST_FLAGS = 0`, equivalent to full host
//!   access — but the trait is uniform so images can opt into host
//!   constraints later.)
//!
//! Together they compose into [`MemMode`]; every cl_mem-backed type's
//! type carries `M: MemMode = ReadWrite` so default ergonomics are
//! unchanged.
//!
//! ## The marker set
//!
//! Kernel-side × host-side, after excluding `WRITE_ONLY` on the buffer
//! side (rust-gpu has no `MaybeUninit` story so write-only `&mut [T]`
//! can't be statically guaranteed):
//!
//! | | Host RW | Host RO | Host NoAccess |
//! |---|---|---|---|
//! | **Kernel RW** | [`ReadWrite`] | [`HostReadOnly`] | [`DeviceScratch`] |
//! | **Kernel RO** | [`ReadOnly`] | [`Frozen`] | *(unnamed)* |
//! | **Kernel WO** | [`WriteOnly`] (images only) | — | — |
//!
//! `WriteOnly` exists because images use explicit `read`/`write`
//! functions inside kernels — rust-gpu CAN prove a write-only image
//! is never read. Buffers passed as `&mut [T]` can't make the same
//! guarantee (the kernel body can always read through the `&mut`).
//!
//! ## Naming notes
//!
//! - `Frozen` (not `Immutable`) — `CL_MEM_IMMUTABLE_EXT` (from the
//!   `cl_ext_immutable_memory_objects` extension) is a separate,
//!   stronger flag that guarantees the *implementation itself* won't
//!   alter the bytes (ROM-region placement, permanent caching).
//!   `Frozen` is the weaker user-visible write-locked-after-creation
//!   semantic; `Immutable` stays reserved for the extension.
//! - `ReadOnly` is the kernel-perspective name (kernels can only
//!   read) — the buffer-side use case is "constant data the kernel
//!   reads, host updates between launches."
//! - `DeviceScratch` — host never observes it; pure intermediate.
//!
//! ## Compile-fail spec (the typestate scheme's enforcement points)
//!
//! Frozen → `&mut [T]` kernel param: rejected because Frozen doesn't
//! impl [`KernelWritable`].
//!
//! ```compile_fail
//! use claspr::{Context, DeviceSlice, Frozen};
//! let ctx = Context::any().unwrap();
//! let frozen: DeviceSlice<u32, Frozen> =
//!     DeviceSlice::from_slice(&ctx, &[0u32; 16]).unwrap();
//! let kernels = claspr_test_kernels::kernels::kernels(&ctx).unwrap();
//! let _ = kernels.scale_u32([16], frozen, 2);  // ← &mut [u32]: ERROR
//! ```
//!
//! Frozen → `.write(...)`: rejected because Frozen doesn't impl
//! [`HostWritable`].
//!
//! ```compile_fail
//! use claspr::{Context, DeviceSlice, Frozen, Launcher};
//! let ctx = Context::any().unwrap();
//! let mut frozen: DeviceSlice<u32, Frozen> =
//!     DeviceSlice::from_slice(&ctx, &[0u32; 16]).unwrap();
//! let _ = frozen.write(&ctx, &[1u32; 16]);  // ← HostWritable: ERROR
//! ```
//!
//! Frozen → `.fill(...)`: rejected because Frozen doesn't impl
//! [`KernelWritable`].
//!
//! ```compile_fail
//! use claspr::{Context, DeviceSlice, Frozen, Launcher};
//! let ctx = Context::any().unwrap();
//! let mut frozen: DeviceSlice<u32, Frozen> =
//!     DeviceSlice::from_slice(&ctx, &[0u32; 16]).unwrap();
//! let _ = frozen.fill(&ctx, 9u32);  // ← KernelWritable: ERROR
//! ```
//!
//! Frozen → `.acquire_host_view()` (mut variant): rejected because
//! Frozen doesn't impl [`HostWritable`] (only Read).
//!
//! ```compile_fail
//! use claspr::{Context, DeviceSlice, Frozen};
//! use claspr_async::HostWritableExt;
//! let ctx = Context::any().unwrap();
//! let frozen: DeviceSlice<u32, Frozen> =
//!     DeviceSlice::from_slice(&ctx, &[0u32; 16]).unwrap();
//! let _ = frozen.acquire_host_view();  // ← HostWritable: ERROR
//! ```
//!
//! Frozen → `.acquire_host_view_read()`: ALLOWED (Frozen impls
//! [`HostReadable`]).
//!
//! ```ignore
//! use claspr::{Context, DeviceSlice, Frozen};
//! use claspr_async::HostReadableExt;
//! let ctx = Context::any()?;
//! let frozen: DeviceSlice<u32, Frozen> =
//!     DeviceSlice::from_slice(&ctx, &[0u32; 16])?;
//! let _read_op = frozen.acquire_host_view_read();  // ✓ compiles
//! ```
//!
//! [`DeviceSlice`]: crate::DeviceSlice
//! [`Image2D`]: crate::Image2D

use opencl3::memory::{
    CL_MEM_HOST_NO_ACCESS, CL_MEM_HOST_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE,
    CL_MEM_WRITE_ONLY,
};
use opencl3::types::cl_mem_flags;

/// Kernel-side access classifier.
pub trait KernelAccess: 'static {
    /// The kernel-side bit (`CL_MEM_READ_WRITE` / `CL_MEM_READ_ONLY` /
    /// `CL_MEM_WRITE_ONLY`).
    const KERNEL_FLAGS: cl_mem_flags;
}

/// Host-side access classifier.
pub trait HostAccess: 'static {
    /// The host-side bits (`CL_MEM_HOST_READ_ONLY` /
    /// `CL_MEM_HOST_NO_ACCESS`, or 0 for default = host RW).
    const HOST_FLAGS: cl_mem_flags;
}

/// Composed access mode — every named marker impls both
/// [`KernelAccess`] and [`HostAccess`].
pub trait MemMode: KernelAccess + HostAccess {
    /// Convenience: the bitwise OR of `KERNEL_FLAGS | HOST_FLAGS`,
    /// ready to pass to `clCreateBuffer` / `clCreateImage`.
    const FLAGS: cl_mem_flags = Self::KERNEL_FLAGS | Self::HOST_FLAGS;
}

impl<T: KernelAccess + HostAccess> MemMode for T {}

// ── Kernel-side classification traits ──────────────────────────────
//
// These split the [`KernelAccess`] trait by *what kernels can do* with
// a buffer/image carrying this marker:
//
// - [`KernelReadable`] — kernel may issue read operations through this
//   slice / image. True for every marker except `WriteOnly` (image-
//   only marker where kernel reads are UB per spec).
// - [`KernelWritable`] — kernel may issue write operations. False for
//   `ReadOnly` and `Frozen` (kernel-side `CL_MEM_READ_ONLY`).
//
// The pair drives the [`KernelSliceReadArg`] /
// [`KernelSliceReadWriteArg`] trait split in `crate::launch`: a buffer
// can be passed to a kernel `&[T]` slice arg iff its marker impls
// `KernelReadable`; a kernel `&mut [T]` slice arg iff its marker impls
// `KernelReadable + KernelWritable` (rust-gpu's `&mut [T]` permits
// reading, so write-only-only isn't expressible at the slice level).
//
// [`KernelSliceReadArg`]: crate::KernelSliceReadArg
// [`KernelSliceReadWriteArg`]: crate::KernelSliceReadWriteArg

/// The kernel may read through this access mode.
pub trait KernelReadable: KernelAccess {}

/// The kernel may write through this access mode.
pub trait KernelWritable: KernelAccess {}

// ReadWrite: read AND write
impl KernelReadable for ReadWrite {}
impl KernelWritable for ReadWrite {}

// ReadOnly: read only
impl KernelReadable for ReadOnly {}

// WriteOnly (image-only): write only — explicitly NOT readable
impl KernelWritable for WriteOnly {}

// HostReadOnly: kernel side is RW
impl KernelReadable for HostReadOnly {}
impl KernelWritable for HostReadOnly {}

// Frozen: kernel side is read-only
impl KernelReadable for Frozen {}

// DeviceScratch: kernel side is RW (host side is what's restricted)
impl KernelReadable for DeviceScratch {}
impl KernelWritable for DeviceScratch {}

// ── Host-side classification traits ────────────────────────────────
//
// Mirror of [`KernelReadable`] / [`KernelWritable`] for the host axis.
// Gate `DeviceSlice::read` / `write` / `acquire_host_view_*` at the
// type level so misuse is a compile error rather than a runtime
// `CL_INVALID_OPERATION` from the OpenCL driver.
//
// - [`HostReadable`] — host may read the buffer's bytes via
//   `clEnqueueReadBuffer` / `clEnqueueMapBuffer(CL_MAP_READ)`. True for
//   every marker except `DeviceScratch` (`CL_MEM_HOST_NO_ACCESS`).
// - [`HostWritable`] — host may write the buffer's bytes via
//   `clEnqueueWriteBuffer` / `clEnqueueMapBuffer(CL_MAP_WRITE)`. False
//   for `HostReadOnly`, `Frozen` (`CL_MEM_HOST_READ_ONLY`) and
//   `DeviceScratch` (`CL_MEM_HOST_NO_ACCESS`).

/// The host may read the buffer's bytes.
pub trait HostReadable: HostAccess {}

/// The host may write the buffer's bytes.
pub trait HostWritable: HostAccess {}

// ReadWrite / ReadOnly / WriteOnly: default host access = RW
impl HostReadable for ReadWrite {}
impl HostWritable for ReadWrite {}
impl HostReadable for ReadOnly {}
impl HostWritable for ReadOnly {}
impl HostReadable for WriteOnly {}
impl HostWritable for WriteOnly {}

// HostReadOnly: host-read-only
impl HostReadable for HostReadOnly {}

// Frozen: host-read-only
impl HostReadable for Frozen {}

// DeviceScratch: host-no-access — neither readable nor writable from
// the host. Don't impl either trait.

// ── Kernel-side variants × Host RW (the "no host restriction" row) ──

/// Default — kernel reads/writes, host reads/writes.
/// `CL_MEM_READ_WRITE`.
#[derive(Clone, Copy, Debug)]
pub struct ReadWrite;
impl KernelAccess for ReadWrite {
    const KERNEL_FLAGS: cl_mem_flags = CL_MEM_READ_WRITE;
}
impl HostAccess for ReadWrite {
    const HOST_FLAGS: cl_mem_flags = 0;
}

/// Kernel reads only; host has full read/write access.
/// `CL_MEM_READ_ONLY`.
///
/// Buffer use case: constant data the kernel only reads (weights,
/// lookup tables), host updates between launches.
/// Image use case: read-only sampled image input.
#[derive(Clone, Copy, Debug)]
pub struct ReadOnly;
impl KernelAccess for ReadOnly {
    const KERNEL_FLAGS: cl_mem_flags = CL_MEM_READ_ONLY;
}
impl HostAccess for ReadOnly {
    const HOST_FLAGS: cl_mem_flags = 0;
}

/// Kernel writes only; host has full read/write access.
/// `CL_MEM_WRITE_ONLY`.
///
/// **Image-only at the typed surface.** Buffers can't safely use this
/// marker because rust-gpu's `&mut [T]` kernel param allows reading
/// (and reading from a `CL_MEM_WRITE_ONLY` buffer is UB). Images use
/// explicit `read`/`write` functions, so write-only images are
/// statically enforceable at the SPIR-V level.
#[derive(Clone, Copy, Debug)]
pub struct WriteOnly;
impl KernelAccess for WriteOnly {
    const KERNEL_FLAGS: cl_mem_flags = CL_MEM_WRITE_ONLY;
}
impl HostAccess for WriteOnly {
    const HOST_FLAGS: cl_mem_flags = 0;
}

// ── Host-restricted variants ────────────────────────────────────────

/// Kernel reads/writes; host can only read (inspect kernel output
/// without modifying). `CL_MEM_HOST_READ_ONLY`.
#[derive(Clone, Copy, Debug)]
pub struct HostReadOnly;
impl KernelAccess for HostReadOnly {
    const KERNEL_FLAGS: cl_mem_flags = CL_MEM_READ_WRITE;
}
impl HostAccess for HostReadOnly {
    const HOST_FLAGS: cl_mem_flags = CL_MEM_HOST_READ_ONLY;
}

/// Kernel reads only; host reads only. Set at construction via
/// `CL_MEM_COPY_HOST_PTR`, never modified again.
/// `CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY`.
///
/// **Not `Immutable`** — `CL_MEM_IMMUTABLE_EXT` is a separate,
/// stronger flag (cl_ext_immutable_memory_objects extension) that
/// guarantees the implementation itself can't alter the bytes.
/// `Frozen` is the weaker user-visible write-lock; `Immutable` stays
/// reserved for the extension.
#[derive(Clone, Copy, Debug)]
pub struct Frozen;
impl KernelAccess for Frozen {
    const KERNEL_FLAGS: cl_mem_flags = CL_MEM_READ_ONLY;
}
impl HostAccess for Frozen {
    const HOST_FLAGS: cl_mem_flags = CL_MEM_HOST_READ_ONLY;
}

/// Kernel reads/writes; host can't touch it at all. Pure intermediate
/// buffer between kernel stages. `CL_MEM_HOST_NO_ACCESS`.
#[derive(Clone, Copy, Debug)]
pub struct DeviceScratch;
impl KernelAccess for DeviceScratch {
    const KERNEL_FLAGS: cl_mem_flags = CL_MEM_READ_WRITE;
}
impl HostAccess for DeviceScratch {
    const HOST_FLAGS: cl_mem_flags = CL_MEM_HOST_NO_ACCESS;
}
