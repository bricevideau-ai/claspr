//! 2D image helpers — format- and access-typed.
//!
//! [`Image2D<A, F>`] is a generic 2D image parameterised on
//! [`ImageAccess`] (`ReadOnly` / `WriteOnly` / `ReadWrite` ZST
//! markers) and [`Format`](format::Format) (`R8G8B8A8Uint`,
//! `R8G8B8A8Unorm`, `R32Float`, etc.). The proc-macro emits
//! matching `&Image2D<A, F>` parameters for `&Image!(format=...,
//! sampled=...)` kernel parameters.
//!
//! ## Format-naming convention
//!
//! Follows OpenCL's `cl_channel_type` distinction precisely, not
//! Vulkan/D3D naming:
//!
//! - `*Uint` / `*Sint` — kernel sees `uint`/`int` values (e.g.
//!   `0..=255` for an 8-bit channel). Maps to
//!   `CL_UNSIGNED_INT8`/`CL_SIGNED_INT8`.
//! - `*Unorm` / `*Snorm` — kernel sees `float` values normalised
//!   to `[0.0, 1.0]` / `[-1.0, 1.0]`. Maps to `CL_UNORM_INT8` /
//!   `CL_SNORM_INT8`.
//! - `*Float` — kernel sees `float`; storage is IEEE float.
//!
//! **Picking the wrong one** typically silently corrupts kernel
//! writes (`Uint` values written into a `Unorm` image get
//! reinterpreted as float bits). When in doubt, match the kernel's
//! `Image!(type=...)` token: `type=u32` → `*Uint`, `type=f32` →
//! `*Float`. The legacy [`Image2DRgba8`] alias resolves to
//! `R8G8B8A8Uint` for rust-gpu's `type=u32` kernel default.

use crate::Result;
use crate::context::Context;
use crate::launch::KernelArg;
use crate::queue::Launcher;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{
    CL_MEM_OBJECT_IMAGE2D, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, ClMem, Image,
};
use opencl3::types::{CL_BLOCKING, cl_image_desc, cl_image_format, cl_mem_flags};
use std::marker::PhantomData;
use std::ptr;

// ── ImageAccess markers ─────────────────────────────────────────────

mod access_sealed {
    pub trait Sealed {}
    impl Sealed for super::ReadOnly {}
    impl Sealed for super::WriteOnly {}
    impl Sealed for super::ReadWrite {}
}

/// Sealed marker for image access mode. Implementors are
/// [`ReadOnly`], [`WriteOnly`], and [`ReadWrite`].
///
/// Corresponds to OpenCL's `CL_MEM_{READ_ONLY, WRITE_ONLY,
/// READ_WRITE}` and SPIR-V's image-AccessQualifier
/// (`ReadOnly`/`WriteOnly`/`ReadWrite`). Pick the strictest mode
/// the kernel uses; for an image written by one kernel and read by
/// another, use `ReadWrite` (requires OpenCL 2.0+).
pub trait ImageAccess: access_sealed::Sealed {
    #[doc(hidden)]
    const CL_FLAGS: cl_mem_flags;
}

/// Image readable by kernels (`CL_MEM_READ_ONLY`).
#[derive(Clone, Copy, Debug)]
pub struct ReadOnly;
impl ImageAccess for ReadOnly {
    const CL_FLAGS: cl_mem_flags = CL_MEM_READ_ONLY;
}

/// Image writable by kernels (`CL_MEM_WRITE_ONLY`).
#[derive(Clone, Copy, Debug)]
pub struct WriteOnly;
impl ImageAccess for WriteOnly {
    const CL_FLAGS: cl_mem_flags = CL_MEM_WRITE_ONLY;
}

/// Image both readable and writable by kernels
/// (`CL_MEM_READ_WRITE`). Requires OpenCL 2.0+ for storage images.
#[derive(Clone, Copy, Debug)]
pub struct ReadWrite;
impl ImageAccess for ReadWrite {
    const CL_FLAGS: cl_mem_flags = CL_MEM_READ_WRITE;
}

// ── Format trait + ZST types ────────────────────────────────────────

/// Image storage formats — channel order + channel type pair from
/// the OpenCL spec, expressed as ZST markers.
///
/// Each format ZST implements [`Format`], which carries the
/// `CHANNEL_ORDER` / `CHANNEL_TYPE` constants the runtime needs
/// and the [`Pixel`](Format::Pixel) associated type used by
/// [`Image2D::download`] to size the host buffer.
pub mod format {
    use opencl3::memory::{
        CL_FLOAT, CL_HALF_FLOAT, CL_R, CL_RG, CL_RGBA, CL_SIGNED_INT8, CL_SIGNED_INT16,
        CL_SIGNED_INT32, CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
        CL_UNSIGNED_INT8, CL_UNSIGNED_INT16, CL_UNSIGNED_INT32,
    };
    use opencl3::types::{cl_channel_order, cl_channel_type};

    mod sealed {
        pub trait Sealed {}
    }

    /// Sealed trait for image storage formats. See the module
    /// docs for the format-naming convention (`*Uint`/`*Unorm`/
    /// `*Float` etc.).
    pub trait Format: sealed::Sealed {
        /// `cl_channel_order` — `CL_R`, `CL_RG`, `CL_RGBA`, …
        const CHANNEL_ORDER: cl_channel_order;
        /// `cl_channel_data_type` — `CL_UNSIGNED_INT8`,
        /// `CL_UNORM_INT8`, `CL_FLOAT`, …
        const CHANNEL_TYPE: cl_channel_type;
        /// The Rust type representing one pixel on the host side.
        /// `[u8; 4]` for RGBA8 integer formats, `f32` for
        /// `R32Float`, `[f32; 4]` for `Rgba32Float`, etc.
        type Pixel: Copy;
    }

    macro_rules! format_zst {
        ($name:ident, $order:ident, $ctype:ident, $pixel:ty) => {
            #[doc = concat!(
                                                    "OpenCL image format: ",
                                                    stringify!($order), " / ", stringify!($ctype),
                                                    ". Pixel type: `", stringify!($pixel), "`."
                                                )]
            #[derive(Clone, Copy, Debug)]
            pub struct $name;
            impl sealed::Sealed for $name {}
            impl Format for $name {
                const CHANNEL_ORDER: cl_channel_order = $order;
                const CHANNEL_TYPE: cl_channel_type = $ctype;
                type Pixel = $pixel;
            }
        };
    }

    // RGBA8 family — `Uint`/`Sint` for integer kernel access,
    // `Unorm`/`Snorm` for normalized-float kernel access. Picking
    // the wrong one silently corrupts kernel writes.
    format_zst!(R8G8B8A8Uint, CL_RGBA, CL_UNSIGNED_INT8, [u8; 4]);
    format_zst!(R8G8B8A8Sint, CL_RGBA, CL_SIGNED_INT8, [i8; 4]);
    format_zst!(R8G8B8A8Unorm, CL_RGBA, CL_UNORM_INT8, [u8; 4]);
    format_zst!(R8G8B8A8Snorm, CL_RGBA, CL_SNORM_INT8, [i8; 4]);

    // RGBA16 family
    format_zst!(R16G16B16A16Uint, CL_RGBA, CL_UNSIGNED_INT16, [u16; 4]);
    format_zst!(R16G16B16A16Sint, CL_RGBA, CL_SIGNED_INT16, [i16; 4]);
    format_zst!(R16G16B16A16Unorm, CL_RGBA, CL_UNORM_INT16, [u16; 4]);
    format_zst!(R16G16B16A16Snorm, CL_RGBA, CL_SNORM_INT16, [i16; 4]);
    format_zst!(R16G16B16A16Float, CL_RGBA, CL_HALF_FLOAT, [u16; 4]); // half = u16 bits

    // RGBA32 family
    format_zst!(R32G32B32A32Float, CL_RGBA, CL_FLOAT, [f32; 4]);
    format_zst!(R32G32B32A32Uint, CL_RGBA, CL_UNSIGNED_INT32, [u32; 4]);
    format_zst!(R32G32B32A32Sint, CL_RGBA, CL_SIGNED_INT32, [i32; 4]);
    /// Alias of [`R32G32B32A32Float`] — common short form.
    pub type Rgba32Float = R32G32B32A32Float;

    // Single- and two-channel
    format_zst!(R32Float, CL_R, CL_FLOAT, f32);
    format_zst!(R32Uint, CL_R, CL_UNSIGNED_INT32, u32);
    format_zst!(R32Sint, CL_R, CL_SIGNED_INT32, i32);
    format_zst!(R16Float, CL_R, CL_HALF_FLOAT, u16);
    format_zst!(R8Unorm, CL_R, CL_UNORM_INT8, u8);

    format_zst!(R32G32Float, CL_RG, CL_FLOAT, [f32; 2]);
    format_zst!(R32G32Uint, CL_RG, CL_UNSIGNED_INT32, [u32; 2]);
}

// ── Image2D ─────────────────────────────────────────────────────────

/// A 2D image with compile-time access mode and storage format.
///
/// `A` is one of [`ReadOnly`] / [`WriteOnly`] / [`ReadWrite`] —
/// matching the kernel-side access qualifier rust-gpu emits for
/// `&Image` vs `&mut Image` parameters. `F` is a
/// [`Format`](format::Format) ZST that picks the channel order +
/// channel type and the per-pixel host element type used by
/// [`download`](Image2D::download).
///
/// Construct via [`Image2D::alloc`]; read back via
/// [`Image2D::download`] (typed pixels) or
/// [`Image2D::download_bytes`] (raw bytes for byte-oriented sinks
/// like the PPM helper). Kernel arguments accept `&Image2D<A, F>`
/// directly.
pub struct Image2D<A: ImageAccess, F: format::Format> {
    image: Image,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    ctx: Context,
    _access: PhantomData<A>,
    _format: PhantomData<F>,
}

impl<A: ImageAccess, F: format::Format> Image2D<A, F> {
    /// Allocate a `width × height` image. Pure context op — no
    /// command queue needed (`clCreateImage` doesn't enqueue
    /// anything).
    ///
    /// Returns an error if the device doesn't advertise image
    /// support — check `ctx.device().cl3().image_support()` first
    /// if you want to fall back gracefully.
    pub fn alloc(ctx: &Context, width: u32, height: u32) -> Result<Self> {
        let format = cl_image_format {
            image_channel_order: F::CHANNEL_ORDER,
            image_channel_data_type: F::CHANNEL_TYPE,
        };
        let desc = cl_image_desc {
            image_type: CL_MEM_OBJECT_IMAGE2D,
            image_width: width as usize,
            image_height: height as usize,
            image_depth: 0,
            image_array_size: 0,
            image_row_pitch: 0,
            image_slice_pitch: 0,
            num_mip_levels: 0,
            num_samples: 0,
            buffer: ptr::null_mut(),
        };
        // SAFETY: null host pointer + CL_MEM_* access flag means
        // OpenCL allocates fresh device memory and ignores the
        // host-pointer contract that makes `Image::create`
        // generally unsafe.
        let image = unsafe {
            Image::create(
                ctx.raw_context(),
                A::CL_FLAGS,
                &format,
                &desc,
                ptr::null_mut(),
            )?
        };
        Ok(Image2D {
            image,
            width,
            height,
            ctx: ctx.clone(),
            _access: PhantomData,
            _format: PhantomData,
        })
    }

    /// Read this image as raw bytes — `Vec<u8>` of length
    /// `width * height * size_of::<F::Pixel>()`. Useful when
    /// handing the data to a byte-oriented sink (file write, PPM
    /// helper, …).
    pub fn download_bytes<L: Launcher>(&self, launcher: &L) -> Result<Vec<u8>> {
        let pixel_count = (self.width as usize) * (self.height as usize);
        let mut bytes = vec![0u8; pixel_count * std::mem::size_of::<F::Pixel>()];
        let origin = [0usize, 0, 0];
        let region = [self.width as usize, self.height as usize, 1];
        // SAFETY: blocking read into a freshly-allocated Vec<u8>;
        // byte count matches the image's pixel layout by
        // construction (F picks both the channel format and the
        // per-pixel size).
        unsafe {
            launcher
                .cl_queue()
                .enqueue_read_image(
                    &self.image,
                    CL_BLOCKING,
                    origin.as_ptr(),
                    region.as_ptr(),
                    0,
                    0,
                    bytes.as_mut_ptr().cast(),
                    &[],
                )?
                .wait()?;
        }
        Ok(bytes)
    }

    /// Read this image into a host `Vec<F::Pixel>` of length
    /// `width * height`. Blocking.
    pub fn download<L: Launcher>(&self, launcher: &L) -> Result<Vec<F::Pixel>>
    where
        F::Pixel: Default,
    {
        let pixel_count = (self.width as usize) * (self.height as usize);
        let mut pixels = vec![<F::Pixel as Default>::default(); pixel_count];
        let origin = [0usize, 0, 0];
        let region = [self.width as usize, self.height as usize, 1];
        // SAFETY: see download_bytes.
        unsafe {
            launcher
                .cl_queue()
                .enqueue_read_image(
                    &self.image,
                    CL_BLOCKING,
                    origin.as_ptr(),
                    region.as_ptr(),
                    0,
                    0,
                    pixels.as_mut_ptr().cast(),
                    &[],
                )?
                .wait()?;
        }
        Ok(pixels)
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Borrow the underlying opencl3 [`Image`] for direct use.
    pub fn image(&self) -> &Image {
        &self.image
    }
}

impl<A: ImageAccess, F: format::Format> KernelArg for Image2D<A, F> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let cl_mem_handle = self.image.get();
        unsafe {
            exec.set_arg(&cl_mem_handle);
        }
    }
}

// ── Back-compat alias ───────────────────────────────────────────────

/// Legacy 2D RGBA8 image — `Image2D<ReadWrite,
/// format::R8G8B8A8Uint>`. Matches rust-gpu's default
/// `Image!(2D, type=u32, sampled=false)` kernel parameter, which
/// writes `uint` values that need `CL_UNSIGNED_INT8` storage (the
/// `*Uint` format), *not* `CL_UNORM_INT8` (the `*Unorm` format).
///
/// New code should spell the generic type directly to make the
/// access mode + format explicit.
pub type Image2DRgba8 = Image2D<ReadWrite, format::R8G8B8A8Uint>;
