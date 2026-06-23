//! Image helpers — format- and access-typed, dimensionality-specific.
//!
//! Three concrete types, one per supported OpenCL image dimensionality:
//!
//! - [`Image1D<A, F>`] — 1D images (`CL_MEM_OBJECT_IMAGE1D`).
//! - [`Image2D<A, F>`] — 2D images (`CL_MEM_OBJECT_IMAGE2D`).
//! - [`Image3D<A, F>`] — 3D images (`CL_MEM_OBJECT_IMAGE3D`).
//!
//! Each is parameterised on [`ImageAccess`] (`ReadOnly` /
//! `WriteOnly` / `ReadWrite` ZST markers) and
//! [`Format`](format::Format) (`R8G8B8A8Uint`, `R8G8B8A8Unorm`,
//! `R32Float`, etc.). The proc-macro picks the right one based on
//! the leading `1D`/`2D`/`3D` ident in the kernel's `Image!(...)`
//! parameter type.
//!
//! Why three concrete types rather than one generic over
//! dimensionality: the underlying `cl_image_desc` shape differs
//! per-dim (1D needs only width; 2D adds height; 3D adds depth),
//! the host-side `download` shape differs (`width` vs
//! `width*height` vs `width*height*depth`), and the SPIR-V
//! `OpTypeImage` `Dim` operand on the kernel side is distinct per
//! dimensionality — so the host-side type must match what the
//! kernel expects, otherwise `clSetKernelArg` rejects the call.
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
use crate::access::{KernelAccess, MemMode};
use crate::buffer::{Buffer, DeviceSlice};
use crate::context::Context;
use crate::eager::{Deps, DeviceOp, DeviceOpExt, ExecMode, Input, Pipe, wrap_event};
use crate::error::Error;
use crate::exec_ctx::ExecutionContext;
use crate::launch::KernelArg;
use crate::queue::Launcher;
use opencl3::event::Event;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{
    CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D,
    CL_MEM_OBJECT_IMAGE2D_ARRAY, CL_MEM_OBJECT_IMAGE3D, ClMem, Image,
};
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_event, cl_image_desc, cl_image_format};
use std::marker::PhantomData;
use std::ptr;

// ── Access markers (re-exported from the shared access module) ─────
//
// Image2D used to define its own sealed `ImageAccess` trait + a private
// `ReadOnly` / `WriteOnly` / `ReadWrite` set of markers. Those are now
// the cross-cutting [`crate::access`] markers; this re-export keeps the
// existing `claspr::image::*` import paths and proc-macro-emitted
// `::claspr::ReadOnly` references working.

pub use crate::access::{ReadOnly, ReadWrite, WriteOnly};

/// Backward-compat alias for [`KernelAccess`].
/// Image2D's marker bound is now `A: KernelAccess` directly.
pub use crate::access::KernelAccess as ImageAccess;

// ── Format trait + ZST types ────────────────────────────────────────

/// Image storage formats — channel order + channel type pair from
/// the OpenCL spec, expressed as ZST markers.
///
/// Each format ZST implements [`Format`](format::Format), which carries the
/// `CHANNEL_ORDER` / `CHANNEL_TYPE` constants the runtime needs
/// and the [`Pixel`](format::Format::Pixel) associated type used by
/// [`Image2D::read`] to size the host buffer.
pub mod format {
    use opencl3::memory::{
        CL_FLOAT, CL_HALF_FLOAT, CL_R, CL_RG, CL_RGBA, CL_SIGNED_INT8, CL_SIGNED_INT16,
        CL_SIGNED_INT32, CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
        CL_UNSIGNED_INT8, CL_UNSIGNED_INT16, CL_UNSIGNED_INT32,
    };
    use opencl3::types::{cl_channel_order, cl_channel_type};

    mod sealed {
        pub trait Sealed {}
        pub trait FamilySealed {}
    }

    /// Sampled-type family of an image format — the kernel-side
    /// view of what `read_imagef`/`read_imageui`/`read_imagei`
    /// returns and what `write_image*` accepts.
    ///
    /// OpenCL deliberately decouples kernel sampled-type from host
    /// storage format: a `type=u32` kernel reads/writes `uint`
    /// values regardless of whether the host allocated
    /// `R8G8B8A8Uint`, `R32Uint`, `R32G32B32A32Uint`, etc. The
    /// runtime translates between the storage format and the
    /// kernel's sampled type. The family marker captures *that*
    /// kernel-side view, not the storage format itself.
    ///
    /// The three marker ZSTs ([`Uint`] / [`Sint`] / [`Float`])
    /// implement this trait; every [`Format`] impl declares its
    /// family via the [`Format::SampledFamily`] associated type.
    ///
    /// Bound used by the proc-macro to constrain host-side
    /// `Image<dim>D<A, F>` arguments to kernels that declare a
    /// matching `type=` keyword (`type=u32` → `Uint`, `type=i32` →
    /// `Sint`, `type=f32` → `Float`).
    pub trait SampledTypeFamily: sealed::FamilySealed {}

    /// Unsigned-integer sampled-type family. Kernel sees `uint`
    /// values via `read_imageui`/`write_imageui`. Implemented for
    /// every `*Uint` storage format (`R8G8B8A8Uint`, `R32Uint`,
    /// `R32G32B32A32Uint`, …).
    #[derive(Clone, Copy, Debug)]
    pub struct Uint;
    impl sealed::FamilySealed for Uint {}
    impl SampledTypeFamily for Uint {}

    /// Signed-integer sampled-type family. Kernel sees `int`
    /// values via `read_imagei`/`write_imagei`. Implemented for
    /// every `*Sint` storage format.
    #[derive(Clone, Copy, Debug)]
    pub struct Sint;
    impl sealed::FamilySealed for Sint {}
    impl SampledTypeFamily for Sint {}

    /// Floating-point sampled-type family. Kernel sees `float`
    /// values via `read_imagef`/`write_imagef`. Implemented by both
    /// `*Float`/`*Unorm`/`*Snorm`/`*HalfFloat` storage formats —
    /// the OpenCL runtime converts each of these to/from float at
    /// access time per the channel data-type spec.
    #[derive(Clone, Copy, Debug)]
    pub struct Float;
    impl sealed::FamilySealed for Float {}
    impl SampledTypeFamily for Float {}

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
        /// Which kernel-side sampled-type family the storage
        /// format translates to / from. See [`SampledTypeFamily`].
        type SampledFamily: SampledTypeFamily;
    }

    macro_rules! format_zst {
        ($name:ident, $order:ident, $ctype:ident, $pixel:ty, $family:ident) => {
            #[doc = concat!(
                                        "OpenCL image format: ",
                                        stringify!($order), " / ", stringify!($ctype),
                                        ". Pixel type: `", stringify!($pixel), "`. ",
                                        "Sampled-type family: [`", stringify!($family), "`]."
                                    )]
            #[derive(Clone, Copy, Debug)]
            pub struct $name;
            impl sealed::Sealed for $name {}
            impl Format for $name {
                const CHANNEL_ORDER: cl_channel_order = $order;
                const CHANNEL_TYPE: cl_channel_type = $ctype;
                type Pixel = $pixel;
                type SampledFamily = $family;
            }
        };
    }

    // RGBA8 family — `Uint`/`Sint` for integer kernel access,
    // `Unorm`/`Snorm` for normalized-float kernel access. Picking
    // the wrong one silently corrupts kernel writes.
    format_zst!(R8G8B8A8Uint, CL_RGBA, CL_UNSIGNED_INT8, [u8; 4], Uint);
    format_zst!(R8G8B8A8Sint, CL_RGBA, CL_SIGNED_INT8, [i8; 4], Sint);
    format_zst!(R8G8B8A8Unorm, CL_RGBA, CL_UNORM_INT8, [u8; 4], Float);
    format_zst!(R8G8B8A8Snorm, CL_RGBA, CL_SNORM_INT8, [i8; 4], Float);

    // RGBA16 family
    format_zst!(R16G16B16A16Uint, CL_RGBA, CL_UNSIGNED_INT16, [u16; 4], Uint);
    format_zst!(R16G16B16A16Sint, CL_RGBA, CL_SIGNED_INT16, [i16; 4], Sint);
    format_zst!(R16G16B16A16Unorm, CL_RGBA, CL_UNORM_INT16, [u16; 4], Float);
    format_zst!(R16G16B16A16Snorm, CL_RGBA, CL_SNORM_INT16, [i16; 4], Float);
    format_zst!(R16G16B16A16Float, CL_RGBA, CL_HALF_FLOAT, [u16; 4], Float); // half = u16 bits

    // RGBA32 family
    format_zst!(R32G32B32A32Float, CL_RGBA, CL_FLOAT, [f32; 4], Float);
    format_zst!(R32G32B32A32Uint, CL_RGBA, CL_UNSIGNED_INT32, [u32; 4], Uint);
    format_zst!(R32G32B32A32Sint, CL_RGBA, CL_SIGNED_INT32, [i32; 4], Sint);
    /// Alias of [`R32G32B32A32Float`] — common short form.
    pub type Rgba32Float = R32G32B32A32Float;

    // Single- and two-channel
    format_zst!(R32Float, CL_R, CL_FLOAT, f32, Float);
    format_zst!(R32Uint, CL_R, CL_UNSIGNED_INT32, u32, Uint);
    format_zst!(R32Sint, CL_R, CL_SIGNED_INT32, i32, Sint);
    format_zst!(R16Float, CL_R, CL_HALF_FLOAT, u16, Float);
    format_zst!(R8Unorm, CL_R, CL_UNORM_INT8, u8, Float);

    format_zst!(R32G32Float, CL_RG, CL_FLOAT, [f32; 2], Float);
    format_zst!(R32G32Uint, CL_RG, CL_UNSIGNED_INT32, [u32; 2], Uint);
}

// ── Image2D ─────────────────────────────────────────────────────────

/// A 2D image with compile-time access mode and storage format.
///
/// `A` is one of [`ReadOnly`] / [`WriteOnly`] / [`ReadWrite`] —
/// matching the kernel-side access qualifier rust-gpu emits for
/// `&Image` vs `&mut Image` parameters. `F` is a
/// [`Format`](format::Format) ZST that picks the channel order +
/// channel type and the per-pixel host element type used by
/// [`read`](Image2D::read) / [`read_alloc`](Image2D::read_alloc).
///
/// Construct via [`Image2D::alloc`]; read back via
/// [`Image2D::read`] (caller-supplied dst) /
/// [`Image2D::read_alloc`] (returns a fresh Vec) /
/// [`Image2D::read_bytes_alloc`] (raw bytes for byte-oriented sinks
/// like the PPM helper). Kernel arguments accept `&Image2D<A, F>`
/// directly.
pub struct Image2D<A: KernelAccess, F: format::Format> {
    image: Image,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    ctx: Context,
    _access: PhantomData<A>,
    _format: PhantomData<F>,
}

impl<A: KernelAccess, F: format::Format> Image2D<A, F> {
    /// Allocate a `width × height` image. Pure context op — no
    /// command queue needed (`clCreateImage` doesn't enqueue
    /// anything).
    ///
    /// Returns an error if the device doesn't advertise image
    /// support — check `ctx.device().cl3().image_support()` first
    /// if you want to fall back gracefully.
    pub fn alloc(ctx: &Context, width: u32, height: u32) -> Result<Self> {
        let image = alloc_image::<A, F>(
            ctx,
            CL_MEM_OBJECT_IMAGE2D,
            width as usize,
            height as usize,
            1,
            0,
        )?;
        Ok(Image2D {
            image,
            width,
            height,
            ctx: ctx.clone(),
            _access: PhantomData,
            _format: PhantomData,
        })
    }

    /// Begin reading this image into a caller-supplied
    /// `Vec<F::Pixel>` of length `width * height`. Returns a lazy
    /// [`ImageRead`] graph node — pick a terminal (`.wait()` blocking,
    /// `.submit()` non-blocking, or `.wait_on`/`.submit_on`).
    pub fn read<'a>(self, dst: &'a mut [F::Pixel]) -> Result<ImageRead<'a, Self, F::Pixel>>
    where
        Self: ImageEnqueue,
        F::Pixel: Send,
    {
        let pixel_count = (self.width as usize) * (self.height as usize);
        let region = self.enqueue_region();
        image_read_op(self, region, dst, pixel_count, "Image2D")
    }

    /// Same as [`read`](Self::read) but raw bytes — caller-supplied
    /// `&mut [u8]` of length `width * height * size_of::<F::Pixel>()`.
    pub fn read_bytes<'a>(self, dst: &'a mut [u8]) -> Result<ImageRead<'a, Self, u8>>
    where
        Self: ImageEnqueue,
    {
        let pixel_count = (self.width as usize) * (self.height as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_read_bytes_op(self, region, dst, expected, "Image2D")
    }

    /// Convenience — `read` into a fresh `Vec`. The Op allocates
    /// and the terminal yields the `Vec`. Matches the existing
    /// `download(&launcher)` ergonomics in a lazy-builder shape.
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        ImageReadAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [self.width as usize, self.height as usize, 1],
            pixel_count: (self.width as usize) * (self.height as usize),
            _format: PhantomData::<F>,
        }
    }

    /// Same as [`read_alloc`](Self::read_alloc) but returns raw bytes.
    pub fn read_bytes_alloc(&self) -> ImageReadBytesAlloc<'_, F> {
        ImageReadBytesAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [self.width as usize, self.height as usize, 1],
            byte_len: (self.width as usize)
                * (self.height as usize)
                * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// Begin writing a typed pixel slice to this image. `pixels.len()`
    /// must equal `width * height` (asserted). Returns the [`ImageWrite`]
    /// graph node.
    pub fn write<'a>(self, pixels: &'a [F::Pixel]) -> ImageWrite<'a, Self, F::Pixel>
    where
        Self: ImageEnqueue,
        F::Pixel: Send + Sync,
    {
        let pixel_count = (self.width as usize) * (self.height as usize);
        let region = self.enqueue_region();
        image_write_op(self, region, pixels, pixel_count, "Image2D")
    }

    /// Same as [`write`](Self::write) but raw bytes — must be
    /// exactly `width * height * size_of::<F::Pixel>()` bytes.
    pub fn write_bytes<'a>(self, bytes: &'a [u8]) -> ImageWrite<'a, Self, u8>
    where
        Self: ImageEnqueue,
    {
        let pixel_count = (self.width as usize) * (self.height as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_write_bytes_op(self, region, bytes, expected, "Image2D")
    }

    /// Begin copying this image into `dst`. Both images must have
    /// the same dimensions and format-compatible pixel sizes
    /// (`clEnqueueCopyImage` surfaces format mismatches as
    /// `CL_IMAGE_FORMAT_MISMATCH` at terminal time).
    pub fn copy_to<A2: KernelAccess + Send + 'static>(
        self,
        dst: Image2D<A2, F>,
    ) -> ImageCopy<Self, Image2D<A2, F>>
    where
        Self: ImageEnqueue,
        F: Send + 'static,
    {
        let region = self.enqueue_region();
        image_copy_op(self, dst, region)
    }

    /// Begin filling every pixel with `pattern`. The 4-component
    /// pattern follows OpenCL's `clEnqueueFillImage` shape —
    /// match `T` to the format's `SampledTypeFamily` (`u32` for
    /// `Uint`, `i32` for `Sint`, `f32` for `Float` / `Unorm` /
    /// `Snorm`).
    pub fn fill<T: Copy + Send + 'static>(self, pattern: [T; 4]) -> ImageFill<Self, T>
    where
        Self: ImageEnqueue,
    {
        let region = self.enqueue_region();
        image_fill_op(self, region, pattern)
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

impl<A: KernelAccess, F: format::Format> KernelArg for Image2D<A, F> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let cl_mem_handle = self.image.get();
        unsafe {
            exec.set_arg(&cl_mem_handle);
        }
    }
}

// ── Internal: shared alloc + read helpers ──────────────────────────
//
// The per-dim wrappers below all funnel through these two functions.
// Each owns its own `width`/`height`/`depth` fields (since
// dimensionality changes the semantics of those), but the
// `clCreateImage` and `clEnqueueReadImage` calls themselves only
// differ in the `cl_image_desc` shape and the `region` triple — so
// we centralise both here.

fn alloc_image<A: KernelAccess, F: format::Format>(
    ctx: &Context,
    image_type: opencl3::types::cl_mem_object_type,
    width: usize,
    height: usize,
    depth: usize,
    array_size: usize,
) -> Result<Image> {
    let format = cl_image_format {
        image_channel_order: F::CHANNEL_ORDER,
        image_channel_data_type: F::CHANNEL_TYPE,
    };
    let desc = cl_image_desc {
        image_type,
        image_width: width,
        image_height: height,
        image_depth: depth,
        image_array_size: array_size,
        image_row_pitch: 0,
        image_slice_pitch: 0,
        num_mip_levels: 0,
        num_samples: 0,
        buffer: ptr::null_mut(),
    };
    // SAFETY: null host pointer + CL_MEM_* access flag means
    // OpenCL allocates fresh device memory and ignores the
    // host-pointer contract that makes `Image::create` generally
    // unsafe.
    let image = unsafe {
        Image::create(
            ctx.raw_context(),
            A::KERNEL_FLAGS,
            &format,
            &desc,
            ptr::null_mut(),
        )?
    };
    Ok(image)
}

// ── Raw enqueue helpers — the fold seam for the eager image ops ──────
//
// Each helper is the `clEnqueue*Image` body the matching Tier-1 image builder
// used to own, lifted out so the eager image graph nodes (`ImageWrite` /
// `ImageRead` / `ImageCopy` / `ImageFill` in this file) can enqueue directly
// against an `Image` without round-tripping through a borrow-based builder.
// `blocking` selects `CL_BLOCKING` vs `CL_NON_BLOCKING` (only `write`/`read`
// have a native blocking flag; copy/fill have none — the caller waits on the
// returned event for their blocking terminal). `deps` is the already-collected
// `cl_event` wait-list (the eager op flattens its `Deps` to raw handles, held
// alive across the call).
//
// The host pointer is a raw `*const`/`*mut c_void` + a `[usize; 3]` region; the
// eager ops hold a real typed slice (`&[E]` / `&mut [E]`, `E: Send`) so the op
// stays `Send`, and pass `slice.as_ptr()` here at enqueue time.

/// Raw `clEnqueueWriteImage` over `image` — body of the former `ImageWriteOp`.
pub(crate) fn write_image_enqueue<L: Launcher + ?Sized>(
    image: &mut Image,
    launcher: &L,
    region: [usize; 3],
    data: *const std::ffi::c_void,
    blocking: bool,
    deps: &[cl_event],
) -> Result<Event> {
    let cl_blocking = if blocking {
        CL_BLOCKING
    } else {
        CL_NON_BLOCKING
    };
    let origin = [0usize, 0, 0];
    // SAFETY: `data` points at the host slice the caller keeps alive across the
    // call; under CL_BLOCKING the driver finishes reading it before returning,
    // under CL_NON_BLOCKING the caller contract (slice outlives the event)
    // covers liveness. `image` must belong to the queue's context.
    let event = unsafe {
        launcher.cl_queue().enqueue_write_image(
            image,
            cl_blocking,
            origin.as_ptr(),
            region.as_ptr(),
            0,
            0,
            data as *mut std::ffi::c_void,
            deps,
        )?
    };
    Ok(event)
}

/// Raw `clEnqueueReadImage` from `image` into `dst` — body of the former
/// `ImageReadOp`.
pub(crate) fn read_image_enqueue<L: Launcher + ?Sized>(
    image: &Image,
    launcher: &L,
    region: [usize; 3],
    dst: *mut std::ffi::c_void,
    blocking: bool,
    deps: &[cl_event],
) -> Result<Event> {
    let cl_blocking = if blocking {
        CL_BLOCKING
    } else {
        CL_NON_BLOCKING
    };
    let origin = [0usize, 0, 0];
    // SAFETY: same context constraint as the write path; `dst` points at the
    // host slice the caller keeps alive; under CL_BLOCKING the driver fills it
    // before returning.
    let event = unsafe {
        launcher.cl_queue().enqueue_read_image(
            image,
            cl_blocking,
            origin.as_ptr(),
            region.as_ptr(),
            0,
            0,
            dst,
            deps,
        )?
    };
    Ok(event)
}

/// Raw `clEnqueueCopyImage` from `src` into `dst` — body of the former
/// `ImageCopyOp`. Non-blocking (copy has no `CL_BLOCKING` flag); the caller
/// waits on the returned event for a blocking terminal.
pub(crate) fn copy_image_enqueue<L: Launcher + ?Sized>(
    src: &Image,
    dst: &mut Image,
    launcher: &L,
    region: [usize; 3],
    deps: &[cl_event],
) -> Result<Event> {
    let origin = [0usize, 0, 0];
    // SAFETY: src/dst must belong to the queue's context; region bounds match by
    // construction (`copy_to` is only callable on same-dim image types).
    let event = unsafe {
        launcher.cl_queue().enqueue_copy_image(
            src,
            dst,
            origin.as_ptr(),
            origin.as_ptr(),
            region.as_ptr(),
            deps,
        )?
    };
    Ok(event)
}

/// Raw `clEnqueueFillImage` over `image` — body of the former `ImageFillOp`.
/// Non-blocking (fill has no `CL_BLOCKING` flag); the caller waits on the
/// returned event for a blocking terminal. `pattern` is a pointer to a
/// 4-component fill value the runtime byte-copies into every pixel in `region`.
pub(crate) fn fill_image_enqueue<L: Launcher + ?Sized>(
    image: &mut Image,
    launcher: &L,
    region: [usize; 3],
    pattern: *const std::ffi::c_void,
    deps: &[cl_event],
) -> Result<Event> {
    let origin = [0usize, 0, 0];
    // SAFETY: `pattern` is a valid 4-component fill value; `image` must belong
    // to the queue's context.
    let event = unsafe {
        launcher.cl_queue().enqueue_fill_image(
            image,
            pattern,
            origin.as_ptr(),
            region.as_ptr(),
            deps,
        )?
    };
    Ok(event)
}

// ── Eager image graph nodes — the image verbs ARE DeviceOps ─────────
//
// Mirroring the buffer fold (`Fill`/`WriteDevice`/`ReadInto`/`CopyTo2`), each
// image verb (`write`/`read`/`copy_to`/`fill` + the `*_bytes` variants) RETURNS
// the eager graph node below instead of a standalone borrow-based builder, so an
// image verb IS a graph node — usable standalone via the concrete-head
// `wait()`/`submit()` (context recovered from the owned image) or the
// launcher-generic `wait_on`/`submit_on`/`sync`, and composable via
// `and_then`/`bundle!`.
//
// The ops are **concrete-head**: every image verb consumes a caller-owned image,
// so the input is always `Input::Concrete`. The region + owning context are
// captured at construction (the per-type methods already compute the dim-shaped
// `[usize; 3]` region). The op holds a real typed host slice (`&[E]` / `&mut
// [E]`, `E: Send`) so it stays `Send` (`DeviceOp: Send`); the raw `*c_void`
// pointer the enqueue helper wants is taken from the slice at execute time.

/// Accessor seam over the owning image types so the generic eager image ops
/// ([`ImageWrite`] / [`ImageRead`] / [`ImageCopy`] / [`ImageFill`]) can reach the
/// underlying `Image` (shared / exclusive) without a per-type op. Implemented by
/// every owning image type via `impl_image_enqueue!`.
///
/// This is a claspr-internal seam — it surfaces in the public image-verb
/// signatures only as a bound (the methods consume an image whose concrete type
/// already implements it). It is not part of the stable API and should not be
/// implemented for foreign types.
pub trait ImageEnqueue: Send + 'static {
    /// Shared borrow of the underlying image (read / copy-src).
    fn image_ref(&self) -> &Image;
    /// Exclusive borrow of the underlying image (write / fill / copy-dst).
    fn image_mut(&mut self) -> &mut Image;
    /// Owning context (for the concrete-head no-launcher terminals).
    fn enqueue_ctx(&self) -> &Context;
    /// Dim-shaped `[width, height|array|1, depth|array|1]` region — the extent
    /// the `clEnqueue*Image` calls operate over.
    fn enqueue_region(&self) -> [usize; 3];
}

macro_rules! impl_image_enqueue {
    ($ty:ident, |$s:ident| $region:expr) => {
        impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static> ImageEnqueue
            for $ty<A, F>
        {
            fn image_ref(&self) -> &Image {
                &self.image
            }
            fn image_mut(&mut self) -> &mut Image {
                &mut self.image
            }
            fn enqueue_ctx(&self) -> &Context {
                &self.ctx
            }
            fn enqueue_region(&self) -> [usize; 3] {
                let $s = self;
                $region
            }
        }
    };
}

impl_image_enqueue!(Image2D, |s| [s.width as usize, s.height as usize, 1]);
impl_image_enqueue!(Image1D, |s| [s.width as usize, 1, 1]);
impl_image_enqueue!(Image3D, |s| [
    s.width as usize,
    s.height as usize,
    s.depth as usize
]);
impl_image_enqueue!(Image1DArray, |s| [
    s.width as usize,
    s.array_size as usize,
    1
]);
impl_image_enqueue!(Image2DArray, |s| [
    s.width as usize,
    s.height as usize,
    s.array_size as usize
]);
impl_image_enqueue!(Image1DBuffer, |s| [s.width as usize, 1, 1]);

/// Recover the owning [`Context`] from a concrete-head image-op input, or a
/// clear "pipe-fed" error for the no-launcher concrete-head terminals.
fn concrete_image_ctx<I: ImageEnqueue>(img: &Input<I>) -> Result<Context> {
    img.concrete()
        .map(|i| i.enqueue_ctx().clone())
        .ok_or(Error::NotSupported(
            "concrete-head terminal (wait/submit) on a pipe-fed image op — use \
             wait_on(&ctx) / sync(&ctx) for piped (graph) inputs",
        ))
}

// ── Leaf: image write (host pixels/bytes → image) ───────────────────

/// Write a caller host slice into an image, yielding the image back for reuse.
/// Returned by `image.write(...)` / `image.write_bytes(...)`. `Output = I`: the
/// image moves in and rebinds out (`let img = img.write(px).wait()?;`).
///
/// Generic over the host element `E` (the format's `Pixel` for `write`, `u8` for
/// `write_bytes`); the device-side byte extent is the captured `region`.
pub struct ImageWrite<'a, I: ImageEnqueue, E> {
    img: Input<I>,
    region: [usize; 3],
    data: &'a [E],
    out: Pipe<I>,
}

impl<I: ImageEnqueue, E: Send + Sync> DeviceOp for ImageWrite<'_, I, E> {
    type Output = I;

    fn output_pipe(&self) -> Pipe<I> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (mut img, deps) = self.img.resolve()?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let data = self.data.as_ptr() as *const std::ffi::c_void;
        match mode {
            ExecMode::Blocking => {
                write_image_enqueue(img.image_mut(), ec, self.region, data, true, &raw)?;
                self.out.put(img, Deps::new());
            }
            ExecMode::Pipelined => {
                let event =
                    write_image_enqueue(img.image_mut(), ec, self.region, data, false, &raw)?;
                self.out.put(img, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("image_write".into());
    }
}

impl<I: ImageEnqueue, E: Send + Sync> ImageWrite<'_, I, E> {
    /// Concrete-head blocking terminal: write on the image's own context default
    /// queue and return the image for reuse.
    pub fn wait(self) -> Result<I> {
        let ctx = concrete_image_ctx(&self.img)?;
        self.sync(&ctx)
    }

    /// Concrete-head non-blocking terminal returning the image plus a completion
    /// [`Event`]. (The host slice must outlive the event.)
    pub fn submit(self) -> Result<(I, crate::Event)> {
        let ctx = concrete_image_ctx(&self.img)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}

// ── Leaf: image read (image → caller host slice) ────────────────────

/// Read an image into a **caller-supplied** host slice, yielding the image back
/// for reuse. Returned by `image.read(&mut dst)` / `image.read_bytes(&mut dst)`.
/// `Output = I`: the image moves in and rebinds out
/// (`let img = img.read(&mut dst).wait()?;`). For a freshly-allocated `Vec`
/// output instead, use `image.read_alloc()` / the Tier-2 `image_download`.
pub struct ImageRead<'a, I: ImageEnqueue, E> {
    img: Input<I>,
    region: [usize; 3],
    dst: &'a mut [E],
    out: Pipe<I>,
}

impl<I: ImageEnqueue, E: Send> DeviceOp for ImageRead<'_, I, E> {
    type Output = I;

    fn output_pipe(&self) -> Pipe<I> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (img, deps) = self.img.resolve()?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let dst = self.dst.as_mut_ptr() as *mut std::ffi::c_void;
        match mode {
            ExecMode::Blocking => {
                read_image_enqueue(img.image_ref(), ec, self.region, dst, true, &raw)?;
                self.out.put(img, Deps::new());
            }
            ExecMode::Pipelined => {
                let event = read_image_enqueue(img.image_ref(), ec, self.region, dst, false, &raw)?;
                self.out.put(img, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("image_read".into());
    }
}

impl<I: ImageEnqueue, E: Send> ImageRead<'_, I, E> {
    /// Concrete-head blocking terminal: read into the caller slice on the image's
    /// own context default queue; return the image for reuse.
    pub fn wait(self) -> Result<I> {
        let ctx = concrete_image_ctx(&self.img)?;
        self.sync(&ctx)
    }

    /// Concrete-head non-blocking terminal returning the image plus a completion
    /// [`Event`]. (The `dst` slice must outlive the event.)
    pub fn submit(self) -> Result<(I, crate::Event)> {
        let ctx = concrete_image_ctx(&self.img)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}

// ── Leaf: image copy (image → image, same dims/format) ──────────────

/// Device-to-device copy from one image into another (matching dims + format).
/// Returned by `src.copy_to(dst)`. `Output = (Src, Dst)`: both images move in
/// and rebind out so they can be reused.
pub struct ImageCopy<Src: ImageEnqueue, Dst: ImageEnqueue> {
    src: Input<Src>,
    dst: Input<Dst>,
    region: [usize; 3],
    src_pipe: Pipe<Src>,
    dst_pipe: Pipe<Dst>,
}

impl<Src: ImageEnqueue, Dst: ImageEnqueue> DeviceOp for ImageCopy<Src, Dst> {
    type Output = (Src, Dst);
    type Handle = (Pipe<Src>, Pipe<Dst>);

    fn output_pipe(&self) -> Pipe<(Src, Dst)> {
        // Multi-output: the value is reconstructed in `collect` from the two
        // element pipes, never this single pipe (which stays empty). Mirrors the
        // buffer `CopyTo2` shape.
        Pipe::new()
    }

    fn handle(&self) -> Self::Handle {
        (self.src_pipe.clone(), self.dst_pipe.clone())
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (src, src_deps) = self.src.resolve()?;
        let (mut dst, dst_deps) = self.dst.resolve()?;
        let mut raw: Vec<crate::cl_event> = src_deps.iter().map(|d| d.as_ref().get()).collect();
        raw.extend(dst_deps.iter().map(|d| d.as_ref().get()));
        // Copy has no native CL_BLOCKING flag — always enqueue non-blocking; a
        // blocking terminal waits on the event via the carried deps.
        let event = copy_image_enqueue(src.image_ref(), dst.image_mut(), ec, self.region, &raw)?;
        let dep = vec![wrap_event(event)];
        self.src_pipe.put(src, dep.clone());
        self.dst_pipe.put(dst, dep);
        Ok(())
    }

    fn collect(self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<((Src, Dst), Deps)> {
        let src_pipe = self.src_pipe.clone();
        let dst_pipe = self.dst_pipe.clone();
        self.execute(ec, mode)?;
        let (src, src_deps) = src_pipe.take().ok_or(Error::NotSupported(
            "eager graph: image copy produced no src",
        ))?;
        let (dst, mut deps) = dst_pipe.take().ok_or(Error::NotSupported(
            "eager graph: image copy produced no dst",
        ))?;
        deps.extend(src_deps);
        Ok(((src, dst), deps))
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("image_copy".into());
    }
}

impl<Src: ImageEnqueue, Dst: ImageEnqueue> ImageCopy<Src, Dst> {
    /// Concrete-head blocking terminal: enqueue the copy on the src image's own
    /// context default queue, wait, and return `(src, dst)`.
    pub fn wait(self) -> Result<(Src, Dst)> {
        let ctx = concrete_image_ctx(&self.src)?;
        self.sync(&ctx)
    }

    /// Concrete-head non-blocking terminal returning `(src, dst)` plus a
    /// completion [`Event`].
    pub fn submit(self) -> Result<((Src, Dst), crate::Event)> {
        let ctx = concrete_image_ctx(&self.src)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}

// ── Leaf: image fill (4-component pattern → every pixel) ─────────────

/// Fill every pixel of an image with a 4-component `pattern`, yielding the image
/// back for reuse. Returned by `image.fill([v; 4])`. `Output = I`. Generic over
/// the pattern element `T` (`u32`/`i32`/`f32` per the format's `SampledFamily`).
pub struct ImageFill<I: ImageEnqueue, T: Copy> {
    img: Input<I>,
    region: [usize; 3],
    pattern: [T; 4],
    out: Pipe<I>,
}

impl<I: ImageEnqueue, T: Copy + Send + 'static> DeviceOp for ImageFill<I, T> {
    type Output = I;

    fn output_pipe(&self) -> Pipe<I> {
        self.out.clone()
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (mut img, deps) = self.img.resolve()?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let pattern = self.pattern;
        // Fill has no native CL_BLOCKING flag — always enqueue non-blocking; a
        // blocking terminal waits on the event via the carried deps.
        let event = fill_image_enqueue(
            img.image_mut(),
            ec,
            self.region,
            pattern.as_ptr() as *const std::ffi::c_void,
            &raw,
        )?;
        self.out.put(img, vec![wrap_event(event)]);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("image_fill".into());
    }
}

impl<I: ImageEnqueue, T: Copy + Send + 'static> ImageFill<I, T> {
    /// Concrete-head blocking terminal: fill on the image's own context default
    /// queue, wait, and return the image for reuse.
    pub fn wait(self) -> Result<I> {
        let ctx = concrete_image_ctx(&self.img)?;
        self.sync(&ctx)
    }

    /// Concrete-head non-blocking terminal returning the image plus a completion
    /// [`Event`].
    pub fn submit(self) -> Result<(I, crate::Event)> {
        let ctx = concrete_image_ctx(&self.img)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}
// ── ImageHostTransfer trait ────────────────────────────────────────
//
// Abstracts the per-image-type alloc / pixel-count / write / read
// surface so a single pair of Tier 2 combinators (`image_upload` /
// `image_download` in `claspr-async`) can handle every image type.
// Each image type implements this with its own `Dims` shape: 1D
// gets `u32`, 2D gets `(u32, u32)`, 3D gets `(u32, u32, u32)`, etc.
//
// This is a trait rather than a generic struct because the per-dim
// `alloc` signatures are genuinely different — there's no
// single-parameter constructor that works for all of them. The
// trait lets users name the image type at the call site
// (`image_upload::<Image2D<RW, R32Uint>>(pixels, (32, 32))`) and
// the combinator dispatches to the right `alloc` + region shape.

/// Polymorphism over the owning image types ([`Image2D`] /
/// [`Image1D`] / [`Image3D`] / [`Image1DArray`] / [`Image2DArray`])
/// so a single Tier 2 transfer combinator can produce or consume
/// any of them. Implemented by every owning image type.
///
/// `Image1DBuffer` is **not** included — it shares storage with a
/// `cl_mem` buffer, so the natural chain shape there is to upload
/// a `DeviceSlice<T>` first then `Image1DBufferView::view_of(&slice)`
/// to view it. The trait's `alloc` would also need an `image-buffer`
/// distinct signature.
pub trait ImageHostTransfer: Sized + Send + 'static {
    /// Dimension args for [`alloc`](Self::alloc). Concrete shape
    /// per image type — `u32` for 1D, `(u32, u32)` for 2D and
    /// 1DArray, `(u32, u32, u32)` for 3D and 2DArray.
    type Dims: Copy + Send + 'static;
    /// The pixel type (`F::Pixel`).
    type Pixel: Send + 'static;

    /// Allocate an image of the given dims on `ctx`.
    fn alloc(ctx: &Context, dims: Self::Dims) -> Result<Self>;

    /// Number of pixels in this image — product of all dimensions.
    /// Used by the combinator to size the host `Vec<Pixel>` on
    /// download.
    fn pixel_count(&self) -> usize;
}

impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static> ImageHostTransfer
    for Image2D<A, F>
where
    F::Pixel: Send + 'static,
{
    type Dims = (u32, u32);
    type Pixel = F::Pixel;
    fn alloc(ctx: &Context, dims: (u32, u32)) -> Result<Self> {
        Image2D::alloc(ctx, dims.0, dims.1)
    }
    fn pixel_count(&self) -> usize {
        (self.width() as usize) * (self.height() as usize)
    }
}

impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static> ImageHostTransfer
    for Image1D<A, F>
where
    F::Pixel: Send + 'static,
{
    type Dims = u32;
    type Pixel = F::Pixel;
    fn alloc(ctx: &Context, width: u32) -> Result<Self> {
        Image1D::alloc(ctx, width)
    }
    fn pixel_count(&self) -> usize {
        self.width() as usize
    }
}

impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static> ImageHostTransfer
    for Image3D<A, F>
where
    F::Pixel: Send + 'static,
{
    type Dims = (u32, u32, u32);
    type Pixel = F::Pixel;
    fn alloc(ctx: &Context, dims: (u32, u32, u32)) -> Result<Self> {
        Image3D::alloc(ctx, dims.0, dims.1, dims.2)
    }
    fn pixel_count(&self) -> usize {
        (self.width() as usize) * (self.height() as usize) * (self.depth() as usize)
    }
}

impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static> ImageHostTransfer
    for Image1DArray<A, F>
where
    F::Pixel: Send + 'static,
{
    type Dims = (u32, u32);
    type Pixel = F::Pixel;
    fn alloc(ctx: &Context, dims: (u32, u32)) -> Result<Self> {
        Image1DArray::alloc(ctx, dims.0, dims.1)
    }
    fn pixel_count(&self) -> usize {
        (self.width() as usize) * (self.array_size() as usize)
    }
}

impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static> ImageHostTransfer
    for Image2DArray<A, F>
where
    F::Pixel: Send + 'static,
{
    type Dims = (u32, u32, u32);
    type Pixel = F::Pixel;
    fn alloc(ctx: &Context, dims: (u32, u32, u32)) -> Result<Self> {
        Image2DArray::alloc(ctx, dims.0, dims.1, dims.2)
    }
    fn pixel_count(&self) -> usize {
        (self.width() as usize) * (self.height() as usize) * (self.array_size() as usize)
    }
}

// ── Per-type method helpers — region builders ──────────────────────
//
// Each `Image*Type::write` / `.read` / `.copy_to` / `.fill` method
// hands its dim-specific `[usize; 3]` region to one of the Op
// constructors below. Centralising the "build an op" step keeps
// per-image-type methods to a single line each.

fn image_write_op<'a, I: ImageEnqueue, T>(
    image: I,
    region: [usize; 3],
    pixels: &'a [T],
    expected_pixel_count: usize,
    type_name: &'static str,
) -> ImageWrite<'a, I, T> {
    assert_eq!(
        pixels.len(),
        expected_pixel_count,
        "{type_name}::write: pixel count {} ≠ expected {}",
        pixels.len(),
        expected_pixel_count,
    );
    ImageWrite {
        img: image.into(),
        region,
        data: pixels,
        out: Pipe::new(),
    }
}

fn image_write_bytes_op<'a, I: ImageEnqueue>(
    image: I,
    region: [usize; 3],
    bytes: &'a [u8],
    expected_bytes: usize,
    type_name: &'static str,
) -> ImageWrite<'a, I, u8> {
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "{type_name}::write_bytes: buffer length {} ≠ expected {}",
        bytes.len(),
        expected_bytes,
    );
    ImageWrite {
        img: image.into(),
        region,
        data: bytes,
        out: Pipe::new(),
    }
}

fn image_read_op<'a, I: ImageEnqueue, T>(
    image: I,
    region: [usize; 3],
    dst: &'a mut [T],
    expected_pixel_count: usize,
    _type_name: &'static str,
) -> Result<ImageRead<'a, I, T>> {
    if dst.len() != expected_pixel_count {
        return Err(Error::LengthMismatch {
            src: expected_pixel_count,
            dst: dst.len(),
        });
    }
    Ok(ImageRead {
        img: image.into(),
        region,
        dst,
        out: Pipe::new(),
    })
}

fn image_read_bytes_op<'a, I: ImageEnqueue>(
    image: I,
    region: [usize; 3],
    dst: &'a mut [u8],
    expected_bytes: usize,
    _type_name: &'static str,
) -> Result<ImageRead<'a, I, u8>> {
    if dst.len() != expected_bytes {
        return Err(Error::LengthMismatch {
            src: expected_bytes,
            dst: dst.len(),
        });
    }
    Ok(ImageRead {
        img: image.into(),
        region,
        dst,
        out: Pipe::new(),
    })
}

fn image_copy_op<Src: ImageEnqueue, Dst: ImageEnqueue>(
    src: Src,
    dst: Dst,
    region: [usize; 3],
) -> ImageCopy<Src, Dst> {
    ImageCopy {
        src: src.into(),
        dst: dst.into(),
        region,
        src_pipe: Pipe::new(),
        dst_pipe: Pipe::new(),
    }
}

fn image_fill_op<I: ImageEnqueue, T: Copy>(
    image: I,
    region: [usize; 3],
    pattern: [T; 4],
) -> ImageFill<I, T> {
    ImageFill {
        img: image.into(),
        region,
        pattern,
        out: Pipe::new(),
    }
}

// ── Read-into-fresh-Vec convenience builders (kept; not folded) ─────
//
// `read_alloc` / `read_bytes_alloc` allocate the destination `Vec` themselves
// and only offer a blocking terminal — the eager `image_download` Tier-2
// combinator covers the non-blocking / chained case. They borrow `&self` (no
// move-out), so they stay standalone builders rather than graph nodes; their
// enqueue body just calls the raw `read_image_enqueue` helper.

/// Convenience builder — `image.read_alloc()`. Allocates a `Vec<F::Pixel>` of
/// the right size at terminal time and yields it. Blocking-only (`.wait()` /
/// `.wait_on(&launcher)`); use the Tier-2 `image_download` for the chained /
/// non-blocking case.
pub struct ImageReadAlloc<'a, F: format::Format> {
    image: &'a Image,
    ctx: &'a Context,
    region: [usize; 3],
    pixel_count: usize,
    _format: PhantomData<F>,
}

impl<F: format::Format> ImageReadAlloc<'_, F>
where
    F::Pixel: Default + Copy,
{
    /// Blocking — allocate the Vec, enqueue + wait on the carried image's
    /// context default queue, return it.
    pub fn wait(self) -> Result<Vec<F::Pixel>> {
        let ctx = self.ctx;
        self.wait_on(ctx)
    }

    /// Blocking with an explicit launcher.
    pub fn wait_on<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Vec<F::Pixel>> {
        let mut pixels = vec![<F::Pixel as Default>::default(); self.pixel_count];
        read_image_enqueue(
            self.image,
            launcher,
            self.region,
            pixels.as_mut_ptr() as *mut std::ffi::c_void,
            true,
            &[],
        )?;
        Ok(pixels)
    }
}

/// Like [`ImageReadAlloc`] but returns raw bytes. Useful for PPM-write paths and
/// byte-oriented sinks that don't want the pixel-type round-trip.
pub struct ImageReadBytesAlloc<'a, F: format::Format> {
    image: &'a Image,
    ctx: &'a Context,
    region: [usize; 3],
    byte_len: usize,
    _format: PhantomData<F>,
}

impl<F: format::Format> ImageReadBytesAlloc<'_, F> {
    /// Blocking — allocate the Vec, enqueue + wait on the carried image's
    /// context default queue, return it.
    pub fn wait(self) -> Result<Vec<u8>> {
        let ctx = self.ctx;
        self.wait_on(ctx)
    }

    /// Blocking with an explicit launcher.
    pub fn wait_on<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; self.byte_len];
        read_image_enqueue(
            self.image,
            launcher,
            self.region,
            bytes.as_mut_ptr() as *mut std::ffi::c_void,
            true,
            &[],
        )?;
        Ok(bytes)
    }
}

// ── Image1D ─────────────────────────────────────────────────────────

/// A 1D image with compile-time access mode and storage format.
///
/// `A` and `F` carry the same meaning as on [`Image2D`]. The
/// underlying OpenCL object is created with
/// `CL_MEM_OBJECT_IMAGE1D`. Use this when the kernel side declares
/// `Image!(1D, type=..., sampled=...)`.
pub struct Image1D<A: KernelAccess, F: format::Format> {
    image: Image,
    width: u32,
    #[allow(dead_code)]
    ctx: Context,
    _access: PhantomData<A>,
    _format: PhantomData<F>,
}

impl<A: KernelAccess, F: format::Format> Image1D<A, F> {
    /// Allocate a `width`-pixel 1D image. Pure context op.
    pub fn alloc(ctx: &Context, width: u32) -> Result<Self> {
        let image = alloc_image::<A, F>(ctx, CL_MEM_OBJECT_IMAGE1D, width as usize, 1, 1, 0)?;
        Ok(Self {
            image,
            width,
            ctx: ctx.clone(),
            _access: PhantomData,
            _format: PhantomData,
        })
    }

    /// See [`Image2D::read`] — same shape, 1D region.
    pub fn read<'a>(self, dst: &'a mut [F::Pixel]) -> Result<ImageRead<'a, Self, F::Pixel>>
    where
        Self: ImageEnqueue,
        F::Pixel: Send,
    {
        let pixel_count = self.width as usize;
        let region = self.enqueue_region();
        image_read_op(self, region, dst, pixel_count, "Image1D")
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(self, dst: &'a mut [u8]) -> Result<ImageRead<'a, Self, u8>>
    where
        Self: ImageEnqueue,
    {
        let expected = (self.width as usize) * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_read_bytes_op(self, region, dst, expected, "Image1D")
    }

    /// See [`Image2D::read_alloc`].
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        ImageReadAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [self.width as usize, 1, 1],
            pixel_count: self.width as usize,
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::read_bytes_alloc`].
    pub fn read_bytes_alloc(&self) -> ImageReadBytesAlloc<'_, F> {
        ImageReadBytesAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [self.width as usize, 1, 1],
            byte_len: (self.width as usize) * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::write`].
    pub fn write<'a>(self, pixels: &'a [F::Pixel]) -> ImageWrite<'a, Self, F::Pixel>
    where
        Self: ImageEnqueue,
        F::Pixel: Send + Sync,
    {
        let pixel_count = self.width as usize;
        let region = self.enqueue_region();
        image_write_op(self, region, pixels, pixel_count, "Image1D")
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(self, bytes: &'a [u8]) -> ImageWrite<'a, Self, u8>
    where
        Self: ImageEnqueue,
    {
        let expected = (self.width as usize) * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_write_bytes_op(self, region, bytes, expected, "Image1D")
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<A2: KernelAccess + Send + 'static>(
        self,
        dst: Image1D<A2, F>,
    ) -> ImageCopy<Self, Image1D<A2, F>>
    where
        Self: ImageEnqueue,
        F: Send + 'static,
    {
        let region = self.enqueue_region();
        image_copy_op(self, dst, region)
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy + Send + 'static>(self, pattern: [T; 4]) -> ImageFill<Self, T>
    where
        Self: ImageEnqueue,
    {
        let region = self.enqueue_region();
        image_fill_op(self, region, pattern)
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Borrow the underlying opencl3 [`Image`].
    pub fn image(&self) -> &Image {
        &self.image
    }
}

impl<A: KernelAccess, F: format::Format> KernelArg for Image1D<A, F> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let cl_mem_handle = self.image.get();
        unsafe {
            exec.set_arg(&cl_mem_handle);
        }
    }
}

// ── Image3D ─────────────────────────────────────────────────────────

/// A 3D image with compile-time access mode and storage format.
///
/// `A` and `F` carry the same meaning as on [`Image2D`]. The
/// underlying OpenCL object is created with
/// `CL_MEM_OBJECT_IMAGE3D`. Use this when the kernel side declares
/// `Image!(3D, type=..., sampled=...)`.
pub struct Image3D<A: KernelAccess, F: format::Format> {
    image: Image,
    width: u32,
    height: u32,
    depth: u32,
    #[allow(dead_code)]
    ctx: Context,
    _access: PhantomData<A>,
    _format: PhantomData<F>,
}

impl<A: KernelAccess, F: format::Format> Image3D<A, F> {
    /// Allocate a `width × height × depth` 3D image. Pure context op.
    pub fn alloc(ctx: &Context, width: u32, height: u32, depth: u32) -> Result<Self> {
        let image = alloc_image::<A, F>(
            ctx,
            CL_MEM_OBJECT_IMAGE3D,
            width as usize,
            height as usize,
            depth as usize,
            0,
        )?;
        Ok(Self {
            image,
            width,
            height,
            depth,
            ctx: ctx.clone(),
            _access: PhantomData,
            _format: PhantomData,
        })
    }

    /// See [`Image2D::read`] — same shape, 3D region.
    pub fn read<'a>(self, dst: &'a mut [F::Pixel]) -> Result<ImageRead<'a, Self, F::Pixel>>
    where
        Self: ImageEnqueue,
        F::Pixel: Send,
    {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        let region = self.enqueue_region();
        image_read_op(self, region, dst, pixel_count, "Image3D")
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(self, dst: &'a mut [u8]) -> Result<ImageRead<'a, Self, u8>>
    where
        Self: ImageEnqueue,
    {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_read_bytes_op(self, region, dst, expected, "Image3D")
    }

    /// See [`Image2D::read_alloc`].
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        ImageReadAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [
                self.width as usize,
                self.height as usize,
                self.depth as usize,
            ],
            pixel_count,
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::read_bytes_alloc`].
    pub fn read_bytes_alloc(&self) -> ImageReadBytesAlloc<'_, F> {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        ImageReadBytesAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [
                self.width as usize,
                self.height as usize,
                self.depth as usize,
            ],
            byte_len: pixel_count * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::write`].
    pub fn write<'a>(self, pixels: &'a [F::Pixel]) -> ImageWrite<'a, Self, F::Pixel>
    where
        Self: ImageEnqueue,
        F::Pixel: Send + Sync,
    {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        let region = self.enqueue_region();
        image_write_op(self, region, pixels, pixel_count, "Image3D")
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(self, bytes: &'a [u8]) -> ImageWrite<'a, Self, u8>
    where
        Self: ImageEnqueue,
    {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_write_bytes_op(self, region, bytes, expected, "Image3D")
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<A2: KernelAccess + Send + 'static>(
        self,
        dst: Image3D<A2, F>,
    ) -> ImageCopy<Self, Image3D<A2, F>>
    where
        Self: ImageEnqueue,
        F: Send + 'static,
    {
        let region = self.enqueue_region();
        image_copy_op(self, dst, region)
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy + Send + 'static>(self, pattern: [T; 4]) -> ImageFill<Self, T>
    where
        Self: ImageEnqueue,
    {
        let region = self.enqueue_region();
        image_fill_op(self, region, pattern)
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Depth in pixels.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Borrow the underlying opencl3 [`Image`].
    pub fn image(&self) -> &Image {
        &self.image
    }
}

impl<A: KernelAccess, F: format::Format> KernelArg for Image3D<A, F> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let cl_mem_handle = self.image.get();
        unsafe {
            exec.set_arg(&cl_mem_handle);
        }
    }
}

// ── KernelImage*Arg traits ──────────────────────────────────────────
//
// Per-dim + per-access trait family. Each variant pins the
// kernel-side access qualifier the kernel declared via `&`/`&mut`
// (plus optional `#[spirv(image_access = ...)]` override on the
// rust-gpu side); the proc-macro on this side picks the right
// trait variant to bound the wrapper's image parameter on.
//
// Three peer traits per dim — exact-access match, no inheritance:
//   `KernelImage<dim>DReadArg<SF>`      — kernel declared ReadOnly
//   `KernelImage<dim>DWriteArg<SF>`     — kernel declared WriteOnly
//   `KernelImage<dim>DReadWriteArg<SF>` — kernel declared ReadWrite
//
// Each is parameterised on a [`format::SampledTypeFamily`] marker
// (`Uint`/`Sint`/`Float`) rather than a concrete `F`. Rationale:
// OpenCL Kernel images carry only sampled-type info at compile time
// (`OpTypeImage` `Image Format = Unknown`); a `type=u32` kernel
// can be paired with any uint-family host storage format
// (`R8G8B8A8Uint`, `R32Uint`, `R32G32B32A32Uint`, …) and the
// runtime translates. Parameterising on family rather than F keeps
// that flexibility while still type-checking the family match.
//
// Why exact-access (not subset):
//   - `clGetKernelArgInfo` returns exactly one of READ_ONLY /
//     WRITE_ONLY / READ_WRITE / NONE; drivers reject mismatched
//     `cl_mem` access flags at `clSetKernelArg` time.
//   - OpenCL C spec forbids sampler-based reads on `read_write`
//     images, so `ReadWrite` is not a strict superset of `ReadOnly`
//     for kernels that might sample.
// Each access marker on the host side maps to exactly one trait
// variant.
//
// All extend [`KernelArg`] so the underlying `clSetKernelArg`
// plumbing is reused, and are sealed in this crate.

mod kernel_image_arg_sealed {
    pub trait Sealed {}
}

// `Image<dim>D<A, F>` is `Send` when both `A: Send` and `F: Send`
// (PhantomData propagation). All access + format marker ZSTs in
// this crate impl `Send` via their `#[derive(Clone, Copy, Debug)]`,
// but the bound has to appear explicitly here for the trait impls
// to satisfy the supertrait `Send + 'static`.

/// Host-side counterpart for a kernel `&Image!(1D, type=...)`
/// parameter (kernel declared `ReadOnly`).
///
/// Implemented only by `Image1D<ReadOnly, F>` where `F`'s sampled
/// family matches the kernel-side `type=` keyword. Other access
/// markers (`WriteOnly`, `ReadWrite`) are intentionally rejected —
/// see the section comment above for the "exact-access" rationale.
pub trait KernelImage1DReadArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&mut Image!(1D, ...,
/// access="write_only")` parameter.
///
/// Implemented only by `Image1D<WriteOnly, F>` where `F`'s sampled
/// family matches. WriteOnly host images can't be read on the
/// kernel side, but they don't need `ImageReadWrite` capability —
/// the right choice for OpenCL 1.2 output kernels.
pub trait KernelImage1DWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&mut Image!(1D, ...)`
/// parameter (kernel declared `ReadWrite` — default for `&mut`).
///
/// Implemented only by `Image1D<ReadWrite, F>` where `F`'s sampled
/// family matches. Requires `ImageReadWrite` capability + OpenCL
/// 2.0+ device support; the rust-gpu codegen auto-declares the
/// capability when emitting any `ReadWrite OpTypeImage`.
pub trait KernelImage1DReadWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&Image!(2D, type=...)`
/// parameter — see [`KernelImage1DReadArg`] for details.
pub trait KernelImage2DReadArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&mut Image!(2D, ...,
/// access="write_only")` parameter — see [`KernelImage1DWriteArg`].
pub trait KernelImage2DWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&mut Image!(2D, ...)`
/// parameter — see [`KernelImage1DReadWriteArg`].
pub trait KernelImage2DReadWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&Image!(buffer, type=...)`
/// parameter — see [`KernelImage1DReadArg`].
///
/// Implemented by [`Image1DBuffer<ReadOnly, F>`] and
/// [`Image1DBuffer<ReadWrite, F>`]. The kernel-side
/// `image1d_buffer_t` reads/writes typed pixels backed by a
/// `cl_mem` buffer object — `clCreateImage` with
/// `CL_MEM_OBJECT_IMAGE1D_BUFFER` shares storage with the buffer
/// it was created from, so the same data can be read as a typed
/// 1D image from a kernel and as raw bytes (or typed elements)
/// through the buffer API.
pub trait KernelImageBufferReadArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&mut Image!(buffer, ...,
/// access="write_only")` parameter — see
/// [`KernelImageBufferReadArg`] for the storage model.
pub trait KernelImageBufferWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&mut Image!(buffer, ...)`
/// parameter (kernel declared `ReadWrite`) — see
/// [`KernelImageBufferReadArg`].
pub trait KernelImageBufferReadWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&Image!(3D, type=...)`
/// parameter — see [`KernelImage1DReadArg`].
pub trait KernelImage3DReadArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&mut Image!(3D, ...,
/// access="write_only")` parameter — see [`KernelImage1DWriteArg`].
pub trait KernelImage3DWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel `&mut Image!(3D, ...)`
/// parameter — see [`KernelImage1DReadWriteArg`].
pub trait KernelImage3DReadWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

// 1D-array and 2D-array trait families — mirror the dim-1/2/3
// pattern. Kernel-side coord is one component wider than the
// non-arrayed form: `Image!(1D, arrayed=true, ...)` takes
// `IVec2(x, layer)`, `Image!(2D, arrayed=true, ...)` takes
// `IVec3(x, y, layer)`. Host-side region for upload/download
// substitutes the array_size dimension for height (1D-array) or
// depth (2D-array) per the OpenCL spec.

/// Host-side counterpart for a kernel
/// `&Image!(1D, arrayed=true, type=...)` parameter — see
/// [`KernelImage1DReadArg`].
pub trait KernelImage1DArrayReadArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel
/// `&mut Image!(1D, arrayed=true, ..., access="write_only")`
/// parameter — see [`KernelImage1DWriteArg`].
pub trait KernelImage1DArrayWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel
/// `&mut Image!(1D, arrayed=true, ...)` parameter — see
/// [`KernelImage1DReadWriteArg`].
pub trait KernelImage1DArrayReadWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel
/// `&Image!(2D, arrayed=true, type=...)` parameter — see
/// [`KernelImage1DReadArg`].
pub trait KernelImage2DArrayReadArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel
/// `&mut Image!(2D, arrayed=true, ..., access="write_only")`
/// parameter — see [`KernelImage1DWriteArg`].
pub trait KernelImage2DArrayWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

/// Host-side counterpart for a kernel
/// `&mut Image!(2D, arrayed=true, ...)` parameter — see
/// [`KernelImage1DReadWriteArg`].
pub trait KernelImage2DArrayReadWriteArg<SF: format::SampledTypeFamily>:
    KernelArg + Send + kernel_image_arg_sealed::Sealed
{
}

// ── Sealed marker impls (one per (Image<dim>D, A, F) combo) ────────
//
// `kernel_image_arg_sealed::Sealed` is required by every
// `KernelImage<dim>D*Arg` trait. We blanket-impl it on every
// concrete `Image<dim>D<A, F>` regardless of access marker, since
// the access-specific gating happens on the per-access trait
// impls below.

impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static>
    kernel_image_arg_sealed::Sealed for Image1D<A, F>
{
}
impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static>
    kernel_image_arg_sealed::Sealed for Image2D<A, F>
{
}
impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static>
    kernel_image_arg_sealed::Sealed for Image3D<A, F>
{
}

// ── Per-(dim, access) impls ──────────────────────────────────────
//
// Each access marker (`ReadOnly`/`WriteOnly`/`ReadWrite`) impls
// one or more trait variants per dim, parameterised on `F`'s
// `SampledFamily` so the proc-macro's
// `<F: format::Format<SampledFamily = K>>` bound on the wrapper
// method picks the right impl per kernel `type=` keyword.
//
// Compatibility partial order (per OpenCL `clSetKernelArg` rules):
//   - `ReadOnly`  host image → satisfies `Read` kernel arg only
//   - `WriteOnly` host image → satisfies `Write` kernel arg only
//   - `ReadWrite` host image → satisfies all three (`Read`,
//     `Write`, `ReadWrite`) — the host promises the cl_mem can be
//     bound to any kernel access qualifier; the runtime constrains
//     only "writing to CL_MEM_READ_ONLY is undefined" and "reading
//     from CL_MEM_WRITE_ONLY is undefined", neither of which fires
//     when ReadWrite is the host flag.
//
// This three-way pattern lets a single `Image2D<ReadWrite, F>` flow
// through a pipeline that mixes write-only producer kernels and
// read-only consumer kernels — the common image-pipeline case —
// without intermediate retype operations on the cl_mem.

// Image1D
impl<F: format::Format + Send + 'static> KernelImage1DReadArg<F::SampledFamily>
    for Image1D<ReadOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage1DWriteArg<F::SampledFamily>
    for Image1D<WriteOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage1DReadArg<F::SampledFamily>
    for Image1D<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage1DWriteArg<F::SampledFamily>
    for Image1D<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage1DReadWriteArg<F::SampledFamily>
    for Image1D<ReadWrite, F>
{
}

// Image2D
impl<F: format::Format + Send + 'static> KernelImage2DReadArg<F::SampledFamily>
    for Image2D<ReadOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage2DWriteArg<F::SampledFamily>
    for Image2D<WriteOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage2DReadArg<F::SampledFamily>
    for Image2D<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage2DWriteArg<F::SampledFamily>
    for Image2D<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage2DReadWriteArg<F::SampledFamily>
    for Image2D<ReadWrite, F>
{
}

// Image3D
impl<F: format::Format + Send + 'static> KernelImage3DReadArg<F::SampledFamily>
    for Image3D<ReadOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage3DWriteArg<F::SampledFamily>
    for Image3D<WriteOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage3DReadArg<F::SampledFamily>
    for Image3D<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage3DWriteArg<F::SampledFamily>
    for Image3D<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage3DReadWriteArg<F::SampledFamily>
    for Image3D<ReadWrite, F>
{
}

// ── Image1DArray ─────────────────────────────────────────────────
//
// A stack of N independent 1D images with the same format. The
// kernel addresses one as `image1d_array_t` with an `IVec2(x,
// layer)` coordinate. Created with `CL_MEM_OBJECT_IMAGE1D_ARRAY`;
// the host-side `region` for upload/download is `[width,
// array_size, 1]` (the array_size dimension takes the slot
// height occupies for 2D).

/// A 1D image array with compile-time access mode and storage
/// format. `array_size` layers of `width` pixels each.
///
/// Use this when the kernel side declares
/// `Image!(1D, arrayed=true, type=..., sampled=...)`. The
/// kernel-side coordinate is `IVec2(x, layer)`.
pub struct Image1DArray<A: KernelAccess, F: format::Format> {
    image: Image,
    width: u32,
    array_size: u32,
    #[allow(dead_code)]
    ctx: Context,
    _access: PhantomData<A>,
    _format: PhantomData<F>,
}

impl<A: KernelAccess, F: format::Format> Image1DArray<A, F> {
    /// Allocate a `width × array_size` image array. Pure context op.
    pub fn alloc(ctx: &Context, width: u32, array_size: u32) -> Result<Self> {
        let image = alloc_image::<A, F>(
            ctx,
            CL_MEM_OBJECT_IMAGE1D_ARRAY,
            width as usize,
            1,
            1,
            array_size as usize,
        )?;
        Ok(Self {
            image,
            width,
            array_size,
            ctx: ctx.clone(),
            _access: PhantomData,
            _format: PhantomData,
        })
    }

    /// See [`Image2D::read`] — region is `[width, array_size, 1]`
    /// per OpenCL spec; layers are laid out contiguously: layer-0
    /// first, then layer-1, etc.
    pub fn read<'a>(self, dst: &'a mut [F::Pixel]) -> Result<ImageRead<'a, Self, F::Pixel>>
    where
        Self: ImageEnqueue,
        F::Pixel: Send,
    {
        let pixel_count = (self.width as usize) * (self.array_size as usize);
        let region = self.enqueue_region();
        image_read_op(self, region, dst, pixel_count, "Image1DArray")
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(self, dst: &'a mut [u8]) -> Result<ImageRead<'a, Self, u8>>
    where
        Self: ImageEnqueue,
    {
        let pixel_count = (self.width as usize) * (self.array_size as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_read_bytes_op(self, region, dst, expected, "Image1DArray")
    }

    /// See [`Image2D::read_alloc`].
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        ImageReadAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [self.width as usize, self.array_size as usize, 1],
            pixel_count: (self.width as usize) * (self.array_size as usize),
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::read_bytes_alloc`].
    pub fn read_bytes_alloc(&self) -> ImageReadBytesAlloc<'_, F> {
        let pixel_count = (self.width as usize) * (self.array_size as usize);
        ImageReadBytesAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [self.width as usize, self.array_size as usize, 1],
            byte_len: pixel_count * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::write`].
    pub fn write<'a>(self, pixels: &'a [F::Pixel]) -> ImageWrite<'a, Self, F::Pixel>
    where
        Self: ImageEnqueue,
        F::Pixel: Send + Sync,
    {
        let pixel_count = (self.width as usize) * (self.array_size as usize);
        let region = self.enqueue_region();
        image_write_op(self, region, pixels, pixel_count, "Image1DArray")
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(self, bytes: &'a [u8]) -> ImageWrite<'a, Self, u8>
    where
        Self: ImageEnqueue,
    {
        let pixel_count = (self.width as usize) * (self.array_size as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_write_bytes_op(self, region, bytes, expected, "Image1DArray")
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<A2: KernelAccess + Send + 'static>(
        self,
        dst: Image1DArray<A2, F>,
    ) -> ImageCopy<Self, Image1DArray<A2, F>>
    where
        Self: ImageEnqueue,
        F: Send + 'static,
    {
        let region = self.enqueue_region();
        image_copy_op(self, dst, region)
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy + Send + 'static>(self, pattern: [T; 4]) -> ImageFill<Self, T>
    where
        Self: ImageEnqueue,
    {
        let region = self.enqueue_region();
        image_fill_op(self, region, pattern)
    }

    /// Width in pixels (per layer).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Number of layers.
    pub fn array_size(&self) -> u32 {
        self.array_size
    }

    /// Borrow the underlying opencl3 [`Image`].
    pub fn image(&self) -> &Image {
        &self.image
    }
}

impl<A: KernelAccess, F: format::Format> KernelArg for Image1DArray<A, F> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let cl_mem_handle = self.image.get();
        unsafe {
            exec.set_arg(&cl_mem_handle);
        }
    }
}

impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static>
    kernel_image_arg_sealed::Sealed for Image1DArray<A, F>
{
}

impl<F: format::Format + Send + 'static> KernelImage1DArrayReadArg<F::SampledFamily>
    for Image1DArray<ReadOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage1DArrayWriteArg<F::SampledFamily>
    for Image1DArray<WriteOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage1DArrayReadArg<F::SampledFamily>
    for Image1DArray<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage1DArrayWriteArg<F::SampledFamily>
    for Image1DArray<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage1DArrayReadWriteArg<F::SampledFamily>
    for Image1DArray<ReadWrite, F>
{
}

// ── Image2DArray ─────────────────────────────────────────────────
//
// A stack of N independent 2D images with the same format. The
// kernel addresses one as `image2d_array_t` with an `IVec3(x, y,
// layer)` coordinate. Created with `CL_MEM_OBJECT_IMAGE2D_ARRAY`;
// the host-side `region` for upload/download is `[width, height,
// array_size]`.

/// A 2D image array with compile-time access mode and storage
/// format. `array_size` layers of `width × height` pixels each.
///
/// Use this when the kernel side declares
/// `Image!(2D, arrayed=true, type=..., sampled=...)`. The
/// kernel-side coordinate is `IVec3(x, y, layer)`.
pub struct Image2DArray<A: KernelAccess, F: format::Format> {
    image: Image,
    width: u32,
    height: u32,
    array_size: u32,
    #[allow(dead_code)]
    ctx: Context,
    _access: PhantomData<A>,
    _format: PhantomData<F>,
}

impl<A: KernelAccess, F: format::Format> Image2DArray<A, F> {
    /// Allocate a `width × height × array_size` image array.
    /// Pure context op.
    pub fn alloc(ctx: &Context, width: u32, height: u32, array_size: u32) -> Result<Self> {
        let image = alloc_image::<A, F>(
            ctx,
            CL_MEM_OBJECT_IMAGE2D_ARRAY,
            width as usize,
            height as usize,
            1,
            array_size as usize,
        )?;
        Ok(Self {
            image,
            width,
            height,
            array_size,
            ctx: ctx.clone(),
            _access: PhantomData,
            _format: PhantomData,
        })
    }

    /// See [`Image2D::read`] — 2D-array region is
    /// `[width, height, array_size]`.
    pub fn read<'a>(self, dst: &'a mut [F::Pixel]) -> Result<ImageRead<'a, Self, F::Pixel>>
    where
        Self: ImageEnqueue,
        F::Pixel: Send,
    {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        let region = self.enqueue_region();
        image_read_op(self, region, dst, pixel_count, "Image2DArray")
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(self, dst: &'a mut [u8]) -> Result<ImageRead<'a, Self, u8>>
    where
        Self: ImageEnqueue,
    {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_read_bytes_op(self, region, dst, expected, "Image2DArray")
    }

    /// See [`Image2D::read_alloc`].
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        ImageReadAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [
                self.width as usize,
                self.height as usize,
                self.array_size as usize,
            ],
            pixel_count,
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::read_bytes_alloc`].
    pub fn read_bytes_alloc(&self) -> ImageReadBytesAlloc<'_, F> {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        ImageReadBytesAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [
                self.width as usize,
                self.height as usize,
                self.array_size as usize,
            ],
            byte_len: pixel_count * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::write`].
    pub fn write<'a>(self, pixels: &'a [F::Pixel]) -> ImageWrite<'a, Self, F::Pixel>
    where
        Self: ImageEnqueue,
        F::Pixel: Send + Sync,
    {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        let region = self.enqueue_region();
        image_write_op(self, region, pixels, pixel_count, "Image2DArray")
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(self, bytes: &'a [u8]) -> ImageWrite<'a, Self, u8>
    where
        Self: ImageEnqueue,
    {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_write_bytes_op(self, region, bytes, expected, "Image2DArray")
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<A2: KernelAccess + Send + 'static>(
        self,
        dst: Image2DArray<A2, F>,
    ) -> ImageCopy<Self, Image2DArray<A2, F>>
    where
        Self: ImageEnqueue,
        F: Send + 'static,
    {
        let region = self.enqueue_region();
        image_copy_op(self, dst, region)
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy + Send + 'static>(self, pattern: [T; 4]) -> ImageFill<Self, T>
    where
        Self: ImageEnqueue,
    {
        let region = self.enqueue_region();
        image_fill_op(self, region, pattern)
    }

    /// Width in pixels (per layer).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels (per layer).
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Number of layers.
    pub fn array_size(&self) -> u32 {
        self.array_size
    }

    /// Borrow the underlying opencl3 [`Image`].
    pub fn image(&self) -> &Image {
        &self.image
    }
}

impl<A: KernelAccess, F: format::Format> KernelArg for Image2DArray<A, F> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let cl_mem_handle = self.image.get();
        unsafe {
            exec.set_arg(&cl_mem_handle);
        }
    }
}

impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static>
    kernel_image_arg_sealed::Sealed for Image2DArray<A, F>
{
}

impl<F: format::Format + Send + 'static> KernelImage2DArrayReadArg<F::SampledFamily>
    for Image2DArray<ReadOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage2DArrayWriteArg<F::SampledFamily>
    for Image2DArray<WriteOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage2DArrayReadArg<F::SampledFamily>
    for Image2DArray<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage2DArrayWriteArg<F::SampledFamily>
    for Image2DArray<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImage2DArrayReadWriteArg<F::SampledFamily>
    for Image2DArray<ReadWrite, F>
{
}

// ── Image1DBuffer ─────────────────────────────────────────────────
//
// Image view of an OpenCL buffer object: the kernel sees an
// `image1d_buffer_t` (SPIR-V `OpTypeImage Dim=Buffer`), the host
// sees a `cl_mem` buffer. Storage is shared — host can use
// `clEnqueueRead/WriteBuffer` against the same `cl_mem` to access
// the same bytes the kernel reads/writes as typed pixels.
//
// Why use it over a normal `DeviceSlice<T>` slice arg: the kernel
// gets format-aware access (e.g. UNORM normalisation, sint/uint
// channel-type interpretation) with hardware support and no
// per-element conversion code. Why use it over [`Image1D`]: the
// max width is `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` (typically GB-
// scale) rather than `CL_DEVICE_IMAGE2D_MAX_*` (typically MB-
// scale).
//
// Created with `CL_MEM_OBJECT_IMAGE1D_BUFFER`. The
// `cl_image_desc::buffer` field of the desc passed to
// `clCreateImage` is the underlying `cl_mem`; we own it
// internally via [`opencl3::memory::Buffer`] so the image and
// its storage drop together.

/// A 1D image-buffer with compile-time access mode and storage
/// format — backs a kernel-side `image1d_buffer_t`
/// (`Image!(buffer, ...)`).
///
/// `A` is the kernel-side access marker; `F` is the channel
/// format. The underlying OpenCL object is created with
/// `CL_MEM_OBJECT_IMAGE1D_BUFFER` over an internally-owned
/// `cl_mem` buffer (one allocation per image-buffer).
///
/// Use this when you want format-aware kernel access (hardware
/// UNORM↔float conversion, etc.) over a 1D-indexed dataset with
/// the buffer-size max rather than the 2D-image-size max.
pub struct Image1DBuffer<A: KernelAccess, F: format::Format> {
    image: Image,
    // `ClBuffer<u8>` — the storage backing the image. We hold it
    // so it stays alive as long as the image-buffer does; OpenCL
    // retains its own ref via the image-create, but we keep an
    // explicit owner here for symmetry with the per-dim wrappers.
    #[allow(dead_code)]
    backing: opencl3::memory::Buffer<u8>,
    width: u32,
    #[allow(dead_code)]
    ctx: Context,
    _access: PhantomData<A>,
    _format: PhantomData<F>,
}

impl<A: KernelAccess, F: format::Format> Image1DBuffer<A, F> {
    /// Allocate an image-buffer of `width` pixels — also allocates
    /// the backing `cl_mem` buffer internally (size `width *
    /// size_of::<F::Pixel>()` bytes).
    pub fn alloc(ctx: &Context, width: u32) -> Result<Self> {
        let pixel_bytes = std::mem::size_of::<F::Pixel>();
        let byte_len = (width as usize) * pixel_bytes;
        // SAFETY: null host pointer + `KERNEL_FLAGS` access flag —
        // OpenCL allocates fresh device memory and ignores the
        // host-pointer contract that makes `Buffer::create` unsafe.
        let backing: opencl3::memory::Buffer<u8> = unsafe {
            opencl3::memory::Buffer::<u8>::create(
                ctx.raw_context(),
                A::KERNEL_FLAGS,
                byte_len,
                ptr::null_mut(),
            )?
        };
        let format = cl_image_format {
            image_channel_order: F::CHANNEL_ORDER,
            image_channel_data_type: F::CHANNEL_TYPE,
        };
        let desc = cl_image_desc {
            image_type: opencl3::memory::CL_MEM_OBJECT_IMAGE1D_BUFFER,
            image_width: width as usize,
            image_height: 0,
            image_depth: 0,
            image_array_size: 0,
            image_row_pitch: 0,
            image_slice_pitch: 0,
            num_mip_levels: 0,
            num_samples: 0,
            buffer: backing.get(),
        };
        // SAFETY: `buffer` in the desc points at the backing
        // `cl_mem` we just allocated; OpenCL retains it and the
        // image shares its storage. Host-ptr null is correct
        // because we're not initialising from a host buffer.
        let image = unsafe {
            Image::create(
                ctx.raw_context(),
                A::KERNEL_FLAGS,
                &format,
                &desc,
                ptr::null_mut(),
            )?
        };
        Ok(Self {
            image,
            backing,
            width,
            ctx: ctx.clone(),
            _access: PhantomData,
            _format: PhantomData,
        })
    }

    /// See [`Image2D::read`] — image-buffer goes through the image
    /// path (`clEnqueueReadImage`), region is `[width, 1, 1]`.
    pub fn read<'a>(self, dst: &'a mut [F::Pixel]) -> Result<ImageRead<'a, Self, F::Pixel>>
    where
        Self: ImageEnqueue,
        F::Pixel: Send,
    {
        let pixel_count = self.width as usize;
        let region = self.enqueue_region();
        image_read_op(self, region, dst, pixel_count, "Image1DBuffer")
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(self, dst: &'a mut [u8]) -> Result<ImageRead<'a, Self, u8>>
    where
        Self: ImageEnqueue,
    {
        let expected = (self.width as usize) * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_read_bytes_op(self, region, dst, expected, "Image1DBuffer")
    }

    /// See [`Image2D::read_alloc`].
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        ImageReadAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [self.width as usize, 1, 1],
            pixel_count: self.width as usize,
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::read_bytes_alloc`].
    pub fn read_bytes_alloc(&self) -> ImageReadBytesAlloc<'_, F> {
        ImageReadBytesAlloc {
            image: &self.image,
            ctx: &self.ctx,
            region: [self.width as usize, 1, 1],
            byte_len: (self.width as usize) * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::write`].
    pub fn write<'a>(self, pixels: &'a [F::Pixel]) -> ImageWrite<'a, Self, F::Pixel>
    where
        Self: ImageEnqueue,
        F::Pixel: Send + Sync,
    {
        let pixel_count = self.width as usize;
        let region = self.enqueue_region();
        image_write_op(self, region, pixels, pixel_count, "Image1DBuffer")
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(self, bytes: &'a [u8]) -> ImageWrite<'a, Self, u8>
    where
        Self: ImageEnqueue,
    {
        let expected = (self.width as usize) * std::mem::size_of::<F::Pixel>();
        let region = self.enqueue_region();
        image_write_bytes_op(self, region, bytes, expected, "Image1DBuffer")
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<A2: KernelAccess + Send + 'static>(
        self,
        dst: Image1DBuffer<A2, F>,
    ) -> ImageCopy<Self, Image1DBuffer<A2, F>>
    where
        Self: ImageEnqueue,
        F: Send + 'static,
    {
        let region = self.enqueue_region();
        image_copy_op(self, dst, region)
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy + Send + 'static>(self, pattern: [T; 4]) -> ImageFill<Self, T>
    where
        Self: ImageEnqueue,
    {
        let region = self.enqueue_region();
        image_fill_op(self, region, pattern)
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Borrow the underlying opencl3 [`Image`].
    pub fn image(&self) -> &Image {
        &self.image
    }
}

impl<A: KernelAccess, F: format::Format> KernelArg for Image1DBuffer<A, F> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let cl_mem_handle = self.image.get();
        unsafe {
            exec.set_arg(&cl_mem_handle);
        }
    }
}

impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static>
    kernel_image_arg_sealed::Sealed for Image1DBuffer<A, F>
{
}

// Per-access impls for Image1DBuffer — same partial order as
// the dim-1/2/3 wrappers above.
impl<F: format::Format + Send + 'static> KernelImageBufferReadArg<F::SampledFamily>
    for Image1DBuffer<ReadOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImageBufferWriteArg<F::SampledFamily>
    for Image1DBuffer<WriteOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImageBufferReadArg<F::SampledFamily>
    for Image1DBuffer<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImageBufferWriteArg<F::SampledFamily>
    for Image1DBuffer<ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImageBufferReadWriteArg<F::SampledFamily>
    for Image1DBuffer<ReadWrite, F>
{
}

// ── Image1DBufferView ─────────────────────────────────────────────
//
// A borrowed image-buffer view over an existing [`DeviceSlice`]:
// the kernel sees an `image1d_buffer_t`, the host holds the slice;
// both refer to the same `cl_mem`. No extra allocation — the
// view's image is created with `desc.buffer = slice.cl_mem`, and
// OpenCL retains the slice's `cl_mem` until the view drops.
//
// Lifetime: `'a` ties the view to a borrow of the slice. As long
// as the view exists, the slice cannot be moved (Rust borrow
// rule); when the view drops, `Image::drop` releases its retain on
// the slice's `cl_mem` and the slice's own retain remains.
//
// Why this is interesting: it lets a single allocation appear as
// both a typed buffer and a format-aware 1D image in the same
// pipeline — write to it with a normal `&mut [T]` kernel arg,
// then read it from a different kernel as an `image1d_buffer_t`
// to get hardware UNORM↔float conversion. No copies, no
// retypes.
//
// Aliasing caveat: like passing the same `DeviceSlice` twice as
// `&mut [T]` args to one launch, holding a view while also
// passing the slice itself to a kernel that writes through it is
// UB at the OpenCL level. claspr doesn't enforce the constraint
// — Rust borrow rules don't track device-side mutation. The user
// is expected to sequence reads + writes through queue
// dependencies (or just not alias).
//
// The view's kernel-access marker is *derived* from the slice's
// `MemMode` (since `MemMode: KernelAccess`), not chosen freely —
// the cl_mem was allocated with the slice's `KERNEL_FLAGS`, and
// the image inherits that.

/// A 1D-image-buffer view over an existing [`DeviceSlice`] —
/// shares storage with the slice, no copy.
///
/// The kernel-side access marker is whatever the slice's
/// `MemMode` resolved to (`MemMode: KernelAccess`), so a
/// `DeviceSlice<f32, ReadWrite>` yields a view that satisfies
/// any of the `KernelImageBuffer*Arg` trait variants; a
/// `DeviceSlice<f32, ReadOnly>` only satisfies
/// `KernelImageBufferReadArg`.
///
/// Width is derived from the slice's byte length and the
/// format's pixel size — the constructor errors if the byte
/// length isn't an exact multiple of `size_of::<F::Pixel>()`.
pub struct Image1DBufferView<'a, M: MemMode, F: format::Format> {
    image: Image,
    width: u32,
    // PhantomData carries (a) the borrow on the slice via `&'a ()`
    // so the view can't outlive the slice, (b) the slice's
    // `MemMode` so the trait impls below can match on M, and (c)
    // the format so trait dispatch picks the right `SampledFamily`.
    _borrow: PhantomData<&'a ()>,
    _mode: PhantomData<M>,
    _format: PhantomData<F>,
}

impl<'a, M: MemMode, F: format::Format> Image1DBufferView<'a, M, F> {
    /// View `slice` as an image-buffer with format `F`. The
    /// kernel-side access is `M`'s `KernelAccess` (i.e. the same
    /// `CL_MEM_READ_ONLY`/`READ_WRITE`/`WRITE_ONLY` the slice was
    /// allocated with — OpenCL won't accept a different access on
    /// the view since the cl_mem is shared).
    ///
    /// Errors if `slice.byte_len() % size_of::<F::Pixel>() != 0` —
    /// a partial trailing pixel is rejected since the kernel-side
    /// indexing would silently read past the data.
    pub fn view_of<T>(slice: &'a DeviceSlice<T, M>) -> Result<Self> {
        let pixel_bytes = std::mem::size_of::<F::Pixel>();
        let byte_len = slice.len() * std::mem::size_of::<T>();
        assert_eq!(
            byte_len % pixel_bytes,
            0,
            "Image1DBufferView::view_of: slice byte length {} is not a multiple of pixel size {} \
             (format = {})",
            byte_len,
            pixel_bytes,
            std::any::type_name::<F>(),
        );
        let width = byte_len / pixel_bytes;
        let format = cl_image_format {
            image_channel_order: F::CHANNEL_ORDER,
            image_channel_data_type: F::CHANNEL_TYPE,
        };
        let desc = cl_image_desc {
            image_type: opencl3::memory::CL_MEM_OBJECT_IMAGE1D_BUFFER,
            image_width: width,
            image_height: 0,
            image_depth: 0,
            image_array_size: 0,
            image_row_pitch: 0,
            image_slice_pitch: 0,
            num_mip_levels: 0,
            num_samples: 0,
            buffer: slice.buffer().get(),
        };
        // SAFETY: `buffer` in desc points at the slice's `cl_mem`,
        // which Rust's borrow rule keeps alive for `'a`. OpenCL
        // additionally `clRetainMemObject`s it. Host-ptr null is
        // correct (no host-side init data — the slice IS the data).
        // Access flags match the slice's because the cl_mem was
        // created with `M::KERNEL_FLAGS`; passing a different
        // access flag here would yield CL_INVALID_VALUE.
        let image = unsafe {
            Image::create(
                slice.ctx().raw_context(),
                M::KERNEL_FLAGS,
                &format,
                &desc,
                ptr::null_mut(),
            )?
        };
        Ok(Self {
            image,
            width: width as u32,
            _borrow: PhantomData,
            _mode: PhantomData,
            _format: PhantomData,
        })
    }

    /// Width in pixels — derived from the slice's byte length and
    /// the format's pixel size.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Borrow the underlying opencl3 [`Image`].
    pub fn image(&self) -> &Image {
        &self.image
    }
}

impl<M: MemMode, F: format::Format> KernelArg for Image1DBufferView<'_, M, F> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let cl_mem_handle = self.image.get();
        unsafe {
            exec.set_arg(&cl_mem_handle);
        }
    }
}

// Sealed marker — same as the owned form. `'static` bound on M
// holds because all access markers (ReadOnly/WriteOnly/ReadWrite)
// are ZSTs that are themselves `'static`, but the view itself
// carries the `'a` lifetime via PhantomData<&'a ()>, so the
// sealed impl needs the explicit `'a` parameter.
impl<M: MemMode + Send, F: format::Format + Send + 'static> kernel_image_arg_sealed::Sealed
    for Image1DBufferView<'_, M, F>
{
}

// Per-(M, access-trait) impls. Same partial-order rules as the
// owned form, but bridged via M's KernelAccess marker:
//   - DeviceSlice<T, ReadOnly>  → view satisfies Read only
//   - DeviceSlice<T, WriteOnly> → view satisfies Write only
//   - DeviceSlice<T, ReadWrite> → view satisfies all three
//
// We list these as concrete-M impls (no blanket over generic M)
// because the trait family is access-discriminated, not
// MemMode-discriminated — coherence requires us to spell out the
// (M, trait) pairs.
impl<F: format::Format + Send + 'static> KernelImageBufferReadArg<F::SampledFamily>
    for Image1DBufferView<'_, ReadOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImageBufferWriteArg<F::SampledFamily>
    for Image1DBufferView<'_, WriteOnly, F>
{
}
impl<F: format::Format + Send + 'static> KernelImageBufferReadArg<F::SampledFamily>
    for Image1DBufferView<'_, ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImageBufferWriteArg<F::SampledFamily>
    for Image1DBufferView<'_, ReadWrite, F>
{
}
impl<F: format::Format + Send + 'static> KernelImageBufferReadWriteArg<F::SampledFamily>
    for Image1DBufferView<'_, ReadWrite, F>
{
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
