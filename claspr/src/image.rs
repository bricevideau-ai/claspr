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
use crate::eager::{
    CbCache, CbWalk, Checkout, Deps, DeviceOp, DeviceOpExt, ExecMode, Input, Pipe,
    cb_collect_external, cb_leaf_build, deps_to_wait_list, new_cb_cache, single_dep,
};
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
// The image access markers are the cross-cutting [`crate::access`] markers; this
// re-export keeps the `claspr::image::*` import paths and proc-macro-emitted
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
// Each helper is a `clEnqueue*Image` body the eager image graph nodes
// (`ImageWrite` / `ImageRead` / `ImageCopy` / `ImageFill` in this file) enqueue
// directly against an `Image`.
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
    img.with_concrete(|i| i.enqueue_ctx().clone())
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

    fn output_pipe(&self) -> Option<Pipe<I>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // In-place: the written image is the lent image → home threads through.
        let (mut img, deps, home) = self.img.resolve_home(ec)?;
        let raw = deps_to_wait_list(&deps);
        let data = self.data.as_ptr() as *const std::ffi::c_void;
        match mode {
            ExecMode::Blocking => {
                write_image_enqueue(img.image_mut(), ec, self.region, data, true, &raw)?;
                self.out.put_home(img, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                let event =
                    write_image_enqueue(img.image_mut(), ec, self.region, data, false, &raw)?;
                self.out.put_home(img, single_dep(event), home);
            }
        }
        Ok(())
    }

    /// Atomicity pre-pass mirror of the slice ops: read-only readiness of the
    /// lent image cell, so a busy/unsatisfiable image op is caught before any
    /// earlier lending op enqueues (see [`Input::check_ready`]).
    fn check_ready(&self) -> Result<()> {
        self.img.check_ready()
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
        self.sync(&ctx).map(Checkout::into_inner)
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
    // Behind a `Mutex` so `execute(&self)` can take the `&mut [E]` it reads into.
    dst: std::sync::Mutex<&'a mut [E]>,
    out: Pipe<I>,
}

impl<I: ImageEnqueue, E: Send> DeviceOp for ImageRead<'_, I, E> {
    type Output = I;

    fn output_pipe(&self) -> Option<Pipe<I>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // In-place: the read image is handed back unchanged → home threads.
        let (img, deps, home) = self.img.resolve_home(ec)?;
        let raw = deps_to_wait_list(&deps);
        let mut dst_guard = self.dst.lock().unwrap();
        let dst = dst_guard.as_mut_ptr() as *mut std::ffi::c_void;
        match mode {
            ExecMode::Blocking => {
                read_image_enqueue(img.image_ref(), ec, self.region, dst, true, &raw)?;
                self.out.put_home(img, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                let event = read_image_enqueue(img.image_ref(), ec, self.region, dst, false, &raw)?;
                self.out.put_home(img, single_dep(event), home);
            }
        }
        Ok(())
    }

    /// Atomicity pre-pass mirror of the slice ops: read-only readiness of the
    /// lent image cell, so a busy/unsatisfiable image op is caught before any
    /// earlier lending op enqueues (see [`Input::check_ready`]).
    fn check_ready(&self) -> Result<()> {
        self.img.check_ready()
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
        self.sync(&ctx).map(Checkout::into_inner)
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
    /// The copy extent. `Some` when a concrete-head `copy_to` computed it at build;
    /// `None` for a pipe-fed graph copy (`eager_image_copy`) where the src image
    /// isn't concrete at build — then it is derived at execute from the lent src.
    region: Option<[usize; 3]>,
    src_pipe: Pipe<Src>,
    dst_pipe: Pipe<Dst>,
    /// Design-v2 CB home: an image→image copy records `clCommandCopyImageKHR` where
    /// the extension provides it, else falls back to software.
    cb_cache: CbCache,
}

impl<Src: ImageEnqueue, Dst: ImageEnqueue> DeviceOp for ImageCopy<Src, Dst> {
    type Output = (Src, Dst);
    type Handle = (Pipe<Src>, Pipe<Dst>);
    type Checkouts = (crate::eager::Checkout<Src>, crate::eager::Checkout<Dst>);

    fn output_pipe(&self) -> Option<Pipe<(Src, Dst)>> {
        // Multi-output: the value is reconstructed in `collect` from the two
        // element pipes, never this single pipe (which stays empty). Mirrors the
        // buffer `CopyTo2` shape.
        None
    }

    fn handle(&self) -> Self::Handle {
        (self.src_pipe.clone(), self.dst_pipe.clone())
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        use opencl3::memory::ClMem;
        // In-place on both sides (image copy preserves type) → each home threads
        // to its own element pipe.
        let (src, src_deps, src_home) = self.src.resolve_home(ec)?;
        let (mut dst, dst_deps, dst_home) = self.dst.resolve_home(ec)?;
        // Region: from the build-time value (concrete `copy_to`) or derived at execute
        // from the lent src (pipe-fed `eager_image_copy`, no concrete src at build).
        let region = self.region.unwrap_or_else(|| src.enqueue_region());

        // ── CB-mode fork (design v2) — image→image copy via clCommandCopyImageKHR.
        // Absent PFN → ineligible → boundary falls back to per-op. ────────────────
        match ec.cb() {
            CbWalk::Off => {}
            CbWalk::Build { builder, ext, .. } => {
                // Per-operand `cb_leaf_build` (external deps + waits + the
                // precise-invalidation reach: note_slot origins + propagate onto the
                // output cell) — the SAME treatment the buffer `CopyTo2` gets. Using
                // it (not raw `cb_collect_external`/`sp_lookup`) is what makes a
                // `mutate_bind` of an image slot invalidate this recorded copy.
                let mut waits = cb_leaf_build(
                    ec,
                    builder,
                    ext,
                    &src_deps,
                    self.src.slot_cell_id(),
                    self.src.pipe_cell_id(),
                    self.src_pipe.cell_id(),
                );
                waits.extend(cb_leaf_build(
                    ec,
                    builder,
                    ext,
                    &dst_deps,
                    self.dst.slot_cell_id(),
                    self.dst.pipe_cell_id(),
                    self.dst_pipe.cell_id(),
                ));
                let smem = crate::record::MemRef::Buffer(src.image_ref().get());
                let dmem = crate::record::MemRef::Buffer(dst.image_ref().get());
                if let Some(sp) =
                    builder.copy_image(smem, dmem, [0, 0, 0], [0, 0, 0], region, &waits)
                {
                    // Multi-output: both output pipes gate on this one copy command.
                    let set = std::collections::BTreeSet::from([sp]);
                    ec.sp_register(self.src_pipe.cell_id(), set.clone());
                    ec.sp_register(self.dst_pipe.cell_id(), set);
                }
                self.src_pipe.put_home(src, Deps::new(), src_home);
                self.dst_pipe.put_home(dst, Deps::new(), dst_home);
                return Ok(());
            }
            CbWalk::LendOnly { ext, .. } => {
                cb_collect_external(ext, &src_deps);
                cb_collect_external(ext, &dst_deps);
                self.src_pipe.put_home(src, Deps::new(), src_home);
                self.dst_pipe.put_home(dst, Deps::new(), dst_home);
                return Ok(());
            }
        }

        let mut merged = src_deps.clone();
        merged.extend(dst_deps.iter().cloned());
        let raw = deps_to_wait_list(&merged);
        // Copy has no native CL_BLOCKING flag — always enqueue non-blocking; a
        // blocking terminal waits on the event via the carried deps.
        let event = copy_image_enqueue(src.image_ref(), dst.image_mut(), ec, region, &raw)?;
        let dep = single_dep(event);
        self.src_pipe.put_home(src, dep.clone(), src_home);
        self.dst_pipe.put_home(dst, dep, dst_home);
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<((Src, Dst), Deps)> {
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

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // Drain each element pipe with its own home → a tuple of independent
        // Checkouts (image copy preserves type, so both homes re-arm correctly).
        let src_pipe = self.src_pipe.clone();
        let dst_pipe = self.dst_pipe.clone();
        self.execute(ec, mode)?;
        let (src, src_deps, src_home) = src_pipe.take_home().ok_or(Error::NotSupported(
            "eager graph: image copy produced no src",
        ))?;
        let (dst, mut deps, dst_home) = dst_pipe.take_home().ok_or(Error::NotSupported(
            "eager graph: image copy produced no dst",
        ))?;
        deps.extend(src_deps);
        Ok((
            (
                crate::eager::Checkout::new(src, src_home),
                crate::eager::Checkout::new(dst, dst_home),
            ),
            deps,
        ))
    }

    /// Atomicity pre-pass mirror of the slice ops: read-only readiness of BOTH
    /// lent image cells (src then dst), so a busy/unsatisfiable operand is caught
    /// before any earlier lending op enqueues (see [`Input::check_ready`]).
    fn check_ready(&self) -> Result<()> {
        self.src.check_ready()?;
        self.dst.check_ready()
    }

    fn bind_slots(&self, binder: &mut crate::SlotBinder) {
        // src/dst may each be a `slot!()` operand; offer the binder to both (execution
        // order: src then dst), short-circuiting once it lands. Non-slot (concrete /
        // pipe) inputs are a no-op in `try_bind_slot`. Mirrors the buffer `CopyTo2`.
        self.src.try_bind_slot(binder);
        if binder.is_consumed() {
            return;
        }
        self.dst.try_bind_slot(binder);
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn cb_addable(&self) -> bool {
        true
    }

    fn cbable_weight(&self) -> usize {
        // An image copy records one clCommandCopyImageKHR (where supported).
        1
    }

    fn cb_restamp(&self, evs: &crate::eager::Deps) {
        // Multi-output: stamp the CB completion event onto BOTH element pipes.
        if let Some((v, _d, h)) = self.src_pipe.take_home() {
            self.src_pipe.put_home(v, evs.clone(), h);
        }
        if let Some((v, _d, h)) = self.dst_pipe.take_home() {
            self.dst_pipe.put_home(v, evs.clone(), h);
        }
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
        // Multi-output: `sync` yields a per-side tuple of Checkouts; extract both.
        let (src_co, dst_co) = self.sync(&ctx)?;
        Ok((src_co.into_inner(), dst_co.into_inner()))
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
    /// Design-v2 CB home: an image fill records `clCommandFillImageKHR` where the
    /// extension provides it, else falls back to software.
    cb_cache: CbCache,
}

impl<I: ImageEnqueue, T: Copy + Send + 'static> DeviceOp for ImageFill<I, T> {
    type Output = I;

    fn output_pipe(&self) -> Option<Pipe<I>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        use opencl3::memory::ClMem;
        // In-place: the filled image is the lent image → home threads through.
        let (mut img, deps, home) = self.img.resolve_home(ec)?;
        let pattern = self.pattern;

        // ── CB-mode fork (design v2) — image fill via clCommandFillImageKHR. An
        // absent PFN marks the build ineligible → boundary falls back to per-op. ──
        match ec.cb() {
            CbWalk::Off => {}
            CbWalk::Build { builder, ext, .. } => {
                // `cb_leaf_build` (not raw `cb_collect_external`/`sp_lookup`) so a
                // filled image SLOT is note_slot'd into the CB's captured set and its
                // reach propagates onto the output — precise invalidation on mutate,
                // matching the buffer `Fill`.
                let waits = cb_leaf_build(
                    ec,
                    builder,
                    ext,
                    &deps,
                    self.img.slot_cell_id(),
                    self.img.pipe_cell_id(),
                    self.out.cell_id(),
                );
                let mem = crate::record::MemRef::Buffer(img.image_ref().get());
                let color = unsafe {
                    std::slice::from_raw_parts(
                        pattern.as_ptr() as *const u8,
                        std::mem::size_of::<[T; 4]>(),
                    )
                };
                if let Some(sp) = builder.fill_image(mem, color, [0, 0, 0], self.region, &waits) {
                    ec.sp_register(self.out.cell_id(), std::collections::BTreeSet::from([sp]));
                }
                self.out.put_home(img, Deps::new(), home);
                return Ok(());
            }
            CbWalk::LendOnly { ext, .. } => {
                cb_collect_external(ext, &deps);
                self.out.put_home(img, Deps::new(), home);
                return Ok(());
            }
        }

        let raw = deps_to_wait_list(&deps);
        // Fill has no native CL_BLOCKING flag — always enqueue non-blocking; a
        // blocking terminal waits on the event via the carried deps.
        let event = fill_image_enqueue(
            img.image_mut(),
            ec,
            self.region,
            pattern.as_ptr() as *const std::ffi::c_void,
            &raw,
        )?;
        self.out.put_home(img, single_dep(event), home);
        Ok(())
    }

    /// Atomicity pre-pass mirror of the slice ops: read-only readiness of the
    /// lent image cell, so a busy/unsatisfiable image op is caught before any
    /// earlier lending op enqueues (see [`Input::check_ready`]).
    fn check_ready(&self) -> Result<()> {
        self.img.check_ready()
    }

    fn bind_slots(&self, binder: &mut crate::SlotBinder) {
        // The filled image may be a `slot!()` operand; a concrete/pipe input is a
        // no-op in `try_bind_slot`.
        self.img.try_bind_slot(binder);
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn cb_addable(&self) -> bool {
        true
    }

    fn cbable_weight(&self) -> usize {
        // An image fill records one clCommandFillImageKHR (where supported).
        1
    }

    fn cb_restamp(&self, evs: &crate::eager::Deps) {
        if let Some((v, _d, h)) = self.out.take_home() {
            self.out.put_home(v, evs.clone(), h);
        }
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
        self.sync(&ctx).map(Checkout::into_inner)
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
// `image_download`) can handle every image type.
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
        dst: std::sync::Mutex::new(dst),
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
        dst: std::sync::Mutex::new(dst),
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
        region: Some(region),
        src_pipe: Pipe::new(),
        dst_pipe: Pipe::new(),
        cb_cache: new_cb_cache(),
    }
}

/// Pipe-fed image→image copy for a graph: both operands accept a concrete image OR
/// an upstream `Pipe`/`Checkout`/`slot!` of one (`impl Into<Input<_>>`), so an image
/// copy can chain off `and_then` (unlike the concrete `copy_to`, which consumes a
/// concrete image by value). The copy extent is derived at execute from the lent
/// src image. `Output = (Src, Dst)`: both images rebind out for reuse.
pub fn eager_image_copy<Src, Dst, S, D>(src: S, dst: D) -> ImageCopy<Src, Dst>
where
    Src: ImageEnqueue,
    Dst: ImageEnqueue,
    S: Into<Input<Src>>,
    D: Into<Input<Dst>>,
{
    ImageCopy {
        src: src.into(),
        dst: dst.into(),
        region: None, // derived at execute from the lent src
        src_pipe: Pipe::new(),
        dst_pipe: Pipe::new(),
        cb_cache: new_cb_cache(),
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
        cb_cache: new_cb_cache(),
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
//
// ## Why `Send + 'static`
//
// The supertrait bound is `KernelArg + Send + 'static + Sealed`. The
// `'static` is what lets an image kernel arg flow through the SAME
// reusable-graph machinery a `DeviceSlice` arg does: the proc-macro
// stores each image arg as an [`Input`](crate::eager::Input)`<I>`
// (an `Arc<Mutex<Option<I>>>` cell), lends it for one run, and
// returns it on the run's [`Checkout`](crate::eager::Checkout) drop —
// re-arming the graph with a STABLE `cl_mem` handle. A cell of `I`
// requires `I: 'static`. Every OWNING image type
// ([`Image1D`]/[`Image2D`]/[`Image3D`]/[`Image1DArray`]/
// [`Image2DArray`]/[`Image1DBuffer`]) is `'static` (it owns its
// `cl_mem` via the opencl3 [`Image`] handle, released on `Drop`), so
// they satisfy the bound and are first-class reusable kernel args.
//
// The borrowed [`Image1DBufferView<'a, …>`](Image1DBufferView)
// deliberately is NOT `'static` (it carries a `'a` borrow of the
// `DeviceSlice` it views), so it does NOT impl these traits — it
// cannot be a reusable kernel arg. Upload a [`DeviceSlice`] and then
// either pass the slice directly (`&[T]` arg) or allocate an owned
// [`Image1DBuffer`] when the kernel needs `image1d_buffer_t`. The
// view remains useful for its host-side accessors; it just can't sit
// in a graph cell.

mod kernel_image_arg_sealed {
    pub trait Sealed {}
}

// `Image<dim>D<A, F>` is `Send` when both `A: Send` and `F: Send`
// (PhantomData propagation). All access + format marker ZSTs in
// this crate impl `Send` via their `#[derive(Clone, Copy, Debug)]`,
// but the bound has to appear explicitly here for the trait impls
// to satisfy the supertrait `Send + 'static`.

macro_rules! kernel_image_arg_traits {
    ($( $name:ident => $doc:literal ),+ $(,)?) => { $(
        #[doc = $doc]
        ///
        /// Marker trait (see the section comment above for the exact-access +
        /// `SampledTypeFamily`-parameterization rationale). Impl'd only by the
        /// matching owning `Image*` type via `impl_image_enqueue!`.
        pub trait $name<SF: format::SampledTypeFamily>:
            KernelArg + Send + 'static + kernel_image_arg_sealed::Sealed
        {
        }
    )+ };
}

// Host-side arg traits, one per (dimensionality x access qualifier). The kernel
// proc-macro selects the variant by `format_ident!("KernelImage{dim}{access}Arg")`,
// so these names are load-bearing.
kernel_image_arg_traits! {
    KernelImage1DReadArg =>
        "Host arg for a kernel `&Image!(1D, ...)` parameter the kernel declared read_only.",
    KernelImage1DWriteArg =>
        "Host arg for a kernel `&Image!(1D, ...)` parameter the kernel declared write_only.",
    KernelImage1DReadWriteArg =>
        "Host arg for a kernel `&Image!(1D, ...)` parameter the kernel declared read_write.",
    KernelImage2DReadArg =>
        "Host arg for a kernel `&Image!(2D, ...)` parameter the kernel declared read_only.",
    KernelImage2DWriteArg =>
        "Host arg for a kernel `&Image!(2D, ...)` parameter the kernel declared write_only.",
    KernelImage2DReadWriteArg =>
        "Host arg for a kernel `&Image!(2D, ...)` parameter the kernel declared read_write.",
    KernelImageBufferReadArg =>
        "Host arg for a kernel `&Image!(1D-buffer, ...)` parameter the kernel declared read_only.",
    KernelImageBufferWriteArg =>
        "Host arg for a kernel `&Image!(1D-buffer, ...)` parameter the kernel declared write_only.",
    KernelImageBufferReadWriteArg =>
        "Host arg for a kernel `&Image!(1D-buffer, ...)` parameter the kernel declared read_write.",
    KernelImage3DReadArg =>
        "Host arg for a kernel `&Image!(3D, ...)` parameter the kernel declared read_only.",
    KernelImage3DWriteArg =>
        "Host arg for a kernel `&Image!(3D, ...)` parameter the kernel declared write_only.",
    KernelImage3DReadWriteArg =>
        "Host arg for a kernel `&Image!(3D, ...)` parameter the kernel declared read_write.",
    KernelImage1DArrayReadArg =>
        "Host arg for a kernel `&Image!(1D-array, ...)` parameter the kernel declared read_only.",
    KernelImage1DArrayWriteArg =>
        "Host arg for a kernel `&Image!(1D-array, ...)` parameter the kernel declared write_only.",
    KernelImage1DArrayReadWriteArg =>
        "Host arg for a kernel `&Image!(1D-array, ...)` parameter the kernel declared read_write.",
    KernelImage2DArrayReadArg =>
        "Host arg for a kernel `&Image!(2D-array, ...)` parameter the kernel declared read_only.",
    KernelImage2DArrayWriteArg =>
        "Host arg for a kernel `&Image!(2D-array, ...)` parameter the kernel declared write_only.",
    KernelImage2DArrayReadWriteArg =>
        "Host arg for a kernel `&Image!(2D-array, ...)` parameter the kernel declared read_write.",
}

// ── Sealed marker impls + per-(dim, access) arg impls ──────────────
//
// `kernel_image_arg_sealed::Sealed` is required by every `KernelImage<dim>D*Arg`
// trait; it is blanket-impl'd on every concrete `Image<dim>D<A, F>` regardless of
// access marker (access-specific gating is on the per-access impls). Each access
// marker then impls one or more trait variants, parameterised on `F::SampledFamily`
// so the proc-macro's `<F: format::Format<SampledFamily = K>>` wrapper bound picks
// the right impl per kernel `type=` keyword.
//
// Compatibility partial order (per OpenCL `clSetKernelArg` rules), emitted by
// `impl_kernel_image_arg_matrix!` for EVERY dim:
//   - `ReadOnly`  host image → satisfies `Read` kernel arg only
//   - `WriteOnly` host image → satisfies `Write` kernel arg only
//   - `ReadWrite` host image → satisfies all three (`Read`, `Write`, `ReadWrite`) —
//     the host promises the cl_mem can bind to any kernel access qualifier; the
//     runtime only forbids writing CL_MEM_READ_ONLY / reading CL_MEM_WRITE_ONLY,
//     neither of which fires when ReadWrite is the host flag.
//
// This lets a single `Image2D<ReadWrite, F>` flow through a pipeline mixing
// write-only producers and read-only consumers without intermediate cl_mem retypes.

/// Emit, for one image family, the `Sealed` impl + the fixed 5-impl access matrix
/// wiring its markers to the `$read`/`$write`/`$readwrite` arg traits. The twin of
/// `kernel_image_arg_traits!` (which DEFINES the traits); this WIRES the impls, so
/// the compatibility partial order above lives in ONE place, not 6 copies.
macro_rules! impl_kernel_image_arg_matrix {
    ($fam:ident, $read:ident, $write:ident, $readwrite:ident) => {
        impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static>
            kernel_image_arg_sealed::Sealed for $fam<A, F>
        {
        }
        impl<F: format::Format + Send + 'static> $read<F::SampledFamily> for $fam<ReadOnly, F> {}
        impl<F: format::Format + Send + 'static> $write<F::SampledFamily> for $fam<WriteOnly, F> {}
        impl<F: format::Format + Send + 'static> $read<F::SampledFamily> for $fam<ReadWrite, F> {}
        impl<F: format::Format + Send + 'static> $write<F::SampledFamily> for $fam<ReadWrite, F> {}
        impl<F: format::Format + Send + 'static> $readwrite<F::SampledFamily>
            for $fam<ReadWrite, F>
        {
        }
    };
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

impl_kernel_image_arg_matrix!(
    Image1D,
    KernelImage1DReadArg,
    KernelImage1DWriteArg,
    KernelImage1DReadWriteArg
);

impl_kernel_image_arg_matrix!(
    Image2D,
    KernelImage2DReadArg,
    KernelImage2DWriteArg,
    KernelImage2DReadWriteArg
);

impl_kernel_image_arg_matrix!(
    Image3D,
    KernelImage3DReadArg,
    KernelImage3DWriteArg,
    KernelImage3DReadWriteArg
);

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

impl_kernel_image_arg_matrix!(
    Image1DArray,
    KernelImage1DArrayReadArg,
    KernelImage1DArrayWriteArg,
    KernelImage1DArrayReadWriteArg
);

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

impl_kernel_image_arg_matrix!(
    Image2DArray,
    KernelImage2DArrayReadArg,
    KernelImage2DArrayWriteArg,
    KernelImage2DArrayReadWriteArg
);

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

impl_kernel_image_arg_matrix!(
    Image1DBuffer,
    KernelImageBufferReadArg,
    KernelImageBufferWriteArg,
    KernelImageBufferReadWriteArg
);

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

// NOTE: `Image1DBufferView<'a, …>` deliberately does NOT impl the
// `KernelImageBuffer*Arg` traits anymore. Those traits now require
// `'static` (so an image arg can flow through the reusable-graph
// `Input`/cell/`Checkout` machinery, exactly like a `DeviceSlice`),
// and the view carries a `'a` borrow of the slice it views — it is
// not `'static`. A borrowed view as a kernel arg was the original
// justification for the image one-shot/consuming fork; with images
// now owned-in-cell, that fork is gone and the view simply isn't a
// reusable kernel arg. To feed slice-backed storage to an
// `image1d_buffer_t` kernel param, allocate an owned
// [`Image1DBuffer`] (which owns its `cl_mem`); the view keeps its
// host-side accessors for the non-kernel-arg cases.

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

// ── RecordableBuffer: stable cl_mem handle for owned images ─────────
//
// Mirrors the `DeviceSlice` impl in `launch.rs` — exposes the owning
// image's backing `cl_mem` + byte length as a [`BufHandle`]. Two
// callers want it:
//   - the home-invariant tests, which read `record_handle().mem` as a
//     stable identity key to assert an image's `cl_mem` is REHOMED
//     (reused) across reusable-graph replays, not re-minted;
//   - the record/replay path (future image support), symmetric with
//     the slice families.
// Every OWNING image type gets the impl (they each hold a `cl_mem` via
// the opencl3 [`Image`]); the borrowed `Image1DBufferView` does not —
// it shares the slice's `cl_mem`, whose handle is reachable through the
// slice's own `RecordableBuffer` impl.

/// `clEnqueue*Image` byte length of an opencl3 [`Image`] — its backing
/// `cl_mem` size. Falls back to `0` only if `clGetMemObjectInfo` fails
/// (it never does for a live image), which keeps the accessor
/// infallible for the identity-key use.
fn image_byte_len(image: &Image) -> usize {
    image.size().unwrap_or(0)
}

macro_rules! impl_recordable_image {
    ($ty:ident) => {
        impl<A: KernelAccess, F: format::Format> crate::record::RecordableBuffer for $ty<A, F> {
            fn record_handle(&self) -> crate::record::BufHandle {
                crate::record::BufHandle {
                    mem: crate::record::MemRef::Buffer(self.image.get()),
                    byte_len: image_byte_len(&self.image),
                }
            }
        }
    };
}
impl_recordable_image!(Image1D);
impl_recordable_image!(Image2D);
impl_recordable_image!(Image3D);
impl_recordable_image!(Image1DArray);
impl_recordable_image!(Image2DArray);
impl_recordable_image!(Image1DBuffer);

// ── Slot machinery: images as first-class reusable-graph slots ──────
//
// With `RecordableBuffer` above, an owned image already flows into a
// slot as a PIPE source (`Tag(pipe)` → `FedByPipe`). These two impls
// make it a full VALUE slot too — `slot!(Tag)` + `bind`/`mutate_bind`
// with an image value — exactly like the buffer families:
//   - `SlotEq` (rebind idempotency / crossed-swap detection) by backing
//     `cl_mem` identity — an image is always a `cl_mem` object, never SVM.
//   - `SlotValue` MOVE-ONLY (`fill_clone` → `None`): an owned image can't
//     be in two cells at once, so it is take-once into the first matching
//     cell, matching `DeviceSlice`/`MappedSlice`/`USMSlice`. (A shared
//     read-only image would ride an `Arc<…>` clone impl, as `DeviceSlice`
//     does — added only when a fan-out use appears.)
// This is what lets a `mutate_bind` of an image slot re-target a built
// graph AND drive the precise command-buffer invalidation for image
// commands (see `ImageCopy`/`ImageFill` reach registration).
macro_rules! impl_image_slot {
    ($ty:ident) => {
        impl<A: KernelAccess, F: format::Format> crate::eager::SlotEq for $ty<A, F> {
            fn slot_eq(&self, other: &Self) -> bool {
                // Owned images are `cl_mem`-backed; identity is handle equality.
                self.image.get() == other.image.get()
            }
        }

        impl<A, F> crate::eager::SlotValue for $ty<A, F>
        where
            A: KernelAccess + Send + 'static,
            F: format::Format + Send + 'static,
        {
            fn fill_clone(&self) -> Option<Box<dyn std::any::Any + Send>> {
                // Move-only: no clone. The binder takes the single image once.
                None
            }
        }
    };
}
impl_image_slot!(Image1D);
impl_image_slot!(Image2D);
impl_image_slot!(Image3D);
impl_image_slot!(Image1DArray);
impl_image_slot!(Image2DArray);
impl_image_slot!(Image1DBuffer);

// ── ToInputImage: a kernel IMAGE arg, concrete-or-pipe ──────────────
//
// The exact image-side twin of [`ToInput`](crate::eager::ToInput) (the
// slice-arg conversion). The proc-macro emits each image kernel arg as
// `impl ToInputImage<SF, Buf = __claspr_D{n}>` and stores the resulting
// `Input<Buf>` in the Op — so an owned image, a `Pipe<image>` (upstream
// output), a `Checkout<image>` (a previous run's result fed straight
// in), or a `slot!(Tag where Tag::Value = image)` all plug into the
// same image-arg position, with `Buf` inferred (no turbofish), exactly
// as the slice families do.
//
// Keyed on the sampled-type-family marker `SF` rather than a slice
// element so it stays a distinct nominal trait from `ToInput<E>` (no
// coherence clash) and so the macro can pin `SF` to the kernel's
// `type=` family. Per-shape impls (owned families + `Pipe` + `Checkout`
// + `SlotHandle`), not a blanket, so they stay disjoint under
// coherence — same discipline as `ToInput`.

/// Image analogue of [`ToInput`](crate::eager::ToInput): convert a
/// kernel image argument (owned image / `Pipe` / `Checkout` / `slot!`)
/// into the [`Input`]`<Buf>` the reusable image kernel Op stores.
/// `Buf` is the concrete owning image type, inferred from the
/// argument; `SF` is the kernel's sampled-type family
/// (`Uint`/`Sint`/`Float`), pinned by the proc-macro so a `Pipe`/`Checkout`
/// of an image flows in without a turbofish.
pub trait ToInputImage<SF: format::SampledTypeFamily> {
    /// The concrete owning image type this arg resolves to — the macro
    /// pins it as the Op's per-image generic and applies the matching
    /// `KernelImage<dim>D<Access>Arg<SF>` bound to it.
    type Buf;
    /// Wrap as a concrete or piped [`Input`].
    fn to_input_image(self) -> Input<Self::Buf>;
}

// A pipe of any image type → a deferred input. `SF` is unconstrained on
// the pipe itself; the macro's `Buf = __D` + `__D: KernelImage…Arg<SF>`
// ties it.
impl<SF: format::SampledTypeFamily, D> ToInputImage<SF> for Pipe<D> {
    type Buf = D;
    fn to_input_image(self) -> Input<D> {
        Input::Pipe(self)
    }
}

/// Implement [`ToInputImage`] for one concrete owning image family.
/// Per-family (not a blanket) so it stays disjoint from the `Pipe<D>`
/// impl under coherence.
macro_rules! impl_to_input_image_owned {
    ($ty:ident) => {
        impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static>
            ToInputImage<F::SampledFamily> for $ty<A, F>
        {
            type Buf = $ty<A, F>;
            fn to_input_image(self) -> Input<$ty<A, F>> {
                Input::from(self)
            }
        }

        // A `Checkout<image>` is usable wherever the bare image is — a
        // reused-graph image output flows straight into the next launch
        // without an explicit `into_inner()`. Consuming the `Checkout`
        // severs its return and feeds the inner image as a concrete
        // `Input`. Distinct nominal type → disjoint under coherence.
        impl<A: KernelAccess + Send + 'static, F: format::Format + Send + 'static>
            ToInputImage<F::SampledFamily> for Checkout<$ty<A, F>>
        {
            type Buf = $ty<A, F>;
            fn to_input_image(self) -> Input<$ty<A, F>> {
                Input::from(self.into_inner())
            }
        }
    };
}
impl_to_input_image_owned!(Image1D);
impl_to_input_image_owned!(Image2D);
impl_to_input_image_owned!(Image3D);
impl_to_input_image_owned!(Image1DArray);
impl_to_input_image_owned!(Image2DArray);
impl_to_input_image_owned!(Image1DBuffer);

// A `slot!(Tag)` whose `Tag::Value` is an owning image type plugs into
// the image-arg position, mirroring the slice `SlotHandle` impl on
// `ToInput`. The macro infers `Buf = Tag::Value` and applies the right
// `KernelImage…Arg<SF>` bound to it. `SlotHandle<Tg>` is a distinct
// nominal type from the bare families / `Pipe` / `Checkout`, so it
// stays disjoint under coherence.
impl<SF, Tg> ToInputImage<SF> for crate::eager::SlotHandle<Tg>
where
    SF: format::SampledTypeFamily,
    Tg: crate::eager::Tag,
{
    type Buf = Tg::Value;
    fn to_input_image(self) -> Input<Tg::Value> {
        self.into_slot_input()
    }
}
