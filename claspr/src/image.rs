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
use crate::error::Error;
use crate::launch::KernelArg;
use crate::op::{ProfileCb, ProfilingInfo, register_profiling_callback};
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
    /// [`ImageReadOp`] — pick a terminal
    /// (`.wait(&launcher)?` blocking, `.submit(&launcher)?`
    /// non-blocking).
    pub fn read<'a>(&'a self, dst: &'a mut [F::Pixel]) -> Result<ImageReadOp<'a, F::Pixel>> {
        let pixel_count = (self.width as usize) * (self.height as usize);
        image_read_op(
            &self.image,
            [self.width as usize, self.height as usize, 1],
            dst,
            pixel_count,
            "Image2D",
        )
    }

    /// Same as [`read`](Self::read) but raw bytes — caller-supplied
    /// `&mut [u8]` of length `width * height * size_of::<F::Pixel>()`.
    pub fn read_bytes<'a>(&'a self, dst: &'a mut [u8]) -> Result<ImageReadOp<'a, u8>> {
        let pixel_count = (self.width as usize) * (self.height as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        image_read_bytes_op(
            &self.image,
            [self.width as usize, self.height as usize, 1],
            dst,
            expected,
            "Image2D",
        )
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
            region: [self.width as usize, self.height as usize, 1],
            pixel_count: (self.width as usize) * (self.height as usize),
            _format: PhantomData::<F>,
        }
    }

    /// Same as [`read_alloc`](Self::read_alloc) but returns raw bytes.
    pub fn read_bytes_alloc(&self) -> ImageReadBytesAlloc<'_, F> {
        ImageReadBytesAlloc {
            image: &self.image,
            region: [self.width as usize, self.height as usize, 1],
            byte_len: (self.width as usize)
                * (self.height as usize)
                * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// Begin writing a typed pixel slice to this image. `pixels.len()`
    /// must equal `width * height` (asserted). Returns a lazy
    /// [`ImageWriteOp`].
    pub fn write<'a>(&'a mut self, pixels: &'a [F::Pixel]) -> ImageWriteOp<'a, F::Pixel> {
        let pixel_count = (self.width as usize) * (self.height as usize);
        image_write_op(
            &mut self.image,
            [self.width as usize, self.height as usize, 1],
            pixels,
            pixel_count,
            "Image2D",
        )
    }

    /// Same as [`write`](Self::write) but raw bytes — must be
    /// exactly `width * height * size_of::<F::Pixel>()` bytes.
    pub fn write_bytes<'a>(&'a mut self, bytes: &'a [u8]) -> ImageWriteOp<'a, u8> {
        let pixel_count = (self.width as usize) * (self.height as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        image_write_bytes_op(
            &mut self.image,
            [self.width as usize, self.height as usize, 1],
            bytes,
            expected,
            "Image2D",
        )
    }

    /// Begin copying this image into `dst`. Both images must have
    /// the same dimensions and format-compatible pixel sizes
    /// (`clEnqueueCopyImage` surfaces format mismatches as
    /// `CL_IMAGE_FORMAT_MISMATCH` at terminal time).
    pub fn copy_to<'a, A2: KernelAccess>(&'a self, dst: &'a mut Image2D<A2, F>) -> ImageCopyOp<'a> {
        image_copy_op(
            &self.image,
            &mut dst.image,
            [self.width as usize, self.height as usize, 1],
        )
    }

    /// Begin filling every pixel with `pattern`. The 4-component
    /// pattern follows OpenCL's `clEnqueueFillImage` shape —
    /// match `T` to the format's `SampledTypeFamily` (`u32` for
    /// `Uint`, `i32` for `Sint`, `f32` for `Float` / `Unorm` /
    /// `Snorm`).
    pub fn fill<T: Copy>(&mut self, pattern: [T; 4]) -> ImageFillOp<'_, T> {
        image_fill_op(
            &mut self.image,
            [self.width as usize, self.height as usize, 1],
            pattern,
        )
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

// ── Image transfer ops ─────────────────────────────────────────────
//
// Same lazy-builder / late-bind-launcher shape as the buffer ops
// (`WriteOp`/`ReadOp`/`CopyOp`/`FillOp`). Each image type's
// `.write` / `.read` / `.read_alloc` / `.copy_to` / `.fill` method
// returns one of these Ops; the caller picks `.wait(&launcher)?`
// (blocking) or `.submit(&launcher)?` (non-blocking, returns
// `Event`) plus the usual `.after(&event)` / `.profiled(cb)`
// modifiers.
//
// The Op types are dimensionality-agnostic — they hold the image
// + a 3-component region (unused dims = 1) + the host
// pointer/length. The per-type methods (`Image2D::write`,
// `Image1D::write`, ...) are thin wrappers that pass the right
// region shape. The actual `enqueue_*_image` call lives in one
// place per op.

/// Lazy builder for `clEnqueueWriteImage`. Constructed via
/// `image.write(...)` / `image.write_bytes(...)` on any image type.
pub struct ImageWriteOp<'a, T> {
    image: &'a mut Image,
    region: [usize; 3],
    data: *const T,
    /// Lifetime tag — the data pointer is borrowed for `'a`. The
    /// Op holds a raw pointer rather than `&'a [T]` because the
    /// builder paths cover both typed-pixel and raw-byte payloads
    /// (the raw bytes case needs `*const u8` not `&[Pixel]`).
    _borrow: PhantomData<&'a [T]>,
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

// SAFETY: `*const T` isn't Send by default but the host pointer
// here is borrowed for `'a` and only read once during the
// blocking or non-blocking `enqueue_write_image` call. The Op is
// moved between functions on the host thread but never crosses
// thread boundaries in claspr today — keeping it !Send is the
// honest answer until Tier 2 needs it.
//
// (We do NOT impl Send.)

impl<'a, T> ImageWriteOp<'a, T> {
    /// Add a queue-side wait dependency. Chainable.
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    /// Add multiple wait-list events at once.
    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    /// Register a completion callback that receives the write's
    /// [`ProfilingInfo`]. Requires the queue to have
    /// `CL_QUEUE_PROFILING_ENABLE`.
    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    /// Sync terminal — enqueue the write with `CL_TRUE` on
    /// `launcher`'s queue; the driver blocks until the image has
    /// been written.
    pub fn wait<L: Launcher + ?Sized>(self, launcher: &L) -> Result<()> {
        let event = enqueue_image_write(self, launcher, CL_BLOCKING)?;
        // CL_BLOCKING already waited for the write at the driver
        // level; we just need to attach the profiling callback if
        // one was registered and let the Event drop.
        drop(event);
        Ok(())
    }

    /// Non-blocking terminal — enqueue the write with `CL_FALSE`
    /// on `launcher`'s queue, return the completion event. `data`
    /// must outlive the event.
    pub fn submit<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Event> {
        enqueue_image_write(self, launcher, CL_NON_BLOCKING)
    }
}

fn enqueue_image_write<'a, T, L: Launcher + ?Sized>(
    op: ImageWriteOp<'a, T>,
    launcher: &L,
    blocking: opencl3::types::cl_bool,
) -> Result<Event> {
    let ImageWriteOp {
        image,
        region,
        data,
        _borrow: _,
        deps,
        profile_cb,
    } = op;
    let origin = [0usize, 0, 0];
    // SAFETY: `data` is borrowed for `'a` (encoded in the
    // PhantomData); under CL_BLOCKING the driver finishes reading
    // it before returning, under CL_NON_BLOCKING the caller
    // contract (data outlives the event) covers liveness.
    let event = unsafe {
        launcher.cl_queue().enqueue_write_image(
            image,
            blocking,
            origin.as_ptr(),
            region.as_ptr(),
            0,
            0,
            data as *mut std::ffi::c_void,
            &deps,
        )?
    };
    if let Some(cb) = profile_cb {
        register_profiling_callback(&event, cb)?;
    }
    Ok(event)
}

/// Lazy builder for `clEnqueueReadImage`. Constructed via
/// `image.read(...)` (caller-supplied dst) or
/// `image.read_alloc()` (op allocates dst). Same `.wait`/`.submit`
/// terminals + `.after`/`.profiled` modifiers as
/// [`ImageWriteOp`].
pub struct ImageReadOp<'a, T> {
    image: &'a Image,
    region: [usize; 3],
    dst: *mut T,
    _borrow: PhantomData<&'a mut [T]>,
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a, T> ImageReadOp<'a, T> {
    /// Add a queue-side wait dependency.
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    /// Add multiple wait-list events at once.
    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    /// Register a completion callback for profiling info.
    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    /// Sync terminal — enqueue the read with `CL_TRUE` on
    /// `launcher`'s queue; the driver blocks until `dst` has been
    /// filled.
    pub fn wait<L: Launcher + ?Sized>(self, launcher: &L) -> Result<()> {
        let event = enqueue_image_read(self, launcher, CL_BLOCKING)?;
        drop(event);
        Ok(())
    }

    /// Non-blocking terminal — enqueue the read with `CL_FALSE`
    /// on `launcher`'s queue, return the completion event. `dst`
    /// is only valid after the event fires; the caller must keep
    /// `dst` alive until then.
    pub fn submit<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Event> {
        enqueue_image_read(self, launcher, CL_NON_BLOCKING)
    }
}

fn enqueue_image_read<'a, T, L: Launcher + ?Sized>(
    op: ImageReadOp<'a, T>,
    launcher: &L,
    blocking: opencl3::types::cl_bool,
) -> Result<Event> {
    let ImageReadOp {
        image,
        region,
        dst,
        _borrow: _,
        deps,
        profile_cb,
    } = op;
    let origin = [0usize, 0, 0];
    // SAFETY: `dst` is borrowed for `'a` (PhantomData) — under
    // CL_BLOCKING the driver fills it before returning; under
    // CL_NON_BLOCKING the caller contract covers liveness.
    let event = unsafe {
        launcher.cl_queue().enqueue_read_image(
            image,
            blocking,
            origin.as_ptr(),
            region.as_ptr(),
            0,
            0,
            dst as *mut std::ffi::c_void,
            &deps,
        )?
    };
    if let Some(cb) = profile_cb {
        register_profiling_callback(&event, cb)?;
    }
    Ok(event)
}

/// Convenience builder — `image.read_alloc()`. Allocates a
/// `Vec<F::Pixel>` of the right size at terminal time and yields
/// it through the terminal's return value. Mirrors the old
/// `download` ergonomics in a lazy-builder shape, but only offers
/// `.wait(&launcher)?` (blocking) because non-blocking + owned-output
/// requires the chain machinery and is properly handled by the
/// Tier 2 `download(image)` combinator instead.
pub struct ImageReadAlloc<'a, F: format::Format> {
    image: &'a Image,
    region: [usize; 3],
    pixel_count: usize,
    _format: PhantomData<F>,
}

impl<'a, F: format::Format> ImageReadAlloc<'a, F>
where
    F::Pixel: Default + Copy,
{
    /// Blocking — allocate the Vec, enqueue + wait, return it.
    pub fn wait<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Vec<F::Pixel>> {
        let mut pixels = vec![<F::Pixel as Default>::default(); self.pixel_count];
        let op = ImageReadOp {
            image: self.image,
            region: self.region,
            dst: pixels.as_mut_ptr(),
            _borrow: PhantomData,
            deps: Vec::new(),
            profile_cb: None,
        };
        op.wait(launcher)?;
        Ok(pixels)
    }
}

/// Like [`ImageReadAlloc`] but returns raw bytes. Useful for
/// PPM-write paths and byte-oriented sinks that don't want the
/// pixel-type round-trip.
pub struct ImageReadBytesAlloc<'a, F: format::Format> {
    image: &'a Image,
    region: [usize; 3],
    byte_len: usize,
    _format: PhantomData<F>,
}

impl<'a, F: format::Format> ImageReadBytesAlloc<'a, F> {
    /// Blocking — allocate the Vec, enqueue + wait, return it.
    pub fn wait<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; self.byte_len];
        let op = ImageReadOp::<u8> {
            image: self.image,
            region: self.region,
            dst: bytes.as_mut_ptr(),
            _borrow: PhantomData,
            deps: Vec::new(),
            profile_cb: None,
        };
        op.wait(launcher)?;
        Ok(bytes)
    }
}

/// Lazy builder for `clEnqueueCopyImage`. Constructed via
/// `src.copy_to(dst)` on any image type — the two images must
/// have matching dimensions and format-compatible pixel sizes
/// (OpenCL surfaces format mismatches as `CL_IMAGE_FORMAT_MISMATCH`
/// at terminal time).
pub struct ImageCopyOp<'a> {
    src: &'a Image,
    dst: &'a mut Image,
    region: [usize; 3],
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a> ImageCopyOp<'a> {
    /// Add a queue-side wait dependency.
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    /// Add multiple wait-list events at once.
    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    /// Register a completion callback for profiling info.
    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    /// Sync terminal — `enqueue + event.wait`. `clEnqueueCopyImage`
    /// has no blocking flag, so `.wait()` is enqueue + event.wait.
    pub fn wait<L: Launcher + ?Sized>(self, launcher: &L) -> Result<()> {
        let event = self.into_event(launcher)?;
        event.wait()?;
        Ok(())
    }

    /// Non-blocking terminal — enqueue and return the event.
    pub fn submit<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Event> {
        self.into_event(launcher)
    }

    pub(crate) fn into_event<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Event> {
        let origin = [0usize, 0, 0];
        // SAFETY: src/dst must belong to the queue's context; the
        // image types' construction enforces this at the
        // type-system level (both came from a Context). Region
        // bounds match by construction (`copy_to` is only callable
        // on same-dim image types, see the per-type method
        // signatures).
        let event = unsafe {
            launcher.cl_queue().enqueue_copy_image(
                self.src,
                self.dst,
                origin.as_ptr(),
                origin.as_ptr(),
                self.region.as_ptr(),
                &self.deps,
            )?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(event)
    }
}

/// Lazy builder for `clEnqueueFillImage`. Constructed via
/// `image.fill(pattern)` on any image type. The pattern is one
/// "fill color" appropriate to the image format —
/// OpenCL's `clEnqueueFillImage` takes a 4-component value (4
/// `f32`s, 4 `i32`s, or 4 `u32`s depending on
/// `cl_channel_data_type`); claspr surfaces this as a generic
/// `[T; 4]` and trusts the caller to use the matching T per the
/// format's `SampledTypeFamily`.
pub struct ImageFillOp<'a, T: Copy> {
    image: &'a mut Image,
    region: [usize; 3],
    pattern: [T; 4],
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a, T: Copy> ImageFillOp<'a, T> {
    /// Add a queue-side wait dependency.
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    /// Add multiple wait-list events at once.
    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    /// Register a completion callback for profiling info.
    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    /// Sync terminal — `enqueue + event.wait`. `clEnqueueFillImage`
    /// has no blocking flag.
    pub fn wait<L: Launcher + ?Sized>(self, launcher: &L) -> Result<()> {
        let event = self.into_event(launcher)?;
        event.wait()?;
        Ok(())
    }

    /// Non-blocking terminal — enqueue and return the event.
    pub fn submit<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Event> {
        self.into_event(launcher)
    }

    pub(crate) fn into_event<L: Launcher + ?Sized>(self, launcher: &L) -> Result<Event> {
        let origin = [0usize, 0, 0];
        // SAFETY: `pattern.as_ptr()` is a valid 4-component fill
        // value the runtime byte-copies into every pixel inside
        // `region`. Image lifetime is borrowed for `'a`.
        let event = unsafe {
            launcher.cl_queue().enqueue_fill_image(
                self.image,
                self.pattern.as_ptr() as *const std::ffi::c_void,
                origin.as_ptr(),
                self.region.as_ptr(),
                &self.deps,
            )?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(event)
    }
}

// ── Per-type method helpers — region builders ──────────────────────
//
// Each `Image*Type::write` / `.read` / `.copy_to` / `.fill` method
// hands its dim-specific `[usize; 3]` region to one of the Op
// constructors below. Centralising the "build an op" step keeps
// per-image-type methods to a single line each.

fn image_write_op<'a, T>(
    image: &'a mut Image,
    region: [usize; 3],
    pixels: &'a [T],
    expected_pixel_count: usize,
    type_name: &'static str,
) -> ImageWriteOp<'a, T> {
    assert_eq!(
        pixels.len(),
        expected_pixel_count,
        "{type_name}::write: pixel count {} ≠ expected {}",
        pixels.len(),
        expected_pixel_count,
    );
    ImageWriteOp {
        image,
        region,
        data: pixels.as_ptr(),
        _borrow: PhantomData,
        deps: Vec::new(),
        profile_cb: None,
    }
}

fn image_write_bytes_op<'a>(
    image: &'a mut Image,
    region: [usize; 3],
    bytes: &'a [u8],
    expected_bytes: usize,
    type_name: &'static str,
) -> ImageWriteOp<'a, u8> {
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "{type_name}::write_bytes: buffer length {} ≠ expected {}",
        bytes.len(),
        expected_bytes,
    );
    ImageWriteOp {
        image,
        region,
        data: bytes.as_ptr(),
        _borrow: PhantomData,
        deps: Vec::new(),
        profile_cb: None,
    }
}

fn image_read_op<'a, T>(
    image: &'a Image,
    region: [usize; 3],
    dst: &'a mut [T],
    expected_pixel_count: usize,
    _type_name: &'static str,
) -> Result<ImageReadOp<'a, T>> {
    if dst.len() != expected_pixel_count {
        return Err(Error::LengthMismatch {
            src: expected_pixel_count,
            dst: dst.len(),
        });
    }
    Ok(ImageReadOp {
        image,
        region,
        dst: dst.as_mut_ptr(),
        _borrow: PhantomData,
        deps: Vec::new(),
        profile_cb: None,
    })
}

fn image_read_bytes_op<'a>(
    image: &'a Image,
    region: [usize; 3],
    dst: &'a mut [u8],
    expected_bytes: usize,
    _type_name: &'static str,
) -> Result<ImageReadOp<'a, u8>> {
    if dst.len() != expected_bytes {
        return Err(Error::LengthMismatch {
            src: expected_bytes,
            dst: dst.len(),
        });
    }
    Ok(ImageReadOp {
        image,
        region,
        dst: dst.as_mut_ptr(),
        _borrow: PhantomData,
        deps: Vec::new(),
        profile_cb: None,
    })
}

fn image_copy_op<'a>(src: &'a Image, dst: &'a mut Image, region: [usize; 3]) -> ImageCopyOp<'a> {
    ImageCopyOp {
        src,
        dst,
        region,
        deps: Vec::new(),
        profile_cb: None,
    }
}

fn image_fill_op<'a, T: Copy>(
    image: &'a mut Image,
    region: [usize; 3],
    pattern: [T; 4],
) -> ImageFillOp<'a, T> {
    ImageFillOp {
        image,
        region,
        pattern,
        deps: Vec::new(),
        profile_cb: None,
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
    pub fn read<'a>(&'a self, dst: &'a mut [F::Pixel]) -> Result<ImageReadOp<'a, F::Pixel>> {
        image_read_op(
            &self.image,
            [self.width as usize, 1, 1],
            dst,
            self.width as usize,
            "Image1D",
        )
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(&'a self, dst: &'a mut [u8]) -> Result<ImageReadOp<'a, u8>> {
        let expected = (self.width as usize) * std::mem::size_of::<F::Pixel>();
        image_read_bytes_op(
            &self.image,
            [self.width as usize, 1, 1],
            dst,
            expected,
            "Image1D",
        )
    }

    /// See [`Image2D::read_alloc`].
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        ImageReadAlloc {
            image: &self.image,
            region: [self.width as usize, 1, 1],
            pixel_count: self.width as usize,
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::read_bytes_alloc`].
    pub fn read_bytes_alloc(&self) -> ImageReadBytesAlloc<'_, F> {
        ImageReadBytesAlloc {
            image: &self.image,
            region: [self.width as usize, 1, 1],
            byte_len: (self.width as usize) * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::write`].
    pub fn write<'a>(&'a mut self, pixels: &'a [F::Pixel]) -> ImageWriteOp<'a, F::Pixel> {
        image_write_op(
            &mut self.image,
            [self.width as usize, 1, 1],
            pixels,
            self.width as usize,
            "Image1D",
        )
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(&'a mut self, bytes: &'a [u8]) -> ImageWriteOp<'a, u8> {
        let expected = (self.width as usize) * std::mem::size_of::<F::Pixel>();
        image_write_bytes_op(
            &mut self.image,
            [self.width as usize, 1, 1],
            bytes,
            expected,
            "Image1D",
        )
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<'a, A2: KernelAccess>(&'a self, dst: &'a mut Image1D<A2, F>) -> ImageCopyOp<'a> {
        image_copy_op(&self.image, &mut dst.image, [self.width as usize, 1, 1])
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy>(&mut self, pattern: [T; 4]) -> ImageFillOp<'_, T> {
        image_fill_op(&mut self.image, [self.width as usize, 1, 1], pattern)
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
    pub fn read<'a>(&'a self, dst: &'a mut [F::Pixel]) -> Result<ImageReadOp<'a, F::Pixel>> {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        image_read_op(
            &self.image,
            [
                self.width as usize,
                self.height as usize,
                self.depth as usize,
            ],
            dst,
            pixel_count,
            "Image3D",
        )
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(&'a self, dst: &'a mut [u8]) -> Result<ImageReadOp<'a, u8>> {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        image_read_bytes_op(
            &self.image,
            [
                self.width as usize,
                self.height as usize,
                self.depth as usize,
            ],
            dst,
            expected,
            "Image3D",
        )
    }

    /// See [`Image2D::read_alloc`].
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        ImageReadAlloc {
            image: &self.image,
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
    pub fn write<'a>(&'a mut self, pixels: &'a [F::Pixel]) -> ImageWriteOp<'a, F::Pixel> {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        image_write_op(
            &mut self.image,
            [
                self.width as usize,
                self.height as usize,
                self.depth as usize,
            ],
            pixels,
            pixel_count,
            "Image3D",
        )
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(&'a mut self, bytes: &'a [u8]) -> ImageWriteOp<'a, u8> {
        let pixel_count = (self.width as usize) * (self.height as usize) * (self.depth as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        image_write_bytes_op(
            &mut self.image,
            [
                self.width as usize,
                self.height as usize,
                self.depth as usize,
            ],
            bytes,
            expected,
            "Image3D",
        )
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<'a, A2: KernelAccess>(&'a self, dst: &'a mut Image3D<A2, F>) -> ImageCopyOp<'a> {
        image_copy_op(
            &self.image,
            &mut dst.image,
            [
                self.width as usize,
                self.height as usize,
                self.depth as usize,
            ],
        )
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy>(&mut self, pattern: [T; 4]) -> ImageFillOp<'_, T> {
        image_fill_op(
            &mut self.image,
            [
                self.width as usize,
                self.height as usize,
                self.depth as usize,
            ],
            pattern,
        )
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
    pub fn read<'a>(&'a self, dst: &'a mut [F::Pixel]) -> Result<ImageReadOp<'a, F::Pixel>> {
        let pixel_count = (self.width as usize) * (self.array_size as usize);
        image_read_op(
            &self.image,
            [self.width as usize, self.array_size as usize, 1],
            dst,
            pixel_count,
            "Image1DArray",
        )
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(&'a self, dst: &'a mut [u8]) -> Result<ImageReadOp<'a, u8>> {
        let pixel_count = (self.width as usize) * (self.array_size as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        image_read_bytes_op(
            &self.image,
            [self.width as usize, self.array_size as usize, 1],
            dst,
            expected,
            "Image1DArray",
        )
    }

    /// See [`Image2D::read_alloc`].
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        ImageReadAlloc {
            image: &self.image,
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
            region: [self.width as usize, self.array_size as usize, 1],
            byte_len: pixel_count * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::write`].
    pub fn write<'a>(&'a mut self, pixels: &'a [F::Pixel]) -> ImageWriteOp<'a, F::Pixel> {
        let pixel_count = (self.width as usize) * (self.array_size as usize);
        image_write_op(
            &mut self.image,
            [self.width as usize, self.array_size as usize, 1],
            pixels,
            pixel_count,
            "Image1DArray",
        )
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(&'a mut self, bytes: &'a [u8]) -> ImageWriteOp<'a, u8> {
        let pixel_count = (self.width as usize) * (self.array_size as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        image_write_bytes_op(
            &mut self.image,
            [self.width as usize, self.array_size as usize, 1],
            bytes,
            expected,
            "Image1DArray",
        )
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<'a, A2: KernelAccess>(
        &'a self,
        dst: &'a mut Image1DArray<A2, F>,
    ) -> ImageCopyOp<'a> {
        image_copy_op(
            &self.image,
            &mut dst.image,
            [self.width as usize, self.array_size as usize, 1],
        )
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy>(&mut self, pattern: [T; 4]) -> ImageFillOp<'_, T> {
        image_fill_op(
            &mut self.image,
            [self.width as usize, self.array_size as usize, 1],
            pattern,
        )
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
    pub fn read<'a>(&'a self, dst: &'a mut [F::Pixel]) -> Result<ImageReadOp<'a, F::Pixel>> {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        image_read_op(
            &self.image,
            [
                self.width as usize,
                self.height as usize,
                self.array_size as usize,
            ],
            dst,
            pixel_count,
            "Image2DArray",
        )
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(&'a self, dst: &'a mut [u8]) -> Result<ImageReadOp<'a, u8>> {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        image_read_bytes_op(
            &self.image,
            [
                self.width as usize,
                self.height as usize,
                self.array_size as usize,
            ],
            dst,
            expected,
            "Image2DArray",
        )
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
    pub fn write<'a>(&'a mut self, pixels: &'a [F::Pixel]) -> ImageWriteOp<'a, F::Pixel> {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        image_write_op(
            &mut self.image,
            [
                self.width as usize,
                self.height as usize,
                self.array_size as usize,
            ],
            pixels,
            pixel_count,
            "Image2DArray",
        )
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(&'a mut self, bytes: &'a [u8]) -> ImageWriteOp<'a, u8> {
        let pixel_count =
            (self.width as usize) * (self.height as usize) * (self.array_size as usize);
        let expected = pixel_count * std::mem::size_of::<F::Pixel>();
        image_write_bytes_op(
            &mut self.image,
            [
                self.width as usize,
                self.height as usize,
                self.array_size as usize,
            ],
            bytes,
            expected,
            "Image2DArray",
        )
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<'a, A2: KernelAccess>(
        &'a self,
        dst: &'a mut Image2DArray<A2, F>,
    ) -> ImageCopyOp<'a> {
        image_copy_op(
            &self.image,
            &mut dst.image,
            [
                self.width as usize,
                self.height as usize,
                self.array_size as usize,
            ],
        )
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy>(&mut self, pattern: [T; 4]) -> ImageFillOp<'_, T> {
        image_fill_op(
            &mut self.image,
            [
                self.width as usize,
                self.height as usize,
                self.array_size as usize,
            ],
            pattern,
        )
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
    pub fn read<'a>(&'a self, dst: &'a mut [F::Pixel]) -> Result<ImageReadOp<'a, F::Pixel>> {
        image_read_op(
            &self.image,
            [self.width as usize, 1, 1],
            dst,
            self.width as usize,
            "Image1DBuffer",
        )
    }

    /// See [`Image2D::read_bytes`].
    pub fn read_bytes<'a>(&'a self, dst: &'a mut [u8]) -> Result<ImageReadOp<'a, u8>> {
        let expected = (self.width as usize) * std::mem::size_of::<F::Pixel>();
        image_read_bytes_op(
            &self.image,
            [self.width as usize, 1, 1],
            dst,
            expected,
            "Image1DBuffer",
        )
    }

    /// See [`Image2D::read_alloc`].
    pub fn read_alloc(&self) -> ImageReadAlloc<'_, F>
    where
        F::Pixel: Default + Copy,
    {
        ImageReadAlloc {
            image: &self.image,
            region: [self.width as usize, 1, 1],
            pixel_count: self.width as usize,
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::read_bytes_alloc`].
    pub fn read_bytes_alloc(&self) -> ImageReadBytesAlloc<'_, F> {
        ImageReadBytesAlloc {
            image: &self.image,
            region: [self.width as usize, 1, 1],
            byte_len: (self.width as usize) * std::mem::size_of::<F::Pixel>(),
            _format: PhantomData::<F>,
        }
    }

    /// See [`Image2D::write`].
    pub fn write<'a>(&'a mut self, pixels: &'a [F::Pixel]) -> ImageWriteOp<'a, F::Pixel> {
        image_write_op(
            &mut self.image,
            [self.width as usize, 1, 1],
            pixels,
            self.width as usize,
            "Image1DBuffer",
        )
    }

    /// See [`Image2D::write_bytes`].
    pub fn write_bytes<'a>(&'a mut self, bytes: &'a [u8]) -> ImageWriteOp<'a, u8> {
        let expected = (self.width as usize) * std::mem::size_of::<F::Pixel>();
        image_write_bytes_op(
            &mut self.image,
            [self.width as usize, 1, 1],
            bytes,
            expected,
            "Image1DBuffer",
        )
    }

    /// See [`Image2D::copy_to`].
    pub fn copy_to<'a, A2: KernelAccess>(
        &'a self,
        dst: &'a mut Image1DBuffer<A2, F>,
    ) -> ImageCopyOp<'a> {
        image_copy_op(&self.image, &mut dst.image, [self.width as usize, 1, 1])
    }

    /// See [`Image2D::fill`].
    pub fn fill<T: Copy>(&mut self, pattern: [T; 4]) -> ImageFillOp<'_, T> {
        image_fill_op(&mut self.image, [self.width as usize, 1, 1], pattern)
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
