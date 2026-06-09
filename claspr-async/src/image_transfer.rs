//! Tier 2 image transfer combinators — [`image_upload`] /
//! [`image_download`] — mirror [`crate::upload`] / [`crate::download`]
//! for buffers, but produce / consume any owning image type that
//! implements [`ImageHostTransfer`] (`Image2D` / `Image1D` /
//! `Image3D` / `Image1DArray` / `Image2DArray`).
//!
//! Single pair of combinators, generic over the image type via the
//! [`ImageHostTransfer`] trait. The trait's `Dims` associated type
//! picks the right shape per image: `u32` for 1D, `(u32, u32)` for
//! 2D / 1DArray, `(u32, u32, u32)` for 3D / 2DArray.
//!
//! ```ignore
//! use claspr_async::{DeviceOperation, image_download, image_upload};
//! use claspr::image::format::R32Uint;
//! use claspr::{Image2D, ReadWrite};
//!
//! let pixels: Vec<u32> = ...;
//! let result: Vec<u32> = image_upload::<Image2D<ReadWrite, R32Uint>>(pixels, (32, 32))
//!     .and_then(|img| kernels.process([32usize, 32], img))
//!     .and_then(image_download::<Image2D<ReadWrite, R32Uint>>)
//!     .sync(&ctx)?;
//! ```
//!
//! Both ops use **non-blocking enqueues**: `image_upload` keeps the
//! source `Vec` alive via `register_drop_callback` until the write
//! event fires; `image_download` moves the destination Vec up the
//! chain and the source image drops at the end of `execute` while
//! OpenCL retains the `cl_mem` until the read completes.
//!
//! `Image1DBuffer` is **not** covered — it shares storage with a
//! `cl_mem` buffer, so the natural chain shape there is to upload
//! a `DeviceSlice<T>` and `Image1DBufferView::view_of(&slice)`.

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use claspr::image::ImageHostTransfer;
use claspr::{Result, register_drop_callback};
use std::marker::PhantomData;

// ── upload ─────────────────────────────────────────────────────────

/// Allocate an image of type `I` with the given `dims` and write
/// `pixels` (length must equal the resulting image's pixel count)
/// into it via a non-blocking enqueue. The chain receives the
/// owned image when the write event fires.
///
/// `pixels` is kept alive until the write completes via the same
/// `register_drop_callback` keep-alive [`crate::upload`] uses.
pub fn image_upload<I>(pixels: Vec<I::Pixel>, dims: I::Dims) -> ImageUpload<I>
where
    I: ImageHostTransfer,
    I::Pixel: Send + 'static,
{
    ImageUpload {
        pixels: Some(pixels),
        dims,
        _ty: PhantomData,
    }
}

/// Combinator built by [`image_upload`].
pub struct ImageUpload<I: ImageHostTransfer> {
    pixels: Option<Vec<I::Pixel>>,
    dims: I::Dims,
    _ty: PhantomData<fn() -> I>,
}

impl<I> DeviceOperation for ImageUpload<I>
where
    I: ImageHostTransfer,
    I::Pixel: Send + 'static,
{
    type Output = I;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(I, Deps)> {
        let pixels = self
            .pixels
            .take()
            .expect("ImageUpload::execute called twice — internal claspr-async bug");
        let mut img = I::alloc(ctx.context(), self.dims)?;
        let event = img
            .write_op(&pixels)
            .after_all(deps_as_events(&deps))
            .submit_on(ctx)?;
        // SAFETY mirror to `Upload`: the runtime is reading from
        // `pixels` until the write event fires. The drop callback
        // keeps it alive until then.
        register_drop_callback(&event, Box::new(pixels))?;
        Ok((img, vec![wrap_event(event)]))
    }
}

// ── download ───────────────────────────────────────────────────────

/// Consume an image of type `I`, allocate a host `Vec<I::Pixel>`
/// of the image's pixel count, and non-blocking-read the image
/// into it. The Vec moves up the chain; the image drops at end of
/// `execute` but OpenCL retains the underlying `cl_mem` until the
/// read completes.
pub fn image_download<I>(img: I) -> ImageDownload<I>
where
    I: ImageHostTransfer,
    I::Pixel: Default + Copy + Send + 'static,
{
    ImageDownload { img: Some(img) }
}

/// Combinator built by [`image_download`].
pub struct ImageDownload<I: ImageHostTransfer> {
    img: Option<I>,
}

impl<I> DeviceOperation for ImageDownload<I>
where
    I: ImageHostTransfer,
    I::Pixel: Default + Copy + Send + 'static,
{
    type Output = Vec<I::Pixel>;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Vec<I::Pixel>, Deps)> {
        let img = self
            .img
            .take()
            .expect("ImageDownload::execute called twice — internal claspr-async bug");
        let pixel_count = img.pixel_count();
        let mut pixels = vec![<I::Pixel as Default>::default(); pixel_count];
        let event = img
            .read_op(&mut pixels)?
            .after_all(deps_as_events(&deps))
            .submit_on(ctx)?;
        Ok((pixels, vec![wrap_event(event)]))
    }
}
