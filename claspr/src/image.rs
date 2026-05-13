//! 2D RGBA8 image helpers.
//!
//! Collapses the `cl_image_format` + `cl_image_desc` + `Image::create`
//! + `enqueue_read_image` boilerplate that the mandelbrot-image and
//!   raymarch samples currently duplicate.

use crate::Result;
use crate::context::Context;
use crate::launch::KernelArg;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{
    CL_MEM_OBJECT_IMAGE2D, CL_MEM_READ_WRITE, CL_RGBA, CL_UNSIGNED_INT8, ClMem, Image,
};
use opencl3::types::{CL_BLOCKING, cl_image_desc, cl_image_format};
use std::ptr;

/// A 2D image with `RGBA8` (4 × `u8`) channels.
///
/// Allocated with `CL_MEM_READ_WRITE` so the same object can be both
/// written by one kernel and read by another (or by `read_image_2d_rgba8`).
pub struct Image2DRgba8 {
    image: Image,
    width: u32,
    height: u32,
}

impl Image2DRgba8 {
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

impl KernelArg for Image2DRgba8 {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let cl_mem_handle = self.image.get();
        unsafe {
            exec.set_arg(&cl_mem_handle);
        }
    }
}

impl Context {
    /// Allocate a `width × height` `RGBA8` 2D image on the device.
    ///
    /// Returns an error if the device doesn't advertise `image_support`
    /// — check `ctx.device().image_support()` first if you want to
    /// fall back gracefully.
    pub fn alloc_image_2d_rgba8(&self, width: u32, height: u32) -> Result<Image2DRgba8> {
        let format = cl_image_format {
            image_channel_order: CL_RGBA,
            image_channel_data_type: CL_UNSIGNED_INT8,
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
        let image = unsafe {
            Image::create(
                self.raw_context(),
                CL_MEM_READ_WRITE,
                &format,
                &desc,
                ptr::null_mut(),
            )?
        };
        Ok(Image2DRgba8 {
            image,
            width,
            height,
        })
    }

    /// Read an `Image2DRgba8` back into a host `Vec<u8>` of length
    /// `width * height * 4` (RGBA8 byte order).
    pub fn read_image_2d_rgba8(&self, img: &Image2DRgba8) -> Result<Vec<u8>> {
        let pixel_count = (img.width as usize) * (img.height as usize);
        let mut pixels = vec![0u8; pixel_count * 4];
        let origin = [0usize, 0, 0];
        let region = [img.width as usize, img.height as usize, 1];
        unsafe {
            self.queue()
                .enqueue_read_image(
                    &img.image,
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
}
