//! `dim_buffer_uint::fill_pattern` expects an image-buffer
//! (`KernelImageBufferWriteArg<Uint>`). `Image2D` impls
//! `KernelImage2D*Arg` only.

use claspr::{image::format::R32Uint, Context, WriteOnly};

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = claspr_test_image_kernels::dim_buffer_uint::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32Uint>::alloc(&ctx, 4, 4).unwrap();
    let _ = kernels.fill_pattern([16usize], img, 16u32);
}
