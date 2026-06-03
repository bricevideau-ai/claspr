//! `dim2_uint::fill_pattern` expects a 2D image
//! (`KernelImage2DWriteArg<Uint>`). `Image1D` only impls the
//! `KernelImage1D*Arg` trait family, not the 2D one.

use claspr::{image::format::R32Uint, Context, WriteOnly};

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = claspr_test_image_kernels::dim2_uint::kernels(&ctx).unwrap();
    let img = claspr::Image1D::<WriteOnly, R32Uint>::alloc(&ctx, 16).unwrap();
    let _ = kernels.fill_pattern([4usize, 4usize], img, 4u32, 4u32);
}
