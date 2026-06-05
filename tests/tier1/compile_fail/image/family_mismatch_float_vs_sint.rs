//! `dim2_float::fill_pattern` declares `type=f32` → bounded
//! `KernelImage2DWriteArg<Float>`. `R32Sint` is in the `Sint`
//! family; doesn't impl the Float variant.

use claspr::{Context, WriteOnly, image::format::R32Sint};

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = claspr_test_image_kernels::dim2_float::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32Sint>::alloc(&ctx, 4, 4).unwrap();
    let _ = kernels.fill_pattern([4usize, 4usize], img, 4u32, 4u32);
}
