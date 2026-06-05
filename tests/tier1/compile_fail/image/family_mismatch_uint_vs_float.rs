//! `dim2_uint::fill_pattern` declares `type=u32` → the proc-macro
//! emits `T: KernelImage2DWriteArg<Uint>`. `R32Float` is in the
//! `Float` family, so it does not impl `KernelImage2DWriteArg<Uint>`.

use claspr::{Context, WriteOnly, image::format::R32Float};

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = claspr_test_image_kernels::dim2_uint::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32Float>::alloc(&ctx, 4, 4).unwrap();
    let _ = kernels.fill_pattern([4usize, 4usize], img, 4u32, 4u32);
}
