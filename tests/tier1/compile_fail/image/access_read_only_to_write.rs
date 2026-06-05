//! `dim2_uint::fill_pattern` takes `&mut Image!(...)` with
//! `image_access="write_only"` → bounded
//! `KernelImage2DWriteArg<Uint>`. `Image2D<ReadOnly, _>` only impls
//! the `Read` variant.

use claspr::{Context, ReadOnly, image::format::R32Uint};

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = claspr_test_image_kernels::dim2_uint::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<ReadOnly, R32Uint>::alloc(&ctx, 4, 4).unwrap();
    let _ = kernels.fill_pattern([4usize, 4usize], img, 4u32, 4u32);
}
