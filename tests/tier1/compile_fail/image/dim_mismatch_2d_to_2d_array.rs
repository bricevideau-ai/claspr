//! Same shape as `dim_mismatch_1d_to_1d_array` but for 2D —
//! passing a non-arrayed `Image2D` to a kernel that expects
//! a 2D-array is rejected at the trait bound.

use claspr::{Context, WriteOnly, image::format::R32Uint};

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = claspr_test_image_kernels::dim2_array_uint::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32Uint>::alloc(&ctx, 8, 4).unwrap();
    let _ = kernels.fill_pattern([8usize, 4usize, 3usize], img, 8u32, 4u32, 3u32);
}
