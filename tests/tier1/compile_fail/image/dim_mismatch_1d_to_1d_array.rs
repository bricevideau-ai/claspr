//! `dim1_array_uint::fill_pattern` expects a 1D-array
//! (`KernelImage1DArrayWriteArg<Uint>`). `Image1D` impls only
//! `KernelImage1D*Arg` (non-arrayed) — the arrayed bit on the
//! kernel side splits the trait family in two, by design.

use claspr::{Context, WriteOnly, image::format::R32Uint};

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = claspr_test_image_kernels::dim1_array_uint::kernels(&ctx).unwrap();
    let img = claspr::Image1D::<WriteOnly, R32Uint>::alloc(&ctx, 16).unwrap();
    let _ = kernels.fill_pattern([16usize, 4usize], img, 16u32, 4u32);
}
