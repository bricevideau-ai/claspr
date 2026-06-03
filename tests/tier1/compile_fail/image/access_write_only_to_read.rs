//! `dim2_float::copy_to_buffer` takes `&Image!(2D, type=f32, …)` →
//! the proc-macro bounds it on `KernelImage2DReadArg<Float>`.
//! `Image2D<WriteOnly, _>` only impls the `Write` variant, so the
//! Read bound should fail.

use claspr::{Context, DeviceSlice, WriteOnly, image::format::R32Float};

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = claspr_test_image_kernels::dim2_float::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32Float>::alloc(&ctx, 4, 4).unwrap();
    let out = DeviceSlice::<f32>::from_slice(&ctx, &[0.0; 16]).unwrap();
    let _ = kernels.copy_to_buffer([4usize, 4usize], img, out, 4u32, 4u32);
}
