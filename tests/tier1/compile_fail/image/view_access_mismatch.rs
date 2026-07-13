//! `Image1DBuffer<ReadOnly, F>` only impls
//! `KernelImageBufferReadArg`. Passing it to a kernel that
//! requires `KernelImageBufferWriteArg` (`&mut Image!(buffer)`
//! with `image_access="write_only"`) must fail at the trait
//! bound — exactly like the owned-image access-mismatch fixtures.
//!
//! (Previously this used a borrowed `Image1DBufferView`. The view
//! is no longer a kernel arg at all — image args are now reusable
//! `DeviceOp` inputs requiring `'static`, which the borrowed view
//! is not. The owned `Image1DBuffer` carries the same access-marker
//! gating, so the access-mismatch check moves to it.)

use claspr::{image::format::R32Uint, Context, Image1DBuffer, ReadOnly};

fn main() {
    let ctx = Context::any().unwrap();
    let img = Image1DBuffer::<ReadOnly, R32Uint>::alloc(&ctx, 16).unwrap();
    let kernels = claspr_test_image_kernels::dim_buffer_uint::kernels(&ctx).unwrap();
    // fill_pattern is bounded KernelImageBufferWriteArg — ReadOnly image doesn't satisfy.
    let _ = kernels.fill_pattern([16usize], img, 16u32);
}
