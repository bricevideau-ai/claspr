//! `Image1DBufferView<'_, ReadOnly, F>` only impls
//! `KernelImageBufferReadArg`. Passing it to a kernel that
//! requires `KernelImageBufferWriteArg` (`&mut Image!(buffer)`
//! with `image_access="write_only"`) must fail at the trait
//! bound — exactly like passing `Image1DBuffer<ReadOnly>` does.

use claspr::{image::format::R32Uint, Context, DeviceSlice, Image1DBufferView, ReadOnly};

fn main() {
    let ctx = Context::any().unwrap();
    let slice = DeviceSlice::<u32, ReadOnly>::from_slice(&ctx, &[0u32; 16]).unwrap();
    let view = Image1DBufferView::<ReadOnly, R32Uint>::view_of(&slice).unwrap();
    let kernels = claspr_test_image_kernels::dim_buffer_uint::kernels(&ctx).unwrap();
    // fill_pattern is bounded KernelImageBufferWriteArg — ReadOnly view doesn't satisfy.
    let _ = kernels.fill_pattern([16usize], view, 16u32);
}
