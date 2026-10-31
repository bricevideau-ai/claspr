//! A `Tile` (one part of a split tensor's shared buffer) fits only `Tensor` /
//! `TensorMut` params, whose kernel view reaches only the tile's elements. A
//! plain slice param (`scale_u32`'s `data: &mut [u32]`) would hand the kernel
//! the WHOLE buffer — overlapping the sibling tiles another kernel may be
//! writing — so it must reject a `Tile`.

use claspr::{Context, DeviceSlice, Tensor};
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = kernels::kernels(&ctx).unwrap();
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, 8).unwrap();
    let mut tiles = Tensor::new(buf, [2, 4]).split([2, 1]).unwrap();
    let tile = tiles.pop().unwrap();
    let _ = kernels.scale_u32([4usize], tile, 2u32);
}
