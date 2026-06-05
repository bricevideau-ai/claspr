//! A `.and_then` follow-up that reaches back to the outer `buf`
//! binding (instead of using the closure's input) must not compile.
//! The first kernel call moves `buf` into the chain; the closure
//! body can only see the buffer that comes out of the upstream
//! `Output` slot.
//!
//! Confirms a chain can't accidentally double-reference a buffer
//! still in flight by reaching past the closure parameter into the
//! outer scope.

use claspr::{Context, DeviceSlice};
use claspr_async::DeviceOperation;
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = kernels::kernels(&ctx).unwrap();
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, 4).unwrap();
    // First op moves `buf`. The closure should use `_first_out`,
    // not the outer `buf` — touching `buf` here is the bug we're
    // catching.
    let _ = kernels
        .fill_u32([4usize], buf, 1u32)
        .and_then(|_first_out| kernels.scale_u32([4usize], buf, 2u32));
}
