//! An `.and_then` follow-up that reaches back to the outer `buf` binding
//! (instead of using the upstream handle the closure is handed) must not
//! compile. The first kernel call moves `buf` into the chain; the closure body
//! can only see the buffer that comes out of the upstream slot.
//!
//! Confirms a chain can't accidentally double-reference a buffer still in
//! flight by reaching past the closure parameter into the outer scope.
//!
//! Unified-API restatement of the deleted `sequential_use_after_move` fixture:
//! `and_then` is now a `DeviceOpExt` verb whose closure receives the upstream's
//! `Handle` slot; reaching back to the moved outer `buf` is still a
//! use-after-move.

use claspr::prelude::*;
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = kernels::kernels(&ctx).unwrap();
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, 4).unwrap();
    // First op moves `buf`. The closure should use `_first_out`, not the outer
    // `buf` — touching `buf` here is the bug we're catching.
    let _ = kernels
        .fill_u32([4usize], buf, 1u32)
        .and_then(|_first_out| kernels.scale_u32([4usize], buf, 2u32));
    let _ = ctx;
}
