//! A handle pipe is a graph edge drained by exactly one op, so `Pipe` is not
//! `Clone`: duplicating a handle to feed one value to two consumers must not
//! compile (the second consumer would find the value already taken). One value
//! read by several ops is shared with `.arc()` / `arc_split` instead.

use claspr::prelude::*;
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let ks = kernels::kernels(&ctx).unwrap();
    let buf = DeviceSlice::<u32>::from_slice(&ctx, &[1; 4]).unwrap();
    let _ = ks.scale_u32([4usize], buf, 2u32).and_then(|x| {
        // Duplicating the handle must be rejected.
        let x2 = x.clone();
        bundle!(
            ks.scale_u32([4usize], x, 3u32),
            ks.scale_u32([4usize], x2, 5u32)
        )
    });
}
