//! Strict scalar-ref binding, direction 1 (#208): a bare `DeviceSlice`
//! must NOT bind to a `&T` scalar-by-reference kernel arg.
//!
//! Before #208 a length-1 `DeviceSlice` satisfied a `&T` arg (it reused
//! `KernelSliceReadArg`). #208 gives scalar-refs a DEDICATED
//! `KernelScalarRefArg` trait impl'd ONLY for `DeviceScalar` (all memory
//! tiers), so a slice — of any length — no longer binds. Passing a
//! `DeviceSlice` to `scale_by_ref_u32`'s `factor: &u32` must be a
//! compile error.

use claspr::{Context, DeviceSlice};
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = kernels::kernels(&ctx).unwrap();
    let data = DeviceSlice::<u32>::alloc_zero(&ctx, 4).unwrap();
    let factor = DeviceSlice::<u32>::alloc_zero(&ctx, 1).unwrap();
    // `factor: &u32` wants `KernelScalarRefArg<u32>`, impl'd only for
    // `DeviceScalar<u32>` — a `DeviceSlice` must be rejected.
    let _ = kernels.scale_by_ref_u32([4usize], data, factor);
}
