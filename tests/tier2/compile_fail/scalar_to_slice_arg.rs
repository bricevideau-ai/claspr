//! Strict scalar-ref binding, direction 2 (#208): a `DeviceScalar` must
//! NOT bind to a `&[T]` slice kernel arg.
//!
//! `DeviceScalar` (`Scalar<DeviceSlice>`) impls the scalar-ref traits
//! but NOT the slice traits (`KernelSliceReadArg` / `…ReadWriteArg`), so
//! passing it to `scale_u32`'s `data: &mut [u32]` slice arg must be a
//! compile error — the exclusion is enforced in BOTH directions.

use claspr::{Context, DeviceScalar};
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = kernels::kernels(&ctx).unwrap();
    let scalar = DeviceScalar::<u32>::new(&ctx, 2).unwrap();
    // `data: &mut [u32]` wants `KernelSliceReadWriteArg<u32>` — a
    // `DeviceScalar` does not impl the slice traits, so this must reject.
    let _ = kernels.scale_u32([1usize], scalar, 2u32);
}
