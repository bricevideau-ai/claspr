//! Tier 2 coverage for f64 arg-marshalling: same fp64 kernels as
//! `tests/tier1/tests/fp64.rs`, composed via async combinators.
//! Validates the proc-macro emits an equivalent Tier 2 surface for
//! f64 as for u32, and that the f64 chain runs through `upload` /
//! `download` cleanly.
//!
//! Skips when the device doesn't advertise `Float64` (most GPUs ship
//! it; llvmpipe needs `RUSTICL_FEATURES=fp64`, set in CI).

use claspr::Context;
use claspr_async::{DeviceOperation, download, upload};
use claspr_test_kernels::kernels_f64 as kernels;

const N: usize = 256;

fn ctx_with_f64() -> Option<Context> {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return None;
    };
    match ctx.device().cl3().double_fp_config() {
        Ok(mask) if mask != 0 => Some(ctx),
        _ => {
            eprintln!("SKIP: device has no Float64 capability");
            None
        }
    }
}

#[test]
fn f64_chain_via_async_combinators() {
    let Some(ctx) = ctx_with_f64() else { return };
    let kernels = kernels::kernels(&ctx).expect("kernels load");

    // upload!(0.0s) → fill(1.0) → scale(4.0) → download → expect 4.0
    // The upload writes finite zeros so the fill_f64's `data[i] * 0.0`
    // sees a defined value.
    let result: Vec<f64> = upload!(vec![0.0f64; N])
        .and_then(|buf| kernels.fill_f64([N], buf, 1.0))
        .and_then(|buf| kernels.scale_f64([N], buf, 4.0))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("f64 chain sync");

    assert_eq!(result.len(), N);
    for (i, &v) in result.iter().enumerate() {
        assert_eq!(v, 4.0, "element {i} mismatch");
    }
}
