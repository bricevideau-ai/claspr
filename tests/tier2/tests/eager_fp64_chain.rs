//! Eager port of `fp64_chain.rs`: f64 arg-marshalling through the eager graph
//! API (`claspr::eager`). Same fp64 kernels as `tests/tier1/tests/fp64.rs`,
//! same N, same fill/scale values, same final assertion — proving the eager
//! surface marshals f64 exactly as it does u32.
//!
//! Old → new mapping (identical to the u32 chains in eager_chain.rs):
//!   `upload!(v)`     → `upload::<f64, claspr::ReadWrite, _>(v)`
//!   `download!(buf)` → `download` (terminal `.and_then(download).sync()`)
//!
//! Skips when the device doesn't advertise `Float64` (llvmpipe needs
//! `RUSTICL_FEATURES=fp64`, set in CI). Guard preserved verbatim.

use claspr::Context;
use claspr::eager::{EagerOpExt, download, upload};
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

/// fp64_chain.rs::f64_chain_via_async_combinators — upload(0.0) → fill(1.0) →
/// scale(4.0) → download → expect 4.0.
#[test]
fn f64_chain_via_async_combinators() {
    let Some(ctx) = ctx_with_f64() else { return };
    let kernels = kernels::kernels(&ctx).expect("kernels load");

    let result: Vec<f64> = upload::<f64, claspr::ReadWrite, _>(vec![0.0f64; N])
        .and_then(|buf| kernels.fill_f64([N], buf, 1.0))
        .and_then(|buf| kernels.scale_f64([N], buf, 4.0))
        .and_then(download)
        .sync(&ctx)
        .expect("f64 chain sync");

    assert_eq!(result.len(), N);
    for (i, &v) in result.iter().enumerate() {
        assert_eq!(v, 4.0, "element {i} mismatch");
    }
}
