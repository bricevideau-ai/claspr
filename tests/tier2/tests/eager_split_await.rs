//! Eager-API port of `split_await.rs`: run a partial chain to completion, do
//! host work in between, then submit a second chain that consumes the first's
//! output. Validates buffers flow correctly across a terminal boundary.
//!
//! Despite the file name, the original uses NO async terminal — it terminates
//! with `.sync()` (and one Tier-1 `.wait()` mid-stream). Both map directly onto
//! the eager `.sync(&ctx)` terminal:
//!   `upload!(v)`              → `upload(v)`
//!   `download!(buf)`          → `.and_then(download)`
//!   Tier-1 `kernel(...).wait()` → eager `kernel(...).sync(&ctx)` (single-output
//!                                kernel `.sync()` yields the `DeviceSlice`,
//!                                same "run to completion, hand back the buffer"
//!                                semantics as the Tier-1 `.wait()`).
//! Both test fns port 1:1 — same N, values, assertions.

use claspr::eager::{DeviceOpExt, download, upload};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const N: usize = 128;

#[test]
fn split_chain_with_host_decision_between() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // First chain: upload + fill. Buffer flows out via .sync().
    let buf = upload(vec![0u32; N])
        .and_then(|b| kernels.fill_u32([N], b, 5))
        .sync(&ctx)
        .expect("first half");

    // Host decision: pick a scale factor based on something we know about the
    // chain's mid-state (here, the value we filled with).
    let factor = if 5 < 10 { 4 } else { 2 };

    // Second chain: take the buffer back in, scale by the decided factor.
    let result = kernels
        .scale_u32([N], buf, factor)
        .and_then(download)
        .sync(&ctx)
        .expect("second half");
    assert!(result.iter().all(|&v| v == 20));
}

#[test]
fn split_chain_then_reuse_buffer_for_independent_work() {
    // Split where the host owns the buffer between halves and uses it for
    // something orthogonal. Validates the buffer's identity / refcount survive
    // the chain boundary.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = upload(vec![0u32; N])
        .and_then(|b| kernels.fill_u32([N], b, 1))
        .sync(&ctx)
        .expect("phase 1");
    // Mid-stream "Tier 1"-style run-to-completion: the kernel Op is a
    // concrete-head `DeviceOp` (it owns its buffer arg), so its no-launcher
    // `.wait()` recovers the context from the buffer and yields it back (the
    // single-output kernel's `Output = DeviceSlice`) — the restored Tier-1
    // terminal, no `&ctx`.
    let buf = kernels
        .scale_u32([N], buf, 10)
        .wait()
        .expect("phase 2 (run-to-completion in the middle)");
    // Pick the second half back up as an eager chain.
    let result = kernels
        .scale_u32([N], buf, 5)
        .and_then(download)
        .sync(&ctx)
        .expect("phase 3");
    assert!(result.iter().all(|&v| v == 50));
}
