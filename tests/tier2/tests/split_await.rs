//! Spike scenario 8 — mixed sync/async: run a partial chain to
//! completion, do host work in between, then submit a second chain
//! that consumes the first's output.
//!
//! Validates that buffers flow correctly across a terminal boundary
//! (`.sync` returns the slice, the caller holds it, a second chain
//! takes it back in). The two halves run as independent submissions
//! with deliberate host work between — useful when the host needs to
//! inspect intermediate state to decide what the second half does.

use claspr::Context;
use claspr_async::{DeviceOperation, download, upload};
use claspr_test_kernels::kernels;

const N: usize = 128;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

#[test]
fn split_chain_with_host_decision_between() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // First chain: upload + fill. Buffer flows out via .sync().
    let buf = upload(vec![0u32; N])
        .and_then(|b| kernels.fill_u32([N], b, 5))
        .sync(&ctx)
        .expect("first half");

    // Host decision: pick a scale factor based on something we know
    // about the chain's mid-state (here, the value we filled with).
    let factor = if 5 < 10 { 4 } else { 2 };

    // Second chain: take the buffer back in, scale by the decided
    // factor, download.
    let result: Vec<u32> = kernels
        .scale_u32([N], buf, factor)
        .and_then(download)
        .sync(&ctx)
        .expect("second half");
    assert!(result.iter().all(|&v| v == 20));
}

#[test]
fn split_chain_then_reuse_buffer_for_independent_work() {
    // Split where the host owns the buffer between halves and uses it
    // for something orthogonal (here: an additional fill). Validates
    // that the buffer's identity / refcount survive the chain boundary.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = upload(vec![0u32; N])
        .and_then(|b| kernels.fill_u32([N], b, 1))
        .sync(&ctx)
        .expect("phase 1");
    // Host-side: nothing to do, just hold the buffer.
    let buf = kernels
        .scale_u32([N], buf, 10)
        .wait(&ctx)
        .expect("phase 2 (Tier 1 in the middle)");
    // Pick the second half back up as a Tier 2 chain.
    let result: Vec<u32> = kernels
        .scale_u32([N], buf, 5)
        .and_then(download)
        .sync(&ctx)
        .expect("phase 3");
    assert!(result.iter().all(|&v| v == 50));
}
