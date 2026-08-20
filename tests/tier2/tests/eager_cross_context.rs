//! Eager port of `cross_context.rs`: cross-**context** buffer flow through the
//! eager graph API. Two separate `Context` instances own disjoint `cl_context`
//! handles, so a `DeviceSlice<T>` allocated in one is *not* valid as a `cl_mem`
//! in the other. The only bridge is host memory: `download` to a `Vec<T>`, then
//! `upload` of that Vec into a chain bound to the other Context.
//!
//! Old → new mapping:
//!   `upload!(v)`     → `upload(v)`
//!   `download!(buf)` → `download` (terminal `.and_then(download).sync(&ctx)`)
//!
//! Pins the same contract: if a future refactor let a `cl_mem` leak across
//! context boundaries, `CL_INVALID_CONTEXT` would surface here.

use claspr::eager::{DeviceOpExt, download, upload};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const N: usize = 64;

/// cross_context.rs::vec_round_trips_between_two_contexts — chain 1 on ctx_a
/// (fill 7 → download), then chain 2 on ctx_b (re-upload that Vec → scale 6 →
/// download). The Vec is the only thing crossing the context boundary.
#[test]
fn vec_round_trips_between_two_contexts() {
    let Some(ctx_a) = ctx() else { return };
    let Some(ctx_b) = ctx() else { return };
    // Kernels are per-Context — loading twice mirrors what a user managing two
    // contexts would do (each has its own cl_program/cl_kernel from the same
    // SPIR-V bytes).
    let kernels_a = kernels::kernels(&ctx_a).expect("load kernels on ctx_a");
    let kernels_b = kernels::kernels(&ctx_b).expect("load kernels on ctx_b");

    // Chain 1 on ctx_a: fill → download. Result is a host-owned Vec.
    let intermediate = upload(vec![0u32; N])
        .and_then(|buf| kernels_a.fill_u32([N], buf, 7))
        .and_then(download)
        .sync(&ctx_a)
        .expect("chain on ctx_a")
        .into_inner();
    assert!(intermediate.iter().all(|&v| v == 7));

    // Chain 2 on ctx_b: re-upload that Vec → scale → download. The `cl_mem`
    // allocated by `upload` here is fresh in ctx_b's address space — proving the
    // Vec is the only bridge.
    let final_result = upload(intermediate)
        .and_then(|buf| kernels_b.scale_u32([N], buf, 6))
        .and_then(download)
        .sync(&ctx_b)
        .expect("chain on ctx_b");
    assert!(final_result.iter().all(|&v| v == 42));
}
