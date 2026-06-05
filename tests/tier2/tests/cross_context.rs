//! Cross-**context** buffer flow — REVIEW.md notes this as "where most
//! users will hit problems first." Two separate `Context` instances
//! own disjoint `cl_context` handles, so a `DeviceSlice<T>` allocated
//! in one is *not* valid as a `cl_mem` in the other. The only path
//! between them is host memory: `download` to a `Vec<T>`, then
//! `upload` of that Vec into a chain bound to the other Context.
//!
//! This test pins that contract — if a future refactor accidentally
//! let a `cl_mem` leak across context boundaries (e.g. by caching
//! something at a global rather than per-Context level), the
//! `CL_INVALID_CONTEXT` would surface here.
//!
//! Two `Context::any()` calls will typically land on the same physical
//! device — that's fine; the OpenCL spec guarantees distinct
//! `cl_context` handles regardless, which is the property we need to
//! validate.

use claspr::Context;
use claspr_async::{DeviceOperation, download, upload};
use claspr_test_kernels::kernels;

const N: usize = 64;

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
fn vec_round_trips_between_two_contexts() {
    let Some(ctx_a) = ctx() else { return };
    let Some(ctx_b) = ctx() else { return };
    // Kernels are per-Context — loading twice mirrors what a user
    // managing two contexts would do (each context has its own
    // cl_program/cl_kernel handles built from the same SPIR-V bytes).
    let kernels_a = kernels::kernels(&ctx_a).expect("load kernels on ctx_a");
    let kernels_b = kernels::kernels(&ctx_b).expect("load kernels on ctx_b");

    // Chain 1 on ctx_a: fill → download. Result is a host-owned Vec.
    let intermediate: Vec<u32> = upload!(vec![0u32; N])
        .and_then(|buf| kernels_a.fill_u32([N], buf, 7))
        .and_then(|buf| download!(buf))
        .sync(&ctx_a)
        .expect("chain on ctx_a");
    assert!(intermediate.iter().all(|&v| v == 7));

    // Chain 2 on ctx_b: re-upload that Vec → scale → download.
    // The `cl_mem` allocated by `upload` here is a fresh one in
    // ctx_b's address space — proving the Vec is the only bridge.
    let final_result: Vec<u32> = upload!(intermediate)
        .and_then(|buf| kernels_b.scale_u32([N], buf, 6))
        .and_then(|buf| download!(buf))
        .sync(&ctx_b)
        .expect("chain on ctx_b");
    assert!(final_result.iter().all(|&v| v == 42));
}
