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

use claspr::Error;
use claspr::eager::{DeviceOpExt, download, scalar_value, upload};
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

// ── Reseed-failure recovery (2026-08-20 adversarial-review Concern) ─────────
//
// Replaying a graph on the WRONG context is the deterministic way to make an
// upload leaf's reseed enqueue fail: the reseed pairs the replay context's
// queue with the seeding context's `cl_mem`, which the spec mandates rejecting
// with `CL_INVALID_CONTEXT`. The contract under test: the failure surfaces as
// the REAL error AND rehomes the taken buffer. Pre-fix the buffer was dropped
// on the `?`, leaving cell-empty + seeded — every later run (including back on
// the correct context) then misreported "already lent … the graph is busy",
// permanently stranding the graph on a potentially transient enqueue error.

/// Slice path (`Upload::execute` reseed arm): seed on ctx_a, fail a replay on
/// ctx_b, then prove the graph still replays cleanly on ctx_a.
#[test]
fn failed_upload_reseed_rehomes_instead_of_stranding() {
    let Some(ctx_a) = ctx() else { return };
    let Some(ctx_b) = ctx() else { return };

    let g = upload(vec![7u32; N]).and_then(download);

    // Run 1 on ctx_a: seeds the persistent upload buffer (owned by ctx_a).
    let out = g.sync(&ctx_a).expect("run 1 on ctx_a");
    assert!(out.iter().all(|&v| v == 7));
    drop(out);

    // Run 2 on ctx_b: the RW upload reseeds on replay; enqueuing ctx_a's
    // buffer on ctx_b's queue must fail with the real enqueue error — NOT the
    // busy-graph diagnosis. Intel legacy NEO doesn't enforce
    // `CL_INVALID_CONTEXT` (the cross-context enqueue silently succeeds — the
    // same laxness that accepts truncated SPIR-V, see tier1/errors.rs), so on
    // a driver where the injection vector doesn't exist, SKIP loudly.
    let Err(err) = g.sync(&ctx_b) else {
        eprintln!("SKIP: driver accepts cross-context enqueue; cannot force a reseed failure");
        return;
    };
    assert!(
        !matches!(&err, Error::NotSupported(m) if m.contains("already lent")),
        "run 2 must surface the real enqueue error, got {err:?}"
    );

    // Run 3 back on ctx_a: the failed reseed rehomed the buffer, so the graph
    // replays (pre-fix: NotSupported "already lent … the graph is busy").
    let out = g
        .sync(&ctx_a)
        .expect("run 3 on ctx_a — graph must not be stranded by a failed reseed");
    assert!(out.iter().all(|&v| v == 7));
}

/// Scalar path (`ScalarUpload::execute` reseed arm): same shape over a
/// length-1 device scalar. (The USM leaf shares the fix but its reseed is a
/// pure host copy with no cross-context failure mode to force.)
#[test]
fn failed_scalar_reseed_rehomes_instead_of_stranding() {
    let Some(ctx_a) = ctx() else { return };
    let Some(ctx_b) = ctx() else { return };

    let g = scalar_value(5u32);

    // Run 1 on ctx_a, then rehome (a bare leaf's Checkout holds the scalar).
    let co = g.sync(&ctx_a).expect("run 1 on ctx_a");
    assert_eq!(co.read_value().expect("read run 1"), 5u32);
    drop(co);

    // Same lax-driver gate as the slice test above.
    let Err(err) = g.sync(&ctx_b) else {
        eprintln!("SKIP: driver accepts cross-context enqueue; cannot force a reseed failure");
        return;
    };
    assert!(
        !matches!(&err, Error::NotSupported(m) if m.contains("already lent")),
        "run 2 must surface the real enqueue error, got {err:?}"
    );

    let co = g
        .sync(&ctx_a)
        .expect("run 3 on ctx_a — graph must not be stranded by a failed reseed");
    assert_eq!(co.read_value().expect("read run 3"), 5u32);
}
