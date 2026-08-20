//! Eager-cluster companion to `svm_chain.rs`: SVM (`MappedSlice<T>`) threaded as
//! a kernel input, exercising auto-registration on every launch + drop ordering.
//!
//! Both tests in the source `svm_chain.rs` are already **pure Tier 1** (the
//! original framing as a Tier-2 `with_context` chain was removed long ago — see
//! the source doc comments): they use `.submit()` / `.wait()` / `.after()` /
//! `.map()` directly, with `&ctx` as the launcher. There is no Tier-2 combinator
//! chain to translate to the eager API, so these are reproduced verbatim. They
//! belong to the same device/svm cluster's coverage and are included here for
//! completeness of the eager-cutover suite.
//!
//! Skips on devices without SVM. Guard preserved verbatim.

use claspr::MappedSlice;
use claspr_test_kernels::kernels;
use claspr_test_support::ctx_with_svm;

const N: usize = 256;

/// svm_chain.rs::mapped_slice_threads_through_typed_launchers — Tier-1
/// fill(6) → scale(7) via submit/after/wait, sum readback. 6 * 7 * N.
#[test]
fn mapped_slice_threads_through_typed_launchers() {
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let (buf, fill_evt) = kernels.fill_u32([N], buf, 6u32).submit().expect("fill");
    let buf = kernels
        .scale_u32([N], buf, 7u32)
        .after(fill_evt)
        .wait()
        .expect("scale");

    let g = buf.map().wait().expect("map");
    let result_sum: u32 = g.iter().copied().sum();
    drop(g);

    assert_eq!(result_sum, 6 * 7 * N as u32);
}

/// svm_chain.rs::many_in_flight_svm_launches_drop_safely — 8 successive scales,
/// no host sync, then drop; the SVMFree wait-list must include every launch.
#[test]
fn many_in_flight_svm_launches_drop_safely() {
    let Some(ctx) = ctx_with_svm() else { return };

    let mut buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    for _ in 0..8 {
        let (returned, _evt) = kernels.scale_u32([N], buf, 1u32).submit().expect("scale");
        buf = returned;
    }
    drop(buf);

    assert_eq!(ctx.error_count(), 0);
}
