//! Wide-arity `bundle!` + the multi-output-branch root-cause lock.
//!
//! Two things this file guards that the cutover suite port had dropped:
//!
//! 1. **Arity 2..=16.** The eager layer originally generated only
//!    `bundle2/3/4`; everything wider was expressed by nesting `bundle2`.
//!    `bundle!(a, …)` (the variadic macro) + `bundleN` for N up to 16 are
//!    restored, mirroring the legacy `bundle!`. Here: a flat 8-way and a flat
//!    16-way bundle of pure values, plus an 8-way bundle of device chains.
//!
//! 2. **Multi-output branches.** A bundle/arc_split/fan_out branch that is
//!    ITSELF multi-output (a nested bundle, the `copy_to` pair, a multi-output
//!    kernel) used to fail with `"a branch produced no output"` because the
//!    outer composite drained the branch's single `output_pipe`, which such a
//!    branch never fills. The `collect` gather seam fixes that — a branch runs
//!    its own reconstruction. These tests pin that: a bundle whose branches are
//!    nested bundles, and a bundle whose branch is a `copy_to` chain.

use claspr::Context;
use claspr::bundle;
use claspr::eager::{DeviceOpExt, alloc_zero, bundle2, download, eager_copy_to, upload, value};
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

/// Flat 8-way bundle via the variadic macro — proves arity 8 + the macro's
/// 8-argument arm both exist and reconstruct the tuple in order.
#[test]
fn eager_bundle_macro_arity8() {
    let Some(ctx) = ctx() else { return };
    let (a, b, c, d, e, f, g, h) = bundle!(
        value(1u32),
        value(2u32),
        value(3u32),
        value(4u32),
        value(5u32),
        value(6u32),
        value(7u32),
        value(8u32),
    )
    .sync(&ctx)
    .expect("8-way bundle");
    assert_eq!((a, b, c, d, e, f, g, h), (1, 2, 3, 4, 5, 6, 7, 8));
}

/// Flat 16-way bundle — the widest arity. Exercises the last `impl_eager_bundle!`
/// invocation and the macro's 16-argument arm.
#[test]
fn eager_bundle_macro_arity16() {
    let Some(ctx) = ctx() else { return };
    let t = bundle!(
        value(0u32),
        value(1u32),
        value(2u32),
        value(3u32),
        value(4u32),
        value(5u32),
        value(6u32),
        value(7u32),
        value(8u32),
        value(9u32),
        value(10u32),
        value(11u32),
        value(12u32),
        value(13u32),
        value(14u32),
        value(15u32),
    )
    .sync(&ctx)
    .expect("16-way bundle");
    // Bundle16's Output is a flat 16-tuple. Check a few representative slots.
    let (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) = t;
    assert_eq!(a, 0);
    assert_eq!(h, 7);
    assert_eq!(p, 15);
    let _ = (b, c, d, e, f, g, i, j, k, l, m, n, o);
}

/// 8-way bundle of independent device chains — each branch uploads, fills via a
/// kernel, downloads. Proves wide arity carries real device work + per-branch
/// event threading, not just pure values.
#[test]
fn eager_bundle_macro_arity8_device_chains() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let branch = |seed: u32| {
        let ks = &ks;
        upload(vec![0u32; N])
            .and_then(move |buf| ks.fill_u32([N], buf, seed))
            .and_then(download)
    };
    let (a, b, c, d, e, f, g, h) = bundle!(
        branch(10),
        branch(11),
        branch(12),
        branch(13),
        branch(14),
        branch(15),
        branch(16),
        branch(17),
    )
    .sync(&ctx)
    .expect("8-way device bundle");
    for (out, want) in [
        (a, 10),
        (b, 11),
        (c, 12),
        (d, 13),
        (e, 14),
        (f, 15),
        (g, 16),
        (h, 17),
    ] {
        assert!(out.iter().all(|&v| v == want), "branch seed {want}");
    }
}

/// ROOT-CAUSE LOCK: a bundle whose branches are THEMSELVES multi-output
/// (nested `bundle2`s). Before the `collect` seam this returned
/// `NotSupported("eager bundle: a branch produced no output")`. Each inner
/// bundle reconstructs its own pair via its `collect` override; the outer
/// bundle gathers those reconstructed values.
#[test]
fn bundle_of_multi_output_branches() {
    let Some(ctx) = ctx() else { return };
    let ((a0, a1), (b0, b1)) = bundle!(
        bundle2(value(1u32), value(2u32)),
        bundle2(value(3u32), value(4u32)),
    )
    .sync(&ctx)
    .expect("bundle of bundles");
    assert_eq!((a0, a1, b0, b1), (1, 2, 3, 4));
}

/// ROOT-CAUSE LOCK: a bundle one of whose branches is a `copy_to` chain (the
/// two-output `CopyTo2` op terminated by selecting `dst` via `download`). The
/// branch is single-output at its tail (download) but routes through a
/// multi-output node — confirms `collect` threads correctly through a mixed
/// branch alongside a plain value branch.
#[test]
fn bundle_with_copy_chain_branch() {
    let Some(ctx) = ctx() else { return };

    let src = upload(vec![9u32; N]).sync(&ctx).expect("upload src");
    let dst = alloc_zero::<u32>(N).sync(&ctx).expect("alloc dst");

    let (copied, marker): (Vec<u32>, u32) = bundle!(
        eager_copy_to(src, dst).and_then(|(_src, dst)| download(dst)),
        value(99u32),
    )
    .sync(&ctx)
    .expect("bundle with copy-chain branch");

    assert!(copied.iter().all(|&v| v == 9), "copy moved the bytes");
    assert_eq!(marker, 99);
}
