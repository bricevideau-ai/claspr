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
//! 2. **Multi-output branches — structure-preserving + re-arming.** A
//!    bundle/arc_split/fan_out branch that is ITSELF multi-output (a nested
//!    bundle, the `copy_to` pair, a multi-output kernel) once failed with
//!    `"a branch produced no output"` (the outer composite drained the branch's
//!    single `output_pipe`, which such a branch never fills). Now the bundle
//!    DELEGATES `gather_checkouts` to each branch, so a multi-output branch
//!    contributes its OWN tuple of per-buffer `Checkout`s (grouped-by-branch,
//!    not collapsed to one `Checkout` over the whole branch tuple) WITH its own
//!    per-buffer return homes. These tests pin: nested-bundle branches; a
//!    `copy_to` chain branch; the ADVERSARIAL re-arm (two multi-output kernel
//!    branches, sync ×2, stable handles); and TRANSITIVE nesting (a bundle whose
//!    branch is itself a multi-output bundle, re-arming at every level).

use claspr::bundle;
use claspr::eager::{DeviceOpExt, alloc_zero, bundle2, download, eager_copy_to, upload, value};
use claspr_test_kernels::kernels;
use claspr_test_support::{N, ctx, handle_of, seeded};

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
    assert_eq!((*a, *b, *c, *d, *e, *f, *g, *h), (1, 2, 3, 4, 5, 6, 7, 8));
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
    assert_eq!(*a, 0);
    assert_eq!(*h, 7);
    assert_eq!(*p, 15);
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
/// bundle reconstructs its own pair via its `gather_checkouts` override; the
/// outer bundle DELEGATES to each branch's own `gather_checkouts`.
///
/// STRUCTURE-PRESERVING `Checkouts`: a multi-output branch contributes ITS
/// tuple of `Checkout`s, not one collapsed `Checkout` over the whole branch
/// tuple. So the outer shape is `((Checkout, Checkout), (Checkout, Checkout))`
/// — grouped-by-branch, per-buffer within each branch. Each inner buffer is
/// individually accessible (no `into_inner`-the-whole-branch needed).
#[test]
fn bundle_of_multi_output_branches() {
    let Some(ctx) = ctx() else { return };
    let ((a0, a1), (b0, b1)) = bundle!(
        bundle2(value(1u32), value(2u32)),
        bundle2(value(3u32), value(4u32)),
    )
    .sync(&ctx)
    .expect("bundle of bundles");
    assert_eq!((*a0, *a1, *b0, *b1), (1, 2, 3, 4));
}

/// ROOT-CAUSE LOCK: a bundle one of whose branches is a `copy_to` chain (the
/// two-output `CopyTo2` op terminated by selecting `dst` via `download`). The
/// branch is single-output at its tail (download) but routes through a
/// multi-output node — confirms `collect` threads correctly through a mixed
/// branch alongside a plain value branch.
#[test]
fn bundle_with_copy_chain_branch() {
    let Some(ctx) = ctx() else { return };

    // `into_inner` to concrete buffers: `eager_copy_to`'s `Src: CopyTo<Dst>` bound
    // can't infer the marker from a `Checkout` (multiple `From<Checkout<…>>` impls).
    let src = upload(vec![9u32; N])
        .sync(&ctx)
        .expect("upload src")
        .into_inner();
    let dst = alloc_zero::<u32>(N)
        .sync(&ctx)
        .expect("alloc dst")
        .into_inner();

    let (copied, marker) = bundle!(
        eager_copy_to(src, dst).and_then(|(_src, dst)| download(dst)),
        value(99u32),
    )
    .sync(&ctx)
    .expect("bundle with copy-chain branch");

    assert!(copied.iter().all(|&v| v == 9), "copy moved the bytes");
    assert_eq!(*marker, 99);
}

/// ADVERSARIAL RE-ARM LOCK — the fixed limitation, proven not papered.
///
/// A `bundle2` of TWO multi-output branches (each `add_u32(a, b, out)` → the
/// per-buffer `(Checkout<a>, Checkout<b>, Checkout<out>)`). This is exactly the
/// case the old design collapsed to `home == None` ("multi-output branch doesn't
/// re-arm through a bundle"). With structure-preserving delegation each branch
/// contributes its OWN per-buffer Checkouts with its OWN homes, so it must:
///   (a) hand back per-buffer Checkouts, individually accessible/droppable,
///   (b) keep STABLE cl_mem handles across two syncs (the re-home), and
///   (c) compute correct data on run 2 (re-seeded operands into stable buffers).
/// Any buffer that does NOT re-home is a FAIL, not a footnote.
#[test]
fn bundle_of_two_multi_output_branches_rearms_x2_stable_handles() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Six caller-owned buffers (two branches × 3 operands), each a distinct cell
    // that must re-arm. Left computes 1+2=3, right computes 10+20=30.
    let la = seeded(&ctx, 1);
    let lb = seeded(&ctx, 2);
    let lo = seeded(&ctx, 0);
    let ra = seeded(&ctx, 10);
    let rb = seeded(&ctx, 20);
    let ro = seeded(&ctx, 0);

    // Snapshot every buffer's stable handle BEFORE the graph runs.
    let (h_la, h_lb, h_lo) = (handle_of(&la), handle_of(&lb), handle_of(&lo));
    let (h_ra, h_rb, h_ro) = (handle_of(&ra), handle_of(&rb), handle_of(&ro));

    // Build once (set-once binds fold the concrete buffers in). REUSABLE `&g`.
    let g = bundle2(ks.add_u32([N], la, lb, lo), ks.add_u32([N], ra, rb, ro));

    // ── RUN 1 ── the shape IS per-branch, per-buffer within each branch:
    // `((Checkout<a>, Checkout<b>, Checkout<out>), (Checkout, Checkout, Checkout))`.
    let ((la1, lb1, lo1), (ra1, rb1, ro1)) = g.sync(&ctx).expect("run 1");

    // (a) each per-buffer Checkout is individually accessible. `map()` (via Deref
    // to DeviceSlice, `&self`) reads WITHOUT severing, so the Checkout still
    // re-homes on drop.
    assert!(
        (*lo1)
            .map()
            .wait()
            .expect("map left out r1")
            .iter()
            .all(|&v| v == 3),
        "run 1: left 1+2=3"
    );
    assert!(
        (*ro1)
            .map()
            .wait()
            .expect("map right out r1")
            .iter()
            .all(|&v| v == 30),
        "run 1: right 10+20=30"
    );

    // Handles observed on run 1 match the originals (lent from their cells).
    assert_eq!(handle_of(&*la1), h_la, "r1 la");
    assert_eq!(handle_of(&*lb1), h_lb, "r1 lb");
    assert_eq!(handle_of(&*lo1), h_lo, "r1 lo");
    assert_eq!(handle_of(&*ra1), h_ra, "r1 ra");
    assert_eq!(handle_of(&*rb1), h_rb, "r1 rb");
    assert_eq!(handle_of(&*ro1), h_ro, "r1 ro");

    // (a') drop ONE buffer of ONE branch individually (left's `b`), map another
    // (right's `out`) — proving per-buffer granularity WITHIN a branch — then drop
    // the rest UNREAD so every cell re-arms (Lent → Bound).
    drop(lb1);
    assert!(
        (*ro1)
            .map()
            .wait()
            .expect("map right out again")
            .iter()
            .all(|&v| v == 30)
    );
    drop((la1, lo1, ra1, rb1, ro1));

    // ── RUN 2 ── the SAME graph re-runs. If any buffer failed to re-home this
    // would error "graph busy" or re-mint a fresh cl_mem.
    let ((la2, lb2, lo2), (ra2, rb2, ro2)) = g.sync(&ctx).expect("run 2 — bundle re-armed");

    // (c) correct data on run 2 (operands re-seeded into the stable buffers).
    assert!(
        (*lo2)
            .map()
            .wait()
            .expect("map left out r2")
            .iter()
            .all(|&v| v == 3),
        "run 2: left 1+2=3"
    );
    assert!(
        (*ro2)
            .map()
            .wait()
            .expect("map right out r2")
            .iter()
            .all(|&v| v == 30),
        "run 2: right 10+20=30"
    );

    // (b) STABLE handles across both syncs — every buffer re-homed to its cell.
    assert_eq!(handle_of(&*la2), h_la, "re-arm: left a stable");
    assert_eq!(handle_of(&*lb2), h_lb, "re-arm: left b stable");
    assert_eq!(handle_of(&*lo2), h_lo, "re-arm: left out stable");
    assert_eq!(handle_of(&*ra2), h_ra, "re-arm: right a stable");
    assert_eq!(handle_of(&*rb2), h_rb, "re-arm: right b stable");
    assert_eq!(handle_of(&*ro2), h_ro, "re-arm: right out stable");
}

/// NESTED-SEAM PROBE — a bundle whose branch is ITSELF a multi-output bundle.
/// `bundle2(bundle2(k1, k2), k3)`: the outer branch 0 is a 2-branch bundle of two
/// multi-output kernels. The delegation must compose TRANSITIVELY — nested-by-
/// branch all the way down — AND re-arm at every level. Shape:
/// `(((C,C,C),(C,C,C)), (C,C,C))`. Two syncs, stable handles throughout.
#[test]
fn nested_bundle_of_multi_output_branches_rearms_transitively() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Three multi-output kernel branches; k1,k2 live under an inner bundle.
    let (a1, b1, o1) = (seeded(&ctx, 1), seeded(&ctx, 2), seeded(&ctx, 0));
    let (a2, b2, o2) = (seeded(&ctx, 3), seeded(&ctx, 4), seeded(&ctx, 0));
    let (a3, b3, o3) = (seeded(&ctx, 5), seeded(&ctx, 6), seeded(&ctx, 0));
    let h = [
        handle_of(&a1),
        handle_of(&b1),
        handle_of(&o1),
        handle_of(&a2),
        handle_of(&b2),
        handle_of(&o2),
        handle_of(&a3),
        handle_of(&b3),
        handle_of(&o3),
    ];

    let g = bundle2(
        bundle2(ks.add_u32([N], a1, b1, o1), ks.add_u32([N], a2, b2, o2)),
        ks.add_u32([N], a3, b3, o3),
    );

    // Run 1 — the nested-by-branch shape destructures all the way down.
    let (((ia1, ib1, io1), (ia2, ib2, io2)), (oa3, ob3, oo3)) = g.sync(&ctx).expect("run 1");
    assert!(
        (*io1).map().wait().unwrap().iter().all(|&v| v == 3),
        "inner k1: 1+2=3"
    );
    assert!(
        (*io2).map().wait().unwrap().iter().all(|&v| v == 7),
        "inner k2: 3+4=7"
    );
    assert!(
        (*oo3).map().wait().unwrap().iter().all(|&v| v == 11),
        "outer k3: 5+6=11"
    );
    drop((ia1, ib1, io1, ia2, ib2, io2, oa3, ob3, oo3));

    // Run 2 — re-arm at every nesting level, stable handles throughout.
    let (((ia1, ib1, io1), (ia2, ib2, io2)), (oa3, ob3, oo3)) = g
        .sync(&ctx)
        .expect("run 2 — nested bundle re-armed at every level");
    assert!(
        (*io2).map().wait().unwrap().iter().all(|&v| v == 7),
        "run 2 inner k2: 3+4=7"
    );
    let got = [
        handle_of(&*ia1),
        handle_of(&*ib1),
        handle_of(&*io1),
        handle_of(&*ia2),
        handle_of(&*ib2),
        handle_of(&*io2),
        handle_of(&*oa3),
        handle_of(&*ob3),
        handle_of(&*oo3),
    ];
    assert_eq!(
        got, h,
        "every buffer at every nesting level re-homed to a stable cl_mem"
    );
}
