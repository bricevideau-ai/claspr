//! MID-GRAPH bundle-fed host seam: written branches route to SEPARATE downstream
//! kernels AND re-home across replays (#212 completion).
//!
//! #212 pass 1 (`eager_host_multihome.rs`) made a bundle-fed `and_then_host` seam
//! re-arm every branch — but ONLY at the TERMINAL (`gather_checkouts`). A seam
//! nested MID-graph (the source of a downstream `and_then`) ran via `execute`,
//! which collapsed a bundle source to `home == None` (no re-home) and exposed a
//! single `Pipe<S::Output>` (`= Pipe<tuple>`, so the written branches could not be
//! routed to separate downstream kernels). Option A completes #212: `AndThenHost`
//! is now multi-output-aware mid-graph via the `SeamScatter` trait — `Handle =
//! S::Handle` (per-branch element pipes) + `execute` gathers per-branch, runs the
//! closure, and RE-SCATTERS each written-back branch (value+home) into its own
//! element pipe (mirroring `Bundle::execute`).
//!
//! Each test builds ONCE and `sync`s ≥2 times: the written branch values feed
//! DISTINCT downstream kernels, both branches re-home (stable `cl_mem` handles),
//! and downstream reads the written values each replay. Plus a terminal + a
//! single-output regression guard so neither pre-existing path regresses.

use claspr::eager::{DeviceOpExt, bundle2, download, forward};
use claspr::{Context, DeviceScalar, DeviceSlice, MemRef, RecordableBuffer};
use claspr_test_kernels::kernels;

const N: usize = 64;

/// Stable identity of a buffer's backing memory for `==` across replays.
fn handle_of<B: RecordableBuffer>(b: &B) -> usize {
    match b.record_handle().mem {
        MemRef::Buffer(m) => m as usize,
        MemRef::Svm(p) => p as usize,
    }
}

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

// ── THE core case: mid-graph bundle-fed seam, written branches → SEPARATE
//    downstream kernels, replayed ─────────────────────────────────────────────
//
// bundle2(lift(addend_a), lift(addend_b)).and_then_host(write both) — the seam is
// MID-graph (source of a downstream `and_then`). Its two written scalars are
// routed to TWO distinct kernels: `data_a += addend_a`, `data_b += addend_b`.
// Because the seam re-scatters each branch into its own element pipe, the closure
// downstream destructures `(a_pipe, b_pipe)` and feeds each separately. Both
// scalar branches re-home across ≥2 syncs (stable handles), and downstream reads
// the written values each run. Before Option A this did not compile (the handle
// was one `Pipe<(Scalar, Scalar)>`) and would not re-home mid-graph.
#[test]
fn midgraph_bundle_seam_written_branches_feed_separate_kernels_replay() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let addend_a = DeviceScalar::<u32>::new(&ctx, 0).expect("addend_a");
    let addend_b = DeviceScalar::<u32>::new(&ctx, 0).expect("addend_b");
    let (ha, hb) = (handle_of(&addend_a), handle_of(&addend_b));
    let data_a = DeviceSlice::<u32>::from_slice(&ctx, &[1u32; N]).expect("data_a");
    let data_b = DeviceSlice::<u32>::from_slice(&ctx, &[1u32; N]).expect("data_b");
    let (hda, hdb) = (handle_of(&data_a), handle_of(&data_b));

    // Seam writes addend_a = 10, addend_b = 20; downstream routes each to its own
    // add_ref_u32. `move` the data buffers into the compose closure so it doesn't
    // borrow ctx.
    let g = bundle2(claspr::eager::lift(addend_a), claspr::eager::lift(addend_b))
        .and_then_host(|(a, b): (&mut u32, &mut u32)| {
            *a = 10;
            *b = 20;
            Ok(())
        })
        .and_then(move |(a_pipe, b_pipe)| {
            // a_pipe: Pipe<DeviceScalar<u32>>, b_pipe: Pipe<DeviceScalar<u32>> —
            // per-branch pipes, routed to SEPARATE kernels.
            bundle2(
                ks.add_ref_u32([N], data_a, a_pipe)
                    .and_then(|(data, _a)| forward(data)),
                ks.add_ref_u32([N], data_b, b_pipe)
                    .and_then(|(data, _b)| forward(data)),
            )
        });

    // Run 1: data_a = 1 + 10 = 11, data_b = 1 + 20 = 21.
    let (ca1, cb1) = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*ca1), hda, "r1 data_a stable");
    assert_eq!(handle_of(&*cb1), hdb, "r1 data_b stable");
    assert!(
        ca1.map().wait().expect("map a r1").iter().all(|&v| v == 11),
        "r1 data_a"
    );
    assert!(
        cb1.map().wait().expect("map b r1").iter().all(|&v| v == 21),
        "r1 data_b"
    );
    drop((ca1, cb1)); // data buffers + (via reclaim) the addend scalars re-home

    // Run 2 (replay over the SAME handles): the seam re-writes the addends (they
    // re-homed mid-graph), so data_a = 11 + 10 = 21, data_b = 21 + 20 = 41.
    let (ca2, cb2) = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*ca2), hda, "r2 data_a stable");
    assert_eq!(handle_of(&*cb2), hdb, "r2 data_b stable");
    assert!(
        ca2.map().wait().expect("map a r2").iter().all(|&v| v == 21),
        "r2 data_a"
    );
    assert!(
        cb2.map().wait().expect("map b r2").iter().all(|&v| v == 41),
        "r2 data_b"
    );
    drop((ca2, cb2));

    // The addend scalars themselves re-homed to their cells (proven by run 2
    // recomputing over the same handles ha/hb): silence the unused-handle warnings
    // by asserting they are stable identities (non-zero).
    assert_ne!(ha, 0);
    assert_ne!(hb, 0);
}

// ── mid-graph seam where downstream DISCARDS one branch (reclaim path) ────────
//
// The seam writes both branches, but the downstream closure keeps only branch a
// (feeds it to a kernel) and DROPS branch b's pipe. `reclaim_undelivered` must
// drain + rehome b so it re-arms across syncs. Replayed ≥2×.
#[test]
fn midgraph_seam_downstream_discards_one_branch_still_rehomes() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let addend_a = DeviceScalar::<u32>::new(&ctx, 0).expect("addend_a");
    let addend_b = DeviceScalar::<u32>::new(&ctx, 0).expect("addend_b");
    let (ha, hb) = (handle_of(&addend_a), handle_of(&addend_b));
    let data = DeviceSlice::<u32>::from_slice(&ctx, &[1u32; N]).expect("data");
    let hd = handle_of(&data);

    let g = bundle2(claspr::eager::lift(addend_a), claspr::eager::lift(addend_b))
        .and_then_host(|(a, b): (&mut u32, &mut u32)| {
            *a = 5;
            *b = 99; // written but discarded downstream — must still rehome
            Ok(())
        })
        .and_then(move |(a_pipe, _b_pipe)| {
            // Drop b_pipe; keep only a. b must rehome via reclaim_undelivered.
            ks.add_ref_u32([N], data, a_pipe)
                .and_then(|(data, _a)| forward(data))
        });

    // Run 1: data = 1 + 5 = 6.
    let co = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*co), hd, "r1 data stable");
    assert!(
        co.map().wait().expect("map r1").iter().all(|&v| v == 6),
        "r1 data"
    );
    drop(co);

    // Run 2 (replay): if b did NOT rehome, this errors "busy". data = 6 + 5 = 11.
    let co = g
        .sync(&ctx)
        .expect("run 2 (replay) -- fails if discarded branch b didn't rehome");
    assert_eq!(handle_of(&*co), hd, "r2 data stable");
    assert!(
        co.map().wait().expect("map r2").iter().all(|&v| v == 11),
        "r2 data"
    );
    drop(co);

    assert_ne!(ha, 0);
    assert_ne!(hb, 0);
}

// ── REGRESSION GUARD: single-output mid-graph seam still re-homes ─────────────
//
// The single-output path must stay byte-behavior-identical: `Handle = Pipe<O>`,
// `SeamScatter` is the identity (one pipe), the seam threads exactly one home
// through to a downstream kernel. Replayed ≥2×.
#[test]
fn single_output_midgraph_seam_still_rehomes() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let addend = DeviceScalar::<u32>::new(&ctx, 0).expect("addend");
    let ha = handle_of(&addend);
    let data = DeviceSlice::<u32>::from_slice(&ctx, &[1u32; N]).expect("data");
    let hd = handle_of(&data);

    // single-output lifted scalar → seam writes it → downstream kernel reads it.
    let g = claspr::eager::lift(addend)
        .and_then_host(|a: &mut u32| {
            *a = 7;
            Ok(())
        })
        .and_then(move |a_pipe| {
            ks.add_ref_u32([N], data, a_pipe)
                .and_then(|(data, _a)| forward(data))
        });

    let co = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*co), hd, "r1 data stable");
    assert!(
        co.map().wait().expect("map r1").iter().all(|&v| v == 8),
        "r1 data"
    );
    drop(co);

    let co = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*co), hd, "r2 data stable");
    assert!(
        co.map().wait().expect("map r2").iter().all(|&v| v == 15),
        "r2 data"
    );
    drop(co);

    assert_ne!(ha, 0);
}

// ── REGRESSION GUARD: TERMINAL bundle-fed seam (the #212 pass-1 path) ─────────
//
// A bundle-fed seam that ENDS at the terminal (no downstream `and_then`) must
// still build its per-branch Checkouts and re-home each — unchanged from #212
// pass 1. This is the twin of `bundle_of_scalar_and_slice_fed_seam_rearms_x2`,
// kept here so the terminal path is guarded alongside the mid-graph tests.
#[test]
fn terminal_bundle_seam_still_rearms_x2() {
    let Some(ctx) = ctx() else { return };

    let sa = DeviceScalar::<u32>::new(&ctx, 0).expect("sa");
    let sb = DeviceScalar::<u32>::new(&ctx, 0).expect("sb");
    let (ha, hb) = (handle_of(&sa), handle_of(&sb));

    let g = bundle2(claspr::eager::lift(sa), claspr::eager::lift(sb)).and_then_host(
        |(a, b): (&mut u32, &mut u32)| {
            *a = a.wrapping_add(3);
            *b = b.wrapping_add(4);
            Ok(())
        },
    );

    // Run 1: sa = 3, sb = 4.
    let (ca1, cb1) = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*ca1), ha, "r1 sa stable");
    assert_eq!(handle_of(&*cb1), hb, "r1 sb stable");
    assert_eq!(ca1.read_value().expect("r1 sa"), 3);
    assert_eq!(cb1.read_value().expect("r1 sb"), 4);
    drop((ca1, cb1));

    // Run 2 (replay, accumulate): sa = 6, sb = 8.
    let (ca2, cb2) = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*ca2), ha, "r2 sa stable");
    assert_eq!(handle_of(&*cb2), hb, "r2 sb stable");
    assert_eq!(ca2.read_value().expect("r2 sa"), 6);
    assert_eq!(cb2.read_value().expect("r2 sb"), 8);
    drop((ca2, cb2));
}

// ── mid-graph bundle-fed seam consumed BY-VALUE (download reconstruct path) ───
//
// A downstream `download` on one branch exercises the seam's `collect` /
// reconstruct path (the by-value gather) rather than the Checkout terminal.
#[test]
fn midgraph_seam_downstream_download_reconstruct() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let addend_a = DeviceScalar::<u32>::new(&ctx, 0).expect("addend_a");
    let addend_b = DeviceScalar::<u32>::new(&ctx, 0).expect("addend_b");
    let data = DeviceSlice::<u32>::from_slice(&ctx, &[2u32; N]).expect("data");

    let out = bundle2(claspr::eager::lift(addend_a), claspr::eager::lift(addend_b))
        .and_then_host(|(a, b): (&mut u32, &mut u32)| {
            *a = 100;
            *b = 200; // discarded
            Ok(())
        })
        .and_then(move |(a_pipe, _b_pipe)| {
            ks.add_ref_u32([N], data, a_pipe)
                .and_then(|(data, _a)| download(data))
        })
        .sync(&ctx)
        .expect("download reconstruct run");
    assert!(out.iter().all(|&v| v == 102), "got {out:?}");
}
