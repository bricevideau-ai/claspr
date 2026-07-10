//! #214: `FanOut` re-arms PER-BRANCH across replays — the dynamic-`Vec` analog of
//! the fixed-arity `bundle!` re-arm (#207/#212).
//!
//! `FanOut`'s terminal `Checkouts` is now `Vec<U::Checkouts>` (one `Checkout` — or
//! nested tuple, for a multi-output branch — per branch, each threading its own
//! return home), and `gather_checkouts` delegates to each branch's own gather. So
//! a fan-out over in-place caller-owned buffers returns every buffer to its cell on
//! drop and the SAME graph replays over stable `cl_mem` handles. (The core
//! caller-buffer re-arm + stable-handle proof lives in
//! `home_invariant::fan_out_caller_buffers_rearm_x2_stable_handles`; this file adds
//! the regression + generality coverage: minted-buffer fan-out still works,
//! multi-output-branch fan-out re-arms, and fan-out composes as a chain head and
//! mid-graph.)

use claspr::eager::{DeviceOpExt, download, fan_out, forward, upload};
use claspr::{Context, DeviceSlice, MemRef, RecordableBuffer};
use claspr_test_kernels::kernels;

const N: usize = 64;

/// Stable identity of a buffer's backing `cl_mem`/SVM ptr for `==` across replays.
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

/// REGRESSION: fan-out over MINTED buffers (`upload` per branch) still works and
/// replays. Minted branches carry `home == None` (nothing to return), so the
/// per-branch re-arm is a no-op there — the graph must still re-run and reseed.
#[test]
fn fan_out_minted_buffers_still_replays() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let ks = &kernels;

    // Each branch mints its own buffer (upload), scales it, downloads. Homogeneous
    // → fan_out. `upload` re-seeds each run, so results are idempotent.
    let g = fan_out(vec![1u32, 2u32, 3u32], move |v| {
        upload(vec![10u32; N])
            .and_then(move |b| ks.scale_u32([N], b, v))
            .and_then(download)
    });

    let r1 = g.sync(&ctx).expect("run 1");
    assert_eq!(r1.len(), 3);
    assert!(r1[0].iter().all(|&x| x == 10), "branch 0: 10×1");
    assert!(r1[1].iter().all(|&x| x == 20), "branch 1: 10×2");
    assert!(r1[2].iter().all(|&x| x == 30), "branch 2: 10×3");
    drop(r1);

    // Run 2 (replay): minted buffers reseed to 10, same results — a fan-out that
    // dropped its branch homes would still work here (minted has no home), but the
    // point is the per-branch gather doesn't REGRESS the minted path.
    let r2 = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(r2.len(), 3);
    assert!(r2[0].iter().all(|&x| x == 10));
    assert!(r2[1].iter().all(|&x| x == 20));
    assert!(r2[2].iter().all(|&x| x == 30));
}

/// GENERALITY: fan-out whose branches are themselves MULTI-OUTPUT ops re-arms every
/// buffer at every branch. Each branch is `add_u32(a, b, out)` over caller-owned
/// buffers (a 3-output op → the branch's `Checkouts` is a `(Checkout, Checkout,
/// Checkout)` tuple), so the fan-out's `Checkouts` is `Vec<(Checkout,Checkout,Checkout)>`.
/// All three buffers per branch must re-home for the replay.
#[test]
fn fan_out_multi_output_branches_rearm_x2() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let ks = &kernels;

    // Two branches, each with its own (a, b, out) caller-owned trio.
    let a0 = DeviceSlice::<u32>::from_slice(&ctx, &[1u32; N]).expect("a0");
    let b0 = DeviceSlice::<u32>::from_slice(&ctx, &[2u32; N]).expect("b0");
    let o0 = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("o0");
    let a1 = DeviceSlice::<u32>::from_slice(&ctx, &[3u32; N]).expect("a1");
    let b1 = DeviceSlice::<u32>::from_slice(&ctx, &[4u32; N]).expect("b1");
    let o1 = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("o1");
    let (ho0, ho1) = (handle_of(&o0), handle_of(&o1));

    // fan_out over the two trios; each branch is add_u32 (multi-output).
    let g = fan_out(vec![(a0, b0, o0), (a1, b1, o1)], move |(a, b, out)| {
        ks.add_u32([N], a, b, out)
    });

    // Run 1: branch 0 out = 1+2 = 3, branch 1 out = 3+4 = 7. Each branch yields a
    // (Checkout<a>, Checkout<b>, Checkout<out>) tuple.
    let cos = g.sync(&ctx).expect("run 1");
    assert_eq!(cos.len(), 2);
    let (_a0, _b0, out0) = &cos[0];
    let (_a1, _b1, out1) = &cos[1];
    assert_eq!(handle_of(&**out0), ho0, "run 1: branch 0 out handle");
    assert_eq!(handle_of(&**out1), ho1, "run 1: branch 1 out handle");
    assert!(
        out0.map()
            .wait()
            .expect("map o0 r1")
            .iter()
            .all(|&v| v == 3),
        "run 1: branch 0 = 3"
    );
    assert!(
        out1.map()
            .wait()
            .expect("map o1 r1")
            .iter()
            .all(|&v| v == 7),
        "run 1: branch 1 = 7"
    );
    drop(cos); // every buffer at every branch re-homes

    // Run 2 (replay over the SAME handles): a/b re-armed to their seeds, so
    // out recomputes 3 and 7 again.
    let cos2 = g.sync(&ctx).expect("run 2 (replay)");
    let (_a0, _b0, out0) = &cos2[0];
    let (_a1, _b1, out1) = &cos2[1];
    assert_eq!(handle_of(&**out0), ho0, "run 2: branch 0 out handle stable");
    assert_eq!(handle_of(&**out1), ho1, "run 2: branch 1 out handle stable");
    assert!(
        out0.map()
            .wait()
            .expect("map o0 r2")
            .iter()
            .all(|&v| v == 3),
        "run 2: branch 0 = 3"
    );
    assert!(
        out1.map()
            .wait()
            .expect("map o1 r2")
            .iter()
            .all(|&v| v == 7),
        "run 2: branch 1 = 7"
    );
    drop(cos2);
}

/// COMPOSITION: a fan-out over MINTED buffers composed MID-graph (source of an
/// `.and_then` that forwards the collapsed `Vec` onward) works and replays. The
/// collapsed-`Vec`-handle by-value path carries `home == None` (the same boundary
/// `bundle!`/`arc_split` document — a `Vec` value has one home slot, `N` branch
/// homes can't ride it), so this exercises the by-value composition, not the
/// per-branch re-home (that rides the Checkout terminal — see the caller-buffer
/// test in home_invariant.rs). Minted branches reseed each run, so it replays.
#[test]
fn fan_out_midgraph_by_value_composes_and_replays() {
    let Some(ctx) = ctx() else { return };

    // Minted branches (upload → download to a host Vec) so there's no caller home
    // to thread; forward the collapsed Vec of results to the terminal. `forward`
    // flattens the fan-out's per-branch Checkouts to a single
    // `Checkout<Vec<Vec<u32>>>` (the by-value shape).
    let g = fan_out(vec![7u32, 8u32], |v| upload(vec![v; N]).and_then(download)).and_then(forward);

    // Run 1: the two minted buffers hold their seeds.
    let co = g.sync(&ctx).expect("run 1");
    assert_eq!(co.len(), 2);
    assert!(co[0].iter().all(|&v| v == 7), "run 1 branch 0 = 7");
    assert!(co[1].iter().all(|&v| v == 8), "run 1 branch 1 = 8");
    drop(co);

    // Run 2 (replay): minted branches reseed, same values. Proves the mid-graph
    // by-value composition re-runs (the collapsed Vec path is sound).
    let co = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(co.len(), 2);
    assert!(co[0].iter().all(|&v| v == 7), "run 2 branch 0 = 7");
    assert!(co[1].iter().all(|&v| v == 8), "run 2 branch 1 = 8");
    drop(co);
}
