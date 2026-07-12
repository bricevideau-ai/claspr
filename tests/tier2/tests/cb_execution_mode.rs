//! CB-as-EXECUTION-MODE (design v2): the whole seam-free device graph runs as ONE
//! automatic, invisible `cl_khr_command_buffer`, homed in the graph and replayed
//! across `sync`s. No user-facing record()/replay() — just `g.sync()` in a loop.
//!
//! These tests assert (a) an all-device graph takes a REAL command buffer on a
//! platform that advertises the extension (introspected via the root's cb_cache),
//! and (b) it produces the right results across replays (build then replay).

use claspr::Context;
use claspr::DeviceSlice;
use claspr::eager::{DeviceOp, DeviceOpExt, fill};
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

/// Whether the graph's root homed a real finalized command buffer after a sync.
fn homed_cb<O: DeviceOp>(g: &O) -> bool {
    g.cb_cache()
        .map(|c| c.lock().unwrap().is_some())
        .unwrap_or(false)
}

/// A single-`fill` graph is an all-device CB region: first `sync` builds + homes a
/// CB (on a CB-capable platform), the second replays it. The fill lands both runs.
#[test]
fn fill_runs_as_command_buffer_and_replays() {
    let Some(ctx) = ctx() else { return };
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let g = fill(buf, 7u32);

    // First sync: build + home the CB (or software fallback if no extension).
    let co = g.sync(&ctx).expect("sync 1");
    let g1 = co.map().wait().expect("read 1");
    assert!(g1.iter().all(|&v| v == 7), "fill 1: {:?}", &g1[..8]);
    drop(g1);
    let built = homed_cb(&g);
    drop(co);

    // Second sync: replay the homed CB (or re-run software).
    let co = g.sync(&ctx).expect("sync 2");
    let g2 = co.map().wait().expect("read 2");
    assert!(g2.iter().all(|&v| v == 7), "fill 2: {:?}", &g2[..8]);
    drop(g2);
    drop(co);

    // A device advertising the extension MUST have homed a real CB (not silently
    // fallen back to per-op execute); one lacking it MUST NOT have.
    if ctx.has_cl_khr_command_buffer() {
        assert!(
            built,
            "device advertises cl_khr_command_buffer but no CB was homed"
        );
    } else {
        assert!(
            !built,
            "device lacks cl_khr_command_buffer but a CB was homed"
        );
    }
    eprintln!(
        "cb-mode fill backend: {}",
        if built {
            "command buffer"
        } else {
            "software (no CB)"
        }
    );
}

/// A fill→kernel chain (in-place scale) is ONE all-device CB region homed at the
/// root AndThen. Build then replay across syncs; the buffer holds fill*scale each
/// run (idempotent — fill resets). The kernel arg-set + ND-range go into the CB.
#[test]
fn fill_then_kernel_chain_runs_as_command_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    // fill(buf, 2) -> scale_u32(_, 5): 2 -> 10 per run (fill resets each run).
    let g = fill(buf, 2u32).and_then(|b| ks.scale_u32([N], b, 5u32));

    for i in 0..3 {
        let co = g.sync(&ctx).unwrap_or_else(|e| panic!("sync {i}: {e:?}"));
        let gd = co.map().wait().expect("read");
        assert!(gd.iter().all(|&v| v == 10), "iter {i}: {:?}", &gd[..8]);
        drop(gd);
        drop(co);
    }
    // The chain's root is the AndThen; it homes the CB when the ext is present.
    if ctx.has_cl_khr_command_buffer() {
        assert!(homed_cb(&g), "chain should home a command buffer");
    }
}

/// A bundle of two in-place kernels over concrete buffers, both fed onward — the
/// multi-branch shape CG uses (bundle2 of axpys). The whole thing is ONE CB with
/// two ND-ranges joined only by their (independent) buffers; it homes at the root
/// AndThen and replays. Exercises the bundle CB fork + per-branch sync points.
#[test]
fn bundle_of_kernels_runs_as_command_buffer() {
    use claspr::eager::bundle2;
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let a = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("a");
    let b = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("b");

    // fill a=2, b=3, then bundle(scale a*5, scale b*7): a->10, b->21 each run.
    let g = fill(a, 2u32).and_then(move |a| {
        fill(b, 3u32)
            .and_then(move |b| bundle2(ks.scale_u32([N], a, 5u32), ks.scale_u32([N], b, 7u32)))
    });

    for i in 0..3 {
        let (ca, cb) = g.sync(&ctx).unwrap_or_else(|e| panic!("sync {i}: {e:?}"));
        let ga = ca.map().wait().expect("read a");
        let gb = cb.map().wait().expect("read b");
        assert!(ga.iter().all(|&v| v == 10), "a iter {i}: {:?}", &ga[..8]);
        assert!(gb.iter().all(|&v| v == 21), "b iter {i}: {:?}", &gb[..8]);
        drop(ga);
        drop(gb);
        drop((ca, cb));
    }
    if ctx.has_cl_khr_command_buffer() {
        assert!(
            homed_cb(&g),
            "bundle-of-kernels should home a command buffer"
        );
    }
}
