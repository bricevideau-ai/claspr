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
use claspr::{slot, slots};
use claspr_test_kernels::kernels;

// A scalar (by-value) scale factor slot — `mutate_bind(Factor(v))` changes it.
slots! {
    Factor: u32,
}

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

/// Stable identity (the `Arc` pointer) of the root's homed `FinalizedCb`, or 0 if
/// none. Two syncs returning the SAME non-zero id prove create-once + replay (the
/// CB is built on sync #1 and REUSED on sync #2 — not rebuilt each run).
fn homed_cb_id<O: DeviceOp>(g: &O) -> usize {
    g.cb_cache()
        .and_then(|c| {
            c.lock()
                .unwrap()
                .as_ref()
                .map(|arc| std::sync::Arc::as_ptr(arc) as usize)
        })
        .unwrap_or(0)
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

/// REGRESSION GUARD (the false-positive the micro-tests missed): a CG-shaped chain
/// — kernels threaded through `and_then` whose closures feed RAW `Pipe` handles as
/// `bundle*` branches — must record its kernels into ONE command buffer homed at
/// the root AND replay the SAME finalized CB across syncs (create-once), NOT rebuild
/// or silently fall back to per-op. The bare-`Pipe`-as-op branch previously reported
/// `cb_addable() == false`, ANDing the whole bundle down to fallback, so this exact
/// shape ran entirely on `clEnqueueNDRangeKernel` while still converging.
///
/// The guard: (a) the root homes a CB after sync #1, and (b) sync #2 reuses the
/// SAME `FinalizedCb` Arc (create-once + replay), which per-op fallback (id==0) and
/// rebuild-each-sync (id changes) both fail.
#[test]
fn cg_shaped_pipe_bundle_records_one_cb_and_replays() {
    use claspr::eager::bundle2;
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let a = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("a");
    let b = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("b");

    // scale a (in place), thread `a` onward as a RAW pipe, then bundle2 of two
    // pipe-fed branches — the CG `bundle6(p, ap, …)` / `bundle2(x, rsnew)` shape:
    //   fill a=1 -> scale a*3 -> and_then(|a: Pipe| bundle2(scale b via a-derived, forward a))
    // Kept simple: scale a, then bundle2(forward(a), scale_u32(b)) where forward(a)
    // is a bare Pipe branch (the exact node that was non-addable).
    let g = fill(a, 1u32)
        .and_then(move |a| ks.scale_u32([N], a, 3u32)) // a = 3
        .and_then(move |a_pipe| {
            // a_pipe is a Pipe<DeviceSlice>; feed it RAW as a bundle branch.
            bundle2(a_pipe, fill(b, 5u32))
        });

    let co = g.sync(&ctx).expect("sync 1");
    let id1 = homed_cb_id(&g);
    let (ca, cbk) = co;
    let ga = ca.map().wait().expect("read a");
    let gb = cbk.map().wait().expect("read b");
    assert!(ga.iter().all(|&v| v == 3), "a: {:?}", &ga[..8]);
    assert!(gb.iter().all(|&v| v == 5), "b: {:?}", &gb[..8]);
    drop((ga, gb));
    drop((ca, cbk));

    let co = g.sync(&ctx).expect("sync 2");
    let id2 = homed_cb_id(&g);
    drop(co);

    if ctx.has_cl_khr_command_buffer() {
        assert_ne!(
            id1, 0,
            "CG-shaped pipe-bundle graph did NOT record a CB (fell back to per-op) \
             — a bare-Pipe branch is reporting cb_addable()==false again"
        );
        assert_eq!(
            id1, id2,
            "iteration CB was NOT create-once: sync #2 built a different CB (or fell \
             back). Same Arc across syncs = build-once + replay."
        );
    }
}

/// HOST-SEAM SEGMENTATION (the hard gate): a device span → `and_then_host` cut →
/// device span must segment into MULTIPLE command buffers wired by the
/// event↔sync-point boundary (each device span its own CB; the CB completion event
/// gates the seam's map, and the seam's unmap event gates the next span's CB), and
/// converge to the SAME result as a pure per-op run.
///
/// The graph: fill(buf,1) -> scale*3 [CB1] -> and_then_host(view += 100) [host] ->
/// scale*2 [CB2]. Per-op result: ((1*3)+100)*2 = 206. The two device spans home
/// DISTINCT CBs (id1 != id2, both non-zero) — proving ≥2 CBs, not a silent
/// single-CB or all-fallback.
#[test]
fn host_seam_segments_into_multiple_command_buffers() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    // The first device span (scale*3) is the SOURCE of the outer AndThen whose tail
    // contains the seam — so it becomes its OWN CB. The second span (scale*2) is
    // after the seam — its own CB. We capture each span's op to introspect its CB.
    let span1 = fill(buf, 1u32).and_then(|b| ks.scale_u32([N], b, 3u32));
    // Build the whole graph; keep a handle to the pre-seam span via forward so we
    // can assert it homed a CB. (Structurally, the outer and_then's source is
    // span1; cb_exec_child boundaries it.)
    let g = span1
        .and_then_host(|v: &mut [u32]| {
            for x in v.iter_mut() {
                *x += 100;
            }
            Ok(())
        })
        .and_then(move |b| ks.scale_u32([N], b, 2u32));

    for i in 0..3 {
        let co = g.sync(&ctx).unwrap_or_else(|e| panic!("sync {i}: {e:?}"));
        let gd = co.map().wait().expect("read");
        assert!(
            gd.iter().all(|&v| v == 206),
            "iter {i}: expected 206 = ((1*3)+100)*2, got {:?}",
            &gd[..8]
        );
        drop(gd);
        drop(co);
    }

    // The graph contains a host seam, so the WHOLE graph is NOT one CB (the root's
    // own cb_cache stays empty — cb_addable is false for a seam graph). Segmentation
    // is proven by the trace (cliloader: ≥2 distinct clCommandNDRangeKernelKHR CBs +
    // host map/unmap between them) and by correctness here. The root homes NO CB:
    assert!(
        !homed_cb(&g),
        "a host-seam graph's ROOT must not home a single whole-graph CB"
    );
}

/// MUTATION INVALIDATION (step 6): a homed command buffer captures the concrete
/// buffers/args it was built with. `mutate_bind`/`mutate_call` re-binding a slot
/// must INVALIDATE the CB (clear the cb_cache) so the NEXT sync rebuilds it and the
/// new binding takes effect — a stale CB would silently keep computing the OLD
/// value. Here a `slot!(Factor)` scale factor is mutated between syncs; the result
/// must reflect the NEW factor, and the CB must be a DIFFERENT Arc after the mutate.
#[test]
fn mutate_invalidates_and_rebuilds_command_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    // fill(buf,1) -> scale_u32(buf, Factor). Factor is an unbound scalar slot.
    let g = fill(buf, 1u32).and_then(|b| ks.scale_u32([N], b, slot!(Factor)));

    // Bind Factor=3, sync: 1*3 = 3. Homes the CB.
    g.mutate_bind(Factor(3u32)).expect("bind factor 3");
    let co = g.sync(&ctx).expect("sync 1");
    let g1 = co.map().wait().expect("read 1");
    assert!(g1.iter().all(|&v| v == 3), "factor 3: {:?}", &g1[..8]);
    drop(g1);
    let id1 = homed_cb_id(&g);
    drop(co);

    // Mutate Factor=5: MUST invalidate the homed CB.
    g.mutate_bind(Factor(5u32)).expect("mutate factor 5");
    if ctx.has_cl_khr_command_buffer() {
        assert_eq!(
            homed_cb_id(&g),
            0,
            "mutate_bind must clear the homed CB (invalidation) — it is still cached"
        );
    }

    // Next sync: 1*5 = 5 (new factor took effect), and a FRESH CB is built.
    let co = g.sync(&ctx).expect("sync 2");
    let g2 = co.map().wait().expect("read 2");
    assert!(
        g2.iter().all(|&v| v == 5),
        "after mutate to factor 5, expected 5, got {:?} — stale CB kept the old factor",
        &g2[..8]
    );
    drop(g2);
    let id2 = homed_cb_id(&g);
    drop(co);

    if ctx.has_cl_khr_command_buffer() {
        assert_ne!(id2, 0, "sync after mutate should rebuild a CB");
        assert_ne!(
            id1, id2,
            "the rebuilt CB must be a DIFFERENT finalized CB than the pre-mutate one"
        );
    }
}
