//! `lift` is SELF-REHOMING: a `lift`ed owned buffer/scalar lends-and-returns
//! from its own [`Cell`], so a graph containing a `lift` REPLAYS across `sync`s
//! over the SAME `cl_mem` handle — exactly like a concrete input to an in-place
//! verb.
//!
//! Before the fix `lift` was move-in-once (`Mutex<Option<T>>`, drained on the
//! first run, no home threaded) — a second `sync` errored "already consumed". Now
//! it holds a `Cell<T>` and `resolve_home`s it (home = its own cell), so the value
//! rehomes on the run's `Checkout` drop and the lift node re-arms.
//!
//! This is the "present an owned buffer/scalar as a re-homing branch" primitive
//! (no device work) that CG strategy 2 needs: `bundle!(lift(a), lift(b), …)
//! .and_then_host(…)` — every lift branch re-arms and the multi-home seam (#212)
//! carries all their `&mut T` write-back views across the replay loop.

use claspr::eager::{DeviceOpExt, bundle2, bundle3};
use claspr::{DeviceScalar, DeviceSlice};
use claspr_test_kernels::kernels;
use claspr_test_support::{ctx, handle_of};

const N: usize = 64;

// ── lift as a LONE branch: a lifted buffer head → host seam, replayed ─────
//
// The minimal self-rehoming proof: a `lift`ed DeviceSlice fed straight into a
// host seam that doubles it in place. The lift node is the buffer's home, so the
// buffer re-homes on Checkout drop and the SAME graph replays over the SAME
// handle. Before the fix, `sync` #2 errored "a lifted resource was already
// consumed".
#[test]
fn lift_buffer_head_seam_replays_stable_handle() {
    let Some(ctx) = ctx() else { return };

    let buf = DeviceSlice::<u32>::from_slice(&ctx, &[3u32; N]).expect("buf");
    let h = handle_of(&buf);

    let g = claspr::eager::lift(buf).and_then_host(|view: &mut [u32]| {
        for x in view.iter_mut() {
            *x = x.wrapping_mul(2);
        }
        Ok(())
    });

    // Run 1: 3 × 2 = 6.
    let co = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*co), h, "lift lone: stable handle r1");
    {
        let view = co.map().wait().expect("map 1");
        assert!(view.iter().all(|&v| v == 6), "run1 {:?}", &view[..4]);
    }
    drop(co);

    // Run 2 (replay over the SAME handle, now 6): 6 × 2 = 12. This is the
    // self-rehome — an old move-once lift would error here.
    let co = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*co), h, "lift lone: stable handle r2");
    {
        let view = co.map().wait().expect("map 2");
        assert!(view.iter().all(|&v| v == 12), "run2 {:?}", &view[..4]);
    }
    drop(co);
}

// ── CG-strategy-2 pattern: bundle of lifted concrete scalars → ONE seam ───
//
// The exact shape CG strategy 2 depends on: several caller-owned DeviceScalars
// presented via `lift` as re-arming branches, bundled into ONE `and_then_host`
// seam that writes each via its `&mut T` view. Every lift branch re-homes to its
// own cell on drop, so the graph replays across the loop over stable handles —
// no per-scalar seams, no `into_inner`.
#[test]
fn bundle_of_lifted_scalars_fed_seam_rearms_x2() {
    let Some(ctx) = ctx() else { return };

    // Two device-resident scalars, a read-view partials-like array, presented as
    // lifted branches. The seam reads `partials` (branch a), writes `alpha` and
    // `nalpha` (branches b, c) from it — a miniature finish_alpha.
    let partials = DeviceScalar::<f32>::new(&ctx, 8.0).expect("partials"); // "Σ" stand-in
    let alpha = DeviceScalar::<f32>::new(&ctx, 0.0).expect("alpha");
    let nalpha = DeviceScalar::<f32>::new(&ctx, 0.0).expect("nalpha");
    let (hp, ha, hn) = (handle_of(&partials), handle_of(&alpha), handle_of(&nalpha));

    // rsold captured by the closure (a host constant here); alpha = rsold / Σ.
    let rsold = 20.0f32;
    let g = bundle3(
        claspr::eager::lift(partials),
        claspr::eager::lift(alpha),
        claspr::eager::lift(nalpha),
    )
    .and_then_host(move |(vp, va, vn): (&mut f32, &mut f32, &mut f32)| {
        let a = rsold / *vp; // 20 / 8 = 2.5
        *va = a;
        *vn = -a;
        Ok(())
    });

    // Run 1.
    let (cp1, ca1, cn1) = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*cp1), hp, "r1 partials stable");
    assert_eq!(handle_of(&*ca1), ha, "r1 alpha stable");
    assert_eq!(handle_of(&*cn1), hn, "r1 nalpha stable");
    assert!((ca1.read_value().expect("r1 alpha") - 2.5).abs() < 1e-6);
    assert!((cn1.read_value().expect("r1 nalpha") + 2.5).abs() < 1e-6);
    drop((cp1, ca1, cn1)); // every lift branch re-arms

    // Run 2 (replay over the SAME handles): partials unchanged (seam doesn't
    // write it), so alpha = 20 / 8 = 2.5 again — proves the re-home + closure
    // re-run.
    let (cp2, ca2, cn2) = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*cp2), hp, "r2 partials stable");
    assert_eq!(handle_of(&*ca2), ha, "r2 alpha stable");
    assert_eq!(handle_of(&*cn2), hn, "r2 nalpha stable");
    assert!((ca2.read_value().expect("r2 alpha") - 2.5).abs() < 1e-6);
    assert!((cn2.read_value().expect("r2 nalpha") + 2.5).abs() < 1e-6);
    drop((cp2, ca2, cn2));
}

// ── lift branch composes with a KERNEL branch in the same seam-fed bundle ──
//
// A mixed bundle: a lifted scalar (write-back branch) alongside a kernel head
// (identity scale over a caller-owned buffer), both fed into one seam. Proves a
// lift branch re-arms in lockstep with a kernel branch's own home threading.
#[test]
fn bundle_of_lift_and_kernel_fed_seam_rearms_x2() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let scalar = DeviceScalar::<u32>::new(&ctx, 0).expect("scalar");
    let slice = DeviceSlice::<u32>::from_slice(&ctx, &[7u32; N]).expect("slice");
    let (hs, hb) = (handle_of(&scalar), handle_of(&slice));

    let g = bundle2(claspr::eager::lift(scalar), ks.scale_u32([N], slice, 1u32)).and_then_host(
        |(vs, vslice): (&mut u32, &mut [u32])| {
            *vs = vs.wrapping_add(5);
            for x in vslice.iter_mut() {
                *x = x.wrapping_add(60);
            }
            Ok(())
        },
    );

    // Run 1: scalar 0 + 5 = 5, slice 7 + 60 = 67.
    let (cs1, cslice1) = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*cs1), hs, "r1 scalar stable");
    assert_eq!(handle_of(&*cslice1), hb, "r1 slice stable");
    assert_eq!(cs1.read_value().expect("r1 scalar"), 5);
    assert!(
        cslice1
            .map()
            .wait()
            .expect("map r1")
            .iter()
            .all(|&v| v == 67),
        "r1 slice"
    );
    drop((cs1, cslice1));

    // Run 2 (replay, same handles): scalar accumulates 5 + 5 = 10 (lift re-homes
    // the mutated value); slice 67 + 60 = 127.
    let (cs2, cslice2) = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*cs2), hs, "r2 scalar stable");
    assert_eq!(handle_of(&*cslice2), hb, "r2 slice stable");
    assert_eq!(cs2.read_value().expect("r2 scalar"), 10);
    assert!(
        cslice2
            .map()
            .wait()
            .expect("map r2")
            .iter()
            .all(|&v| v == 127),
        "r2 slice"
    );
    drop((cs2, cslice2));
}
