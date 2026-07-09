//! Bundle- / multi-output-fed host seam: EVERY branch re-arms (#212).
//!
//! The adversarial matrix from `eager_bundle_wide.rs`
//! (`bundle_of_two_multi_output_branches_rearms_x2_stable_handles`,
//! `nested_bundle_of_multi_output_branches_rearms_transitively`) CROSSED with the
//! replay proof from `eager_host_and_profile.rs`
//! (`and_then_host_replays_and_reruns_each_sync`).
//!
//! Before the fix, an `and_then_host` fed by a bundle / multi-output source
//! gathered its source via `collect_home`, which collapses a tuple output to ONE
//! home slot (`home == None` for a multi-output source by construction). So no
//! branch re-armed its origin cell and the graph broke on the 2nd `sync`. The fix
//! gives the seam `type Checkouts = S::Checkouts` and a `gather_checkouts` override
//! that SPLITs the source's per-branch checkouts (value + homes), maps the tuple
//! value through the closure, then REASSEMBLEs re-threading each ORIGINAL home — so
//! every branch re-arms and the graph replays.
//!
//! Each test: build ONCE, `sync` ≥2 times, assert (a) stable `cl_mem` handles
//! across syncs (the re-home), (b) the closure re-runs each sync (a `Fn` seam,
//! not one-shot), (c) values correct each time. Covered: bundle of single-output
//! branches; a bundle containing a multi-output (nested-bundle) branch; a mix of
//! `DeviceScalar` + `DeviceSlice` + `MappedSlice`; and a single-output-fed seam
//! regression guard for #211.

use claspr::eager::{DeviceOpExt, acquire_mapped_view, bundle2, bundle3, release_mapped_view};
use claspr::{Context, DeviceScalar, DeviceSlice, MappedSlice, MemRef, RecordableBuffer, SvmLevel};
use claspr_test_kernels::kernels;

const N: usize = 64;

/// Stable identity of a buffer's backing memory (raw `cl_mem`/SVM ptr as `usize`)
/// for `==` across replays. Works on a `DeviceSlice`/`DeviceScalar`/`MappedSlice`
/// and (via `Deref`) on a `Checkout` of any of them.
fn handle_of<B: RecordableBuffer>(b: &B) -> usize {
    match b.record_handle().mem {
        MemRef::Buffer(m) => m as usize,
        MemRef::Svm(p) => p as usize,
    }
}

/// Allocate + fill a `DeviceSlice<u32>` of `N` elements with `v`.
fn seeded(ctx: &Context, v: u32) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::alloc_zero(ctx, N)
        .expect("alloc")
        .fill(v)
        .wait()
        .expect("seed")
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

// ── #211 regression guard: single-output-fed seam still replays ──────────
//
// The single-output source path must stay byte-behavior-identical after #212:
// `S::Checkouts` collapses to `Checkout<O>`, `CheckoutSplit` is the identity
// (one value + one home), and the seam threads exactly one home. This is the
// same shape as `and_then_host_replays_and_reruns_each_sync`, kept here so a
// regression in the single-output path is caught alongside the multi-home tests.
#[test]
fn single_output_fed_seam_still_replays() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let buf = seeded(&ctx, 3);
    let h = handle_of(&buf);

    // identity kernel head over a caller-owned buffer, then a host seam doubling
    // in place. The buffer re-homes to its concrete cell across replays.
    let g = ks
        .scale_u32([N], buf, 1u32)
        .and_then_host(|view: &mut [u32]| {
            for x in view.iter_mut() {
                *x = x.wrapping_mul(2);
            }
            Ok(())
        });

    // Run 1: 3 × 1 × 2 = 6.
    let co = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*co), h, "single-output: stable handle r1");
    {
        let view = co.map().wait().expect("map 1");
        assert!(view.iter().all(|&v| v == 6), "run1 {:?}", &view[..4]);
    }
    drop(co);

    // Run 2 (replay over the SAME handle, now 6): 6 × 1 × 2 = 12.
    let co = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*co), h, "single-output: stable handle r2");
    {
        let view = co.map().wait().expect("map 2");
        assert!(view.iter().all(|&v| v == 12), "run2 {:?}", &view[..4]);
    }
    drop(co);
}

// ── bundle of single-output branches → seam, replayed ────────────────────
//
// The core #212 case: two single-output kernel branches (each an identity scale
// over a caller-owned buffer) bundled and fed into ONE host seam. Before the fix
// this broke on `sync` #2 ("already lent and not returned") because neither
// branch's home rode through the collapsed `collect_home`. Now the seam's
// `gather_checkouts` splits the `(Checkout<a>, Checkout<b>)` pair, maps the
// `(&mut [u32], &mut [u32])` tuple view, and reassembles re-threading BOTH homes.
#[test]
fn bundle_of_single_output_branches_fed_seam_rearms_x2() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let a = seeded(&ctx, 1);
    let b = seeded(&ctx, 10);
    let (ha, hb) = (handle_of(&a), handle_of(&b));

    // Two single-output branches → a 2-tuple `Output`; the seam maps
    // `(&mut [u32], &mut [u32])` and adds a distinct constant to each branch so a
    // wrong-branch mixup would be caught.
    let g = bundle2(ks.scale_u32([N], a, 1u32), ks.scale_u32([N], b, 1u32)).and_then_host(
        |(va, vb): (&mut [u32], &mut [u32])| {
            for x in va.iter_mut() {
                *x = x.wrapping_add(100);
            }
            for x in vb.iter_mut() {
                *x = x.wrapping_add(200);
            }
            Ok(())
        },
    );

    // Run 1: a = 1 + 100 = 101, b = 10 + 200 = 210.
    let (ca1, cb1) = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*ca1), ha, "r1 a stable");
    assert_eq!(handle_of(&*cb1), hb, "r1 b stable");
    assert!(
        ca1.map()
            .wait()
            .expect("map a r1")
            .iter()
            .all(|&v| v == 101),
        "r1 a"
    );
    assert!(
        cb1.map()
            .wait()
            .expect("map b r1")
            .iter()
            .all(|&v| v == 210),
        "r1 b"
    );
    drop((ca1, cb1)); // both re-arm

    // Run 2 (replay, same handles): a = 101 + 100 = 201, b = 210 + 200 = 410.
    let (ca2, cb2) = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*ca2), ha, "r2 a stable");
    assert_eq!(handle_of(&*cb2), hb, "r2 b stable");
    assert!(
        ca2.map()
            .wait()
            .expect("map a r2")
            .iter()
            .all(|&v| v == 201),
        "r2 a"
    );
    assert!(
        cb2.map()
            .wait()
            .expect("map b r2")
            .iter()
            .all(|&v| v == 410),
        "r2 b"
    );
    drop((ca2, cb2));
}

// ── bundle CONTAINING a multi-output (nested) branch → seam, replayed ─────
//
// The nested / transitive case: the outer bundle's first branch is ITSELF a
// multi-output kernel (`add_u32(a, b, out)` → `(a, b, out)`), the second is a
// single-output identity. So `S::Output = ((a, b, out), c)` and `S::Checkouts =
// ((Checkout, Checkout, Checkout), Checkout)`. `CheckoutSplit` must descend into
// the nested tuple, hand the closure `((&mut[], &mut[], &mut[]), &mut[])`, and
// reassemble re-threading ALL FOUR homes at BOTH nesting levels. This is the seam
// twin of `nested_bundle_of_multi_output_branches_rearms_transitively`.
// The nested-tuple seam view (`((&mut[], &mut[], &mut[]), &mut[])`) is an
// intrinsically nested type — the whole point of the transitive-nesting test — so
// the `type_complexity` lint on the closure annotation is expected here.
#[allow(clippy::type_complexity)]
#[test]
fn bundle_with_nested_multi_output_branch_fed_seam_rearms_transitively() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Branch 1 (multi-output): out = a + b = 3 + 4 = 7; a, b threaded back too.
    let a = seeded(&ctx, 3);
    let b = seeded(&ctx, 4);
    let out = seeded(&ctx, 0);
    // Branch 2 (single-output identity): c stays 9.
    let c = seeded(&ctx, 9);
    let (ha, hb, ho, hc) = (handle_of(&a), handle_of(&b), handle_of(&out), handle_of(&c));

    // The seam view is `((&mut[a], &mut[b], &mut[out]), &mut[c])`. Bump `out` and
    // `c` by distinct constants; leave a, b so their re-home + values are checked.
    let g = bundle2(ks.add_u32([N], a, b, out), ks.scale_u32([N], c, 1u32)).and_then_host(
        |((_va, _vb, vout), vc): ((&mut [u32], &mut [u32], &mut [u32]), &mut [u32])| {
            for x in vout.iter_mut() {
                *x = x.wrapping_add(1000);
            }
            for x in vc.iter_mut() {
                *x = x.wrapping_add(2000);
            }
            Ok(())
        },
    );

    // Run 1: out = (3+4) + 1000 = 1007, c = 9 + 2000 = 2009.
    let ((ca1, cb1, cout1), cc1) = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*ca1), ha, "r1 a stable");
    assert_eq!(handle_of(&*cb1), hb, "r1 b stable");
    assert_eq!(handle_of(&*cout1), ho, "r1 out stable");
    assert_eq!(handle_of(&*cc1), hc, "r1 c stable");
    assert!(
        cout1
            .map()
            .wait()
            .expect("map out r1")
            .iter()
            .all(|&v| v == 1007),
        "r1 out"
    );
    assert!(
        cc1.map()
            .wait()
            .expect("map c r1")
            .iter()
            .all(|&v| v == 2009),
        "r1 c"
    );
    drop((ca1, cb1, cout1, cc1)); // every buffer at every level re-arms

    // Run 2 (replay, same handles). add_u32 recomputes out = a + b over the
    // re-homed a=3, b=4 → 7, then +1000 = 1007 again. c = 2009 + 1 (identity) +
    // 2000 = 4009.
    let ((ca2, cb2, cout2), cc2) = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*ca2), ha, "r2 a stable");
    assert_eq!(handle_of(&*cb2), hb, "r2 b stable");
    assert_eq!(handle_of(&*cout2), ho, "r2 out stable");
    assert_eq!(handle_of(&*cc2), hc, "r2 c stable");
    assert!(
        cout2
            .map()
            .wait()
            .expect("map out r2")
            .iter()
            .all(|&v| v == 1007),
        "r2 out"
    );
    assert!(
        cc2.map()
            .wait()
            .expect("map c r2")
            .iter()
            .all(|&v| v == 4009),
        "r2 c"
    );
    drop((ca2, cb2, cout2, cc2));
}

// ── MIXED buffer kinds: DeviceScalar + DeviceSlice + MappedSlice → seam ───
//
// The buffer-kind-generality case the mandate calls out: a bundle whose branches
// span THREE buffer kinds fed into one seam.
//   - DeviceScalar<u32>  (View = &mut u32)      — write-scalar head, re-homes.
//   - DeviceSlice<u32>   (View = &mut [u32])    — identity scale, re-homes.
//   - MappedSlice<u32>   (via its Mappable host-view, View = &mut [u32]).
// MappedSlice/USMSlice are not directly `Mappable`; a MappedSlice reaches the seam
// through `acquire_mapped_view` (its host-view IS `Mappable`), so the branch is a
// `lift(mapped) → acquire_mapped_view` chain, released back after the seam. The
// seam maps the mixed `(&mut u32, &mut [u32], &mut [u32])` tuple, splitting three
// heterogeneous per-branch homes and reassembling each. The DeviceScalar +
// DeviceSlice branches re-arm their caller cells (asserted stable); the
// MappedSlice is a lifted resource (move-in-once head) so it does not re-arm a
// caller cell — its write-through is verified by re-mapping the released buffer.
// (The ×2 replay of the re-arming kinds is pinned by
// `bundle_of_scalar_and_slice_fed_seam_rearms_x2`; a lifted MappedSlice is out of
// scope for a ×2 replay by construction.)
//
// SVM-gated: skips on devices without coarse-grain SVM (MappedSlice needs it).
#[test]
fn bundle_of_mixed_buffer_kinds_fed_seam() {
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: no SVM support (MappedSlice branch needs it)");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Branch A: DeviceScalar written to 5 (single-output, re-homes to its cell).
    let scalar = DeviceScalar::<u32>::new(&ctx, 0).expect("alloc scalar");
    let hs = handle_of(&scalar);
    // Branch B: DeviceSlice identity scale (re-homes).
    let slice = seeded(&ctx, 7);
    let hb = handle_of(&slice);
    // Branch C: MappedSlice via its host-view (a lifted, move-in-once branch).
    let mapped0 = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("MappedSlice alloc");

    // The seam maps the mixed tuple and returns it as a single value (the seam's
    // Handle is one `Pipe<S::Output>`), so the terminal `Checkouts` is
    // `(Checkout<Scalar>, Checkout<DeviceSlice>, Checkout<MappedSliceHostView>)` —
    // three heterogeneous homes split + reassembled.
    let g = bundle3(
        ks.write_scalar_u32([1usize], scalar, 5u32),
        ks.scale_u32([N], slice, 1u32),
        claspr::eager::lift(mapped0).and_then(acquire_mapped_view),
    )
    .and_then_host(
        |(vs, vslice, vmapped): (&mut u32, &mut [u32], &mut [u32])| {
            *vs = vs.wrapping_add(50); // scalar: 5 + 50 = 55
            for x in vslice.iter_mut() {
                *x = x.wrapping_add(60); // slice: 7 + 60 = 67
            }
            for x in vmapped.iter_mut() {
                *x = 71; // mapped write-through
            }
            Ok(())
        },
    );

    let (cs, cslice, cmapped) = g.sync(&ctx).expect("mixed run");
    assert_eq!(handle_of(&*cs), hs, "scalar stable");
    assert_eq!(handle_of(&*cslice), hb, "slice stable");
    assert_eq!(cs.read_value().expect("read scalar"), 55, "scalar");
    assert!(
        cslice
            .map()
            .wait()
            .expect("map slice")
            .iter()
            .all(|&v| v == 67),
        "slice"
    );
    // Release the mapped host-view back to its `MappedSlice`, then re-map to
    // confirm the seam's write-through committed.
    let mut mapped = release_mapped_view(cmapped.into_inner())
        .sync(&ctx)
        .expect("release mapped view")
        .into_inner();
    {
        let guard = mapped.map_mut().wait().expect("re-map mapped");
        assert!(guard.iter().all(|&v| v == 71), "mapped write-through");
    }
    drop((cs, cslice));
}

/// Multi-sync replay of the re-arming kinds (DeviceScalar + DeviceSlice) fed
/// through one seam — the ×2 stable-handle proof for the scalar+slice kinds,
/// without the move-once MappedSlice branch (covered for write-through above).
#[test]
fn bundle_of_scalar_and_slice_fed_seam_rearms_x2() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let scalar = DeviceScalar::<u32>::new(&ctx, 0).expect("alloc scalar");
    let slice = seeded(&ctx, 7);
    let (hs, hb) = (handle_of(&scalar), handle_of(&slice));

    // Branch A writes the scalar to 5 each run (write_scalar re-seeds from the
    // by-value arg), branch B is an identity scale (accumulates). The seam bumps
    // the scalar by 50 and the slice by 60.
    let g = bundle2(
        ks.write_scalar_u32([1usize], scalar, 5u32),
        ks.scale_u32([N], slice, 1u32),
    )
    .and_then_host(|(vs, vslice): (&mut u32, &mut [u32])| {
        *vs = vs.wrapping_add(50);
        for x in vslice.iter_mut() {
            *x = x.wrapping_add(60);
        }
        Ok(())
    });

    // Run 1: scalar = 5 + 50 = 55, slice = 7 + 60 = 67.
    let (cs1, cslice1) = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*cs1), hs, "r1 scalar stable");
    assert_eq!(handle_of(&*cslice1), hb, "r1 slice stable");
    assert_eq!(cs1.read_value().expect("read r1"), 55, "r1 scalar");
    assert!(
        cslice1
            .map()
            .wait()
            .expect("map slice r1")
            .iter()
            .all(|&v| v == 67),
        "r1 slice"
    );
    drop((cs1, cslice1));

    // Run 2 (replay, same handles): scalar re-written to 5, +50 = 55 again; slice
    // accumulates: 67 (identity) + 60 = 127.
    let (cs2, cslice2) = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(handle_of(&*cs2), hs, "r2 scalar stable");
    assert_eq!(handle_of(&*cslice2), hb, "r2 slice stable");
    assert_eq!(cs2.read_value().expect("read r2"), 55, "r2 scalar");
    assert!(
        cslice2
            .map()
            .wait()
            .expect("map slice r2")
            .iter()
            .all(|&v| v == 127),
        "r2 slice"
    );
    drop((cs2, cslice2));
}
