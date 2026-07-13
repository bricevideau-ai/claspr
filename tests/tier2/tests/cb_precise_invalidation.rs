//! PRECISE per-slot command-buffer invalidation (design v2, precise pass).
//!
//! `mutate_bind`/`mutate_call` re-binding a slot in a built, replayed graph makes
//! any homed command buffer that BAKED that slot's buffer/scalar stale. The precise
//! pass clears ONLY the CBs whose recorded commands depend on a mutated slot —
//! versus the old coarse "clear every CB in the graph". Correctness is identical
//! (a stale CB never survives); the win is that untouched regions keep their homed
//! CB across the mutate and skip the rebuild.
//!
//! These tests introspect the graph's homed CBs by their stable `Arc` identity (the
//! `#[doc(hidden)]` `DeviceOp::collect_cb_ids` walk) and assert the precise SET:
//!
//! - **precision** — mutating region A's slot clears A's CB but LEAVES B's CB Arc
//!   untouched (a coarse clear-all would drop both; a no-op invalidation would keep
//!   both AND compute the wrong answer);
//! - **reach** — a buffer slot threaded through region 1's kernel, across a host
//!   seam, into region 2's CB IS reached: mutating it clears region 2's CB too (the
//!   pipe-reachability substrate carried the origin across the seam — without it,
//!   region 2's CB would silently replay the OLD buffer);
//! - **combinator recursion** — a CB homed UNDER a `bundle` branch (below an interior
//!   seam) is still reached by the mutate walk (the recursion the bundle/seam
//!   `invalidate_cbs`/`collect_cb_ids` overrides close).
//!
//! Each device region is a NESTED chain of ≥2 device commands so it homes its OWN
//! command buffer (a single post-seam op is weight-1 and runs per-op, homing no CB).
//! Where the platform lacks `cl_khr_command_buffer` no CB is ever homed; the id sets
//! are empty and the SET assertions are guarded by `has_cl_khr_command_buffer()`.
//! Results are asserted UNCONDITIONALLY on every path.

use claspr::eager::{Checkout, DeviceOp, DeviceOpExt, bundle2, fill};
use claspr::image::format::R32G32B32A32Uint;
use claspr::{Context, DeviceSlice, Image2D, ReadWrite, eager_image_copy};
use claspr::{slot, slots};
use claspr_test_kernels::kernels;

const N: usize = 64;
// Image dims for the image-slot reach test (W*H == N so the pixel count matches).
const W: u32 = 8;
const H: u32 = 8;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            None
        }
    }
}

// Two independent scalar factor slots (one per device region / branch) + a buffer
// slot for the cross-seam reach test.
slots! {
    Factor1: u32,
    Factor2: u32,
    Buf: DeviceSlice<u32>,
    SrcImg: Image2D<ReadWrite, R32G32B32A32Uint>,
}

/// Every homed `FinalizedCb` id in the graph, as a sorted set (the `Arc` pointers).
fn cb_ids<O: DeviceOp>(g: &O) -> std::collections::BTreeSet<usize> {
    let mut v = Vec::new();
    g.collect_cb_ids(&mut v);
    v.into_iter().collect()
}

/// A no-op host seam over a `&mut [u32]` view — segments the graph into two CB
/// regions WITHOUT changing the data (so the arithmetic stays easy to predict).
fn noop_seam(_v: &mut [u32]) -> claspr::Result<()> {
    Ok(())
}

/// Seed a fresh `DeviceSlice<u32>` of `N` elements to `v`.
fn seeded(ctx: &Context, v: u32) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::alloc_zero(ctx, N)
        .expect("alloc")
        .fill(v)
        .wait()
        .expect("seed")
}

type RgbaImg = Image2D<ReadWrite, R32G32B32A32Uint>;

/// Seed a fresh `W×H` RGBA image with the constant pixel `[base, base+1, base+2, base+3]`.
fn seeded_image(ctx: &Context, base: u32) -> RgbaImg {
    RgbaImg::alloc(ctx, W, H)
        .expect("alloc image")
        .fill([base, base + 1, base + 2, base + 3])
        .wait()
        .expect("seed image")
}

/// PRECISION: two device regions split by a host seam, each a nested weight-2 chain
/// capturing a DIFFERENT scalar slot. Mutating region 1's `Factor1` must clear region
/// 1's CB while region 2's CB Arc stays put — the whole point of "precise". A coarse
/// clear-all would drop region 2's CB too; a broken invalidation would keep region
/// 1's stale CB and compute the old factor.
#[test]
fn distinct_slots_mutate_clears_only_its_region() {
    let Some(ctx) = ctx() else { return };
    let has_cb = ctx.has_cl_khr_command_buffer();
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ks = &ks;
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    // region 1: fill(buf,2) -> [ scale(Factor1) -> scale(*1) ]   nested weight-2 -> CB {Factor1}
    //   -- host seam (no-op) --
    // region 2: [ scale(Factor2) -> scale(*1) ]                  nested weight-2 -> CB {Factor2}
    let g = fill(buf, 2u32)
        .and_then(|b| {
            ks.scale_u32([N], b, slot!(Factor1))
                .and_then(|b2| ks.scale_u32([N], b2, 1u32))
        })
        .and_then_host(noop_seam)
        .and_then(|b| {
            ks.scale_u32([N], b, slot!(Factor2))
                .and_then(|b2| ks.scale_u32([N], b2, 1u32))
        });

    // Bind both factors, sync: 2 * 3 * 1 * 5 * 1 = 30. Homes both regions' CBs.
    g.mutate_bind(Factor1(3u32)).expect("bind f1");
    g.mutate_bind(Factor2(5u32)).expect("bind f2");
    let co = g.sync(&ctx).expect("sync 1");
    let g1 = co.map().wait().expect("read 1");
    assert!(g1.iter().all(|&v| v == 30), "f1=3 f2=5: {:?}", &g1[..8]);
    drop(g1);
    drop(co);

    let s0 = cb_ids(&g);
    if has_cb {
        assert_eq!(
            s0.len(),
            2,
            "expected two homed region CBs, got {}",
            s0.len()
        );
    }

    // Mutate ONLY Factor1. Region 1's CB must be cleared; region 2's must survive.
    g.mutate_bind(Factor1(4u32)).expect("mutate f1");
    let s1 = cb_ids(&g);
    if has_cb {
        assert!(
            s1.is_subset(&s0),
            "no CB should be spuriously created by a mutate: s1={s1:?} s0={s0:?}"
        );
        assert_eq!(
            s1.len(),
            1,
            "mutate of Factor1 must clear EXACTLY region 1's CB, leaving region 2's: \
             s0={s0:?} s1={s1:?}"
        );
        assert!(
            s1.iter().all(|id| s0.contains(id)),
            "the surviving CB id must be the original region-2 CB (no rebuild): \
             s0={s0:?} s1={s1:?}"
        );
    }

    // Next sync: 2 * 4 * 1 * 5 * 1 = 40 (new Factor1 took effect; region 2 replayed
    // its RETAINED CB on the updated buffer).
    let co = g.sync(&ctx).expect("sync 2");
    let g2 = co.map().wait().expect("read 2");
    assert!(
        g2.iter().all(|&v| v == 40),
        "after mutate Factor1=4 expected 40, got {:?}",
        &g2[..8]
    );
    drop(g2);
    drop(co);
}

/// REACH: a buffer slot threaded in place through region 1's kernel, across a host
/// seam, into region 2's CB. Mutating that buffer slot must clear BOTH regions'
/// CBs — region 2 captured the slot ONLY via the pipe-reachability substrate that
/// carried the origin across the seam. Without the cross-seam reach forward, region
/// 2's CB would survive and replay the OLD `cl_mem` → wrong answer. The result check
/// is the load-bearing proof; the id-set check corroborates.
#[test]
fn buffer_slot_threaded_across_seam_clears_downstream_cb() {
    let Some(ctx) = ctx() else { return };
    let has_cb = ctx.has_cl_khr_command_buffer();
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ks = &ks;

    // region 1: scale(Buf,2) -> scale(*1)   in place on the Buf SLOT [CB1 {Buf}]
    //   -- host seam (no-op) --  Buf's cl_mem threads onward unchanged
    // region 2: [ scale(*3) -> scale(*1) ]   on the threaded Buf     [CB2 {Buf} via reach]
    let g = ks
        .scale_u32([N], slot!(Buf), 2u32)
        .and_then(|b| ks.scale_u32([N], b, 1u32))
        .and_then_host(noop_seam)
        .and_then(|b| {
            ks.scale_u32([N], b, 3u32)
                .and_then(|b2| ks.scale_u32([N], b2, 1u32))
        });

    // Bind Buf := [10; N], sync: 10 * 2 * 3 = 60.
    g.mutate_bind(Buf(seeded(&ctx, 10))).expect("bind Buf=10");
    let co = g.sync(&ctx).expect("sync 1");
    let r1 = co.map().wait().expect("read 1");
    assert!(r1.iter().all(|&v| v == 60), "Buf=10: {:?}", &r1[..8]);
    drop(r1);
    drop(co);

    let s0 = cb_ids(&g);
    if has_cb {
        assert_eq!(s0.len(), 2, "expected two region CBs, got {}", s0.len());
    }

    // Mutate Buf to a DIFFERENT buffer ([7; N]). BOTH regions' CBs baked the old
    // cl_mem and must be cleared — region 2 only because the reach substrate carried
    // Buf's origin across the seam into its captured_slots.
    g.mutate_bind(Buf(seeded(&ctx, 7))).expect("mutate Buf=7");
    let s1 = cb_ids(&g);
    if has_cb {
        assert!(
            s1.is_disjoint(&s0),
            "mutating the threaded Buf must clear BOTH regions' CBs (region 2's \
             survived → cross-seam reach missing): s0={s0:?} s1={s1:?}"
        );
    }

    // Next sync: 7 * 2 * 3 = 42. A surviving region-2 CB would have replayed the old
    // buffer (10*2*3=60) — so this value IS the reach proof.
    let co = g.sync(&ctx).expect("sync 2");
    let r2 = co.map().wait().expect("read 2");
    assert!(
        r2.iter().all(|&v| v == 42),
        "after mutate Buf=7 expected 42, got {:?} — stale downstream CB kept old buffer",
        &r2[..8]
    );
    drop(r2);
    drop(co);
}

/// COMBINATOR RECURSION: a `bundle2` of two branches, EACH containing an interior
/// host seam so each branch homes its pre-seam CB on an INTERIOR node (under the
/// bundle). Mutating branch A's slot must reach that nested CB — the mutate walk
/// recurses bundle -> branch -> (pre-seam) source — and must leave branch B's nested
/// CB untouched (precision within the bundle). The default own-cache-only
/// invalidation would strand both nested CBs; a coarse clear-all would drop both.
#[test]
fn bundle_branch_interior_cb_precise_recursion() {
    let Some(ctx) = ctx() else { return };
    let has_cb = ctx.has_cl_khr_command_buffer();
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ks = &ks;

    let a = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("a");
    let b = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("b");

    // branchA: fill(a,2) -> scale(Factor1) [CB, interior] -- seam -- scale(*1)
    //   -> A = 2 * Factor1
    // branchB: fill(b,10) -> scale(Factor2) [CB, interior] -- seam -- scale(*1)
    //   -> B = 10 * Factor2
    let g = bundle2(
        fill(a, 2u32)
            .and_then(|x| ks.scale_u32([N], x, slot!(Factor1)))
            .and_then_host(noop_seam)
            .and_then(|x| ks.scale_u32([N], x, 1u32)),
        fill(b, 10u32)
            .and_then(|x| ks.scale_u32([N], x, slot!(Factor2)))
            .and_then_host(noop_seam)
            .and_then(|x| ks.scale_u32([N], x, 1u32)),
    );

    g.mutate_bind(Factor1(3u32)).expect("bind f1");
    g.mutate_bind(Factor2(4u32)).expect("bind f2");
    let (ca, cb) = g.sync(&ctx).expect("sync 1");
    let ra = ca.map().wait().expect("read a");
    let rb = cb.map().wait().expect("read b");
    assert!(ra.iter().all(|&v| v == 6), "A f1=3: {:?}", &ra[..8]);
    assert!(rb.iter().all(|&v| v == 40), "B f2=4: {:?}", &rb[..8]);
    drop((ra, rb));
    drop((ca, cb));

    let s0 = cb_ids(&g);
    if has_cb {
        assert_eq!(
            s0.len(),
            2,
            "each branch homes one interior pre-seam CB: got {}",
            s0.len()
        );
    }

    // Mutate Factor1 (branch A's slot): branch A's interior CB must be cleared via
    // bundle -> branch recursion; branch B's interior CB must survive.
    g.mutate_bind(Factor1(5u32)).expect("mutate f1");
    let s1 = cb_ids(&g);
    if has_cb {
        assert_eq!(
            s1.len(),
            1,
            "exactly branch A's interior CB should clear (recursion + precision): \
             s0={s0:?} s1={s1:?}"
        );
        assert!(
            s1.iter().all(|id| s0.contains(id)),
            "the survivor must be branch B's original interior CB: s0={s0:?} s1={s1:?}"
        );
    }

    let (ca, cb) = g.sync(&ctx).expect("sync 2");
    let ra = ca.map().wait().expect("read a");
    let rb = cb.map().wait().expect("read b");
    assert!(
        ra.iter().all(|&v| v == 10),
        "after mutate f1=5 expected A=10, got {:?} — nested CB kept old factor",
        &ra[..8]
    );
    assert!(
        rb.iter().all(|&v| v == 40),
        "branch B must be unaffected by a Factor1 mutate: {:?}",
        &rb[..8]
    );
    drop((ra, rb));
    drop((ca, cb));
}

/// A buffer SLOT in a `fill()` POSITION (not a kernel arg) captured by a CB. Before
/// the leaf-`bind_slots` fix, `Fill` never offered its buffer input to the binder, so
/// `mutate_bind(Buf(..))` failed `SlotNoSuchTag` — the `fill`/`write`/`download` slot
/// positions the docs advertise type-checked but were unbindable. With `bind_slots`
/// wired (and `Fill` already using `cb_leaf_build`), a filled slot is captured AND
/// precisely invalidated: mutating `Buf` clears the `fill -> scale` CB. `fill`
/// overwrites, so the arithmetic is Buf-independent (21 either way) — the id-set is
/// the load-bearing proof that the mutate reached the CB (a SlotNoSuchTag would panic
/// the mutate outright, a missed reach would leave the id set unchanged).
#[test]
fn fill_slot_position_mutate_clears_its_cb() {
    let Some(ctx) = ctx() else { return };
    let has_cb = ctx.has_cl_khr_command_buffer();
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ks = &ks;

    // weight-2 CB capturing the Buf slot in a FILL position: fill(Buf, 7) -> scale(*3).
    let g = fill(slot!(Buf).into_slot_input(), 7u32).and_then(|b| ks.scale_u32([N], b, 3u32));
    if has_cb {
        assert_eq!(g.cbable_weight(), 2, "fill + scale = weight 2");
    }

    g.mutate_bind(Buf(seeded(&ctx, 0))).expect("bind Buf");
    let co = g.sync(&ctx).expect("sync 1");
    let r1 = co.map().wait().expect("read 1");
    assert!(
        r1.iter().all(|&v| v == 21),
        "fill 7 * 3 = 21: {:?}",
        &r1[..8]
    );
    drop(r1);
    drop(co);

    let s0 = cb_ids(&g);
    if has_cb {
        assert_eq!(s0.len(), 1, "one homed CB, got {}", s0.len());
    }

    // Mutate Buf to a fresh buffer: the CB baked the old buffer's cl_mem, so it MUST
    // clear (Fill::bind_slots exposes the slot; Fill's cb_leaf_build note_slots it).
    g.mutate_bind(Buf(seeded(&ctx, 0))).expect("mutate Buf");
    let s1 = cb_ids(&g);
    if has_cb {
        assert!(
            s1.is_empty(),
            "mutating the filled slot must clear its CB: s0={s0:?} s1={s1:?}"
        );
    }

    let co = g.sync(&ctx).expect("sync 2");
    let r2 = co.map().wait().expect("read 2");
    assert!(
        r2.iter().all(|&v| v == 21),
        "after mutate the fresh buffer is filled+scaled: {:?}",
        &r2[..8]
    );
    drop(r2);
    drop(co);
}

/// REACH through an IMAGE command. An image slot (`SrcImg`) feeds a weight-2 image
/// command buffer: `bundle2(copy(SrcImg -> dst), copy(other -> mid))` records ONE CB
/// (a root bundle records a single CB, not one per branch), and the first branch
/// BAKES `SrcImg`'s concrete `cl_image` into it. Mutating `SrcImg` to a different
/// image MUST clear that CB — else replay reads the STALE original image and `dst`
/// keeps the old pixels. This guards the image ops' precise-invalidation reach:
/// `ImageCopy`/`ImageFill` must `note_slot`/`cb_reach_extend` via `cb_leaf_build` like
/// the buffer leaves. Before that fix they recorded the command but never noted the
/// captured slot, so this mutate silently no-op'd → stale replay. Both branches'
/// outputs are gathered by the bundle, so `SrcImg` rehomes across the replay. The
/// `dst` result is the load-bearing correctness proof; the id-set check corroborates.
#[test]
fn image_slot_mutate_clears_its_cb() {
    let Some(ctx) = ctx() else { return };
    if !ctx.device().cl3().image_support().unwrap_or(false) {
        eprintln!("SKIP: device has no image support");
        return;
    }
    let has_cb = ctx.has_cl_khr_command_buffer();

    // Two source images pre-seeded to DISTINCT constant pixels; the second bundle
    // branch (`other -> mid`) exists only to make the bundle weight-2 (one CB).
    let src_a = seeded_image(&ctx, 11); // [11,12,13,14]
    let src_b = seeded_image(&ctx, 22); // [22,23,24,25]
    let other = seeded_image(&ctx, 90);
    let dst = RgbaImg::alloc(&ctx, W, H).expect("alloc dst");
    let mid = RgbaImg::alloc(&ctx, W, H).expect("alloc mid");

    let g = bundle2(
        eager_image_copy(slot!(SrcImg).into_slot_input(), dst),
        eager_image_copy(other, mid),
    );
    if has_cb {
        assert_eq!(g.cbable_weight(), 2, "two image copies = weight 2");
    }

    // Read `dst` = the second element of the first branch's (src, dst) output tuple.
    type Co = Checkout<RgbaImg>;
    let read_dst = |g: &_, ctx: &Context| -> Vec<[u32; 4]> {
        let ((_si, d), (_o, _m)): ((Co, Co), (Co, Co)) = DeviceOpExt::sync(g, ctx).expect("sync");
        let out = d.read_alloc().wait().expect("read dst");
        drop(((_si, d), (_o, _m)));
        out
    };

    // Bind SrcImg = src_a, sync: dst must carry src_a's pixels. Homes the CB.
    g.mutate_bind(SrcImg(src_a)).expect("bind src_a");
    let got_a = read_dst(&g, &ctx);
    assert!(
        got_a.iter().all(|&px| px == [11, 12, 13, 14]),
        "SrcImg=src_a: dst should carry src_a pixels; got {:?}",
        &got_a[..2]
    );

    let s0 = cb_ids(&g);
    if has_cb {
        assert_eq!(s0.len(), 1, "expected one homed image CB, got {}", s0.len());
    }

    // Mutate SrcImg = src_b. The CB baked src_a's cl_image via the first copy, so it
    // MUST be cleared. Pre-fix: the image copy never note_slot'd SrcImg, so the CB's
    // captured-slot set was empty → this mutate cleared nothing → the id set stays
    // non-empty AND the replay below reads the stale src_a.
    g.mutate_bind(SrcImg(src_b)).expect("mutate src_b");
    let s1 = cb_ids(&g);
    if has_cb {
        assert!(
            s1.is_empty(),
            "mutating the captured image slot must clear its CB: s0={s0:?} s1={s1:?}"
        );
    }

    // Replay: dst must now reflect src_b — the load-bearing proof that no stale image
    // handle survived in a replayed command buffer.
    let got_b = read_dst(&g, &ctx);
    assert!(
        got_b.iter().all(|&px| px == [22, 23, 24, 25]),
        "after mutate SrcImg=src_b, dst must reflect src_b, not stale src_a; got {:?}",
        &got_b[..2]
    );
}
