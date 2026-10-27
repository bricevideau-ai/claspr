//! **Keyed command-buffer cache** (`CbLru`) — a `mutate_bind` that only changes a
//! slot's VALUE must not throw the recorded command buffer away.
//!
//! Background (`NOTES.md`, "Graph mutation invalidates the recorded CB wholesale"):
//! invalidation used to be precise about WHICH CBs died but had no notion of how
//! much of one changed, so rebinding a single scalar forced a full re-record — 24
//! `clCommandNDRangeKernelKHR` per step in the miniWeather port. Real replay loops
//! cycle through a handful of binding states, so the CB for the state you are
//! returning to was usually recorded moments ago and thrown out.
//!
//! These tests assert the RECORD COUNT (per graph, via `cb_records_of`), not
//! timings — the count is deterministic and is the thing that actually regressed.

use claspr::eager::{DeviceOp, DeviceOpExt, bundle2, download, fill};
use claspr::{DeviceSlice, slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::{N, cb_records_of, ctx, homed_cb_count, seeded};

slots! {
    Factor: u32,
    Pre: u32,
    Src: DeviceSlice<u32>,
    Dst: DeviceSlice<u32>,
}

/// Command buffers `g`'s root records while running `f`. Per-GRAPH, not the
/// process-wide counter — cargo runs these tests in parallel, and a global delta
/// would count the other tests' records too.
fn records_during<O: DeviceOp>(g: &O, f: impl FnOnce()) -> u64 {
    let before = cb_records_of(g);
    f();
    cb_records_of(g) - before
}

/// A weight-3 chain whose middle kernel scales by a slot. `fill` resets each run, so
/// the result is a pure function of the bound factor.
macro_rules! scaled_chain {
    ($ctx:expr, $ks:expr) => {{
        let buf = DeviceSlice::<u32>::alloc_zero($ctx, N).expect("alloc");
        fill(buf, 2u32)
            .and_then(|b| $ks.scale_u32([N], b, slot!(Factor)))
            .and_then(|b| $ks.scale_u32([N], b, 1u32))
    }};
}

/// **The regression.** Alternating one scalar between two values must record TWO
/// command buffers total — one per state — and then replay both forever. Before the
/// keyed cache this recorded once per sync.
#[test]
fn alternating_scalar_records_two_cbs_not_one_per_sync() {
    let Some(ctx) = ctx() else { return };
    if !ctx.has_cl_khr_command_buffer() {
        eprintln!("SKIP: device lacks cl_khr_command_buffer");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = scaled_chain!(&ctx, ks);
    g.mutate_bind(Factor(3u32)).expect("bind");

    let n_sync = 12;
    let records = records_during(&g, || {
        for i in 0..n_sync {
            g.mutate_bind(Factor(if i % 2 == 0 { 3u32 } else { 4u32 }))
                .expect("mutate");
            let co = g.sync(&ctx).expect("sync");
            let v = co.map().wait().expect("read");
            let want = if i % 2 == 0 { 6 } else { 8 };
            assert!(v.iter().all(|&x| x == want), "iter {i}: {:?}", &v[..4]);
            drop(v);
            drop(co);
        }
    });
    assert_eq!(
        records, 2,
        "two binding states over {n_sync} syncs must record exactly two CBs"
    );
    // Both states are resident: the current one plus the parked other.
    assert_eq!(homed_cb_count(&g), 2, "both states should be cached");
}

/// Re-binding a slot to the value it ALREADY holds must not disturb the cache at
/// all: same digest, so no rotation, no re-record. (Previously this re-recorded,
/// because invalidation keyed on "was this cell touched", not "did it change".)
#[test]
fn rebinding_the_same_value_does_not_re_record() {
    let Some(ctx) = ctx() else { return };
    if !ctx.has_cl_khr_command_buffer() {
        eprintln!("SKIP: device lacks cl_khr_command_buffer");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = scaled_chain!(&ctx, ks);
    g.mutate_bind(Factor(3u32)).expect("bind");
    let co = g.sync(&ctx).expect("warm-up sync");
    drop(co);

    let records = records_during(&g, || {
        for _ in 0..8 {
            g.mutate_bind(Factor(3u32)).expect("same-value rebind");
            drop(g.sync(&ctx).expect("sync"));
        }
    });
    assert_eq!(records, 0, "same-value rebind must not re-record");
}

/// A **buffer** ping-pong (gray-scott `run_swap`'s shape) is the other tier: the
/// mutation changes `cl_mem` identity, not a scalar. Two crossed states, two records.
#[test]
fn buffer_pingpong_records_two_cbs() {
    let Some(ctx) = ctx() else { return };
    if !ctx.has_cl_khr_command_buffer() {
        eprintln!("SKIP: device lacks cl_khr_command_buffer");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = bundle2(
        ks.scale_u32([N], slot!(Src), 1u32),
        ks.scale_u32([N], slot!(Dst), 1u32),
    );
    g.mutate_call((Src(seeded(&ctx, 1)), Dst(seeded(&ctx, 1))))
        .expect("initial bind");

    let records = records_during(&g, || {
        for _ in 0..10 {
            let (src_co, dst_co) = g.sync(&ctx).expect("sync");
            // Cross them: each slot adopts the other's buffer.
            g.mutate_call((Src(dst_co), Dst(src_co))).expect("swap");
        }
    });
    assert_eq!(records, 2, "a period-2 buffer swap must record two CBs");
}

/// A cycle LONGER than the cache holds must still be correct — it just re-records.
/// This pins the cap as real rather than an unbounded pile of live CB handles: five
/// states through a four-deep cache evict, so the count keeps climbing.
#[test]
fn cycle_longer_than_cap_evicts_and_stays_correct() {
    let Some(ctx) = ctx() else { return };
    if !ctx.has_cl_khr_command_buffer() {
        eprintln!("SKIP: device lacks cl_khr_command_buffer");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = scaled_chain!(&ctx, ks);
    g.mutate_bind(Factor(1u32)).expect("bind");

    // 3 states fit under CB_LRU_CAP (4 parked + 1 current): record 3, then hit.
    let three = records_during(&g, || {
        for i in 0..12 {
            let f = 1 + (i % 3) as u32;
            g.mutate_bind(Factor(f)).expect("mutate");
            let co = g.sync(&ctx).expect("sync");
            let v = co.map().wait().expect("read");
            assert!(v.iter().all(|&x| x == 2 * f), "f={f}: {:?}", &v[..4]);
            drop(v);
            drop(co);
        }
    });
    assert_eq!(three, 3, "a 3-cycle fits the cache: one record per state");

    // 8 states do NOT fit; correctness is unaffected, records keep coming.
    let many = records_during(&g, || {
        for i in 0..16 {
            let f = 10 + (i % 8) as u32;
            g.mutate_bind(Factor(f)).expect("mutate");
            let co = g.sync(&ctx).expect("sync");
            let v = co.map().wait().expect("read");
            assert!(v.iter().all(|&x| x == 2 * f), "f={f}: {:?}", &v[..4]);
            drop(v);
            drop(co);
        }
    });
    assert!(
        many > 8,
        "an 8-cycle through a 5-slot cache must evict and re-record, got {many}"
    );
    assert!(
        homed_cb_count(&g) <= claspr::eager::CB_LRU_CAP + 1,
        "cache must stay capped, holding {}",
        homed_cb_count(&g)
    );
}

/// A chain whose first scale's slot is bound BEFORE composing (`call` on the
/// kernel, inside the `and_then` closure — the curry-then-compose idiom) and whose
/// second scale's slot stays open for the root. `fill` resets each run.
macro_rules! precurried_chain {
    ($ctx:expr, $ks:expr) => {{
        let buf = DeviceSlice::<u32>::alloc_zero($ctx, N).expect("alloc");
        fill(buf, 2u32)
            .and_then(|b| $ks.scale_u32([N], b, slot!(Pre)).call((Pre(5u32),)))
            .and_then(|b| $ks.scale_u32([N], b, slot!(Factor)))
            .and_then(download)
    }};
}

/// A slot bound before composing was bound on the SUBGRAPH, before the root node
/// (and its CB cache) existed, so the root never saw it bound. That used to make
/// every recorded CB unkeyable — never parked — and alternating the root's slot
/// re-recorded on every switch (6 records in 6 syncs instead of 2).
#[test]
fn slot_bound_before_composing_keeps_the_cache_keyed() {
    let Some(ctx) = ctx() else { return };
    if !ctx.has_cl_khr_command_buffer() {
        eprintln!("SKIP: device lacks cl_khr_command_buffer");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = precurried_chain!(&ctx, ks);
    let records = records_during(&g, || {
        for i in 0..12u32 {
            let f = if i % 2 == 0 { 3 } else { 7 };
            g.mutate_bind(Factor(f)).expect("mutate");
            let out = g.sync(&ctx).expect("sync");
            assert!(
                out.iter().all(|&v| v == 2 * 5 * f),
                "sync {i}: got {}",
                out[0]
            );
        }
    });
    assert_eq!(records, 2, "one CB per state, then replay");
}

/// The pre-composition binding is keyed as "unchanged since the root was built".
/// Re-binding that slot later through the root must invalidate those CBs: results
/// stay correct through the change and back (an extra record is fine; a stale
/// replay is not).
#[test]
fn rebinding_a_precurried_slot_through_the_root_stays_correct() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = precurried_chain!(&ctx, ks);
    for (pre, f) in [(5, 3), (5, 7), (11, 3), (11, 7), (5, 3), (5, 7), (11, 3)] {
        g.mutate_call((Pre(pre), Factor(f))).expect("mutate");
        let out = g.sync(&ctx).expect("sync");
        assert!(
            out.iter().all(|&v| v == 2 * pre * f),
            "pre={pre} f={f}: got {}",
            out[0]
        );
    }
}
