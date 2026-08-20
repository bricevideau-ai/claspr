//! Arced / arc_split / fan_out in a command buffer (design v2). These structural
//! combinators add no device command of their own, but they DELEGATE to their
//! source/branches — an all-device sub-tree under them records into ONE CB.
//!
//! The existing arced/fan_out samples (eager_diamond, eager_fan_out_rearm) start with
//! `upload` and end with `download` (host↔device transfers = seams), so they stay
//! per-op. These tests use CONCRETE pre-allocated buffers + device kernels (no
//! upload/download, no alloc_zero — which is itself a synchronous host op and not a
//! CB command), so the span under the combinator is fully CB-addable and takes a CB.

use claspr::DeviceSlice;
use claspr::eager::{DeviceOp, DeviceOpExt, arc_split, arced, bundle2, fan_out, fill, forward};
use claspr_test_kernels::kernels;
use claspr_test_support::{ctx, homed_cb};

const N: usize = 64;

/// All-device arced + arc_split: `fill` a concrete buffer on-device, `arced` it, split
/// to two read-only kernel branches, combine. No upload/download/alloc → one command
/// buffer, homed at the root, replayed. Proves Arced + ArcSplit delegate their
/// source's CB-addability and alias its sync points to every branch.
#[test]
fn arc_split_all_device_runs_as_command_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ks = &ks;

    // Concrete buffers (alloc is a host op → done up front, outside the CB span).
    let shared = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("shared");
    let a_in = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("a_in");
    let a_out = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("a_out");
    let b_in = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("b_in");
    let b_out = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("b_out");
    let combined = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("combined");

    // shared = fill(5); arced → split to two branches; branch adds a device-filled
    // operand; results summed. All commands are device (fill/add) → CB-addable span.
    let g = arc_split::<2, _>(arced(fill(shared, 5u32))).and_then(move |[s1, s2]| {
        bundle2(
            // branch A: outA = s1 + fill(a_in, 10)
            fill(a_in, 10u32)
                .and_then(move |ain| ks.add_u32([N], s1, ain, a_out))
                .and_then(|(_s, _a, out)| forward(out)),
            // branch B: outB = s2 + fill(b_in, 20)
            fill(b_in, 20u32)
                .and_then(move |bin| ks.add_u32([N], s2, bin, b_out))
                .and_then(|(_s, _b, out)| forward(out)),
        )
        .and_then(move |(ao, bo)| {
            ks.add_u32([N], ao, bo, combined)
                .and_then(|(_a, _b, out)| forward(out))
        })
    });

    // (5+10) + (5+20) = 40 per element. arc_split is a single-sync fan (its read-only
    // Arc-clone branches don't thread homes for a replay loop — same as eager_diamond,
    // which syncs once); we verify the CB is homed on that one sync + the result.
    let co = g.sync(&ctx).expect("sync");
    let got = co.map().wait().expect("read");
    assert!(got.iter().all(|&v| v == 40), "arc_split: {:?}", &got[..8]);
    drop(got);
    drop(co);
    if ctx.has_cl_khr_command_buffer() {
        assert!(
            homed_cb(&g),
            "all-device arc_split should home a command buffer"
        );
    }
}

/// All-device fan_out: N independent `fill` kernels over concrete buffers, no
/// download/alloc in the span. Records every branch's fill into ONE CB, homed at
/// the FanOut. Proves FanOut delegates all-branches CB-addability + sums weight.
#[test]
fn fan_out_all_device_runs_as_command_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ks = &ks;

    let vals: Vec<u32> = vec![3, 4, 5];
    // One concrete buffer per branch (alloc up front, host op).
    let bufs: Vec<DeviceSlice<u32>> = (0..vals.len())
        .map(|_| DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("branch buf"))
        .collect();

    let inputs: Vec<(u32, DeviceSlice<u32>)> = vals.iter().copied().zip(bufs).collect();
    let g = fan_out(inputs, move |(v, buf)| ks.fill_u32([N], buf, v));
    assert!(g.cbable_weight() >= 2, "3 fill branches ≥ 2 commands");

    for i in 0..3 {
        let outs = g.sync(&ctx).unwrap_or_else(|e| panic!("sync {i}: {e:?}"));
        assert_eq!(outs.len(), vals.len(), "iter {i}");
        for (b, co) in outs.into_iter().enumerate() {
            let got = co.map().wait().expect("read branch");
            assert!(
                got.iter().all(|&x| x == vals[b]),
                "iter {i} branch {b}: {:?}",
                &got[..8]
            );
        }
    }
    if ctx.has_cl_khr_command_buffer() {
        assert!(
            homed_cb(&g),
            "all-device fan_out should home a command buffer"
        );
    }
}
