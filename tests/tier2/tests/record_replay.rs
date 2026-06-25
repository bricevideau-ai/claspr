//! Layer-1 record/replay (software backend, no command buffer).
//!
//! Proves a recordable eager graph can be recorded ONCE and replayed MANY times
//! against the same concrete buffer — the first reusable-pipeline primitive.
//!
//! `record()` takes `&self`, so the graph (and the concrete buffer it owns)
//! survives recording; we replay it N times, then consume the graph through the
//! normal `.wait()` terminal to read the buffer back and assert the fill landed.

use claspr::Context;
use claspr::DeviceSlice;
use claspr::eager::fill;
use claspr::record::RecordExt;
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

/// Record `fill(buf, 7)` once; replay it twice; then run the same graph
/// normally and confirm the buffer holds 7 (the recording did real work).
#[test]
fn fill_records_once_replays_twice() {
    let Some(ctx) = ctx() else { return };

    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let graph = fill(buf, 7u32);

    // Record WITHOUT consuming the graph (&self walk).
    let recorded = graph.record().expect("record");

    // Replay the recording on the device twice.
    recorded.replay(&ctx).expect("replay 1");
    recorded.replay(&ctx).expect("replay 2");

    // Drop the recording (ends the borrow), then consume the graph the normal
    // way to read the buffer back. The fill value must be present.
    drop(recorded);
    let filled = graph.wait().expect("terminal fill");
    let mut readback = vec![0u32; N];
    let out = filled.read(&mut readback).wait().expect("read");
    let _ = out;
    assert!(
        readback.iter().all(|&v| v == 7),
        "expected all 7 after fill, got {:?}",
        &readback[..8]
    );
}

/// Record a kernel (`scale_u32([N], buf, 3)`) over a concrete buffer; replay it
/// twice. Each replay scales in place, so a buffer seeded with 1 holds 3 after
/// one replay and 9 after two.
#[test]
fn kernel_records_once_replays_twice() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Seed a concrete buffer with all-ones.
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let buf = buf.fill(1u32).wait().expect("seed");

    // Record scale-by-3 over the concrete buffer (no execution).
    let graph = ks.scale_u32([N], buf, 3u32);
    let recorded = graph.record().expect("record kernel");

    recorded.replay(&ctx).expect("replay 1"); // 1 -> 3
    recorded.replay(&ctx).expect("replay 2"); // 3 -> 9
    drop(recorded);

    let scaled = graph.wait().expect("terminal scale"); // 9 -> 27
    let mut readback = vec![0u32; N];
    let out = scaled.read(&mut readback).wait().expect("read");
    let _ = out;
    assert!(
        readback.iter().all(|&v| v == 27),
        "expected all 27 (1*3*3*3), got {:?}",
        &readback[..8]
    );
}

/// A multi-stage recorded chain: upload-seeded fill then kernel, replayed twice.
#[test]
fn fill_then_kernel_chain_records_and_replays() {
    use claspr::eager::DeviceOpExt;
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    // fill(buf, 2) -> scale_u32(_, 5): 2 -> 10 per run.
    let graph = fill(buf, 2u32).and_then(|b| ks.scale_u32([N], b, 5u32));
    let recorded = graph.record().expect("record chain");

    recorded.replay(&ctx).expect("replay 1");
    recorded.replay(&ctx).expect("replay 2");
    drop(recorded);

    // Terminal run: fill resets to 2, scale -> 10.
    let out = graph.sync(&ctx).expect("sync chain");
    let mut readback = vec![0u32; N];
    let r = out.read(&mut readback).wait().expect("read");
    let _ = r;
    assert!(
        readback.iter().all(|&v| v == 10),
        "expected all 10 (2*5), got {:?}",
        &readback[..8]
    );
}

/// Repeated replays are stable over many iterations.
#[test]
fn fill_replays_many_times() {
    let Some(ctx) = ctx() else { return };
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let graph = fill(buf, 0xABCD_u32);
    let recorded = graph.record().expect("record");
    for i in 0..16 {
        recorded
            .replay(&ctx)
            .unwrap_or_else(|e| panic!("replay {i}: {e:?}"));
    }
}
