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
