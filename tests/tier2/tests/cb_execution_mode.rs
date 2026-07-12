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
