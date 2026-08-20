//! Eager-API port of `host_and_profile.rs` — the `and_then_host` mid-chain host
//! work cases. The two `.profiled` cases are ported in `eager_profile.rs` (the
//! eager `.profiled` hook is `DeviceProfileExt`); this file keeps only the
//! host-seam half. All four originals are accounted for across the two files.
//!
//! Old → new mapping:
//!   `value(v).and_then(|x| upload!(x))` → `upload(v)`
//!   `.and_then_host(|view|…)`           → same method on `DeviceOpExt`

use claspr::DeviceSlice;
use claspr::eager::{DeviceOpExt, upload, value};
use claspr::{slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx_profiling;
use std::sync::{Arc, Mutex};

const N: usize = 128;

// ── and_then_host ────────────────────────────────────────────────────

#[test]
fn and_then_host_sum_between_device_stages() {
    let Some(ctx) = ctx_profiling(false) else {
        return;
    };

    let kernels = kernels::kernels(&ctx).expect("load kernels");
    // upload + fill + (host) sum-in-place via mapped view. The closure returns
    // Result<()>; the reduction value flows out via the canonical
    // Arc<Mutex<_>> side-effect channel.
    let sum_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&sum_cell);
    let _final_buf = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 3))
        .and_then_host(move |slice: &mut [u32]| {
            *cell.lock().unwrap() = slice.iter().sum();
            Ok(())
        })
        .sync(&ctx)
        .expect("and_then_host chain");
    let sum = *sum_cell.lock().unwrap();
    assert_eq!(sum, 3 * N as u32);
}

#[test]
fn and_then_host_error_propagates() {
    let Some(ctx) = ctx_profiling(false) else {
        return;
    };
    // Closure returns Err → the eager host seam surfaces the original Rust
    // variant at the terminal (not the OpenCl(-1) cascade).
    let err = value(())
        .and_then_host(|()| -> claspr::Result<()> { Err(claspr::Error::SvmNotAvailable) })
        .sync(&ctx)
        .expect_err("expected error");
    assert!(matches!(err, claspr::Error::SvmNotAvailable), "got {err:?}",);
}

// ── REUSABLE host seam: replay across syncs (#211) ───────────────────
//
// The host seam used to be a one-shot `FnOnce` (stored `Mutex<Option<F>>`,
// `.take()`n on first execute) — a graph containing one could only be `sync`'d
// once. It is now `Fn` (kept in an `Arc`, cloned per run for the worker thread),
// so a host-seam graph REPLAYS and the closure re-runs every `sync`. These two
// tests prove replay + that the closure genuinely re-executes each run.

slots! { Buf: DeviceSlice<u32> }

/// A host-seam graph `sync`'d TWICE over the SAME buffer handle. The seam DOUBLES
/// the mapped view each run, so the buffer QUADRUPLES across two syncs — which can
/// only happen if the `Fn` closure re-ran on the replay (a one-shot `FnOnce` would
/// error on the 2nd `sync`). `scale_u32(slot, 1)` is an identity kernel head that
/// makes the slot-bound buffer flow into the seam.
#[test]
fn and_then_host_replays_and_reruns_each_sync() {
    let Some(ctx) = ctx_profiling(false) else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Reusable graph: identity kernel head over a slot-bound buffer, then a host
    // seam that doubles every element in place. The slot is behind the seam; `bind`
    // reaches it (the seam forwards `bind_slots`) and the buffer rehomes to its
    // cell across replays (the seam threads the source's home). Closure captures
    // NOTHING and only borrows the view — the right shape for something replayed.
    let buf = DeviceSlice::<u32>::from_slice(&ctx, &[3u32; N]).expect("seed buffer");
    let g = kernels
        .scale_u32([N], slot!(Buf), 1u32)
        .and_then_host(|view: &mut [u32]| {
            for slot in view.iter_mut() {
                *slot = slot.wrapping_mul(2);
            }
            Ok(())
        })
        .bind(Buf(buf));

    // Run 1: 3 × 1 (kernel) × 2 (host seam) = 6. Borrowing `map` so the Buf slot
    // rehomes on the Checkout's drop and the graph re-arms for replay.
    let co = g.sync(&ctx).expect("host-seam run 1");
    {
        let view = co.map().wait().expect("map 1");
        assert!(view.iter().all(|&v| v == 6), "run1 got {:?}", &view[..4]);
    }
    drop(co); // rehome Buf for replay

    // Run 2 (replay over the SAME handle, now holding 6): 6 × 1 × 2 = 12. Proves
    // the seam re-ran — a one-shot FnOnce would have errored here instead.
    let co = g.sync(&ctx).expect("host-seam run 2 (replay)");
    {
        let view = co.map().wait().expect("map 2");
        assert!(view.iter().all(|&v| v == 12), "run2 got {:?}", &view[..4]);
    }
    drop(co);
}

/// Replay stress: the SAME host-seam graph `sync`'d N times in a loop, with an
/// external `Arc<Mutex<u32>>` counter incremented by the closure each run. After
/// the loop the counter equals the run count — direct proof the `Fn` closure fired
/// on every replay (and that captures are shared by borrow, not move-consumed).
#[test]
fn and_then_host_loop_reruns_closure_every_iteration() {
    let Some(ctx) = ctx_profiling(false) else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let calls = Arc::new(Mutex::new(0u32));
    let calls_c = Arc::clone(&calls);

    let buf = DeviceSlice::<u32>::from_slice(&ctx, &[0u32; N]).expect("seed buffer");
    let g = kernels
        .scale_u32([N], slot!(Buf), 1u32)
        .and_then_host(move |_view: &mut [u32]| {
            *calls_c.lock().unwrap() += 1;
            Ok(())
        })
        .bind(Buf(buf));

    const RUNS: u32 = 5;
    for _ in 0..RUNS {
        let co = g.sync(&ctx).expect("loop replay sync");
        drop(co); // rehome for the next iteration
    }
    assert_eq!(
        *calls.lock().unwrap(),
        RUNS,
        "the host seam closure must re-run once per sync"
    );
}

// ── profile ──────────────────────────────────────────────────────────
//
// `profile_chain_fires_callback_when_profiling_on` and
// `profile_chain_errors_when_profiling_off` are ported in `eager_profile.rs`
// (the eager `.profiled` hook is `DeviceProfileExt::profiled`).
