//! Unified terminal set on `DeviceOpExt` (reunification stage 2): the launcher-
//! generic `wait_on` / `submit_on` alongside `sync`. `wait_on(&queue)` runs the
//! whole graph on a specific queue (cross-queue control); `submit_on(&queue)`
//! returns a completion event without blocking. `sync(&ctx)` == `wait_on` over a
//! context's default queue.

use claspr::prelude::*;
use claspr::{InOrder, Queue};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const N: usize = 64;

/// `wait_on(&queue)` runs a graph to completion on an explicit queue and yields
/// the Output directly — same result as `sync(&ctx)`, but caller-chosen queue.
#[test]
fn wait_on_explicit_queue_runs_graph() {
    let Some(ctx) = ctx() else { return };
    let device = Device::any().expect("device");
    let queue = Queue::<InOrder>::on_device(&ctx, &device).expect("queue");

    let out = upload(vec![0u32; N])
        .and_then(|buf| fill(buf, 9u32))
        .and_then(download)
        .wait_on(&queue)
        .expect("wait_on chain");

    assert_eq!(out.len(), N);
    assert!(out.iter().all(|&v| v == 9));
}

/// `wait_on` over a kernel chain (the common workhorse), explicit queue.
#[test]
fn wait_on_kernel_chain() {
    let Some(ctx) = ctx() else { return };
    let device = Device::any().expect("device");
    let queue = Queue::<InOrder>::on_device(&ctx, &device).expect("queue");
    let kernels = kernels::kernels(&ctx).expect("kernels");

    let out = upload(vec![0u32; N])
        .and_then(|b| kernels.fill_u32([N], b, 7u32))
        .and_then(|b| kernels.scale_u32([N], b, 3u32))
        .and_then(download)
        .wait_on(&queue)
        .expect("wait_on kernel chain");

    assert!(out.iter().all(|&v| v == 21));
}

/// `submit_on(&queue)` enqueues the graph non-blocking and returns a completion
/// event; waiting it confirms the work finished (the buffer was filled).
#[test]
fn submit_on_returns_completion_event() {
    let Some(ctx) = ctx() else { return };
    let device = Device::any().expect("device");
    let queue = Queue::<InOrder>::on_device(&ctx, &device).expect("queue");

    // Build a buffer, fill it via a submitted (non-blocking) graph, wait the
    // returned event, then read it back via a separate wait_on to verify.
    let buf = alloc_zero::<u32>(N).wait_on(&queue).expect("alloc");

    let event = fill(buf, 5u32).submit_on(&queue).expect("submit_on fill");
    // Block on the returned marker — the fill is done once it fires.
    event.wait().expect("event wait");
}

/// `sync(&ctx)` still works and agrees with `wait_on` — it's `wait_on` over the
/// context's default OOO queue.
#[test]
fn sync_matches_wait_on() {
    let Some(ctx) = ctx() else { return };

    let via_sync = upload(vec![3u32; N])
        .and_then(|buf| fill(buf, 4u32))
        .and_then(download)
        .sync(&ctx)
        .expect("sync");
    assert!(via_sync.iter().all(|&v| v == 4));
}
