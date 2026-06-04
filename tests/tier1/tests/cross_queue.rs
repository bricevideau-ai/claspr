//! `.after(event)` between two **distinct** command queues — the
//! genuine cross-queue case the API was designed for. `basic.rs`'s
//! similar test reuses the same queue; this file uses two separate
//! `Queue::<InOrder>::on_device` instances so the event-wait-list
//! handshake is what's actually doing the ordering.

use claspr::{Context, Device, DeviceSlice, InOrder, Queue};
use claspr_test_kernels::kernels;
use std::panic;

const N: usize = 512;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

#[test]
fn after_event_orders_launch_on_second_queue() {
    let Some(ctx) = ctx() else { return };
    let device: Device = ctx.device().clone();
    let q_producer = Queue::<InOrder>::on_device(&ctx, &device).expect("producer queue");
    let q_consumer = Queue::<InOrder>::on_device(&ctx, &device).expect("consumer queue");
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");

    // Producer queue fills with 6. Submit returns an Event we hand to
    // the consumer queue as a wait-list dep. Without `.after(event)`,
    // the consumer launch could (in principle) start before the
    // producer's fill — they're on different queues with no implicit
    // ordering between them.
    let (buf, fill_event) = kernels
        .fill_u32([N], buf, 6)
        .submit(&q_producer)
        .expect("submit fill on producer");
    let buf = kernels
        .scale_u32([N], buf, 7)
        .after(fill_event)
        .wait(&q_consumer)
        .expect("scale after on consumer");

    let mut out = vec![0u32; N];
    buf.read(&mut out).wait(&ctx).expect("read");
    assert!(out.iter().all(|&v| v == 42));
}

#[test]
fn after_all_orders_launch_after_multiple_cross_queue_events() {
    // Two producer queues, each fills its own buffer. A third queue's
    // launch waits on both via `.after_all`. Validates that the wait-
    // list takes more than one event (the empty `.after_all` case is
    // trivially right; multi-event is the real test).
    let Some(ctx) = ctx() else { return };
    let device: Device = ctx.device().clone();
    let q1 = Queue::<InOrder>::on_device(&ctx, &device).expect("q1");
    let q2 = Queue::<InOrder>::on_device(&ctx, &device).expect("q2");
    let q_combine = Queue::<InOrder>::on_device(&ctx, &device).expect("q_combine");
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let a = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc a");
    let b = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc b");
    let out = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc out");

    let (a, ev_a) = kernels.fill_u32([N], a, 10).submit(&q1).expect("fill a");
    let (b, ev_b) = kernels.fill_u32([N], b, 32).submit(&q2).expect("fill b");

    let (_a, _b, out) = kernels
        .add_u32([N], a, b, out)
        .after_all([ev_a, ev_b])
        .wait(&q_combine)
        .expect("add after_all");

    let mut host = vec![0u32; N];
    out.read(&mut host).wait(&ctx).expect("read");
    assert!(host.iter().all(|&v| v == 42));
}

#[test]
fn after_with_cross_context_event_panics_clearly() {
    // Build two independent Contexts (each over the default device),
    // submit a kernel on Context A to get an event, then try to use
    // that event as an `.after()` dep on Context B's queue. The
    // cross-context check must fire as a clear Rust panic instead of
    // letting OpenCL surface a cryptic CL_INVALID_CONTEXT later.
    let Some(_skip_guard) = ctx() else { return };
    let Ok(dev) = Device::any() else { return };
    let ctx_a = Context::builder().device(&dev).build().expect("ctx_a");
    let ctx_b = Context::builder().device(&dev).build().expect("ctx_b");
    let kernels_a = kernels::kernels(&ctx_a).expect("kernels_a");

    // Produce an event on ctx_a.
    let buf_a = DeviceSlice::<u32>::alloc(&ctx_a, N).expect("alloc on ctx_a");
    let (_buf_a, event_from_a) = kernels_a
        .fill_u32([N], buf_a, 1)
        .submit(&ctx_a)
        .expect("submit on ctx_a");

    // Try to use it as an after-dep on ctx_b — should panic.
    let kernels_b = kernels::kernels(&ctx_b).expect("kernels_b");
    let buf_b = DeviceSlice::<u32>::alloc(&ctx_b, N).expect("alloc on ctx_b");
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _ = kernels_b
            .fill_u32([N], buf_b, 2)
            .after(event_from_a)
            .wait(&ctx_b);
    }));
    assert!(
        result.is_err(),
        "cross-context .after() must panic; got Ok instead",
    );
}
