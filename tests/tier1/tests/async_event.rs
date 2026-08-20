//! The Tier-1 async surface: `EventFuture`'s real Pending→wake path
//! and `IntoFuture for LaunchOp`.
//!
//! The tier2 `.run(&ctx).await` tests drive the async terminal, but
//! always through `block_on` over chains that may already be complete
//! at first poll — the `clSetEventCallback` → `AtomicWaker` bridge
//! was never *provably* exercised, and `IntoFuture for LaunchOp` had
//! zero tests. These pin both.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::task::Poll;

use claspr::{
    Context, DeviceSlice, EventFutureExt, LaunchOp, complete_user_event, create_user_event,
};
use claspr_test_kernels::kernels;
use futures::executor::block_on;
use futures::task::noop_waker;

const N: usize = 1024;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

/// Drive `EventFuture` through a genuine Pending → callback-wake →
/// Ready cycle using a user event completed from another thread.
///
/// The first (manual) poll happens while the event is still
/// `CL_SUBMITTED`, so it MUST return `Pending` and register the
/// callback. The completing thread then sleeps briefly before firing,
/// so `block_on`'s poll almost certainly parks and is woken by the
/// `clSetEventCallback` → `AtomicWaker` bridge (if the thread wins
/// the race anyway, the test still passes — it just exercises the
/// Ready path instead).
#[test]
fn user_event_future_pending_then_callback_wake() {
    let Some(ctx) = ctx() else { return };
    let ev = create_user_event(&ctx).expect("create user event");
    let raw = ev.get() as usize;

    let mut fut = ev.into_future();

    // Manual first poll with a no-op waker: the event is still
    // CL_SUBMITTED, so this must be Pending.
    let waker = noop_waker();
    let mut poll_cx = std::task::Context::from_waker(&waker);
    assert!(
        matches!(Pin::new(&mut fut).poll(&mut poll_cx), Poll::Pending),
        "user event is not complete; first poll must be Pending"
    );

    // Complete the event from another thread after a delay. The
    // future owns the only `Event`, so the completer reconstructs a
    // borrowed view from the raw handle and `forget`s it afterwards —
    // dropping it would release a refcount this thread doesn't own.
    // (The handle stays valid: `fut` keeps the event alive.)
    let completer = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let borrowed = claspr::Event::new(raw as *mut _);
        // 0 == CL_COMPLETE.
        complete_user_event(&borrowed, 0).expect("complete user event");
        std::mem::forget(borrowed);
    });

    block_on(&mut fut).expect("future must resolve Ok after completion");
    completer.join().expect("completer thread");
}

/// `IntoFuture for LaunchOp` — the lower-level Tier-1 builder awaited
/// directly. Launch fill_u32 via `LaunchOp::new`, await it, verify
/// the buffer contents landed.
#[test]
fn launch_op_into_future_runs_kernel() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let kernel = ks.kernel("fill_u32");

    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let op = LaunchOp::new(&ctx, &kernel, [N], (&buf, 0xABCD_EF01u32));
    block_on(op.into_future()).expect("awaited launch");

    let mut out = vec![0u32; N];
    buf.read(&mut out).wait().expect("read back");
    assert!(out.iter().all(|&v| v == 0xABCD_EF01));
}
