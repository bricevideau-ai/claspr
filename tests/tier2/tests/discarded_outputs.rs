//! A kernel whose outputs a graph discards still runs for effect, so every
//! terminal must cover it — the non-waiting ones (`submit_on`, async `run`)
//! included, whose completion would otherwise fire before it has finished
//! (and its buffers return home while it may still write them).
//!
//! The discarded kernel is held back by a user event; the independent tail is
//! free to run; the terminal's completion must NOT fire until the held kernel
//! has run. Caveat: this only discriminates on a device that really runs ready
//! commands past a blocked one AND doesn't batch the two kernels into one
//! command buffer — on PoCL (one CB) and rusticl (in-order execution) it
//! passes with or without the fix; it guards the fix on such hardware.

use claspr::eager::DeviceOpExt;
use claspr::{DeviceSlice, Event, complete_user_event, create_user_event};
use claspr_test_kernels::kernels;
use claspr_test_support::{N, ctx};
use opencl3::event::{CL_COMPLETE, retain_event};
use std::time::{Duration, Instant};

/// Whether `event` completes within `timeout`.
fn completes_within(event: &Event, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if event
            .command_execution_status()
            .map(|s| s.0 == CL_COMPLETE)
            .unwrap_or(false)
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

#[test]
fn submit_on_covers_a_kernel_whose_outputs_are_discarded() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let gate = create_user_event(&ctx).expect("user event");
    // SAFETY: a second owned handle on the same event (retained for it).
    let gate_dep = unsafe {
        retain_event(gate.get()).expect("retain");
        Event::new(gate.get())
    };
    let alloc = || DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let g = ks
        .slow_pair_u32([N], alloc(), alloc(), 1)
        .after(gate_dep)
        .and_then(|(_a, _b)| ks.fill_u32([N], alloc(), 1u32));
    let queue = ctx.default_outoforder_queue(ctx.device()).expect("queue");
    let done = g.submit_on(&*queue).expect("submit");
    let early = completes_within(&done, Duration::from_millis(500));
    complete_user_event(&gate, CL_COMPLETE).expect("release the gate");
    done.wait().expect("drain");
    assert!(
        !early,
        "the terminal completed while the discarded kernel was still held back"
    );
}
