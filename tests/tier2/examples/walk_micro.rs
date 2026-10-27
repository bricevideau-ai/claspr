//! **Replay-walk micro-bench** — the per-node cost of walking a Tier-2 graph on
//! `sync`, with NO device commands: a `lift`ed buffer followed by L `forward`
//! passthrough links, replayed ITERS times, at L = 50, 100, 200.
//!
//! What it guards: the walk must stay LINEAR in chain length. `and_then` erases
//! the chain it extends, so every link is reached through a `DeviceDynOp`, and
//! any per-node query that recurses down the spine (instead of being memoized —
//! see `PredMemo`) makes the walk O(n²): the per-link figure then GROWS with L
//! (seen once as 102 / 138 / 190 ns at 50 / 100 / 200). A flat per-link column
//! is the healthy shape; on PoCL/M3 it reads ~73 ns. The absolute number is
//! mostly the terminal's lend/rehome lifecycle, not dispatch.
//!
//! ```sh
//! OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors \
//!   cargo run -p claspr-tier2-tests --example walk_micro --release
//! ```
//!
//! `ITERS` (default 20000) overrides the replay count.
use claspr::eager::{DeviceOpExt, forward, lift};
use claspr::{Context, DeviceSlice};
use std::time::Instant;

/// Ten `forward` links appended to `$g`.
macro_rules! ten {
    ($g:expr) => {
        $g.and_then(forward)
            .and_then(forward)
            .and_then(forward)
            .and_then(forward)
            .and_then(forward)
            .and_then(forward)
            .and_then(forward)
            .and_then(forward)
            .and_then(forward)
            .and_then(forward)
    };
}

/// Fifty links: five `ten!`s.
macro_rules! fifty {
    ($g:expr) => {
        ten!(ten!(ten!(ten!(ten!($g)))))
    };
}

/// Warm up, then time `iters` syncs of `g` and print the median per-sync and
/// per-link cost.
macro_rules! bench {
    ($ctx:expr, $iters:expr, $links:expr, $g:expr) => {{
        let g = $g;
        for _ in 0..200 {
            drop(g.sync($ctx).expect("warm-up sync"));
        }
        let mut samples = Vec::with_capacity($iters);
        for _ in 0..$iters {
            let t = Instant::now();
            let out = g.sync($ctx).expect("sync");
            samples.push(t.elapsed().as_nanos() as u64);
            drop(out);
        }
        samples.sort_unstable();
        let median_us = samples[$iters / 2] as f64 / 1000.0;
        println!(
            "links={:<4} per-sync median={median_us:>7.2}us  per-link={:>6.1}ns",
            $links,
            median_us * 1000.0 / $links as f64
        );
    }};
}

fn main() {
    let ctx = Context::any().expect("an OpenCL device");
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20000);
    let source = || lift(DeviceSlice::<u32>::alloc_zero(&ctx, 16).expect("buffer"));
    println!(
        "device: {}  iters={iters}",
        ctx.device().name().unwrap_or_else(|_| "?".into())
    );
    bench!(&ctx, iters, 50, fifty!(source()));
    bench!(&ctx, iters, 100, fifty!(fifty!(source())));
    bench!(&ctx, iters, 200, fifty!(fifty!(fifty!(fifty!(source())))));
}
