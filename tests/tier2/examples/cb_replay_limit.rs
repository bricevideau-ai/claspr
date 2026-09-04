//! Repro: sustained replay of ONE cached command buffer eventually fails with
//! `CL_INVALID_OPERATION` (-59) on pocl, at a NON-DETERMINISTIC iteration.
//!
//! Pre-existing (reproduces with the keyed cache reverted). claspr waits correctly —
//! `wait_deps_reconcile` does `clWaitForEvents` on the CB's completion event before
//! returning — so after `sync` the buffer should be back in the Executable state and
//! legal to re-enqueue. Runs both queue orders so a report can say whether the
//! out-of-order path is implicated.
//!
//! ```sh
//! OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors \
//!   cargo run -p claspr-tier2-tests --example cb_replay_limit --release
//! ```
use claspr::eager::{DeviceOpExt, fill};
use claspr::{Context, DeviceSlice};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

/// Replay a weight-3 all-device graph `iters` times on `launcher`, returning the
/// iteration that failed (or `None` if it survived).
fn replay_until_failure<L: claspr::Launcher + ?Sized>(
    ctx: &Context,
    launcher: &L,
    iters: usize,
) -> claspr::Result<Option<(usize, claspr::Error)>> {
    let n = 64usize;
    let ks = kernels::kernels(ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(ctx, n)?;
    let g = fill(buf, 2u32)
        .and_then(|b| ks.scale_u32([n], b, 5u32))
        .and_then(|b| ks.scale_u32([n], b, 1u32));
    for i in 0..iters {
        match g.wait_on(launcher) {
            Ok(co) => drop(co),
            Err(e) => return Ok(Some((i, e))),
        }
    }
    Ok(None)
}

fn main() -> claspr::Result<()> {
    let Some(ctx) = ctx() else { return Ok(()) };
    if !ctx.has_cl_khr_command_buffer() {
        eprintln!("SKIP: no cl_khr_command_buffer");
        return Ok(());
    }
    let iters: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6000);
    let device = ctx.device().clone();

    for trial in 0..3 {
        let ooo = ctx.default_outoforder_queue(&device)?;
        match replay_until_failure(&ctx, &*ooo, iters)? {
            Some((i, e)) => println!("out-of-order trial {trial}: FAILED at {i}: {e:?}"),
            None => println!("out-of-order trial {trial}: OK ({iters} replays)"),
        }
        let inorder = ctx.default_inorder_queue(&device)?;
        match replay_until_failure(&ctx, &inorder, iters)? {
            Some((i, e)) => println!("in-order     trial {trial}: FAILED at {i}: {e:?}"),
            None => println!("in-order     trial {trial}: OK ({iters} replays)"),
        }
    }
    Ok(())
}
