//! Eager-API port of `run_await.rs`: the `.run(&ctx).await` async terminal.
//!
//! The chain is built and submitted exactly like the sync case; the difference
//! is the terminal — instead of `.sync(&ctx)`, the user calls `.run(&ctx)` and
//! `.await`s the resulting [`claspr::DeviceChainFuture`]. Under the hood,
//! completion is signaled by an `clEnqueueMarkerWithWaitList` event whose
//! `CL_COMPLETE` callback wakes the future's waker (the Tier-1 `EventFuture`
//! machinery, gated behind the `async-events` feature — on by default).
//!
//! Uses `futures::executor::block_on` so the test harness doesn't need a full
//! async runtime — same harness as the old `run_await.rs`.
//!
//! Old → new mapping (mirrors `eager_chain.rs` / `eager_error.rs`):
//!   `value(v).and_then(|x| upload!(x))` → `upload::<u32, ReadWrite, _>(v)`
//!   `download!(buf)`                    → `.and_then(download)`
//!   pure host-value arithmetic chain    → lifted single `value(..)` (the eager
//!     `and_then` hands a `Pipe<u32>`, not the host scalar — same DEVIATION as
//!     `eager_chain.rs::value_passthrough`).
//!   chain error via `with_context`      → `.and_then_host(|_| Err(..))` (the
//!     eager host seam surfaces the closure's exact `Error` variant).

use claspr::eager::{DeviceOpExt, download, upload, value};
use claspr::{Context, Error};
use claspr_test_kernels::kernels;
use futures::executor::block_on;

const N: usize = 256;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

/// run_await.rs::await_simple_chain — upload → fill kernel → download via `.await`.
#[test]
fn await_simple_chain() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let chain = upload::<u32, claspr::ReadWrite, _>(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 0x1234_5678))
        .and_then(download);

    let result: Vec<u32> = block_on(chain.run(&ctx)).expect("await chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 0x1234_5678));
}

/// run_await.rs::await_pure_value_chain — a pure host value resolved via `.await`.
///
/// DEVIATION FROM run_await.rs (same as `eager_chain.rs::value_passthrough`):
/// the old test chained `value(n).and_then(|n| value(n + ..))` transforming the
/// concrete `u32` between stages. In the eager API `and_then`'s closure receives
/// a `Pipe<u32>`, not the value, so in-graph host arithmetic is impossible. We
/// assert the same final value (84) by lifting the computed value up front and
/// awaiting it — exercising the `.run().await` terminal over a pure-value op.
#[test]
fn await_pure_value_chain() {
    let Some(ctx) = ctx() else { return };

    let computed = 10u32.wrapping_add(32).wrapping_mul(2);
    let result: u32 = block_on(value(computed).run(&ctx)).expect("await pure chain");
    assert_eq!(result, 84);
}

/// run_await.rs::await_propagates_chain_error — a chain `Err` surfaces at
/// `.run().await`.
///
/// The eager host seam runs its closure synchronously inside `execute` and
/// returns the closure's exact `Error` variant; `run` turns that synchronous
/// `Err` into `DeviceChainFuture::Errored`, which surfaces on the first poll.
#[test]
fn await_propagates_chain_error() {
    let Some(ctx) = ctx() else { return };

    let chain = upload::<u32, claspr::ReadWrite, _>(vec![0u32; 16]).and_then_host(
        |view: &mut [u32]| -> claspr::Result<()> {
            Err(Error::LengthMismatch {
                src: view.len(),
                dst: 8,
            })
        },
    );

    let err = block_on(chain.run(&ctx)).expect_err("chain should error");
    assert!(matches!(err, Error::LengthMismatch { .. }), "got {err:?}");
}

/// Multi-output terminal via `.run().await`. Previously `run` was single-output
/// only (it drained `output_pipe`, which multi-output ops never fill, so a
/// bundle terminal returned `NotSupported`). Now `run` gathers via `collect`,
/// the same arity-agnostic seam `sync` uses, so a bundle resolves to its
/// reconstructed tuple over the async terminal too.
#[test]
fn await_multi_output_bundle() {
    use claspr::eager::bundle2;
    let Some(ctx) = ctx() else { return };

    let (a, b): (u32, u32) =
        block_on(bundle2(value(11u32), value(22u32)).run(&ctx)).expect("await bundle");
    assert_eq!((a, b), (11, 22));
}
