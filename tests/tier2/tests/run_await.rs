//! Sub-step-3.4 coverage: the `.run(&ctx).await` async terminal.
//!
//! The chain is built and submitted exactly like the sync case; the
//! difference is the terminal — instead of `.sync(&ctx)`, the user
//! calls `.run(&ctx)` and `.await`s the resulting [`ChainFuture`].
//! Under the hood, completion is signaled by an
//! `clEnqueueMarkerWithWaitList` event whose `CL_COMPLETE` callback
//! wakes the future's waker.
//!
//! Uses `futures::executor::block_on` so the test harness doesn't
//! need a full async runtime.

use claspr::Context;
use claspr_async::{DeviceOperation, DeviceOperationHostExt, download, upload, value};
use claspr_test_kernels::kernels;
use futures::executor::block_on;

const N: usize = 256;

#[test]
fn await_simple_chain() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let chain = value(vec![0u32; N])
        .and_then(upload)
        .and_then(|buf| kernels.fill_u32([N], buf, 0x1234_5678))
        .and_then(download);

    let result: Vec<u32> = block_on(chain.run(&ctx)).expect("await chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 0x1234_5678));
}

#[test]
fn await_pure_value_chain() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let chain = value(10u32)
        .and_then(|n| value(n.wrapping_add(32)))
        .and_then(|n| value(n.wrapping_mul(2)));

    let result: u32 = block_on(chain.run(&ctx)).expect("await pure chain");
    assert_eq!(result, 84);
}

#[test]
fn await_propagates_chain_error() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    // Synthesize a chain error from inside an `.and_then_host` step
    // and assert it surfaces at `.run().await`. The original test
    // used a `with_context` closure to do a fallible Tier 1 read
    // with a mismatched destination size; with `with_context`
    // removed, we construct the same error variant directly. The
    // test's intent (chain Err → terminal Err) is preserved; the
    // mechanism (where the error originates) is different.
    let chain = value(vec![0u32; 16]).and_then(upload).and_then_host(
        |view: &mut [u32]| -> claspr::Result<()> {
            Err(claspr::Error::LengthMismatch {
                src: view.len(),
                dst: 8,
            })
        },
    );

    let err = block_on(chain.run(&ctx)).expect_err("chain should error");
    assert!(
        matches!(err, claspr::Error::LengthMismatch { .. }),
        "got {err:?}",
    );
}
