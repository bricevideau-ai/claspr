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

use claspr::{Context, DeviceSlice};
use claspr_async::{DeviceOperation, value, with_context};
use claspr_test_kernels::kernels;
use futures::executor::block_on;

const N: usize = 256;

#[test]
fn await_simple_chain() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let chain = value(vec![0u32; N])
        .and_then(|host| with_context(move |ec| DeviceSlice::upload(ec, &host)))
        .and_then(|buf| {
            with_context(move |ec| {
                let kernels = kernels::kernels(ec.context())?;
                kernels.fill_u32(ec, [N], &buf, 0x1234_5678).wait()?;
                Ok::<_, claspr::Error>(buf)
            })
        })
        .and_then(|buf| {
            with_context(move |ec| {
                let mut out = vec![0u32; N];
                buf.download(ec, &mut out).wait()?;
                Ok::<_, claspr::Error>(out)
            })
        });

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

    // Force the chain to fail by passing a bogus length to a download
    // — DeviceSlice::download returns LengthMismatch if dst.len() !=
    // self.len().
    let chain = value(vec![0u32; 16])
        .and_then(|host| with_context(move |ec| DeviceSlice::upload(ec, &host)))
        .and_then(|buf| {
            with_context(move |ec| {
                let mut wrong_size = vec![0u32; 8];
                // This errors synchronously inside execute.
                buf.download(ec, &mut wrong_size).wait()?;
                Ok::<_, claspr::Error>(wrong_size)
            })
        });

    let err = block_on(chain.run(&ctx)).expect_err("chain should error");
    assert!(
        matches!(err, claspr::Error::LengthMismatch { .. }),
        "got {err:?}",
    );
}
