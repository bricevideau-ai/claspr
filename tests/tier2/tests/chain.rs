//! Sub-step-1 coverage: `DeviceOperation` + `value` + `with_context` +
//! `.and_then` + `.sync()`.
//!
//! Builds a small chain that uploads a vector, runs `fill_u32` on it
//! to overwrite every element, then downloads — end-to-end through
//! the Tier 2 surface (no direct Tier 1 `.wait()` at the user's call
//! site, though the kernel itself enqueues via the chain's
//! ExecutionContext-as-Launcher path).

use claspr::{Context, DeviceSlice};
use claspr_async::{DeviceOperation, value, with_context};
use claspr_test_kernels::kernels;

const N: usize = 256;
const FILL_VALUE: u32 = 0xfeed_cafe;

#[test]
fn linear_chain_fill_and_download() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let result: Vec<u32> = value(vec![0u32; N])
        .and_then(|host| with_context(move |ec| DeviceSlice::upload(ec, &host)))
        .and_then(|buf| {
            with_context(move |ec| {
                // ExecutionContext implements Launcher, so the Tier 1
                // launch op enqueues on the chain's OOO queue.
                let kernels = kernels::kernels(ec.context())?;
                kernels.fill_u32(ec, [N], &buf, FILL_VALUE).wait()?;
                Ok::<_, claspr::Error>(buf)
            })
        })
        .and_then(|buf| {
            with_context(move |ec| {
                let mut out = vec![0u32; N];
                buf.download(ec, &mut out)?;
                Ok::<_, claspr::Error>(out)
            })
        })
        .sync(&ctx)
        .expect("chain sync");

    assert_eq!(result.len(), N);
    assert!(
        result.iter().all(|&v| v == FILL_VALUE),
        "expected every element to be {FILL_VALUE:#x}",
    );
}

#[test]
fn value_passthrough() {
    // No device needed for this smoke test — the chain only carries
    // host values.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let out = value(42u32)
        .and_then(|n| value(n.wrapping_add(1)))
        .and_then(|n| value(n.wrapping_mul(2)))
        .sync(&ctx)
        .expect("value chain");
    assert_eq!(out, 86);
}
