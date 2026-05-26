//! Sub-step-1 + transfer coverage: end-to-end chain that mirrors the
//! spike's scenario_1 shape — `upload → kernel → download` with no
//! per-step `with_context` boilerplate for the buffer transfers.

use claspr::Context;
use claspr_async::{DeviceOperation, download, upload, value, with_context};
use claspr_test_kernels::kernels;
use std::sync::Arc;

const N: usize = 256;
const FILL_VALUE: u32 = 0xfeed_cafe;

#[test]
fn linear_chain_upload_kernel_download() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let result: Vec<u32> = upload(vec![0u32; N])
        // Kernel still needs with_context until Phase 4 emits _op variants.
        .and_then(|buf| {
            with_context(move |ec| {
                let kernels = kernels::kernels(ec.context())?;
                kernels.fill_u32(ec, [N], &buf, FILL_VALUE).wait()?;
                Ok::<_, claspr::Error>(buf)
            })
        })
        .and_then(download)
        .sync(&ctx)
        .expect("chain sync");

    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == FILL_VALUE));
}

#[test]
fn value_passthrough() {
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

#[test]
fn upload_accepts_arc_source_caller_retains_clone() {
    // Validates the polymorphic UploadSource path — Arc<[T]> input,
    // caller keeps its own clone, both stay alive until OpenCL is
    // done.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let shared: Arc<[u32]> = Arc::from(vec![7u32; N]);
    let kept_by_caller = Arc::clone(&shared);

    let result: Vec<u32> = upload(Arc::clone(&shared))
        .and_then(download)
        .sync(&ctx)
        .expect("arc upload");
    assert!(result.iter().all(|&v| v == 7));
    // Caller's clone is still usable; data heap is alive.
    assert_eq!(kept_by_caller[0], 7);
}
