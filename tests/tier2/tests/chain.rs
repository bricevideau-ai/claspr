//! Sub-step-1 + transfer coverage: end-to-end chain that mirrors the
//! spike's scenario_1 shape — `upload → kernel → download` with no
//! per-step `with_context` boilerplate.
//!
//! `_kernel_op` chains use the proc-macro-emitted `_op` variant
//! (Phase 4): the kernel call is just another node, no inner
//! `with_context` wrap.

use claspr::Context;
use claspr_async::{DeviceOperation, bundle, download, upload, value};
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

    // Build the chain once so we can borrow `kernels` for its full
    // lifetime. Each `_op` call returns a struct that holds `&kernels`
    // — the chain executes to completion before `kernels` drops.
    let kernels = kernels::kernels(&ctx).expect("kernels load");

    let result: Vec<u32> = upload!(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, FILL_VALUE))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("chain sync");

    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == FILL_VALUE));
}

#[test]
fn three_slice_kernel_op_threads_tuple_output() {
    // `add_u32` takes `(a, b, out)` — the emitted Output is the
    // 3-tuple of slices. Destructure to keep going.
    //
    // Parallel uploads via `bundle!` — claspr-async clones the input
    // deps into each child and joins their events with a marker, so
    // the three uploads pipeline on the OOO queue.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("kernels load");

    let result: Vec<u32> = bundle!(
        upload!(vec![3u32; N]),
        upload!(vec![4u32; N]),
        upload!(vec![0u32; N]),
    )
    .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
    .and_then(|(_a, _b, out)| download!(out))
    .sync(&ctx)
    .expect("add chain");
    assert!(result.iter().all(|&v| v == 7));
}

#[test]
fn kernel_op_chains_two_kernels() {
    // fill → scale, both via _op. No with_context anywhere.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("kernels load");

    let result: Vec<u32> = upload!(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 5))
        .and_then(|buf| kernels.scale_u32([N], buf, 7))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("fill+scale chain");
    assert!(result.iter().all(|&v| v == 35));
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

    let result: Vec<u32> = upload!(Arc::clone(&shared))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("arc upload");
    assert!(result.iter().all(|&v| v == 7));
    // Caller's clone is still usable; data heap is alive.
    assert_eq!(kept_by_caller[0], 7);
}
