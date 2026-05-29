//! Lazy buffer-alloc ops + `.and_then_with_context`.
//!
//! Two ergonomic primitives added together:
//!
//! - `device_slice_alloc` / `host_buffer_alloc` / `shared_buffer_alloc`:
//!   lazy `DeviceOperation` versions of the synchronous constructors,
//!   making bundle-hoisted-alloc and `and_then_with_context`-of-an-op
//!   chains express cleanly.
//! - `.and_then_with_context`: chain method that hands the closure
//!   both `ec` and the previous output, returns an op directly (no
//!   `Result<value>` / `Ok(...)` wrapping needed when paired with the
//!   new alloc combinators).
//!
//! These tests prove both the per-op alloc surface and the composed
//! shape (hoisted bundle + chained `and_then_with_context`) work
//! end-to-end on a real device.

use claspr::{Buffer, Context, Error, HostBuffer, SharedBuffer, SvmLevel};
use claspr_async::{
    DeviceOperation, bundle, device_slice_alloc, download, host_buffer_alloc, shared_buffer_alloc,
    upload, value,
};
use claspr_test_kernels::kernels;

const N: usize = 64;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

// ── alloc-op shape tests ───────────────────────────────────────────

#[test]
fn device_slice_alloc_produces_buffer_usable_in_kernel() {
    // Alloc lazily, fill on the same chain, download — proves the
    // resulting DeviceSlice<T> is a valid, kernel-launchable buffer.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = device_slice_alloc::<u32>(N)
        .and_then(|buf| kernels.fill_u32([N], buf, 42))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc + fill chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 42));
}

#[test]
fn host_buffer_alloc_produces_buffer_of_requested_len() {
    // HostBuffer's contents are uninit at alloc time — don't read
    // them. Just confirm the alloc op produces a buffer of the
    // requested length, materialised through the chain terminal.
    let Some(ctx) = ctx() else { return };
    let buf: HostBuffer<u32> = host_buffer_alloc::<u32>(N).sync(&ctx).expect("alloc");
    assert_eq!(buf.len(), N);
    // Drop here exercises the HostBuffer unmap-on-Drop path, which
    // is the failure mode most likely to regress if the alloc op
    // didn't go through `HostBuffer::alloc` cleanly.
}

#[test]
fn shared_buffer_alloc_succeeds_or_surfaces_svm_not_available() {
    let Some(ctx) = ctx() else { return };
    let result = shared_buffer_alloc::<u32>(N).sync(&ctx);
    if ctx.svm_capability() == SvmLevel::None {
        // The synchronous SharedBuffer::alloc gates on SVM; the
        // lazy op should surface the same Err at execute time.
        // (SharedBuffer doesn't impl Debug so we can't print it on
        // failure — match on the Err arm directly.)
        let err = result.err().expect("expected SvmNotAvailable");
        assert!(
            matches!(err, Error::SvmNotAvailable),
            "expected SvmNotAvailable on no-SVM device, got {err:?}",
        );
    } else {
        let buf: SharedBuffer<u32> = result.expect("alloc");
        assert_eq!(buf.len(), N);
    }
}

// ── composed shapes ─────────────────────────────────────────────────

#[test]
fn hoisted_bundle_uploads_and_alloc_feed_three_arg_kernel() {
    // cuda-oxide's preferred idiom: hoist every input/temp to a
    // top-level bundle, then thread the tuple into a multi-arg
    // kernel. Proves device_slice_alloc composes inside bundle! with
    // upload, and the typed launcher's tuple Output destructures
    // cleanly into the next .and_then.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = bundle!(
        upload(vec![1u32; N]),
        upload(vec![2u32; N]),
        device_slice_alloc::<u32>(N),
    )
    .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
    .and_then(|(_a, _b, out)| download(out))
    .sync(&ctx)
    .expect("hoisted bundle chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 3));
}

#[test]
fn and_then_with_context_closure_returns_kernel_op_directly() {
    // The whole point of the new combinator: closure receives `ec` +
    // the previous output, returns an op DIRECTLY (no Result, no
    // Ok wrap). Reading `ec.device()` inside the closure forces ec
    // to be used (proving it's actually in scope, not just an unused
    // parameter); the kernel call returns the next chain op.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = upload(vec![6u32; N])
        .and_then_with_context(|ec, buf| {
            // Use ec — proves it's accessible without a wrapping
            // with_context layer.
            let _dev = ec.device().clone();
            kernels.scale_u32([N], buf, 5)
        })
        .and_then(download)
        .sync(&ctx)
        .expect("and_then_with_context chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 30));
}

#[test]
fn and_then_with_context_composes_lazy_alloc_inside_closure() {
    // The user's original "mid-chain temp buffer" pseudo-code shape:
    // grab `ec`, alloc a temp via a lazy op inside the closure, run
    // a kernel that uses both the upstream buf and the temp. No
    // synchronous fallibility needed — the alloc lazy-fails at
    // execute time if it ever does.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = upload(vec![7u32; N])
        .and_then_with_context(|_ec, buf| {
            // Build a sub-chain: pair the upstream buf with a fresh
            // alloc, then add them via a 3-arg kernel.
            bundle!(
                value(buf),
                upload(vec![1u32; N]),
                device_slice_alloc::<u32>(N)
            )
            .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
        })
        .and_then(|(_a, _b, out)| download(out))
        .sync(&ctx)
        .expect("alloc-via-and_then_with_context chain");
    assert!(result.iter().all(|&v| v == 8));
}
