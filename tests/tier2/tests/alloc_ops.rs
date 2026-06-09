//! Lazy buffer-alloc ops + `.and_then_with_context`.
//!
//! Two ergonomic primitives added together:
//!
//! - `device_slice_alloc` / `mapped_slice_alloc`:
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

use claspr::{Buffer, Context, Error, MappedSlice, SvmLevel};
use claspr_async::{
    DeviceOperation, FillUninit, WriteUninit, bundle, device_slice_alloc_uninit,
    device_slice_alloc_zero, download, mapped_slice_alloc_uninit, mapped_slice_alloc_zero, upload,
    value,
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
    let result: Vec<u32> = device_slice_alloc_zero!(u32, N)
        .and_then(|buf| kernels.fill_u32([N], buf, 42))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("alloc + fill chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 42));
}

#[test]
fn mapped_slice_alloc_succeeds_or_surfaces_svm_not_available() {
    let Some(ctx) = ctx() else { return };
    let result = mapped_slice_alloc_zero!(u32, N).sync(&ctx);
    if ctx.svm_capability() == SvmLevel::None {
        // The synchronous MappedSlice::alloc gates on SVM; the
        // lazy op should surface the same Err at execute time.
        let err = result.expect_err("expected SvmNotAvailable");
        assert!(
            matches!(err, Error::SvmNotAvailable),
            "expected SvmNotAvailable on no-SVM device, got {err:?}",
        );
    } else {
        let buf: MappedSlice<u32> = result.expect("alloc");
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
        upload!(vec![1u32; N]),
        upload!(vec![2u32; N]),
        device_slice_alloc_zero!(u32, N),
    )
    .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
    .and_then(|(_a, _b, out)| download!(out))
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
    let result: Vec<u32> = upload!(vec![6u32; N])
        .and_then_with_context(|ec, buf| {
            // Use ec — proves it's accessible without a wrapping
            // with_context layer.
            let _dev = ec.device().clone();
            kernels.scale_u32([N], buf, 5)
        })
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("and_then_with_context chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 30));
}

// ── direct Uninit trait paths ──────────────────────────────────────
//
// The sugar macros (`device_slice_alloc_zero!`, `device_slice_filled!`,
// `upload!`) expand to alloc_uninit + FillUninit::fill / WriteUninit::write.
// These tests exercise the underlying trait verbs directly so the
// macros aren't the only coverage for the compositional path.

#[test]
fn alloc_uninit_then_fill_via_trait_verb() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = device_slice_alloc_uninit!(u32, N)
        .and_then(|u| u.fill(99u32))
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("alloc_uninit + fill chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 198));
}

#[test]
fn alloc_uninit_then_write_via_trait_verb() {
    let Some(ctx) = ctx() else { return };
    let result: Vec<u32> = device_slice_alloc_uninit!(u32, N)
        .and_then(|u| u.write((0u32..N as u32).collect::<Vec<_>>()))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("alloc_uninit + write chain");
    assert_eq!(result, (0u32..N as u32).collect::<Vec<_>>());
}

#[test]
fn mapped_alloc_uninit_then_fill_via_trait_verb() {
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: no SVM");
        return;
    }
    let buf: MappedSlice<u32> = mapped_slice_alloc_uninit!(u32, N)
        .and_then(|u| u.fill(7u32))
        .sync(&ctx)
        .expect("mapped alloc_uninit + fill");
    assert_eq!(buf.len(), N);
    let g = buf.map().wait().expect("map");
    assert!(g.iter().all(|&v| v == 7));
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
    let result: Vec<u32> = upload!(vec![7u32; N])
        .and_then_with_context(|_ec, buf| {
            // Build a sub-chain: pair the upstream buf with a fresh
            // alloc, then add them via a 3-arg kernel.
            bundle!(
                value(buf),
                upload!(vec![1u32; N]),
                device_slice_alloc_zero!(u32, N)
            )
            .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
        })
        .and_then(|(_a, _b, out)| download!(out))
        .sync(&ctx)
        .expect("alloc-via-and_then_with_context chain");
    assert!(result.iter().all(|&v| v == 8));
}
