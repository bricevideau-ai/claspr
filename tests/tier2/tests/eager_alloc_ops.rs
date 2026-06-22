//! Eager-API port of `alloc_ops.rs`. Same N, same values, same assertions,
//! rewritten against `claspr::eager`.
//!
//! Old → new mapping:
//!   `device_slice_alloc_zero!(u32, N)`  → `alloc_zero::<u32, ReadWrite>(N)`
//!   `download!(buf)`                    → `download`
//!   `bundle!(a, b, c)`                  → `bundle3(a, b, c)`
//!   `value(x)`                          → `value(x)`
//!   `.and_then_with_context(...)`       → `.and_then_with_context(...)`
//!
//! Uninit trait-verb shapes: the closure layer's `device_slice_alloc_uninit!` is
//! a lazy op producing a `DeviceSliceUninit`. The eager analog is the
//! `device_alloc_uninit` / `mapped_alloc_uninit` PRODUCING leaf — it allocates at
//! execute, so the uninit is GRAPH-PRODUCED and threaded into the fill/write verb
//! (`device_alloc_uninit(N).and_then(|u| fill_device_uninit(u, v))`), faithful to
//! the old lazy op. Same N, same 99/×2 = 198 and 0..N values.

use claspr::eager::{
    DeviceOpExt, alloc_zero, bundle3, device_alloc_uninit, download, fill_device_uninit,
    fill_mapped_uninit, lift, mapped_alloc_uninit, upload, write_device_uninit,
};
use claspr::{Buffer, Context, Error, MappedSlice, SvmLevel};
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

/// alloc_ops.rs::device_slice_alloc_produces_buffer_usable_in_kernel.
#[test]
fn device_slice_alloc_produces_buffer_usable_in_kernel() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = alloc_zero::<u32, claspr::ReadWrite>(N)
        .and_then(|buf| kernels.fill_u32([N], buf, 42))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc + fill chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 42));
}

/// alloc_ops.rs::mapped_slice_alloc_succeeds_or_surfaces_svm_not_available.
///
/// The eager API has no `mapped_slice_alloc_zero` op; `MappedSlice::alloc_zero`
/// (Tier 1) is the runtime primitive the lazy op wraps, so we call it directly
/// — same gate, same Err on a no-SVM device.
#[test]
fn mapped_slice_alloc_succeeds_or_surfaces_svm_not_available() {
    let Some(ctx) = ctx() else { return };
    let result = MappedSlice::<u32>::alloc_zero(&ctx, N);
    if ctx.svm_capability() == SvmLevel::None {
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

/// alloc_ops.rs::hoisted_bundle_uploads_and_alloc_feed_three_arg_kernel.
#[test]
fn hoisted_bundle_uploads_and_alloc_feed_three_arg_kernel() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = bundle3(
        upload::<u32, claspr::ReadWrite, _>(vec![1u32; N]),
        upload::<u32, claspr::ReadWrite, _>(vec![2u32; N]),
        alloc_zero::<u32, claspr::ReadWrite>(N),
    )
    .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
    .and_then(|(_a, _b, out)| download(out))
    .sync(&ctx)
    .expect("hoisted bundle chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 3));
}

/// alloc_ops.rs::and_then_with_context_closure_returns_kernel_op_directly.
#[test]
fn and_then_with_context_closure_returns_kernel_op_directly() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = upload::<u32, claspr::ReadWrite, _>(vec![6u32; N])
        .and_then_with_context(|ec, buf| {
            let _dev = ec.device().clone();
            kernels.scale_u32([N], buf, 5)
        })
        .and_then(download)
        .sync(&ctx)
        .expect("and_then_with_context chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 30));
}

// ── direct Uninit trait paths ──────────────────────────────────────

/// alloc_ops.rs::alloc_uninit_then_fill_via_trait_verb — uninit head + the
/// eager `fill_device_uninit` verb, then scale ×2 = 198.
#[test]
fn alloc_uninit_then_fill_via_trait_verb() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = device_alloc_uninit::<u32, claspr::ReadWrite>(N)
        .and_then(|u| fill_device_uninit(u, 99u32))
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc_uninit + fill chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 198));
}

/// alloc_ops.rs::alloc_uninit_then_write_via_trait_verb — uninit head + the
/// eager `write_device_uninit` verb writing 0..N.
#[test]
fn alloc_uninit_then_write_via_trait_verb() {
    let Some(ctx) = ctx() else { return };
    let result: Vec<u32> = device_alloc_uninit::<u32, claspr::ReadWrite>(N)
        .and_then(|u| write_device_uninit(u, (0u32..N as u32).collect::<Vec<_>>()))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc_uninit + write chain");
    assert_eq!(result, (0u32..N as u32).collect::<Vec<_>>());
}

/// alloc_ops.rs::mapped_alloc_uninit_then_fill_via_trait_verb — SVM analog via
/// the graph-produced `mapped_alloc_uninit` leaf + `fill_mapped_uninit`.
#[test]
fn mapped_alloc_uninit_then_fill_via_trait_verb() {
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: no SVM");
        return;
    }
    let buf: MappedSlice<u32> = mapped_alloc_uninit::<u32, claspr::ReadWrite>(N)
        .and_then(|u| fill_mapped_uninit(u, 7u32))
        .sync(&ctx)
        .expect("mapped alloc_uninit + fill");
    assert_eq!(buf.len(), N);
    let g = buf.map().wait().expect("map");
    assert!(g.iter().all(|&v| v == 7));
}

/// alloc_ops.rs::and_then_with_context_composes_lazy_alloc_inside_closure —
/// the mid-chain temp-buffer shape: grab `ec`, build a sub-chain bundling the
/// upstream buf with a fresh alloc, add via a 3-arg kernel. 7 + 1 = 8.
#[test]
fn and_then_with_context_composes_lazy_alloc_inside_closure() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = upload::<u32, claspr::ReadWrite, _>(vec![7u32; N])
        .and_then_with_context(|_ec, buf| {
            // `and_then_with_context`'s Handle is a single `Pipe<Output>`, not a
            // per-element tuple-of-pipes, so the multi-output `add_u32` must be
            // reduced to one pipe (download) INSIDE the sub-chain — the outer
            // chain can't destructure its tuple. Same 7 + 1 = 8 result.
            bundle3(
                lift(buf),
                upload::<u32, claspr::ReadWrite, _>(vec![1u32; N]),
                alloc_zero::<u32, claspr::ReadWrite>(N),
            )
            .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
            .and_then(|(_a, _b, out)| download(out))
        })
        .sync(&ctx)
        .expect("alloc-via-and_then_with_context chain");
    assert!(result.iter().all(|&v| v == 8));
}
