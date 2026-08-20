//! Expansion + execution regression for the chain-entry macro family.
//!
//! Nothing in-tree invoked these macros, so their `$crate::…`
//! expansion paths silently rotted when the crate root was reshaped —
//! every advertised `device_slice_alloc_zero!`-style entry point was
//! broken at user expansion until 2026-08. This file invokes every
//! exported chain-entry macro (both arms where practical) so a path
//! break is a compile error in CI, and runs each through a terminal so
//! the expansion is semantically right, not just name-resolvable.

use claspr::eager::{DeviceOpExt, download, fill_device_uninit};
use claspr::{
    device_scalar_alloc, device_scalar_zero, device_slice, device_slice_alloc_uninit,
    device_slice_alloc_zero, device_slice_filled, device_slice_from_slice, mapped_slice,
    mapped_slice_alloc_uninit, mapped_slice_alloc_zero, mapped_slice_filled,
    mapped_slice_from_slice, mapped_slice_upload, usm_slice, usm_slice_alloc_uninit,
    usm_slice_alloc_zero,
};
use claspr_test_support::{ctx, ctx_with_svm};

const N: usize = 16;

#[test]
fn device_tier_macros_expand_and_run() {
    let Some(ctx) = ctx() else { return };

    let zeros: Vec<u32> = device_slice_alloc_zero!(u32, N)
        .and_then(download)
        .sync(&ctx)
        .expect("alloc_zero chain")
        .into_inner();
    assert_eq!(zeros, vec![0u32; N]);

    let filled: Vec<u32> = device_slice_filled!(7u32, N)
        .and_then(download)
        .sync(&ctx)
        .expect("filled chain")
        .into_inner();
    assert_eq!(filled, vec![7u32; N]);

    let uninit_then_fill: Vec<u32> = device_slice_alloc_uninit!(u32, N)
        .and_then(|u| fill_device_uninit(u, 3u32))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc_uninit chain")
        .into_inner();
    assert_eq!(uninit_then_fill, vec![3u32; N]);

    let from_slice: Vec<u32> = device_slice_from_slice!(vec![1u32, 2, 3, 4])
        .and_then(download)
        .sync(&ctx)
        .expect("from_slice chain")
        .into_inner();
    assert_eq!(from_slice, vec![1, 2, 3, 4]);

    let repeat: Vec<u32> = device_slice![9u32; N]
        .and_then(download)
        .sync(&ctx)
        .expect("device_slice![v; n]")
        .into_inner();
    assert_eq!(repeat, vec![9u32; N]);

    let literal: Vec<u32> = device_slice![5u32, 6, 7]
        .and_then(download)
        .sync(&ctx)
        .expect("device_slice![a, b, c]")
        .into_inner();
    assert_eq!(literal, vec![5, 6, 7]);

    // Scalar twins — creation + terminal proves the expansion path;
    // scalar readback semantics are covered by the cg example/tests.
    let _ = device_scalar_alloc!(2.5f32)
        .sync(&ctx)
        .expect("scalar alloc");
    let _ = device_scalar_zero!(u32).sync(&ctx).expect("scalar zero");
}

#[test]
fn svm_tier_macros_expand_and_run() {
    let Some(ctx) = ctx_with_svm() else { return };

    let _ = mapped_slice_alloc_zero!(u32, N)
        .sync(&ctx)
        .expect("mapped alloc_zero");
    let _ = mapped_slice_alloc_uninit!(u32, N)
        .and_then(|u| claspr::eager::fill_mapped_uninit(u, 1u32))
        .sync(&ctx)
        .expect("mapped alloc_uninit + fill");
    let _ = mapped_slice_filled!(4u32, N)
        .sync(&ctx)
        .expect("mapped filled");
    let _ = mapped_slice_from_slice!(vec![1u32, 2])
        .sync(&ctx)
        .expect("mapped from_slice");
    let _ = mapped_slice_upload!(vec![3u32, 4])
        .sync(&ctx)
        .expect("mapped upload");
    let _ = mapped_slice![8u32; N]
        .sync(&ctx)
        .expect("mapped_slice![v; n]");
    let _ = mapped_slice![1u32, 2, 3]
        .sync(&ctx)
        .expect("mapped_slice![a, b, c]");
}

#[test]
fn usm_tier_macros_expand_and_run() {
    // usm_slice! requires fine-grain-system SVM (wraps a host Vec).
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() != claspr::SvmLevel::FineSystem {
        eprintln!("SKIP: no fine-grain-system SVM");
        return;
    }

    let _ = usm_slice!(vec![1u32, 2, 3])
        .sync(&ctx)
        .expect("usm_slice!(vec)");
    let _ = usm_slice![0u32; N].sync(&ctx).expect("usm_slice![v; n]");
    let _ = usm_slice![4u32, 5].sync(&ctx).expect("usm_slice![a, b, c]");
    let _ = usm_slice_alloc_uninit!(u32, N)
        .and_then(|u| claspr::eager::fill_usm_uninit(u, 2u32))
        .sync(&ctx)
        .expect("usm alloc_uninit + fill");
    let _ = usm_slice_alloc_zero!(u32, N)
        .sync(&ctx)
        .expect("usm alloc_zero");
}
