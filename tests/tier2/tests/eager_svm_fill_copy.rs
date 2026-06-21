//! Eager port of `svm_fill_copy.rs`: `MappedSlice::fill` / `copy_to` (Tier 1)
//! plus the Tier-2 mapped-slice filled/upload chains, expressed through the
//! eager graph API where one exists.
//!
//! The Tier-1 tests (fill, copy_to, copy-length-mismatch, write,
//! write-length-mismatch, write-after-fill) have no Tier-2 combinator chain and
//! are reproduced verbatim — same suite coverage.
//!
//! Tier-2 mapped chains — old → new mapping:
//!   `mapped_slice_filled!(v, N)` → `fill_mapped_uninit(MappedSlice::alloc_uninit(&ctx, N)?, v)`
//!         The eager API has no `mapped_slice_alloc_zero` / filled PRODUCING leaf
//!         over a Context; the uninit head is built synchronously (Tier 1
//!         `MappedSlice::alloc_uninit`) and threaded into the eager
//!         `fill_mapped_uninit` leaf — the exact pattern eager_alloc_ops.rs uses
//!         for `mapped_alloc_uninit_then_fill_via_trait_verb`.
//!   `mapped_slice_upload!(v)`    → concrete `MappedSlice::from_slice(&ctx, &v)`
//!         fed into the eager kernel op (the eager API has no mapped-upload
//!         PRODUCING leaf; eager_buffer_ops.rs threads MappedSlice concretely the
//!         same way). Same values, same assertions.
//!   `mapped_slice![v; N]`       → repeat arm → fill_mapped_uninit path.
//!   `mapped_slice![a, b, c]`    → literal arm → from_slice path.
//!
//! DEVIATION (no-SVM path): `tier2_mapped_slice_filled_surfaces_svm_not_available`
//! relied on the lazy Tier-2 op deferring the SVM-availability check to execute
//! time. The eager mapped path has no Context-bound producing leaf — the uninit
//! head is the Tier-1 `MappedSlice::alloc_uninit`, which itself surfaces
//! `Error::SvmNotAvailable` on a no-SVM device (see its rustdoc). We assert that
//! same error at the same boundary.
//!
//! Skips on devices without SVM. Guard preserved verbatim.

use claspr::eager::{EagerOpExt, fill_mapped_uninit};
use claspr::{Buffer, Context, MappedSlice, SvmLevel};
use claspr_test_kernels::kernels;

const N: usize = 64;

fn ctx_with_svm() -> Option<Context> {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return None;
    };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: device has no SVM");
        return None;
    }
    Some(ctx)
}

/// svm_fill_copy.rs::tier1_svm_fill_writes_pattern — Tier-1, reproduced verbatim.
#[test]
fn tier1_svm_fill_writes_pattern() {
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    buf.fill(0xDEAD_BEEFu32).wait().expect("fill");

    let g = buf.map().wait().expect("map");
    assert!(g.iter().all(|&v| v == 0xDEAD_BEEF));
    drop(g);
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

/// svm_fill_copy.rs::tier1_svm_copy_to_propagates_contents — Tier-1, verbatim.
#[test]
fn tier1_svm_copy_to_propagates_contents() {
    let Some(ctx) = ctx_with_svm() else { return };
    let src = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc src");
    let dst = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc dst");

    src.fill(42u32).wait().expect("fill src");
    src.copy_to(&dst).wait().expect("copy src→dst");

    let g = dst.map().wait().expect("map dst");
    assert!(g.iter().all(|&v| v == 42));
    drop(g);
    drop(src);
    drop(dst);
    assert_eq!(ctx.error_count(), 0);
}

/// svm_fill_copy.rs::tier1_svm_copy_length_mismatch_errors — Tier-1, verbatim.
#[test]
fn tier1_svm_copy_length_mismatch_errors() {
    let Some(ctx) = ctx_with_svm() else { return };
    let src = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc src");
    let dst = MappedSlice::<u32>::alloc_zero(&ctx, N / 2).expect("alloc dst");

    let err = src.copy_to(&dst).wait().expect_err("length mismatch");
    assert!(
        matches!(err, claspr::Error::LengthMismatch { src: 64, dst: 32 }),
        "got {err:?}",
    );
}

/// svm_fill_copy.rs::tier1_svm_write_copies_host_data_into_buffer — Tier-1, verbatim.
#[test]
fn tier1_svm_write_copies_host_data_into_buffer() {
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let host: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(11)).collect();
    buf.write(&host).wait().expect("write");

    let g = buf.map().wait().expect("map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, (i as u32).wrapping_mul(11));
    }
    drop(g);
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

/// svm_fill_copy.rs::tier1_svm_write_length_mismatch_errors — Tier-1, verbatim.
#[test]
fn tier1_svm_write_length_mismatch_errors() {
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let host = vec![0u32; N / 2];
    let err = buf.write(&host).wait().expect_err("length mismatch");
    assert!(
        matches!(err, claspr::Error::LengthMismatch { src: 32, dst: 64 }),
        "got {err:?}",
    );
}

/// svm_fill_copy.rs::tier1_svm_write_after_all_chains_after_fill — Tier-1, verbatim.
#[test]
fn tier1_svm_write_after_all_chains_after_fill() {
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let host: Vec<u32> = (0..N as u32).collect();

    let fill_evt = buf.fill(99u32).submit().expect("fill submit");
    buf.write(&host)
        .after(&fill_evt)
        .wait()
        .expect("write after fill");

    let g = buf.map().wait().expect("map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, i as u32);
    }
    drop(g);
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

/// svm_fill_copy.rs::tier2_mapped_slice_filled_threads_into_kernel — eager
/// `fill_mapped_uninit` over a concrete uninit head → scale 2 → 10.
#[test]
fn tier2_mapped_slice_filled_threads_into_kernel() {
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let uninit = MappedSlice::<u32>::alloc_uninit(&ctx, N).expect("mapped alloc_uninit");
    let buf = fill_mapped_uninit(uninit, 5u32)
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .sync(&ctx)
        .expect("filled svm chain");
    assert_eq!(buf.len(), N);
    let g = buf.map().wait().expect("map");
    assert!(g.iter().all(|&v| v == 10));
}

/// svm_fill_copy.rs::tier2_mapped_slice_upload_threads_into_kernel — concrete
/// `MappedSlice::from_slice` → eager kernel scale 3.
#[test]
fn tier2_mapped_slice_upload_threads_into_kernel() {
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let src: MappedSlice<u32> =
        MappedSlice::from_slice(&ctx, &[1u32, 2, 3, 4, 5, 6, 7, 8]).expect("svm upload");
    let buf = kernels
        .scale_u32([8], src, 3)
        .sync(&ctx)
        .expect("upload + scale");
    assert_eq!(buf.len(), 8);
    let g = buf.map().wait().expect("map");
    assert_eq!(&g[..], &[3u32, 6, 9, 12, 15, 18, 21, 24]);
}

/// svm_fill_copy.rs::macro_mapped_slice_repeat_arm — `mapped_slice![v; N]` →
/// repeat arm → `mapped_slice_filled!` → eager fill_mapped_uninit path.
#[test]
fn macro_mapped_slice_repeat_arm() {
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let uninit = MappedSlice::<u32>::alloc_uninit(&ctx, N).expect("mapped alloc_uninit");
    let buf = fill_mapped_uninit(uninit, 4u32)
        .and_then(|buf| kernels.scale_u32([N], buf, 5))
        .sync(&ctx)
        .expect("macro repeat");
    let g = buf.map().wait().expect("map");
    assert!(g.iter().all(|&v| v == 20));
}

/// svm_fill_copy.rs::macro_mapped_slice_literal_arm — `mapped_slice![a, b, c]` →
/// literal arm → `mapped_slice_upload!` → concrete from_slice path.
#[test]
fn macro_mapped_slice_literal_arm() {
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let src: MappedSlice<u32> =
        MappedSlice::from_slice(&ctx, &[10u32, 20, 30, 40]).expect("svm upload");
    let buf = kernels
        .scale_u32([4], src, 2)
        .sync(&ctx)
        .expect("macro literal");
    let g = buf.map().wait().expect("map");
    assert_eq!(&g[..], &[20u32, 40, 60, 80]);
}

/// svm_fill_copy.rs::tier2_mapped_slice_filled_surfaces_svm_not_available —
/// DEVIATION (see module doc): the eager mapped path has no Context-bound lazy
/// producing leaf, so the SVM-availability check fires at the Tier-1 uninit
/// alloc (`MappedSlice::alloc_uninit`), which surfaces the same
/// `Error::SvmNotAvailable` on a no-SVM device.
#[test]
fn tier2_mapped_slice_filled_surfaces_svm_not_available() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    if ctx.svm_capability() != SvmLevel::None {
        eprintln!("SKIP: device supports SVM, can't test no-SVM path here");
        return;
    }
    let err = MappedSlice::<u32>::alloc_uninit(&ctx, N).expect_err("expected SvmNotAvailable");
    assert!(matches!(err, claspr::Error::SvmNotAvailable), "got {err:?}",);
}
