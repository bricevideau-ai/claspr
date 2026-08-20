//! Eager port of `svm_fill_copy.rs`: `MappedSlice::fill` / `copy_to` (Tier 1)
//! plus the Tier-2 mapped-slice filled/upload chains, expressed through the
//! eager graph API where one exists.
//!
//! The Tier-1 tests (fill, copy_to, copy-length-mismatch, write,
//! write-length-mismatch, write-after-fill) have no Tier-2 combinator chain and
//! are reproduced verbatim — same suite coverage.
//!
//! Tier-2 mapped chains — old → new mapping:
//!   `mapped_slice_filled!(v, N)` → `mapped_alloc_uninit(N).and_then(|u| fill_mapped_uninit(u, v))`
//!         The eager `mapped_alloc_uninit` PRODUCING leaf allocates the uninit
//!         `MappedSlice` at execute (graph-produced, like the old
//!         `MappedSliceAllocUninit`), so the no-SVM check defers to the terminal.
//!   `mapped_slice_upload!(v)`    → concrete `MappedSlice::from_slice(&ctx, &v)`
//!         fed into the eager kernel op (the eager API has no mapped-upload
//!         PRODUCING leaf; eager_buffer_ops.rs threads MappedSlice concretely the
//!         same way). Same values, same assertions.
//!   `mapped_slice![v; N]`       → repeat arm → fill_mapped_uninit path.
//!   `mapped_slice![a, b, c]`    → literal arm → from_slice path.
//!
//! Skips on devices without SVM. Guard preserved verbatim.

use claspr::eager::{DeviceOpExt, fill_mapped_uninit, mapped_alloc_uninit};
use claspr::{Buffer, MappedSlice, SvmLevel};
use claspr_test_kernels::kernels;
use claspr_test_support::{ctx, ctx_with_svm};

const N: usize = 64;

/// svm_fill_copy.rs::tier1_svm_fill_writes_pattern — eager move-out form: `fill`
/// consumes the buffer and rebinds it out.
#[test]
fn tier1_svm_fill_writes_pattern() {
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let buf = buf.fill(0xDEAD_BEEFu32).wait().expect("fill");

    let g = buf.map().wait().expect("map");
    assert!(g.iter().all(|&v| v == 0xDEAD_BEEF));
    drop(g);
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

/// svm_fill_copy.rs::tier1_svm_copy_to_propagates_contents — eager move-out:
/// `copy_to` consumes both and yields `(src, dst)`; the named `sync` terminal.
#[test]
fn tier1_svm_copy_to_propagates_contents() {
    let Some(ctx) = ctx_with_svm() else { return };
    let src = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc src");
    let dst = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc dst");

    let src = src.fill(42u32).wait().expect("fill src");
    let (src, dst) = src.copy_to(dst).sync(&ctx).expect("copy src→dst");

    let g = dst.map().wait().expect("map dst");
    assert!(g.iter().all(|&v| v == 42));
    drop(g);
    drop(src);
    drop(dst);
    assert_eq!(ctx.error_count(), 0);
}

/// svm_fill_copy.rs::tier1_svm_copy_length_mismatch_errors — eager move-out form.
#[test]
fn tier1_svm_copy_length_mismatch_errors() {
    let Some(ctx) = ctx_with_svm() else { return };
    let src = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc src");
    let dst = MappedSlice::<u32>::alloc_zero(&ctx, N / 2).expect("alloc dst");

    let err = src.copy_to(dst).sync(&ctx).expect_err("length mismatch");
    assert!(
        matches!(err, claspr::Error::LengthMismatch { src: 64, dst: 32 }),
        "got {err:?}",
    );
}

/// svm_fill_copy.rs::tier1_svm_write_copies_host_data_into_buffer — eager
/// move-out: `write` takes the host data by value and rebinds the buffer out.
#[test]
fn tier1_svm_write_copies_host_data_into_buffer() {
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let host: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(11)).collect();
    let buf = buf.write(host).wait().expect("write");

    let g = buf.map().wait().expect("map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, (i as u32).wrapping_mul(11));
    }
    drop(g);
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

/// svm_fill_copy.rs::tier1_svm_write_length_mismatch_errors — eager move-out form.
#[test]
fn tier1_svm_write_length_mismatch_errors() {
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let host = vec![0u32; N / 2];
    let err = buf.write(host).wait().expect_err("length mismatch");
    assert!(
        matches!(err, claspr::Error::LengthMismatch { src: 32, dst: 64 }),
        "got {err:?}",
    );
}

/// svm_fill_copy.rs::tier1_svm_write_after_all_chains_after_fill — eager move-out:
/// sequential fill-then-write rebinds the buffer; the write overwrites the fill,
/// same ordering semantics as the old `.after(&fill_evt)` (the `wait()` on `fill`
/// completes it before `write` enqueues).
#[test]
fn tier1_svm_write_after_all_chains_after_fill() {
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let host: Vec<u32> = (0..N as u32).collect();

    let buf = buf.fill(99u32).wait().expect("fill");
    let buf = buf.write(host).wait().expect("write after fill");

    let g = buf.map().wait().expect("map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, i as u32);
    }
    drop(g);
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

/// svm_fill_copy.rs::tier2_mapped_slice_filled_threads_into_kernel — graph-produced
/// `mapped_alloc_uninit` → `fill_mapped_uninit` → scale 2 → 10.
#[test]
fn tier2_mapped_slice_filled_threads_into_kernel() {
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = mapped_alloc_uninit::<u32>(N)
        .and_then(|u| fill_mapped_uninit(u, 5u32))
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

    let buf = mapped_alloc_uninit::<u32>(N)
        .and_then(|u| fill_mapped_uninit(u, 4u32))
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

/// svm_fill_copy.rs::tier2_mapped_slice_filled_surfaces_svm_not_available — on a
/// no-SVM device the `mapped_alloc_uninit` PRODUCING leaf defers its allocation
/// to execute, so `SvmNotAvailable` surfaces AT THE TERMINAL (`.sync()`), not
/// eagerly — faithful to the old lazy `mapped_slice_filled!` op's "at execute".
#[test]
fn tier2_mapped_slice_filled_surfaces_svm_not_available() {
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() != SvmLevel::None {
        eprintln!("SKIP: device supports SVM, can't test no-SVM path here");
        return;
    }
    let err = mapped_alloc_uninit::<u32>(N)
        .and_then(|u| fill_mapped_uninit(u, 0u32))
        .sync(&ctx)
        .expect_err("expected SvmNotAvailable");
    assert!(matches!(err, claspr::Error::SvmNotAvailable), "got {err:?}",);
}
