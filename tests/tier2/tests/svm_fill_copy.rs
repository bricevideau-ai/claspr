//! `MappedSlice::fill` / `MappedSlice::copy_to` + the Tier 2
//! `mapped_slice_filled` op — SVM analogs of `DeviceSlice::fill` /
//! `DeviceSlice::copy_to` / `device_slice_filled`.
//!
//! Skips on devices without SVM (most desktop GPUs have it; llvmpipe
//! reports SVM 2.0 fine, but the no-SVM error path is also worth
//! exercising).

use claspr::{Buffer, Context, MappedSlice, SvmLevel};
use claspr_async::{DeviceOperation, mapped_slice, mapped_slice_filled, mapped_slice_upload};
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

#[test]
fn tier1_svm_fill_writes_pattern() {
    // Alloc MappedSlice + fill + read back via map. Pattern lands
    // in every slot, error_count stays clean.
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    buf.fill(0xDEAD_BEEFu32).wait(&ctx).expect("fill");

    let g = buf.map().wait(&ctx).expect("map");
    assert!(g.iter().all(|&v| v == 0xDEAD_BEEF));
    drop(g);
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

#[test]
fn tier1_svm_copy_to_propagates_contents() {
    // Fill src, copy_to dst, read dst — dst must see src's data.
    // Also exercises auto-register on BOTH src and dst so Drop on
    // either side waits for the copy.
    let Some(ctx) = ctx_with_svm() else { return };
    let src = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc src");
    let dst = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc dst");

    src.fill(42u32).wait(&ctx).expect("fill src");
    src.copy_to(&dst).wait(&ctx).expect("copy src→dst");

    let g = dst.map().wait(&ctx).expect("map dst");
    assert!(g.iter().all(|&v| v == 42));
    drop(g);
    // Drop src first — last_use should include the copy event,
    // so the free queues after.
    drop(src);
    drop(dst);
    assert_eq!(ctx.error_count(), 0);
}

#[test]
fn tier1_svm_copy_length_mismatch_errors() {
    // src and dst must have the same length — surfaces our typed
    // LengthMismatch (checked before the unsafe enqueue).
    let Some(ctx) = ctx_with_svm() else { return };
    let src = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc src");
    let dst = MappedSlice::<u32>::alloc_zero(&ctx, N / 2).expect("alloc dst");

    let err = src.copy_to(&dst).wait(&ctx).expect_err("length mismatch");
    assert!(
        matches!(err, claspr::Error::LengthMismatch { src: 64, dst: 32 }),
        "got {err:?}",
    );
}

#[test]
fn tier1_svm_write_copies_host_data_into_buffer() {
    // Alloc + Tier 1 .write(host_data).wait — buffer ends up with
    // host_data's bytes, readable via map. Most basic shape; mirrors
    // tier1_svm_fill_writes_pattern but with a host source instead of
    // a single fill pattern.
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let host: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(11)).collect();
    buf.write(&host).wait(&ctx).expect("write");

    let g = buf.map().wait(&ctx).expect("map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, (i as u32).wrapping_mul(11));
    }
    drop(g);
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

#[test]
fn tier1_svm_write_length_mismatch_errors() {
    // data.len() != owner.len → typed LengthMismatch surfaces
    // before the unsafe enqueue. Same gate shape as svm_copy.
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let host = vec![0u32; N / 2];
    let err = buf.write(&host).wait(&ctx).expect_err("length mismatch");
    assert!(
        matches!(err, claspr::Error::LengthMismatch { src: 32, dst: 64 }),
        "got {err:?}",
    );
}

#[test]
fn tier1_svm_write_after_all_chains_after_fill() {
    // fill(99) → write(host_data) — the write's `.after_all(...)`
    // gates on the fill's event so the write lands AFTER the fill.
    // Final state: host_data, not 99s.
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let host: Vec<u32> = (0..N as u32).collect();

    let fill_evt = buf.fill(99u32).submit(&ctx).expect("fill submit");
    buf.write(&host)
        .after(&fill_evt)
        .wait(&ctx)
        .expect("write after fill");

    let g = buf.map().wait(&ctx).expect("map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, i as u32);
    }
    drop(g);
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

#[test]
fn tier2_mapped_slice_filled_threads_into_kernel() {
    // Lazy alloc+fill on SVM, then run a kernel that takes the
    // MappedSlice as a slice arg (via the KernelSliceArg<T>
    // widening), then read back via map.
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // mapped_slice_filled produces MappedSlice<u32>; scale by 2.
    let buf = mapped_slice_filled!(5u32, N)
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .sync(&ctx)
        .expect("filled svm chain");
    assert_eq!(buf.len(), N);
    let g = buf.map().wait(&ctx).expect("map");
    assert!(g.iter().all(|&v| v == 10));
}

#[test]
fn tier2_mapped_slice_upload_threads_into_kernel() {
    // alloc SVM + clEnqueueSVMMemcpy from a host literal, then scale
    // by 3 in a kernel. The host source is kept alive by the drop
    // callback registered on the memcpy event; the buffer's last_use
    // includes the memcpy event so Drop's SVMFree waits.
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = mapped_slice_upload!(vec![1u32, 2, 3, 4, 5, 6, 7, 8])
        .and_then(|buf| kernels.scale_u32([8], buf, 3))
        .sync(&ctx)
        .expect("upload + scale");
    assert_eq!(buf.len(), 8);
    let g = buf.map().wait(&ctx).expect("map");
    assert_eq!(&g[..], &[3u32, 6, 9, 12, 15, 18, 21, 24]);
}

#[test]
fn macro_mapped_slice_repeat_arm() {
    // `mapped_slice![v; N]` → mapped_slice_filled!(v, N).
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = mapped_slice![4u32; N]
        .and_then(|buf| kernels.scale_u32([N], buf, 5))
        .sync(&ctx)
        .expect("macro repeat");
    let g = buf.map().wait(&ctx).expect("map");
    assert!(g.iter().all(|&v| v == 20));
}

#[test]
fn macro_mapped_slice_literal_arm() {
    // `mapped_slice![a, b, c]` → mapped_slice_upload!(vec![a, b, c]).
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = mapped_slice![10u32, 20, 30, 40]
        .and_then(|buf| kernels.scale_u32([4], buf, 2))
        .sync(&ctx)
        .expect("macro literal");
    let g = buf.map().wait(&ctx).expect("map");
    assert_eq!(&g[..], &[20u32, 40, 60, 80]);
}

#[test]
fn tier2_mapped_slice_filled_surfaces_svm_not_available() {
    // Run without SVM: should surface SvmNotAvailable at execute.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    if ctx.svm_capability() != SvmLevel::None {
        eprintln!("SKIP: device supports SVM, can't test no-SVM path here");
        return;
    }
    let err = mapped_slice_filled!(0u32, N)
        .sync(&ctx)
        .expect_err("expected SvmNotAvailable");
    assert!(matches!(err, claspr::Error::SvmNotAvailable), "got {err:?}",);
}
