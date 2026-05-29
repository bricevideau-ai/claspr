//! `SharedBuffer::fill` / `SharedBuffer::copy_to` + the Tier 2
//! `shared_buffer_filled` op — SVM analogs of `DeviceSlice::fill` /
//! `DeviceSlice::copy_to` / `device_slice_filled`.
//!
//! Skips on devices without SVM (most desktop GPUs have it; llvmpipe
//! reports SVM 2.0 fine, but the no-SVM error path is also worth
//! exercising).

use claspr::{Buffer, Context, SharedBuffer, SvmLevel};
use claspr_async::{DeviceOperation, shared_buffer, shared_buffer_filled, shared_buffer_upload};
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
    // Alloc SharedBuffer + fill + read back via map. Pattern lands
    // in every slot, error_count stays clean.
    let Some(ctx) = ctx_with_svm() else { return };
    let buf = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc");
    buf.fill(&ctx, 0xDEAD_BEEFu32).wait().expect("fill");

    let g = buf.map(&ctx).expect("map");
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
    let src = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc src");
    let dst = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc dst");

    src.fill(&ctx, 42u32).wait().expect("fill src");
    src.copy_to(&dst, &ctx).wait().expect("copy src→dst");

    let g = dst.map(&ctx).expect("map dst");
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
    let src = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc src");
    let dst = SharedBuffer::<u32>::alloc(&ctx, N / 2).expect("alloc dst");

    let err = src.copy_to(&dst, &ctx).wait().expect_err("length mismatch");
    assert!(
        matches!(err, claspr::Error::LengthMismatch { src: 64, dst: 32 }),
        "got {err:?}",
    );
}

#[test]
fn tier2_shared_buffer_filled_threads_into_kernel() {
    // Lazy alloc+fill on SVM, then run a kernel that takes the
    // SharedBuffer as a slice arg (via the KernelSliceArg<T>
    // widening), then read back via map.
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // shared_buffer_filled produces SharedBuffer<u32>; scale by 2.
    let buf = shared_buffer_filled(5u32, N)
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .sync(&ctx)
        .expect("filled svm chain");
    assert_eq!(buf.len(), N);
    let g = buf.map(&ctx).expect("map");
    assert!(g.iter().all(|&v| v == 10));
}

#[test]
fn tier2_shared_buffer_upload_threads_into_kernel() {
    // alloc SVM + clEnqueueSVMMemcpy from a host literal, then scale
    // by 3 in a kernel. The host source is kept alive by the drop
    // callback registered on the memcpy event; the buffer's last_use
    // includes the memcpy event so Drop's SVMFree waits.
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = shared_buffer_upload::<u32, _>(vec![1u32, 2, 3, 4, 5, 6, 7, 8])
        .and_then(|buf| kernels.scale_u32([8], buf, 3))
        .sync(&ctx)
        .expect("upload + scale");
    assert_eq!(buf.len(), 8);
    let g = buf.map(&ctx).expect("map");
    assert_eq!(&g[..], &[3u32, 6, 9, 12, 15, 18, 21, 24]);
}

#[test]
fn macro_shared_buffer_repeat_arm() {
    // `shared_buffer![v; N]` → shared_buffer_filled(v, N).
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = shared_buffer![4u32; N]
        .and_then(|buf| kernels.scale_u32([N], buf, 5))
        .sync(&ctx)
        .expect("macro repeat");
    let g = buf.map(&ctx).expect("map");
    assert!(g.iter().all(|&v| v == 20));
}

#[test]
fn macro_shared_buffer_literal_arm() {
    // `shared_buffer![a, b, c]` → shared_buffer_upload(vec![a, b, c]).
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = shared_buffer![10u32, 20, 30, 40]
        .and_then(|buf| kernels.scale_u32([4], buf, 2))
        .sync(&ctx)
        .expect("macro literal");
    let g = buf.map(&ctx).expect("map");
    assert_eq!(&g[..], &[20u32, 40, 60, 80]);
}

#[test]
fn tier2_shared_buffer_filled_surfaces_svm_not_available() {
    // Run without SVM: should surface SvmNotAvailable at execute.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    if ctx.svm_capability() != SvmLevel::None {
        eprintln!("SKIP: device supports SVM, can't test no-SVM path here");
        return;
    }
    let err = shared_buffer_filled(0u32, N)
        .sync(&ctx)
        .expect_err("expected SvmNotAvailable");
    assert!(matches!(err, claspr::Error::SvmNotAvailable), "got {err:?}",);
}
