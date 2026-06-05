//! Tier 2 in-place buffer ops — `device_slice_fill` /
//! `device_slice_copy` / `device_slice_write` and the SVM analogues
//! `mapped_slice_fill` / `mapped_slice_copy`.
//!
//! Each test exercises one of the ops in a realistic chain shape:
//! upstream alloc / upload, the in-place op, then a downstream read
//! that asserts the op's effect propagated correctly. Marker-bound
//! enforcement is checked separately by the `compile_fail/buffer_ops_*`
//! fixtures via `safety_compile_fail.rs`.

use claspr::{Buffer, Context, Error, MappedSlice, SvmLevel};
use claspr_async::{
    CopyTo, DeviceOperation, bundle, device_slice_alloc_zero, device_slice_fill,
    device_slice_write, download, mapped_slice_alloc_zero, mapped_slice_fill, mapped_slice_upload,
    upload,
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

fn ctx_with_svm() -> Option<Context> {
    let c = ctx()?;
    if c.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: device has no SVM");
        return None;
    }
    Some(c)
}

// ── device_slice_fill ──────────────────────────────────────────────

#[test]
fn device_slice_fill_in_place() {
    // Alloc zeros, fill with 7, download — expect all 7s. Most basic
    // shape proving the op works at all.
    let Some(ctx) = ctx() else { return };
    let result: Vec<u32> = device_slice_alloc_zero!(u32, N)
        .and_then(|buf| device_slice_fill(buf, 7u32))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("alloc + fill + download chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 7));
}

#[test]
fn device_slice_fill_chains_after_upload() {
    // upload!(values) → fill(99) — fill must wait for upload's event
    // before clEnqueueFillBuffer lands, otherwise the upload could
    // arrive AFTER and overwrite. Result must be all 99s.
    let Some(ctx) = ctx() else { return };
    let input: Vec<u32> = (0..N as u32).collect();
    let result: Vec<u32> = upload!(input)
        .and_then(|buf| device_slice_fill(buf, 99u32))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("upload + fill chain");
    assert!(result.iter().all(|&v| v == 99));
}

#[test]
fn device_slice_fill_event_threads_to_kernel() {
    // fill(7) → kernel(scale_u32 ×2). Without the fill event being
    // wired into the kernel launch's wait-list, the kernel could run
    // before the fill, producing 0*2 = 0 instead of 7*2 = 14.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = device_slice_alloc_zero!(u32, N)
        .and_then(|buf| device_slice_fill(buf, 7u32))
        .and_then(|buf| kernels.scale_u32([N], buf, 2u32))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("fill → kernel → download chain");
    assert!(result.iter().all(|&v| v == 14));
}

// ── device_slice_copy ──────────────────────────────────────────────

#[test]
fn device_slice_copy_propagates_src_to_dst() {
    // upload src + alloc(dst) → copy → download!(dst) — dst sees src.
    let Some(ctx) = ctx() else { return };
    let src_data: Vec<u32> = (0..N as u32).map(|i| i * 3).collect();
    let result: Vec<u32> = bundle!(upload!(src_data.clone()), device_slice_alloc_zero!(u32, N))
        .and_then(|(src, dst)| src.copy_to(dst))
        .and_then(|(_src, dst)| download!(dst))
        .sync(&ctx)
        .expect("upload + alloc + copy + download");
    assert_eq!(result, src_data);
}

#[test]
fn device_slice_copy_returns_both_buffers() {
    // After copy, src is still readable (op output is `(src, dst)`).
    // Download src too and assert it matches what we uploaded.
    let Some(ctx) = ctx() else { return };
    let src_data: Vec<u32> = (10..10 + N as u32).collect();
    let (src_out, dst_out): (Vec<u32>, Vec<u32>) =
        bundle!(upload!(src_data.clone()), device_slice_alloc_zero!(u32, N))
            .and_then(|(src, dst)| src.copy_to(dst))
            .and_then(|(src, dst)| bundle!(download!(src), download!(dst)))
            .sync(&ctx)
            .expect("copy + both downloads");
    assert_eq!(src_out, src_data, "src unchanged after copy");
    assert_eq!(dst_out, src_data, "dst received src's bytes");
}

#[test]
fn device_slice_copy_length_mismatch_errors() {
    // src=10, dst=5 → expect Error::LengthMismatch { src: 10, dst: 5 }.
    // The Tier 1 CopyOp checks at into_event time; the Tier 2 wrapper
    // surfaces the same Err.
    let Some(ctx) = ctx() else { return };
    let result = bundle!(upload!(vec![0u32; 10]), device_slice_alloc_zero!(u32, 5))
        .and_then(|(src, dst)| src.copy_to(dst))
        .and_then(|(_src, dst)| download!(dst))
        .sync(&ctx);
    let err = result.expect_err("copy of mismatched lengths should error");
    assert!(
        matches!(err, Error::LengthMismatch { src: 10, dst: 5 }),
        "expected LengthMismatch {{ src: 10, dst: 5 }}, got {err:?}",
    );
}

// ── device_slice_write ─────────────────────────────────────────────

#[test]
fn device_slice_write_into_existing_buffer() {
    // alloc → write(host vec) → download — buffer sees what we wrote.
    let Some(ctx) = ctx() else { return };
    let data: Vec<u32> = (1..=N as u32).collect();
    let result: Vec<u32> = device_slice_alloc_zero!(u32, N)
        .and_then(|buf| device_slice_write(buf, data.clone()))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("alloc + write + download");
    assert_eq!(result, data);
}

#[test]
fn device_slice_write_overwrites_kernel_output() {
    // kernel(fill = 99) → write(host vec [1..N]) → download. The
    // write must wait for the kernel to complete (else the kernel
    // could overwrite our host data). Final state: host data.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let data: Vec<u32> = (1..=N as u32).collect();
    let result: Vec<u32> = device_slice_alloc_zero!(u32, N)
        .and_then(|buf| kernels.fill_u32([N], buf, 99u32))
        .and_then(|buf| device_slice_write(buf, data.clone()))
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("kernel + write + download");
    assert_eq!(result, data);
}

// ── mapped_slice_fill ──────────────────────────────────────────────

#[test]
fn mapped_slice_fill_in_place() {
    // SVM analog of device_slice_fill_in_place. Read back via
    // Tier 1 .map() since download!(buf) is DeviceSlice-only.
    let Some(ctx) = ctx_with_svm() else { return };
    let buf: MappedSlice<u32> = mapped_slice_alloc_zero!(u32, N)
        .and_then(|buf| mapped_slice_fill(buf, 7u32))
        .sync(&ctx)
        .expect("alloc + fill");
    assert_eq!(buf.len(), N);
    let g = buf.map().wait(&ctx).expect("map for read-back");
    assert!(g.iter().all(|&v| v == 7));
}

// ── mapped_slice_copy ──────────────────────────────────────────────

#[test]
fn mapped_slice_copy_propagates_src_to_dst() {
    // SVM analog of device_slice_copy_propagates_src_to_dst.
    let Some(ctx) = ctx_with_svm() else { return };
    let src_data: Vec<u32> = (0..N as u32).map(|i| i + 1000).collect();
    let (_src, dst): (MappedSlice<u32>, MappedSlice<u32>) = bundle!(
        mapped_slice_upload!(src_data.clone()),
        mapped_slice_alloc_zero!(u32, N),
    )
    .and_then(|(src, dst)| src.copy_to(dst))
    .sync(&ctx)
    .expect("upload + alloc + copy");
    let g = dst.map().wait(&ctx).expect("map dst for read-back");
    assert_eq!(&*g, src_data.as_slice());
}

// ── CopyTo with Uninit destination ────────────────────────────────

#[test]
fn copy_to_device_slice_uninit_propagates_and_transitions_to_init() {
    // Exercises the (DeviceSlice, DeviceSliceUninit) CopyTo impl.
    // The Uninit→Init transition happens inside the op — user writes
    // `src.copy_to(uninit_dst)` and gets back a fully-initialised
    // DeviceSlice. No `unsafe { assume_init() }` at the call site.
    let Some(ctx) = ctx() else { return };
    let src_data: Vec<u32> = (100..100 + N as u32).collect();
    let result: Vec<u32> = bundle!(
        upload!(src_data.clone()),
        claspr_async::device_slice_alloc_uninit!(u32, N),
    )
    .and_then(|(src, uninit_dst)| src.copy_to(uninit_dst))
    .and_then(|(_src, dst)| download!(dst))
    .sync(&ctx)
    .expect("upload + alloc_uninit + copy_to + download");
    assert_eq!(result, src_data);
}

#[test]
fn copy_to_mapped_slice_uninit_propagates_and_transitions_to_init() {
    // SVM analog of the above. Exercises (MappedSlice, MappedSliceUninit).
    let Some(ctx) = ctx_with_svm() else { return };
    let src_data: Vec<u32> = (200..200 + N as u32).collect();
    let (_src, dst): (MappedSlice<u32>, MappedSlice<u32>) = bundle!(
        mapped_slice_upload!(src_data.clone()),
        claspr_async::mapped_slice_alloc_uninit!(u32, N),
    )
    .and_then(|(src, uninit_dst)| src.copy_to(uninit_dst))
    .sync(&ctx)
    .expect("svm upload + alloc_uninit + copy_to");
    let g = dst.map().wait(&ctx).expect("map dst");
    assert_eq!(&*g, src_data.as_slice());
}

#[test]
fn copy_to_cross_type_mapped_to_usm_svm_memcpy() {
    // Cross-type (MappedSlice → USMSlice). Goes through
    // clEnqueueSVMMemcpy with src.ptr (SVM) → dst.ptr (host ptr).
    // Requires fine-grain-system SVM (USMSlice's gate).
    let Some(ctx) = ctx_with_svm() else { return };
    if ctx.svm_capability() != SvmLevel::FineSystem {
        eprintln!("SKIP: cross-type mapped→usm needs FineSystem SVM");
        return;
    }
    let src_data: Vec<u32> = (300..300 + N as u32).collect();
    let (_src, dst): (MappedSlice<u32>, claspr::USMSlice<u32>) = bundle!(
        mapped_slice_upload!(src_data.clone()),
        claspr_async::usm_slice_alloc_zero!(u32, N),
    )
    .and_then(|(src, dst)| src.copy_to(dst))
    .sync(&ctx)
    .expect("svm upload + usm alloc + cross-type copy");
    // USMSlice is host memory — read directly via Deref.
    assert_eq!(&dst[..], src_data.as_slice());
}
