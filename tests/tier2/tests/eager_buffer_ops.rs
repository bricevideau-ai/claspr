//! Eager-API port of `buffer_ops.rs`. Same N, same data, same assertions,
//! rewritten against `claspr::eager`.
//!
//! Old → new mapping:
//!   `device_slice_alloc_zero!(u32, N)`  → `alloc_zero::<u32, ReadWrite>(N)`
//!   `device_slice_fill(buf, v)`         → `fill(buf, v)`
//!   `device_slice_write(buf, vec)`      → `write(buf, vec)` (eager `write` op:
//!       the same non-blocking `clEnqueueWriteBuffer`, not a host-view seam).
//!   `src.copy_to(dst)`                  → `eager_copy_to(src, dst)` (2-output:
//!       `.and_then(|(_src, dst)| download(dst))`).
//!   `upload!(v)`                        → `upload::<T, ReadWrite, _>(v)`
//!   `download!(buf)`                    → `download`
//!   `bundle!(a, b)`                     → `bundle2(a, b)`
//!
//! SVM/USM cases (`mapped_slice_*`, cross-type `copy_to`): the eager API has no
//! `mapped_slice_alloc_zero` / `mapped_slice_upload` PRODUCING op, but every
//! such buffer is built synchronously (Tier 1 `MappedSlice::alloc_zero` /
//! `from_slice` / `USMSlice::alloc_zero`) and threaded as a CONCRETE
//! `eager_copy_to` head — `eager_copy_to`'s `Src: CopyTo<Dst>` bound covers all
//! the same Mapped→Mapped / Mapped→USM pairs. Same values, same assertions.

use claspr::eager::{
    EagerOpExt, alloc_zero, bundle2, download, eager_copy_to, fill, upload, write,
};
use claspr::{Buffer, Context, DeviceSlice, Error, MappedSlice, SvmLevel};
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

/// buffer_ops.rs::device_slice_fill_in_place.
#[test]
fn device_slice_fill_in_place() {
    let Some(ctx) = ctx() else { return };
    let result: Vec<u32> = alloc_zero::<u32, claspr::ReadWrite>(N)
        .and_then(|buf| fill(buf, 7u32))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc + fill + download chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 7));
}

/// buffer_ops.rs::device_slice_fill_chains_after_upload.
#[test]
fn device_slice_fill_chains_after_upload() {
    let Some(ctx) = ctx() else { return };
    let input: Vec<u32> = (0..N as u32).collect();
    let result: Vec<u32> = upload::<u32, claspr::ReadWrite, _>(input)
        .and_then(|buf| fill(buf, 99u32))
        .and_then(download)
        .sync(&ctx)
        .expect("upload + fill chain");
    assert!(result.iter().all(|&v| v == 99));
}

/// buffer_ops.rs::device_slice_fill_event_threads_to_kernel.
#[test]
fn device_slice_fill_event_threads_to_kernel() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = alloc_zero::<u32, claspr::ReadWrite>(N)
        .and_then(|buf| fill(buf, 7u32))
        .and_then(|buf| kernels.scale_u32([N], buf, 2u32))
        .and_then(download)
        .sync(&ctx)
        .expect("fill → kernel → download chain");
    assert!(result.iter().all(|&v| v == 14));
}

// ── device_slice_copy ──────────────────────────────────────────────

/// buffer_ops.rs::device_slice_copy_propagates_src_to_dst.
#[test]
fn device_slice_copy_propagates_src_to_dst() {
    let Some(ctx) = ctx() else { return };
    let src_data: Vec<u32> = (0..N as u32).map(|i| i * 3).collect();
    // `eager_copy_to`'s `Src: CopyTo<Dst>` bound is on the concrete buffer
    // types, so its heads are concrete buffers (or pipes of them), not upstream
    // ops — build them synchronously first (mirrors eager_cutover::device_copy_eager).
    let src = upload::<u32, claspr::ReadWrite, _>(src_data.clone())
        .sync(&ctx)
        .expect("upload src");
    let dst = alloc_zero::<u32, claspr::ReadWrite>(N)
        .sync(&ctx)
        .expect("alloc dst");
    let result: Vec<u32> = eager_copy_to(src, dst)
        .and_then(|(_src, dst)| download(dst))
        .sync(&ctx)
        .expect("upload + alloc + copy + download");
    assert_eq!(result, src_data);
}

/// buffer_ops.rs::device_slice_copy_returns_both_buffers — after copy, src is
/// still readable; download both.
#[test]
fn device_slice_copy_returns_both_buffers() {
    let Some(ctx) = ctx() else { return };
    let src_data: Vec<u32> = (10..10 + N as u32).collect();
    let src = upload::<u32, claspr::ReadWrite, _>(src_data.clone())
        .sync(&ctx)
        .expect("upload src");
    let dst = alloc_zero::<u32, claspr::ReadWrite>(N)
        .sync(&ctx)
        .expect("alloc dst");
    let (src_out, dst_out): (Vec<u32>, Vec<u32>) = eager_copy_to(src, dst)
        .and_then(|(src, dst)| bundle2(download(src), download(dst)))
        .sync(&ctx)
        .expect("copy + both downloads");
    assert_eq!(src_out, src_data, "src unchanged after copy");
    assert_eq!(dst_out, src_data, "dst received src's bytes");
}

/// buffer_ops.rs::device_slice_copy_length_mismatch_errors — src=10, dst=5.
#[test]
fn device_slice_copy_length_mismatch_errors() {
    let Some(ctx) = ctx() else { return };
    let src = upload::<u32, claspr::ReadWrite, _>(vec![0u32; 10])
        .sync(&ctx)
        .expect("upload src");
    let dst = alloc_zero::<u32, claspr::ReadWrite>(5)
        .sync(&ctx)
        .expect("alloc dst");
    let result = eager_copy_to(src, dst)
        .and_then(|(_src, dst)| download(dst))
        .sync(&ctx);
    let err = result.expect_err("copy of mismatched lengths should error");
    assert!(
        matches!(err, Error::LengthMismatch { src: 10, dst: 5 }),
        "expected LengthMismatch {{ src: 10, dst: 5 }}, got {err:?}",
    );
}

// ── device_slice_write ─────────────────────────────────────────────

/// buffer_ops.rs::device_slice_write_into_existing_buffer — write a host vec into
/// an existing buffer via the eager `write` op (`device_slice_write`'s analog: a
/// non-blocking `clEnqueueWriteBuffer`, NOT a map/host-memcpy seam). alloc →
/// write → download.
#[test]
fn device_slice_write_into_existing_buffer() {
    let Some(ctx) = ctx() else { return };
    let data: Vec<u32> = (1..=N as u32).collect();
    let result: Vec<u32> = alloc_zero::<u32, claspr::ReadWrite>(N)
        .and_then(|buf| write(buf, data.clone()))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc + write + download");
    assert_eq!(result, data);
}

/// buffer_ops.rs::device_slice_write_overwrites_kernel_output — kernel fill 99,
/// then the `write` of [1..N] must wait for the kernel (its event threads through
/// the chain's deps into the write's wait-list); final state = host data.
#[test]
fn device_slice_write_overwrites_kernel_output() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let data: Vec<u32> = (1..=N as u32).collect();
    let result: Vec<u32> = alloc_zero::<u32, claspr::ReadWrite>(N)
        .and_then(|buf| kernels.fill_u32([N], buf, 99u32))
        .and_then(|buf| write(buf, data.clone()))
        .and_then(download)
        .sync(&ctx)
        .expect("kernel + write + download");
    assert_eq!(result, data);
}

// ── mapped_slice_fill ──────────────────────────────────────────────

/// buffer_ops.rs::mapped_slice_fill_in_place — SVM analog. The eager API has no
/// mapped-fill verb over an existing `MappedSlice`; `MappedSlice::fill` (Tier 1)
/// is the primitive both layers call, so the alloc + fill use it directly.
#[test]
fn mapped_slice_fill_in_place() {
    let Some(ctx) = ctx_with_svm() else { return };
    let buf: MappedSlice<u32> = MappedSlice::alloc_zero(&ctx, N).expect("alloc");
    buf.fill(7u32).wait().expect("fill");
    assert_eq!(buf.len(), N);
    let g = buf.map().wait().expect("map for read-back");
    assert!(g.iter().all(|&v| v == 7));
}

// ── mapped_slice_copy ──────────────────────────────────────────────

/// buffer_ops.rs::mapped_slice_copy_propagates_src_to_dst — SVM analog via two
/// concrete `MappedSlice` heads + `eager_copy_to`.
#[test]
fn mapped_slice_copy_propagates_src_to_dst() {
    let Some(ctx) = ctx_with_svm() else { return };
    let src_data: Vec<u32> = (0..N as u32).map(|i| i + 1000).collect();
    let src = MappedSlice::from_slice(&ctx, &src_data).expect("upload src");
    let dst = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc dst");
    let (_src, dst): (MappedSlice<u32>, MappedSlice<u32>) = eager_copy_to(src, dst)
        .sync(&ctx)
        .expect("upload + alloc + copy");
    let g = dst.map().wait().expect("map dst for read-back");
    assert_eq!(&*g, src_data.as_slice());
}

// ── CopyTo with Uninit destination ────────────────────────────────

/// buffer_ops.rs::copy_to_device_slice_uninit_propagates_and_transitions_to_init
/// — DeviceSlice → DeviceSliceUninit via `eager_copy_to`.
#[test]
fn copy_to_device_slice_uninit_propagates_and_transitions_to_init() {
    let Some(ctx) = ctx() else { return };
    let src_data: Vec<u32> = (100..100 + N as u32).collect();
    let src = upload::<u32, claspr::ReadWrite, _>(src_data.clone())
        .sync(&ctx)
        .expect("upload src");
    let uninit_dst = DeviceSlice::<u32>::alloc_uninit(&ctx, N).expect("alloc_uninit");
    let result: Vec<u32> = eager_copy_to(src, uninit_dst)
        .and_then(|(_src, dst)| download(dst))
        .sync(&ctx)
        .expect("upload + alloc_uninit + copy_to + download");
    assert_eq!(result, src_data);
}

/// buffer_ops.rs::copy_to_mapped_slice_uninit_propagates_and_transitions_to_init
/// — SVM analog: MappedSlice → MappedSliceUninit.
#[test]
fn copy_to_mapped_slice_uninit_propagates_and_transitions_to_init() {
    let Some(ctx) = ctx_with_svm() else { return };
    let src_data: Vec<u32> = (200..200 + N as u32).collect();
    let src = MappedSlice::from_slice(&ctx, &src_data).expect("svm upload");
    let uninit_dst = MappedSlice::<u32>::alloc_uninit(&ctx, N).expect("mapped alloc_uninit");
    let (_src, dst): (MappedSlice<u32>, MappedSlice<u32>) = eager_copy_to(src, uninit_dst)
        .sync(&ctx)
        .expect("svm upload + alloc_uninit + copy_to");
    let g = dst.map().wait().expect("map dst");
    assert_eq!(&*g, src_data.as_slice());
}

/// buffer_ops.rs::copy_to_cross_type_mapped_to_usm_svm_memcpy — cross-type
/// MappedSlice → USMSlice (FineSystem SVM) via `eager_copy_to`.
#[test]
fn copy_to_cross_type_mapped_to_usm_svm_memcpy() {
    let Some(ctx) = ctx_with_svm() else { return };
    if ctx.svm_capability() != SvmLevel::FineSystem {
        eprintln!("SKIP: cross-type mapped→usm needs FineSystem SVM");
        return;
    }
    let src_data: Vec<u32> = (300..300 + N as u32).collect();
    let src = MappedSlice::from_slice(&ctx, &src_data).expect("svm upload");
    let dst = claspr::USMSlice::<u32>::alloc_zero(&ctx, N).expect("usm alloc");
    let (_src, dst): (MappedSlice<u32>, claspr::USMSlice<u32>) = eager_copy_to(src, dst)
        .sync(&ctx)
        .expect("svm upload + usm alloc + cross-type copy");
    // USMSlice is host memory — read directly via Deref.
    assert_eq!(&dst[..], src_data.as_slice());
}
