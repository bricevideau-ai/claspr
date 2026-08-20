//! Eager-API port of `marker_aware_constructors.rs`. Same N, same data, same
//! assertions, rewritten against `claspr::eager`.
//!
//! Old → new mapping:
//!   `device_slice_from_slice!(data; M)`  → `upload_as(data, M)` (witness arg)
//!       (`upload`'s leaf does a synchronous `DeviceSlice::from_slice`
//!       CL_MEM_COPY_HOST_PTR create — no `Fillable` bound, so it works for any
//!       marker incl. `Frozen`, exactly like `device_slice_from_slice!`. The
//!       marker is inferred from the witness, no turbofish.)
//!   `device_slice_alloc_zero!(u32, N; M)` → `alloc_zero_as::<u32, _>(N, M)`
//!   `device_slice_filled!(v, N; M)`       → `alloc_zero_as::<u32,_>(N, M).fill(v)`
//!   `device_slice_alloc_uninit!(u32, N)`  → concrete `DeviceSliceUninit` head
//!       (the eager API has no uninit-producing device op; the uninit is built
//!       synchronously and threaded as the chain head, then kernel-written —
//!       same compositional path as the original).
//!   `download!(buf)`                      → `download`

use claspr::eager::{DeviceOpExt, alloc_zero_as, download, fill, upload_as};
use claspr::{DeviceSlice, Frozen, HostReadOnly, ReadOnly};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const N: usize = 32;

/// marker_aware::from_slice_with_frozen_marker_round_trips — `upload` uses
/// `from_slice` (no Fillable bound), so the Frozen marker round-trips.
#[test]
fn from_slice_with_frozen_marker_round_trips() {
    let Some(ctx) = ctx() else { return };
    let data: Vec<u32> = (0..N as u32).collect();
    let result = upload_as(data.clone(), Frozen)
        .and_then(download)
        .sync(&ctx)
        .expect("from_slice Frozen + download chain");
    assert_eq!(*result, data);
}

/// marker_aware::from_slice_with_read_only_marker.
#[test]
fn from_slice_with_read_only_marker() {
    let Some(ctx) = ctx() else { return };
    let data: Vec<u32> = (10..10 + N as u32).collect();
    let result = upload_as(data.clone(), ReadOnly)
        .and_then(download)
        .sync(&ctx)
        .expect("from_slice ReadOnly + download chain");
    assert_eq!(*result, data);
}

/// marker_aware::alloc_zero_with_host_read_only_via_device_kernel_path —
/// HostReadOnly is Fillable (DeviceKernel fill strategy), so `alloc_zero` works.
#[test]
fn alloc_zero_with_host_read_only_via_device_kernel_path() {
    let Some(ctx) = ctx() else { return };
    let result = alloc_zero_as::<u32, _>(N, HostReadOnly)
        .and_then(download)
        .sync(&ctx)
        .expect("alloc_zero HostReadOnly + download chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 0));
}

/// marker_aware::filled_with_read_only_marker — `device_slice_filled!(7, N;
/// ReadOnly)` → `alloc_zero_as::<u32, _>(N, ReadOnly).fill(7)`.
#[test]
fn filled_with_read_only_marker() {
    let Some(ctx) = ctx() else { return };
    let result = alloc_zero_as::<u32, _>(N, ReadOnly)
        .and_then(|buf| fill(buf, 7u32))
        .and_then(download)
        .sync(&ctx)
        .expect("filled ReadOnly + download chain");
    assert!(result.iter().all(|&v| v == 7));
}

/// marker_aware::alloc_uninit_returns_uninit_op_output — uninit head consumed
/// via assume_init + kernel-write, then download.
#[test]
fn alloc_uninit_returns_uninit_op_output() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let uninit = DeviceSlice::<u32>::alloc_uninit(&ctx, N).expect("alloc_uninit");
    // SAFETY: fill_u32 kernel writes every slot. The eager API has no
    // uninit-producing device op, so the uninit is built synchronously and
    // threaded as the concrete chain head; `assume_init` is applied up front
    // (the kernel writes every element below), matching the original's intent.
    let buf = unsafe { uninit.assume_init() };
    let result = kernels
        .fill_u32([N], buf, 99)
        .and_then(download)
        .sync(&ctx)
        .expect("alloc_uninit + kernel-write + download chain");
    assert!(result.iter().all(|&v| v == 99));
}
