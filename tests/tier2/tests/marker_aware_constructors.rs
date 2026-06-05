//! Tier 2 marker-aware constructor macros — verify the new from_slice
//! and alloc_uninit macros work with non-default markers, and that
//! marker propagation flows through chains correctly.

use claspr::{Context, DeviceSlice, Frozen, HostReadOnly, ReadOnly};
use claspr_async::{
    DeviceOperation, device_slice_alloc_uninit, device_slice_alloc_zero, device_slice_filled,
    device_slice_from_slice, download,
};
use claspr_test_kernels::kernels;

const N: usize = 32;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

#[test]
fn from_slice_with_frozen_marker_round_trips() {
    // Frozen lacks Fillable (no write path post-creation), so
    // device_slice_alloc_zero! / device_slice_filled! would reject.
    // device_slice_from_slice! works for any marker via
    // CL_MEM_COPY_HOST_PTR.
    let Some(ctx) = ctx() else { return };
    let data: Vec<u32> = (0..N as u32).collect();
    let result: Vec<u32> = device_slice_from_slice!(data.clone(); Frozen)
        .and_then(|buf: DeviceSlice<u32, Frozen>| download!(buf))
        .sync(&ctx)
        .expect("from_slice Frozen + download chain");
    assert_eq!(result, data);
}

#[test]
fn from_slice_with_read_only_marker() {
    // ReadOnly: kernel-RO + host-RW. Initialize via from_slice,
    // download to verify.
    let Some(ctx) = ctx() else { return };
    let data: Vec<u32> = (10..10 + N as u32).collect();
    let result: Vec<u32> = device_slice_from_slice!(data.clone(); ReadOnly)
        .and_then(|buf: DeviceSlice<u32, ReadOnly>| download!(buf))
        .sync(&ctx)
        .expect("from_slice ReadOnly + download chain");
    assert_eq!(result, data);
}

#[test]
fn alloc_zero_with_host_read_only_via_device_kernel_path() {
    // HostReadOnly: not HostWritable, so FILL_STRATEGY = DeviceKernel.
    // alloc_zero internally fills via the built-in fill kernel. Then
    // download verifies (HostReadOnly IS HostReadable).
    let Some(ctx) = ctx() else { return };
    let result: Vec<u32> = device_slice_alloc_zero!(u32, N; HostReadOnly)
        .and_then(|buf: DeviceSlice<u32, HostReadOnly>| download!(buf))
        .sync(&ctx)
        .expect("alloc_zero HostReadOnly + download chain");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 0));
}

#[test]
fn filled_with_read_only_marker() {
    // ReadOnly is HostWritable → FILL_STRATEGY = Runtime (fast path).
    // Tier 2 device_slice_filled! macro should work for ReadOnly.
    let Some(ctx) = ctx() else { return };
    let result: Vec<u32> = device_slice_filled!(7u32, N; ReadOnly)
        .and_then(|buf: DeviceSlice<u32, ReadOnly>| download!(buf))
        .sync(&ctx)
        .expect("filled ReadOnly + download chain");
    assert!(result.iter().all(|&v| v == 7));
}

#[test]
fn alloc_uninit_returns_uninit_op_output() {
    // device_slice_alloc_uninit! produces an Op whose Output is
    // DeviceSliceUninit<T, M>. Chain consumes the uninit via
    // assume_init + kernel-write, then downloads.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let result: Vec<u32> = device_slice_alloc_uninit!(u32, N)
        .and_then(|uninit| {
            // SAFETY: fill_u32 kernel writes every slot.
            let buf = unsafe { uninit.assume_init() };
            kernels.fill_u32([N], buf, 99)
        })
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("alloc_uninit + kernel-write + download chain");
    assert!(result.iter().all(|&v| v == 99));
}
