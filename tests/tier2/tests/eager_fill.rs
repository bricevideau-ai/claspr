//! Eager-API port of `fill.rs`. Same N, same fill/scale values, same
//! assertions, rewritten against `claspr::eager`.
//!
//! Old → new mapping:
//!   `device_slice_filled!(v, N)`     → `alloc_zero(N).and_then(|b| fill(b, v))`
//!   `device_slice![v; N]` (repeat)   → same (device-side fill, no host alloc)
//!   `device_slice![a, b, c]` (list)  → `upload(vec![a, b, c])`
//!   `download!(buf)`                 → `download`
//!
//! The Tier-1 `DeviceSlice::fill` test (`tier1_fill_writes_value_to_every_element`)
//! is reproduced byte-for-byte: it uses no Tier-2 op, only the Tier-1 fill verb,
//! so the eager port is identical (it is the runtime primitive both layers call).

use claspr::Context;
use claspr::DeviceSlice;
use claspr::eager::{DeviceOpExt, alloc_zero, download, fill, upload};
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

/// fill.rs::tier1_fill_writes_value_to_every_element — Tier 1 FillOp directly.
#[test]
fn tier1_fill_writes_value_to_every_element() {
    let Some(ctx) = ctx() else { return };
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let buf = buf.fill(42u32).wait().expect("fill");

    let mut readback = vec![0u32; N];
    buf.read(&mut readback).wait().expect("read");
    assert!(readback.iter().all(|&v| v == 42));
}

/// fill.rs::tier2_device_slice_filled_threads_into_chain — lazy alloc+fill →
/// scale ×2 → download. `device_slice_filled!(7, N)` → `alloc_zero(N).fill(7)`.
#[test]
fn tier2_device_slice_filled_threads_into_chain() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result: Vec<u32> = alloc_zero::<u32, claspr::ReadWrite>(N)
        .and_then(|buf| fill(buf, 7u32))
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .and_then(download)
        .sync(&ctx)
        .expect("filled chain");
    assert!(result.iter().all(|&v| v == 14));
}

/// fill.rs::macro_repeat_arm_matches_device_slice_filled — `device_slice![3; N]`
/// is the same device-side fill shape: `alloc_zero(N).fill(3)`.
#[test]
fn macro_repeat_arm_matches_device_slice_filled() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result: Vec<u32> = alloc_zero::<u32, claspr::ReadWrite>(N)
        .and_then(|buf| fill(buf, 3u32))
        .and_then(|buf| kernels.scale_u32([N], buf, 5))
        .and_then(download)
        .sync(&ctx)
        .expect("macro repeat chain");
    assert!(result.iter().all(|&v| v == 15));
}

/// fill.rs::macro_literal_list_arm_uploads_host_vec — `device_slice![1,2,3,4]`
/// → `upload(vec![1,2,3,4])`.
#[test]
fn macro_literal_list_arm_uploads_host_vec() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result: Vec<u32> = upload::<u32, claspr::ReadWrite, _>(vec![1u32, 2, 3, 4])
        .and_then(|buf| kernels.scale_u32([4], buf, 10))
        .and_then(download)
        .sync(&ctx)
        .expect("macro literal chain");
    assert_eq!(result, vec![10, 20, 30, 40]);
}
