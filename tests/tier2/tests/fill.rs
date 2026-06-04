//! `DeviceSlice::fill` (Tier 1) + `device_slice_filled` / the
//! `device_slice!` macro (Tier 2).
//!
//! Coverage:
//! - Tier 1 fill on an existing DeviceSlice → readback verifies the
//!   pattern landed in every slot.
//! - Tier 2 `device_slice_filled` op in a chain → downstream kernel
//!   reads the filled buffer.
//! - Macro `[value; count]` arm → same as `device_slice_filled`.
//! - Macro `[a, b, c]` arm → equivalent to `upload(vec![a, b, c])`.

use claspr::{Context, DeviceSlice};
use claspr_async::{DeviceOperation, device_slice, download};
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

#[test]
fn tier1_fill_writes_value_to_every_element() {
    // Alloc, fill, read back. No host upload, no kernel — exercise
    // the Tier 1 FillOp directly.
    let Some(ctx) = ctx() else { return };
    let mut buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");
    buf.fill(42u32).wait(&ctx).expect("fill");

    let mut readback = vec![0u32; N];
    buf.read(&mut readback).wait(&ctx).expect("read");
    assert!(readback.iter().all(|&v| v == 42));
}

#[test]
fn tier2_device_slice_filled_threads_into_chain() {
    // Lazy alloc+fill via `device_slice_filled` → scale by 2 →
    // download. Asserts the fill landed before scale read the buffer.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result: Vec<u32> = claspr_async::device_slice_filled(7u32, N)
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .and_then(download)
        .sync(&ctx)
        .expect("filled chain");
    assert!(result.iter().all(|&v| v == 14));
}

#[test]
fn macro_repeat_arm_matches_device_slice_filled() {
    // `device_slice![value; count]` → device-side fill (no host alloc).
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result: Vec<u32> = device_slice![3u32; N]
        .and_then(|buf| kernels.scale_u32([N], buf, 5))
        .and_then(download)
        .sync(&ctx)
        .expect("macro repeat chain");
    assert!(result.iter().all(|&v| v == 15));
}

#[test]
fn macro_literal_list_arm_uploads_host_vec() {
    // `device_slice![a, b, c, d]` → expands to upload(vec![a, b, c, d]).
    // The buffer must carry those exact values to the next stage.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Use 4 elements so the launch grid stays small.
    let result: Vec<u32> = device_slice![1u32, 2, 3, 4]
        .and_then(|buf| kernels.scale_u32([4], buf, 10))
        .and_then(download)
        .sync(&ctx)
        .expect("macro literal chain");
    assert_eq!(result, vec![10, 20, 30, 40]);
}
