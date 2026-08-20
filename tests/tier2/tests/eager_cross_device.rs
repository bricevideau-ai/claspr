//! Eager port of `cross_device.rs`: cross-device pipeline within a single
//! multi-device Context, expressed through the eager graph API. The shared
//! `cl_context` makes a `DeviceSlice<T>` valid on either device's queue, so the
//! chain spans devices via `.on_device_at(i)`, which resolves the device by
//! index against the running context at execute.
//!
//! Old → new mapping:
//!   `upload!(v)`                  → `upload(v)`
//!   `download!(buf)`              → `download`
//!   `kernel(...).on_device(dev)`  → `.on_device_at(i)` on the eager kernel op
//!
//! NOTE (eager seam): routing flows through the pipe-fed `.and_then`, so the
//! routed downstream's `clEnqueue*` carries the upstream's completion event on
//! its wait-list — a real per-command edge. Neither test asserts cross-device
//! event ORDERING (they assert final values and the download→reupload ownership
//! lifecycle), so the port is faithful.
//!
//! Skips when only one device is available (no real multi-device platform AND no
//! sub-device partition support). Guard copied verbatim from cross_device.rs.

use claspr::eager::{DeviceOpExt, download, upload};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx_two_devices;

const N: usize = 64;

/// cross_device.rs::pipeline_spans_two_devices_via_mapped_slice — stage 1
/// (fill 3) on device 0, stage 2 (scale 4) on device 1, then download. 3×4 = 12.
#[test]
fn pipeline_spans_two_devices_via_mapped_slice() {
    let Some((ctx, _dev_a, _dev_b)) = ctx_two_devices() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let result = upload(vec![0u32; N])
        .and_then(move |buf| kernels_ref.fill_u32([N], buf, 3).on_device_at(0))
        .and_then(move |buf| kernels_ref.scale_u32([N], buf, 4).on_device_at(1))
        .and_then(download)
        .sync(&ctx)
        .expect("cross-device chain");
    assert!(result.iter().all(|&v| v == 12));
}

/// cross_device.rs::downloaded_vec_can_be_reuploaded_into_a_fresh_chain —
/// chain 1 ends with download (host-owned Vec); chain 2 re-uploads that Vec.
/// Both run on the chain's default queue (no per-op routing here). 5×6 = 30.
#[test]
fn downloaded_vec_can_be_reuploaded_into_a_fresh_chain() {
    let Some((ctx, _dev_a, _dev_b)) = ctx_two_devices() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let intermediate = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 5))
        .and_then(download)
        .sync(&ctx)
        .expect("chain 1")
        .into_inner();
    assert!(intermediate.iter().all(|&v| v == 5));

    let final_result = upload(intermediate)
        .and_then(|buf| kernels.scale_u32([N], buf, 6))
        .and_then(download)
        .sync(&ctx)
        .expect("chain 2");
    assert!(final_result.iter().all(|&v| v == 30));
}
