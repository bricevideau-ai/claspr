//! Spike scenario 14 — cross-device pipeline within a single
//! multi-device Context. The shared `cl_context` makes events on one
//! device's queue valid as deps for ops on another device's queue,
//! so the chain spans devices naturally via `.on_device(&dev)` /
//! `transfer_to_device(buf, &dev)` (both non-blocking; the chain
//! never host-blocks).
//!
//! Device handles come from `ec.context().devices()` inside
//! `.and_then_with_context` closures (via the `ec.device_at(i)`
//! shortcut), not from external captures — chain stays portable
//! across contexts.
//!
//! Skips when only one device is available (no real multi-device
//! platform AND no sub-device partition support).

use claspr::device::Platform;
use claspr::{Context, Device};
use claspr_async::{DeviceOperation, download, upload};
use claspr_test_kernels::kernels;

const N: usize = 64;

/// Three-stage discovery, matching tier1/tests/multi_device.rs:
/// real multi-device → sub-device partition → skip. See that file
/// for the rationale.
fn ctx_two_devices() -> Option<(Context, Device, Device)> {
    if let Ok(platforms) = Platform::all() {
        for p in platforms {
            if let Ok(devs) = p.devices()
                && devs.len() >= 2
            {
                let dev_a = devs[0].clone();
                let dev_b = devs[1].clone();
                let ctx = Context::builder()
                    .devices(&[dev_a.clone(), dev_b.clone()])
                    .build()
                    .ok()?;
                return Some((ctx, dev_a, dev_b));
            }
        }
    }
    if let Ok(devs) = Device::all() {
        for parent in devs {
            if parent.partition_max_sub_devices().unwrap_or(0) < 2 {
                continue;
            }
            let cu = parent.max_compute_units().unwrap_or(0);
            if cu < 2 {
                continue;
            }
            // partition_equally takes CUs-per-sub-device, not number of
            // sub-devices — see its rustdoc. cu/2 yields 2 sub-devices.
            let Ok(subs) = parent.partition_equally(cu / 2) else {
                continue;
            };
            if subs.len() < 2 {
                continue;
            }
            let dev_a = subs[0].clone();
            let dev_b = subs[1].clone();
            let ctx = Context::builder()
                .devices(&[dev_a.clone(), dev_b.clone()])
                .build()
                .ok()?;
            return Some((ctx, dev_a, dev_b));
        }
    }
    eprintln!(
        "SKIP: no platform with ≥2 devices and no partitionable device \
         (CL_DEVICE_PARTITION_EQUALLY with max_sub_devices ≥ 2)",
    );
    None
}

#[test]
fn pipeline_spans_two_devices_via_mapped_slice() {
    // Both devices share the cl_context, so a `DeviceSlice<T>` is
    // valid on either device's queue. Stage 1 on device 0, stage 2
    // on device 1; the chain is fully non-blocking — each kernel is
    // routed via `.on_device(ec.device_at(i))`, no `.wait()` inside
    // any closure. Device handles are pulled from `ec` rather than
    // captured from outer scope (chain stays portable across
    // contexts).
    let Some((ctx, _dev_a, _dev_b)) = ctx_two_devices() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let result: Vec<u32> = upload(vec![0u32; N])
        .and_then_with_context(move |ec, buf| {
            kernels_ref.fill_u32([N], buf, 3).on_device(ec.device_at(0))
        })
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 4)
                .on_device(ec.device_at(1))
        })
        .and_then(download)
        .sync(&ctx)
        .expect("cross-device chain");
    assert!(result.iter().all(|&v| v == 12));
}

#[test]
fn downloaded_vec_can_be_reuploaded_into_a_fresh_chain() {
    // Chain 1 ends with `download` (host-owned `Vec<T>`); chain 2
    // starts with `upload` of that same Vec. Pins the memory-lifecycle
    // path so a future regression that broke ownership semantics
    // (e.g. download returning a borrow rather than an owned Vec) would
    // surface here. The two chains share a multi-device Context but
    // both run on the chain's default OOO queue — per-op device routing
    // is a separate open question and not what this test exercises.
    let Some((ctx, _dev_a, _dev_b)) = ctx_two_devices() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let intermediate: Vec<u32> = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 5))
        .and_then(download)
        .sync(&ctx)
        .expect("chain 1");
    assert!(intermediate.iter().all(|&v| v == 5));

    let final_result: Vec<u32> = upload(intermediate)
        .and_then(|buf| kernels.scale_u32([N], buf, 6))
        .and_then(download)
        .sync(&ctx)
        .expect("chain 2");
    assert!(final_result.iter().all(|&v| v == 30));
}
