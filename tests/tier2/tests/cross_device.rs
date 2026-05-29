//! Spike scenario 14 — cross-device pipeline within a single
//! multi-device Context. The shared `cl_context` makes events on one
//! device's queue valid as deps for ops on another device's queue,
//! so the chain naturally spans devices.
//!
//! Skips when only one device is available. Today's claspr-async
//! `ExecutionContext` picks the context's default OOO queue, which
//! is per-device — so the chain runs on whichever device the context
//! considers default. Cross-device routing would require
//! `op.on_device(&dev_b)` per-op (Tier 2 open question 4) or manual
//! `with_context` + per-device queues. We exercise the latter form
//! to validate the multi-device path doesn't break the chain plumbing.

use claspr::device::Platform;
use claspr::{Context, Device, DeviceSlice};
use claspr_async::{DeviceOperation, download, upload, with_context};
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
fn pipeline_spans_two_devices_via_shared_buffer() {
    // Both devices share the cl_context, so a `DeviceSlice<T>` is
    // valid on either device's queue. Stage 1 on device A, stage 2
    // on device B; the buffer's cl_mem refcount + per-queue command
    // ordering does the cross-device sync.
    //
    // We use `with_context` to opt into a specific queue per stage
    // (the default OOO routing isn't aware of cross-device intent).
    let Some((ctx, dev_a, dev_b)) = ctx_two_devices() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let q_a = claspr::Queue::<claspr::InOrder>::on_device(&ctx, &dev_a).expect("queue on dev_a");
    let q_b = claspr::Queue::<claspr::InOrder>::on_device(&ctx, &dev_b).expect("queue on dev_b");
    let q_a_ref = &q_a;
    let q_b_ref = &q_b;

    let result: Vec<u32> = upload(vec![0u32; N])
        .and_then(move |buf| {
            with_context(move |_ec| {
                let buf = kernels_ref.fill_u32([N], buf, 3).wait(q_a_ref)?;
                Ok::<DeviceSlice<u32>, claspr::Error>(buf)
            })
        })
        .and_then(move |buf| {
            with_context(move |_ec| {
                let buf = kernels_ref.scale_u32([N], buf, 4).wait(q_b_ref)?;
                Ok::<DeviceSlice<u32>, claspr::Error>(buf)
            })
        })
        .and_then(download)
        .sync(&ctx)
        .expect("cross-device chain");
    assert!(result.iter().all(|&v| v == 12));
}

#[test]
fn download_from_device_b_can_be_reuploaded_to_device_a() {
    // The "cross-context buffer flow" REVIEW.md identifies as where
    // most users hit problems first: end a chain on device B with a
    // download, take the Vec back, then feed it to a fresh chain
    // that uploads to device A and processes it further.
    //
    // The Vec is fully host-owned between the two chains — no
    // cl_mem aliasing risk — but the test pins the memory-lifecycle
    // path so a future regression that broke ownership semantics
    // (e.g. download returning a borrow rather than an owned Vec)
    // would surface here.
    let Some((ctx, dev_a, dev_b)) = ctx_two_devices() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;
    let q_a = claspr::Queue::<claspr::InOrder>::on_device(&ctx, &dev_a).expect("queue on dev_a");
    let q_b = claspr::Queue::<claspr::InOrder>::on_device(&ctx, &dev_b).expect("queue on dev_b");
    let q_a_ref = &q_a;
    let q_b_ref = &q_b;

    // Chain 1: fill on device B, download to host.
    let intermediate: Vec<u32> = upload(vec![0u32; N])
        .and_then(move |buf| {
            with_context(move |_ec| {
                let buf = kernels_ref.fill_u32([N], buf, 5).wait(q_b_ref)?;
                Ok::<DeviceSlice<u32>, claspr::Error>(buf)
            })
        })
        .and_then(download)
        .sync(&ctx)
        .expect("chain 1 (device B → host)");
    assert!(intermediate.iter().all(|&v| v == 5));

    // Chain 2: re-upload to device A, scale, download.
    let final_result: Vec<u32> = upload(intermediate)
        .and_then(move |buf| {
            with_context(move |_ec| {
                let buf = kernels_ref.scale_u32([N], buf, 6).wait(q_a_ref)?;
                Ok::<DeviceSlice<u32>, claspr::Error>(buf)
            })
        })
        .and_then(download)
        .sync(&ctx)
        .expect("chain 2 (host → device A)");
    assert!(final_result.iter().all(|&v| v == 30));
}
