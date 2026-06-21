//! Eager-API port of `transfer_to_device.rs`: explicit `DeviceSlice` migration
//! as an eager graph leaf.
//!
//! The closure layer exposes `claspr_async::transfer_to_device(buf, &dev)`, a
//! `DeviceOperation` that enqueues a `clEnqueueMigrateMemObjects` on the target
//! device's default OOO queue. The eager analog is
//! [`claspr::eager::transfer_to_device`] — a buffer-consuming **leaf** (same
//! family as `download` / `fill`, taking `impl Into<Input<DeviceSlice<…>>>`), not
//! a wrapping method like `.on_device`. It composes via `and_then_with_context`
//! so the target `Device` is pulled from `ec` (the portable idiom, no external
//! Device captures), exactly mirroring the old:
//!   `.and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(i)))`
//!
//! Old → new mapping:
//!   `claspr_async::transfer_to_device(buf, dev)` → `claspr::eager::transfer_to_device(buf, dev)`
//!   `upload!(v)`                                 → `upload::<u32, ReadWrite, _>(v)`
//!   `download!(buf)`                             → `download`
//!   `.and_then_with_context(...)`                → same name on `EagerOpExt`
//!   `kernel(...).on_device(dev)`                 → same `.on_device(...)` eager op
//!
//! ## Single-device runners (pocl)
//!
//! The old tests routed to `device_at(1)` and skipped when only one device was
//! available. On the common single-device pocl runner, migrating a buffer to the
//! device it already lives on is a valid **no-op exercise**: the migrate is still
//! enqueued and its event still threads through the chain's deps, so it proves
//! the op enqueues + composes correctly even where there is no second device. So
//! these tests pick the target index as `min(desired, num_devices - 1)` — they
//! run for real everywhere, and exercise true cross-device movement when a second
//! device exists.

use claspr::device::Platform;
use claspr::eager::{EagerOpExt, download, transfer_to_device, upload};
use claspr::{Context, Device, ReadWrite};
use claspr_test_kernels::kernels;

const N: usize = 64;

/// Build a context: prefer a real two-device platform, then a sub-device
/// partition, else fall back to a single default device (so the tests still run
/// on pocl). Mirrors `transfer_to_device.rs`'s discovery, with a single-device
/// fallback appended.
fn ctx() -> Option<Context> {
    if let Ok(platforms) = Platform::all() {
        for p in platforms {
            if let Ok(devs) = p.devices()
                && devs.len() >= 2
                && let Ok(ctx) = Context::builder()
                    .devices(&[devs[0].clone(), devs[1].clone()])
                    .build()
            {
                return Some(ctx);
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
            let Ok(subs) = parent.partition_equally(cu / 2) else {
                continue;
            };
            if subs.len() < 2 {
                continue;
            }
            if let Ok(ctx) = Context::builder()
                .devices(&[subs[0].clone(), subs[1].clone()])
                .build()
            {
                return Some(ctx);
            }
        }
    }
    // Single-device fallback — migrate-to-same-device is a valid no-op exercise.
    match Context::any() {
        Ok(ctx) => Some(ctx),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device available");
            None
        }
    }
}

/// transfer_to_device.rs::transfer_to_device_completes_in_chain —
/// upload → transfer → kernel.on_device → download. The transfer is a queue
/// command (non-blocking); the chain completes without an explicit `.wait()`,
/// without hang, and the result matches. Routes to `device_at(1)` where a second
/// device exists, else `device_at(0)` (migrate no-op).
#[test]
fn transfer_to_device_completes_in_chain() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;
    // Desired second device, clamped to what's actually present.
    let target = (ctx.devices().len() - 1).min(1);

    let result: Vec<u32> = upload::<u32, ReadWrite, _>(vec![5u32; N])
        .and_then_with_context(move |ec, buf| transfer_to_device(buf, ec.device_at(target)))
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 4)
                .on_device(ec.device_at(target))
        })
        .and_then(download)
        .sync(&ctx)
        .expect("transfer + on_device chain");

    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 20));
}

/// transfer_to_device.rs::transfer_then_on_device_matches_scenario_14_shape —
/// the literal scenario-14 shape (transfer → scale → transfer → scale →
/// transfer → download). Regression test for the cross-device pipeline. Each
/// transfer/scale is routed via `device_at(i)`, clamped to the available device
/// count (so on single-device pocl every stage runs on device 0 and the
/// migrates are no-ops, but still enqueue + thread deps).
#[test]
fn transfer_then_on_device_matches_scenario_14_shape() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;
    let ndev = ctx.devices().len();
    let d0 = 0usize;
    let d1 = (ndev - 1).min(1);

    let result: Vec<u32> = upload::<u32, ReadWrite, _>(vec![1u32; N])
        .and_then_with_context(move |ec, buf| transfer_to_device(buf, ec.device_at(d0)))
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 2)
                .on_device(ec.device_at(d0))
        })
        .and_then_with_context(move |ec, buf| transfer_to_device(buf, ec.device_at(d1)))
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 10)
                .on_device(ec.device_at(d1))
        })
        .and_then_with_context(move |ec, buf| transfer_to_device(buf, ec.device_at(d0)))
        .and_then(download)
        .sync(&ctx)
        .expect("scenario-14 chain");

    assert_eq!(result.len(), N);
    // 1 * 2 * 10 = 20
    assert!(result.iter().all(|&v| v == 20));
}
