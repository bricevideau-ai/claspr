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
//!   `upload!(v)`                                 → `upload(v)`
//!   `download!(buf)`                             → `download`
//!   `.and_then_with_context(...)`                → same name on `DeviceOpExt`
//!   `kernel(...).on_device(dev)`                 → same `.on_device(...)` eager op
//!
//! Both tests need a genuine two-device context (a real two-device platform or a
//! sub-device partition) and **skip otherwise** — same as `transfer_to_device.rs`.
//! They route to `device_at(0)` / `device_at(1)` and exercise real cross-device
//! migration; there is no single-device no-op fallback (migrating to the device a
//! buffer already lives on wouldn't test cross-device movement).

use claspr::device::Platform;
use claspr::eager::{DeviceOpExt, download, transfer_to_device, upload};
use claspr::{Context, Device};
use claspr_test_kernels::kernels;

const N: usize = 64;

/// Build a genuine two-device context: prefer a real two-device platform, then a
/// sub-device partition. Returns `None` (test skips) when no two-device context
/// is available — mirrors `transfer_to_device.rs::ctx_two_devices`.
fn ctx_two_devices() -> Option<Context> {
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
    eprintln!("SKIP: no two-device context available (real or sub-device)");
    None
}

/// transfer_to_device.rs::transfer_to_device_completes_in_chain —
/// upload → transfer to dev[1] → kernel.on_device(dev[1]) → download. The
/// transfer is a queue command (non-blocking); the chain completes without an
/// explicit `.wait()`, without hang, and the result matches.
#[test]
fn transfer_to_device_completes_in_chain() {
    let Some(ctx) = ctx_two_devices() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let result: Vec<u32> = upload(vec![5u32; N])
        .and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(1)))
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 4)
                .on_device(ec.device_at(1))
        })
        .and_then(download)
        .sync(&ctx)
        .expect("transfer + on_device chain");

    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 20));
}

/// transfer_to_device.rs::transfer_then_on_device_matches_scenario_14_shape —
/// the literal scenario-14 shape (transfer → scale → transfer → scale →
/// transfer → download). Regression test for the cross-device pipeline.
#[test]
fn transfer_then_on_device_matches_scenario_14_shape() {
    let Some(ctx) = ctx_two_devices() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let result: Vec<u32> = upload(vec![1u32; N])
        .and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(0)))
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 2)
                .on_device(ec.device_at(0))
        })
        .and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(1)))
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 10)
                .on_device(ec.device_at(1))
        })
        .and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(0)))
        .and_then(download)
        .sync(&ctx)
        .expect("scenario-14 chain");

    assert_eq!(result.len(), N);
    // 1 * 2 * 10 = 20
    assert!(result.iter().all(|&v| v == 20));
}
