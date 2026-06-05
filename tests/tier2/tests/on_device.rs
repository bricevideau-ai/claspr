//! `.on_device(&dev)` — per-op device routing for Tier 2 chains.
//!
//! These tests pull device handles from `ec.context().devices()`
//! (via the `ec.device_at(i)` shortcut) inside
//! `.and_then_with_context` closures — the idiom the spike + real
//! user code should use. No external Device captures.
//!
//! Skips when only one device is available (no sub-device partition).

use claspr::device::Platform;
use claspr::{Context, Device, Error};
use claspr_async::{DeviceOperation, DeviceOperationHostExt, bundle, download, upload};
use claspr_test_kernels::kernels;

const N: usize = 64;

/// Three-stage discovery, matching tier1/tests/multi_device.rs and
/// the helper in cross_device.rs: real multi-device → sub-device
/// partition → skip.
fn ctx_two_devices() -> Option<Context> {
    if let Ok(platforms) = Platform::all() {
        for p in platforms {
            if let Ok(devs) = p.devices()
                && devs.len() >= 2
            {
                let ctx = Context::builder()
                    .devices(&[devs[0].clone(), devs[1].clone()])
                    .build()
                    .ok()?;
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
            let ctx = Context::builder()
                .devices(&[subs[0].clone(), subs[1].clone()])
                .build()
                .ok()?;
            return Some(ctx);
        }
    }
    eprintln!("SKIP: no two-device context available (real or sub-device)");
    None
}

#[test]
fn on_device_routes_chain_to_devices_from_context() {
    // Two scale stages, one per device, plus a final download. The
    // chain is fully non-blocking — no `.wait()` anywhere, no
    // `with_context` ceremony. Device identity resolved from `ec`
    // each stage.
    let Some(ctx) = ctx_two_devices() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let result: Vec<u32> = upload!(vec![1u32; N])
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 3)
                .on_device(ec.device_at(0))
        })
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 4)
                .on_device(ec.device_at(1))
        })
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("on_device chain");

    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 12));
}

#[test]
fn on_device_preserves_host_error_slot_across_routing() {
    // An `.and_then_host` failure inside a chain routed through
    // `.on_device(&dev_b)` should still surface the rich variant at
    // the terminal — proving the child EC built by OnDevice shares
    // the parent's host-error slot.
    let Some(ctx) = ctx_two_devices() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let chain = upload!(vec![1u32; N])
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 2)
                .on_device(ec.device_at(1))
        })
        // DeviceSlice's Mappable view is `&mut [T]`. Closure returns
        // Err — the rich variant must survive across the routed
        // child EC's host-error slot back to the chain terminal.
        .and_then_host(|_view: &mut [u32]| -> claspr::Result<()> {
            Err(Error::Build {
                log: "routed-chain abort".to_string(),
            })
        });

    let err = chain.sync(&ctx).expect_err("expected branch B error");
    assert!(
        matches!(&err, Error::Build { log } if log == "routed-chain abort"),
        "got {err:?}",
    );
}

#[test]
fn on_device_bundle_runs_branches_on_distinct_devices() {
    // A bundle whose two branches are routed to different devices;
    // both should run, and the chain should produce both outputs.
    // Smoke test for "multi-device branches don't hang" (which would
    // happen without the terminal flush_all_outoforder_queues fix —
    // dev_b's queue would never get pushed under rusticl).
    let Some(ctx) = ctx_two_devices() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let (a, b): (Vec<u32>, Vec<u32>) = bundle!(
        upload!(vec![1u32; N])
            .and_then_with_context(move |ec, buf| {
                kernels_ref
                    .scale_u32([N], buf, 7)
                    .on_device(ec.device_at(0))
            })
            .and_then(|buf| download!(buf)),
        upload!(vec![1u32; N])
            .and_then_with_context(move |ec, buf| {
                kernels_ref
                    .scale_u32([N], buf, 11)
                    .on_device(ec.device_at(1))
            })
            .and_then(|buf| download!(buf)),
    )
    .sync(&ctx)
    .expect("bundle across devices");

    assert!(a.iter().all(|&v| v == 7));
    assert!(b.iter().all(|&v| v == 11));
}
