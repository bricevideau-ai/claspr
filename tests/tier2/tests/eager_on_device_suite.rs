//! Eager port of `on_device.rs`: `.on_device(&dev)` per-op device routing for
//! eager graph chains. Device handles are pulled from `ec.device_at(i)` inside
//! `.and_then_with_context` closures — the portable idiom (no external Device
//! captures). Mirrors `eager_cutover::eager_on_device`.
//!
//! Named `eager_on_device_suite` to avoid a file-stem clash with the existing
//! `eager_cutover::eager_on_device` test fn / harness expectations.
//!
//! Old → new mapping:
//!   `upload!(v)`                  → `upload(v)`
//!   `download!(buf)`              → `download`
//!   `.and_then_with_context(...)` → same name on `DeviceOpExt`
//!   `kernel(...).on_device(dev)`  → same `.on_device(...)` on the eager kernel op
//!   `bundle!(a, b)`              → `bundle2(a, b)`
//!
//! NOTE (known eager seam): `and_then_with_context` passes the upstream VALUE,
//! not a pipe, so a routed downstream enqueues without an explicit event edge to
//! the source — terminal completion is correct; mid-chain OOO ordering relies on
//! the driver. None of these tests assert cross-device event ORDERING (they
//! assert final values / error surfacing / both-branches-ran), so the port is
//! faithful.
//!
//! Skips when only one device is available (no real multi-device platform AND no
//! sub-device partition). Guard copied verbatim from on_device.rs.

use claspr::device::Platform;
use claspr::eager::{DeviceOpExt, bundle2, download, upload};
use claspr::{Context, Device, Error};
use claspr_test_kernels::kernels;

const N: usize = 64;

/// Three-stage discovery: real multi-device → sub-device partition → skip.
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

/// on_device.rs::on_device_routes_chain_to_devices_from_context — two scale
/// stages, one per device, plus a final download. Device identity resolved from
/// `ec` each stage. 1 × 3 × 4 = 12.
#[test]
fn on_device_routes_chain_to_devices_from_context() {
    let Some(ctx) = ctx_two_devices() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let result: Vec<u32> = upload(vec![1u32; N])
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
        .and_then(download)
        .sync(&ctx)
        .expect("on_device chain");

    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 12));
}

/// on_device.rs::on_device_preserves_host_error_slot_across_routing — an
/// `.and_then_host` failure inside a routed chain must still surface the rich
/// variant at the terminal, proving the routed child EC shares the parent's
/// host-error slot.
#[test]
fn on_device_preserves_host_error_slot_across_routing() {
    let Some(ctx) = ctx_two_devices() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let chain = upload(vec![1u32; N])
        .and_then_with_context(move |ec, buf| {
            kernels_ref
                .scale_u32([N], buf, 2)
                .on_device(ec.device_at(1))
        })
        // DeviceSlice's Mappable view is `&mut [T]`. Closure returns Err — the
        // rich variant must survive across the routed child EC's host-error slot
        // back to the chain terminal.
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

/// on_device.rs::on_device_bundle_runs_branches_on_distinct_devices — a bundle
/// whose two branches are routed to different devices; both run, both outputs
/// produced. Smoke test for "multi-device branches don't hang".
#[test]
fn on_device_bundle_runs_branches_on_distinct_devices() {
    let Some(ctx) = ctx_two_devices() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let (a, b): (Vec<u32>, Vec<u32>) = bundle2(
        upload(vec![1u32; N])
            .and_then_with_context(move |ec, buf| {
                kernels_ref
                    .scale_u32([N], buf, 7)
                    .on_device(ec.device_at(0))
            })
            .and_then(download),
        upload(vec![1u32; N])
            .and_then_with_context(move |ec, buf| {
                kernels_ref
                    .scale_u32([N], buf, 11)
                    .on_device(ec.device_at(1))
            })
            .and_then(download),
    )
    .sync(&ctx)
    .expect("bundle across devices");

    assert!(a.iter().all(|&v| v == 7));
    assert!(b.iter().all(|&v| v == 11));
}
