//! `transfer_to_device(buf, &dev)` — explicit `DeviceSlice` migration
//! between devices in a multi-device context.
//!
//! Migration may or may not involve real data movement (depends on
//! topology — see the combinator's module docs). These tests are
//! smoke-shape: assert "doesn't hang, doesn't error, downstream sees
//! correct data", since we can't observe whether the migrate hardware-
//! moved bytes from the runtime API alone.

use claspr::device::Platform;
use claspr::{Context, Device};
use claspr_async::{DeviceOperation, download, transfer_to_device, upload};
use claspr_test_kernels::kernels;

const N: usize = 64;

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
fn transfer_to_device_completes_in_chain() {
    // upload → transfer to dev[1] → kernel.on_device(dev[1]) → download.
    // The transfer is a queue command (non-blocking); downstream
    // kernel waits on its event via the chain's deps. The whole
    // chain completes without `.wait()`, without hang, and the
    // result matches.
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

#[test]
fn transfer_then_on_device_matches_scenario_14_shape() {
    // The literal scenario-14 shape (transfer → scale → transfer →
    // scale → transfer → download). Regression test for the spike's
    // cross-device pipeline.
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
