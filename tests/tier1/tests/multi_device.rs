//! Multi-device runtime path — the case where a single `Context`
//! spans ≥2 devices and per-device queues + buffer flow work as
//! expected.
//!
//! Discovery falls back in three stages:
//!
//! 1. **Real multi-device**: any platform with ≥2 devices.
//! 2. **Sub-devices**: any device that supports `CL_DEVICE_PARTITION_EQUALLY`
//!    with `partition_max_sub_devices >= 2` — partitioned into two
//!    so the test still exercises the multi-device API path on
//!    single-CPU-device boxes (pocl + rusticl both support this).
//! 3. **Skip**: no platform has ≥2 devices and no device can be
//!    partitioned. Tests print a SKIP and return.
//!
//! REVIEW.md item 2 called this out as a merge blocker; the
//! partition fallback makes the tests actually fire on common dev
//! environments.

use claspr::device::Platform;
use claspr::{Context, Device, DeviceSlice, InOrder, Queue};
use claspr_test_kernels::kernels;

const N: usize = 256;

/// Returns `Some((ctx, dev_a, dev_b))` for a 2-device context built
/// from either real multi-device discovery or sub-device partitioning.
/// Returns `None` (with a SKIP printed) when neither is available.
fn ctx_two_devices() -> Option<(Context, Device, Device)> {
    // 1. Real multi-device: any platform with ≥2 devices.
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
    // 2. Sub-devices: any device with PARTITION_EQUALLY + ≥ 2 CUs.
    // `partition_equally`'s arg is *CUs per sub-device*, not number of
    // sub-devices — `parent.partition_equally(cu / 2)` on a CU-count
    // ≥ 2 parent yields exactly 2 sub-devices.
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
            let dev_a = subs[0].clone();
            let dev_b = subs[1].clone();
            let ctx = Context::builder()
                .devices(&[dev_a.clone(), dev_b.clone()])
                .build()
                .ok()?;
            return Some((ctx, dev_a, dev_b));
        }
    }
    // 3. Skip.
    eprintln!(
        "SKIP: no platform with ≥2 devices and no partitionable device \
         (CL_DEVICE_PARTITION_EQUALLY with max_sub_devices ≥ 2)",
    );
    None
}

#[test]
fn context_builder_accepts_two_devices() {
    let Some((ctx, _a, _b)) = ctx_two_devices() else {
        return;
    };
    assert_eq!(ctx.devices().len(), 2);
}

#[test]
fn per_device_queues_are_independent() {
    let Some((ctx, dev_a, dev_b)) = ctx_two_devices() else {
        return;
    };
    let q_a = Queue::<InOrder>::on_device(&ctx, &dev_a).expect("queue on dev_a");
    let q_b = Queue::<InOrder>::on_device(&ctx, &dev_b).expect("queue on dev_b");
    // Queue handles must be distinct cl_command_queues.
    assert_ne!(q_a.raw().get(), q_b.raw().get());
}

#[test]
fn launch_runs_on_each_device_via_proc_macro_launcher() {
    // The macro-emitted method takes `impl Launcher`; the per-device
    // Queue impls Launcher. Same kernel, different queues, both run.
    let Some((ctx, dev_a, dev_b)) = ctx_two_devices() else {
        return;
    };
    let q_a = Queue::<InOrder>::on_device(&ctx, &dev_a).expect("queue on dev_a");
    let q_b = Queue::<InOrder>::on_device(&ctx, &dev_b).expect("queue on dev_b");
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Two independent buffers, one launch per device.
    let buf_a = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc a");
    let buf_b = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc b");
    let buf_a = kernels.fill_u32([N], buf_a, 7).wait_on(&q_a).expect("a");
    let buf_b = kernels.fill_u32([N], buf_b, 9).wait_on(&q_b).expect("b");

    let mut host_a = vec![0u32; N];
    let mut host_b = vec![0u32; N];
    buf_a.read(&mut host_a).wait_on(&q_a).expect("read a");
    buf_b.read(&mut host_b).wait_on(&q_b).expect("read b");
    assert!(host_a.iter().all(|&v| v == 7));
    assert!(host_b.iter().all(|&v| v == 9));
}
