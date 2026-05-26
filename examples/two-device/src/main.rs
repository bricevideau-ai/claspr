//! Multi-device claspr example: demonstrates [`Context::for_devices`],
//! [`Queue::on_device`], and same-context cross-buffer copy
//! ([`DeviceSlice::copy_to`]) without depending on kernel
//! compilation.
//!
//! Picks two devices if available (preferring two physical devices,
//! else two sub-devices via `partition_equally`, else falls back to
//! two queues on the same single device). For each queue, allocates
//! a buffer, uploads host data, copies one buffer's contents into a
//! third buffer through the other queue, then verifies the
//! round-trip.
//!
//! No `#[claspr::device]` module / no kernel — staged this way
//! because the surrounding multi-device API is what this stage of
//! the runtime is meant to prove out. A kernel-launching multi-
//! device example will land alongside the real two-physical-device
//! testbed.

use claspr::{Buffer, Context, Device, DeviceSlice, InOrder, Queue};

const N: usize = 64;

/// Pick a [`DeviceConfig`] — two devices if the system offers
/// them, sub-devices if the lone device can be partitioned,
/// otherwise just one device that hosts both queues.
enum DeviceConfig {
    Two([Device; 2]),
    OneShared(Device),
}

fn pick_devices() -> claspr::Result<Option<DeviceConfig>> {
    let all = Device::all()?;
    if all.is_empty() {
        return Ok(None);
    }
    if all.len() >= 2 {
        println!(
            "two-device: using two physical devices ({}, {})",
            all[0].name()?,
            all[1].name()?,
        );
        return Ok(Some(DeviceConfig::Two([all[0].clone(), all[1].clone()])));
    }
    // Try sub-device partitioning. `partition_equally`'s argument
    // is "compute units per sub-device" — for exactly 2 sub-devices
    // from an N-CU parent, ask for N/2.
    let parent = &all[0];
    let cu = parent.max_compute_units()?;
    if cu >= 2
        && let Ok(parts) = parent.partition_equally(cu / 2)
        && parts.len() >= 2
    {
        println!(
            "two-device: partitioned `{}` into two halves of {} CUs each",
            parent.name()?,
            cu / 2
        );
        return Ok(Some(DeviceConfig::Two([
            parts[0].clone(),
            parts[1].clone(),
        ])));
    }
    println!(
        "two-device: only one device (`{}`), no usable partition — running both queues on it",
        parent.name()?,
    );
    Ok(Some(DeviceConfig::OneShared(parent.clone())))
}

fn run() -> claspr::Result<bool> {
    let cfg = match pick_devices()? {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no OpenCL device");
            return Ok(false);
        }
    };

    let (ctx, q0, q1, devices_used) = match cfg {
        DeviceConfig::Two([d0, d1]) => {
            let ctx = Context::for_devices(&[d0.clone(), d1.clone()])?;
            let q0 = Queue::<InOrder>::on_device(&ctx, &d0)?;
            let q1 = Queue::<InOrder>::on_device(&ctx, &d1)?;
            (ctx, q0, q1, 2)
        }
        DeviceConfig::OneShared(d) => {
            let ctx = Context::for_device(&d)?;
            let q0 = Queue::<InOrder>::new(&ctx)?;
            let q1 = Queue::<InOrder>::new(&ctx)?;
            (ctx, q0, q1, 1)
        }
    };

    // Stage 1: two halves uploaded to two queues.
    let inputs: Vec<u32> = (1..=N as u32).collect();
    let half = N / 2;
    let buf0 = DeviceSlice::upload(&q0, &inputs[..half])?;
    let buf1 = DeviceSlice::upload(&q1, &inputs[half..])?;

    // Stage 2: cross-buffer copy through q1 — allocate a fresh
    // buffer on the shared context, copy buf0's data into it
    // (queued on q1), wait. Exercises the
    // `DeviceSlice::copy_to` path within a possibly-multi-device
    // context.
    let mut mirror = DeviceSlice::alloc(&ctx, half)?;
    buf0.copy_to(&mut mirror, &q1).wait()?;

    // Stage 3: download back via the respective queues and verify.
    let mut out0 = vec![0u32; half];
    let mut out1 = vec![0u32; N - half];
    let mut mirror_out = vec![0u32; half];
    buf0.read(&q0, &mut out0).wait()?;
    buf1.read(&q1, &mut out1).wait()?;
    mirror.read(&q1, &mut mirror_out).wait()?;

    assert_eq!(out0, inputs[..half], "buf0 round-trip mismatch");
    assert_eq!(out1, inputs[half..], "buf1 round-trip mismatch");
    assert_eq!(mirror_out, inputs[..half], "cross-buffer copy mismatch");

    println!(
        "two-device: {} elements across {} device(s) via {} queue(s); buf0 len={}, buf1 len={}, cross-copy verified",
        N,
        devices_used,
        2,
        buf0.len(),
        buf1.len(),
    );
    Ok(true)
}

fn main() -> claspr::Result<()> {
    if !run()? {
        std::process::exit(0);
    }
    Ok(())
}
