//! Shared helpers for claspr's integration-test crates.
//!
//! Every suite in `tests/tier1` and `tests/tier2` needs the same
//! preamble: acquire a device (or skip), seed a buffer, snapshot a
//! buffer's backing-memory identity. Those helpers used to be
//! copy-pasted per test file; this crate is the single home.
//!
//! ## Skip discipline
//!
//! A test that can't get the device/feature it needs prints a line
//! starting with `SKIP:` to stderr and returns early — the suite
//! stays green, but the skip is greppable. On a machine with no
//! OpenCL ICD at all, that turns *every* test into a silent pass;
//! to forbid that in CI, set the env var `CLASPR_REQUIRE_DEVICE`
//! (any non-empty value) and [`ctx`] panics instead of skipping.
//! Feature gates (image support, SVM, a *second* device) still skip
//! under `CLASPR_REQUIRE_DEVICE` — the variable asserts that a
//! device exists, not what it can do.

use claspr::device::Platform;
use claspr::eager::DeviceOp;
use claspr::{Context, Device, DeviceSlice, MemRef, RecordableBuffer, SvmLevel};

#[cfg(feature = "ui-test")]
pub mod ui;

/// The canonical little buffer length shared by [`seeded`] and the
/// assertions around it. Small enough to eyeball in a failure dump,
/// big enough to span a few work-groups.
pub const N: usize = 64;

/// Whether `CLASPR_REQUIRE_DEVICE` is set (to any non-empty value).
fn require_device() -> bool {
    std::env::var_os("CLASPR_REQUIRE_DEVICE").is_some_and(|v| !v.is_empty())
}

/// Any usable OpenCL context, or `None` with a `SKIP:` line printed.
///
/// If `CLASPR_REQUIRE_DEVICE` is set, a missing device panics
/// instead — so CI against a real ICD can't silently report
/// all-green while executing nothing.
pub fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(err) => {
            if require_device() {
                panic!("CLASPR_REQUIRE_DEVICE is set but no OpenCL device is available: {err}");
            }
            eprintln!("SKIP: no OpenCL device ({err})");
            None
        }
    }
}

/// A single-device context with profiling on or off — for the
/// profiling-callback suites, which need both states. Same
/// no-device discipline as [`ctx`] (`SKIP:` or panic under
/// `CLASPR_REQUIRE_DEVICE`); a builder failure on a real device is
/// a hard error, not a skip.
pub fn ctx_profiling(profiling: bool) -> Option<Context> {
    let dev = match Device::any() {
        Ok(d) => d,
        Err(err) => {
            if require_device() {
                panic!("CLASPR_REQUIRE_DEVICE is set but no OpenCL device is available: {err}");
            }
            eprintln!("SKIP: no OpenCL device ({err})");
            return None;
        }
    };
    Some(
        Context::builder()
            .device(&dev)
            .profiling(profiling)
            .build()
            .expect("build context"),
    )
}

/// [`ctx`], additionally requiring image support. A device without
/// images is a plain `SKIP:` even under `CLASPR_REQUIRE_DEVICE`.
pub fn ctx_with_images() -> Option<Context> {
    let ctx = ctx()?;
    if !ctx.device().cl3().image_support().unwrap_or(false) {
        eprintln!("SKIP: device has no image support");
        return None;
    }
    Some(ctx)
}

/// [`ctx`], additionally requiring some level of SVM. A device
/// without SVM is a plain `SKIP:` even under `CLASPR_REQUIRE_DEVICE`.
pub fn ctx_with_svm() -> Option<Context> {
    let ctx = ctx()?;
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: device has no SVM");
        return None;
    }
    Some(ctx)
}

/// Returns `Some((ctx, dev_a, dev_b))` for a 2-device context, or
/// `None` with a `SKIP:` line printed.
///
/// Discovery falls back in three stages:
///
/// 1. **Real multi-device**: any platform with ≥2 devices.
/// 2. **Sub-devices**: any device that supports `CL_DEVICE_PARTITION_EQUALLY`
///    with `partition_max_sub_devices >= 2` — partitioned into two so the
///    multi-device API path still fires on single-CPU-device boxes (pocl +
///    rusticl both support this).
/// 3. **Skip.** Fewer than two devices stays a plain `SKIP:` even under
///    `CLASPR_REQUIRE_DEVICE` (the variable asserts a device exists, not
///    two) — but *zero* devices still panics through [`ctx`] when it's set.
pub fn ctx_two_devices() -> Option<(Context, Device, Device)> {
    // 1. Real multi-device: any platform with ≥2 devices. A context-build
    // failure on one platform must not silently abort discovery — say so
    // loudly and try the next platform (then the sub-device stage).
    if let Ok(platforms) = Platform::all() {
        for p in platforms {
            if let Ok(devs) = p.devices()
                && devs.len() >= 2
            {
                let dev_a = devs[0].clone();
                let dev_b = devs[1].clone();
                match Context::builder()
                    .devices(&[dev_a.clone(), dev_b.clone()])
                    .build()
                {
                    Ok(ctx) => return Some((ctx, dev_a, dev_b)),
                    Err(e) => {
                        eprintln!(
                            "SKIP candidate: two-device context build failed on platform {} \
                             ({e}); trying next",
                            p.name()
                        );
                    }
                }
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
            match Context::builder()
                .devices(&[dev_a.clone(), dev_b.clone()])
                .build()
            {
                Ok(ctx) => return Some((ctx, dev_a, dev_b)),
                Err(e) => {
                    eprintln!("SKIP candidate: sub-device context build failed ({e}); trying next");
                }
            }
        }
    }
    // 3. Skip — but a machine with NO device at all must still trip
    // CLASPR_REQUIRE_DEVICE, so probe through `ctx()` (which panics
    // then) before settling for the plain two-device skip.
    if require_device() {
        let _ = ctx();
    }
    eprintln!("SKIP: no two-device context available (real or sub-device)");
    None
}

/// Allocate + fill a `DeviceSlice<u32>` of [`N`] elements with `v`.
pub fn seeded(ctx: &Context, v: u32) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::alloc_zero(ctx, N)
        .expect("alloc")
        .fill(v)
        .wait()
        .expect("seed")
}

/// The stable identity of a buffer's backing memory: the raw `cl_mem`
/// (or SVM) pointer as a `usize`, for `==` identity comparison across
/// runs. Reads through the public `RecordableBuffer::record_handle()`
/// — works on a bare `DeviceSlice`, on images, and (via `Deref`) on a
/// live `Checkout` of any of them.
pub fn handle_of<B: RecordableBuffer>(b: &B) -> usize {
    match b.record_handle().mem {
        MemRef::Buffer(m) => m as usize,
        MemRef::Svm(p) => p as usize,
    }
}

/// Whether the graph's root homed a real finalized command buffer
/// after a sync.
pub fn homed_cb<O: DeviceOp>(g: &O) -> bool {
    g.cb_cache()
        .map(|c| c.lock().unwrap().is_some())
        .unwrap_or(false)
}
