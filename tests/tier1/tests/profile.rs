//! `.profiled()` validation: the queue-side `CL_QUEUE_PROFILING_ENABLE`
//! check must fail cleanly at terminal time on a non-profiling queue,
//! and on a profiling-enabled queue the closure must fire with a
//! monotonic timestamp set.
//!
//! Uses the `fill_u32` kernel from `claspr-test-kernels` — small and
//! portable to avoid spirv-builder / pocl gotchas that would mask
//! runtime-side bugs we're actually trying to surface.

use claspr::{Context, Device, DeviceSlice};
use claspr_test_kernels::kernels;

const N: usize = 1024;

/// Convenience: build a single-device context with profiling on/off.
/// Returns `None` (with an eprintln SKIP) if there's no OpenCL device.
fn ctx(profiling: bool) -> Option<Context> {
    let dev = match Device::any() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
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

#[test]
fn errors_when_queue_lacks_profiling() {
    let Some(ctx) = ctx(false) else {
        return;
    };
    assert!(!ctx.profiling());

    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");

    let err = kernels
        .fill_u32(&ctx, [N], &buf, 7)
        .profiled(|_info| panic!("callback must not fire when profiling is off"))
        .wait()
        .expect_err("expected ProfilingDisabled");
    assert!(
        matches!(err, claspr::Error::ProfilingDisabled),
        "got {err:?}"
    );
}

#[test]
fn delivers_monotonic_timestamps_when_enabled() {
    let Some(ctx) = ctx(true) else {
        return;
    };
    assert!(ctx.profiling());

    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");

    let (tx, rx) = std::sync::mpsc::channel();
    kernels
        .fill_u32(&ctx, [N], &buf, 42)
        .profiled(move |info| {
            tx.send(info).expect("send profiling info");
        })
        .wait()
        .expect("wait");

    let info = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("callback fired")
        .expect("profiling info Ok");
    assert!(
        info.queued <= info.submit && info.submit <= info.start && info.start <= info.end,
        "non-monotonic timestamps: {info:?}",
    );
}

#[test]
fn fill_then_download_round_trip() {
    // Sanity for the test-kernel library itself: fill_u32 + download
    // gives back the expected pattern. Catches regressions in the
    // tests/kernels crate before they look like LaunchOp bugs.
    let Some(ctx) = ctx(false) else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");
    kernels
        .fill_u32(&ctx, [N], &buf, 99)
        .wait()
        .expect("fill_u32");

    let mut out = vec![0u32; N];
    buf.read(&ctx, &mut out).wait().expect("download");
    assert!(out.iter().all(|&x| x == 99));
}
