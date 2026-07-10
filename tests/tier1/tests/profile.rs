//! `.profiled()` validation: the queue-side `CL_QUEUE_PROFILING_ENABLE`
//! check must fail cleanly at terminal time on a non-profiling queue,
//! and on a profiling-enabled queue the closure must fire with a
//! monotonic timestamp set.
//!
//! Uses the `fill_u32` kernel from `claspr-test-kernels` — small and
//! portable to avoid spirv-builder / pocl gotchas that would mask
//! runtime-side bugs we're actually trying to surface.

use claspr::eager::DeviceOpExt;
use claspr::{Context, Device, DeviceSlice};
use claspr_test_kernels::kernels;
use std::sync::{Arc, Mutex};

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
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    let result = kernels
        .fill_u32([N], buf, 7)
        .profiled(|_info| panic!("callback must not fire when profiling is off"))
        .wait();
    let err = match result {
        Ok(_) => panic!("expected ProfilingDisabled error"),
        Err(e) => e,
    };
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
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    let (tx, rx) = std::sync::mpsc::channel();
    let _buf = kernels
        .fill_u32([N], buf, 42)
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

/// #216: the INHERENT `.profiled()` on a generated kernel `Op` is REUSABLE — a
/// kernel op built once, `.profiled(cb)`, and replayed (`.sync()`'d ≥2×) re-fires
/// the callback on BOTH runs with valid `ProfilingInfo`. Before the fix the
/// generated `profile_cb` was a one-shot `Mutex<Option<FnOnce>>` drained via
/// `.take()`, so run 2 silently launched with NO callback (timing on run 1 only) —
/// inconsistent with the Tier 2 `.profiled()` combinator (#213), which re-fires.
/// Now it's an `Arc<Fn>` re-supplying a fresh shim each run. Mirrors #213's
/// `profile_combinator_replays_and_refires_callback_each_sync`, on a bare kernel op.
///
/// (`.wait()` consumes the op; the replay path is the Tier-2 borrow terminal
/// `.sync()` — a bare kernel op is a `DeviceOp`, so `.sync()` runs its single
/// enqueue and returns a re-homing `Checkout`.)
#[test]
fn inherent_profiled_replays_and_refires_callback_each_sync() {
    let Some(ctx) = ctx(true) else {
        return;
    };
    assert!(ctx.profiling());

    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    // Collect every callback firing (fires on an OpenCL callback thread → shared
    // behind an Arc<Mutex>). The `Fn` closure captures the Arc by clone.
    let seen: Arc<Mutex<Vec<claspr::ProfilingInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_cb = Arc::clone(&seen);
    // The kernel op (caller-owned `buf` → in-place fill) re-homes on Checkout drop,
    // so the same op replays. `.profiled` sets the reusable Arc<Fn> callback.
    let g = kernels.fill_u32([N], buf, 42).profiled(move |info| {
        if let Ok(info) = info {
            seen_cb.lock().unwrap().push(info);
        }
    });

    // Run 1 then re-arm; Run 2 (replay). A one-shot FnOnce would have fired only on
    // run 1 (silent no-op on run 2) — this test fails if run 2 doesn't re-fire.
    let co = g.sync(&ctx).expect("profiled kernel run 1");
    drop(co);
    let co = g.sync(&ctx).expect("profiled kernel run 2 (replay)");
    drop(co);

    // Callbacks fire on OpenCL callback threads after each run's marker completes;
    // poll until both have landed (or time out).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while seen.lock().unwrap().len() < 2 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.len(),
        2,
        "inherent .profiled() callback should fire once per sync (got {})",
        seen.len()
    );
    for (i, info) in seen.iter().enumerate() {
        assert!(
            info.queued <= info.submit && info.submit <= info.start && info.start <= info.end,
            "run {i}: non-monotonic timestamps: {info:?}",
        );
    }
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
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let buf = kernels.fill_u32([N], buf, 99).wait().expect("fill_u32");

    let mut out = vec![0u32; N];
    buf.read(&mut out).wait().expect("download");
    assert!(out.iter().all(|&x| x == 99));
}
