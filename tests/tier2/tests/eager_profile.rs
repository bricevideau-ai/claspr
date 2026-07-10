//! Eager-API port of `host_and_profile.rs`'s `.profiled` cases:
//!   - callback fires with `Ok(ProfilingInfo)` when the queue has profiling on,
//!   - `.profiled` returns `Err(ProfilingDisabled)` when profiling is off.
//!
//! Old → new mapping:
//!   `value(v).and_then(|x| upload!(x))` → `upload(v)`
//!   `.profiled(cb)`                     → `DeviceProfileExt::profiled`

use claspr::eager::{DeviceOpExt, DeviceProfileExt, upload, value};
use claspr::{Context, Device};
use claspr_test_kernels::kernels;
use std::sync::{Arc, Mutex};

const N: usize = 128;

fn ctx(profiling: bool) -> Option<Context> {
    let dev = Device::any().ok()?;
    Context::builder()
        .device(&dev)
        .profiling(profiling)
        .build()
        .ok()
}

#[test]
fn profile_chain_fires_callback_when_profiling_on() {
    let Some(ctx) = ctx(true) else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    assert!(ctx.profiling());

    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let (tx, rx) = std::sync::mpsc::channel();
    let _checkout = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 7))
        .profiled(move |info| {
            tx.send(info).expect("send profiling info");
        })
        .sync(&ctx)
        .expect("profiled chain");

    let info = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("callback fired")
        .expect("profiling Ok");
    assert!(
        info.queued <= info.submit && info.submit <= info.start && info.start <= info.end,
        "non-monotonic timestamps: {info:?}",
    );
}

/// #213: the `.profiled()` COMBINATOR ([`DeviceProfileExt::profiled`], which wraps
/// a multi-stage sub-chain — distinct from the inherent one-shot `.profiled()` on a
/// single kernel `Op`) is REUSABLE — `sync`'d twice, the `Fn` callback fires on
/// BOTH runs with valid profiling data. Before the fix the combinator's callback
/// was a one-shot `FnOnce` drained via `Mutex<Option<_>>::take()`, so the 2nd
/// `sync` errored "callback already consumed". Mirrors
/// `and_then_host_replays_and_reruns_each_sync`: build once, `sync` ≥2×, assert the
/// callback fired each time (an `Arc<Mutex<Vec<_>>>` collector) with monotonic
/// timestamps both runs.
///
/// The source is `upload(..).and_then(kernel)` — an [`AndThen`], which has no
/// inherent `.profiled()`, so `.profiled()` resolves to the `DeviceProfileExt`
/// combinator (the subject of #213). `upload` re-seeds its buffer each run and
/// rehomes it, so the graph replays over stable handles.
#[test]
fn profile_combinator_replays_and_refires_callback_each_sync() {
    let Some(ctx) = ctx(true) else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    assert!(ctx.profiling());

    let kernels = kernels::kernels(&ctx).expect("load kernels");
    // Collect every callback firing (fires on an OpenCL callback thread, so shared
    // behind an Arc<Mutex>). The `Fn` closure captures the Arc by clone — the right
    // shape for something replayed.
    let seen: Arc<Mutex<Vec<claspr::ProfilingInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_cb = Arc::clone(&seen);
    let g = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 7))
        .profiled(move |info| {
            if let Ok(info) = info {
                seen_cb.lock().unwrap().push(info);
            }
        });

    // Run 1 then re-arm; Run 2 (replay). A one-shot FnOnce would have errored on
    // run 2 with "callback already consumed" instead.
    let out = g.sync(&ctx).expect("profiled combinator run 1");
    drop(out);
    let out = g.sync(&ctx).expect("profiled combinator run 2 (replay)");
    drop(out);

    // The callbacks fire on OpenCL callback threads after each run's marker
    // completes; poll until both have landed (or time out).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while seen.lock().unwrap().len() < 2 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.len(),
        2,
        "combinator callback should fire once per sync (got {})",
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
fn profile_chain_errors_when_profiling_off() {
    let Some(ctx) = ctx(false) else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    assert!(!ctx.profiling());

    let err = value(())
        .profiled(|_| panic!("callback must not fire when profiling is off"))
        .sync(&ctx)
        .expect_err("expected ProfilingDisabled");
    assert!(matches!(err, claspr::Error::ProfilingDisabled), "{err:?}");
}
