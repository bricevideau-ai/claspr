//! Eager-API port of `host_and_profile.rs`'s `.profiled` cases:
//!   - callback fires with `Ok(ProfilingInfo)` when the queue has profiling on,
//!   - `.profiled` returns `Err(ProfilingDisabled)` when profiling is off.
//!
//! Old → new mapping:
//!   `value(v).and_then(|x| upload!(x))` → `upload::<u32, ReadWrite, _>(v)`
//!   `.profiled(cb)`                     → `EagerProfileExt::profiled`

use claspr::eager::{EagerOpExt, EagerProfileExt, upload, value};
use claspr::{Context, Device};
use claspr_test_kernels::kernels;

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
    upload::<u32, claspr::ReadWrite, _>(vec![0u32; N])
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
