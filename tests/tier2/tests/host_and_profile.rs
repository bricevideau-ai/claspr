//! Sub-step-3.5 (and_then_host) + sub-step-3.7 (`.profiled`)
//! coverage.

use claspr::{Context, Device, DeviceSlice};
use claspr_async::{
    DeviceOperation, DeviceOperationHostExt, DeviceOperationProfileExt, value, with_context,
};
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

// ── and_then_host ────────────────────────────────────────────────────

#[test]
fn and_then_host_sum_between_device_stages() {
    let Some(ctx) = ctx(false) else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    // upload + fill + download + (host) sum + check
    let sum: u32 = value(vec![0u32; N])
        .and_then(|host| with_context(move |ec| DeviceSlice::upload(ec, &host)))
        .and_then(|buf| {
            with_context(move |ec| {
                let kernels = kernels::kernels(ec.context())?;
                kernels.fill_u32(ec, [N], &buf, 3).wait()?;
                Ok::<_, claspr::Error>(buf)
            })
        })
        .and_then(|buf| {
            with_context(move |ec| {
                let mut out = vec![0u32; N];
                buf.download(ec, &mut out)?;
                Ok::<_, claspr::Error>(out)
            })
        })
        .and_then_host(|out| Ok(out.iter().sum::<u32>()))
        .sync(&ctx)
        .expect("and_then_host chain");
    assert_eq!(sum, 3 * N as u32);
}

#[test]
fn and_then_host_error_propagates() {
    let Some(ctx) = ctx(false) else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let err = value(())
        .and_then_host(|_| -> claspr::Result<u32> { Err(claspr::Error::SvmNotAvailable) })
        .sync(&ctx)
        .expect_err("expected error");
    assert!(matches!(err, claspr::Error::SvmNotAvailable));
}

// ── profile ──────────────────────────────────────────────────────────

#[test]
fn profile_chain_fires_callback_when_profiling_on() {
    let Some(ctx) = ctx(true) else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    assert!(ctx.profiling());

    let (tx, rx) = std::sync::mpsc::channel();
    value(vec![0u32; N])
        .and_then(|host| with_context(move |ec| DeviceSlice::upload(ec, &host)))
        .and_then(|buf| {
            with_context(move |ec| {
                let kernels = kernels::kernels(ec.context())?;
                kernels.fill_u32(ec, [N], &buf, 7).wait()?;
                Ok::<_, claspr::Error>(buf)
            })
        })
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
