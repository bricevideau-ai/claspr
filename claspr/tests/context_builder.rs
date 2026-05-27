//! Phase 2 tests: [`Context::builder`], per-device default queues,
//! profiling round-trip.
//!
//! Skip-on-no-device pattern matches the rest of claspr's test suite —
//! every test `return Ok(())`s with an `eprintln!` if there's no OpenCL
//! runtime to talk to.

use claspr::{Context, Device, Queue};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn any_ctx() -> TestResult {
    if Context::any().is_err() {
        eprintln!("SKIP: no OpenCL device");
        return Ok(());
    }
    Ok(())
}

#[test]
fn empty_builder_fails() {
    // No skip needed — this never touches the runtime.
    let res = Context::builder().build();
    assert!(res.is_err(), "expected error when no devices selected");
}

#[test]
fn builder_with_one_device_matches_for_device() -> TestResult {
    if any_ctx().is_err() {
        return Ok(());
    }
    let Ok(dev) = Device::any() else {
        eprintln!("SKIP: no OpenCL device");
        return Ok(());
    };

    let via_builder = Context::builder().device(&dev).build()?;
    let via_shortcut = Context::for_device(&dev)?;

    assert_eq!(via_builder.devices().len(), 1);
    assert_eq!(via_shortcut.devices().len(), 1);
    assert_eq!(via_builder.device().raw_id(), dev.raw_id());
    assert_eq!(via_shortcut.device().raw_id(), dev.raw_id());
    assert!(!via_builder.profiling());
    assert!(!via_shortcut.profiling());
    Ok(())
}

#[test]
fn profiling_round_trips_through_default_queues() -> TestResult {
    let Ok(dev) = Device::any() else {
        eprintln!("SKIP: no OpenCL device");
        return Ok(());
    };

    let ctx_off = Context::builder().device(&dev).profiling(false).build()?;
    let ctx_on = Context::builder().device(&dev).profiling(true).build()?;
    assert!(!ctx_off.profiling());
    assert!(ctx_on.profiling());

    // The default in-order queue exists in both. Profiling only
    // affects the bitmask used to create it; we can't easily inspect
    // that, but we can verify queue construction succeeded.
    let _ = ctx_off.default_inorder_queue(&dev)?;
    let _ = ctx_on.default_inorder_queue(&dev)?;

    // An out-of-order queue created via Queue::new should also inherit
    // profiling from the context.
    let _q_off = Queue::<claspr::OutOfOrder>::new(&ctx_off)?;
    let _q_on = Queue::<claspr::OutOfOrder>::new(&ctx_on)?;
    Ok(())
}

#[test]
fn default_inorder_queue_returns_stable_reference() -> TestResult {
    let Ok(dev) = Device::any() else {
        eprintln!("SKIP: no OpenCL device");
        return Ok(());
    };
    let ctx = Context::builder().device(&dev).build()?;
    let a = ctx.default_inorder_queue(&dev)? as *const _;
    let b = ctx.default_inorder_queue(&dev)? as *const _;
    let c = ctx.default_inorder_queue(&dev)? as *const _;
    assert_eq!(a, b);
    assert_eq!(b, c);
    Ok(())
}

#[test]
fn default_outoforder_queue_is_lazy_and_stable() -> TestResult {
    let Ok(dev) = Device::any() else {
        eprintln!("SKIP: no OpenCL device");
        return Ok(());
    };
    let ctx = Context::builder().device(&dev).build()?;
    // First call creates; subsequent calls return Arc clones of the
    // same underlying Queue (until an invalidate is triggered).
    let a = std::sync::Arc::as_ptr(&ctx.default_outoforder_queue(&dev)?);
    let b = std::sync::Arc::as_ptr(&ctx.default_outoforder_queue(&dev)?);
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn default_outoforder_queue_rebuilds_after_invalidate() -> TestResult {
    let Ok(dev) = Device::any() else {
        eprintln!("SKIP: no OpenCL device");
        return Ok(());
    };
    let ctx = Context::builder().device(&dev).build()?;
    // Hold a clone of the first Arc so the underlying Queue can't be
    // freed by the invalidate (which would let the allocator reuse
    // the address and obscure the test).
    let first = ctx.default_outoforder_queue(&dev)?;
    let a = std::sync::Arc::as_ptr(&first);
    ctx.invalidate_default_outoforder_queue(&dev);
    let second = ctx.default_outoforder_queue(&dev)?;
    let b = std::sync::Arc::as_ptr(&second);
    assert_ne!(a, b, "invalidate should force a fresh Queue allocation");
    Ok(())
}

#[test]
fn foreign_device_yields_invalid_argument() -> TestResult {
    let Ok(devs) = Device::all() else {
        eprintln!("SKIP: no OpenCL devices");
        return Ok(());
    };
    if devs.len() < 2 {
        eprintln!("SKIP: need >= 2 devices to ask about a foreign one");
        return Ok(());
    }
    let ctx = Context::builder().device(&devs[0]).build()?;
    // devs[1] is a real device, just not part of this context.
    let err = ctx.default_inorder_queue(&devs[1]).unwrap_err();
    assert!(
        matches!(err, claspr::Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
    Ok(())
}

#[test]
fn multi_device_each_gets_its_own_queue() -> TestResult {
    let Ok(devs) = Device::all() else {
        eprintln!("SKIP: no OpenCL devices");
        return Ok(());
    };
    if devs.len() < 2 {
        eprintln!("SKIP: need >= 2 devices");
        return Ok(());
    }
    // Filter to one platform — OpenCL requires a context's devices to
    // share a platform. Take the first device's platform and keep
    // anyone matching it.
    let plat_id = devs[0].platform().raw_id();
    let same_plat: Vec<Device> = devs
        .into_iter()
        .filter(|d| d.platform().raw_id() == plat_id)
        .collect();
    if same_plat.len() < 2 {
        eprintln!("SKIP: need >= 2 devices on the same platform");
        return Ok(());
    }

    let ctx = Context::builder().devices(&same_plat[..2]).build()?;
    assert_eq!(ctx.devices().len(), 2);
    let q0 = ctx.default_inorder_queue(&same_plat[0])? as *const _;
    let q1 = ctx.default_inorder_queue(&same_plat[1])? as *const _;
    assert_ne!(q0, q1, "per-device queues must be distinct");
    Ok(())
}

#[test]
fn launcher_routes_to_devices_zero_default_queue() -> TestResult {
    let Ok(dev) = Device::any() else {
        eprintln!("SKIP: no OpenCL device");
        return Ok(());
    };
    let ctx = Context::builder().device(&dev).build()?;
    use claspr::Launcher;
    // The default in-order queue for devices[0] is eagerly populated
    // at build time and is what `Launcher::cl_queue(&ctx)` returns.
    let via_launcher = ctx.cl_queue() as *const _;
    let via_default = ctx.default_inorder_queue(&dev)?.raw() as *const _;
    assert_eq!(via_launcher, via_default);
    Ok(())
}
