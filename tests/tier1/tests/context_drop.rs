//! Leak-regression tests for the `Context` / default-queue Arc cycle.
//!
//! Background: `Context` is `Arc<ContextInner>`; `ContextInner` used to
//! store its per-device DEFAULT queues as `Queue` wrappers, and each
//! `Queue` is `Arc<QueueInner>` with a STRONG `ctx: Context` back-edge.
//! That formed an Arc cycle from birth (the in-order default is built
//! eagerly), so `ContextInner` never reached strong-count 0, opencl3's
//! `clReleaseContext` / `clReleaseCommandQueue` never ran, and every
//! `cl_context` + default `cl_command_queue` leaked (Intel cliloader
//! --leak-checking: 16 cl_context allocs, 0 releases).
//!
//! The fix stores the defaults as RAW handles (no `Queue`, no `ctx`
//! back-edge); user queues keep their strong `ctx`. These two tests pin
//! both directions of that invariant WITHOUT cliloader — purely via the
//! Rust `Arc` graph (a `Weak` to `ContextInner` upgrades iff the inner
//! is still alive, i.e. iff its `Drop` — which fires the OpenCL releases
//! — has NOT yet run).

use claspr::{Context, Device, InOrder, Queue};

fn dev() -> Option<Device> {
    match Device::any() {
        Ok(d) => Some(d),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

/// Cycle-broken: touch BOTH default queue paths, drop the owned handles
/// they return, drop the `Context` — and the `ContextInner` must be
/// gone (`Weak` can't upgrade), proving the release path fired and no
/// Arc cycle pins it alive.
#[test]
fn default_queues_do_not_pin_context() {
    let Some(dev) = dev() else { return };
    let ctx = Context::for_device(&dev).expect("build context");

    // Exercise both default-queue accessors; both hand back OWNED
    // wrappers that strong-hold `ctx`. Dropping them here must NOT leave
    // a dangling strong ref inside `ContextInner`.
    {
        let _in = ctx
            .default_inorder_queue(&dev)
            .expect("default in-order queue");
        let _oo = ctx
            .default_outoforder_queue(&dev)
            .expect("default out-of-order queue");
        // _in, _oo drop here.
    }

    // The only strong `Context` is `ctx`; downgrade and drop it.
    let weak = ctx.__test_weak();
    assert_eq!(
        ctx.__test_strong_count(),
        1,
        "no extra strong ContextInner refs should survive after the \
         default-queue wrappers dropped (cycle would show >1)"
    );
    assert_eq!(ctx.error_count(), 0, "no release errors expected");
    drop(ctx);

    assert!(
        Context::__test_weak_is_dead(weak.as_ref()),
        "ContextInner must be dropped once the last Context handle is \
         gone — a surviving Arc cycle (default queue strong-holding ctx) \
         is exactly the leak this test guards against"
    );
}

/// User-queue-outlives-context: a `Queue::new` user queue KEEPS the
/// context alive (it strong-holds `ctx`), so the `Weak` still upgrades
/// after the `Context` handle drops; the queue stays usable; and only
/// once the queue ALSO drops does `ContextInner` finally release.
#[test]
fn user_queue_outlives_its_context() {
    let Some(dev) = dev() else { return };
    let ctx = Context::for_device(&dev).expect("build context");

    let q = Queue::<InOrder>::new(&ctx).expect("build user queue");
    let weak = ctx.__test_weak();
    drop(ctx);

    assert!(
        !Context::__test_weak_is_dead(weak.as_ref()),
        "a live user Queue must keep its ContextInner alive after the \
         Context handle drops (user queues intentionally strong-hold ctx)"
    );

    // The queue is still fully functional against its retained context.
    q.finish()
        .expect("user queue still usable after Context dropped");

    drop(q);
    assert!(
        Context::__test_weak_is_dead(weak.as_ref()),
        "once the last user Queue drops, ContextInner must release \
         (cl_context + queues freed)"
    );
}
