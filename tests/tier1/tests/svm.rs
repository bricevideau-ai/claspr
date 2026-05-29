//! MappedSlice round-trip + cross-queue Drop ordering.
//!
//! Tests both the basic map/map_mut RAII pattern AND the
//! `register_use` cross-queue Drop fix. The Drop test deliberately
//! schedules work on a separate command queue, pushes that work's
//! event onto the MappedSlice's in-flight-use list, then lets
//! MappedSlice drop — its `clEnqueueSVMFree` must queue-order
//! behind the cross-queue work via the event wait-list instead of
//! deadlocking or freeing while the unmap is still in flight.

use claspr::{Context, Device, InOrder, Launcher, MappedSlice, OutOfOrder, Queue, SvmLevel};
use claspr_test_kernels::kernels;
use std::sync::Arc;

const N: usize = 256;

fn ctx_with_svm() -> Option<Context> {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return None;
    };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: device has no SVM");
        return None;
    }
    Some(ctx)
}

#[test]
fn map_mut_then_map_round_trip() {
    let Some(ctx) = ctx_with_svm() else { return };

    let mut buf = MappedSlice::<u32>::alloc(&ctx, N).expect("alloc");
    {
        let mut view = buf.map_mut(&ctx).expect("map_mut");
        for (i, slot) in view.iter_mut().enumerate() {
            *slot = i as u32;
        }
    } // unmap on Drop

    let view = buf.map(&ctx).expect("map");
    for (i, &v) in view.iter().enumerate() {
        assert_eq!(v, i as u32);
    }
}

#[test]
fn drop_orders_after_cross_queue_unmap_via_last_use() {
    // The whole point of #94: MappedSlice can be used on a queue
    // distinct from its Context's default in-order queue, and Drop's
    // clEnqueueSVMFree must wait for that cross-queue work to finish
    // before reclaiming the allocation.
    //
    // We simulate the path MappedSliceHostView takes: enqueue an
    // unmap on a separate queue, wrap the event in Arc<Event>, set
    // it as last_use, then drop. The drop's free queue-orders on the
    // unmap event via wait-list — no host-side wait, no UB.
    let Some(ctx) = ctx_with_svm() else { return };

    let device: Device = ctx.device().clone();
    let other_queue = Queue::<InOrder>::on_device(&ctx, &device).expect("aux queue");

    let mut buf = MappedSlice::<u32>::alloc(&ctx, N).expect("alloc");
    // Write something so the unmap has data to flush.
    {
        let mut view = buf.map_mut(&other_queue).expect("map_mut on aux");
        for slot in view.iter_mut() {
            *slot = 99;
        }
    } // unmap-and-set-last_use happens inside MappedWriteGuard::Drop

    // Sanity: at Drop time, last_use may be set. Drop's free uses it
    // as a wait-list entry, so this drop should not block the host
    // and should not panic on use-after-free in the runtime.
    drop(buf);
    // The Context's sticky-error counter would catch a CL error.
    assert_eq!(
        ctx.error_count(),
        0,
        "Drop free should not have recorded any errors"
    );
}

#[test]
fn explicit_register_use_orders_drop_after_cross_queue_event() {
    // Demonstrate the public register_use API directly: enqueue work
    // on a separate queue, manually feed the event back into the
    // MappedSlice, drop. Same expected outcome as above but via the
    // user-facing API path (not the host_view internal helper).
    let Some(ctx) = ctx_with_svm() else { return };
    let device: Device = ctx.device().clone();
    let other_queue = Queue::<InOrder>::on_device(&ctx, &device).expect("aux queue");

    let buf = MappedSlice::<u32>::alloc(&ctx, N).expect("alloc");
    // Issue a marker on the aux queue to manufacture an event.
    // SAFETY: empty wait-list is always valid.
    let marker =
        unsafe { other_queue.raw().enqueue_marker_with_wait_list(&[]) }.expect("aux marker");
    buf.register_use(Arc::new(marker));

    drop(buf);
    assert_eq!(
        ctx.error_count(),
        0,
        "Drop with explicit register_use should not have recorded any errors"
    );
}

#[test]
fn kernel_launches_on_ooo_queue_register_themselves_for_drop() {
    // The auto-registration path: kernel launches that take a
    // MappedSlice as a KernelArg should record their completion event
    // on the buffer's in-flight-use list (via
    // `KernelArg::register_completion`). Drop's `clEnqueueSVMFree`
    // wait-list then includes every launch, so the free queue-orders
    // after all of them even when they pile up concurrently on an OOO
    // queue.
    //
    // No explicit sync between the launches and the drop — if the
    // wiring is wrong, the free races with in-flight launches and
    // either deadlocks the queue or trips the runtime. Either way the
    // context's sticky-error counter or a hang would surface it.
    let Some(ctx) = ctx_with_svm() else { return };
    let device: Device = ctx.device().clone();
    let ooo = Queue::<OutOfOrder>::on_device(&ctx, &device).expect("ooo queue");
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let mut buf = MappedSlice::<u32>::alloc(&ctx, N).expect("alloc");
    // Issue several launches on the OOO queue, all consuming `&buf`
    // via the typed launcher (now generic over KernelSliceArg, so
    // `MappedSlice<u32>` flows through `kernels.fill_u32` directly).
    // On an OOO queue, the events may finish in any order — without
    // Vec accumulation, only the most recently registered would gate
    // the free, which would race the others.
    for value in 0..4u32 {
        let (returned, event) = kernels
            .fill_u32([N], buf, value)
            .submit(&ooo)
            .expect("submit");
        buf = returned;
        // Hold each Event briefly; auto-register has already happened
        // via the MappedSlice KernelArg impl.
        drop(event);
    }
    // Drop the buffer WITHOUT explicit sync. The Vec inside `buf` has
    // 4 events queued from the launches; clEnqueueSVMFree's wait_list
    // gates the free behind all of them.
    drop(buf);
    // Drain the queue so any auto-recorded errors land in the context.
    ooo.raw().finish().expect("finish ooo");
    ctx.cl_queue().finish().expect("finish ctx default");
    assert_eq!(ctx.error_count(), 0);
}

#[test]
fn alloc_then_immediate_drop_uses_empty_wait_list() {
    // The simplest case: nothing in flight when Drop fires →
    // `clEnqueueSVMFree` runs with an empty wait_list. Exercises the
    // None/empty arm of the Vec-drain path.
    let Some(ctx) = ctx_with_svm() else { return };
    {
        let _buf = MappedSlice::<u32>::alloc(&ctx, N).expect("alloc");
    } // immediate drop, no use registered
    ctx.cl_queue().finish().expect("finish");
    assert_eq!(ctx.error_count(), 0);
}

#[test]
fn read_only_map_via_map_guard() {
    // Existing tests use map_mut. This exercises the read-only path
    // (`map(launcher)` → MappedReadGuard derefs to &[T]).
    let Some(ctx) = ctx_with_svm() else { return };
    let mut buf = MappedSlice::<u32>::alloc(&ctx, N).expect("alloc");
    {
        // Populate via map_mut...
        let mut g = buf.map_mut(&ctx).expect("map_mut");
        for (i, slot) in g.iter_mut().enumerate() {
            *slot = (i as u32).wrapping_mul(7);
        }
    } // unmap

    // ...then read back via read-only map.
    let g = buf.map(&ctx).expect("map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, (i as u32).wrapping_mul(7));
    }
}

#[test]
fn multi_kernel_svm_pipeline_via_typed_launchers() {
    // Two-stage compute pipeline operating entirely on a MappedSlice:
    // fill → scale → read back via map. Both kernels run on an OOO
    // queue and take the SVM as a `KernelArg`; the auto-registration
    // path (KernelArg::register_completion) records each launch's
    // event on `buf`, so the buffer's eventual Drop can wait on every
    // in-flight use.
    //
    // MappedSlice flows through `kernels.fill_u32` / `kernels.scale_u32`
    // directly thanks to the `KernelSliceArg<T>` widening on the
    // emitted typed launchers — same surface as DeviceSlice.
    let Some(ctx) = ctx_with_svm() else { return };
    let device: Device = ctx.device().clone();
    let ooo = Queue::<OutOfOrder>::on_device(&ctx, &device).expect("ooo queue");
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = MappedSlice::<u32>::alloc(&ctx, N).expect("alloc");

    // Stage 1: fill_u32 with 4.
    let (buf, fill_event) = kernels
        .fill_u32([N], buf, 4u32)
        .submit(&ooo)
        .expect("submit fill");

    // Stage 2: scale_u32 by 5, ordered after fill via `.after`.
    let buf = kernels
        .scale_u32([N], buf, 5u32)
        .after(fill_event)
        .wait(&ooo)
        .expect("scale after fill");

    // Read result via map: every slot is 4 * 5 = 20.
    let g = buf.map(&ctx).expect("map");
    assert!(g.iter().all(|&v| v == 20));
}
