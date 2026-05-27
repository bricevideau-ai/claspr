//! SharedBuffer round-trip + cross-queue Drop ordering.
//!
//! Tests both the basic map/map_mut RAII pattern AND the
//! `register_use` cross-queue Drop fix. The Drop test deliberately
//! schedules work on a separate command queue, pushes that work's
//! event onto the SharedBuffer's in-flight-use list, then lets
//! SharedBuffer drop — its `clEnqueueSVMFree` must queue-order
//! behind the cross-queue work via the event wait-list instead of
//! deadlocking or freeing while the unmap is still in flight.

use claspr::{Context, Device, InOrder, Launcher, OutOfOrder, Queue, SharedBuffer, SvmLevel};
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

    let mut buf = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc");
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
    // The whole point of #94: SharedBuffer can be used on a queue
    // distinct from its Context's default in-order queue, and Drop's
    // clEnqueueSVMFree must wait for that cross-queue work to finish
    // before reclaiming the allocation.
    //
    // We simulate the path SharedBufferHostView takes: enqueue an
    // unmap on a separate queue, wrap the event in Arc<Event>, set
    // it as last_use, then drop. The drop's free queue-orders on the
    // unmap event via wait-list — no host-side wait, no UB.
    let Some(ctx) = ctx_with_svm() else { return };

    let device: Device = ctx.device().clone();
    let other_queue = Queue::<InOrder>::on_device(&ctx, &device).expect("aux queue");

    let mut buf = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc");
    // Write something so the unmap has data to flush.
    {
        let mut view = buf.map_mut(&other_queue).expect("map_mut on aux");
        for slot in view.iter_mut() {
            *slot = 99;
        }
    } // unmap-and-set-last_use happens inside SharedWriteGuard::Drop

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
    // SharedBuffer, drop. Same expected outcome as above but via the
    // user-facing API path (not the host_view internal helper).
    let Some(ctx) = ctx_with_svm() else { return };
    let device: Device = ctx.device().clone();
    let other_queue = Queue::<InOrder>::on_device(&ctx, &device).expect("aux queue");

    let buf = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc");
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
    // SharedBuffer as a KernelArg should record their completion event
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

    let buf = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc");
    // Issue several launches on the OOO queue, all consuming `&buf`.
    // On an OOO queue, the events may finish in any order — without
    // Vec accumulation, only the most recently registered would gate
    // the free, which would race the others.
    //
    // We can't pass `&SharedBuffer` to the typed `kernels.fill_u32`
    // (which is typed against `DeviceSlice<u32>`), so we drop to the
    // lower-level path: build the LaunchOp manually via a kernel
    // handle and the `KernelArgs` tuple.
    use claspr::{IntoLaunchSpec, LaunchOp};
    for value in 0..4u32 {
        let kernel = kernels.kernel("fill_u32");
        let event = LaunchOp::new(&ooo, &kernel, [N].into_launch_spec(), (&buf, value))
            .submit()
            .expect("submit");
        // Hold each Event briefly; auto-register has already happened.
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
