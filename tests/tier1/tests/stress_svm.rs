//! Stress test: many concurrent OOO launches on one `SharedBuffer`.
//!
//! `SharedBuffer::last_use` accumulates `Arc<Event>` per launch and
//! only drains at Drop. Today the largest in-tree run is 4 iterations
//! (`tests/tier1/tests/svm.rs::kernel_launches_on_ooo_queue_register_themselves_for_drop`),
//! which doesn't pin the "many in flight at once" case. This test
//! runs 1024 launches without intermediate sync, then drops the
//! buffer — the Drop's `clEnqueueSVMFree` must wait on all 1024
//! events before reclaiming the allocation, and host memory must
//! stay bounded (the Vec is `O(launches)` and is freed at Drop).
//!
//! Smoke-test shape: assert no crash, no hang, no sticky CL error.
//! Matches the rationale in `tier1/tests/drop_safety.rs` ("These
//! tests don't directly observe the deferral — they're correctness
//! assertions: if a drop fired the underlying release eagerly, the
//! runtime would either hang, crash, or surface a sticky error").

use claspr::{Context, OutOfOrder, Queue, SharedBuffer, SvmLevel};
use claspr_test_kernels::kernels;

const N: usize = 256;
/// At least 1000 per REVIEW.md item #5. `fill_u32` is one
/// accumulator op per work-item; 1024 iterations on llvmpipe complete
/// in well under a second while still fully exercising the Vec
/// accumulation surface.
const LAUNCHES: u32 = 1024;

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
fn thousand_ooo_launches_on_one_sharedbuffer_drop_safely() {
    let Some(ctx) = ctx_with_svm() else { return };
    let device = ctx.device().clone();
    let ooo = Queue::<OutOfOrder>::on_device(&ctx, &device).expect("ooo queue");
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let mut buf = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc");
    // Each iteration submits without blocking on prior events —
    // each launch auto-registers via KernelArg::register_completion
    // and ends up in the buffer's last_use Vec.
    for value in 0..LAUNCHES {
        let (returned, _evt) = kernels
            .fill_u32([N], buf, value)
            .submit(&ooo)
            .expect("submit");
        buf = returned;
    }

    // Drop here triggers clEnqueueSVMFree with a wait list of
    // LAUNCHES events. If the Vec drained eagerly somewhere along
    // the way, or if event refcounts got mismanaged, this would
    // hang, crash, or trip error_count.
    drop(buf);

    use claspr::Launcher;
    ooo.raw().finish().expect("finish ooo");
    ctx.cl_queue().finish().expect("finish ctx default");
    assert_eq!(ctx.error_count(), 0);
}
