//! Drop-while-in-flight: every buffer kind must defer its physical
//! release until pending GPU work finishes, so dropping a buffer
//! while a non-blocking command still references it is safe.
//!
//! These tests don't directly observe the deferral — they're
//! correctness assertions: if a drop fired the underlying release
//! eagerly, the runtime would either hang, crash, or surface a
//! sticky error on the context. We force every drop, then poke the
//! context for errors and finish the queue.

use claspr::{Context, DeviceSlice, Launcher};
use claspr_test_kernels::kernels;

const N: usize = 4096;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

#[test]
fn device_slice_drop_while_kernel_in_flight() {
    // submit() returns immediately; without an explicit wait, dropping
    // `buf` next leaves an in-flight launch referencing the cl_mem.
    // The runtime must keep the buffer alive (cl_mem is refcounted)
    // until the launch completes. We finish the queue to force
    // completion, then check the context's sticky error counter.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");
    let (buf, _event) = kernels.fill_u32([N], buf, 1).submit(&ctx).expect("submit");
    drop(buf);
    // Force pending work to complete via the context's default queue.
    ctx.cl_queue().finish().expect("finish");
    assert_eq!(ctx.error_count(), 0, "no release errors expected");
}

// (MappedSlice drop-while-in-flight is covered separately in
// tests/svm.rs, which exercises the cross-queue `register_use` path —
// the proc-macro-emitted typed launch only accepts `&DeviceSlice<T>`,
// so MappedSlice needs the lower-level `ctx.launch` route or the
// host-view RAII flow.)
