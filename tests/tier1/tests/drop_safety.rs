//! Drop-while-in-flight: every buffer kind must defer its physical
//! release until pending GPU work finishes, so dropping a buffer
//! while a non-blocking command still references it is safe.
//!
//! Covered kinds: `DeviceSlice` (cl_mem — release is refcounted by
//! the runtime), `MappedSlice` and `USMSlice` (SVM — the free must be
//! event-gated behind the launch's completion, the mechanism under
//! test). `DeviceScalar` is a length-1 `DeviceSlice`, so it rides the
//! same cl_mem path and isn't tested separately.
//!
//! These tests don't directly observe the deferral — they're
//! correctness assertions: if a drop fired the underlying release
//! eagerly, the runtime would either hang, crash, or surface a
//! sticky error on the context. We force every drop, then poke the
//! context for errors and finish the queue.

use claspr::{Context, DeviceSlice, Launcher, MappedSlice, SvmLevel, USMSlice};
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
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let (buf, _event) = kernels.fill_u32([N], buf, 1).submit().expect("submit");
    drop(buf);
    // Force pending work to complete via the context's default queue.
    ctx.cl_queue().finish().expect("finish");
    assert_eq!(ctx.error_count(), 0, "no release errors expected");
}

#[test]
fn mapped_slice_drop_while_kernel_in_flight() {
    // SVM path: unlike cl_mem, an SVM free is NOT refcounted past
    // in-flight commands by the runtime — claspr must gate the free
    // behind the launch's completion event (the `last_use`
    // registration). Dropping right after submit() exercises exactly
    // that gate.
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: no SVM support");
        return;
    }
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let (buf, _event) = kernels.fill_u32([N], buf, 1).submit().expect("submit");
    drop(buf);
    ctx.cl_queue().finish().expect("finish");
    assert_eq!(ctx.error_count(), 0, "no release errors expected");
}

#[test]
fn usm_slice_drop_while_kernel_in_flight() {
    // Fine-grain-system SVM over a host Vec — same event-gated-free
    // requirement as MappedSlice, different allocation path.
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() != SvmLevel::FineSystem {
        eprintln!("SKIP: no fine-grain-system SVM");
        return;
    }
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let buf = USMSlice::<u32>::new(&ctx, vec![0u32; N]).expect("alloc");
    let (buf, _event) = kernels.fill_u32([N], buf, 1).submit().expect("submit");
    drop(buf);
    ctx.cl_queue().finish().expect("finish");
    assert_eq!(ctx.error_count(), 0, "no release errors expected");
}
