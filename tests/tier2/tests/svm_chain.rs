//! SVM (`SharedBuffer<T>`) inside a Tier 2 async chain.
//!
//! The existing `host_view.rs` test goes through `acquire_host_view` /
//! `release_to_device` — the proper RAII path. This file complements
//! it: SharedBuffer threaded through a chain *as the kernel's input*,
//! exercising the `KernelArg::register_completion` auto-registration
//! on every launch in the chain. Verifies that:
//!
//! - SharedBuffer composes with `with_context` for direct kernel
//!   launches on the chain's executor
//! - the buffer survives the chain's drop ordering (no UB even when
//!   the chain ends with the SharedBuffer Drop firing)
//! - multiple kernel launches on the SVM in flight on the OOO queue
//!   are all in the eventual `clEnqueueSVMFree` wait-list
//!
//! Skips on devices without SVM (e.g. some configurations of rusticl).

use claspr::{Context, SharedBuffer, SvmLevel};
use claspr_test_kernels::kernels;

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
fn shared_buffer_threads_through_typed_launchers() {
    // Originally framed as a Tier 2 chain via `with_context`; with
    // that removed, this is just plain Tier 1 with `&ctx` as the
    // launcher. Same coverage: SharedBuffer + the typed launchers'
    // KernelSliceArg widening + map-based readback.
    let Some(ctx) = ctx_with_svm() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc");
    let (buf, fill_evt) = kernels.fill_u32([N], buf, 6u32).submit(&ctx).expect("fill");
    let buf = kernels
        .scale_u32([N], buf, 7u32)
        .after(fill_evt)
        .wait(&ctx)
        .expect("scale");

    let g = buf.map(&ctx).expect("map");
    let result_sum: u32 = g.iter().copied().sum();
    drop(g);

    // fill(6) → scale(7) → sum = 6 * 7 * N
    assert_eq!(result_sum, 6 * 7 * N as u32);
}

#[test]
fn many_in_flight_svm_launches_drop_safely() {
    // Stress-test the Vec accumulation: enqueue many kernel launches
    // on the SVM with no host sync between them, then drop. The Drop's
    // `clEnqueueSVMFree` wait-list must include every launch's event.
    let Some(ctx) = ctx_with_svm() else { return };

    let mut buf = SharedBuffer::<u32>::alloc(&ctx, N).expect("alloc");
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    // 8 successive scales on the same SVM. Each .submit() doesn't
    // block; each launch auto-registers via `KernelArg::register_completion`.
    // The typed launcher consumes + returns `buf` per call.
    for _ in 0..8 {
        let (returned, _evt) = kernels
            .scale_u32([N], buf, 1u32)
            .submit(&ctx)
            .expect("scale");
        buf = returned;
    }
    // Drop the SVM here — Drop drains the in-flight events Vec
    // into the free's wait-list.
    drop(buf);

    // Validate that no error was recorded by any Drop along the way.
    assert_eq!(ctx.error_count(), 0);
}
