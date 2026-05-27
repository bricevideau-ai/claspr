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

use claspr::{Context, IntoLaunchSpec, LaunchOp, SharedBuffer, SvmLevel};
use claspr_async::{DeviceOperation, with_context};
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
fn shared_buffer_threads_through_async_chain() {
    let Some(ctx) = ctx_with_svm() else { return };

    let result_sum: u32 = with_context(|ec| {
        // Allocate SVM inside the chain so the buffer's ownership
        // belongs to this chain's scope. fill_u32 + scale_u32 launches
        // both go through the typed Tier 1 API (which only accepts
        // DeviceSlice for the slice arg), so we drop to LaunchOp for
        // the SVM arg path.
        let buf = SharedBuffer::<u32>::alloc(ec.context(), N)?;
        let kernels = kernels::kernels(ec.context())?;
        let fill_kernel = kernels.kernel("fill_u32");
        let scale_kernel = kernels.kernel("scale_u32");

        // Stage 1: fill.
        let fill_evt =
            LaunchOp::new(ec, &fill_kernel, [N].into_launch_spec(), (&buf, 6u32)).submit()?;
        // Stage 2: scale, ordered after fill, blocking.
        LaunchOp::new(ec, &scale_kernel, [N].into_launch_spec(), (&buf, 7u32))
            .after(&fill_evt)
            .wait()?;

        // Read back via map.
        let g = buf.map(ec)?;
        let s = g.iter().copied().sum::<u32>();
        drop(g);
        Ok::<u32, claspr::Error>(s)
    })
    .sync(&ctx)
    .expect("svm chain");

    // fill(6) → scale(7) → sum = 6 * 7 * N
    assert_eq!(result_sum, 6 * 7 * N as u32);
}

#[test]
fn many_in_flight_svm_launches_drop_safely() {
    // Stress-test the Vec accumulation: enqueue many kernel launches
    // on the SVM with no host sync between them, then drop. The Drop's
    // `clEnqueueSVMFree` wait-list must include every launch's event.
    let Some(ctx) = ctx_with_svm() else { return };

    with_context(|ec| {
        let buf = SharedBuffer::<u32>::alloc(ec.context(), N)?;
        let kernels = kernels::kernels(ec.context())?;
        // 8 successive scales on the same SVM. Each .submit() doesn't
        // block, each launch auto-registers via `KernelArg::register_completion`.
        for _ in 0..8 {
            let k = kernels.kernel("scale_u32");
            let _evt = LaunchOp::new(ec, &k, [N].into_launch_spec(), (&buf, 1u32)).submit()?;
        }
        // Drop the SVM here — Drop drains the in-flight events Vec
        // into the free's wait-list.
        drop(buf);
        Ok::<(), claspr::Error>(())
    })
    .sync(&ctx)
    .expect("svm stress");

    // Validate that no error was recorded by any Drop along the way.
    assert_eq!(ctx.error_count(), 0);
}
