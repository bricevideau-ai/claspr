//! Integration tests for `MappedSlice::Drop` correctness.
//!
//! Validates the Phase 0 fix: `MappedSlice::Drop` now uses
//! `clEnqueueSVMFree` (queue-ordered) instead of the immediate
//! `clSVMFree` (UB if commands in flight per the CL spec).
//!
//! These tests are skip-on-no-device — they `return Ok(())` if the
//! host has no OpenCL device or no SVM support, so CI without GPUs
//! still passes.

use claspr::{Context, MappedSlice, SvmLevel};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn ctx_with_svm() -> Result<Option<Context>> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(_) => return Ok(None),
    };
    if ctx.svm_capability() == SvmLevel::None {
        return Ok(None);
    }
    Ok(Some(ctx))
}

#[test]
fn alloc_and_drop_basic() -> Result<()> {
    let Some(ctx) = ctx_with_svm()? else {
        eprintln!("SKIP: no OpenCL device or no SVM support");
        return Ok(());
    };

    let buf = MappedSlice::<u32>::alloc_zero(&ctx, 1024)?;
    drop(buf);

    // The drop enqueues an SVM free. The context's default queue
    // tracks it; nothing else should have errored.
    assert_eq!(ctx.error_count(), 0);
    Ok(())
}

#[test]
fn alloc_drop_many_in_sequence() -> Result<()> {
    // Stress: lots of allocations, all dropped without explicit sync.
    // Each Drop enqueues an SVM free on the default queue; the runtime
    // sequences them. If any errored, the sticky counter catches it.
    let Some(ctx) = ctx_with_svm()? else {
        eprintln!("SKIP: no OpenCL device or no SVM support");
        return Ok(());
    };

    for _ in 0..32 {
        let buf = MappedSlice::<u32>::alloc_zero(&ctx, 256)?;
        drop(buf);
    }

    assert_eq!(ctx.error_count(), 0);
    Ok(())
}

#[test]
fn alloc_drop_with_map_in_between() -> Result<()> {
    // A more realistic flow: alloc, map (which is queue-ordered),
    // unmap (also queue-ordered, on guard Drop), then drop the buffer.
    // The SVM free enqueued by buffer Drop will be ordered after the
    // unmap, so it can't race the still-pending unmap command.
    let Some(ctx) = ctx_with_svm()? else {
        eprintln!("SKIP: no OpenCL device or no SVM support");
        return Ok(());
    };

    {
        let mut buf = MappedSlice::<u32>::alloc_zero(&ctx, 16)?;
        {
            let mut guard = buf.map_mut().wait(&ctx)?;
            for (i, x) in guard.iter_mut().enumerate() {
                *x = i as u32;
            }
        }
        // buf drops here, after the map guard's unmap completes.
    }

    assert_eq!(ctx.error_count(), 0);
    Ok(())
}
