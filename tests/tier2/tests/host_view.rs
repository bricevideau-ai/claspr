//! Phase 3.6 coverage — `HostAccessible` three-stage round-trip,
//! mirroring spike scenario 16:
//!
//!   upload → kernel → acquire → and_then_host → release → kernel → download
//!
//! Exercises that the host edit (`view[0] += 100`) actually round-trips
//! through device memory and shows up at download time.

use claspr::{Buffer, Context, HostBuffer, SharedBuffer, SvmLevel};
use claspr_async::{
    DeviceOperation, DeviceOperationHostExt, HostAccessibleExt, download, upload, value,
};
use claspr_test_kernels::kernels;

const N: usize = 64;

#[test]
fn acquire_host_edit_release_round_trip() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Stage 1: upload all-3s
    // Stage 2: kernel scale_u32 by 2 → all 6s
    // Stage 3: acquire host view, set view[0] = 999 (host edit)
    // Stage 4: release back to device
    // Stage 5: kernel scale_u32 by 10 → all 60s except [0]=9990
    // Stage 6: download

    let result: Vec<u32> = upload(vec![3u32; N])
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .and_then(|buf| buf.acquire_host_view())
        .and_then_host(|mut view| {
            // At this point the d2h has completed; view derefs to the
            // current device state. Pretty-print and check, then edit.
            assert_eq!(view[0], 6);
            assert!(view.iter().all(|&v| v == 6));
            view[0] = 999;
            Ok(view)
        })
        .and_then(|view| view.release_to_device())
        .and_then(|buf| kernels.scale_u32([N], buf, 10))
        .and_then(download)
        .sync(&ctx)
        .expect("host_view chain");

    assert_eq!(result[0], 9990, "host edit should round-trip to device");
    for &v in &result[1..] {
        assert_eq!(v, 60, "untouched elements should be 6 * 10 = 60");
    }
}

#[test]
fn acquire_immediately_release_is_a_round_trip() {
    // Acquire with no host edit, release. Verifies the round-trip
    // doesn't corrupt data even without a kernel between the two.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let result: Vec<u32> = upload(vec![42u32; N])
        .and_then(|buf| buf.acquire_host_view())
        .and_then_host(Ok)
        .and_then(|view| view.release_to_device())
        .and_then(download)
        .sync(&ctx)
        .expect("round-trip");
    assert!(result.iter().all(|&v| v == 42));
}

#[test]
fn host_buffer_acquire_release_is_zero_copy_passthrough() {
    // HostBuffer is permanently mapped — acquire/release should be
    // no-ops that just pass the buffer through. Host edit through
    // the view IS the device state (zero-copy), so once the chain
    // ends and we Deref the HostBuffer, we see the edit.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let buf = HostBuffer::<u32>::from_slice(&ctx, &vec![5u32; N]).expect("alloc HostBuffer");

    let returned_buf: HostBuffer<u32> = value(buf)
        .and_then(|b| b.acquire_host_view())
        .and_then_host(|mut view| {
            // view derefs to the always-mapped slice; edit is immediately
            // visible on the host (zero-copy).
            view[0] = 999;
            Ok(view)
        })
        .and_then(|view| view.release_to_device())
        .sync(&ctx)
        .expect("HostBuffer round-trip");

    assert_eq!(returned_buf[0], 999, "host edit should be visible");
    assert_eq!(returned_buf[1], 5);
    assert_eq!(returned_buf.len(), N);
}

#[test]
fn shared_buffer_acquire_release_round_trip() {
    // SharedBuffer (coarse-grain SVM): acquire = clEnqueueSVMMap,
    // release = clEnqueueSVMUnmap. Edit via DerefMut, release,
    // re-acquire to verify the edit landed.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: no SVM support on this device");
        return;
    }

    let buf = SharedBuffer::<u32>::alloc(&ctx, N).expect("SharedBuffer alloc");

    // Stage 1: map, set every element to 11, unmap, re-map to verify.
    let mut buf = value(buf)
        .and_then(|b| b.acquire_host_view())
        .and_then_host(|mut view| {
            for x in view.iter_mut() {
                *x = 11;
            }
            Ok(view)
        })
        .and_then(|view| view.release_to_device())
        .sync(&ctx)
        .expect("SharedBuffer round-trip");

    // Re-acquire via Tier 1 to read back without another chain — the
    // existing SharedReadGuard/SharedWriteGuard are what the
    // standalone API offers. Confirms our HostAccessibleExt round-tripped
    // through SVM correctly.
    let guard = buf.map_mut(&ctx).expect("re-map");
    assert!(
        guard.iter().all(|&v| v == 11),
        "host writes should persist via SVM"
    );
    drop(guard);
}
