//! Phase 3.6 coverage — `HostAccessible` three-stage round-trip,
//! mirroring spike scenario 16:
//!
//!   upload → kernel → acquire → and_then_host → release → kernel → download
//!
//! Exercises that the host edit (`view[0] += 100`) actually round-trips
//! through device memory and shows up at download time.

use claspr::Context;
use claspr_async::{
    DeviceOperation, DeviceOperationHostExt, HostAccessibleExt, download, upload, with_context,
};
use claspr_test_kernels::kernels;

const N: usize = 64;

#[test]
fn acquire_host_edit_release_round_trip() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    // Stage 1: upload all-3s
    // Stage 2: kernel scale_u32 by 2 → all 6s
    // Stage 3: acquire host view, set view[0] = 999 (host edit)
    // Stage 4: release back to device
    // Stage 5: kernel scale_u32 by 10 → all 60s except [0]=9990
    // Stage 6: download

    let result: Vec<u32> = upload(vec![3u32; N])
        .and_then(|buf| {
            with_context(move |ec| {
                let kernels = kernels::kernels(ec.context())?;
                kernels.scale_u32(ec, [N], &buf, 2).wait()?;
                Ok::<_, claspr::Error>(buf)
            })
        })
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
        .and_then(|buf| {
            with_context(move |ec| {
                let kernels = kernels::kernels(ec.context())?;
                kernels.scale_u32(ec, [N], &buf, 10).wait()?;
                Ok::<_, claspr::Error>(buf)
            })
        })
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
        .and_then_host(|view| Ok(view))
        .and_then(|view| view.release_to_device())
        .and_then(download)
        .sync(&ctx)
        .expect("round-trip");
    assert!(result.iter().all(|&v| v == 42));
}
