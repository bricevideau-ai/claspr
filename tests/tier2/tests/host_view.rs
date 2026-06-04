//! `HostAccessible` coverage — acquire/release wrapped around an
//! `and_then_host` closure that touches the view's bytes directly.
//!
//! Under the async `and_then_host` the closure receives the mapped
//! slice as its argument — `&mut [T]` for the read/write variant,
//! `&[T]` for the read-only variant. No method on the view needs to
//! be called inside the closure; the view passes through `and_then`
//! to `release_to_device` unchanged.

use claspr::{Context, MappedSlice, SvmLevel};
use claspr_async::{
    DeviceOperation, DeviceOperationHostExt, HostReadableExt, HostWritableExt, download, upload,
    value,
};
use claspr_test_kernels::kernels;

const N: usize = 64;

// ── DeviceSlice (real map/unmap) ────────────────────────────────────

#[test]
fn acquire_host_edit_release_round_trip() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // upload all-3s → scale by 2 (all 6s) → host view (edit [0]=999)
    // → release → scale by 10 (all 60s, except [0]=9990) → download.
    let result: Vec<u32> = upload(vec![3u32; N])
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .and_then(|buf| buf.acquire_host_view())
        .and_then_host(|slice: &mut [u32]| {
            assert_eq!(slice[0], 6);
            assert!(slice.iter().all(|&v| v == 6));
            slice[0] = 999;
            Ok(())
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
    // Acquire, no host edit, release. Verifies the map/unmap pair
    // doesn't corrupt data even without anything between them.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let result: Vec<u32> = upload(vec![42u32; N])
        .and_then(|buf| buf.acquire_host_view())
        .and_then_host(|_slice: &mut [u32]| Ok(()))
        .and_then(|view| view.release_to_device())
        .and_then(download)
        .sync(&ctx)
        .expect("round-trip");
    assert!(result.iter().all(|&v| v == 42));
}

#[test]
fn acquire_host_view_read_inspects_without_writeback() {
    // Read-only variant: map with CL_MAP_READ only. Closure sees
    // `&[u32]` — can read but not mutate. Unmap is cheaper because
    // the runtime knows it doesn't have to commit anything back.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Fill on device, inspect via read-only host view, then continue
    // device work using the same buffer (release_to_device hands it
    // back).
    let sum_cell = std::sync::Arc::new(std::sync::Mutex::new(0u32));
    let cell = std::sync::Arc::clone(&sum_cell);

    let result: Vec<u32> = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 7))
        .and_then(|buf| buf.acquire_host_view_read())
        .and_then_host(move |slice: &[u32]| {
            // Type is &[u32] — no DerefMut, no &mut. Read-only.
            assert!(slice.iter().all(|&v| v == 7));
            *cell.lock().unwrap() = slice.iter().sum();
            Ok(())
        })
        .and_then(|view| view.release_to_device())
        // The buffer is unchanged because we mapped READ-only —
        // downstream device work sees the original fill.
        .and_then(|buf| kernels.scale_u32([N], buf, 3))
        .and_then(download)
        .sync(&ctx)
        .expect("read-only view chain");

    assert_eq!(*sum_cell.lock().unwrap(), 7 * N as u32);
    assert!(result.iter().all(|&v| v == 21), "7 * 3 = 21");
}

#[test]
fn acquire_host_view_read_just_inspect_and_drop() {
    // Read-only view at the end of a chain: nothing else needed,
    // release_to_device + download.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let first_cell = std::sync::Arc::new(std::sync::Mutex::new(0u32));
    let cell = std::sync::Arc::clone(&first_cell);

    let _buf = upload(vec![13u32; N])
        .and_then(|buf| kernels.scale_u32([N], buf, 4))
        .and_then(|buf| buf.acquire_host_view_read())
        .and_then_host(move |slice: &[u32]| {
            *cell.lock().unwrap() = slice[0];
            Ok(())
        })
        .and_then(|view| view.release_to_device())
        .sync(&ctx)
        .expect("read-only inspect chain");

    assert_eq!(*first_cell.lock().unwrap(), 52, "13 * 4 = 52");
}

// ── MappedSlice (coarse-grain SVM map/unmap) ───────────────────────

#[test]
fn mapped_slice_acquire_release_round_trip() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: no SVM support on this device");
        return;
    }

    let buf = MappedSlice::<u32>::alloc(&ctx, N).expect("MappedSlice alloc");

    let mut buf = value(buf)
        .and_then(|b| b.acquire_host_view())
        .and_then_host(|slice: &mut [u32]| {
            for x in slice.iter_mut() {
                *x = 11;
            }
            Ok(())
        })
        .and_then(|view| view.release_to_device())
        .sync(&ctx)
        .expect("MappedSlice round-trip");

    // Re-acquire via Tier 1 to read back without another chain.
    let guard = buf.map_mut().wait(&ctx).expect("re-map");
    assert!(
        guard.iter().all(|&v| v == 11),
        "host writes should persist via SVM"
    );
    drop(guard);
}

#[test]
fn mapped_slice_acquire_release_read_only() {
    // Read-only SVM map: clEnqueueSVMMap with CL_MAP_READ only.
    // Closure sees `&[u32]` (no DerefMut, no &mut access). Unmap is
    // cheaper since the runtime knows no writes to commit.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: no SVM support on this device");
        return;
    }

    // Seed the MappedSlice via Tier 1 first.
    let mut buf = MappedSlice::<u32>::alloc(&ctx, N).expect("MappedSlice alloc");
    {
        let mut guard = buf.map_mut().wait(&ctx).expect("seed map");
        for (i, x) in guard.iter_mut().enumerate() {
            *x = 100 + i as u32;
        }
    }

    let first_cell = std::sync::Arc::new(std::sync::Mutex::new(0u32));
    let cell = std::sync::Arc::clone(&first_cell);
    let sum_cell = std::sync::Arc::new(std::sync::Mutex::new(0u32));
    let scell = std::sync::Arc::clone(&sum_cell);

    let buf = value(buf)
        .and_then(|b| b.acquire_host_view_read())
        .and_then_host(move |slice: &[u32]| {
            // Type is &[u32] — no mutation possible.
            *cell.lock().unwrap() = slice[0];
            *scell.lock().unwrap() = slice.iter().sum();
            Ok(())
        })
        .and_then(|view| view.release_to_device())
        .sync(&ctx)
        .expect("MappedSlice read-only chain");

    assert_eq!(*first_cell.lock().unwrap(), 100);
    let expected_sum: u32 = (0..N as u32).map(|i| 100 + i).sum();
    assert_eq!(*sum_cell.lock().unwrap(), expected_sum);

    // Buffer is unchanged — re-map and verify the original seed
    // values survived (read-only map doesn't write back).
    let mut buf = buf;
    let guard = buf.map_mut().wait(&ctx).expect("verify map");
    for (i, &v) in guard.iter().enumerate() {
        assert_eq!(v, 100 + i as u32, "read-only map must not modify data");
    }
    drop(guard);
}
