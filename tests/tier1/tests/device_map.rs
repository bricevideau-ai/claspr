//! Tier 1 `DeviceSlice::map` / `map_mut` coverage —
//! zero-copy host access on the cl_mem path, blocking and
//! non-blocking terminals + the explicit `release()` path for
//! cross-queue chain ordering.
//!
//! Parallels `svm.rs` for the SVM (`MappedSlice`) side. Tests both
//! halves so they don't drift apart.

use claspr::{Device, DeviceSlice, InOrder, Queue, ReadOnly};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const N: usize = 256;

#[test]
fn map_mut_then_map_round_trip() {
    // Mirror of svm.rs's basic map_mut + map test for the cl_mem
    // path. Seed via map_mut (writes commit back on unmap), inspect
    // via map.
    let Some(ctx) = ctx() else { return };
    let mut buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    {
        let mut g = buf.map_mut().wait().expect("map_mut");
        for (i, slot) in g.iter_mut().enumerate() {
            *slot = (i as u32).wrapping_mul(3);
        }
    } // unmap on drop

    let g = buf.map().wait().expect("map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, (i as u32).wrapping_mul(3));
    }
}

#[test]
fn submit_returns_pending_then_wait_yields_guard() {
    // Non-blocking cl_mem map: `.submit()` returns DeviceMapReadPending
    // with an Event handle; `.wait()` blocks and yields the guard.
    let Some(ctx) = ctx() else { return };
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    // Seed via write.
    let seed: Vec<u32> = (0..N as u32).collect();
    let buf = buf.write(seed).wait().expect("seed write");

    let pending = buf.map().submit().expect("submit map");
    // Event handle exposed for chain ordering.
    let _evt_borrow = pending.event();
    let g = pending.wait().expect("pending.wait");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, i as u32);
    }
}

#[test]
fn submit_mut_pending_derefmut_after_wait() {
    let Some(ctx) = ctx() else { return };
    let mut buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    let pending = buf.map_mut().submit().expect("submit map_mut");
    let mut g = pending.wait().expect("pending.wait");
    for (i, slot) in g.iter_mut().enumerate() {
        *slot = (i as u32).wrapping_mul(13);
    }
    drop(g);

    let g = buf.map().wait().expect("readback map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, (i as u32).wrapping_mul(13));
    }
}

#[test]
fn read_only_marker_allows_map_but_not_map_mut() {
    // ReadOnly is HostReadable + HostWritable (kernel-RO, host-RW).
    // So both map and map_mut should compile. We just exercise map
    // here; map_mut is exercised on ReadWrite above. The explicit
    // marker-rejection tests (HostReadOnly/Frozen rejecting .map_mut,
    // DeviceScratch rejecting .map) live in compile_fail/.
    let Some(ctx) = ctx() else { return };
    let buf = DeviceSlice::<u32, ReadOnly>::from_slice(&ctx, &vec![7u32; N]).expect("from_slice");
    let g = buf.map().wait().expect("map");
    assert!(g.iter().all(|&v| v == 7));
}

#[test]
fn release_returns_unmap_event_for_cross_queue_chaining() {
    // The explicit-release path: consume the guard, get the unmap
    // event back, use it to gate a subsequent kernel launch on a
    // different queue. Mirrors the cl_mem cross-queue ordering use
    // case the deferred-by-design `last_use` register doesn't apply
    // to (cl_mem is refcounted, not last_use-tracked).
    let Some(ctx) = ctx() else { return };
    let device: Device = ctx.device().clone();
    let aux_queue = Queue::<InOrder>::on_device(&ctx, &device).expect("aux queue");

    let mut buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    // Map on aux queue, write a sentinel, release explicitly.
    let mut g = buf.map_mut().wait_on(&aux_queue).expect("map_mut on aux");
    for (i, slot) in g.iter_mut().enumerate() {
        *slot = (i as u32).wrapping_add(100);
    }
    let unmap_event = g.release().expect("release returns event");
    // unmap_event would be threaded into a follow-up enqueue's
    // wait-list via `.after(&evt)` in real code. Here we just wait it.
    unmap_event.wait().expect("unmap event wait");

    // Now use the buffer in a kernel launch on the default queue.
    // The release ensured the unmap commit was visible.
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let buf = kernels.scale_u32([N], buf, 2u32).wait().expect("scale_u32");

    let g = buf.map().wait().expect("readback map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, (i as u32).wrapping_add(100).wrapping_mul(2));
    }
}

#[test]
fn map_then_kernel_launch_round_trip() {
    // End-to-end: alloc → map_mut to seed → kernel scales it →
    // map to readback. All Tier 1, no `.read()`/`.write()` copies.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let mut buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    {
        let mut g = buf.map_mut().wait().expect("seed map");
        for (i, slot) in g.iter_mut().enumerate() {
            *slot = (i as u32).wrapping_add(1);
        }
    } // unmap before kernel launch

    let buf = kernels.scale_u32([N], buf, 3u32).wait().expect("scale_u32");

    let g = buf.map().wait().expect("readback map");
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(v, (i as u32).wrapping_add(1).wrapping_mul(3));
    }
}
