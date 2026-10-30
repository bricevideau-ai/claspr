//! `PerItem<T>` kernel arguments: a written buffer handed to each work-item as
//! its own element, so no two work-items hold a `&mut` to the same element.
//!
//! claspr-build lowers the parameter to the raw `&mut [T]` plus a per-item view
//! (adding a global-id parameter if the kernel has none); the host side checks
//! at launch that the grid has exactly as many work-items as the buffer has
//! elements, so every element is owned by exactly one work-item. A mismatch is
//! `GridMismatch` before anything is enqueued, and leaves a graph retryable.
//! The typed launcher then dispatches an unguarded device twin that indexes
//! without bounds checks; the kernel's own entry point (what a raw launch gets)
//! re-checks the grid uniformly and does nothing on a mismatch.

use claspr::eager::{DeviceOpExt, download};
use claspr::{
    DeviceScratch, DeviceSlice, Error, LaunchOp, LaunchSpec, MappedSlice, SvmLevel, USMSlice, slot,
    slots,
};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const N: usize = 256;

slots! {
    Grid: LaunchSpec,
}

fn buf(ctx: &claspr::Context, v: u32) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::from_slice(ctx, &[v; N]).expect("alloc")
}

/// A `PerItem` output, written once per work-item — by a kernel that declares no
/// global id of its own.
#[test]
fn per_item_output_is_written_once_per_work_item() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (_a, _b, out) = ks
        .add_per_item_u32([N], buf(&ctx, 3), buf(&ctx, 4), buf(&ctx, 0))
        .wait()
        .expect("launch");
    let mut host = vec![0u32; N];
    out.read(&mut host).wait().expect("read");
    assert!(host.iter().all(|&v| v == 7), "got {:?}", &host[..4]);
}

/// Read-modify-write of the work-item's own element through the view.
#[test]
fn per_item_in_place_update() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let data = ks
        .scale_per_item_u32([N], buf(&ctx, 5), 3)
        .wait()
        .expect("launch");
    let mut host = vec![0u32; N];
    data.read(&mut host).wait().expect("read");
    assert!(host.iter().all(|&v| v == 15), "got {:?}", &host[..4]);
}

/// A grid with more work-items than elements (would index past the end) or
/// fewer (would leave elements unvisited), in 1 or 2 dimensions, is rejected
/// before the launch.
#[test]
fn grid_must_cover_the_buffer_exactly() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    for grid in [
        LaunchSpec::from([N + 1]),
        LaunchSpec::from([N - 1]),
        LaunchSpec::from([N, 2]),
    ] {
        let err = ks
            .add_per_item_u32(grid, buf(&ctx, 3), buf(&ctx, 4), buf(&ctx, 0))
            .wait()
            .expect_err("mismatched grid");
        match err {
            Error::GridMismatch { arg, grid: g, len } => {
                assert_eq!((arg, len), ("out", N));
                assert_eq!(g, grid.global());
            }
            other => panic!("expected GridMismatch, got {other:?}"),
        }
    }
}

/// In a graph, a rejected launch hands its lent buffers back: rebinding the
/// grid makes the same graph run, with the right result.
#[test]
fn rejected_launch_leaves_the_graph_retryable() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let g = ks
        .add_per_item_u32(slot!(Grid), buf(&ctx, 3), buf(&ctx, 4), buf(&ctx, 0))
        .and_then(|(_a, _b, out)| download(out))
        .bind(Grid(LaunchSpec::from([N + 1])));
    let err = g.sync(&ctx).expect_err("oversized grid");
    assert!(matches!(err, Error::GridMismatch { .. }), "got {err:?}");
    g.mutate_bind(Grid(LaunchSpec::from([N])))
        .expect("rebind grid");
    for run in 1..=2 {
        let out = g.sync(&ctx).expect("sync after rebinding");
        assert!(out.iter().all(|&v| v == 7), "run {run}: {:?}", &out[..4]);
    }
}

/// `PerItemOut`: a kernel initialises an UNINITIALISED buffer and the launch
/// hands back the initialised `DeviceSlice` — the "kernel-write-only" pattern
/// with no `unsafe assume_init`.
#[test]
fn per_item_out_initialises_an_uninitialised_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let uninit = DeviceSlice::<u32>::alloc_uninit(&ctx, N).expect("alloc uninit");
    let (_a, out): (_, DeviceSlice<u32>) = ks
        .scaled_copy_out_u32([N], buf(&ctx, 3), uninit, 5)
        .wait()
        .expect("launch");
    let mut host = vec![0u32; N];
    out.read(&mut host).wait().expect("read");
    assert!(host.iter().all(|&v| v == 15), "got {:?}", &host[..4]);
}

/// An initialised buffer is accepted by a `PerItemOut` parameter too.
#[test]
fn per_item_out_accepts_an_initialised_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (_a, out) = ks
        .scaled_copy_out_u32([N], buf(&ctx, 2), buf(&ctx, 9), 4)
        .wait()
        .expect("launch");
    let mut host = vec![0u32; N];
    out.read(&mut host).wait().expect("read");
    assert!(host.iter().all(|&v| v == 8), "got {:?}", &host[..4]);
}

/// The grid must cover an uninitialised output exactly — otherwise some of it
/// would be handed back "initialised" without having been written.
#[test]
fn per_item_out_rejects_a_grid_that_would_leave_elements_unwritten() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let uninit = DeviceSlice::<u32>::alloc_uninit(&ctx, N).expect("alloc uninit");
    let err = ks
        .scaled_copy_out_u32([N - 1], buf(&ctx, 3), uninit, 5)
        .wait()
        .expect_err("short grid");
    assert!(
        matches!(
            err,
            Error::GridMismatch {
                arg: "out",
                len: N,
                ..
            }
        ),
        "got {err:?}"
    );
}

/// A graph over an uninitialised output cell replays: each run's output is
/// returned to the cell (forgetting initialisation) and rewritten by the next.
#[test]
fn per_item_out_graph_replays_from_an_uninitialised_cell() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let uninit = DeviceSlice::<u32>::alloc_uninit(&ctx, N).expect("alloc uninit");
    let g = ks
        .scaled_copy_out_u32([N], buf(&ctx, 7), uninit, 3)
        .and_then(|(_a, out)| download(out));
    for run in 1..=3 {
        let out = g.sync(&ctx).expect("sync");
        assert!(out.iter().all(|&v| v == 21), "run {run}: {:?}", &out[..4]);
    }
}

/// `DeviceScratch` memory (the host can never write it) initialised by a
/// kernel, then consumed by another kernel — no zero-fill, no `unsafe`.
#[test]
fn per_item_out_initialises_device_scratch() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let scratch = DeviceSlice::<u32, DeviceScratch>::alloc_uninit(&ctx, N).expect("alloc scratch");
    let (_a, scratch) = ks
        .scaled_copy_out_u32([N], buf(&ctx, 2), scratch, 10)
        .wait()
        .expect("initialise scratch");
    let (_s, _b, out) = ks
        .add_u32([N], scratch, buf(&ctx, 1), buf(&ctx, 0))
        .wait()
        .expect("consume scratch");
    let mut host = vec![0u32; N];
    out.read(&mut host).wait().expect("read");
    assert!(host.iter().all(|&v| v == 21), "got {:?}", &host[..4]);
}

/// `PerItemIn` inputs are read element-wise through a read-only view.
#[test]
fn per_item_in_inputs_are_read_element_wise() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (_a, _b, out) = ks
        .add_per_item_in_u32([N], buf(&ctx, 3), buf(&ctx, 4), buf(&ctx, 0))
        .wait()
        .expect("launch");
    let mut host = vec![0u32; N];
    out.read(&mut host).wait().expect("read");
    assert!(host.iter().all(|&v| v == 7), "got {:?}", &host[..4]);
}

/// The grid check bounds `PerItemIn` INPUTS too: an input shorter than the grid
/// (which a plain `&[T]` would read past the end of) is rejected, naming it.
#[test]
fn per_item_in_input_shorter_than_the_grid_is_rejected() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let short = DeviceSlice::<u32>::from_slice(&ctx, &[3; N / 2]).expect("short input");
    let err = ks
        .add_per_item_in_u32([N], short, buf(&ctx, 4), buf(&ctx, 0))
        .wait()
        .expect_err("short input");
    assert!(
        matches!(err, Error::GridMismatch { arg: "a", len, .. } if len == N / 2),
        "got {err:?}"
    );
}

/// A read-only shared buffer (`Arc<DeviceSlice>`) is a valid `PerItemIn`.
#[test]
fn per_item_in_accepts_a_shared_read_only_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let shared = std::sync::Arc::new(buf(&ctx, 10));
    for addend in [1u32, 2] {
        let (_a, _b, out) = ks
            .add_per_item_in_u32(
                [N],
                std::sync::Arc::clone(&shared),
                buf(&ctx, addend),
                buf(&ctx, 0),
            )
            .wait()
            .expect("launch");
        let mut host = vec![0u32; N];
        out.read(&mut host).wait().expect("read");
        assert!(
            host.iter().all(|&v| v == 10 + addend),
            "got {:?}",
            &host[..4]
        );
    }
}

/// Two write-only outputs, initialised from uninitialised buffers, through a
/// kernel whose early-return path must also prove both writes.
#[test]
fn per_item_out_two_outputs_with_an_early_return() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let input: Vec<u32> = (0..N as u32)
        .map(|i| if i % 4 == 0 { 0 } else { (i << 16) | (i + 7) })
        .collect();
    let a = DeviceSlice::<u32>::from_slice(&ctx, &input).expect("a");
    let lo = DeviceSlice::<u32>::alloc_uninit(&ctx, N).expect("lo");
    let hi = DeviceSlice::<u32>::alloc_uninit(&ctx, N).expect("hi");
    let (_a, lo, hi) = ks.split_halves_u32([N], a, lo, hi).wait().expect("launch");
    let (mut l, mut h) = (vec![0u32; N], vec![0u32; N]);
    lo.read(&mut l).wait().expect("read lo");
    hi.read(&mut h).wait().expect("read hi");
    for (i, &v) in input.iter().enumerate() {
        assert_eq!((l[i], h[i]), (v & 0xffff, v >> 16), "element {i}");
    }
}

/// The packed `(x, y, z)` element `i` of a `coords_u32` run on a `w × h × …`
/// grid holds: row-major, x fastest.
fn expected_coords(i: usize, w: usize, h: usize) -> u32 {
    let (x, y, z) = (i % w, (i / w) % h, i / (w * h));
    ((z << 20) | (y << 10) | x) as u32
}

/// 2-D and 3-D grids: work-item `(x, y, z)` owns element `x + w * (y + h * z)`,
/// and any grid shape with one work-item per element passes the check.
#[test]
fn per_item_partitions_two_and_three_dimensional_grids() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    for grid in [
        LaunchSpec::from([16, N / 16]),
        LaunchSpec::from([4, 8, N / 32]),
        LaunchSpec::from([N, 1]),
    ] {
        let out = DeviceSlice::<u32>::alloc_uninit(&ctx, N).expect("alloc uninit");
        let out = ks.coords_u32(grid, out).wait().expect("launch");
        let mut host = vec![0u32; N];
        out.read(&mut host).wait().expect("read");
        let g = grid.global();
        let (w, h) = (g[0], g.get(1).copied().unwrap_or(1));
        for (i, &v) in host.iter().enumerate() {
            assert_eq!(v, expected_coords(i, w, h), "grid {g:?}, element {i}");
        }
    }
}

/// An uninitialised SVM (`MappedSlice`) buffer initialised by a `PerItemOut`
/// kernel; the launch hands back the initialised `MappedSlice`.
#[test]
fn per_item_out_initialises_an_uninitialised_mapped_slice() {
    let Some(ctx) = claspr_test_support::ctx_with_svm() else {
        return;
    };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let uninit = MappedSlice::<u32>::alloc_uninit(&ctx, N).expect("alloc uninit");
    let (_a, out): (_, MappedSlice<u32>) = ks
        .scaled_copy_out_u32([N], buf(&ctx, 6), uninit, 7)
        .wait()
        .expect("launch");
    let view = out.map().wait().expect("map");
    assert!(view.iter().all(|&v| v == 42), "got {:?}", &view[..4]);
}

/// An uninitialised USM (`USMSlice`, fine-grain system SVM) buffer initialised
/// by a `PerItemOut` kernel.
#[test]
fn per_item_out_initialises_an_uninitialised_usm_slice() {
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() != SvmLevel::FineSystem {
        eprintln!("SKIP: no fine-grain-system SVM");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let uninit = USMSlice::<u32>::alloc_uninit(&ctx, N).expect("alloc uninit");
    let (_a, out): (_, USMSlice<u32>) = ks
        .scaled_copy_out_u32([N], buf(&ctx, 5), uninit, 9)
        .wait()
        .expect("launch");
    assert!(out.iter().all(|&v| v == 45), "got {:?}", &out[..4]);
}

/// A raw launch of a per-item kernel's own entry point (which skips the host
/// grid check) runs its in-kernel guard: a matching grid computes as usual, and
/// a grid that doesn't cover every per-item buffer does nothing — no work-item
/// reads past the short input.
#[test]
fn raw_launch_runs_the_in_kernel_grid_guard() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let kernel = ks.kernel("add_per_item_in_u32");
    let (a, b, out) = (buf(&ctx, 3), buf(&ctx, 4), buf(&ctx, 0));
    LaunchOp::new(&ctx, &kernel, [N], (&a, &b, &out))
        .wait()
        .expect("launch");
    let mut host = vec![0u32; N];
    out.read(&mut host).wait().expect("read");
    assert!(host.iter().all(|&v| v == 7), "got {:?}", &host[..4]);

    let short = DeviceSlice::<u32>::from_slice(&ctx, &[3; N / 2]).expect("alloc");
    let out = buf(&ctx, 0);
    LaunchOp::new(&ctx, &kernel, [N], (&short, &b, &out))
        .wait()
        .expect("launch");
    out.read(&mut host).wait().expect("read");
    assert!(
        host.iter().all(|&v| v == 0),
        "guard let a work-item through: {:?}",
        &host[..4]
    );
}

/// The unguarded twin the typed launcher dispatches is not handed out raw.
#[test]
fn raw_lookup_of_the_unchecked_twin_panics() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ks.kernel("add_per_item_u32__claspr_unchecked")
    }))
    .expect_err("the unchecked twin must be refused");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or("");
    assert!(msg.contains("unchecked twin"), "unexpected panic: {msg}");
}
