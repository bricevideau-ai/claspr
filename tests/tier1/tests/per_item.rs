//! `PerItem<T>` kernel arguments: a written buffer handed to each work-item as
//! its own element, so no two work-items hold a `&mut` to the same element.
//!
//! claspr-build lowers the parameter to the raw `&mut [T]` plus a per-item view
//! (adding a global-id parameter if the kernel has none); the host side checks
//! at launch that the grid is 1-D and exactly as long as the buffer, so every
//! element is owned by exactly one work-item. A mismatch is `GridMismatch`
//! before anything is enqueued, and leaves a graph retryable.

use claspr::eager::{DeviceOpExt, download};
use claspr::{DeviceSlice, Error, LaunchSpec, slot, slots};
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

/// A grid that would index past the end, one that would leave elements
/// unwritten, and a 2-D grid are all rejected before the launch.
#[test]
fn grid_must_be_one_dimensional_and_cover_the_buffer_exactly() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    for grid in [
        LaunchSpec::from([N + 1]),
        LaunchSpec::from([N - 1]),
        LaunchSpec::from([N, 1]),
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
