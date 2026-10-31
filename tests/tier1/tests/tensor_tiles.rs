//! `Tensor::split` (parts per axis) / `Tensor::tiles` (tile size per axis) hand
//! out disjoint `Tile`s of one buffer that kernels can
//! WRITE concurrently (same device), correct by construction: a tile fits only
//! `Tensor` / `TensorMut` params (never a plain slice — see the compile-fail
//! fixture `tile_to_slice_arg`), its kernel view reaches only its elements, and
//! `split` refuses a view whose indices alias. `Tile::join` takes them all back.

use claspr::eager::download;
use claspr::{DeviceSlice, Shape, Tensor, Tile};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const ROWS: usize = 10;
const COLS: usize = 6;

fn zeros(ctx: &claspr::Context) -> Tensor<DeviceSlice<u32>, 2> {
    Tensor::new(
        DeviceSlice::<u32>::from_slice(ctx, &[0; ROWS * COLS]).expect("alloc"),
        [ROWS, COLS],
    )
}

/// Three row bands written by three launches submitted together on the
/// out-of-order queue (free to overlap), then joined back into one tensor.
#[test]
fn tiles_are_written_concurrently_then_joined() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let tiles = zeros(&ctx).split([3, 1]).expect("split");
    let extents: Vec<[usize; 2]> = tiles.iter().map(|t| t.shape().ext).collect();
    assert_eq!(extents, [[4, COLS], [4, COLS], [2, COLS]]);
    let launched: Vec<_> = tiles
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            let ext = t.shape().ext;
            ks.fill_view_u32([ext[1], ext[0]], t, 100 + i as u32)
                .submit()
                .expect("submit")
        })
        .collect();
    let tiles: Vec<Tile<u32, _, 2>> = launched
        .into_iter()
        .map(|(t, ev)| {
            ev.wait().expect("wait");
            t
        })
        .collect();
    let m = Tile::join(tiles).expect("join");
    assert_eq!(m.shape(), Shape::contiguous([ROWS, COLS]));
    let got = download(m).wait().expect("download");
    for r in 0..ROWS {
        let band = 100 + (r / 4) as u32;
        assert!(
            got[r * COLS..(r + 1) * COLS].iter().all(|&v| v == band),
            "row {r}: {:?}",
            &got[r * COLS..(r + 1) * COLS]
        );
    }
}

/// A 2-D grid of tiles from a tile size (`tiles([4, 4])` over 10 x 6: 3 x 2
/// tiles, the last row / column of tiles taking the remainder), each filled
/// from its grid `index()`, then joined.
#[test]
fn a_grid_of_tiles_from_a_tile_size() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let tiles = zeros(&ctx).tiles([4, 4]).expect("tiles");
    let grid: Vec<([usize; 2], [usize; 2])> =
        tiles.iter().map(|t| (t.index(), t.shape().ext)).collect();
    assert_eq!(
        grid,
        [
            ([0, 0], [4, 4]),
            ([0, 1], [4, 2]),
            ([1, 0], [4, 4]),
            ([1, 1], [4, 2]),
            ([2, 0], [2, 4]),
            ([2, 1], [2, 2])
        ]
    );
    let launched: Vec<_> = tiles
        .into_iter()
        .map(|t| {
            let ([i, j], ext) = (t.index(), t.shape().ext);
            ks.fill_view_u32([ext[1], ext[0]], t, (10 * i + j) as u32)
                .submit()
                .expect("submit")
        })
        .collect();
    let tiles: Vec<Tile<u32, _, 2>> = launched
        .into_iter()
        .map(|(t, ev)| {
            ev.wait().expect("wait");
            t
        })
        .collect();
    let got = download(Tile::join(tiles).expect("join"))
        .wait()
        .expect("download");
    for r in 0..ROWS {
        for c in 0..COLS {
            assert_eq!(
                got[r * COLS + c],
                (10 * (r / 4) + c / 4) as u32,
                "({r}, {c})"
            );
        }
    }
}

/// A tile downloads its own view (here the right half of the columns).
#[test]
fn a_tile_downloads_its_view() {
    let Some(ctx) = ctx() else { return };
    let v: Vec<u32> = (0..(ROWS * COLS) as u32).collect();
    let m = Tensor::new(
        DeviceSlice::<u32>::from_slice(&ctx, &v).expect("alloc"),
        [ROWS, COLS],
    );
    let mut tiles = m.split([1, 2]).expect("split");
    let right = download(tiles.pop().expect("two tiles"))
        .wait()
        .expect("download");
    let expected: Vec<u32> = (0..ROWS)
        .flat_map(|r| (COLS / 2..COLS).map(move |c| (r, c)))
        .map(|(r, c)| v[r * COLS + c])
        .collect();
    assert_eq!(right, expected);
}

/// `split` refuses a view whose indices alias: its tiles would not be disjoint.
#[test]
fn split_refuses_an_aliasing_view() {
    let Some(ctx) = ctx() else { return };
    let buf = DeviceSlice::<u32>::from_slice(&ctx, &[0; COLS]).expect("alloc");
    // Every row is the same row (stride 0): rows "0..5" and "5..10" overlap.
    let broadcast = Shape {
        offset: 0,
        ext: [ROWS, COLS],
        stride: [0, 1],
    };
    assert!(Tensor::new(buf, broadcast).split([2, 1]).is_err());
}

/// `join` needs every tile of the split; a partial set is handed back.
#[test]
fn join_needs_every_tile() {
    let Some(ctx) = ctx() else { return };
    let mut tiles = zeros(&ctx).split([2, 1]).expect("split");
    let last = tiles.pop().expect("two tiles");
    let tiles = Tile::join(tiles).expect_err("a tile is missing");
    let all: Vec<_> = tiles.into_iter().chain([last]).collect();
    assert!(Tile::join(all).is_ok());
}
