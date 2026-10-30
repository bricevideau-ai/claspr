//! Tensor views to and from the host: `download_tensor` (a view → a dense
//! row-major `Vec`), `upload_tensor` (a host array → a contiguous `Tensor`) and
//! `write_tensor` (a host array → a view of an existing buffer, e.g. a tile).
//! A rank-1–3 view with a contiguous innermost axis is one rect copy
//! (`clEnqueue{Read,Write}BufferRect`); `download_tensor` gathers any other
//! view on the host, `write_tensor` refuses it.

use claspr::eager::{DeviceOpExt, download, download_tensor, upload_tensor, write_tensor};
use claspr::{DeviceSlice, Error, Shape, Tensor};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

fn iota(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| i * 3 + 1).collect()
}

fn tensor<const R: usize>(
    ctx: &claspr::Context,
    v: &[u32],
    shape: Shape<R>,
) -> Tensor<DeviceSlice<u32>, R> {
    Tensor::new(
        DeviceSlice::<u32>::from_slice(ctx, v).expect("alloc"),
        shape,
    )
}

#[test]
fn download_a_whole_tensor() {
    let Some(ctx) = ctx() else { return };
    let v = iota(24);
    let got = download_tensor(tensor(&ctx, &v, Shape::contiguous([4, 6])))
        .wait()
        .expect("download");
    assert_eq!(got, v);
}

/// A tile of a matrix and a sub-box of a 3-D tensor: one rect copy each.
#[test]
fn download_a_tile_and_a_sub_box() {
    let Some(ctx) = ctx() else { return };
    let v = iota(48);
    let tile = Shape::contiguous([6, 8]).slice(0, 1..4).slice(1, 2..7);
    let got = download_tensor(tensor(&ctx, &v, tile))
        .wait()
        .expect("tile");
    let expected: Vec<u32> = (1..4)
        .flat_map(|r| (2..7).map(move |c| (r, c)))
        .map(|(r, c)| v[r * 8 + c])
        .collect();
    assert_eq!(got, expected);

    let sub = Shape::contiguous([3, 4, 4])
        .slice(0, 1..3)
        .slice(1, 1..3)
        .slice(2, 0..3);
    let got = download_tensor(tensor(&ctx, &v, sub))
        .wait()
        .expect("sub-box");
    let expected: Vec<u32> = (1..3)
        .flat_map(|z| (1..3).flat_map(move |y| (0..3).map(move |x| (z, y, x))))
        .map(|(z, y, x)| v[z * 16 + y * 4 + x])
        .collect();
    assert_eq!(got, expected);
}

/// No rect copy: a transposed view (strided innermost axis) and a rank-4 view
/// are read whole and gathered.
#[test]
fn download_views_without_a_rect() {
    let Some(ctx) = ctx() else { return };
    let v = iota(24);
    let transposed = Shape {
        offset: 0,
        ext: [6, 4],
        stride: [1, 6],
    };
    let got = download_tensor(tensor(&ctx, &v, transposed))
        .wait()
        .expect("transposed");
    let expected: Vec<u32> = (0..6)
        .flat_map(|c| (0..4).map(move |r| (r, c)))
        .map(|(r, c)| v[r * 6 + c])
        .collect();
    assert_eq!(got, expected);

    let rank4 = Shape::contiguous([2, 2, 3, 2]).slice(2, 1..3);
    let got = download_tensor(tensor(&ctx, &v, rank4))
        .wait()
        .expect("rank 4");
    let expected: Vec<u32> = (0..2)
        .flat_map(|a| {
            (0..2).flat_map(move |b| (1..3).flat_map(move |c| (0..2).map(move |d| (a, b, c, d))))
        })
        .map(|(a, b, c, d)| v[a * 12 + b * 6 + c * 2 + d])
        .collect();
    assert_eq!(got, expected);
}

/// A kernel's output `Tensor` downloads straight from its pipe; the chain starts
/// from `upload_tensor` and replays (the uploaded tensor reseeds each run).
#[test]
fn upload_transpose_download_replays() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (n0, n1) = (3, 5);
    let v = iota(n0 * n1);
    let out = Tensor::new(
        DeviceSlice::<u32>::alloc_uninit(&ctx, n0 * n1).expect("alloc"),
        [n1, n0],
    );
    let g = upload_tensor(v.clone(), [n0, n1])
        .and_then(|a| ks.transpose_u32([n0, n1], a, out))
        .and_then(|(_a, t)| download_tensor(t));
    let expected: Vec<u32> = (0..n1)
        .flat_map(|j| (0..n0).map(move |i| (i, j)))
        .map(|(i, j)| v[i * n1 + j])
        .collect();
    for run in 1..=2 {
        assert_eq!(
            g.sync(&ctx).expect("sync").into_inner(),
            expected,
            "run {run}"
        );
    }
}

/// `write_tensor` fills one tile of a buffer and leaves the rest untouched.
#[test]
fn write_a_tile() {
    let Some(ctx) = ctx() else { return };
    let (rows, cols) = (5, 7);
    let tile = Shape::contiguous([rows, cols])
        .slice(0, 1..3)
        .slice(1, 2..6);
    let t = tensor(&ctx, &vec![0; rows * cols], tile);
    let data: Vec<u32> = (1..=8).collect();
    let t = write_tensor(t, data.clone())
        .sync(&ctx)
        .expect("write")
        .into_inner();
    let got = download(t.into_buffer()).wait().expect("download");
    for r in 0..rows {
        for c in 0..cols {
            let inside = (1..3).contains(&r) && (2..6).contains(&c);
            let want = if inside {
                data[(r - 1) * 4 + (c - 2)]
            } else {
                0
            };
            assert_eq!(got[r * cols + c], want, "({r}, {c})");
        }
    }
}

#[test]
fn write_refuses_a_view_without_a_rect() {
    let Some(ctx) = ctx() else { return };
    let transposed = Shape {
        offset: 0,
        ext: [3, 2],
        stride: [1, 3],
    };
    let t = tensor(&ctx, &[0; 6], transposed);
    let err = write_tensor(t, vec![1u32; 6])
        .sync(&ctx)
        .map(|_| ())
        .expect_err("no rect");
    assert!(matches!(err, Error::NotSupported(_)), "got {err:?}");
}
