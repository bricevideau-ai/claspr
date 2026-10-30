//! `Tensor<T, R>` / `TensorMut<T, R>` kernel arguments. On the host the value is
//! one `claspr::Tensor` — a buffer plus its `Shape<R>` (offset, extents,
//! strides) — which the launcher decomposes (on the wire: the buffer, then the
//! shape by value) and the handle hands back, so it pipes into the next kernel
//! as is. A `Tensor` also fills a plain slice / `PerItem*` param (its buffer).
//! The launch checks the shape stays inside the buffer (`TensorShapeMismatch`).

use claspr::eager::{DeviceOpExt, download};
use claspr::{Buffer, DeviceSlice, Error, Shape, Tensor};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

fn upload(ctx: &claspr::Context, v: &[u32]) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::from_slice(ctx, v).expect("alloc")
}

fn read(d: DeviceSlice<u32>) -> Vec<u32> {
    let mut v = vec![0u32; d.len()];
    d.read(&mut v).wait().expect("read");
    v
}

fn iota(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| i * 3 + 1).collect()
}

/// Transpose twice through a pipe: the first launch's output `Tensor` (written
/// as a `PerItemOut` — a tensor over an uninitialised buffer, which the launch
/// initialises) is the second's `Tensor` input, unchanged, and the round trip
/// restores the matrix.
#[test]
fn a_tensor_pipes_into_the_next_kernel() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (n0, n1) = (5, 7);
    let v = iota(n0 * n1);
    let a = Tensor::new(upload(&ctx, &v), [n0, n1]);
    // An uninitialised buffer: the first transpose (a `PerItemOut`) initialises it.
    let b = Tensor::new(
        DeviceSlice::<u32>::alloc_uninit(&ctx, n0 * n1).expect("alloc"),
        [n1, n0],
    );
    let c = upload(&ctx, &vec![0; n0 * n1]);
    let (b, c) = ks
        .transpose_u32([n0, n1], a, b)
        .and_then(|(_a, b)| ks.transpose_u32([n1, n0], b, c))
        .sync(&ctx)
        .expect("launch");
    let (b, c) = (b.into_inner(), c.into_inner());
    assert_eq!(b.shape(), Shape::contiguous([n1, n0]));
    let expected: Vec<u32> = (0..n1)
        .flat_map(|j| (0..n0).map(move |i| (i, j)))
        .map(|(i, j)| v[i * n1 + j])
        .collect();
    assert_eq!(read(b.into_buffer()), expected);
    assert_eq!(read(c), v);
}

/// The same kernel over the whole matrix and over a tile of it: a tile is the
/// same buffer viewed with another shape (`with_shape` + `Shape::slice`).
#[test]
fn rows_of_a_matrix_and_of_a_tile() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (rows, cols) = (6, 9);
    let v = iota(rows * cols);
    let sum = |r: usize, c: std::ops::Range<usize>| -> u32 { c.map(|j| v[r * cols + j]).sum() };

    let m = Tensor::new(upload(&ctx, &v), [rows, cols]);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, rows).expect("alloc");
    let (m, out) = ks
        .tensor_row_sums_u32([rows], m, out)
        .wait()
        .expect("launch");
    assert_eq!(
        read(out),
        (0..rows).map(|r| sum(r, 0..cols)).collect::<Vec<_>>()
    );

    let tile = m.shape().slice(0, 2..5).slice(1, 3..7);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, 3).expect("alloc");
    let (_, out) = ks
        .tensor_row_sums_u32([3], m.with_shape(tile), out)
        .wait()
        .expect("launch");
    assert_eq!(read(out), (2..5).map(|r| sum(r, 3..7)).collect::<Vec<_>>());
}

#[test]
fn rank_three_sub_views() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let v = iota(60);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, 4).expect("alloc");
    let t = Tensor::new(upload(&ctx, &v), [4, 3, 5]);
    let (_, out) = ks.tensor3_pick_u32([4], t, out).wait().expect("launch");
    let expected: Vec<u32> = (0..4).map(|i| v[i * 15 + 5 + 2]).collect();
    assert_eq!(read(out), expected);
}

#[test]
fn tensor_mut_writes_distinct_elements() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let n = 6;
    let d: Vec<u32> = (0..n as u32).map(|i| 100 + i).collect();
    let m = Tensor::new(upload(&ctx, &vec![0; n * n]), [n, n]);
    let (m, _) = ks
        .set_diagonal_u32([n], m, upload(&ctx, &d))
        .wait()
        .expect("launch");
    let got = read(m.into_buffer());
    for i in 0..n {
        for j in 0..n {
            assert_eq!(got[i * n + j], if i == j { d[i] } else { 0 }, "({i}, {j})");
        }
    }
}

/// A shape reaching past the buffer is rejected before launch.
#[test]
fn shape_must_stay_inside_the_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, 4).expect("alloc");
    let t = Tensor::new(upload(&ctx, &iota(19)), [4, 5]); // 20 elements, 19 in the buffer
    match ks.tensor_row_sums_u32([4], t, out).wait() {
        Err(Error::TensorShapeMismatch { arg, len, .. }) => assert_eq!((arg, len), ("src", 19)),
        other => panic!("expected TensorShapeMismatch, got {:?}", other.map(|_| ())),
    }
}

/// A reusable graph replays from uninitialised `Tensor` and buffer cells: each
/// run initialises them, and each return to its cell forgets it again.
#[test]
fn a_graph_replays_from_an_uninitialised_tensor() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (n0, n1) = (4, 6);
    let v = iota(n0 * n1);
    let uninit = || DeviceSlice::<u32>::alloc_uninit(&ctx, n0 * n1).expect("alloc");
    let g = ks
        .transpose_u32(
            [n0, n1],
            Tensor::new(upload(&ctx, &v), [n0, n1]),
            Tensor::new(uninit(), [n1, n0]),
        )
        .and_then(|(_a, b)| ks.transpose_u32([n1, n0], b, uninit()))
        .and_then(|(_b, c)| download(c));
    for run in 1..=3 {
        assert_eq!(g.sync(&ctx).expect("sync"), v, "run {run}");
    }
}
