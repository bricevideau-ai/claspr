//! `Tensor<T, R>` / `TensorMut<T, R>` kernel arguments: a buffer plus a
//! `Shape<R>` (offset, extents, strides) passed by value right after it — the
//! launcher takes the buffer (piped) then `<name>_shape` (not piped). The launch
//! checks the shape stays inside the buffer (`TensorShapeMismatch` otherwise).

use claspr::{Buffer, DeviceSlice, Error, Shape};
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

#[test]
fn transpose_reads_any_multi_index() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (n0, n1) = (5, 7);
    let v = iota(n0 * n1);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, n0 * n1).expect("alloc");
    let (_, out) = ks
        .transpose_u32([n0, n1], upload(&ctx, &v), Shape::contiguous([n0, n1]), out)
        .wait()
        .expect("launch");
    let expected: Vec<u32> = (0..n1)
        .flat_map(|j| (0..n0).map(move |i| (i, j)))
        .map(|(i, j)| v[i * n1 + j])
        .collect();
    assert_eq!(read(out), expected);
}

/// The same kernel over the whole matrix and over a tile of it (`Shape::slice`):
/// a tile is just another shape of the same buffer.
#[test]
fn rows_of_a_matrix_and_of_a_tile() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (rows, cols) = (6, 9);
    let v = iota(rows * cols);
    let sum = |r: usize, c: std::ops::Range<usize>| -> u32 { c.map(|j| v[r * cols + j]).sum() };

    let whole = Shape::contiguous([rows, cols]);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, rows).expect("alloc");
    let (m, out) = ks
        .tensor_row_sums_u32([rows], upload(&ctx, &v), whole, out)
        .wait()
        .expect("launch");
    assert_eq!(
        read(out),
        (0..rows).map(|r| sum(r, 0..cols)).collect::<Vec<_>>()
    );

    let tile = whole.slice(0, 2..5).slice(1, 3..7);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, 3).expect("alloc");
    let (_, out) = ks
        .tensor_row_sums_u32([3], m, tile, out)
        .wait()
        .expect("launch");
    assert_eq!(read(out), (2..5).map(|r| sum(r, 3..7)).collect::<Vec<_>>());
}

#[test]
fn rank_three_sub_views() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ext = [4, 3, 5];
    let v = iota(60);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, 4).expect("alloc");
    let (_, out) = ks
        .tensor3_pick_u32([4], upload(&ctx, &v), Shape::contiguous(ext), out)
        .wait()
        .expect("launch");
    let expected: Vec<u32> = (0..4).map(|i| v[i * 15 + 5 + 2]).collect();
    assert_eq!(read(out), expected);
}

#[test]
fn tensor_mut_writes_distinct_elements() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let n = 6;
    let d: Vec<u32> = (0..n as u32).map(|i| 100 + i).collect();
    let (m, _) = ks
        .set_diagonal_u32(
            [n],
            upload(&ctx, &vec![0; n * n]),
            Shape::contiguous([n, n]),
            upload(&ctx, &d),
        )
        .wait()
        .expect("launch");
    let got = read(m);
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
    let shape = Shape::contiguous([4, 5]); // 20 elements over a 19-element buffer
    match ks
        .tensor_row_sums_u32([4], upload(&ctx, &iota(19)), shape, out)
        .wait()
    {
        Err(Error::TensorShapeMismatch { arg, len, .. }) => assert_eq!((arg, len), ("src", 19)),
        other => panic!("expected TensorShapeMismatch, got {:?}", other.map(|_| ())),
    }
}
