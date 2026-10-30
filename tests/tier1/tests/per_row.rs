//! `PerRow<T>` / `PerRowIn<T>` kernel arguments: the buffer is split into one
//! equal row per work-item (any 1-, 2- or 3-D grid, row `i` for the row-major
//! linear id `i`), and the launch checks the length divides exactly
//! (`RowGridMismatch` otherwise).

use claspr::{Buffer, DeviceSlice, Error};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const ROWS: usize = 12;
const COLS: usize = 7;

fn matrix(ctx: &claspr::Context) -> (Vec<u32>, DeviceSlice<u32>) {
    let v: Vec<u32> = (0..(ROWS * COLS) as u32)
        .map(|i| (i * 37) % 101 + 5)
        .collect();
    let d = DeviceSlice::<u32>::from_slice(ctx, &v).expect("alloc");
    (v, d)
}

fn read(d: DeviceSlice<u32>) -> Vec<u32> {
    let mut v = vec![0u32; d.len()];
    d.read(&mut v).wait().expect("read");
    v
}

#[test]
fn per_row_in_reads_this_work_items_row() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (v, m) = matrix(&ctx);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, ROWS).expect("alloc");
    let (_, out) = ks.row_sums_u32([ROWS], m, out).wait().expect("launch");
    let expected: Vec<u32> = v.chunks(COLS).map(|r| r.iter().sum()).collect();
    assert_eq!(read(out), expected);
}

/// Rows follow the row-major linear id, so a 2-D grid works the same.
#[test]
fn per_row_on_a_two_dimensional_grid() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (v, m) = matrix(&ctx);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, ROWS).expect("alloc");
    let (_, out) = ks.row_sums_u32([4, 3], m, out).wait().expect("launch");
    let expected: Vec<u32> = v.chunks(COLS).map(|r| r.iter().sum()).collect();
    assert_eq!(read(out), expected);
}

#[test]
fn per_row_updates_its_row_in_place() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (v, m) = matrix(&ctx);
    let m = ks.subtract_row_min_u32([ROWS], m).wait().expect("launch");
    let expected: Vec<u32> = v
        .chunks(COLS)
        .flat_map(|r| {
            let min = *r.iter().min().unwrap();
            r.iter().map(move |x| x - min)
        })
        .collect();
    assert_eq!(read(m), expected);
}

/// A length the grid doesn't split into equal rows is rejected before launch.
#[test]
fn per_row_length_must_split_evenly() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (_, m) = matrix(&ctx);
    match ks.subtract_row_min_u32([ROWS + 1], m).wait() {
        Err(Error::RowGridMismatch { arg, len, .. }) => assert_eq!((arg, len), ("m", ROWS * COLS)),
        other => panic!("expected RowGridMismatch, got {:?}", other.map(|_| ())),
    }
}
