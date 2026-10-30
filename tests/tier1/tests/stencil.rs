//! Stencil kernel arguments: `Periodic<T>` / `Clamped<T>` (unpadded buffer, the
//! grid is the buffer), `Halo<T, R..>` (the buffer is the grid plus `R` halo
//! cells per side) — all `Stencil<T, B>` — and `Interior<T, R..>`, the
//! read-write centre of a halo-padded buffer. The launch checks the buffer is
//! exactly the (padded) grid; a mismatch is `HaloGridMismatch` / `GridMismatch`.

use claspr::{Buffer, DeviceSlice, Error};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const W: usize = 8;
const H: usize = 6;

fn field(ctx: &claspr::Context, w: usize, h: usize) -> (Vec<u32>, DeviceSlice<u32>) {
    let v: Vec<u32> = (0..(w * h) as u32).map(|i| i * 7 + 1).collect();
    let d = DeviceSlice::<u32>::from_slice(ctx, &v).expect("alloc");
    (v, d)
}

fn read(d: DeviceSlice<u32>) -> Vec<u32> {
    let mut v = vec![0u32; d.len()];
    d.read(&mut v).wait().expect("read");
    v
}

/// Host reference: sum of the 4 neighbours of `(x, y)` in a `w x h` field,
/// resolving an out-of-range coordinate with `fix`.
fn neighbours(v: &[u32], w: usize, h: usize, fix: impl Fn(isize, usize) -> usize) -> Vec<u32> {
    let mut out = vec![0; w * h];
    for y in 0..h {
        for x in 0..w {
            let at =
                |dx: isize, dy: isize| v[fix(y as isize + dy, h) * w + fix(x as isize + dx, w)];
            out[y * w + x] = at(-1, 0) + at(1, 0) + at(0, -1) + at(0, 1);
        }
    }
    out
}

#[test]
fn periodic_wraps_at_the_edges() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (v, f) = field(&ctx, W, H);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, W * H).expect("alloc");
    let (_, out) = ks
        .neighbours_periodic_u32([W, H], f, out)
        .wait()
        .expect("launch");
    let wrap = |i: isize, e: usize| i.rem_euclid(e as isize) as usize;
    assert_eq!(read(out), neighbours(&v, W, H, wrap));
}

#[test]
fn clamped_reads_the_edge_cell() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (v, f) = field(&ctx, W, H);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, W * H).expect("alloc");
    let (_, out) = ks
        .neighbours_clamped_u32([W, H], f, out)
        .wait()
        .expect("launch");
    let clamp = |i: isize, e: usize| i.clamp(0, e as isize - 1) as usize;
    assert_eq!(read(out), neighbours(&v, W, H, clamp));
}

/// `Halo<u32, 1>`: the buffer is `[W + 2, H + 2]`, the grid the `[W, H]`
/// interior; work-item `(x, y)` is padded cell `(x + 1, y + 1)`.
#[test]
fn halo_reads_into_the_padding() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (pw, ph) = (W + 2, H + 2);
    let (v, f) = field(&ctx, pw, ph);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, W * H).expect("alloc");
    let (_, out) = ks
        .neighbours_halo_u32([W, H], f, out)
        .wait()
        .expect("launch");
    let mut expected = vec![0; W * H];
    for y in 0..H {
        for x in 0..W {
            let at = |dx: isize, dy: isize| {
                v[(y as isize + 1 + dy) as usize * pw + (x as isize + 1 + dx) as usize]
            };
            expected[y * W + x] = at(-1, 0) + at(1, 0) + at(0, -1) + at(0, 1) + 1000 * at(0, 0);
        }
    }
    assert_eq!(read(out), expected);
}

/// `Stencil<u32, Padded<0, 0, 1>>` on a 3-D grid: padded along z only.
#[test]
fn spelled_out_stencil_pads_one_axis() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (w, h, d) = (4, 3, 5);
    let (v, f) = field(&ctx, w * h, d + 2);
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, w * h * d).expect("alloc");
    let (_, out) = ks
        .z_neighbours_u32([w, h, d], f, out)
        .wait()
        .expect("launch");
    let plane = w * h;
    let expected: Vec<u32> = (0..w * h * d)
        .map(|i| {
            let (xy, z) = (i % plane, i / plane);
            v[z * plane + xy] + v[(z + 2) * plane + xy]
        })
        .collect();
    assert_eq!(read(out), expected);
}

/// `Interior<u32, 1>` writes the interior and leaves the halo untouched.
#[test]
fn interior_writes_only_the_interior() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (pw, ph) = (W + 2, H + 2);
    let (v, src) = field(&ctx, pw, ph);
    let dst = DeviceSlice::<u32>::from_slice(&ctx, &vec![9u32; pw * ph]).expect("alloc");
    let (_, dst) = ks
        .interior_add_one_u32([W, H], src, dst)
        .wait()
        .expect("launch");
    let got = read(dst);
    for y in 0..ph {
        for x in 0..pw {
            let interior = (1..=W).contains(&x) && (1..=H).contains(&y);
            let want = if interior { v[y * pw + x] + 1 } else { 9 };
            assert_eq!(got[y * pw + x], want, "cell ({x}, {y})");
        }
    }
}

/// A `Halo` buffer that isn't exactly the padded grid is rejected before launch.
#[test]
fn halo_buffer_must_be_the_padded_grid() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (_, f) = field(&ctx, W, H); // unpadded: one halo cell per side missing
    let out = DeviceSlice::<u32>::alloc_uninit(&ctx, W * H).expect("alloc");
    match ks.neighbours_halo_u32([W, H], f, out).wait() {
        Err(Error::HaloGridMismatch { arg, pad, len, .. }) => {
            assert_eq!((arg, pad, len), ("f", [1, 1, 0], W * H));
        }
        other => panic!("expected HaloGridMismatch, got {:?}", other.map(|_| ())),
    }
}
