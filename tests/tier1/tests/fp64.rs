//! Tier 1 coverage for f64 runtime arg-marshalling.
//!
//! The bulk of `tests/kernels/` is u32-only by design (max
//! portability), so claspr's typed-launch + `KernelArg` path for
//! `f64` slices + scalars has never been exercised through these
//! tests. These three cases close that gap:
//!
//! 1. `fill_f64` writes the scalar to every slot.
//! 2. `scale_f64` multiplies every slot by the scalar.
//! 3. The two compose in a fill → scale pipeline through the typed
//!    launcher (`.wait(&ctx)`).
//!
//! Each test skips when the device doesn't advertise the `Float64`
//! capability (most GPUs do, llvmpipe needs `RUSTICL_FEATURES=fp64`
//! — already set in CI). f64 kernel-side codegen lives in
//! `rust-gpu`'s upstream difftest suite; here we're purely validating
//! that the runtime hands the right bytes to `clSetKernelArg` for
//! f64 scalars and that `DeviceSlice<f64>` round-trips correctly.

use claspr::{Context, DeviceSlice};
use claspr_test_kernels::kernels_f64 as kernels;

const N: usize = 256;

fn ctx_with_f64() -> Option<Context> {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return None;
    };
    // `CL_DEVICE_DOUBLE_FP_CONFIG` returns a bit-field of rounding
    // modes the device supports for f64. Non-zero = some level of
    // f64 support; zero = none. (opencl3's `double_fp_config()` may
    // also `Err` on devices that don't recognise the query — treat
    // that as "no f64.")
    match ctx.device().cl3().double_fp_config() {
        Ok(mask) if mask != 0 => Some(ctx),
        _ => {
            eprintln!("SKIP: device has no Float64 capability");
            None
        }
    }
}

#[test]
fn fill_f64_writes_value_to_every_element() {
    let Some(ctx) = ctx_with_f64() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // alloc → write 0.0s so the `data[i] * 0.0` in fill_f64 sees a
    // finite value (uninit could be NaN — see the kernel-lib module
    // docs). Then fill, then read back.
    let initial = vec![0.0f64; N];
    let mut readback = vec![-1.0f64; N];
    let mut buf = DeviceSlice::<f64>::alloc_zero(&ctx, N).expect("alloc");
    buf.write(&initial).wait(&ctx).expect("write zeros");

    let buf = kernels
        .fill_f64([N], buf, 1.5)
        .wait(&ctx)
        .expect("fill_f64");
    buf.read(&mut readback).wait(&ctx).expect("read");

    for (i, &v) in readback.iter().enumerate() {
        assert_eq!(v, 1.5, "element {i} mismatch");
    }
}

#[test]
fn scale_f64_multiplies_each_element() {
    let Some(ctx) = ctx_with_f64() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let initial = vec![2.0f64; N];
    let mut readback = vec![-1.0f64; N];
    let mut buf = DeviceSlice::<f64>::alloc_zero(&ctx, N).expect("alloc");
    buf.write(&initial).wait(&ctx).expect("write 2.0s");

    let buf = kernels
        .scale_f64([N], buf, 3.0)
        .wait(&ctx)
        .expect("scale_f64");
    buf.read(&mut readback).wait(&ctx).expect("read");

    for (i, &v) in readback.iter().enumerate() {
        assert_eq!(v, 6.0, "element {i} mismatch");
    }
}

#[test]
fn fill_then_scale_pipeline_via_typed_launchers() {
    let Some(ctx) = ctx_with_f64() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let initial = vec![0.0f64; N];
    let mut readback = vec![-1.0f64; N];
    let mut buf = DeviceSlice::<f64>::alloc_zero(&ctx, N).expect("alloc");
    buf.write(&initial).wait(&ctx).expect("write zeros");

    let buf = kernels
        .fill_f64([N], buf, 0.25)
        .wait(&ctx)
        .expect("fill_f64");
    let buf = kernels
        .scale_f64([N], buf, 8.0)
        .wait(&ctx)
        .expect("scale_f64");
    buf.read(&mut readback).wait(&ctx).expect("read");

    for (i, &v) in readback.iter().enumerate() {
        assert_eq!(v, 2.0, "element {i} mismatch");
    }
}
