//! Image kernels in a command buffer (design v2). An image kernel arg is a single
//! `cl_mem`, set exactly like a buffer object, so an image kernel records into a CB
//! via `clCommandNDRangeKernelKHR` like any buffer kernel — no image-specific FFI.
//!
//! This asserts a >= 2-command IMAGE-kernel chain (fill_pattern -> copy_to_buffer)
//! homes ONE command buffer on a CB-capable device and replays correctly.
//!
//! ICD note: rust-gpu image kernels SEGV on rusticl/llvmpipe (a known rust-gpu/llvmpipe
//! issue), so this test SKIPs there; it runs on pocl 7.2-pre and Intel iris. It uses a
//! 2D image (not 3D / arrayed) to avoid the rust-gpu vec3-coord write bug on those.

use claspr::eager::{DeviceOp, DeviceOpExt};
use claspr::image::format::R32Float;
use claspr::{Context, DeviceSlice, Image2D, ReadWrite};
use claspr_test_image_kernels::dim2_float;

const W: u32 = 8;
const H: u32 = 8;
const N: usize = (W * H) as usize;

/// Skip on no device, no image support, or rusticl/llvmpipe (image-kernel SEGV).
fn ctx() -> Option<Context> {
    let c = match Context::any() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return None;
        }
    };
    if !c.device().cl3().image_support().unwrap_or(false) {
        eprintln!("SKIP: device has no image support");
        return None;
    }
    // rust-gpu image kernels SEGV on rusticl/llvmpipe — skip there.
    let name = c.device().name().unwrap_or_default().to_lowercase();
    if name.contains("llvmpipe") {
        eprintln!("SKIP: rust-gpu image kernels SEGV on llvmpipe");
        return None;
    }
    Some(c)
}

fn homed_cb<O: DeviceOp>(g: &O) -> bool {
    g.cb_cache()
        .map(|c| c.lock().unwrap().is_some())
        .unwrap_or(false)
}

/// A two-image-kernel chain — `fill_pattern` (write image) then `copy_to_buffer`
/// (read image -> out buffer) — is a weight-2 all-device span: ONE command buffer,
/// homed at the root AndThen, replayed across syncs. The out buffer holds the
/// kernel-written pattern each run.
#[test]
fn image_kernel_chain_runs_as_command_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = dim2_float::kernels(&ctx).expect("load image kernels");

    let img = Image2D::<ReadWrite, R32Float>::alloc(&ctx, W, H).expect("alloc image");
    let out = DeviceSlice::<f32>::alloc_zero(&ctx, N).expect("alloc out");

    // fill_pattern writes pixel (x,y) = Vec4(x, y, 0, 1); copy_to_buffer reads back
    // the .x component into `out`, so out[y*W + x] == x. Two image-kernel commands →
    // weight 2 → CB.
    let g = ks
        .fill_pattern([W as usize, H as usize], img, W, H)
        .and_then(move |img| ks.copy_to_buffer([W as usize, H as usize], img, out, W, H));
    assert_eq!(
        g.cbable_weight(),
        2,
        "fill + copy image kernels = two commands"
    );

    let expected: Vec<f32> = (0..H).flat_map(|_y| (0..W).map(|x| x as f32)).collect();

    for i in 0..3 {
        let (_img_co, out_co) = g.sync(&ctx).unwrap_or_else(|e| panic!("sync {i}: {e:?}"));
        let got = out_co.map().wait().expect("read out");
        assert_eq!(&got[..], &expected[..], "iter {i}");
        drop(got);
        drop((_img_co, out_co));
    }

    // On a CB-capable device the chain MUST have taken a real command buffer — the
    // image args are recorded via clCommandNDRangeKernelKHR like buffer args.
    if ctx.has_cl_khr_command_buffer() {
        assert!(
            homed_cb(&g),
            "image-kernel chain should home a command buffer on a CB-capable device"
        );
    }
}
