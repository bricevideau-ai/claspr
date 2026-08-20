//! Image kernels in a command buffer (design v2). An image kernel arg is a single
//! `cl_mem`, set exactly like a buffer object, so an image kernel records into a CB
//! via `clCommandNDRangeKernelKHR` like any buffer kernel — no image-specific FFI.
//!
//! This asserts a >= 2-command IMAGE-kernel chain (fill_pattern -> copy_to_buffer)
//! homes ONE command buffer on a CB-capable device and replays correctly.
//!
//! Runs on all three ICDs — via the CB path where the driver's command buffer
//! advertises the commands (pocl), else per-op (see `ctx`). Both prior image caveats
//! are fixed: the rusticl/llvmpipe image-kernel SEGV (Mesa 26.1.4 / LLVM 21) and the
//! 3D/2D-array vec3-coord write bug (a rust-gpu codegen bug — coordinates now widened
//! to 4 components for Kernel targets on the opencl-kernel-support branch; pocl was
//! always spec-conformant). This test uses a 2D image simply because that's the
//! shape the dim2_float test kernels expose.

use claspr::eager::{DeviceOp, DeviceOpExt};
use claspr::image::format::R32Float;
use claspr::{DeviceSlice, Image2D, ReadWrite};
use claspr_test_image_kernels::dim2_float;
use claspr_test_support::{ctx_with_images, homed_cb};

const W: u32 = 8;
const H: u32 = 8;
const N: usize = (W * H) as usize;

/// A two-image-kernel chain — `fill_pattern` (write image) then `copy_to_buffer`
/// (read image -> out buffer) — is a weight-2 all-device span: ONE command buffer,
/// homed at the root AndThen, replayed across syncs. The out buffer holds the
/// kernel-written pattern each run.
#[test]
fn image_kernel_chain_runs_as_command_buffer() {
    let Some(ctx) = ctx_with_images() else { return };
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

/// An image FILL then image→image COPY is a weight-2 all-device span:
/// `clCommandFillImageKHR` + `clCommandCopyImageKHR` in ONE command buffer (where the
/// extension provides the image commands; else software fallback). `eager_image_copy`
/// is the pipe-fed graph copy (the concrete `copy_to` can't chain off `and_then`).
#[test]
fn image_fill_then_copy_runs_as_command_buffer() {
    let Some(ctx) = ctx_with_images() else { return };
    use claspr::eager_image_copy;
    use claspr::image::format::R32G32B32A32Uint;
    let pattern: [u32; 4] = [11, 22, 33, 44];

    let src = Image2D::<ReadWrite, R32G32B32A32Uint>::alloc(&ctx, W, H).expect("alloc src");
    let dst = Image2D::<ReadWrite, R32G32B32A32Uint>::alloc(&ctx, W, H).expect("alloc dst");

    // fill(src) -> eager_image_copy(src -> dst): two image commands → weight 2 → CB.
    // Annotate the piped src so `eager_image_copy`'s Src generic is pinned.
    let g = src.fill(pattern).and_then(
        move |src: claspr::Pipe<Image2D<ReadWrite, R32G32B32A32Uint>>| eager_image_copy(src, dst),
    );
    assert_eq!(g.cbable_weight(), 2, "image fill + copy = two commands");

    type Img = Image2D<ReadWrite, R32G32B32A32Uint>;
    for i in 0..3 {
        let (_src_co, dst_co): (claspr::eager::Checkout<Img>, claspr::eager::Checkout<Img>) =
            g.sync(&ctx).unwrap_or_else(|e| panic!("sync {i}: {e:?}"));
        let got: Vec<[u32; 4]> = dst_co.read_alloc().wait().expect("read dst");
        assert_eq!(got.len(), N, "iter {i} len");
        assert!(
            got.iter().all(|&px| px == pattern),
            "iter {i}: copy should carry the fill pattern; got {:?}",
            &got[..2]
        );
        drop((_src_co, dst_co));
    }

    if ctx.has_cl_khr_command_buffer() && homed_cb(&g) {
        eprintln!("image fill+copy: recorded a command buffer");
    } else {
        eprintln!("image fill+copy: software fallback (image CB commands unavailable)");
    }
}

/// An image FILL followed by a kernel that READS the image is a weight-2 all-device
/// span: `clCommandFillImageKHR` + `clCommandNDRangeKernelKHR` in ONE command buffer.
#[test]
fn image_fill_then_kernel_runs_as_command_buffer() {
    let Some(ctx) = ctx_with_images() else { return };
    let ks = dim2_float::kernels(&ctx).expect("load image kernels");
    let fill_x: f32 = 7.0;

    let img = Image2D::<ReadWrite, R32Float>::alloc(&ctx, W, H).expect("alloc image");
    let out = DeviceSlice::<f32>::alloc_zero(&ctx, N).expect("alloc out");

    // img.fill([fill_x; 4]) writes fill_x to every pixel channel; copy_to_buffer reads
    // .x into out. Fill (image command) + kernel = weight 2 → CB.
    let g = img
        .fill([fill_x; 4])
        .and_then(move |img| ks.copy_to_buffer([W as usize, H as usize], img, out, W, H));
    assert_eq!(g.cbable_weight(), 2, "image fill + kernel = two commands");

    for i in 0..3 {
        let (_img_co, out_co) = g.sync(&ctx).unwrap_or_else(|e| panic!("sync {i}: {e:?}"));
        let got = out_co.map().wait().expect("read out");
        assert!(
            got.iter().all(|&v| v == fill_x),
            "iter {i}: every out element should be the fill .x ({fill_x}); got {:?}",
            &got[..4]
        );
        drop(got);
        drop((_img_co, out_co));
    }

    // Where the driver's command buffer supports clCommandFillImageKHR (>= 0.9.4) the
    // chain homes a real CB; otherwise it falls back to software — both correct.
    if ctx.has_cl_khr_command_buffer() && homed_cb(&g) {
        eprintln!("image fill+kernel: recorded a command buffer");
    } else {
        eprintln!("image fill+kernel: software fallback (image fill CB command unavailable)");
    }
}
