//! Runtime exercise of claspr's image trait dispatch.
//!
//! What this file actually proves (over and above the examples,
//! which only ever use the default `R8G8B8A8Uint` 2D RGBA8 shape):
//!
//! - Non-default formats round-trip through `clCreateImage` +
//!   `clEnqueueWriteImage` + `clEnqueueReadImage` for every
//!   sampled-type family the trait dispatch advertises:
//!   `R32Uint` (Uint, single channel), `R32Sint` (Sint, single
//!   channel), `R32Float` (Float, single channel), and
//!   `R32G32B32A32Uint` (Uint, four channel).
//! - The kernel-side `&Image!(2D, type=u32, sampled=false)` /
//!   `&Image!(2D, type=f32, …)` / `&Image!(2D, type=i32, …)`
//!   actually accepts host-side `Image2D<A, F>` whose
//!   `F::SampledFamily` matches the kernel's `type=`. Mismatches
//!   are covered by `image_compile_fail.rs` (compile-fail tests
//!   via `trybuild`).
//! - The `WriteOnly` (host) and `ReadOnly` (host) access markers
//!   each bind to a kernel whose access qualifier matches
//!   (write-only fill kernel ↔ `Image2D<WriteOnly, _>`; read-only
//!   image→buffer kernel ↔ `Image2D<ReadOnly, _>`).
//!
//! All tests skip cleanly when no OpenCL device is available or
//! the device doesn't advertise image support.

use claspr::{
    Context, DeviceSlice, Image1D, Image1DBuffer, Image1DBufferView, Image3D, ReadOnly, ReadWrite,
    WriteOnly,
    image::format::{R8G8B8A8Uint, R32Float, R32G32B32A32Uint, R32Sint, R32Uint},
};

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<Context> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return None;
        }
    };
    if !ctx.device().cl3().image_support().unwrap_or(false) {
        eprintln!("SKIP: device has no image support");
        return None;
    }
    Some(ctx)
}

/// Write-only 2D Uint image, default `R8G8B8A8Uint` format
/// (this is what the examples already exercise — included here
/// as a baseline so the new trait dispatch can be compared
/// against the existing-success path on this machine).
#[test]
fn fill_pattern_rgba8_uint() {
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim2_uint::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R8G8B8A8Uint>::alloc(&ctx, W, H).unwrap();
    let img = kernels
        .fill_pattern([W as usize, H as usize], img, W, H)
        .wait(&ctx)
        .unwrap();
    let bytes = img.download_bytes(&ctx).unwrap();
    // Pixel (0,0) is value 0 → R=0; pixel (1,0) → R=1; pixel (0,1) → R=W (16).
    // Each pixel is 4 bytes (RGBA8); R channel is byte 0 of each pixel.
    assert_eq!(bytes[0], 0); // pixel (0,0) R
    assert_eq!(bytes[4], 1); // pixel (1,0) R
    assert_eq!(bytes[(W as usize) * 4], W as u8); // pixel (0,1) R
    assert_eq!(bytes[3], 0xFF); // pixel (0,0) A
}

/// Write-only 2D Uint image, **non-default** `R32Uint` format
/// (single-channel u32). Proves the Uint family accepts formats
/// other than RGBA8.
#[test]
fn fill_pattern_r32_uint() {
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim2_uint::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32Uint>::alloc(&ctx, W, H).unwrap();
    let img = kernels
        .fill_pattern([W as usize, H as usize], img, W, H)
        .wait(&ctx)
        .unwrap();
    let pixels: Vec<u32> = img.download(&ctx).unwrap();
    // The kernel writes (x + y*W, 0, 0, 0xFFFF_FFFF) per pixel.
    // R32Uint is single-channel so only the .x part survives the
    // read-back — the other components are dropped by the hardware
    // per the OpenCL image storage spec.
    assert_eq!(pixels.len(), (W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let got = pixels[(y * W + x) as usize];
            let want = x + y * W;
            assert_eq!(got, want, "pixel ({x},{y}): got {got}, want {want}");
        }
    }
}

/// Write-only 2D Uint image, four-channel `R32G32B32A32Uint`
/// format. Proves the kernel's UVec4-output write_imageui survives
/// the wider channel layout.
#[test]
fn fill_pattern_rgba32_uint() {
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim2_uint::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32G32B32A32Uint>::alloc(&ctx, W, H).unwrap();
    let img = kernels
        .fill_pattern([W as usize, H as usize], img, W, H)
        .wait(&ctx)
        .unwrap();
    let pixels: Vec<[u32; 4]> = img.download(&ctx).unwrap();
    for y in 0..H {
        for x in 0..W {
            let got = pixels[(y * W + x) as usize];
            assert_eq!(got[0], x + y * W);
            assert_eq!(got[1], 0);
            assert_eq!(got[2], 0);
            assert_eq!(got[3], 0xFFFF_FFFF);
        }
    }
}

/// Write-only 2D Float image, `R32Float` format. Proves the
/// Float family.
#[test]
fn fill_pattern_r32_float() {
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim2_float::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32Float>::alloc(&ctx, W, H).unwrap();
    let img = kernels
        .fill_pattern([W as usize, H as usize], img, W, H)
        .wait(&ctx)
        .unwrap();
    let pixels: Vec<f32> = img.download(&ctx).unwrap();
    for y in 0..H {
        for x in 0..W {
            let got = pixels[(y * W + x) as usize];
            // Single-channel format keeps only .x = px as f32.
            let want = x as f32;
            assert_eq!(got, want, "pixel ({x},{y}): got {got}, want {want}");
        }
    }
}

/// Write-only 2D Sint image, `R32Sint` format. Proves the Sint
/// family.
#[test]
fn fill_pattern_r32_sint() {
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim2_sint::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32Sint>::alloc(&ctx, W, H).unwrap();
    let img = kernels
        .fill_pattern([W as usize, H as usize], img, W, H)
        .wait(&ctx)
        .unwrap();
    let pixels: Vec<i32> = img.download(&ctx).unwrap();
    for y in 0..H {
        for x in 0..W {
            let got = pixels[(y * W + x) as usize];
            // Kernel writes (px - py, -(px - py), 0, 1); single channel keeps .x.
            let want = (x as i32) - (y as i32);
            assert_eq!(got, want, "pixel ({x},{y}): got {got}, want {want}");
        }
    }
}

/// Read-only 2D Float image (host-seeded via `upload`) →
/// kernel copies pixels into a `DeviceSlice<f32>`. Proves the
/// `&Image` (ReadOnly access qualifier) kernel-param path and
/// the host `upload` API at the same time.
#[test]
fn read_only_float_image_to_buffer() {
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim2_float::kernels(&ctx).unwrap();

    let mut img = claspr::Image2D::<ReadOnly, R32Float>::alloc(&ctx, W, H).unwrap();
    let mut seed = vec![0.0f32; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            seed[(y * W + x) as usize] = (x as f32) + (y as f32) * 100.0;
        }
    }
    img.upload(&ctx, &seed).unwrap();

    // Seed with finite values so the `out[i] * 0.0` trick in the
    // kernel produces a clean zero (NaN otherwise).
    let zeros = vec![0.0f32; (W * H) as usize];
    let out = DeviceSlice::<f32>::from_slice(&ctx, &zeros).unwrap();

    let (_img, out) = kernels
        .copy_to_buffer([W as usize, H as usize], img, out, W, H)
        .wait(&ctx)
        .unwrap();
    let mut result = vec![0.0f32; (W * H) as usize];
    out.read(&ctx, &mut result).wait().unwrap();
    assert_eq!(result, seed, "kernel-read pixels should match host-seeded");
}

/// Read-only 2D Sint image, host-seeded → kernel-copied. Sint
/// family + `&Image` read path.
#[test]
fn read_only_sint_image_to_buffer() {
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim2_sint::kernels(&ctx).unwrap();

    let mut img = claspr::Image2D::<ReadOnly, R32Sint>::alloc(&ctx, W, H).unwrap();
    let mut seed = vec![0i32; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            seed[(y * W + x) as usize] = (x as i32) - (y as i32) * 10;
        }
    }
    img.upload(&ctx, &seed).unwrap();

    let zeros = vec![0i32; (W * H) as usize];
    let out = DeviceSlice::<i32>::from_slice(&ctx, &zeros).unwrap();

    let (_img, out) = kernels
        .copy_to_buffer([W as usize, H as usize], img, out, W, H)
        .wait(&ctx)
        .unwrap();
    let mut result = vec![0i32; (W * H) as usize];
    out.read(&ctx, &mut result).wait().unwrap();
    assert_eq!(result, seed);
}

/// 1D image — write-only, `R32Uint` format. Proves the
/// `Image1D` runtime + the `KernelImage1DWriteArg<Uint>` trait
/// bound + the rust-gpu auto-declare of `OpCapability Image1D`
/// for `OpTypeImage Dim=1D` on Kernel (without that auto-declare,
/// spirv-val rejects the module).
#[test]
fn dim1_fill_pattern_r32_uint() {
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim1_uint::kernels(&ctx).unwrap();
    let img = Image1D::<WriteOnly, R32Uint>::alloc(&ctx, W).unwrap();
    let img = kernels
        .fill_pattern([W as usize], img, W)
        .wait(&ctx)
        .unwrap();
    let pixels: Vec<u32> = img.download(&ctx).unwrap();
    assert_eq!(pixels.len(), W as usize);
    for x in 0..W {
        assert_eq!(pixels[x as usize], x, "pixel {x}");
    }
}

/// 3D image — write-only, `R32Uint` format. Proves the
/// `Image3D` runtime + the `KernelImage3DWriteArg<Uint>` trait
/// bound emitted by the proc-macro for `&mut Image!(3D, …)`.
/// (Dim=3D doesn't carry a per-Dim capability requirement in
/// the SPIR-V core spec, so it works on the same OpenCL 1.2
/// `image()` build preset as Dim=2D.)
#[test]
fn dim3_fill_pattern_r32_uint() {
    const D: u32 = 4;
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim3_uint::kernels(&ctx).unwrap();
    let img = Image3D::<WriteOnly, R32Uint>::alloc(&ctx, W, H, D).unwrap();
    let img = kernels
        .fill_pattern([W as usize, H as usize, D as usize], img, W, H, D)
        .wait(&ctx)
        .unwrap();
    let pixels: Vec<u32> = img.download(&ctx).unwrap();
    assert_eq!(pixels.len(), (W * H * D) as usize);
    for z in 0..D {
        for y in 0..H {
            for x in 0..W {
                let idx = (z * W * H + y * W + x) as usize;
                let want = x + y * W + z * W * H;
                assert_eq!(pixels[idx], want, "voxel ({x},{y},{z})");
            }
        }
    }
}

/// Image-buffer (1D image backed by a `cl_mem` buffer) — write
/// path. Proves the `Image1DBuffer` runtime + the new
/// `KernelImageBufferWriteArg<Uint>` trait bound + the rust-gpu
/// auto-declare of `OpCapability ImageBuffer` for
/// `OpTypeImage Dim=Buffer`.
#[test]
fn dim_buffer_fill_pattern_r32_uint() {
    const N: u32 = 64;
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim_buffer_uint::kernels(&ctx).unwrap();
    let img = Image1DBuffer::<WriteOnly, R32Uint>::alloc(&ctx, N).unwrap();
    let img = kernels
        .fill_pattern([N as usize], img, N)
        .wait(&ctx)
        .unwrap();
    let pixels: Vec<u32> = img.download(&ctx).unwrap();
    assert_eq!(pixels.len(), N as usize);
    for x in 0..N {
        assert_eq!(pixels[x as usize], x.wrapping_mul(3));
    }
}

/// Image-buffer **view** over an existing `DeviceSlice`.
/// Allocates a `DeviceSlice<u32, ReadWrite>` and seeds it via
/// `from_slice`, constructs an `Image1DBufferView<'_, ReadWrite,
/// R32Uint>` over it (no copy — shared cl_mem), and reads it
/// through the image-buffer kernel. The kernel-read values
/// should match the host-written ones byte-for-byte.
#[test]
fn dim_buffer_view_of_slice() {
    const N: u32 = 64;
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim_buffer_uint::kernels(&ctx).unwrap();

    // Host writes through the slice path.
    let seed: Vec<u32> = (0..N).map(|x| x.wrapping_mul(13).wrapping_add(5)).collect();
    let slice = DeviceSlice::<u32>::from_slice(&ctx, &seed).unwrap();

    // No copy — the view shares the slice's cl_mem.
    let view = Image1DBufferView::<ReadWrite, R32Uint>::view_of(&slice).unwrap();
    assert_eq!(view.width(), N);

    // Output slice for the kernel to write into.
    let zeros = vec![0u32; N as usize];
    let out = DeviceSlice::<u32>::from_slice(&ctx, &zeros).unwrap();

    // Kernel reads the view (as `image1d_buffer_t`), writes to out.
    let (_view, out) = kernels
        .copy_to_buffer([N as usize], view, out, N)
        .wait(&ctx)
        .unwrap();

    let mut result = vec![0u32; N as usize];
    out.read(&ctx, &mut result).wait().unwrap();
    assert_eq!(
        result, seed,
        "kernel-read pixels through view should match host-seeded slice"
    );
}

/// Image-buffer view over a slice with reinterpret: the slice
/// is `DeviceSlice<u32>` but the view sees it as `R32Uint`
/// (matching), exercising the byte-length arithmetic in
/// `view_of` (16 u32 elements = 16 R32Uint pixels = 64 bytes).
#[test]
fn dim_buffer_view_width_derived_from_slice_bytes() {
    const N: u32 = 16;
    let Some(ctx) = ctx() else { return };
    let zeros = vec![0u32; N as usize];
    let slice = DeviceSlice::<u32>::from_slice(&ctx, &zeros).unwrap();
    let view = Image1DBufferView::<ReadWrite, R32Uint>::view_of(&slice).unwrap();
    // 16 u32 = 64 bytes; pixel size 4 → 16 pixels.
    assert_eq!(view.width(), N);
}

/// Image-buffer read path. Seed via `upload`, kernel reads
/// pixels and copies the .x channel to a `DeviceSlice<u32>`.
/// Proves the `&Image!(buffer, …)` → `KernelImageBufferReadArg`
/// bound + cross-arg (image-buffer + slice) on the same launch.
#[test]
fn dim_buffer_read_to_slice() {
    const N: u32 = 64;
    let Some(ctx) = ctx() else { return };
    let kernels = claspr_test_image_kernels::dim_buffer_uint::kernels(&ctx).unwrap();

    let mut img = Image1DBuffer::<ReadOnly, R32Uint>::alloc(&ctx, N).unwrap();
    let seed: Vec<u32> = (0..N).map(|x| x * 7 + 1).collect();
    img.upload(&ctx, &seed).unwrap();

    let zeros = vec![0u32; N as usize];
    let out = DeviceSlice::<u32>::from_slice(&ctx, &zeros).unwrap();

    let (_img, out) = kernels
        .copy_to_buffer([N as usize], img, out, N)
        .wait(&ctx)
        .unwrap();
    let mut result = vec![0u32; N as usize];
    out.read(&ctx, &mut result).wait().unwrap();
    assert_eq!(result, seed);
}
