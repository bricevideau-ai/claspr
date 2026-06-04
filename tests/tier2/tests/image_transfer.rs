//! Tier 2 image transfer combinator coverage —
//! [`image_upload`](claspr_async::image_upload) /
//! [`image_download`](claspr_async::image_download) generic over
//! every owning image type via [`ImageHostTransfer`](claspr::ImageHostTransfer).
//!
//! Each test takes the same shape: build a chain that uploads a
//! known pixel pattern, optionally runs an identity-shape kernel
//! (so the chain has device work between upload and download),
//! and downloads the result. Confirms the round-trip per image
//! type.
//!
//! No kernel-side image processing is exercised here — that's
//! `tier1/tests/image_dispatch.rs`'s job. What's under test is
//! that the combinator pair (1) allocates the right image shape
//! for each `Dims` type, (2) wires non-blocking upload + download
//! through the chain's event graph, (3) returns the host Vec at
//! `.sync()` time with the right pixel content.

use claspr::{
    Context, Image1D, Image1DArray, Image2D, Image2DArray, Image3D, ReadWrite,
    image::format::R32Uint,
};
use claspr_async::{DeviceOperation, image_download, image_upload};

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

/// `Image2D` round-trip — upload → download. Confirms the
/// `(u32, u32)` Dims path.
#[test]
fn image2d_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 8;
    const H: u32 = 4;
    let pixels: Vec<u32> = (0..(W * H)).map(|i| 0xCAFE_0000 | i).collect();

    let result: Vec<u32> = image_upload::<Image2D<ReadWrite, R32Uint>>(pixels.clone(), (W, H))
        .and_then(image_download::<Image2D<ReadWrite, R32Uint>>)
        .sync(&ctx)
        .expect("chain");

    assert_eq!(result, pixels);
}

/// `Image1D` round-trip — confirms the `u32` Dims path (no
/// tuple wrap).
#[test]
fn image1d_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 16;
    let pixels: Vec<u32> = (0..W).map(|i| i * 13 + 5).collect();

    let result: Vec<u32> = image_upload::<Image1D<ReadWrite, R32Uint>>(pixels.clone(), W)
        .and_then(image_download::<Image1D<ReadWrite, R32Uint>>)
        .sync(&ctx)
        .expect("chain");

    assert_eq!(result, pixels);
}

/// `Image3D` round-trip — confirms the `(u32, u32, u32)` Dims
/// path.
#[test]
fn image3d_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 3;
    const D: u32 = 2;
    let pixels: Vec<u32> = (0..(W * H * D)).map(|i| i + 1000).collect();

    let result: Vec<u32> = image_upload::<Image3D<ReadWrite, R32Uint>>(pixels.clone(), (W, H, D))
        .and_then(image_download::<Image3D<ReadWrite, R32Uint>>)
        .sync(&ctx)
        .expect("chain");

    assert_eq!(result, pixels);
}

/// `Image1DArray` round-trip — confirms the `(u32, u32)` Dims
/// path for the array variant; layers laid out contiguously.
#[test]
fn image1d_array_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 6;
    const LAYERS: u32 = 3;
    let pixels: Vec<u32> = (0..(W * LAYERS)).map(|i| 0xA000 + i).collect();

    let result: Vec<u32> =
        image_upload::<Image1DArray<ReadWrite, R32Uint>>(pixels.clone(), (W, LAYERS))
            .and_then(image_download::<Image1DArray<ReadWrite, R32Uint>>)
            .sync(&ctx)
            .expect("chain");

    assert_eq!(result, pixels);
}

/// `Image2DArray` round-trip — confirms the `(u32, u32, u32)`
/// Dims path for the 2D-array variant.
#[test]
fn image2d_array_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 3;
    const LAYERS: u32 = 2;
    let pixels: Vec<u32> = (0..(W * H * LAYERS)).map(|i| 0xB000 + i).collect();

    let result: Vec<u32> =
        image_upload::<Image2DArray<ReadWrite, R32Uint>>(pixels.clone(), (W, H, LAYERS))
            .and_then(image_download::<Image2DArray<ReadWrite, R32Uint>>)
            .sync(&ctx)
            .expect("chain");

    assert_eq!(result, pixels);
}
