//! Eager port of `image_transfer.rs`: image transfer combinator coverage via
//! the eager graph API. [`image_upload`](claspr::eager::image_upload) /
//! [`image_download`](claspr::eager::image_download) generic over every owning
//! image type via [`ImageHostTransfer`](claspr::ImageHostTransfer).
//!
//! Old → new mapping:
//!   `claspr_async::image_upload::<I>(px, dims)`   → `claspr::eager::image_upload::<I>(px, dims)`
//!   `claspr_async::image_download::<I>`           → `claspr::eager::image_download::<I>`
//!   `.and_then(image_download::<I>).sync()`       → same, terminal yields the Vec
//!
//! Same shapes, same pixel patterns, same N, same round-trip assertions.

use claspr::eager::{DeviceOpExt, image_download, image_upload};
use claspr::{
    Context, Image1D, Image1DArray, Image2D, Image2DArray, Image3D, ReadWrite,
    image::format::R32Uint,
};

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

/// `Image2D` round-trip — upload → download. Confirms the `(u32, u32)` Dims path.
#[test]
fn image2d_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 8;
    const H: u32 = 4;
    let pixels: Vec<u32> = (0..(W * H)).map(|i| 0xCAFE_0000 | i).collect();

    let result = image_upload::<Image2D<ReadWrite, R32Uint>>(pixels.clone(), (W, H))
        .and_then(image_download::<Image2D<ReadWrite, R32Uint>>)
        .sync(&ctx)
        .expect("chain")
        .into_inner();

    assert_eq!(result, pixels);
}

/// `Image1D` round-trip — confirms the `u32` Dims path (no tuple wrap).
#[test]
fn image1d_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 16;
    let pixels: Vec<u32> = (0..W).map(|i| i * 13 + 5).collect();

    let result = image_upload::<Image1D<ReadWrite, R32Uint>>(pixels.clone(), W)
        .and_then(image_download::<Image1D<ReadWrite, R32Uint>>)
        .sync(&ctx)
        .expect("chain")
        .into_inner();

    assert_eq!(result, pixels);
}

/// `Image3D` round-trip — confirms the `(u32, u32, u32)` Dims path.
#[test]
fn image3d_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 3;
    const D: u32 = 2;
    let pixels: Vec<u32> = (0..(W * H * D)).map(|i| i + 1000).collect();

    let result = image_upload::<Image3D<ReadWrite, R32Uint>>(pixels.clone(), (W, H, D))
        .and_then(image_download::<Image3D<ReadWrite, R32Uint>>)
        .sync(&ctx)
        .expect("chain")
        .into_inner();

    assert_eq!(result, pixels);
}

/// `Image1DArray` round-trip — confirms the `(u32, u32)` Dims path for the
/// array variant; layers laid out contiguously.
#[test]
fn image1d_array_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 6;
    const LAYERS: u32 = 3;
    let pixels: Vec<u32> = (0..(W * LAYERS)).map(|i| 0xA000 + i).collect();

    let result = image_upload::<Image1DArray<ReadWrite, R32Uint>>(pixels.clone(), (W, LAYERS))
        .and_then(image_download::<Image1DArray<ReadWrite, R32Uint>>)
        .sync(&ctx)
        .expect("chain")
        .into_inner();

    assert_eq!(result, pixels);
}

/// `Image2DArray` round-trip — confirms the `(u32, u32, u32)` Dims path for the
/// 2D-array variant.
#[test]
fn image2d_array_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 3;
    const LAYERS: u32 = 2;
    let pixels: Vec<u32> = (0..(W * H * LAYERS)).map(|i| 0xB000 + i).collect();

    let result = image_upload::<Image2DArray<ReadWrite, R32Uint>>(pixels.clone(), (W, H, LAYERS))
        .and_then(image_download::<Image2DArray<ReadWrite, R32Uint>>)
        .sync(&ctx)
        .expect("chain")
        .into_inner();

    assert_eq!(result, pixels);
}
