//! Tiny [PPM (P6)](https://en.wikipedia.org/wiki/Netpbm#File_formats)
//! writer. Just enough to round-trip an RGBA8 image read from a kernel
//! to a file you can open in any image viewer.

use crate::Result;
use std::path::Path;

/// Write an `RGBA8` byte buffer to a PPM (P6) file at `path`.
///
/// The alpha channel is dropped — PPM only stores RGB. `pixels` must be
/// exactly `width * height * 4` bytes long.
pub fn write_ppm_rgba8(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<()> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or("image dimensions overflow usize")?;
    if pixels.len() != expected {
        return Err(format!(
            "pixel buffer is {} bytes, expected {} ({}x{} RGBA8)",
            pixels.len(),
            expected,
            width,
            height
        )
        .into());
    }
    let mut out = Vec::with_capacity(expected / 4 * 3 + 32);
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    for chunk in pixels.chunks_exact(4) {
        out.push(chunk[0]);
        out.push(chunk[1]);
        out.push(chunk[2]);
    }
    std::fs::write(path, &out)?;
    Ok(())
}
