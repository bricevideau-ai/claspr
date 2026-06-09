//! Sobel edge-detection image kernel, packaged as a reusable claspr
//! library crate. Reads from one RGBA8 image, writes the gradient
//! magnitude (as a grayscale image) to another. Companion to
//! `examples/mandelbrot-kernel/`; the two are composed in
//! `examples/image-pipeline/`.
//!
//! ```ignore
//! let sobel = sobel_kernel::kernels(&ctx)?;
//! let (input, output) = sobel.sobel([w, h], input_image, output_image, w, h).wait()?;
//! ```

#[claspr::device]
pub mod gpu {
    // spirv-std imports cfg-gated — sobel-kernel doesn't pull
    // spirv-std as a host dep. Builtin params + kernel bodies (where
    // these names appear) are discarded before host name resolution
    // by the `#[claspr::kernel]` macro, so resolving them only on
    // the spirv side is enough.
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        cl::Int2,
        glam::{USizeVec3, UVec4},
        num_traits::Float,
    };

    /// Rec. 601 luminance from RGB byte components. Pure scalar Rust
    /// — works equally on both sides, host doesn't need spirv-std to
    /// see this helper.
    fn luminance_rgb(r: u32, g: u32, b: u32) -> f32 {
        0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
    }

    #[claspr::kernel]
    pub fn sobel(
        #[spirv(global_invocation_id)] id: USizeVec3,
        input: &Image!(2D, type=u32, sampled=false),
        #[spirv(image_access = "write_only")] output: &mut Image!(2D, type=u32, sampled=false),
        width: u32,
        height: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        if px >= width || py >= height {
            return;
        }

        // Read a 3x3 luminance window with edge-clamped sampling.
        let x = px as i32;
        let y = py as i32;
        let read_lum = |dx: i32, dy: i32| -> f32 {
            let cx = (x + dx).clamp(0, width as i32 - 1);
            let cy = (y + dy).clamp(0, height as i32 - 1);
            let pixel: UVec4 = unsafe { input.read(Int2::new(cx, cy)) };
            luminance_rgb(pixel.x, pixel.y, pixel.z)
        };

        // Sobel kernels:
        //   Gx = | -1  0  1 |     Gy = | -1 -2 -1 |
        //        | -2  0  2 |          |  0  0  0 |
        //        | -1  0  1 |          |  1  2  1 |
        let l00 = read_lum(-1, -1);
        let l10 = read_lum(0, -1);
        let l20 = read_lum(1, -1);
        let l01 = read_lum(-1, 0);
        let l21 = read_lum(1, 0);
        let l02 = read_lum(-1, 1);
        let l12 = read_lum(0, 1);
        let l22 = read_lum(1, 1);

        let gx = -l00 + l20 - 2.0 * l01 + 2.0 * l21 - l02 + l22;
        let gy = -l00 - 2.0 * l10 - l20 + l02 + 2.0 * l12 + l22;

        let mag = (gx * gx + gy * gy).sqrt();
        let v = if mag >= 255.0 { 255 } else { mag as u32 };
        unsafe {
            output.write(Int2::new(x, y), UVec4::new(v, v, v, 255));
        }
    }
}

pub use gpu::{Kernels, kernels};
