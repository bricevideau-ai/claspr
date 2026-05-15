//! Mandelbrot-set image kernel, packaged as a reusable claspr library
//! crate. Pair with `examples/sobel-kernel/` and `examples/image-pipeline/`
//! to see the cross-crate composition story end to end.
//!
//! The host calls into this library by:
//!
//! ```ignore
//! let mandelbrot = mandelbrot_kernel::kernels(&ctx)?;
//! mandelbrot.mandelbrot(&ctx, [w, h], &image, w, h, max_iter)?;
//! ```
//!
//! No build.rs in the consuming crate is needed — this library carries
//! its own.

#[claspr::device]
pub mod gpu {
    // The host crate doesn't depend on spirv-std, so spirv-std use
    // statements are cfg-gated to the kernel-side build. The
    // `#[claspr::kernel]` proc-macro discards builtin params and the
    // kernel body before host name resolution touches them, so these
    // imports are only ever resolved on the spirv target.
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        cl::Int2,
        glam::{USizeVec3, UVec4},
    };

    /// Map iteration count to an `(R, G, B)` u32 triple. Pure integer
    /// math so the kernel doesn't need any transcendental imports —
    /// three different multiples of `t = iter * 255 / max_iter` mod
    /// 256 give a striped escape colour ramp (R / G / B band each
    /// time the iteration count crosses a threshold).
    fn color(iter: u32, max_iter: u32) -> (u32, u32, u32) {
        if iter >= max_iter {
            (0, 0, 0)
        } else {
            let t = (iter * 255 / max_iter) & 0xff;
            (t, (t * 7) & 0xff, (t * 13) & 0xff)
        }
    }

    #[claspr::kernel]
    pub fn mandelbrot(
        #[spirv(global_invocation_id)] id: USizeVec3,
        image: &mut Image!(2D, type=u32, sampled=false),
        width: u32,
        height: u32,
        max_iter: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        if px >= width || py >= height {
            return;
        }

        // Map pixel coordinates to the standard Mandelbrot view:
        //   x in [-2.5, 1.0], y in [-1.0, 1.0]
        let cx = (px as f32 / width as f32) * 3.5 - 2.5;
        let cy = (py as f32 / height as f32) * 2.0 - 1.0;

        let mut zx = 0.0f32;
        let mut zy = 0.0f32;
        let mut iter = 0u32;
        while iter < max_iter {
            let zx2 = zx * zx;
            let zy2 = zy * zy;
            if zx2 + zy2 > 4.0 {
                break;
            }
            let new_zx = zx2 - zy2 + cx;
            let new_zy = 2.0 * zx * zy + cy;
            zx = new_zx;
            zy = new_zy;
            iter += 1;
        }

        let (r, g, b) = color(iter, max_iter);
        let coord = Int2::new(px as i32, py as i32);
        unsafe {
            image.write(coord, UVec4::new(r, g, b, 255));
        }
    }
}

// Re-export the typed launch handle + loader at the crate root so
// downstream code can write `mandelbrot_kernel::kernels(&ctx)?` and
// `mandelbrot_kernel::Kernels` without going through `::gpu::`.
pub use gpu::{Kernels, kernels};
