//! Mandelbrot-set image kernel, packaged as a reusable claspr library
//! crate. Pair with `examples/sobel-kernel/` and `examples/image-pipeline/`
//! to see the cross-crate composition story end to end.
//!
//! The host calls into this library by:
//!
//! ```ignore
//! let mandelbrot = mandelbrot_kernel::kernels(&ctx)?;
//! let image = mandelbrot.mandelbrot([w, h], image, w, h, max_iter).wait()?;
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
    use num_complex::Complex32;
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        cl::Int2,
        glam::{USizeVec3, UVec4},
    };

    /// Iterate the Mandelbrot recurrence `z = z² + c` from `z = 0`
    /// until escape (|z|² > 4) or `max_iter`. Outlined into its own
    /// function — rust-gpu's optimiser may keep this as a separate
    /// `OpFunction` rather than inlining into the kernel, which used
    /// to trip both pocl and rusticl on image kernels; see
    /// `[[reference_pocl_image_complex_hang]]` for the history.
    /// Both runtimes handle this shape cleanly now.
    ///
    /// Pure Rust + `num_complex` only — host-callable for validation.
    pub fn mandelbrot_iter(c: Complex32, max_iter: u32) -> u32 {
        let mut z = Complex32::new(0.0, 0.0);
        let mut i = 0u32;
        while i < max_iter {
            if z.norm_sqr() > 4.0 {
                break;
            }
            z = z * z + c;
            i += 1;
        }
        i
    }

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
        #[spirv(image_access = "write_only")] image: &mut Image!(2D, type=u32, sampled=false),
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

        let iter = mandelbrot_iter(Complex32::new(cx, cy), max_iter);

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
