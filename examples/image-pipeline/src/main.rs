//! End-to-end claspr **library composition** demo. Pulls in two
//! independent kernel libraries from the workspace —
//! `mandelbrot-kernel` and `sobel-kernel` — and runs them as a
//! two-stage image pipeline:
//!
//!   1. Render a Mandelbrot fractal into image A.
//!   2. Run Sobel edge detection: read image A, write the gradient
//!      magnitude (grayscale) into image B.
//!   3. Read image B back to host, save as PPM.
//!
//! What this demonstrates that the existing single-crate examples
//! don't:
//!
//! - **Library crates expose kernels at the crate root** via
//!   `pub use gpu::{Kernels, kernels};`, so the consumer reads
//!   `mandelbrot_kernel::kernels(&ctx)` rather than going through
//!   `::gpu::`.
//! - **The consumer needs no `build.rs` of its own** — each kernel
//!   library carries its own (each one's `build.rs` calls
//!   `claspr_build::compile_from_host("src/lib.rs")` and writes its
//!   SPV blob into its own `OUT_DIR`). The two libraries' generated
//!   modules don't collide; they live in different
//!   `<lib_pkg>::gpu::Kernels` types.
//! - **Two image kernels with different signatures** chain naturally:
//!   `mandelbrot` writes to one image, `sobel` reads from it and writes
//!   to a second.
//!
//! Run with `cargo run -p image-pipeline`. Writes `image-pipeline.ppm`.

use claspr::{Context, write_ppm_rgba8};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const MAX_ITER: u32 = 256;

/// Run the two-library pipeline and return the Sobel output pixels (RGBA8), or
/// `None` if there's no image-capable device (so the test can SKIP gracefully).
fn run() -> claspr::Result<Option<Vec<u8>>> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return Ok(None);
        }
    };

    if !ctx.device().cl3().image_support().unwrap_or(false) {
        eprintln!("SKIP: device has no image support");
        return Ok(None);
    }

    // Each library exposes its own typed launch handle. Loaded once
    // per Context; reused across launches.
    let mandelbrot = mandelbrot_kernel::kernels(&ctx)?;
    let sobel = sobel_kernel::kernels(&ctx)?;

    // Two RGBA8 images: `fractal` is mandelbrot's destination + sobel's
    // input; `edges` is sobel's destination.
    let fractal = claspr::Image2DRgba8::alloc(&ctx, WIDTH, HEIGHT)?;
    let edges = claspr::Image2DRgba8::alloc(&ctx, WIDTH, HEIGHT)?;

    // Stage 1: render the Mandelbrot set into `fractal`.
    let fractal = mandelbrot
        .mandelbrot(
            [WIDTH as usize, HEIGHT as usize],
            fractal,
            WIDTH,
            HEIGHT,
            MAX_ITER,
        )
        .wait()?;

    // Stage 2: edge-detect `fractal` into `edges`.
    let (_fractal, edges) = sobel
        .sobel(
            [WIDTH as usize, HEIGHT as usize],
            fractal,
            edges,
            WIDTH,
            HEIGHT,
        )
        .wait()?;

    let pixels = edges.read_bytes_alloc().wait()?;
    Ok(Some(pixels))
}

fn main() -> claspr::Result<()> {
    if let Some(pixels) = run()? {
        let ppm_path = "image-pipeline.ppm";
        write_ppm_rgba8(ppm_path, WIDTH, HEIGHT, &pixels)?;
        println!(
            "image-pipeline: wrote {ppm_path} ({WIDTH}x{HEIGHT}, mandelbrot → sobel via two library crates)",
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end regression for the cross-crate library composition: the
    /// mandelbrot→sobel pipeline runs without error and produces a full-size
    /// RGBA8 buffer that isn't uniformly blank — i.e. both library kernels
    /// actually executed and the second read the first's output. Guards against
    /// composition regressions that `cargo run` would print-but-not-catch.
    #[test]
    fn pipeline_produces_nonblank_output() {
        let Some(pixels) = run().expect("pipeline run") else {
            return; // no image-capable device — SKIP
        };
        assert_eq!(
            pixels.len(),
            (WIDTH * HEIGHT * 4) as usize,
            "output must be a full WIDTH*HEIGHT RGBA8 buffer"
        );
        // Sobel edge magnitudes on a Mandelbrot render are non-uniform: some
        // pixels are edges (bright), most aren't (dark). A pipeline that silently
        // produced nothing would leave an all-equal buffer.
        let first = pixels[0];
        assert!(
            pixels.iter().any(|&b| b != first),
            "edge-detected output must not be a uniform buffer"
        );
    }
}
