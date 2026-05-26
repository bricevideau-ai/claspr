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

fn run() -> claspr::Result<bool> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return Ok(false);
        }
    };

    if !ctx.device().cl3().image_support().unwrap_or(false) {
        eprintln!("SKIP: device has no image support");
        return Ok(false);
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
    mandelbrot
        .mandelbrot(
            &ctx,
            [WIDTH as usize, HEIGHT as usize],
            &fractal,
            WIDTH,
            HEIGHT,
            MAX_ITER,
        )
        .wait()?;

    // Stage 2: edge-detect `fractal` into `edges`.
    sobel
        .sobel(
            &ctx,
            [WIDTH as usize, HEIGHT as usize],
            &fractal,
            &edges,
            WIDTH,
            HEIGHT,
        )
        .wait()?;

    let pixels = edges.download_bytes(&ctx)?;
    let ppm_path = "image-pipeline.ppm";
    write_ppm_rgba8(ppm_path, WIDTH, HEIGHT, &pixels)?;
    println!(
        "image-pipeline: wrote {ppm_path} ({WIDTH}x{HEIGHT}, mandelbrot → sobel via two library crates)",
    );
    Ok(true)
}

fn main() -> claspr::Result<()> {
    let _ = run()?;
    Ok(())
}
