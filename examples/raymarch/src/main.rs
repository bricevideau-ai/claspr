//! Multi-file claspr example: a ray-marched signed-distance-field
//! scene with sun lighting + soft shadows + distance fog. Writes the
//! result to `raymarch.ppm`. Demonstrates that a `#[claspr::device]`
//! module can be split across files using ordinary `mod foo;`
//! declarations — claspr-build follows them with rustc's normal
//! file-resolution rules and inlines the bodies into the generated
//! kernel sub-crate.
//!
//! `#![feature(proc_macro_hygiene)]` is required at crate level
//! because `mod foo;` (file modules) inside a proc-macro's input is
//! gated by that feature on nightly (rust-lang/rust#54727). Single-
//! file kernel modules don't need it; only the multi-file form does.
#![feature(proc_macro_hygiene)]

//!
//! Layout mirrors `examples/collatz`: top-level is host code (use
//! statements, `mod compiled`, `fn main`), and the entire device side
//! lives inside one `#[claspr::device] mod gpu { ... }` block. The
//! build script lifts the module body verbatim into the generated
//! kernel sub-crate; `claspr-build`'s slim preamble adds only
//! `#![cfg_attr(target_arch = "spirv", no_std)]`, so the user's `use
//! spirv_std::*;` lines come along to the kernel crate as written.
//!
//! The host validates a handful of pixels by calling `gpu::pixel_color`
//! directly — the same function the kernel calls per pixel — to prove
//! the host arms of `cl::Float3` arithmetic + `opencl_std::*` math
//! intrinsics produce sensible output, on top of the round-trip
//! through OpenCL.

use claspr::Context;

#[claspr::device]
mod gpu {
    // Submodules — bodies live in src/gpu/scene.rs and src/gpu/shading.rs.
    // claspr-build follows these `mod` declarations the same way rustc
    // does, inlining each file into the generated kernel sub-crate.
    pub mod scene;
    pub mod shading;

    use spirv_std::Image;
    use spirv_std::cl::{Float3, Int2};
    use spirv_std::glam::{USizeVec3, UVec4};

    // f32::exp on bare `f32` (no std) needs `num_traits::Float` in
    // scope on the kernel side — the libm intercept rewrites it to
    // `OpExtInst <OpenCL.std> exp`. On host it comes from std.
    #[cfg(target_arch = "spirv")]
    use spirv_std::num_traits::Float;

    use scene::{GROUND_BIAS, GROUND_Y, march, ray_at, scene_normal};
    use shading::{COLOR_BLOB, COLOR_GROUND, FOG_DENSITY, shade, sky, sun_dir};

    // ── Camera ─────────────────────────────────────────────────
    pub const CAM_RO: Float3 = Float3::new(3.0, 1.6, 4.0);
    pub const CAM_TARGET: Float3 = Float3::ZERO;
    pub const CAM_WORLD_UP: Float3 = Float3::Y;
    pub const FOV_SCALE: f32 = 0.7;

    /// Per-pixel colour at NDC `(u, v)`. Same code path runs on both
    /// the kernel (per work item) and the host (validation harness in
    /// `main.rs`).
    pub fn pixel_color(u: f32, v: f32) -> Float3 {
        let forward = (CAM_TARGET - CAM_RO).normalize();
        let right = forward.cross(CAM_WORLD_UP).normalize();
        let cam_up = right.cross(forward);
        let rd = (forward + (right * (u * FOV_SCALE)) + (cam_up * (v * FOV_SCALE))).normalize();

        let sun = sun_dir();
        let (hit, t) = march(CAM_RO, rd);

        if hit {
            let p = ray_at(CAM_RO, rd, t);
            let n = scene_normal(p);
            let base = if p.y() < GROUND_Y + GROUND_BIAS {
                COLOR_GROUND
            } else {
                COLOR_BLOB
            };
            let surf = shade(p, n, CAM_RO, sun, base);
            let fog = (-t * FOG_DENSITY).exp();
            sky(rd).lerp(surf, fog)
        } else {
            sky(rd)
        }
    }

    #[claspr::kernel]
    pub fn raymarch(
        #[spirv(global_invocation_id)] id: USizeVec3,
        image: &mut Image!(2D, type=u32, sampled=false),
        width: u32,
        height: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        if px >= width || py >= height {
            return;
        }

        // Pixel → NDC, with y flipped so origin is top-left.
        let aspect = width as f32 / height as f32;
        let u = (2.0 * (px as f32 + 0.5) / width as f32 - 1.0) * aspect;
        let v = 1.0 - 2.0 * (py as f32 + 0.5) / height as f32;

        let color = pixel_color(u, v);

        // Saturate, scale to 0..255, pack into the (R, G, B, A=255)
        // tuple the OpenCL CL_RGBA8 image expects.
        let rgb = (color.clamp(Float3::ZERO, Float3::ONE) * 255.0).as_uint3();
        let rgba = rgb.extend(255);
        let out = UVec4::from_array(rgba.to_array());

        let coord = Int2::new(px as i32, py as i32);
        unsafe {
            image.write(coord, out);
        }
    }
}

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn run() -> claspr::Result<bool> {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return Ok(false);
        }
    };

    if !ctx.device().image_support().unwrap_or(false) {
        eprintln!("SKIP: device has no image support");
        return Ok(false);
    }

    let kernels = gpu::kernels(&ctx)?;
    let img = ctx.alloc_image_2d_rgba8(WIDTH, HEIGHT)?;
    kernels.raymarch(&ctx, [WIDTH as usize, HEIGHT as usize], &img, WIDTH, HEIGHT)?;
    let pixels = ctx.read_image_2d_rgba8(&img)?;

    // Host vs. device pixel comparison. Walking every pixel through
    // `pixel_color` on the CPU is doable but slow — we stride by `STEP`
    // so the binary stays snappy while still covering a few thousand
    // pixels spread across the frame.
    //
    // Tolerance: OpenCL math intrinsics (sqrt/sin/cos/pow/exp) have
    // implementation-defined precision per the spec, so we don't
    // assume bit-for-bit matches between pocl's CPU JIT and host
    // libm. ±2 per channel passes everywhere we've measured; bump
    // if real divergence shows up on a different runtime.
    const STEP: u32 = 20;
    const TOL: u8 = 2;
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let (mut compared, mut max_diff): (usize, u8) = (0, 0);
    for py in (0..HEIGHT).step_by(STEP as usize) {
        for px in (0..WIDTH).step_by(STEP as usize) {
            let pixel_base = ((py * WIDTH + px) * 4) as usize;
            let device = [
                pixels[pixel_base],
                pixels[pixel_base + 1],
                pixels[pixel_base + 2],
            ];

            let u = (2.0 * (px as f32 + 0.5) / WIDTH as f32 - 1.0) * aspect;
            let v = 1.0 - 2.0 * (py as f32 + 0.5) / HEIGHT as f32;
            let host_color = gpu::pixel_color(u, v).to_array();
            let host = [
                (host_color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (host_color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (host_color[2].clamp(0.0, 1.0) * 255.0) as u8,
            ];

            for c in 0..3 {
                let diff = device[c].abs_diff(host[c]);
                assert!(
                    diff <= TOL,
                    "device/host mismatch at ({px},{py}) channel {c}: \
                     device={} host={} diff={diff} > tol {TOL}",
                    device[c],
                    host[c],
                );
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            compared += 1;
        }
    }

    let ppm_path = "raymarch.ppm";
    claspr::write_ppm_rgba8(ppm_path, WIDTH, HEIGHT, &pixels)?;
    println!(
        "raymarch: wrote {ppm_path} ({WIDTH}x{HEIGHT}, {compared} pixels host-validated, max channel delta {max_diff})",
    );

    Ok(true)
}

fn main() -> claspr::Result<()> {
    let _ = run()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::gpu::pixel_color;

    /// Top-left corner — purely sky (no scene geometry in view), and
    /// blue-dominant.
    #[test]
    fn pixel_color_top_left_is_blue_sky() {
        let c = pixel_color(-1.0, 1.0).to_array();
        assert!(
            c[2] > c[0] && c[2] > c[1],
            "expected blue-dominant sky at top-left, got {c:?}",
        );
    }

    /// Centre of frame — should hit geometry and produce a finite,
    /// in-range colour.
    #[test]
    fn pixel_color_centre_is_finite() {
        let c = pixel_color(0.0, 0.0).to_array();
        for (i, &x) in c.iter().enumerate() {
            assert!(
                x.is_finite() && (-0.01..=1.01).contains(&x),
                "component {i} out of range: {x}",
            );
        }
    }
}
