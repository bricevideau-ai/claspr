//! Single-file claspr example: a ray-marched signed-distance-field
//! scene with sun lighting + soft shadows + distance fog. Writes the
//! result to `raymarch.ppm`.
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
    use spirv_std::arch::opencl_std as ocl;
    use spirv_std::cl::{Float3, Int2};
    use spirv_std::glam::{USizeVec3, UVec4};
    use spirv_std::{Image, spirv};

    // f32::cos/sin/powf/exp on bare `f32` (no std) need
    // `num_traits::Float` in scope on the kernel side — the libm
    // intercept then rewrites them to `OpExtInst <OpenCL.std> {cos,
    // sin, pow, exp, …}`. On host they come from std, so the import
    // is unused there — cfg-gate to keep the host build clean.
    #[cfg(target_arch = "spirv")]
    use spirv_std::num_traits::Float;

    // ── Numeric tolerances ─────────────────────────────────────
    pub const EPSILON: f32 = 0.001;

    // ── Scene ──────────────────────────────────────────────────
    pub const SPHERE_A: Float3 = Float3::new(-0.7, 0.0, 0.0);
    pub const SPHERE_B: Float3 = Float3::new(0.6, -0.2, 0.0);
    pub const RADIUS_A: f32 = 0.9;
    pub const RADIUS_B: f32 = 0.7;
    pub const GROUND_Y: f32 = -0.9;
    pub const SMIN_K: f32 = 0.45;
    pub const GROUND_BIAS: f32 = 0.01;

    // ── Primary ray march ──────────────────────────────────────
    pub const MAX_STEPS: u32 = 96;
    pub const MAX_DIST: f32 = 30.0;

    // ── Soft-shadow march ──────────────────────────────────────
    pub const SHADOW_STEPS: u32 = 32;
    pub const SHADOW_MAX_DIST: f32 = 12.0;
    pub const SHADOW_T_START: f32 = 0.02;
    pub const SHADOW_K: f32 = 8.0;
    pub const SHADOW_STEP_MIN: f32 = 0.05;
    pub const SHADOW_STEP_MAX: f32 = 0.5;

    // ── Camera ─────────────────────────────────────────────────
    pub const CAM_RO: Float3 = Float3::new(3.0, 1.6, 4.0);
    pub const CAM_TARGET: Float3 = Float3::ZERO;
    pub const CAM_WORLD_UP: Float3 = Float3::Y;
    pub const FOV_SCALE: f32 = 0.7;

    // ── Sun & shading ──────────────────────────────────────────
    pub const SUN_AZ: f32 = 0.7;
    pub const SUN_EL: f32 = 0.6;
    pub const SUN_COLOR: Float3 = Float3::new(1.0, 0.95, 0.85);
    pub const AMBIENT: f32 = 0.15;
    pub const DIFFUSE: f32 = 0.7;
    pub const SPECULAR_POWER: f32 = 32.0;
    pub const FOG_DENSITY: f32 = 0.06;

    // ── Sky gradient ───────────────────────────────────────────
    pub const SKY_ZENITH: Float3 = Float3::new(0.30, 0.55, 0.85);
    pub const SKY_BAND: Float3 = Float3::new(0.85, 0.78, 0.62);
    pub const SKY_HORIZON_LO: f32 = -0.05;
    pub const SKY_HORIZON_HI: f32 = 0.45;

    // ── Surface palette ────────────────────────────────────────
    pub const COLOR_GROUND: Float3 = Float3::new(0.55, 0.55, 0.60);
    pub const COLOR_BLOB: Float3 = Float3::new(0.85, 0.55, 0.40);

    pub fn ray_at(ro: Float3, rd: Float3, t: f32) -> Float3 {
        rd.mul_add(Float3::splat(t), ro)
    }

    pub fn smin(a: f32, b: f32, k: f32) -> f32 {
        let h = ocl::clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
        ocl::mix(b, a, h) - k * h * (1.0 - h)
    }

    pub fn scene_sdf(p: Float3) -> f32 {
        let d_a = p.distance(SPHERE_A) - RADIUS_A;
        let d_b = p.distance(SPHERE_B) - RADIUS_B;
        let blob = smin(d_a, d_b, SMIN_K);
        let plane = p.y() - GROUND_Y;
        blob.min(plane)
    }

    pub fn scene_normal(p: Float3) -> Float3 {
        let ex = Float3::new(EPSILON, 0.0, 0.0);
        let ey = Float3::new(0.0, EPSILON, 0.0);
        let ez = Float3::new(0.0, 0.0, EPSILON);
        let dx = scene_sdf(p + ex) - scene_sdf(p - ex);
        let dy = scene_sdf(p + ey) - scene_sdf(p - ey);
        let dz = scene_sdf(p + ez) - scene_sdf(p - ez);
        Float3::new(dx, dy, dz).normalize()
    }

    pub fn march(ro: Float3, rd: Float3) -> (bool, f32) {
        let mut t = 0.0f32;
        let mut i = 0u32;
        while i < MAX_STEPS {
            let d = scene_sdf(ray_at(ro, rd, t));
            if d < EPSILON {
                return (true, t);
            }
            t += d;
            if t > MAX_DIST {
                break;
            }
            i += 1;
        }
        (false, t)
    }

    pub fn soft_shadow(ro: Float3, rd: Float3) -> f32 {
        let mut t = SHADOW_T_START;
        let mut res = 1.0f32;
        let mut i = 0u32;
        while i < SHADOW_STEPS {
            let d = scene_sdf(ray_at(ro, rd, t));
            if d < EPSILON {
                return 0.0;
            }
            res = res.min(SHADOW_K * d / t);
            t += ocl::clamp(d, SHADOW_STEP_MIN, SHADOW_STEP_MAX);
            if t > SHADOW_MAX_DIST {
                break;
            }
            i += 1;
        }
        ocl::clamp(res, 0.0, 1.0)
    }

    pub fn sky(rd: Float3) -> Float3 {
        let h = ocl::smoothstep(SKY_HORIZON_LO, SKY_HORIZON_HI, rd.y());
        SKY_BAND.lerp(SKY_ZENITH, h)
    }

    pub fn shade(p: Float3, n: Float3, ro: Float3, sun: Float3, base: Float3) -> Float3 {
        let view = (ro - p).normalize();
        let ndotl = ocl::clamp(n.dot(sun), 0.0, 1.0);
        let shadow = soft_shadow(p, sun);

        let refl = n * (2.0 * view.dot(n)) - view;
        let spec = ocl::clamp(refl.dot(sun), 0.0, 1.0).powf(SPECULAR_POWER) * shadow;

        let diff = DIFFUSE * ndotl * shadow;
        base * (SUN_COLOR * diff + Float3::splat(AMBIENT)) + SUN_COLOR * spec
    }

    /// Per-pixel colour at NDC `(u, v)`. Same code path runs on both
    /// the kernel (per work item) and the host (validation harness in
    /// `main.rs`).
    pub fn pixel_color(u: f32, v: f32) -> Float3 {
        let forward = (CAM_TARGET - CAM_RO).normalize();
        let right = forward.cross(CAM_WORLD_UP).normalize();
        let cam_up = right.cross(forward);
        let rd = (forward + (right * (u * FOV_SCALE)) + (cam_up * (v * FOV_SCALE))).normalize();

        let sun = Float3::new(
            SUN_EL.cos() * SUN_AZ.sin(),
            SUN_EL.sin(),
            SUN_EL.cos() * SUN_AZ.cos(),
        )
        .normalize();

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

    let kernels = compiled::Kernels::load(&ctx)?;
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
