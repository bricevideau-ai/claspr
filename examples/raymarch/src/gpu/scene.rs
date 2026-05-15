//! Scene geometry: SDF, march, and the constants that describe the
//! scene primitives. Lifted into the kernel sub-crate by claspr-build
//! when it sees `mod scene;` inside the `#[claspr::device] mod gpu` in
//! `src/main.rs`.

use spirv_std::arch::opencl_std as ocl;
use spirv_std::cl::Float3;

// ── Numeric tolerance ─────────────────────────────────────────
pub const EPSILON: f32 = 0.001;

// ── Scene primitives ──────────────────────────────────────────
pub const SPHERE_A: Float3 = Float3::new(-0.7, 0.0, 0.0);
pub const SPHERE_B: Float3 = Float3::new(0.6, -0.2, 0.0);
pub const RADIUS_A: f32 = 0.9;
pub const RADIUS_B: f32 = 0.7;
pub const GROUND_Y: f32 = -0.9;
pub const SMIN_K: f32 = 0.45;
// Bias when picking the ground vs. blob palette: hit just above the
// plane still counts as ground (avoids flicker on grazing hits).
pub const GROUND_BIAS: f32 = 0.01;

// ── Primary ray march ─────────────────────────────────────────
pub const MAX_STEPS: u32 = 96;
pub const MAX_DIST: f32 = 30.0;

pub fn ray_at(ro: Float3, rd: Float3, t: f32) -> Float3 {
    rd.mul_add(Float3::splat(t), ro)
}

/// Polynomial smooth-min — blends two SDFs over a radius `k`. The
/// scalar `clamp` stays as `ocl::clamp` because `f32::clamp` from
/// `core` inlines as branchy Rust source on SPIR-V (no fast-path),
/// while `ocl::clamp` lowers to a single `OpExtInst Fclamp`.
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
