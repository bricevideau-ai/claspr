//! Sun, sky, soft shadows, and surface shading. Lifted into the kernel
//! sub-crate by claspr-build when it follows `mod shading;` from
//! `src/main.rs`'s `#[claspr::device] mod gpu`.
//!
//! Cross-file imports work as expected: `use super::scene::...` here
//! resolves through the kernel sub-crate's module tree just as it does
//! in the host build, because claspr-build inlines the submodule bodies
//! while preserving their `mod` structure.

use spirv_std::arch::opencl_std as ocl;
use spirv_std::cl::Float3;

// `f32::cos`/`sin`/`powf` on bare `f32` (no std) need
// `num_traits::Float` in scope on the kernel side — the libm
// intercept rewrites them to `OpExtInst <OpenCL.std> {cos, sin, pow}`.
// On host they come from std.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::scene::{EPSILON, ray_at, scene_sdf};

// ── Soft-shadow march ─────────────────────────────────────────
pub const SHADOW_STEPS: u32 = 32;
pub const SHADOW_MAX_DIST: f32 = 12.0;
pub const SHADOW_T_START: f32 = 0.02; // step away from the surface to dodge self-shadow
pub const SHADOW_K: f32 = 8.0; // penumbra width (smaller = softer)
pub const SHADOW_STEP_MIN: f32 = 0.05; // clamp the per-step distance
pub const SHADOW_STEP_MAX: f32 = 0.5;

// ── Sun & shading ─────────────────────────────────────────────
pub const SUN_AZ: f32 = 0.7;
pub const SUN_EL: f32 = 0.6;
pub const SUN_COLOR: Float3 = Float3::new(1.0, 0.95, 0.85);
pub const AMBIENT: f32 = 0.15;
pub const DIFFUSE: f32 = 0.7;
pub const SPECULAR_POWER: f32 = 32.0;
pub const FOG_DENSITY: f32 = 0.06;

// ── Sky gradient ──────────────────────────────────────────────
pub const SKY_ZENITH: Float3 = Float3::new(0.30, 0.55, 0.85);
pub const SKY_BAND: Float3 = Float3::new(0.85, 0.78, 0.62);
pub const SKY_HORIZON_LO: f32 = -0.05;
pub const SKY_HORIZON_HI: f32 = 0.45;

// ── Surface palette ───────────────────────────────────────────
pub const COLOR_GROUND: Float3 = Float3::new(0.55, 0.55, 0.60);
pub const COLOR_BLOB: Float3 = Float3::new(0.85, 0.55, 0.40);

/// Sun direction from spherical (azimuth, elevation) coords.
pub fn sun_dir() -> Float3 {
    Float3::new(
        SUN_EL.cos() * SUN_AZ.sin(),
        SUN_EL.sin(),
        SUN_EL.cos() * SUN_AZ.cos(),
    )
    .normalize()
}

/// Cone-marched soft shadow. `SHADOW_K` controls penumbra width.
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

    // Phong specular: reflect view about normal, dot with sun, raise to power.
    let refl = n * (2.0 * view.dot(n)) - view;
    let spec = ocl::clamp(refl.dot(sun), 0.0, 1.0).powf(SPECULAR_POWER) * shadow;

    let diff = DIFFUSE * ndotl * shadow;
    base * (SUN_COLOR * diff + Float3::splat(AMBIENT)) + SUN_COLOR * spec
}
