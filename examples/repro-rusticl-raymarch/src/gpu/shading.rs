//! Lighting + sky. (Copy of examples/raymarch/src/gpu/shading.rs.)

use spirv_std::arch::opencl_std as ocl;
use spirv_std::cl::Float3;

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::scene::{EPSILON, ray_at, scene_sdf};

pub const SHADOW_STEPS: u32 = 32;
pub const SHADOW_MAX_DIST: f32 = 12.0;
pub const SHADOW_T_START: f32 = 0.02;
pub const SHADOW_K: f32 = 8.0;
pub const SHADOW_STEP_MIN: f32 = 0.05;
pub const SHADOW_STEP_MAX: f32 = 0.5;

pub const SUN_AZ: f32 = 0.7;
pub const SUN_EL: f32 = 0.6;
pub const SUN_COLOR: Float3 = Float3::new(1.0, 0.95, 0.85);
pub const AMBIENT: f32 = 0.15;
pub const DIFFUSE: f32 = 0.7;
pub const SPECULAR_POWER: f32 = 32.0;
pub const FOG_DENSITY: f32 = 0.06;

pub const SKY_ZENITH: Float3 = Float3::new(0.30, 0.55, 0.85);
pub const SKY_BAND: Float3 = Float3::new(0.85, 0.78, 0.62);
pub const SKY_HORIZON_LO: f32 = -0.05;
pub const SKY_HORIZON_HI: f32 = 0.45;

pub const COLOR_GROUND: Float3 = Float3::new(0.55, 0.55, 0.60);
pub const COLOR_BLOB: Float3 = Float3::new(0.85, 0.55, 0.40);

pub fn sun_dir() -> Float3 {
    Float3::new(
        SUN_EL.cos() * SUN_AZ.sin(),
        SUN_EL.sin(),
        SUN_EL.cos() * SUN_AZ.cos(),
    )
    .normalize()
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
