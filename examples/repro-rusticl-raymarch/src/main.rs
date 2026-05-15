// `#[claspr::device] mod gpu { mod foo; mod bar; ... }` (multi-file
// device module) is gated on nightly because file modules in
// proc-macro input are unstable (rust-lang/rust#54727). The flag is
// crate-level so the proc-macro can't auto-inject it.
#![feature(proc_macro_hygiene)]

//! Bisecting reproducer for the raymarch rusticl crash.
//!
//! The full raymarch sample (examples/raymarch) segfaults on rusticl
//! during program load + launch. Several plausible triggers, given
//! what raymarch does that the passing samples don't:
//!
//!   - heavy `cl::Float3` ops (OpTypeVector + OpDot/OpExtInst length etc.)
//!   - heavy `OpenCL.std` math intrinsics (sqrt/sin/cos/pow/exp/clamp/mix/smoothstep)
//!   - long compute loops (96-step ray march, 32-step soft shadow)
//!   - many helper-function calls from the kernel (per the bitwise
//!     reproducer's finding: helper-with-multi-arm-match crashes
//!     rusticl)
//!
//! We start with the full pixel_color path as a kernel and progressively
//! strip it down. The first variant that PASSES on rusticl tells us
//! what tipped the previous variant over.
//!
//! Run on rusticl:
//!   OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd \
//!   RUSTICL_ENABLE=llvmpipe \
//!   cargo run -p repro-rusticl-raymarch
//!
//! The line printed last (without OK / ERROR) identifies which
//! variant killed the process.

use claspr::Context;
use spirv_std::cl::Float3;

const W: u32 = 320;
const H: u32 = 180;

/// Absolute minimum: write a constant `Float3` to a `&mut [Float3]`
/// slice. No math. No `OpenCL.std` intrinsics. If this crashes,
/// the trigger is just "writing a vector type to a vector slice".
#[claspr::device]
pub mod write_const_float3 {
    use spirv_std::cl::Float3;
    use spirv_std::glam::USizeVec3;

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [Float3],
        width: u32,
        _height: u32,
    ) {
        let px = _id.x as u32;
        let py = _id.y as u32;
        let c = Float3::new(0.5, 0.6, 0.7);
        unsafe {
            *out.get_unchecked_mut((py * width + px) as usize) = c;
        }
    }
}

/// Like the above but builds the Float3 from per-pixel scalar
/// inputs. Adds `Float3::new(u, v, 1.0)` (composite construction).
#[claspr::device]
pub mod write_per_pixel_float3 {
    use spirv_std::cl::Float3;
    use spirv_std::glam::USizeVec3;

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [Float3],
        width: u32,
        _height: u32,
    ) {
        let px = _id.x as u32;
        let py = _id.y as u32;
        let u = px as f32 / width as f32 - 0.5;
        let v = py as f32 / width as f32 - 0.5;
        let c = Float3::new(u, v, 1.0);
        unsafe {
            *out.get_unchecked_mut((py * width + px) as usize) = c;
        }
    }
}

/// Hand-rolled normalize using OpenCL.std intrinsics directly: emits
/// `length` + componentwise divide. Avoids `Float3::normalize()` which
/// rust-gpu lowers via an out-of-line helper function. If this passes
/// → trigger is the `OpFunctionCall %ext_normalize` shape rust-gpu
/// emits, not the math itself.
#[claspr::device]
pub mod write_normalized_inline {
    use spirv_std::cl::Float3;
    use spirv_std::glam::USizeVec3;

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [Float3],
        width: u32,
        _height: u32,
    ) {
        let px = _id.x as u32;
        let py = _id.y as u32;
        let u = px as f32 / width as f32 - 0.5;
        let v = py as f32 / width as f32 - 0.5;
        let raw = Float3::new(u, v, 1.0);
        // Inline normalize: c / length(c).
        let len = spirv_std::arch::opencl_std::length(raw);
        let c = raw * (1.0 / len);
        unsafe {
            *out.get_unchecked_mut((py * width + px) as usize) = c;
        }
    }
}

/// Adds `.normalize()` — `OpExtInst <OpenCL.std> normalize` on a 3-vector.
#[claspr::device]
pub mod write_normalized_float3 {
    use spirv_std::cl::Float3;
    use spirv_std::glam::USizeVec3;

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [Float3],
        width: u32,
        _height: u32,
    ) {
        let px = _id.x as u32;
        let py = _id.y as u32;
        let u = px as f32 / width as f32 - 0.5;
        let v = py as f32 / width as f32 - 0.5;
        let c = Float3::new(u, v, 1.0).normalize();
        unsafe {
            *out.get_unchecked_mut((py * width + px) as usize) = c;
        }
    }
}

/// Just `sky(rd)` — one helper, lerp + smoothstep only, no loops.
/// If this crashes, the trigger is simpler than the bitwise case.
#[claspr::device]
pub mod just_sky {
    use spirv_std::cl::Float3;
    use spirv_std::glam::USizeVec3;

    pub fn sky(rd: Float3) -> Float3 {
        let h = spirv_std::arch::opencl_std::smoothstep(-0.05_f32, 0.45_f32, rd.y());
        Float3::new(0.85, 0.78, 0.62).lerp(Float3::new(0.30, 0.55, 0.85), h)
    }

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [Float3],
        width: u32,
        _height: u32,
    ) {
        let px = _id.x as u32;
        let py = _id.y as u32;
        let u = px as f32 / width as f32 - 0.5;
        let v = py as f32 / width as f32 - 0.5;
        let rd = Float3::new(u, v, 1.0).normalize();
        let c = sky(rd);
        unsafe {
            *out.get_unchecked_mut((py * width + px) as usize) = c;
        }
    }
}

/// Same as `just_sky` but with `sky`'s body inlined into the kernel.
/// If this passes and `just_sky` crashes → helper call alone is the
/// trigger.
#[claspr::device]
pub mod just_sky_inlined {
    use spirv_std::cl::Float3;
    use spirv_std::glam::USizeVec3;

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [Float3],
        width: u32,
        _height: u32,
    ) {
        let px = _id.x as u32;
        let py = _id.y as u32;
        let u = px as f32 / width as f32 - 0.5;
        let v = py as f32 / width as f32 - 0.5;
        let rd = Float3::new(u, v, 1.0).normalize();
        let h = spirv_std::arch::opencl_std::smoothstep(-0.05_f32, 0.45_f32, rd.y());
        let c = Float3::new(0.85, 0.78, 0.62).lerp(Float3::new(0.30, 0.55, 0.85), h);
        unsafe {
            *out.get_unchecked_mut((py * width + px) as usize) = c;
        }
    }
}

#[claspr::device]
pub mod gpu {
    pub mod scene;
    pub mod shading;

    use spirv_std::cl::Float3;
    use spirv_std::glam::USizeVec3;

    pub const CAM_RO: Float3 = Float3::new(3.0, 1.6, 4.0);
    pub const CAM_TARGET: Float3 = Float3::ZERO;
    pub const CAM_WORLD_UP: Float3 = Float3::Y;
    pub const FOV_SCALE: f32 = 0.7;

    use scene::{GROUND_BIAS, GROUND_Y, march, ray_at, scene_normal};
    use shading::{COLOR_BLOB, COLOR_GROUND, FOG_DENSITY, shade, sky, sun_dir};

    #[cfg(target_arch = "spirv")]
    use spirv_std::num_traits::Float;

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

    /// Writes `pixel_color` straight to a Float3 output slice (skip the
    /// image kernel for a smaller reproducer; if this crashes, the
    /// image part is exonerated).
    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [Float3],
        width: u32,
        height: u32,
    ) {
        let px = _id.x as u32;
        let py = _id.y as u32;
        if px >= width || py >= height {
            return;
        }
        let aspect = width as f32 / height as f32;
        let u = (2.0 * (px as f32 + 0.5) / width as f32 - 1.0) * aspect;
        let v = 1.0 - 2.0 * (py as f32 + 0.5) / height as f32;
        let c = pixel_color(u, v);
        unsafe {
            *out.get_unchecked_mut((py * width + px) as usize) = c;
        }
    }
}

fn try_step(name: &str, f: impl FnOnce() -> claspr::Result<()>) {
    eprint!("  {name:24} ... ");
    match f() {
        Ok(()) => eprintln!("OK"),
        Err(e) => eprintln!("ERROR: {e}"),
    }
}

fn main() -> claspr::Result<()> {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return Ok(());
        }
    };
    eprintln!(
        "Device: {} ({})",
        ctx.device().name()?,
        ctx.device().vendor()?
    );

    let n = (W * H) as usize;
    let mut out = vec![Float3::ZERO; n];

    macro_rules! variant {
        ($name:literal, $mod:ident) => {
            try_step($name, || {
                let k = $mod::kernels(&ctx)?;
                let buf = ctx.alloc::<Float3>(n)?;
                k.run(&ctx, [W as usize, H as usize], &buf, W, H)?;
                ctx.download(&buf, &mut out)?;
                Ok(())
            });
        };
    }

    variant!("write_const_float3", write_const_float3);
    variant!("write_per_pixel_float3", write_per_pixel_float3);
    variant!("write_normalized_inline", write_normalized_inline);
    variant!("write_normalized_float3", write_normalized_float3);
    variant!("just_sky_inlined", just_sky_inlined);
    variant!("just_sky", just_sky);
    variant!("full_pixel_color", gpu);

    Ok(())
}
