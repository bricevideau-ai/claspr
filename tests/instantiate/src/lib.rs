//! End-to-end coverage for `#[claspr::device(instantiate(...))]`.
//!
//! One device module written against a placeholder scalar type `Real`,
//! stamped once per listed width. Each stamp is its own kernel
//! sub-crate, its own SPIR-V module, and its own host-side sub-module
//! (`gpu::f64`, `gpu::f32`) with a full typed-launcher surface — the
//! placeholder resolves through an injected `pub type Real = <ty>;` on
//! both sides, so no signature is ever textually substituted.
//!
//! What this locks in:
//!
//! - The build side stamps N kernel sub-crates from one module body
//!   and the stamps' SPIR-V genuinely differ (no `.spv` aliasing —
//!   regression-guarded separately in `tests/explicit-compile`).
//! - Capability hygiene per stamp: the f64 stamp declares `Float64`;
//!   the f32 stamp's SPIR-V must NOT mention it, so it stays loadable
//!   on devices without fp64 (the whole point of stamping instead of
//!   compiling both widths into one module — see NOTES.md).
//! - Both stamps dispatch correctly through their typed launchers.

#[claspr::device(instantiate(Real = [f64, f32]))]
pub mod gpu {
    /// `out[i] = a[i] * factor + offset`, all in the stamped width.
    #[claspr::kernel]
    pub fn axpb(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] a: &[Real],
        #[spirv(cross_workgroup)] out: &mut [Real],
        factor: Real,
        offset: Real,
    ) {
        let i = id.x;
        out[i] = a[i] * factor + offset;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claspr::{Context, DeviceSlice};
    use claspr_test_support::ctx;

    const N: usize = 64;

    /// Same gate as `tier1/tests/fp64.rs`: non-zero
    /// `CL_DEVICE_DOUBLE_FP_CONFIG` means some level of f64 support.
    fn ctx_with_f64() -> Option<Context> {
        let ctx = ctx()?;
        match ctx.device().cl3().double_fp_config() {
            Ok(mask) if mask != 0 => Some(ctx),
            _ => {
                eprintln!("SKIP: device has no Float64 capability");
                None
            }
        }
    }

    /// Walk a SPIR-V module's instruction stream and collect the
    /// operand of every `OpCapability` (opcode 17).
    fn declared_capabilities(spv: &[u8]) -> Vec<u32> {
        assert!(
            spv.len() >= 20 && spv.len().is_multiple_of(4),
            "malformed SPIR-V"
        );
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(words[0], 0x0723_0203, "bad SPIR-V magic");
        let mut caps = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let word_count = (words[i] >> 16) as usize;
            let opcode = words[i] & 0xffff;
            assert!(word_count > 0, "zero-length instruction at word {i}");
            if opcode == 17 {
                caps.push(words[i + 1]);
            }
            i += word_count;
        }
        caps
    }

    const FLOAT64_CAP: u32 = 10;

    #[test]
    fn stamps_embed_distinct_spirv() {
        assert_ne!(
            gpu::f64::SPV_BYTES,
            gpu::f32::SPV_BYTES,
            "the two stamps embedded identical SPIR-V — stamping is aliasing one build",
        );
        assert_eq!(gpu::f64::ENTRY_POINTS, gpu::f32::ENTRY_POINTS);
        assert_eq!(gpu::f64::ENTRY_POINTS, &["axpb"]);
    }

    /// The portability core of the feature: the f32 stamp must be
    /// loadable on devices with no fp64 at all, so its module must not
    /// declare `Float64`. The f64 stamp is the positive control (it
    /// must declare it), which also proves the walker works.
    #[test]
    fn f32_stamp_declares_no_float64() {
        let f64_caps = declared_capabilities(gpu::f64::SPV_BYTES);
        let f32_caps = declared_capabilities(gpu::f32::SPV_BYTES);
        assert!(
            f64_caps.contains(&FLOAT64_CAP),
            "f64 stamp should declare Float64; declared: {f64_caps:?}",
        );
        assert!(
            !f32_caps.contains(&FLOAT64_CAP),
            "f32 stamp must NOT declare Float64 (breaks non-fp64 devices); \
             declared: {f32_caps:?}",
        );
    }

    /// One driver body for every stamped width — the host-side mirror
    /// of the single kernel source above. Today the width is threaded
    /// through a macro because each stamp has its own concrete
    /// `Kernels` type; the planned generated `trait GpuKernels<Real>`
    /// (NOTES.md, instantiate design) turns this macro into a plain
    /// `fn run<R, K: GpuKernels<R>>` — the macro marks exactly the
    /// boilerplate that trait will erase.
    macro_rules! axpb_round_trip {
        ($name:ident, $real:ty, $stamp:path, $ctx:expr) => {
            #[test]
            fn $name() {
                let Some(ctx) = $ctx else { return };
                use $stamp as stamp;
                let kernels = stamp::kernels(&ctx).expect("load stamp");

                let input: Vec<$real> = (0..N).map(|i| i as $real).collect();
                let a = DeviceSlice::<$real>::alloc_zero(&ctx, N).expect("alloc a");
                let a = a.write(input.clone()).wait().expect("write a");
                let out = DeviceSlice::<$real>::alloc_zero(&ctx, N).expect("alloc out");

                let (_a, out) = kernels
                    .axpb([N], a, out, 2.0 as $real, 0.5 as $real)
                    .wait()
                    .expect("axpb");

                let mut got = vec![0.0 as $real; N];
                out.read(&mut got).wait().expect("readback");
                let want: Vec<$real> = input.iter().map(|&x| x * 2.0 + 0.5).collect();
                assert_eq!(got, want);
            }
        };
    }

    axpb_round_trip!(f32_stamp_runs, f32, crate::gpu::f32, ctx());
    axpb_round_trip!(f64_stamp_runs, f64, crate::gpu::f64, ctx_with_f64());
}
