//! Trivial rust-gpu kernel crate exercising the explicit
//! `claspr_build::compile(...)` path.
//!
//! Shape mirrors `tests/kernels`'s `fill_u32`. No fp64 / no images
//! / no groups so the OpenCL 1.2 target accepts it on every ICD.

#![no_std]

use spirv_std::glam::USizeVec3;
use spirv_std::spirv;

/// Under the `alt` feature the fill is biased so the two compiled
/// variants of this crate produce observably different kernels — the
/// parent crate's build script compiles both and its tests assert the
/// embedded SPIR-V blobs (and runtime behavior) stay distinct.
#[cfg(feature = "alt")]
const BIAS: u32 = 1000;
#[cfg(not(feature = "alt"))]
const BIAS: u32 = 0;

#[spirv(kernel)]
pub fn fill_u32(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
    value: u32,
) {
    let i = id.x;
    data[i] = value + BIAS;
}
