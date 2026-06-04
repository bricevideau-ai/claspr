//! Trivial rust-gpu kernel crate exercising the explicit
//! `claspr_build::compile(...)` path.
//!
//! Shape mirrors `tests/kernels`'s `fill_u32`. No fp64 / no images
//! / no groups so the OpenCL 1.2 target accepts it on every ICD.

#![no_std]

use spirv_std::glam::USizeVec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn fill_u32(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
    value: u32,
) {
    let i = id.x;
    data[i] = value;
}
