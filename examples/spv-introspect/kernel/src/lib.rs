//! Two-kernel demo SPIR-V for the runtime-introspection example.
//!
//! - `fill_u32(data, value)` — single mutable slice + scalar.
//! - `add_u32(a, b, out)` — two read-only slices + one mutable slice.
//!
//! Distinct arg shapes let the introspection walker show off
//! different `cl_kernel_arg_address_qualifier` values
//! (CL_KERNEL_ARG_ADDRESS_GLOBAL for slices, _PRIVATE for scalars)
//! and varying mutability on the same kernel.

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

#[spirv(kernel)]
pub fn add_u32(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] a: &[u32],
    #[spirv(cross_workgroup)] b: &[u32],
    #[spirv(cross_workgroup)] out: &mut [u32],
) {
    let i = id.x;
    out[i] = a[i].wrapping_add(b[i]);
}
