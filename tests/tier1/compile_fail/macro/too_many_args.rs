//! `KernelArgs` (claspr/src/launch.rs) only has tuple impls up to
//! arity 8, so a kernel with 9+ runtime args (buffers + images +
//! scalars, the grid excluded) is unsupported. The `#[claspr::kernel]`
//! / `claspr::kernels!` expansion must surface a friendly,
//! span-attributed error at the kernel definition BEFORE the cryptic
//! downstream `KernelArgs is not implemented for (A, …, I)` trait error
//! can fire.
//!
//! This kernel has 9 runtime args (one buffer + eight scalars) plus a
//! `#[spirv(global_invocation_id)]` builtin (dropped) — so the count is
//! 9, one over the ceiling of 8.

use claspr::kernels;

kernels! {
    pub mod gpu {
        fn too_many_args(
            #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
            #[spirv(cross_workgroup)] data: &mut [u32],
            a: u32,
            b: u32,
            c: u32,
            d: u32,
            e: u32,
            f: u32,
            g: u32,
            h: u32,
        );
    }
}

fn main() {}
