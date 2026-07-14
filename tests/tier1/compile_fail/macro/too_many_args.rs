//! `KernelArgs` (claspr/src/launch.rs) only has tuple impls up to
//! arity 16, so a kernel with 17+ runtime args (buffers + images +
//! scalars, the grid excluded) is unsupported. The `#[claspr::kernel]`
//! / `claspr::kernels!` expansion must surface a friendly,
//! span-attributed error at the kernel definition BEFORE the cryptic
//! downstream `KernelArgs is not implemented for (A, …, Q)` trait error
//! can fire.
//!
//! This kernel has 17 runtime args (one buffer + sixteen scalars) plus a
//! `#[spirv(global_invocation_id)]` builtin (dropped) — so the count is
//! 17, one over the ceiling of 16.

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
            i: u32,
            j: u32,
            k: u32,
            l: u32,
            m: u32,
            n: u32,
            o: u32,
            p: u32,
        );
    }
}

fn main() {}
