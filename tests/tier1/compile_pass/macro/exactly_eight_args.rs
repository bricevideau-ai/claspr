//! Boundary check: 8 runtime args (one buffer + seven scalars) is the
//! MAXIMUM `KernelArgs` supports, so this must still compile cleanly.
//! Pairs with `compile_fail/macro/too_many_args.rs` (9 args → friendly
//! error). The `#[spirv(global_invocation_id)]` builtin is dropped and
//! the grid is a separate launcher arg, so neither counts toward the 8.

use claspr::kernels;

kernels! {
    pub mod gpu {
        fn exactly_eight_args(
            #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
            #[spirv(cross_workgroup)] data: &mut [u32],
            a: u32,
            b: u32,
            c: u32,
            d: u32,
            e: u32,
            f: u32,
            g: u32,
        );
    }
}

fn main() {}
