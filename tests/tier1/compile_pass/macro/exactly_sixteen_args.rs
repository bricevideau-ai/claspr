//! Boundary check: 16 runtime args (one buffer + fifteen scalars) is the
//! MAXIMUM `KernelArgs` supports, so this must still compile cleanly.
//! Pairs with `compile_fail/macro/too_many_args.rs` (17 args → friendly
//! error). The `#[spirv(global_invocation_id)]` builtin is dropped and
//! the grid is a separate launcher arg, so neither counts toward the 16.

use claspr::kernels;

kernels! {
    pub mod gpu {
        fn exactly_sixteen_args(
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
        );
    }
}

fn main() {}
