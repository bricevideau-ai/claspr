//! A slice parameter without `#[spirv(cross_workgroup)]` is the single
//! most likely new-user mistake. It must error at the parameter with a
//! "did you forget" hint — historically it was classified as a *scalar*
//! and fell through to a confusing trait-bound error against generated
//! code at the launch site.

use claspr::kernels;

kernels! {
    pub mod gpu {
        fn forgot_qualifier(
            #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
            data: &mut [u32],
        );
    }
}

fn main() {}
