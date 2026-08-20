//! An unrecognised `Image!` sampled type (`type=q32` here) must be a
//! compile error. Historically it fell into the `Uint` catch-all, so a
//! typo'd sampled type launched with the wrong family and failed (or
//! misread) at runtime instead of at the parameter.

use claspr::kernels;

kernels! {
    pub mod gpu {
        fn bad_sampled_type(
            #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
            img: &mut Image!(2D, type=q32),
        );
    }
}

fn main() {}
