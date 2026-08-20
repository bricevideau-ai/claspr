//! A typo'd SPIR-V builtin qualifier must be a compile error at the
//! attribute, not a silently dropped parameter. Historically anything
//! unrecognised was treated as a builtin and deleted from the host
//! launch wrapper, so `global_invocation_idx` (note the typo) would
//! produce a launcher whose signature silently desynced from the
//! kernel as written.

use claspr::kernels;

kernels! {
    pub mod gpu {
        fn typod_builtin(
            #[spirv(global_invocation_idx)] _id: spirv_std::glam::USizeVec3,
            #[spirv(cross_workgroup)] data: &mut [u32],
        );
    }
}

fn main() {}
