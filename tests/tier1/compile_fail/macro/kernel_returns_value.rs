//! A `#[claspr::kernel]` fn writes through its buffer/image args and must return
//! `()`. A non-`()` return type is unsupported; the macro must reject it with a
//! span-attributed error AT THE KERNEL DEFINITION, not let it fall through to a
//! confusing downstream error in generated code.

use claspr::kernels;

kernels! {
    pub mod gpu {
        fn returns_value(
            #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
            #[spirv(cross_workgroup)] data: &mut [u32],
        ) -> u32;
    }
}

fn main() {}
