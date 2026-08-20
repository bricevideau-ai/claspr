//! Vulkan-style `#[spirv(uniform, binding = ...)]` attributes have no
//! meaning on the OpenCL kernel target. They must be rejected with a
//! pointed error, not classified as a "builtin" and silently dropped
//! from the host launch wrapper.

use claspr::kernels;

kernels! {
    pub mod gpu {
        fn vulkan_shaped(
            #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
            #[spirv(uniform, binding = 0)] data: &mut [u32],
        );
    }
}

fn main() {}
