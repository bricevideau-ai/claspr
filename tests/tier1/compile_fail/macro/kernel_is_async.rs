//! A `#[claspr::kernel]` fn is a device entry point and cannot be `async`. The
//! macro must reject it with a span-attributed error rather than emit host code
//! that fails opaquely.

use claspr::kernels;

kernels! {
    pub mod gpu {
        async fn is_async(
            #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
            #[spirv(cross_workgroup)] data: &mut [u32],
        );
    }
}

fn main() {}
