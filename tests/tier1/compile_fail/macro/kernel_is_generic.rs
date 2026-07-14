//! A `#[claspr::kernel]` fn cannot be generic — the host wrapper synthesizes its
//! own generics from the buffer/image params. A user-written type parameter must
//! be rejected at the macro boundary with a clear diagnostic.

use claspr::kernels;

kernels! {
    pub mod gpu {
        fn is_generic<X>(
            #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
            #[spirv(cross_workgroup)] data: &mut [u32],
        );
    }
}

fn main() {}
