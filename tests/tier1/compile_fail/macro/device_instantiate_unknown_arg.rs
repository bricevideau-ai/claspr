//! Unknown `#[claspr::device(...)]` arguments must be rejected with a
//! pointed error naming the one supported argument, not ignored.

#[claspr::device(frobnicate = 3)]
pub mod gpu {
    #[claspr::kernel]
    pub fn noop(#[spirv(cross_workgroup)] data: &mut [u32]) {
        data[0] = 1;
    }
}

fn main() {}
