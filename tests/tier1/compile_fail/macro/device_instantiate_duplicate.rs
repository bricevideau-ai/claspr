//! Duplicate instantiate types would stamp the same module twice under
//! the same name — reject with the offending type named.

#[claspr::device(instantiate(Real = [f32, f32]))]
pub mod gpu {
    #[claspr::kernel]
    pub fn noop(#[spirv(cross_workgroup)] data: &mut [Real]) {
        data[0] = data[0];
    }
}

fn main() {}
