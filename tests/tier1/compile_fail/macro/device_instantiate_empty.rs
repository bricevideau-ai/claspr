//! `instantiate` with an empty type list is meaningless — the module
//! would produce zero stamps. Reject at expansion, not at build time.

#[claspr::device(instantiate(Real = []))]
pub mod gpu {
    #[claspr::kernel]
    pub fn noop(#[spirv(cross_workgroup)] data: &mut [Real]) {
        data[0] = data[0];
    }
}

fn main() {}
