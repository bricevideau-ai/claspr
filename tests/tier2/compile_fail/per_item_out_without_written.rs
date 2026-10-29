//! A kernel with write-only `PerItemOut` outputs must return `Written` — the
//! proof, built only from each output's `write`, that every output is written
//! on every path that completes. Omitting it is rejected at the kernel's
//! signature (claspr-build then enforces the proof's fields on the device side).

struct Kernels;

#[claspr::kernel(kernels = Kernels)]
pub fn forgets_written(
    #[spirv(cross_workgroup)] a: &[u32],
    #[spirv(cross_workgroup)] out: PerItemOut<u32>,
) {
}

fn main() {}
