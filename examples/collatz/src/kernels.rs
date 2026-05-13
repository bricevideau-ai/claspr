//! Single-source kernel module for the collatz example.
//!
//! What's in here serves *both* compilation paths:
//!
//! - **Host compilation**: each `#[claspr::kernel]` function is rewritten
//!   by the proc-macro into a host-side launch wrapper. Helper
//!   functions and consts are compiled normally.
//! - **Kernel compilation**: `examples/collatz/build.rs` calls
//!   `claspr_build::compile_from_host("src/kernels.rs")` which copies
//!   this whole file into a generated kernel sub-crate (translating
//!   `#[claspr::kernel]` → `#[spirv(kernel)]`) and runs `spirv-builder`
//!   on it.
//!
//! Notes about the constraints this dual life imposes:
//!
//! - We avoid `use glam::USizeVec3` at module scope because the host
//!   crate doesn't depend on glam — instead the parameter types use
//!   the absolute path `::glam::USizeVec3`, which only matters in the
//!   kernel-crate path (the host proc-macro drops builtin params
//!   before name resolution touches them).
//! - The helper `collatz` function below has no claspr attribute, so
//!   it compiles on both sides (pure-Rust body, valid for both host
//!   and SPIR-V targets).

/// Collatz sequence length for `n` (1-indexed input → number of steps
/// to reach 1), or `None` on overflow / zero input. Plain pure-Rust;
/// gets pulled into the kernel crate via the whole-file extraction in
/// `claspr_build::compile_from_host`.
pub fn collatz(mut n: u32) -> Option<u32> {
    let mut i = 0;
    if n == 0 {
        return None;
    }
    while n != 1 {
        n = if n.is_multiple_of(2) {
            n / 2
        } else {
            if n >= 0x5555_5555 {
                return None;
            }
            3 * n + 1
        };
        i += 1;
    }
    Some(i)
}

#[claspr::kernel(kernels = crate::compiled::Kernels)]
pub fn collatz_kernel(
    #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
) {
    let index = _id.x;
    data[index] = collatz(data[index]).unwrap_or(u32::MAX);
}
