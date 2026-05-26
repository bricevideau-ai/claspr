//! Simple, portable kernels used by claspr's runtime integration tests.
//!
//! Shapes mirror the patterns from rust-gpu's `tests/compiletests/ui/lang/kernel/slices/`
//! (fill / copy / safe_read_write / dynamic_index) — anything outside
//! that set risks tripping spirv-opt / spirv-val edge cases we'd
//! otherwise be debugging instead of validating the runtime.
//!
//! Design constraints (per [`IMPLEMENTATION-PLAN.md`] Phase 5):
//!
//! - **OpenCL 1.2 only**, max portability across pocl / rusticl / Intel.
//! - **`u32` slices + scalars only** — no vector or struct args.
//! - **Read-then-write bodies only** (`data[i] = f(data[i], ...)`).
//!   Writes that don't first read the slice trip rust-gpu's pipeline
//!   on the same workspace where read-then-write builds clean.
//! - **One operation per kernel** — runtime tests compose them.

#[claspr::device]
pub mod kernels {
    /// Replace every element with `value` (encoded as `data[i] * 0 + value`
    /// so the codegen sees a read-then-write — see module docs for why).
    #[claspr::kernel]
    pub fn fill_u32(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        value: u32,
    ) {
        let i = id.x;
        data[i] = data[i].wrapping_mul(0).wrapping_add(value);
    }

    /// Element-wise `out[i] = a[i] + b[i]`.
    #[claspr::kernel]
    pub fn add_u32(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] a: &[u32],
        #[spirv(cross_workgroup)] b: &[u32],
        #[spirv(cross_workgroup)] out: &mut [u32],
    ) {
        let i = id.x;
        out[i] = a[i].wrapping_add(b[i]);
    }

    /// Multiply every element by `factor` in place.
    #[claspr::kernel]
    pub fn scale_u32(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        factor: u32,
    ) {
        let i = id.x;
        data[i] = data[i].wrapping_mul(factor);
    }

    /// `dst[i] = src[i]`. The `wrapping_add(0)` keeps spirv-opt from
    /// dropping the kernel — the simpler `dst[i] = src[i]` survives
    /// rust-gpu codegen but vanishes during opt, leaving the kernel
    /// out of the entry-point list.
    #[claspr::kernel]
    pub fn copy_u32(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] src: &[u32],
        #[spirv(cross_workgroup)] dst: &mut [u32],
    ) {
        let i = id.x;
        dst[i] = src[i].wrapping_add(0);
    }
}
