//! Simple, portable kernels used by claspr's runtime integration tests.
//!
//! Shapes mirror the patterns from rust-gpu's `tests/compiletests/ui/lang/kernel/slices/`
//! (fill / copy / safe_read_write / dynamic_index) — anything outside
//! that set risks tripping spirv-opt / spirv-val edge cases we'd
//! otherwise be debugging instead of validating the runtime.
//!
//! ## Two device modules
//!
//! - [`mod@kernels`] — u32-only kernels (`fill_u32` / `add_u32` /
//!   `scale_u32` / `copy_u32`). Compiled without `Capability::Float64`
//!   so the emitted SPIR-V is consumable by every backend, including
//!   devices that don't support fp64 at all (e.g. rusticl/iris on
//!   Ice Lake, which SEGVs when handed a `Float64`-declaring program).
//! - [`mod@kernels_f64`] — `fill_f64` / `scale_f64`. Compiled *with*
//!   `Capability::Float64`. Runtime tests that load this module skip
//!   when the device doesn't advertise fp64.
//!
//! The split is the reason — keeping the fp64 kernels in their own
//! module means u32-only tests never load a program that mentions
//! `OpCapability Float64`, even transitively. See `build.rs` for how
//! the per-module capability set is selected.
//!
//! ## Design constraints (per `IMPLEMENTATION-PLAN.md` Phase 5)
//!
//! - **OpenCL 1.2 only**, max portability across pocl / rusticl / Intel.
//! - **`u32` and `f64` scalars / slices only** — no vector or struct
//!   args.
//! - **Read-then-write bodies only** (`data[i] = f(data[i], ...)`).
//!   Writes that don't first read the slice trip rust-gpu's pipeline
//!   on the same workspace where read-then-write builds clean. For
//!   the `f64` fills, callers must write a *finite* initial value
//!   first (alloc → write → fill) — otherwise the `data[i] * 0.0`
//!   trick that dodges spirv-opt produces NaN on uninit memory.
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

#[claspr::device]
pub mod kernels_f64 {
    /// Replace every element with `value` — the `f64` analogue of
    /// `fill_u32`. The `data[i] * 0.0 + value` shape preserves the
    /// read-then-write the codegen wants. NaN-safe only when callers
    /// have written a finite initial value first (see module docs).
    #[claspr::kernel]
    pub fn fill_f64(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [f64],
        value: f64,
    ) {
        let i = id.x;
        data[i] = data[i] * 0.0 + value;
    }

    /// Multiply every element by `factor` in place — the `f64`
    /// analogue of `scale_u32`. Naturally read-then-write.
    #[claspr::kernel]
    pub fn scale_f64(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [f64],
        factor: f64,
    ) {
        let i = id.x;
        data[i] = data[i] * factor;
    }
}
