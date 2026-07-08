//! Simple, portable kernels used by claspr's runtime integration tests.
//!
//! Shapes mirror the patterns from rust-gpu's
//! `tests/compiletests/ui/lang/kernel/slices/` (fill / copy /
//! safe_read_write / dynamic_index) — well-trodden ground.
//!
//! ## Two device modules
//!
//! - [`mod@kernels`] — u32-only kernels (`fill_u32` / `add_u32` /
//!   `scale_u32` / `copy_u32` / `local_id_u32` / `global_id_u32`).
//!   Compiled without `Capability::Float64` so the emitted SPIR-V is
//!   consumable by every backend, including devices that don't
//!   support fp64 at all (e.g. rusticl/iris on Ice Lake, which SEGVs
//!   when handed a `Float64`-declaring program).
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
//! - **One operation per kernel** — runtime tests compose them.

#[claspr::device]
pub mod kernels {
    /// Replace every element with `value`.
    #[claspr::kernel]
    pub fn fill_u32(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        value: u32,
    ) {
        let i = id.x;
        data[i] = value;
    }

    /// Element-wise `out[i] = a[i] + b[i]`.
    #[claspr::kernel]
    pub fn add_u32(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
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
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        factor: u32,
    ) {
        let i = id.x;
        data[i] = data[i].wrapping_mul(factor);
    }

    /// `dst[i] = src[i]`.
    #[claspr::kernel]
    pub fn copy_u32(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] src: &[u32],
        #[spirv(cross_workgroup)] dst: &mut [u32],
    ) {
        let i = id.x;
        dst[i] = src[i];
    }

    /// Write `local_invocation_id().x` into the global slot — used by
    /// runtime tests to verify the local work-size took effect. With
    /// `local=[L]`, the output should be `[0, 1, …, L-1]` repeating
    /// once per workgroup.
    #[claspr::kernel]
    pub fn local_id_u32(
        #[spirv(global_invocation_id)] gid: spirv_std::glam::USizeVec3,
        #[spirv(local_invocation_id)] lid: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
    ) {
        data[gid.x] = lid.x as u32;
    }

    /// Write `global_invocation_id().x` into the global slot — used by
    /// runtime tests to verify `global_offset` took effect. With
    /// `global_size=[N]` and `global_offset=[K]`,
    /// `global_invocation_id().x` ranges over `K..K+N` and writes land
    /// at the same indices.
    #[claspr::kernel]
    pub fn global_id_u32(
        #[spirv(global_invocation_id)] gid: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
    ) {
        data[gid.x] = gid.x as u32;
    }

    /// Multiply every element by a scalar passed **by reference** —
    /// exercises `#[spirv(cross_workgroup)] &u32` (a read scalar-ref,
    /// lowered to a bare pointer-to-scalar with no length operand).
    #[claspr::kernel]
    pub fn scale_by_ref_u32(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        #[spirv(cross_workgroup)] factor: &u32,
    ) {
        let i = id.x;
        data[i] = data[i].wrapping_mul(*factor);
    }

    /// Write a by-value scalar through a `&mut u32` output scalar-ref —
    /// exercises `#[spirv(cross_workgroup)] &mut u32` threading to
    /// `Output` (host-readable after the launch). Single-element grid.
    #[claspr::kernel]
    pub fn write_scalar_u32(
        #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] out: &mut u32,
        val: u32,
    ) {
        *out = val;
    }

    /// `f32` twin of [`scale_by_ref_u32`] — a `&f32` READ scalar-ref
    /// scales every element. Proves the scalar-ref path is generic over
    /// the element type (not f32-special-cased, not u32-special-cased).
    #[claspr::kernel]
    pub fn scale_by_ref_f32(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [f32],
        #[spirv(cross_workgroup)] factor: &f32,
    ) {
        let i = id.x;
        data[i] = data[i] * *factor;
    }

    /// `f32` twin of [`write_scalar_u32`] — a `&mut f32` OUTPUT
    /// scalar-ref written by value.
    #[claspr::kernel]
    pub fn write_scalar_f32(
        #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] out: &mut f32,
        val: f32,
    ) {
        *out = val;
    }

    /// Add a device-resident `&u32` scalar to every element in place —
    /// used by the host-write-then-kernel-read seam test: a host seam
    /// WRITES `addend` (a `&mut u32` DeviceScalar) mid-graph, then this
    /// kernel READS it as `&u32` in the SAME graph.
    #[claspr::kernel]
    pub fn add_ref_u32(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        #[spirv(cross_workgroup)] addend: &u32,
    ) {
        let i = id.x;
        data[i] = data[i].wrapping_add(*addend);
    }
}

#[claspr::device]
pub mod kernels_f64 {
    /// Replace every element with `value` — the `f64` analogue of
    /// `fill_u32`.
    #[claspr::kernel]
    pub fn fill_f64(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [f64],
        value: f64,
    ) {
        let i = id.x;
        data[i] = value;
    }

    /// Multiply every element by `factor` in place — the `f64`
    /// analogue of `scale_u32`.
    #[claspr::kernel]
    pub fn scale_f64(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [f64],
        factor: f64,
    ) {
        let i = id.x;
        data[i] = data[i] * factor;
    }
}
