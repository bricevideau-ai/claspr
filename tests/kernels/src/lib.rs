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
//! - **`u32` and `f64` scalars / slices** — no vector args. One
//!   `#[repr(C)]` struct, [`kernels::Counter`], is passed BY REFERENCE
//!   (`&Counter` / `&mut Counter` over a `DeviceScalar<Counter>`) to pin
//!   user structs as scalar-ref args.
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

    /// Write `global_invocation_id().x` into the global slot — an
    /// identity probe used by the slot-generalization tests (and the
    /// natural probe for a global-offset launch option, if
    /// `LaunchSpec` ever grows one; it has no offset support today).
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

    /// `out[i] = a[i] + b[i]` with the output as a `PerItem`: each work-item
    /// owns one element, so the kernel needs no index arithmetic or global id
    /// (claspr adds one) and the launch checks the grid covers `out` exactly.
    #[claspr::kernel]
    pub fn add_per_item_u32(
        #[spirv(cross_workgroup)] a: &[u32],
        #[spirv(cross_workgroup)] b: &[u32],
        #[spirv(cross_workgroup)] out: PerItem<u32>,
    ) {
        let i = out.index();
        *out = a[i].wrapping_add(b[i]);
    }

    /// Multiply every element by `factor` in place, through a `PerItem` view
    /// (read and write the work-item's own element).
    #[claspr::kernel]
    pub fn scale_per_item_u32(
        #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: PerItem<u32>,
        factor: u32,
    ) {
        *data = data.wrapping_mul(factor);
    }

    /// `out[i] = a[i] + b[i]` with `PerItemIn` inputs: each read element-wise,
    /// so the launch's grid check bounds the inputs as well as the output.
    #[claspr::kernel]
    pub fn add_per_item_in_u32(
        #[spirv(cross_workgroup)] a: PerItemIn<u32>,
        #[spirv(cross_workgroup)] b: PerItemIn<u32>,
        #[spirv(cross_workgroup)] out: PerItem<u32>,
    ) {
        *out = a.wrapping_add(*b);
    }

    /// `out[i] = a[i] * factor` into a write-only `PerItemOut`: `out` may be
    /// an UNINITIALISED buffer, which the launch initialises. Returning
    /// `Written` (built only from `out.write(..)`) is the compile-time proof
    /// that every element is written; the kernel cannot read `out`.
    #[claspr::kernel]
    pub fn scaled_copy_out_u32(
        #[spirv(cross_workgroup)] a: &[u32],
        #[spirv(cross_workgroup)] out: PerItemOut<u32>,
        factor: u32,
    ) -> Written {
        let i = out.index();
        Written {
            out: out.write(a[i].wrapping_mul(factor)),
        }
    }

    /// Each work-item writes its own grid coordinates, packed as
    /// `z << 20 | y << 10 | x`, through a `PerItemOut` — pins the row-major
    /// work-item → element mapping on 2-D and 3-D grids.
    #[claspr::kernel]
    pub fn coords_u32(#[spirv(cross_workgroup)] out: PerItemOut<u32>) -> Written {
        let c = out.coord();
        Written {
            out: out.write(((c.z << 20) | (c.y << 10) | c.x) as u32),
        }
    }

    /// Two write-only outputs, matched to `Written`'s fields by name, with an
    /// early return (which must also prove both writes): `lo = a & 0xffff`,
    /// `hi = a >> 16`, or both 0 for a zero input.
    #[claspr::kernel]
    pub fn split_halves_u32(
        #[spirv(cross_workgroup)] a: PerItemIn<u32>,
        #[spirv(cross_workgroup)] lo: PerItemOut<u32>,
        #[spirv(cross_workgroup)] hi: PerItemOut<u32>,
    ) -> Written {
        if *a == 0 {
            return Written {
                lo: lo.write(0),
                hi: hi.write(0),
            };
        }
        Written {
            hi: hi.write(*a >> 16),
            lo: lo.write(*a & 0xffff),
        }
    }

    /// Transpose: on the `[n0, n1]` grid, work-item `(i, j)` writes element
    /// `j * n0 + i` of `out` (row-major `[n1, n0]`) from `src[[i, j]]` — a
    /// `Tensor<u32, 2>` read at any multi-index.
    #[claspr::kernel]
    pub fn transpose_u32(
        #[spirv(cross_workgroup)] src: Tensor<u32, 2>,
        #[spirv(cross_workgroup)] out: PerItemOut<u32>,
    ) -> Written {
        let c = out.coord();
        Written {
            out: out.write(src[[c.x, c.y]]),
        }
    }

    /// `out[i] = sum of src.sub(i)` — rank-1 rows of a (possibly strided) view.
    #[claspr::kernel]
    pub fn tensor_row_sums_u32(
        #[spirv(cross_workgroup)] src: Tensor<u32, 2>,
        #[spirv(cross_workgroup)] out: PerItemOut<u32>,
    ) -> Written {
        let row = src.sub(out.index());
        let mut sum = 0u32;
        let mut j = 0usize;
        while j < row.len() {
            sum += row[j];
            j += 1;
        }
        Written {
            out: out.write(sum),
        }
    }

    /// `out[i] = t[[i, 1, 2]]` via `sub(i).sub(1)[2]` — rank 3 down to rank 1.
    #[claspr::kernel]
    pub fn tensor3_pick_u32(
        #[spirv(cross_workgroup)] t: Tensor<u32, 3>,
        #[spirv(cross_workgroup)] out: PerItemOut<u32>,
    ) -> Written {
        let i = out.index();
        Written {
            out: out.write(t.sub(i).sub(1)[2]),
        }
    }

    /// `m[[i, i]] = d[i]` — a `TensorMut` written at work-item-distinct elements.
    #[claspr::kernel]
    pub fn set_diagonal_u32(
        #[spirv(cross_workgroup)] m: TensorMut<u32, 2>,
        #[spirv(cross_workgroup)] d: PerItemIn<u32>,
    ) {
        let i = d.index();
        m[[i, i]] = *d;
    }

    /// `t[[row, col]] = v` for the work-item at `(col, row)` — writes its own
    /// element of a `TensorMut` (a whole tensor, or one tile of a split).
    #[claspr::kernel]
    pub fn fill_view_u32(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] t: TensorMut<u32, 2>,
        v: u32,
    ) {
        t[[id.y, id.x]] = v;
    }

    /// `out[i] = sum of row i` — `PerRowIn`: the buffer is one equal row per
    /// work-item, and this work-item reads its own.
    #[claspr::kernel]
    pub fn row_sums_u32(
        #[spirv(cross_workgroup)] m: PerRowIn<u32>,
        #[spirv(cross_workgroup)] out: PerItemOut<u32>,
    ) -> Written {
        let mut sum = 0u32;
        let mut j = 0usize;
        while j < m.len() {
            sum += m[j];
            j += 1;
        }
        Written {
            out: out.write(sum),
        }
    }

    /// Subtract each row's minimum from it, in place — `PerRow`: this
    /// work-item's row, exclusively.
    #[claspr::kernel]
    pub fn subtract_row_min_u32(#[spirv(cross_workgroup)] m: PerRow<u32>) {
        let mut min = m[0];
        let mut j = 1usize;
        while j < m.len() {
            if m[j] < min {
                min = m[j];
            }
            j += 1;
        }
        j = 0;
        while j < m.len() {
            m[j] -= min;
            j += 1;
        }
    }

    /// Sum of the 4 neighbours on a periodic `[W, H]` grid (`Periodic`: the
    /// buffer is the grid; offsets past an edge wrap).
    #[claspr::kernel]
    pub fn neighbours_periodic_u32(
        #[spirv(cross_workgroup)] f: Periodic<u32>,
        #[spirv(cross_workgroup)] out: PerItemOut<u32>,
    ) -> Written {
        let s = f.at([-1, 0]) + f.at([1, 0]) + f.at([0, -1]) + f.at([0, 1]);
        Written { out: out.write(s) }
    }

    /// The same with clamped edges (`Clamped`: past an edge reads the edge).
    #[claspr::kernel]
    pub fn neighbours_clamped_u32(
        #[spirv(cross_workgroup)] f: Clamped<u32>,
        #[spirv(cross_workgroup)] out: PerItemOut<u32>,
    ) -> Written {
        let s = f.at([-1, 0]) + f.at([1, 0]) + f.at([0, -1]) + f.at([0, 1]);
        Written { out: out.write(s) }
    }

    /// The same over a buffer padded with one halo cell per side (`Halo<u32, 1>`:
    /// the buffer is `[W + 2, H + 2]`, the grid is the `[W, H]` interior), plus
    /// 1000 x the centre.
    #[claspr::kernel]
    pub fn neighbours_halo_u32(
        #[spirv(cross_workgroup)] f: Halo<u32, 1>,
        #[spirv(cross_workgroup)] out: PerItemOut<u32>,
    ) -> Written {
        let s = f.at([-1, 0]) + f.at([1, 0]) + f.at([0, -1]) + f.at([0, 1]) + 1000 * *f;
        Written { out: out.write(s) }
    }

    /// 3-D halo, spelled out as `Stencil<_, Padded<..>>`: the z neighbours
    /// (`Padded<0, 0, 1>`: padded along z only).
    #[claspr::kernel]
    pub fn z_neighbours_u32(
        #[spirv(cross_workgroup)] f: Stencil<u32, Padded<0, 0, 1>>,
        #[spirv(cross_workgroup)] out: PerItemOut<u32>,
    ) -> Written {
        Written {
            out: out.write(f.at([0, 0, -1]) + f.at([0, 0, 1])),
        }
    }

    /// `dst = src + 1` on the interior of two halo-padded buffers (`Interior`
    /// writes this work-item's cell; the halo is left as it is).
    #[claspr::kernel]
    pub fn interior_add_one_u32(
        #[spirv(cross_workgroup)] src: Halo<u32, 1>,
        #[spirv(cross_workgroup)] dst: Interior<u32, 1>,
    ) {
        *dst = *src + 1;
    }

    /// A user struct passed BY REFERENCE to kernels, held on the host side in
    /// a `DeviceScalar<Counter>` — e.g. per-step state a graph advances itself.
    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    pub struct Counter {
        pub value: u32,
        pub step: u32,
    }

    /// Add the counter's current `value` to every element in place (a struct
    /// READ through `&Counter`).
    #[claspr::kernel]
    pub fn add_counter_u32(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        #[spirv(cross_workgroup)] counter: &Counter,
    ) {
        let i = id.x;
        data[i] = data[i].wrapping_add(counter.value);
    }

    /// Advance the counter by its `step` (a struct WRITE through
    /// `&mut Counter`; launch with one work-item).
    #[claspr::kernel]
    pub fn advance_counter(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] counter: &mut Counter,
    ) {
        if id.x == 0 {
            counter.value = counter.value.wrapping_add(counter.step);
        }
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
