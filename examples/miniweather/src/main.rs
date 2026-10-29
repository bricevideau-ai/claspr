//! miniWeather — single-source Rust + OpenCL, in one file, at two precisions.
//!
//! Same numerics as `miniWeather_serial.F90` (Matt Norman's miniWeather):
//! finite-volume, 3-stage RK, alternating Strang splitting. Init and the
//! mass/energy reductions run on the host in f64; the timestep runs on the
//! OpenCL device at whichever width the device supports (or `--precision`).
//!
//! This is the flagship `instantiate` showcase: the device module below is
//! written ONCE against the placeholder `Real` and stamped per width by
//! `#[claspr::device(instantiate(Real = [f64, f32]))]` — no per-width kernel
//! copies, no separate kernel crate, no hand-stamped signature lists. The
//! HOST driver is width-generic too: ONE `fn run<Real, K: gpu::Kernels<Real>>`
//! drives both stamps through the generated `gpu::Kernels<Real>` trait (the
//! only per-width text left in this file is the two-line dispatch in `main`
//! and the `Width` cast helper).
//!
//! ## Two graph shapes, proven equivalent
//!
//! The timestep is a Tier-2 eager graph built once and replayed with `sync`.
//! It is built in TWO shapes, and `main` runs both and compares:
//!
//! - **`Shape::Unrolled`** — two timesteps per graph (48 dispatches). Fusing
//!   the pair makes the x-first/z-first alternation and the launch extents
//!   LITERALS, so a steady-state replay mutates nothing at all. One binding
//!   state, one recorded command buffer.
//! - **`Shape::PerTimestep`** — one timestep per graph (24 dispatches),
//!   mirroring the reference's `perform_timestep` one-to-one. Direction,
//!   extents and hyperviscosity ride SLOTS and flip every step — the
//!   reference's `direction_switch`, expressed as a rebind rather than as
//!   unrolled code. Two binding states, both held by the keyed command-buffer
//!   cache, so it still records twice rather than once per sync.
//!
//! The unrolled shape exists only because direction used to have to be a
//! literal: before the keyed cache, alternating a slot re-recorded the whole
//! command buffer every step. It no longer does, so the natural shape is now
//! affordable — it costs a few percent of wall time for one extra `sync` per
//! pair of steps, and reads like the Fortran.
//!
//! All nine buffers are threaded through every dispatch as pipes (they rehome
//! automatically each replay); the read-only hydrostatic arrays are `Arc`-shared
//! rather than threaded, so they cost no pass-through parameters.
//!
//! Self-validating: `cargo test -p miniweather-example` runs short thermal
//! integrations at both widths and checks the domain-integrated mass/energy
//! drift, f32-vs-f64 agreement, and that the two graph shapes above are
//! BIT-IDENTICAL. Full runs:
//! `cargo run -p miniweather-example -- --nx 400 --nz 200 --sim-time 400`.

use claspr::eager::{DeviceOp, DeviceOpExt, bundle4, lift};
use claspr::{Context, DeviceSlice, LaunchSpec, slot, slots};

// ── Physical + scheme constants (host, f64) ─────────────────────────────────
use std::f64::consts::PI;
const GRAV: f64 = 9.8;
const CP: f64 = 1004.0;
const CV: f64 = 717.0;
const RD: f64 = 287.0;
const P0: f64 = 1.0e5;
const C0: f64 = 27.5629410929725921310572974482;
const GAMMA: f64 = 1.40027894002789400278940027894;
const XLEN: f64 = 2.0e4;
const ZLEN: f64 = 1.0e4;
const HV_BETA: f64 = 0.05;
const CFL: f64 = 1.50;
const MAX_SPEED: f64 = 450.0;
const HS: usize = 2;
// Digits mirror miniWeather_serial.F90 verbatim.
#[allow(clippy::excessive_precision)]
const QPOINTS: [f64; 3] = [
    0.112701665379258311482073460022,
    0.500000000000000000000000000000,
    0.887298334620741688517926539980,
];
// Digits mirror miniWeather_serial.F90 verbatim.
#[allow(clippy::excessive_precision)]
const QWEIGHTS: [f64; 3] = [
    0.277777777777777777777777777779,
    0.444444444444444444444444444444,
    0.277777777777777777777777777779,
];

// ═════════════════════════════════════════════════════════════════════════════
// Device side: ONE precision-generic module, stamped per width by
// `instantiate` — `gpu::f64` and `gpu::f32` each get their own kernel
// sub-crate, SPIR-V module, and typed host surface, with
// `pub type Real = <ty>;` injected on both sides. The f64 stamp declares
// `Float64` automatically; the f32 stamp builds without fp64 permission,
// so its module loads on devices with no double support at all.
//
// Every kernel takes the same nine-buffer "caravan" in the same order
// (state, state_tmp, flux, tend, hy_dens_cell, hy_dens_theta_cell,
// hy_dens_int, hy_dens_theta_int, hy_pressure_int) so the Tier-2 chain
// threads them uniformly; unused ones are just passed through. `dir`
// selects x (0) or z (1); the `_s`/`_t` suffix picks which of
// state/state_tmp is the forcing. All float literals are unsuffixed so
// they infer to `Real`.
// ═════════════════════════════════════════════════════════════════════════════

#[claspr::device(instantiate(Real = [f64, f32]))]
pub mod gpu {
    use spirv_std::arch::opencl_std as ocl;
    use spirv_std::glam::USizeVec3;
    const HS: usize = 2;
    // Digits mirror miniWeather_serial.F90 verbatim.
    #[allow(clippy::excessive_precision)]
    const C0: Real = 27.5629410929725921310572974482;
    #[allow(clippy::excessive_precision)]
    const GAMMA: Real = 1.40027894002789400278940027894;
    const GRAV: Real = 9.8;
    const ZLEN: Real = 1.0e4;

    #[allow(clippy::too_many_arguments)]
    fn halo_body(
        f: &mut [Real],
        hy_d: &[Real],
        hy_dt: &[Real],
        gid: usize,
        nx: usize,
        nz: usize,
        dir: u32,
        inj: u32,
        dz: Real,
    ) {
        let nxp = nx + 2 * HS;
        let nzp = nz + 2 * HS;
        let plane = nzp * nxp;
        if dir == 0 {
            // periodic x halos, one work-item per interior row
            if gid >= nz {
                return;
            }
            let kk = gid + HS;
            let mut ll = 0usize;
            while ll < 4 {
                let b = ll * plane + kk * nxp;
                f[b] = f[b + nx];
                f[b + 1] = f[b + nx + 1];
                f[b + nx + 2] = f[b + 2];
                f[b + nx + 3] = f[b + 3];
                ll += 1;
            }
            if inj != 0 {
                let z = (gid as Real + 0.5) * dz;
                let d = z - 3.0 * ZLEN / 4.0;
                let ad = if d < 0.0 { -d } else { d };
                if ad <= ZLEN / 16.0 {
                    let hd = hy_d[kk];
                    let hdt = hy_dt[kk];
                    let mut i = 0usize;
                    while i < HS {
                        let dens = f[kk * nxp + i];
                        f[plane + kk * nxp + i] = (dens + hd) * 50.0;
                        f[3 * plane + kk * nxp + i] = (dens + hd) * 298.0 - hdt;
                        i += 1;
                    }
                }
            }
        } else {
            // z halos, one work-item per column (including x halos)
            if gid >= nxp {
                return;
            }
            let i = gid;
            let rows = [0usize, 1, nz + HS, nz + HS + 1];
            let srcs = [HS, HS, nz + HS - 1, nz + HS - 1];
            let mut j = 0usize;
            while j < 4 {
                let row = rows[j];
                let src = srcs[j];
                f[2 * plane + row * nxp + i] = 0.0; // WMOM
                f[plane + row * nxp + i] = f[plane + src * nxp + i] / hy_d[src] * hy_d[row]; // UMOM
                f[row * nxp + i] = f[src * nxp + i]; // DENS
                f[3 * plane + row * nxp + i] = f[3 * plane + src * nxp + i]; // RHOT
                j += 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn flux_body(
        f: &[Real],
        flux: &mut [Real],
        hy_d: &[Real],
        hy_dt: &[Real],
        hy_di: &[Real],
        hy_dti: &[Real],
        hy_pi: &[Real],
        gid: usize,
        nx: usize,
        nz: usize,
        dir: u32,
        hv: Real,
    ) {
        let nxp = nx + 2 * HS;
        let nzp = nz + 2 * HS;
        let plane = nzp * nxp;
        let fplane = (nz + 1) * (nx + 1);
        let mut vals = [0.0; 4];
        let mut d3 = [0.0; 4];
        if dir == 0 {
            if gid >= (nx + 1) * nz {
                return;
            }
            let j = gid % (nx + 1);
            let k = gid / (nx + 1);
            let kk = k + HS;
            let mut ll = 0usize;
            while ll < 4 {
                let b = ll * plane + kk * nxp + j;
                let s1 = f[b];
                let s2 = f[b + 1];
                let s3 = f[b + 2];
                let s4 = f[b + 3];
                vals[ll] = -s1 / 12.0 + 7.0 * s2 / 12.0 + 7.0 * s3 / 12.0 - s4 / 12.0;
                d3[ll] = -s1 + 3.0 * s2 - 3.0 * s3 + s4;
                ll += 1;
            }
            let r = vals[0] + hy_d[kk];
            let u = vals[1] / r;
            let w = vals[2] / r;
            let t = (vals[3] + hy_dt[kk]) / r;
            let p = C0 * ocl::pow(r * t, GAMMA);
            let fb = k * (nx + 1) + j;
            flux[fb] = r * u - hv * d3[0];
            flux[fplane + fb] = r * u * u + p - hv * d3[1];
            flux[2 * fplane + fb] = r * u * w - hv * d3[2];
            flux[3 * fplane + fb] = r * u * t - hv * d3[3];
        } else {
            if gid >= nx * (nz + 1) {
                return;
            }
            let i = gid % nx;
            let k = gid / nx;
            let ii = i + HS;
            let mut ll = 0usize;
            while ll < 4 {
                let b = ll * plane + k * nxp + ii;
                let s1 = f[b];
                let s2 = f[b + nxp];
                let s3 = f[b + 2 * nxp];
                let s4 = f[b + 3 * nxp];
                vals[ll] = -s1 / 12.0 + 7.0 * s2 / 12.0 + 7.0 * s3 / 12.0 - s4 / 12.0;
                d3[ll] = -s1 + 3.0 * s2 - 3.0 * s3 + s4;
                ll += 1;
            }
            let r = vals[0] + hy_di[k];
            let u = vals[1] / r;
            let mut w = vals[2] / r;
            let t = (vals[3] + hy_dti[k]) / r;
            let p = C0 * ocl::pow(r * t, GAMMA) - hy_pi[k];
            let mut d30 = d3[0];
            if k == 0 || k == nz {
                w = 0.0;
                d30 = 0.0;
            }
            let fb = k * (nx + 1) + i;
            flux[fb] = r * w - hv * d30;
            flux[fplane + fb] = r * w * u - hv * d3[1];
            flux[2 * fplane + fb] = r * w * w + p - hv * d3[2];
            flux[3 * fplane + fb] = r * w * t - hv * d3[3];
        }
    }

    /// One `tend` element — variable `ll` at interior cell `(i, k)`: the flux
    /// divergence along `dir`, plus gravity on WMOM for z.
    #[allow(clippy::too_many_arguments)]
    fn tend_at(
        flux: &[Real],
        f: &[Real],
        i: usize,
        k: usize,
        ll: usize,
        nx: usize,
        nz: usize,
        dir: u32,
        dx: Real,
        dz: Real,
    ) -> Real {
        let fb = ll * (nz + 1) * (nx + 1) + k * (nx + 1) + i;
        if dir == 0 {
            -(flux[fb + 1] - flux[fb]) / dx
        } else {
            let mut v = -(flux[fb + (nx + 1)] - flux[fb]) / dz;
            if ll == 2 {
                v -= f[(k + HS) * (nx + 2 * HS) + (i + HS)] * GRAV;
            }
            v
        }
    }

    /// Index into a halo-padded state array of the interior cell a
    /// `[nx, nz, 4]` work-item `c` owns (`c.z` is the variable plane).
    fn interior(c: USizeVec3, nx: usize, nz: usize) -> usize {
        let nxp = nx + 2 * HS;
        c.z * (nz + 2 * HS) * nxp + (c.y + HS) * nxp + (c.x + HS)
    }

    // ── entry points: caravan order is fixed across all kernels ─────────────

    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn halo_s(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] state: &mut [Real],
        #[spirv(cross_workgroup)] _state_tmp: &[Real],
        #[spirv(cross_workgroup)] _flux: &[Real],
        #[spirv(cross_workgroup)] _tend: &[Real],
        #[spirv(cross_workgroup)] hy_d: &[Real],
        #[spirv(cross_workgroup)] hy_dt: &[Real],
        nx: u32,
        nz: u32,
        dir: u32,
        inj: u32,
        dz: Real,
    ) {
        halo_body(
            state,
            hy_d,
            hy_dt,
            id.x,
            nx as usize,
            nz as usize,
            dir,
            inj,
            dz,
        );
    }

    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn halo_t(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] _state: &[Real],
        #[spirv(cross_workgroup)] state_tmp: &mut [Real],
        #[spirv(cross_workgroup)] _flux: &[Real],
        #[spirv(cross_workgroup)] _tend: &[Real],
        #[spirv(cross_workgroup)] hy_d: &[Real],
        #[spirv(cross_workgroup)] hy_dt: &[Real],
        nx: u32,
        nz: u32,
        dir: u32,
        inj: u32,
        dz: Real,
    ) {
        halo_body(
            state_tmp,
            hy_d,
            hy_dt,
            id.x,
            nx as usize,
            nz as usize,
            dir,
            inj,
            dz,
        );
    }

    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn flux_s(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] state: &[Real],
        #[spirv(cross_workgroup)] _state_tmp: &[Real],
        #[spirv(cross_workgroup)] flux: &mut [Real],
        #[spirv(cross_workgroup)] _tend: &[Real],
        #[spirv(cross_workgroup)] hy_d: &[Real],
        #[spirv(cross_workgroup)] hy_dt: &[Real],
        #[spirv(cross_workgroup)] hy_di: &[Real],
        #[spirv(cross_workgroup)] hy_dti: &[Real],
        #[spirv(cross_workgroup)] hy_pi: &[Real],
        nx: u32,
        nz: u32,
        dir: u32,
        hv: Real,
    ) {
        flux_body(
            state,
            flux,
            hy_d,
            hy_dt,
            hy_di,
            hy_dti,
            hy_pi,
            id.x,
            nx as usize,
            nz as usize,
            dir,
            hv,
        );
    }

    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn flux_t(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] _state: &[Real],
        #[spirv(cross_workgroup)] state_tmp: &[Real],
        #[spirv(cross_workgroup)] flux: &mut [Real],
        #[spirv(cross_workgroup)] _tend: &[Real],
        #[spirv(cross_workgroup)] hy_d: &[Real],
        #[spirv(cross_workgroup)] hy_dt: &[Real],
        #[spirv(cross_workgroup)] hy_di: &[Real],
        #[spirv(cross_workgroup)] hy_dti: &[Real],
        #[spirv(cross_workgroup)] hy_pi: &[Real],
        nx: u32,
        nz: u32,
        dir: u32,
        hv: Real,
    ) {
        flux_body(
            state_tmp,
            flux,
            hy_d,
            hy_dt,
            hy_di,
            hy_dti,
            hy_pi,
            id.x,
            nx as usize,
            nz as usize,
            dir,
            hv,
        );
    }

    // The tend stage writes ALL of `tend` (4 planes of nz x nx, row-major in
    // exactly the `[nx, nz, 4]` grid order) — one work-item per element, and
    // `-> Written` proves no element is left stale from the previous stage.
    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn tend_s(
        #[spirv(cross_workgroup)] state: &[Real],
        #[spirv(cross_workgroup)] _state_tmp: &[Real],
        #[spirv(cross_workgroup)] flux: &[Real],
        #[spirv(cross_workgroup)] tend: PerItemOut<Real>,
        nx: u32,
        nz: u32,
        dir: u32,
        dx: Real,
        dz: Real,
    ) -> Written {
        let c = tend.coord();
        let v = tend_at(
            flux,
            state,
            c.x,
            c.y,
            c.z,
            nx as usize,
            nz as usize,
            dir,
            dx,
            dz,
        );
        Written {
            tend: tend.write(v),
        }
    }

    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn tend_t(
        #[spirv(cross_workgroup)] _state: &[Real],
        #[spirv(cross_workgroup)] state_tmp: &[Real],
        #[spirv(cross_workgroup)] flux: &[Real],
        #[spirv(cross_workgroup)] tend: PerItemOut<Real>,
        nx: u32,
        nz: u32,
        dir: u32,
        dx: Real,
        dz: Real,
    ) -> Written {
        let c = tend.coord();
        let v = tend_at(
            flux,
            state_tmp,
            c.x,
            c.y,
            c.z,
            nx as usize,
            nz as usize,
            dir,
            dx,
            dz,
        );
        Written {
            tend: tend.write(v),
        }
    }

    /// stages 1 & 2: state_tmp = state + dt * tend, one work-item per `tend`
    /// element on the `[nx, nz, 4]` grid.
    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn update_a(
        #[spirv(cross_workgroup)] state: &[Real],
        #[spirv(cross_workgroup)] state_tmp: &mut [Real],
        #[spirv(cross_workgroup)] _flux: &[Real],
        #[spirv(cross_workgroup)] tend: PerItemIn<Real>,
        nx: u32,
        nz: u32,
        dt: Real,
    ) {
        let c = interior(tend.coord(), nx as usize, nz as usize);
        state_tmp[c] = state[c] + dt * *tend;
    }

    /// stage 3: state = state + dt * tend (in place)
    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn update_b(
        #[spirv(cross_workgroup)] state: &mut [Real],
        #[spirv(cross_workgroup)] _state_tmp: &[Real],
        #[spirv(cross_workgroup)] _flux: &[Real],
        #[spirv(cross_workgroup)] tend: PerItemIn<Real>,
        nx: u32,
        nz: u32,
        dt: Real,
    ) {
        let c = interior(tend.coord(), nx as usize, nz as usize);
        state[c] = state[c] + dt * *tend;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Host: init, reductions, output (all f64), CLI, and the Tier-2 driver.
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq)]
enum Case {
    Thermal,
    Collision,
    DensityCurrent,
    Injection,
}

struct Config {
    nx: usize,
    nz: usize,
    sim_time: f64,
    case: Case,
    dump: Option<String>,
    precision: Option<bool>, // Some(true) => force f64, Some(false) => force f32
}

fn hydro_const_theta(z: f64) -> (f64, f64) {
    let theta0 = 300.0;
    let exner0 = 1.0;
    let t = theta0;
    let exner = exner0 - GRAV * z / (CP * theta0);
    let p = P0 * exner.powf(CP / RD);
    let rt = (p / C0).powf(1.0 / GAMMA);
    (rt / t, t)
}

fn sample_ellipse_cosine(x: f64, z: f64, amp: f64, x0: f64, z0: f64, xrad: f64, zrad: f64) -> f64 {
    let dist = (((x - x0) / xrad).powi(2) + ((z - z0) / zrad).powi(2)).sqrt() * PI / 2.0;
    if dist <= PI / 2.0 {
        amp * dist.cos().powi(2)
    } else {
        0.0
    }
}

/// (r, u, w, t, hr, ht) at a point, per data spec.
fn case_state(case: Case, x: f64, z: f64) -> (f64, f64, f64, f64, f64, f64) {
    let (hr, ht) = hydro_const_theta(z);
    let t = match case {
        Case::Injection => 0.0,
        Case::DensityCurrent => {
            sample_ellipse_cosine(x, z, -20.0, XLEN / 2.0, 5000.0, 4000.0, 2000.0)
        }
        Case::Thermal => sample_ellipse_cosine(x, z, 3.0, XLEN / 2.0, 2000.0, 2000.0, 2000.0),
        Case::Collision => {
            sample_ellipse_cosine(x, z, 20.0, XLEN / 2.0, 2000.0, 2000.0, 2000.0)
                + sample_ellipse_cosine(x, z, -20.0, XLEN / 2.0, 8000.0, 2000.0, 2000.0)
        }
    };
    (0.0, 0.0, 0.0, t, hr, ht)
}

struct Init {
    state: Vec<f64>,
    hy_d: Vec<f64>,
    hy_dt: Vec<f64>,
    hy_di: Vec<f64>,
    hy_dti: Vec<f64>,
    hy_pi: Vec<f64>,
}

fn initialize(cfg: &Config) -> Init {
    let (nx, nz) = (cfg.nx, cfg.nz);
    let dx = XLEN / nx as f64;
    let dz = ZLEN / nz as f64;
    let nxp = nx + 2 * HS;
    let nzp = nz + 2 * HS;
    let plane = nzp * nxp;
    let mut state = vec![0.0f64; 4 * plane];
    for kh in 0..nzp {
        let fk = kh as f64 - 1.0; // Fortran k = kh - HS + 1, so k - 0.5 = kh - 1.5 ... see below
        for ih in 0..nxp {
            let fi = ih as f64 - 1.0;
            // Fortran cell centers: (i - 0.5)*dx for i in 1-hs ..= nx+hs.
            // With ih = i + hs - 1: i - 0.5 = ih - 1.5.
            let c = kh * nxp + ih;
            for kk in 0..3 {
                for ii in 0..3 {
                    let x = (fi - 0.5) * dx + (QPOINTS[ii] - 0.5) * dx;
                    let z = (fk - 0.5) * dz + (QPOINTS[kk] - 0.5) * dz;
                    let (r, u, w, t, hr, ht) = case_state(cfg.case, x, z);
                    let qw = QWEIGHTS[ii] * QWEIGHTS[kk];
                    state[c] += r * qw;
                    state[plane + c] += (r + hr) * u * qw;
                    state[2 * plane + c] += (r + hr) * w * qw;
                    state[3 * plane + c] += ((r + hr) * (t + ht) - hr * ht) * qw;
                }
            }
        }
    }
    let mut hy_d = vec![0.0f64; nzp];
    let mut hy_dt = vec![0.0f64; nzp];
    for kh in 0..nzp {
        let fk = kh as f64 - 1.0;
        for kk in 0..3 {
            let z = (fk - 0.5) * dz + (QPOINTS[kk] - 0.5) * dz;
            let (_, _, _, _, hr, ht) = case_state(cfg.case, 0.0, z);
            hy_d[kh] += hr * QWEIGHTS[kk];
            hy_dt[kh] += hr * ht * QWEIGHTS[kk];
        }
    }
    let mut hy_di = vec![0.0f64; nz + 1];
    let mut hy_dti = vec![0.0f64; nz + 1];
    let mut hy_pi = vec![0.0f64; nz + 1];
    for k in 0..=nz {
        let z = k as f64 * dz;
        let (_, _, _, _, hr, ht) = case_state(cfg.case, 0.0, z);
        hy_di[k] = hr;
        hy_dti[k] = hr * ht;
        hy_pi[k] = C0 * (hr * ht).powf(GAMMA);
    }
    Init {
        state,
        hy_d,
        hy_dt,
        hy_di,
        hy_dti,
        hy_pi,
    }
}

/// Domain-integrated mass and total energy, exactly as the Fortran computes.
fn reductions(state: &[f64], init: &Init, nx: usize, nz: usize) -> (f64, f64) {
    let dx = XLEN / nx as f64;
    let dz = ZLEN / nz as f64;
    let nxp = nx + 2 * HS;
    let plane = (nz + 2 * HS) * nxp;
    let (mut mass, mut te) = (0.0f64, 0.0f64);
    for k in 0..nz {
        for i in 0..nx {
            let c = (k + HS) * nxp + (i + HS);
            let r = state[c] + init.hy_d[k + HS];
            let u = state[plane + c] / r;
            let w = state[2 * plane + c] / r;
            let th = (state[3 * plane + c] + init.hy_dt[k + HS]) / r;
            let p = C0 * (r * th).powf(GAMMA);
            let t = th / (P0 / p).powf(RD / CP);
            let ke = r * (u * u + w * w);
            mass += r * dx * dz;
            te += (ke + r * CV * t) * dx * dz;
        }
    }
    (mass, te)
}

/// theta/u/w perturbation fields, (nz, nx) row-major, as the Fortran `output`.
fn output_fields(state: &[f64], init: &Init, nx: usize, nz: usize) -> Vec<f64> {
    let nxp = nx + 2 * HS;
    let plane = (nz + 2 * HS) * nxp;
    let mut out = vec![0.0f64; 3 * nz * nx];
    for k in 0..nz {
        for i in 0..nx {
            let c = (k + HS) * nxp + (i + HS);
            let hd = init.hy_d[k + HS];
            let hdt = init.hy_dt[k + HS];
            let dens = state[c];
            out[k * nx + i] = (state[3 * plane + c] + hdt) / (hd + dens) - hdt / hd;
            out[nz * nx + k * nx + i] = state[plane + c] / (hd + dens);
            out[2 * nz * nx + k * nx + i] = state[2 * plane + c] / (hd + dens);
        }
    }
    out
}

// The dt-dependent scalar slots, declared ONCE for every width via the
// generic-value `slots!` arm (`Tag<R>: R`) — each width instantiation is an
// independent slot identity (`KeyFor<R>`), so the f64 and f32 graphs can
// never cross-match. The graph is one DOUBLE step (x-first step, then
// z-first step); directions and launch extents are literals, so these
// scalars — per-step RK stage dts and hyperviscosity per (step, half,
// stage) — are the only slots, re-bound at most twice per run.
slots! {
    Dt1A<R>: R, Dt1B<R>: R, Dt1C<R>: R,
    Dt2A<R>: R, Dt2B<R>: R, Dt2C<R>: R,
    H1A1<R>: R, H1A2<R>: R, H1A3<R>: R, // step 1, x half
    H1B1<R>: R, H1B2<R>: R, H1B3<R>: R, // step 1, z half
    H2A1<R>: R, H2A2<R>: R, H2A3<R>: R, // step 2, z half
    H2B1<R>: R, H2B2<R>: R, H2B3<R>: R, // step 2, x half
}

// Slots for the NON-UNROLLED graph: one graph = one timestep = two halves,
// mirroring the reference `perform_timestep`. Everything that differs between
// the two halves is a slot rather than a literal, so the SAME graph runs
// x-then-z on odd steps and z-then-x on even ones. Both halves of a timestep
// share the stage dts (the reference passes dt/3, dt/2, dt/1 to all six
// `semi_discrete_step` calls), so there are three of those, not six.
slots! {
    PtA<R>: R, PtB<R>: R, PtC<R>: R,       // stage dts, shared by both halves
    PvA1<R>: R, PvA2<R>: R, PvA3<R>: R,    // hyperviscosity, first half
    PvB1<R>: R, PvB2<R>: R, PvB3<R>: R,    // hyperviscosity, second half
}

// Direction and launch extents are width-INDEPENDENT (u32 / LaunchSpec), so
// unlike the dt-carrying tags above these need no generic parameter: the f64
// and f32 graphs can share the tag type because `bind_slots` walks one graph
// and only ever fills that graph's cells.
slots! {
    PdA: u32, PdB: u32,                    // DIR_X = 0 / DIR_Z = 1, per half
    PgA: LaunchSpec, PfA: LaunchSpec,      // halo / flux extents, first half
    PgB: LaunchSpec, PfB: LaunchSpec,      // halo / flux extents, second half
}

// ── Tier-2 driver, instantiated per precision ────────────────────────────────

/// The example's numeric-width bound: host-boundary casts for a
/// width-generic driver (claspr itself stays numeric-agnostic).
trait Width:
    Copy + Default + Send + Sync + 'static + claspr::SlotValue + claspr::eager::SlotEq
{
    fn of(v: f64) -> Self;
    fn to_f64(self) -> f64;
}
impl Width for f64 {
    fn of(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}
impl Width for f32 {
    fn of(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// Command buffers graph `g` has recorded over its life, read from its root's
/// own cache. Not the process-wide `cb_records_total()`: that also counts every
/// OTHER graph's records — under `cargo test` the concurrently running tests'
/// — which made the per-timestep CB-count assertion flake (~1 in 8 parallel runs).
fn cbs_recorded<O: DeviceOp>(g: &O) -> u64 {
    g.cb_cache()
        .map(|c| c.lock().unwrap().records())
        .unwrap_or(0)
}

/// Which graph shape to build. Both compute the same thing; they differ only in
/// whether the x-first/z-first alternation is UNROLLED into the graph or carried
/// by slots and flipped per step.
///
/// Only ONE is built per `run` call: the other would be dead weight. Stack is no
/// longer the constraint — `and_then` erases the chain it extends, so a debug
/// build of `run` (both arms in one frame) needs under 256 KiB of a test
/// thread's 2 MiB (measured 2026-10-27).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    /// Two timesteps per graph, directions and extents as literals. One binding
    /// state, so a steady-state replay mutates nothing.
    Unrolled,
    /// One timestep per graph, mirroring the reference's `perform_timestep`.
    /// Direction, extents and hyperviscosity ride slots; two binding states,
    /// both held by the keyed command-buffer cache.
    PerTimestep,
}

/// One device run of one shape.
struct RunOut {
    /// Initial state after the precision round-trip — what the device evolved from.
    state0: Vec<f64>,
    /// Final state.
    final_: Vec<f64>,
    secs: f64,
    /// `sync` calls performed (the unrolled shape advances TWO timesteps per sync).
    syncs: u64,
    /// Command buffers RECORDED — the number the keyed cache exists to hold down.
    cbs: u64,
}

/// Run the case on the device with ONE graph `shape`. Running both and comparing
/// the final states bit-for-bit is the equivalence proof — separate graphs over
/// separate buffers from identical host state. Mirrors gray-scott's
/// swap-vs-immutable pairing.
fn run<Real, K>(ctx: &Context, cfg: &Config, init: &Init, shape: Shape) -> claspr::Result<RunOut>
where
    K: gpu::Kernels<Real>,
    Real: Width,
    claspr::ScalarInput<Real>: From<Real>,
{
    let (nx, nz) = (cfg.nx, cfg.nz);
    let dx = XLEN / nx as f64;
    let dzf = ZLEN / nz as f64;
    let nxp = nx + 2 * HS;
    let state_len = 4 * (nz + 2 * HS) * nxp;

    let cast = |v: &[f64]| -> Vec<Real> { v.iter().map(|&x| Real::of(x)).collect() };
    let state_h = cast(&init.state);
    let state0: Vec<f64> = state_h.iter().map(|&x| x.to_f64()).collect();

    let ks = K::kernels(ctx)?;
    let s = DeviceSlice::<Real>::from_vec(ctx, state_h.clone())?;
    let t = DeviceSlice::<Real>::from_vec(ctx, state_h)?;
    let fx = DeviceSlice::<Real>::alloc_zero(ctx, 4 * (nz + 1) * (nx + 1))?;
    let td = DeviceSlice::<Real>::alloc_zero(ctx, 4 * nz * nx)?;

    // Read-only hydrostatic arrays: Arc-shared, one clone bound as
    // a literal at every dispatch site that reads them (the Arc
    // fan-out pattern; no pipe threading, no slots).
    let hd = std::sync::Arc::new(DeviceSlice::<Real>::from_vec(ctx, cast(&init.hy_d))?);
    let hdt = std::sync::Arc::new(DeviceSlice::<Real>::from_vec(ctx, cast(&init.hy_dt))?);
    let hdi = std::sync::Arc::new(DeviceSlice::<Real>::from_vec(ctx, cast(&init.hy_di))?);
    let hdti = std::sync::Arc::new(DeviceSlice::<Real>::from_vec(ctx, cast(&init.hy_dti))?);
    let hpi = std::sync::Arc::new(DeviceSlice::<Real>::from_vec(ctx, cast(&init.hy_pi))?);
    let (hd, hdt, hdi, hdti, hpi) = (&hd, &hdt, &hdi, &hdti, &hpi);

    let nxu = nx as u32;
    let nzu = nz as u32;
    let inj = if cfg.case == Case::Injection {
        1u32
    } else {
        0u32
    };
    let dz_t = Real::of(dzf);
    let dx_t = Real::of(dx);
    // tend/update grid: one work-item per `tend` element (the per-item views'
    // launch check enforces nx * nz * 4 == td.len()).
    let gt = LaunchSpec::from([nx, nz, 4]);
    let ks = &ks;

    // One graph = one DOUBLE timestep: step 1 is x-first (halves
    // x, z), step 2 is z-first (halves z, x). Directions and
    // launch extents are literals, so a steady-state replay
    // mutates nothing and the recorded command buffer is reused.
    let gh_x = LaunchSpec::from([nz]);
    let gh_z = LaunchSpec::from([nxp]);
    let gf_x = LaunchSpec::from([(nx + 1) * nz]);
    let gf_z = LaunchSpec::from([nx * (nz + 1)]);

    // Twelve dispatches of one half-step (3 RK stages x 4 kernels),
    // appended to a chain whose output is the four mutable buffers `(s, t, fx,
    // td)` — they ride the pipes; the hy arrays enter as Arc clones per site.
    // The first half is appended to `seed()`, which presents the owned buffers
    // as such a chain. Free identifiers (ks, gt, hd, ...) resolve at the
    // expansion site.
    // rustfmt oscillates on the parameter continuation line of a
    // macro_rules! nested this deep (adds indentation every pass)
    // — freeze it.
    #[rustfmt::skip]
        macro_rules! half {
            ($g:expr, $gh:expr, $gf:expr, $d:expr, $hv1:ident, $hv2:ident, $hv3:ident,
                                     $dta:ident, $dtb:ident, $dtc:ident) => {
                $g.and_then(move |(s, t, fx, td)| {
                    ks.halo_s(
                        $gh,
                        s,
                        t,
                        fx,
                        td,
                        hd.clone(),
                        hdt.clone(),
                        nxu,
                        nzu,
                        $d,
                        inj,
                        dz_t,
                    )
                })
                .and_then(move |(s, t, fx, td, _hd, _hdt)| {
                    ks.flux_s(
                        $gf,
                        s,
                        t,
                        fx,
                        td,
                        hd.clone(),
                        hdt.clone(),
                        hdi.clone(),
                        hdti.clone(),
                        hpi.clone(),
                        nxu,
                        nzu,
                        $d,
                        slot!($hv1<Real>),
                    )
                })
                .and_then(move |(s, t, fx, td, _h1, _h2, _h3, _h4, _h5)| {
                    ks.tend_s(gt, s, t, fx, td, nxu, nzu, $d, dx_t, dz_t)
                })
                .and_then(move |(s, t, fx, td)| {
                    ks.update_a(gt, s, t, fx, td, nxu, nzu, slot!($dta<Real>))
                })
                .and_then(move |(s, t, fx, td)| {
                    ks.halo_t(
                        $gh,
                        s,
                        t,
                        fx,
                        td,
                        hd.clone(),
                        hdt.clone(),
                        nxu,
                        nzu,
                        $d,
                        inj,
                        dz_t,
                    )
                })
                .and_then(move |(s, t, fx, td, _hd, _hdt)| {
                    ks.flux_t(
                        $gf,
                        s,
                        t,
                        fx,
                        td,
                        hd.clone(),
                        hdt.clone(),
                        hdi.clone(),
                        hdti.clone(),
                        hpi.clone(),
                        nxu,
                        nzu,
                        $d,
                        slot!($hv2<Real>),
                    )
                })
                .and_then(move |(s, t, fx, td, _h1, _h2, _h3, _h4, _h5)| {
                    ks.tend_t(gt, s, t, fx, td, nxu, nzu, $d, dx_t, dz_t)
                })
                .and_then(move |(s, t, fx, td)| {
                    ks.update_a(gt, s, t, fx, td, nxu, nzu, slot!($dtb<Real>))
                })
                .and_then(move |(s, t, fx, td)| {
                    ks.halo_t(
                        $gh,
                        s,
                        t,
                        fx,
                        td,
                        hd.clone(),
                        hdt.clone(),
                        nxu,
                        nzu,
                        $d,
                        inj,
                        dz_t,
                    )
                })
                .and_then(move |(s, t, fx, td, _hd, _hdt)| {
                    ks.flux_t(
                        $gf,
                        s,
                        t,
                        fx,
                        td,
                        hd.clone(),
                        hdt.clone(),
                        hdi.clone(),
                        hdti.clone(),
                        hpi.clone(),
                        nxu,
                        nzu,
                        $d,
                        slot!($hv3<Real>),
                    )
                })
                .and_then(move |(s, t, fx, td, _h1, _h2, _h3, _h4, _h5)| {
                    ks.tend_t(gt, s, t, fx, td, nxu, nzu, $d, dx_t, dz_t)
                })
                .and_then(move |(s, t, fx, td)| {
                    ks.update_b(gt, s, t, fx, td, nxu, nzu, slot!($dtc<Real>))
                })
            };
        }

    // The chain's starting point: the four owned buffers as graph edges (`lift`
    // presents an owned buffer as a self-rehoming input, no device work), so
    // every half — the first included — is appended with `half!`.
    let seed = move || bundle4(lift(s), lift(t), lift(fx), lift(td));

    // hv(d, stage, dt) with the dt = 0 no-op convention
    let hv = |d: f64, q: f64, dt: f64| -> Real {
        if dt == 0.0 {
            Real::of(0.0)
        } else {
            Real::of(-HV_BETA * d / (16.0 * (dt / q)))
        }
    };

    let dt0 = dx.min(dzf) / MAX_SPEED * CFL;
    let bind_all = |dt1: f64, dt2: f64| {
        (
            Dt1A(Real::of(dt1 / 3.0)),
            Dt1B(Real::of(dt1 / 2.0)),
            Dt1C(Real::of(dt1)),
            Dt2A(Real::of(dt2 / 3.0)),
            Dt2B(Real::of(dt2 / 2.0)),
            Dt2C(Real::of(dt2)),
            H1A1(hv(dx, 3.0, dt1)),
            H1A2(hv(dx, 2.0, dt1)),
            H1A3(hv(dx, 1.0, dt1)),
            H1B1(hv(dzf, 3.0, dt1)),
            H1B2(hv(dzf, 2.0, dt1)),
            H1B3(hv(dzf, 1.0, dt1)),
            H2A1(hv(dzf, 3.0, dt2)),
            H2A2(hv(dzf, 2.0, dt2)),
            H2A3(hv(dzf, 1.0, dt2)),
            H2B1(hv(dx, 3.0, dt2)),
            H2B2(hv(dx, 2.0, dt2)),
            H2B3(hv(dx, 1.0, dt2)),
        )
    };
    // ── Graph B bindings: one timestep, direction chosen by slot. ────────────
    // `first_is_x` IS the reference's `direction_switch`. Everything that
    // distinguishes the two halves — direction flag, halo/flux extents, and the
    // hyperviscosity coefficients (which carry the grid spacing, dx vs dz) —
    // flips together, so the graph has exactly TWO binding states while dt holds
    // steady. Both live in the keyed CB cache, so neither is ever re-recorded.
    let bind_per = |first_is_x: bool, dt: f64| {
        let (da, db) = if first_is_x {
            (0u32, 1u32)
        } else {
            (1u32, 0u32)
        };
        let (gha, gfa, ghb, gfb) = if first_is_x {
            (gh_x, gf_x, gh_z, gf_z)
        } else {
            (gh_z, gf_z, gh_x, gf_x)
        };
        // Spacing follows the direction: hv_coef = -hv_beta * d / (16*dt).
        let (sa, sb) = if first_is_x { (dx, dzf) } else { (dzf, dx) };
        (
            PtA(Real::of(dt / 3.0)),
            PtB(Real::of(dt / 2.0)),
            PtC(Real::of(dt)),
            PvA1(hv(sa, 3.0, dt)),
            PvA2(hv(sa, 2.0, dt)),
            PvA3(hv(sa, 1.0, dt)),
            PvB1(hv(sb, 3.0, dt)),
            PvB2(hv(sb, 2.0, dt)),
            PvB3(hv(sb, 1.0, dt)),
            PdA(da),
            PdB(db),
            PgA(gha),
            PfA(gfa),
            PgB(ghb),
            PfB(gfb),
        )
    };

    // Only the selected shape is built (see `Shape`).
    let (final_state, secs, syncs, cbs) = match shape {
        Shape::Unrolled => {
            // ── Graph A: the UNROLLED double timestep (x-first step, then z-first). ──
            // Directions and extents are literals, so a steady-state replay mutates
            // nothing at all. Two timesteps per sync.
            let g0 = half!(seed(), gh_x, gf_x, 0u32, H1A1, H1A2, H1A3, Dt1A, Dt1B, Dt1C);
            let g1 = half!(g0, gh_z, gf_z, 1u32, H1B1, H1B2, H1B3, Dt1A, Dt1B, Dt1C);
            let g2 = half!(g1, gh_z, gf_z, 1u32, H2A1, H2A2, H2A3, Dt2A, Dt2B, Dt2C);
            let g = half!(g2, gh_x, gf_x, 0u32, H2B1, H2B2, H2B3, Dt2A, Dt2B, Dt2C);

            let g = g.call(bind_all(dt0, dt0));

            let t_unrolled = std::time::Instant::now();
            let mut etime = 0.0f64;
            let mut dt = dt0;
            let mut bound = (dt0, dt0);
            let mut final_state = vec![Real::default(); state_len];
            let mut nstep = 0u64;
            while etime < cfg.sim_time {
                // derive this pair's two step dts with the Fortran's exact
                // clamp arithmetic; dt2 = 0 means "no-op second step"
                if etime + dt > cfg.sim_time {
                    dt = cfg.sim_time - etime;
                }
                let dt1 = dt;
                let e1 = etime + dt1;
                let dt2 = if e1 < cfg.sim_time {
                    if e1 + dt > cfg.sim_time {
                        cfg.sim_time - e1
                    } else {
                        dt
                    }
                } else {
                    0.0
                };
                if (dt1, dt2) != bound {
                    // One call: the rebind is all-or-nothing.
                    g.mutate_call(bind_all(dt1, dt2))?;
                    bound = (dt1, dt2);
                }
                if dt2 != 0.0 {
                    dt = dt2; // the Fortran dt variable persists across steps
                }
                let co = g.sync(ctx)?;
                etime = e1 + dt2;
                nstep += if dt2 != 0.0 { 2 } else { 1 };
                if etime >= cfg.sim_time {
                    let (cs, _ct, _cfx, _ctd) = co;
                    cs.read_into(&mut final_state)?;
                }
            }
            let secs_unrolled = t_unrolled.elapsed().as_secs_f64();
            let cbs_unrolled = cbs_recorded(&g);

            (final_state, secs_unrolled, nstep.div_ceil(2), cbs_unrolled)
        }
        Shape::PerTimestep => {
            // ── Graph B: the NON-UNROLLED single timestep. ───────────────────────────
            // One graph = one `perform_timestep`: two halves, three RK stages each.
            // Direction and extents ride slots, so the SAME graph is x-then-z on odd
            // steps and z-then-x on even ones — the reference's `direction_switch`,
            // expressed as a rebind instead of as unrolled code. Two binding states,
            // both resident in the keyed command-buffer cache, so the steady state
            // records nothing (see NOTES, "keyed LRU").
            let p0 = half!(
                seed(),
                slot!(PgA),
                slot!(PfA),
                slot!(PdA),
                PvA1,
                PvA2,
                PvA3,
                PtA,
                PtB,
                PtC
            );
            let gp = half!(
                p0,
                slot!(PgB),
                slot!(PfB),
                slot!(PdB),
                PvB1,
                PvB2,
                PvB3,
                PtA,
                PtB,
                PtC
            );

            let gp = gp.call(bind_per(true, dt0));

            // ── Graph B loop: one sync per timestep, exactly the reference's loop. ───
            //   do while (etime < sim_time)
            //     if (etime + dt > sim_time) dt = sim_time - etime
            //     call perform_timestep(...); etime = etime + dt
            let t_per_step = std::time::Instant::now();
            let mut etime_p = 0.0f64;
            let mut dt_p = dt0;
            let mut first_is_x = true;
            let mut bound_p = (true, dt0);
            let mut final_p = vec![Real::default(); state_len];
            let mut nstep_p = 0u64;
            while etime_p < cfg.sim_time {
                if etime_p + dt_p > cfg.sim_time {
                    dt_p = cfg.sim_time - etime_p;
                }
                if (first_is_x, dt_p) != bound_p {
                    gp.mutate_call(bind_per(first_is_x, dt_p))?;
                    bound_p = (first_is_x, dt_p);
                }
                let co = gp.sync(ctx)?;
                etime_p += dt_p;
                nstep_p += 1;
                first_is_x = !first_is_x;
                if etime_p >= cfg.sim_time {
                    let (cs, _ct, _cfx, _ctd) = co;
                    cs.read_into(&mut final_p)?;
                }
            }
            let secs_per_step = t_per_step.elapsed().as_secs_f64();
            let cbs_per_step = cbs_recorded(&gp);

            (final_p, secs_per_step, nstep_p, cbs_per_step)
        }
    };
    eprintln!("{shape:?}: {syncs} syncs, {cbs} command buffers recorded");
    Ok(RunOut {
        state0,
        final_: final_state.iter().map(|&x| x.to_f64()).collect(),
        secs,
        syncs,
        cbs,
    })
}

// ── CLI ──────────────────────────────────────────────────────────────────────

fn parse_args() -> Config {
    let mut cfg = Config {
        nx: 100,
        nz: 50,
        sim_time: 400.0,
        case: Case::Thermal,
        dump: None,
        precision: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || {
            args.next()
                .unwrap_or_else(|| panic!("missing value for {a}"))
        };
        match a.as_str() {
            "--nx" => cfg.nx = val().parse().expect("--nx"),
            "--nz" => cfg.nz = val().parse().expect("--nz"),
            "--sim-time" => cfg.sim_time = val().parse().expect("--sim-time"),
            "--dump" => cfg.dump = Some(val()),
            "--precision" => {
                cfg.precision = match val().as_str() {
                    "f64" => Some(true),
                    "f32" => Some(false),
                    p => panic!("unknown precision {p} (f32|f64)"),
                }
            }
            "--case" => {
                cfg.case = match val().as_str() {
                    "thermal" => Case::Thermal,
                    "collision" => Case::Collision,
                    "density_current" => Case::DensityCurrent,
                    "injection" => Case::Injection,
                    c => {
                        eprintln!("case {c} not supported by the claspr port");
                        std::process::exit(2);
                    }
                }
            }
            other => panic!("unknown arg {other}"),
        }
    }
    cfg
}

fn main() -> claspr::Result<()> {
    let cfg = parse_args();
    let ctx = Context::any()?;
    let dev = ctx.device();
    let fp64 = dev
        .cl3()
        .double_fp_config()
        .map(|v| v != 0)
        .unwrap_or(false);
    let use_f64 = cfg.precision.unwrap_or(fp64);
    if use_f64 && !fp64 {
        eprintln!("device has no fp64 support; use --precision f32");
        std::process::exit(2);
    }
    eprintln!(
        "device: {} | precision: {}",
        dev.cl3().name().unwrap_or_else(|_| "?".into()),
        if use_f64 { "f64" } else { "f32" },
    );

    println!("nx_glob, nz_glob: {} {}", cfg.nx, cfg.nz);
    println!("dx,dz: {} {}", XLEN / cfg.nx as f64, ZLEN / cfg.nz as f64);
    println!(
        "dt: {}",
        (XLEN / cfg.nx as f64).min(ZLEN / cfg.nz as f64) / MAX_SPEED * CFL
    );

    let init = initialize(&cfg);
    let t0 = std::time::Instant::now();
    // Run BOTH shapes and compare. They are one meta-kernel: same kernels, same
    // order, same scalars — the only difference is whether the direction
    // alternation is unrolled into the graph or carried by slots the keyed
    // command-buffer cache keys on. Byte equality of the final state is the proof.
    let go = |shape| -> claspr::Result<RunOut> {
        if use_f64 {
            run::<f64, gpu::f64::Kernels>(&ctx, &cfg, &init, shape)
        } else {
            run::<f32, gpu::f32::Kernels>(&ctx, &cfg, &init, shape)
        }
    };
    let a = go(Shape::Unrolled)?;
    let b = go(Shape::PerTimestep)?;
    let (state0, state1) = (&a.state0, &a.final_);
    eprintln!("device loop: {:.3}s", t0.elapsed().as_secs_f64());

    println!(
        "unrolled double-step:     {:.3}s  {:>4} syncs  {:>2} CBs recorded  {:.0}us/sync",
        a.secs,
        a.syncs,
        a.cbs,
        a.secs * 1e6 / a.syncs as f64
    );
    println!(
        "non-unrolled per-step:    {:.3}s  {:>4} syncs  {:>2} CBs recorded  {:.0}us/sync",
        b.secs,
        b.syncs,
        b.cbs,
        b.secs * 1e6 / b.syncs as f64
    );
    println!(
        "per-timestep vs unrolled: {:+.1}% wall -- final state {}",
        (b.secs / a.secs - 1.0) * 100.0,
        if a.final_ == b.final_ {
            "BIT-IDENTICAL".to_string()
        } else {
            let worst = a
                .final_
                .iter()
                .zip(&b.final_)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max);
            format!("DIFFERS (max abs {worst:.3e})")
        }
    );

    let (mass0, te0) = reductions(state0, &init, cfg.nx, cfg.nz);
    let (mass, te) = reductions(state1, &init, cfg.nx, cfg.nz);
    println!("d_mass: {:.15e}", (mass - mass0) / mass0);
    println!("d_te:   {:.15e}", (te - te0) / te0);

    if let Some(path) = &cfg.dump {
        let fields = output_fields(state1, &init, cfg.nx, cfg.nz);
        let mut bytes = Vec::with_capacity(fields.len() * 8);
        for v in &fields {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write dump");
    }
    Ok(())
}

// ── Self-validation ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> Config {
        Config {
            nx: 100,
            nz: 50,
            sim_time: 400.0,
            case: Case::Thermal,
            dump: None,
            precision: None,
        }
    }

    fn has_fp64(ctx: &Context) -> bool {
        ctx.device()
            .cl3()
            .double_fp_config()
            .map(|v| v != 0)
            .unwrap_or(false)
    }

    /// f64 stamp, thermal 400s on the default grid: mass is conserved to
    /// roundoff and total energy drifts only slightly (observed on PoCL:
    /// d_mass 1.3e-14, d_te -4.14e-5 — bit-identical to the pre-instantiate
    /// explicit-compile port).
    #[test]
    fn f64_thermal_conserves_mass_and_energy() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        if !has_fp64(&ctx) {
            eprintln!("SKIP: device has no Float64 capability");
            return;
        }
        let cfg = test_cfg();
        let init = initialize(&cfg);
        let r = run::<f64, gpu::f64::Kernels>(&ctx, &cfg, &init, Shape::Unrolled).expect("run f64");
        let (s0, s1) = (&r.state0, &r.final_);
        let (m0, e0) = reductions(s0, &init, cfg.nx, cfg.nz);
        let (m1, e1) = reductions(s1, &init, cfg.nx, cfg.nz);
        let dm = ((m1 - m0) / m0).abs();
        let de = ((e1 - e0) / e0).abs();
        assert!(dm < 1e-12, "f64 mass drift too large: {dm:e}");
        assert!(de < 5e-4, "f64 energy drift too large: {de:e}");
    }

    /// f32 stamp alone (no fp64 gate — this must run on ANY device, which is
    /// the point of stamping: the f32 module never declares Float64).
    /// Observed on PoCL: d_mass 1.2e-11, d_te -4.14e-5.
    #[test]
    fn f32_thermal_conserves_mass_and_energy() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        let cfg = test_cfg();
        let init = initialize(&cfg);
        let r = run::<f32, gpu::f32::Kernels>(&ctx, &cfg, &init, Shape::Unrolled).expect("run f32");
        let (s0, s1) = (&r.state0, &r.final_);
        let (m0, e0) = reductions(s0, &init, cfg.nx, cfg.nz);
        let (m1, e1) = reductions(s1, &init, cfg.nx, cfg.nz);
        let dm = ((m1 - m0) / m0).abs();
        let de = ((e1 - e0) / e0).abs();
        assert!(dm < 1e-8, "f32 mass drift too large: {dm:e}");
        assert!(de < 5e-4, "f32 energy drift too large: {de:e}");
    }

    /// **The two graph shapes are one meta-kernel.** The unrolled graph hard-codes
    /// the x-first/z-first alternation by fusing two timesteps; the non-unrolled
    /// one carries direction, extents and hyperviscosity in slots and flips them
    /// per step, mirroring the reference's `direction_switch`. Same kernels, same
    /// order, same scalars — so the final states must agree BIT-for-bit, not
    /// approximately. Separate graphs over separate buffers, so agreement is a
    /// real equivalence proof rather than aliasing.
    ///
    /// Also pins the payoff: the per-timestep shape alternates between two
    /// binding states and must record a SMALL CONSTANT number of command buffers
    /// (two states plus the final clipped-dt one), not one per sync. Without the
    /// keyed cache this would be one record per sync.
    #[test]
    fn unrolled_and_per_timestep_are_bit_identical() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        let cfg = test_cfg();
        let init = initialize(&cfg);
        let go = |shape| {
            if has_fp64(&ctx) {
                run::<f64, gpu::f64::Kernels>(&ctx, &cfg, &init, shape)
            } else {
                run::<f32, gpu::f32::Kernels>(&ctx, &cfg, &init, shape)
            }
            .expect("run")
        };
        let a = go(Shape::Unrolled);
        let b = go(Shape::PerTimestep);
        assert_eq!(
            a.final_.len(),
            b.final_.len(),
            "both shapes must produce the same state length"
        );
        if a.final_ != b.final_ {
            let (i, x, y) = a
                .final_
                .iter()
                .zip(&b.final_)
                .enumerate()
                .find(|(_, (x, y))| x != y)
                .map(|(i, (x, y))| (i, *x, *y))
                .expect("lengths equal and vectors differ");
            panic!("first divergence at {i}: unrolled {x:.17e} vs per-timestep {y:.17e}");
        }
        // A CB-capable device should hold this to a handful; a device without the
        // extension records none at all, so only assert the upper bound.
        assert!(
            b.cbs <= 4,
            "per-timestep shape recorded {} command buffers over {} syncs -- the \
             keyed cache should hold it to the two direction states plus the final \
             clipped-dt one",
            b.cbs,
            b.syncs
        );
    }

    /// Cross-width agreement: the two stamps run the SAME kernel source, so
    /// after 400s the output fields must agree to single-precision accuracy
    /// (observed max |f64 - f32| = 3.1e-4 on PoCL).
    #[test]
    fn f32_matches_f64_within_single_precision() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        if !has_fp64(&ctx) {
            eprintln!("SKIP: device has no Float64 capability");
            return;
        }
        let cfg = test_cfg();
        let init = initialize(&cfg);
        let s64 = run::<f64, gpu::f64::Kernels>(&ctx, &cfg, &init, Shape::Unrolled)
            .expect("run f64")
            .final_;
        let s32 = run::<f32, gpu::f32::Kernels>(&ctx, &cfg, &init, Shape::Unrolled)
            .expect("run f32")
            .final_;
        let f64_fields = output_fields(&s64, &init, cfg.nx, cfg.nz);
        let f32_fields = output_fields(&s32, &init, cfg.nx, cfg.nz);
        let max_diff = f64_fields
            .iter()
            .zip(&f32_fields)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 5e-3,
            "f32 and f64 stamps diverged: max field diff {max_diff:e}",
        );
    }
}
