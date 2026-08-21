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
//! per-width HOST driver still goes through `make_runner!`; that macro is
//! precisely the boilerplate the planned generated `GpuKernels<Real>` trait
//! will erase (see claspr NOTES.md).
//!
//! The whole timestep (6 semi-steps x 4 kernels = 24 dispatches) is ONE
//! Tier-2 eager graph, built once and replayed with `sync` every DOUBLE
//! step. All nine buffers are literals threaded through every dispatch as
//! pipes (they rehome automatically each replay); the per-step direction
//! alternation and launch extents are literals, so a steady-state replay
//! mutates nothing and the recorded command buffer is reused; only the
//! dt-dependent scalars are slots, re-bound when the clamped dt changes.
//!
//! Self-validating: `cargo test -p miniweather-example` runs short thermal
//! integrations at both widths and checks the domain-integrated mass/energy
//! drift (and f32-vs-f64 agreement). Full runs:
//! `cargo run -p miniweather-example -- --nx 400 --nz 200 --sim-time 400`.

use claspr::eager::DeviceOpExt;
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

    #[allow(clippy::too_many_arguments)]
    fn tend_body(
        flux: &[Real],
        tend: &mut [Real],
        f: &[Real],
        gid: usize,
        nx: usize,
        nz: usize,
        dir: u32,
        dx: Real,
        dz: Real,
    ) {
        if gid >= nx * nz {
            return;
        }
        let i = gid % nx;
        let k = gid / nx;
        let nxp = nx + 2 * HS;
        let fplane = (nz + 1) * (nx + 1);
        let mut ll = 0usize;
        while ll < 4 {
            let o = ll * nz * nx + k * nx + i;
            if dir == 0 {
                let fb = ll * fplane + k * (nx + 1) + i;
                tend[o] = -(flux[fb + 1] - flux[fb]) / dx;
            } else {
                let fb = ll * fplane + k * (nx + 1) + i;
                let mut v = -(flux[fb + (nx + 1)] - flux[fb]) / dz;
                if ll == 2 {
                    v -= f[(k + HS) * nxp + (i + HS)] * GRAV;
                }
                tend[o] = v;
            }
            ll += 1;
        }
    }

    fn update_from_body(
        dst: &mut [Real],
        src: &[Real],
        tend: &[Real],
        gid: usize,
        nx: usize,
        nz: usize,
        dt: Real,
    ) {
        if gid >= nx * nz {
            return;
        }
        let i = gid % nx;
        let k = gid / nx;
        let nxp = nx + 2 * HS;
        let plane = (nz + 2 * HS) * nxp;
        let mut ll = 0usize;
        while ll < 4 {
            let c = ll * plane + (k + HS) * nxp + (i + HS);
            dst[c] = src[c] + dt * tend[ll * nz * nx + k * nx + i];
            ll += 1;
        }
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

    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn tend_s(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] state: &[Real],
        #[spirv(cross_workgroup)] _state_tmp: &[Real],
        #[spirv(cross_workgroup)] flux: &[Real],
        #[spirv(cross_workgroup)] tend: &mut [Real],
        nx: u32,
        nz: u32,
        dir: u32,
        dx: Real,
        dz: Real,
    ) {
        tend_body(
            flux,
            tend,
            state,
            id.x,
            nx as usize,
            nz as usize,
            dir,
            dx,
            dz,
        );
    }

    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn tend_t(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] _state: &[Real],
        #[spirv(cross_workgroup)] state_tmp: &[Real],
        #[spirv(cross_workgroup)] flux: &[Real],
        #[spirv(cross_workgroup)] tend: &mut [Real],
        nx: u32,
        nz: u32,
        dir: u32,
        dx: Real,
        dz: Real,
    ) {
        tend_body(
            flux,
            tend,
            state_tmp,
            id.x,
            nx as usize,
            nz as usize,
            dir,
            dx,
            dz,
        );
    }

    /// stages 1 & 2: state_tmp = state + dt * tend
    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn update_a(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] state: &[Real],
        #[spirv(cross_workgroup)] state_tmp: &mut [Real],
        #[spirv(cross_workgroup)] _flux: &[Real],
        #[spirv(cross_workgroup)] tend: &[Real],
        nx: u32,
        nz: u32,
        dt: Real,
    ) {
        update_from_body(state_tmp, state, tend, id.x, nx as usize, nz as usize, dt);
    }

    /// stage 3: state = state + dt * tend (in place)
    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn update_b(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] state: &mut [Real],
        #[spirv(cross_workgroup)] _state_tmp: &[Real],
        #[spirv(cross_workgroup)] _flux: &[Real],
        #[spirv(cross_workgroup)] tend: &[Real],
        nx: u32,
        nz: u32,
        dt: Real,
    ) {
        if id.x >= (nx as usize) * (nz as usize) {
            return;
        }
        let (nx, nz) = (nx as usize, nz as usize);
        let i = id.x % nx;
        let k = id.x / nx;
        let nxp = nx + 2 * HS;
        let plane = (nz + 2 * HS) * nxp;
        let mut ll = 0usize;
        while ll < 4 {
            let c = ll * plane + (k + HS) * nxp + (i + HS);
            state[c] = state[c] + dt * tend[ll * nz * nx + k * nx + i];
            ll += 1;
        }
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

// ── Tier-2 driver, instantiated per precision ────────────────────────────────

macro_rules! make_runner {
    ($runner:ident, $stamp:path, $ty:ty) => {
        mod $runner {
            use super::*;


            /// Run the case on the device; returns (initial, final) state as
            /// f64 (initial = after the precision round-trip, i.e. exactly
            /// what the device evolved from).
            pub fn run(
                ctx: &Context,
                cfg: &Config,
                init: &Init,
            ) -> claspr::Result<(Vec<f64>, Vec<f64>)> {
                let (nx, nz) = (cfg.nx, cfg.nz);
                let dx = XLEN / nx as f64;
                let dzf = ZLEN / nz as f64;
                let nxp = nx + 2 * HS;
                let state_len = 4 * (nz + 2 * HS) * nxp;

                let cast = |v: &[f64]| -> Vec<$ty> { v.iter().map(|&x| x as $ty).collect() };
                let state_h = cast(&init.state);
                let state0: Vec<f64> = state_h.iter().map(|&x| x as f64).collect();

                use $stamp as stamp;
                let ks = stamp::kernels(ctx)?;
                let s = DeviceSlice::<$ty>::from_vec(ctx, state_h.clone())?;
                let t = DeviceSlice::<$ty>::from_vec(ctx, state_h)?;
                let fx = DeviceSlice::<$ty>::alloc_zero(ctx, 4 * (nz + 1) * (nx + 1))?;
                let td = DeviceSlice::<$ty>::alloc_zero(ctx, 4 * nz * nx)?;
                // Read-only hydrostatic arrays: Arc-shared, one clone bound as
                // a literal at every dispatch site that reads them (the Arc
                // fan-out pattern; no pipe threading, no slots).
                let hd = std::sync::Arc::new(DeviceSlice::<$ty>::from_vec(ctx, cast(&init.hy_d))?);
                let hdt =
                    std::sync::Arc::new(DeviceSlice::<$ty>::from_vec(ctx, cast(&init.hy_dt))?);
                let hdi =
                    std::sync::Arc::new(DeviceSlice::<$ty>::from_vec(ctx, cast(&init.hy_di))?);
                let hdti =
                    std::sync::Arc::new(DeviceSlice::<$ty>::from_vec(ctx, cast(&init.hy_dti))?);
                let hpi =
                    std::sync::Arc::new(DeviceSlice::<$ty>::from_vec(ctx, cast(&init.hy_pi))?);
                let (hd, hdt, hdi, hdti, hpi) = (&hd, &hdt, &hdi, &hdti, &hpi);

                let nxu = nx as u32;
                let nzu = nz as u32;
                let inj = if cfg.case == Case::Injection {
                    1u32
                } else {
                    0u32
                };
                let dz_t = dzf as $ty;
                let dx_t = dx as $ty;
                let gt = LaunchSpec::from([nx * nz]);
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
                // Twelve dispatches of one half-step (3 RK stages x 4 kernels),
                // appended to an existing chain. The four mutable buffers ride
                // the pipes; the hy arrays enter as Arc clones per site. Free
                // identifiers (ks, gt, hd, ...) resolve at the expansion site.
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
                                slot!($hv1<$ty>),
                            )
                        })
                        .and_then(move |(s, t, fx, td, _h1, _h2, _h3, _h4, _h5)| {
                            ks.tend_s(gt, s, t, fx, td, nxu, nzu, $d, dx_t, dz_t)
                        })
                        .and_then(move |(s, t, fx, td)| {
                            ks.update_a(gt, s, t, fx, td, nxu, nzu, slot!($dta<$ty>))
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
                                slot!($hv2<$ty>),
                            )
                        })
                        .and_then(move |(s, t, fx, td, _h1, _h2, _h3, _h4, _h5)| {
                            ks.tend_t(gt, s, t, fx, td, nxu, nzu, $d, dx_t, dz_t)
                        })
                        .and_then(move |(s, t, fx, td)| {
                            ks.update_a(gt, s, t, fx, td, nxu, nzu, slot!($dtb<$ty>))
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
                                slot!($hv3<$ty>),
                            )
                        })
                        .and_then(move |(s, t, fx, td, _h1, _h2, _h3, _h4, _h5)| {
                            ks.tend_t(gt, s, t, fx, td, nxu, nzu, $d, dx_t, dz_t)
                        })
                        .and_then(move |(s, t, fx, td)| {
                            ks.update_b(gt, s, t, fx, td, nxu, nzu, slot!($dtc<$ty>))
                        })
                    };
                }

                // Seed dispatch takes the owned buffers; every later half is
                // uniform. Step 1 = x-first (x half via seed+rest, then z),
                // step 2 = z-first (z, then x).
                let g0 = ks
                    .halo_s(
                        gh_x,
                        s,
                        t,
                        fx,
                        td,
                        hd.clone(),
                        hdt.clone(),
                        nxu,
                        nzu,
                        0u32,
                        inj,
                        dz_t,
                    )
                    .and_then(move |(s, t, fx, td, _hd, _hdt)| {
                        ks.flux_s(
                            gf_x,
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
                            0u32,
                            slot!(H1A1<$ty>),
                        )
                    })
                    .and_then(move |(s, t, fx, td, _h1, _h2, _h3, _h4, _h5)| {
                        ks.tend_s(gt, s, t, fx, td, nxu, nzu, 0u32, dx_t, dz_t)
                    })
                    .and_then(move |(s, t, fx, td)| {
                        ks.update_a(gt, s, t, fx, td, nxu, nzu, slot!(Dt1A<$ty>))
                    })
                    .and_then(move |(s, t, fx, td)| {
                        ks.halo_t(
                            gh_x,
                            s,
                            t,
                            fx,
                            td,
                            hd.clone(),
                            hdt.clone(),
                            nxu,
                            nzu,
                            0u32,
                            inj,
                            dz_t,
                        )
                    })
                    .and_then(move |(s, t, fx, td, _hd, _hdt)| {
                        ks.flux_t(
                            gf_x,
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
                            0u32,
                            slot!(H1A2<$ty>),
                        )
                    })
                    .and_then(move |(s, t, fx, td, _h1, _h2, _h3, _h4, _h5)| {
                        ks.tend_t(gt, s, t, fx, td, nxu, nzu, 0u32, dx_t, dz_t)
                    })
                    .and_then(move |(s, t, fx, td)| {
                        ks.update_a(gt, s, t, fx, td, nxu, nzu, slot!(Dt1B<$ty>))
                    })
                    .and_then(move |(s, t, fx, td)| {
                        ks.halo_t(
                            gh_x,
                            s,
                            t,
                            fx,
                            td,
                            hd.clone(),
                            hdt.clone(),
                            nxu,
                            nzu,
                            0u32,
                            inj,
                            dz_t,
                        )
                    })
                    .and_then(move |(s, t, fx, td, _hd, _hdt)| {
                        ks.flux_t(
                            gf_x,
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
                            0u32,
                            slot!(H1A3<$ty>),
                        )
                    })
                    .and_then(move |(s, t, fx, td, _h1, _h2, _h3, _h4, _h5)| {
                        ks.tend_t(gt, s, t, fx, td, nxu, nzu, 0u32, dx_t, dz_t)
                    })
                    .and_then(move |(s, t, fx, td)| {
                        ks.update_b(gt, s, t, fx, td, nxu, nzu, slot!(Dt1C<$ty>))
                    });
                let g1 = half!(g0, gh_z, gf_z, 1u32, H1B1, H1B2, H1B3, Dt1A, Dt1B, Dt1C);
                let g2 = half!(g1, gh_z, gf_z, 1u32, H2A1, H2A2, H2A3, Dt2A, Dt2B, Dt2C);
                let g = half!(g2, gh_x, gf_x, 0u32, H2B1, H2B2, H2B3, Dt2A, Dt2B, Dt2C);

                // hv(d, stage, dt) with the dt = 0 no-op convention
                let hv = |d: f64, q: f64, dt: f64| -> $ty {
                    if dt == 0.0 {
                        0.0 as $ty
                    } else {
                        (-HV_BETA * d / (16.0 * (dt / q))) as $ty
                    }
                };

                let dt0 = dx.min(dzf) / MAX_SPEED * CFL;
                let bind_all = |dt1: f64, dt2: f64| {
                    (
                        Dt1A((dt1 / 3.0) as $ty),
                        Dt1B((dt1 / 2.0) as $ty),
                        Dt1C(dt1 as $ty),
                        Dt2A((dt2 / 3.0) as $ty),
                        Dt2B((dt2 / 2.0) as $ty),
                        Dt2C(dt2 as $ty),
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
                let b = bind_all(dt0, dt0);
                let g = g
                    .bind(b.0)
                    .bind(b.1)
                    .bind(b.2)
                    .bind(b.3)
                    .bind(b.4)
                    .bind(b.5)
                    .bind(b.6)
                    .bind(b.7)
                    .bind(b.8)
                    .bind(b.9)
                    .bind(b.10)
                    .bind(b.11)
                    .bind(b.12)
                    .bind(b.13)
                    .bind(b.14)
                    .bind(b.15)
                    .bind(b.16)
                    .bind(b.17);

                let mut etime = 0.0f64;
                let mut dt = dt0;
                let mut bound = (dt0, dt0);
                let mut final_state = vec![<$ty>::default(); state_len];
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
                        let b = bind_all(dt1, dt2);
                        g.mutate_call((b.0, b.1, b.2, b.3))?;
                        g.mutate_call((b.4, b.5, b.6, b.7))?;
                        g.mutate_call((b.8, b.9, b.10, b.11))?;
                        g.mutate_call((b.12, b.13, b.14, b.15))?;
                        g.mutate_call((b.16, b.17))?;
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
                        let view = (*cs).map().wait()?;
                        final_state.copy_from_slice(&view);
                    }
                }
                eprintln!("steps: {nstep}");
                let final_f64: Vec<f64> = final_state.iter().map(|&x| x as f64).collect();
                Ok((state0, final_f64))
            }
        }
    };
}

make_runner!(runner_f64, super::gpu::f64, f64);
make_runner!(runner_f32, super::gpu::f32, f32);

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
    let (state0, state1) = if use_f64 {
        runner_f64::run(&ctx, &cfg, &init)?
    } else {
        runner_f32::run(&ctx, &cfg, &init)?
    };
    eprintln!("device loop: {:.3}s", t0.elapsed().as_secs_f64());

    let (mass0, te0) = reductions(&state0, &init, cfg.nx, cfg.nz);
    let (mass, te) = reductions(&state1, &init, cfg.nx, cfg.nz);
    println!("d_mass: {:.15e}", (mass - mass0) / mass0);
    println!("d_te:   {:.15e}", (te - te0) / te0);

    if let Some(path) = &cfg.dump {
        let fields = output_fields(&state1, &init, cfg.nx, cfg.nz);
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
        let (s0, s1) = runner_f64::run(&ctx, &cfg, &init).expect("run f64");
        let (m0, e0) = reductions(&s0, &init, cfg.nx, cfg.nz);
        let (m1, e1) = reductions(&s1, &init, cfg.nx, cfg.nz);
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
        let (s0, s1) = runner_f32::run(&ctx, &cfg, &init).expect("run f32");
        let (m0, e0) = reductions(&s0, &init, cfg.nx, cfg.nz);
        let (m1, e1) = reductions(&s1, &init, cfg.nx, cfg.nz);
        let dm = ((m1 - m0) / m0).abs();
        let de = ((e1 - e0) / e0).abs();
        assert!(dm < 1e-8, "f32 mass drift too large: {dm:e}");
        assert!(de < 5e-4, "f32 energy drift too large: {de:e}");
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
        let (_, s64) = runner_f64::run(&ctx, &cfg, &init).expect("run f64");
        let (_, s32) = runner_f32::run(&ctx, &cfg, &init).expect("run f32");
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
