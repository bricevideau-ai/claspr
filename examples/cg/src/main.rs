//! claspr Tier 2: **matrix-free Conjugate Gradient** — a genuinely MIXED graph.
//!
//! Where `gray-scott` is a *pure device* solver (fixed step count, no host in the
//! loop → the whole graph trivially records into one command buffer), CG is the
//! opposite and more interesting shape: dense device inner work whose OUTER control
//! flow is inherently a HOST decision. That makes it the honest test workload for the
//! eventual command-buffer partitioner — the CB-able device regions must be
//! *discovered* between interpreted host cuts, not assumed to be the whole graph.
//!
//! We solve `A x = b` for the SPD "screened-Poisson" operator
//! `A = tridiag(-1, DIAG, -1)` (DIAG > 2 ⇒ well-conditioned, CG converges in tens of
//! iterations), matrix-FREE: `A` is never materialised, only applied by a stencil
//! kernel. The classic CG iteration:
//!
//! ```text
//!   r = b - A x0   (x0 = 0 ⇒ r = b);   p = r;   rsold = r·r
//!   repeat:
//!     Ap   = A p                         ── device (spmv)
//!     pAp  = p·Ap                        ── device reduce → host scalar
//!     α    = rsold / pAp                 ── HOST decision (feeds the next kernels)
//!     x   += α p                         ── device (axpy)
//!     r   -= α Ap                        ── device (axpy)
//!     rsnew = r·r                        ── device reduce → host scalar
//!     if √rsnew < tol: STOP              ── HOST branch == the loop bound
//!     β    = rsnew / rsold               ── HOST decision
//!     p    = r + β p                     ── device (xpby)
//!     rsold = rsnew
//! ```
//!
//! ## Why this is a mixed graph (the point)
//!
//! Each iteration is THREE device regions separated by TWO host cuts:
//!
//! ```text
//!   [ REGION A: spmv → dot_partial ]        ← one CB-able chain (2 kernels)
//!        │ download partials
//!        ▼
//!   ( HOST: pAp = Σpartials ; α = rsold/pAp )   ← interpreted cut (α feeds region B)
//!        │
//!        ▼
//!   [ REGION B: axpy(x)  ‖  { axpy(r) → norm2_partial } ]   ← CB-able BUNDLE (branch)
//!        │ download partials
//!        ▼
//!   ( HOST: rsnew = Σpartials ; STOP? ; β )     ← interpreted cut + LOOP BOUND
//!        │
//!        ▼
//!   [ REGION C: xpby ]                       ← CB-able chain (1 kernel)
//! ```
//!
//! The host steps are NOT bolted on — α, β and the convergence test are the
//! algorithm's own control flow, computed from device reductions and fed back as
//! kernel args / the loop condition. A command-buffer backend would record regions
//! A/B/C once (their buffer handles are stable across iterations — bound once, updated
//! in place) and replay them, while the host cuts stay interpreted. That "find the
//! maximal recordable region, stop at the host seam, resume" structure is exactly
//! what a pure device solver never forces.
//!
//! Reductions are done matrix-free too: a `dot_partial` kernel where each of `G`
//! lanes grid-strides a slice into `partials[g]`, and the host finishes the `G`-way
//! sum (a tiny download). No workgroup memory / barriers — robust on every ICD.

use claspr::eager::{DeviceOpExt, bundle2, bundle3};
use claspr::{Context, DeviceSlice};

#[claspr::device]
pub mod gpu {
    /// Problem size (1D grid). Small so the ~tens of CG iterations stay quick on a
    /// CPU ICD; the structure is identical at any `N`.
    pub const N: usize = 512;
    /// Dot-product partial-sum lanes: each lane grid-strides `N/G` elements into one
    /// `partials` slot; the host sums the `G` partials. Keeps the reduction
    /// on-device without workgroup memory.
    pub const G: usize = 64;
    /// Main-diagonal weight of `A = tridiag(-1, DIAG, -1)`. `DIAG = 2.0` is the exact
    /// 1D Poisson (ill-conditioned, cond ~ (N/π)²); `> 2.0` is a screened-Poisson /
    /// Helmholtz shift `(-∇² + κ²)`, well-conditioned so CG converges fast — nicer for
    /// a demo, still a genuine SPD system.
    pub const DIAG: f32 = 2.5;

    /// Matrix-free SpMV: `ap = A p` for `A = tridiag(-1, DIAG, -1)` with Dirichlet
    /// (zero) boundaries. `ap[i] = DIAG·p[i] − p[i−1] − p[i+1]`.
    #[claspr::kernel]
    pub fn spmv(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] p: &[f32],
        #[spirv(cross_workgroup)] ap: &mut [f32],
    ) {
        let i = id.x as usize;
        if i >= N {
            return;
        }
        let left = if i == 0 { 0.0 } else { p[i - 1] };
        let right = if i + 1 >= N { 0.0 } else { p[i + 1] };
        ap[i] = DIAG * p[i] - left - right;
    }

    /// Partial dot product `a·b`: lane `g` accumulates `a[i]*b[i]` over its
    /// grid-stride slice (`i = g, g+G, g+2G, …`) into `partials[g]`. Dispatched over
    /// `[G]`; the host sums the `G` partials into the scalar.
    #[claspr::kernel]
    pub fn dot_partial(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] a: &[f32],
        #[spirv(cross_workgroup)] b: &[f32],
        #[spirv(cross_workgroup)] partials: &mut [f32],
    ) {
        let g = id.x as usize;
        if g >= G {
            return;
        }
        let mut acc = 0.0f32;
        let mut i = g;
        while i < N {
            acc += a[i] * b[i];
            i += G;
        }
        partials[g] = acc;
    }

    /// Partial squared-norm `a·a` — the one-argument twin of [`dot_partial`], used for
    /// `r·r`. (A slot can't be bound to two kernel args at once — `a` moved — so `r·r`
    /// needs its own kernel rather than `dot_partial(r, r, …)`.)
    #[claspr::kernel]
    pub fn norm2_partial(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] a: &[f32],
        #[spirv(cross_workgroup)] partials: &mut [f32],
    ) {
        let g = id.x as usize;
        if g >= G {
            return;
        }
        let mut acc = 0.0f32;
        let mut i = g;
        while i < N {
            acc += a[i] * a[i];
            i += G;
        }
        partials[g] = acc;
    }

    /// `axpy`: `y += s · x`, in place. `x += α p` uses `s = α`; `r −= α Ap` uses the
    /// SAME kernel with `s = −α` — so the step scalar (a host-computed value) is what
    /// flows back into the device work.
    #[claspr::kernel]
    pub fn axpy(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] y: &mut [f32],
        #[spirv(cross_workgroup)] x: &[f32],
        s: f32,
    ) {
        let i = id.x as usize;
        if i >= N {
            return;
        }
        y[i] += s * x[i];
    }

    /// `xpby`: `p = r + β · p`, in place — the CG direction update. `β` is the
    /// host-computed `rsnew / rsold`.
    #[claspr::kernel]
    pub fn xpby(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] p: &mut [f32],
        #[spirv(cross_workgroup)] r: &[f32],
        beta: f32,
    ) {
        let i = id.x as usize;
        if i >= N {
            return;
        }
        p[i] = r[i] + beta * p[i];
    }
}

use gpu::{DIAG, G, N};

const TOL: f32 = 1e-5;
const MAXITER: usize = 1000;

/// Host application of the same operator `A` — used to build a right-hand side with a
/// KNOWN solution (`b = A x_true`) so the solve can be checked against `x_true`.
fn host_poisson(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let left = if i == 0 { 0.0 } else { x[i - 1] };
            let right = if i + 1 >= n { 0.0 } else { x[i + 1] };
            DIAG * x[i] - left - right
        })
        .collect()
}

/// Host finish of a device partial reduction: map the `G`-element `partials` buffer
/// and sum it into the scalar. This tiny download IS the host cut — the point where a
/// CB-able device region ends and interpreted control flow resumes.
fn sum_dev(buf: &DeviceSlice<f32>) -> claspr::Result<f32> {
    let guard = buf.map().wait()?;
    Ok(guard.iter().copied().sum())
}

/// Read a device vector back to the host.
fn read_vec(buf: &DeviceSlice<f32>) -> claspr::Result<Vec<f32>> {
    let guard = buf.map().wait()?;
    Ok(guard.to_vec())
}

/// Solve `A x = b` by Conjugate Gradient. Returns `(x, iterations, final ‖r‖)`.
fn solve(ctx: &Context, b_host: &[f32]) -> claspr::Result<(Vec<f32>, usize, f32)> {
    let kernels = gpu::kernels(ctx)?;

    // Device-resident CG vectors, bound once and updated in place across iterations
    // (so a CB backend would see STABLE buffer handles → record-once/replay-many):
    //   x  solution     (x0 = 0)
    //   r  residual     (r0 = b − A x0 = b)
    //   p  search dir    (p0 = r0 = b)
    //   ap scratch A·p   (overwritten each iteration by spmv)
    //   partials  G-way partial sums for the dot / norm2 reductions
    let mut x: DeviceSlice<f32> = DeviceSlice::alloc_zero(ctx, N)?;
    let mut r: DeviceSlice<f32> = DeviceSlice::from_slice(ctx, b_host)?;
    let mut p: DeviceSlice<f32> = DeviceSlice::from_slice(ctx, b_host)?;
    let mut ap: DeviceSlice<f32> = DeviceSlice::alloc_zero(ctx, N)?;
    let mut partials: DeviceSlice<f32> = DeviceSlice::alloc_zero(ctx, G)?;

    // rsold = r0·r0 = b·b (host — we already hold b on the host).
    let mut rsold: f32 = b_host.iter().map(|v| v * v).sum();

    let mut iters = 0usize;
    let final_rnorm;
    loop {
        // ── REGION A (device chain): ap = A p ; partials = p·ap ──────────────
        // Two kernels, one dataflow chain (spmv's `ap` feeds dot_partial) → a single
        // CB-able region. `spmv` returns (p, ap); dot_partial returns (p, ap, partials)
        // — PER-ELEMENT Checkouts, so each buffer can be lent onward individually.
        let (p_co, ap_co, part_co) = kernels
            .spmv([N], p, ap)
            .and_then(|(p, ap)| kernels.dot_partial([G], p, ap, partials))
            .sync(ctx)?;
        // Read the reduction on the host (map borrows the Checkout — does NOT consume,
        // so p_co/ap_co/part_co stay lendable). This tiny download is the host cut.
        let pap = sum_dev(&part_co)?;

        // ── HOST CUT 1: the step length α (feeds region B's kernels) ─────────
        let alpha = rsold / pap;

        // ── REGION B (device BUNDLE — two recordable branches) ──────────────
        //   branch 1:  x += α p       (x moved in;  p LENT from region A)
        //   branch 2:  r −= α ap  →  partials = r·r   (r moved in;  ap, partials LENT)
        // Region A's Checkouts flow DIRECTLY in as lent kernel inputs — no `into_inner`,
        // no rebind. A stays busy while B runs; each lent buffer returns to A's graph on
        // its Checkout's drop (return-on-drop), so the recover below is a single sever.
        // The branches are independent (disjoint buffers) → a `bundle2` a CB backend
        // records as two subtrees under one enqueue. `axpy(y, x, s)` returns (y, x), so
        // branch 1 hands back (x, p) and branch 2 (r, ap, partials).
        let (xp_co, rap_co) = bundle2(
            kernels.axpy([N], x, p_co, alpha),
            kernels.axpy([N], r, ap_co, -alpha).and_then(|(r, ap)| {
                kernels
                    .norm2_partial([G], r, part_co)
                    .and_then(move |(r, partials)| bundle3(r, ap, partials))
            }),
        )
        .sync(ctx)?;
        // A bundle branch's `sync` yields ONE Checkout over its WHOLE output tuple
        // (can't extract one element), and the CG buffers form a CYCLE across
        // iterations (p: C→A) that outlives these throwaway graphs — so HERE we must
        // recover owned buffers with `into_inner`. That severs region A's lent cells
        // (p/ap/partials) and hands back region B's owned outputs (x/r) in one go.
        let (r_inner, ap_inner, part_inner) = rap_co.into_inner();
        let rsnew = sum_dev(&part_inner)?;
        let (x_inner, p_inner) = xp_co.into_inner();
        x = x_inner;
        p = p_inner;
        r = r_inner;
        ap = ap_inner;
        partials = part_inner;

        // ── HOST CUT 2: convergence test (== the loop bound) + β ─────────────
        iters += 1;
        let rnorm = rsnew.sqrt();
        if rnorm < TOL || iters >= MAXITER {
            final_rnorm = rnorm;
            break;
        }
        let beta = rsnew / rsold;

        // ── REGION C (device): p = r + β p ──────────────────────────────────
        // Single kernel → per-element Checkouts; recover owned p/r for the next
        // iteration (the cycle boundary again forces a sever).
        let (p_co, r_co) = kernels.xpby([N], p, r, beta).sync(ctx)?;
        p = p_co.into_inner();
        r = r_co.into_inner();

        rsold = rsnew;
    }

    let x_host = read_vec(&x)?;
    Ok((x_host, iters, final_rnorm))
}

fn run(ctx: Context) -> claspr::Result<()> {
    // A known solution (a discrete parabola, ~zero at the boundaries), scaled into a
    // small range so f32 stays comfortable. Build b = A x_true so we can check the
    // solve against x_true.
    let x_true: Vec<f32> = (0..N)
        .map(|i| ((i + 1) as f32) * ((N - i) as f32) / (N as f32 * N as f32))
        .collect();
    let b = host_poisson(&x_true);

    let (x, iters, rnorm) = solve(&ctx, &b)?;

    let err = x
        .iter()
        .zip(&x_true)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "cg: N={N}, DIAG={DIAG} → converged in {iters} iters, |r|={rnorm:e}, \
         max|x - x_true|={err:e}"
    );
    assert!(rnorm < TOL, "CG did not reach tolerance (|r|={rnorm:e})");
    assert!(err < 1e-3, "solution error too large: {err:e}");
    Ok(())
}

fn main() -> claspr::Result<()> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            return Ok(());
        }
    };
    run(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_converges_to_known_solution() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        run(ctx).expect("CG solve");
    }
}
