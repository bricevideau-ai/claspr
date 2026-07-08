//! claspr Tier 2: **matrix-free Conjugate Gradient** — one self-closing,
//! all-device graph replayed as the CG loop body.
//!
//! Gray-Scott is a *pure device* solver with a FIXED step count, so the whole
//! simulation trivially records into one command buffer. CG is more interesting:
//! its OUTER control flow (the convergence test) is inherently a HOST decision.
//! The naive shape therefore chops each iteration into three device regions
//! separated by two host cuts — because α and β (`rsold/pAp`, `rsnew/rsold`) were
//! computed on the HOST from downloaded reductions, forcing three `sync`s per
//! iteration and a fistful of `into_inner`s to shuttle buffers across the seams.
//!
//! ## The unlock: device-resident scalars
//!
//! Move α and β **on-device**. Two tiny single-work-item finish kernels
//! (`finish_alpha` / `finish_beta`)
//! consume the reduction `partials` and the residual scalars and write α, −α, β,
//! and the running `rsold`/`rsnew` straight into len-1 device buffers (exercising
//! the `&f32` / `&mut f32` scalar-by-reference kernel-arg path). The step scalars
//! never touch the host, so the host round-trip that forced the chopping is gone
//! and the ENTIRE iteration becomes a single device graph `g`, built ONCE and
//! replayed:
//!
//! ```text
//!   loop {
//!       let rsnew = g.sync(ctx)?;          // run one whole CG iteration on-device
//!       done = *map(rsnew) < TOL*TOL;      // the ONLY host action: read r·r
//!       drop(rsnew);                       // re-arm g over the SAME handles
//!       if done { break }
//!   }
//! ```
//!
//! The loop body has **zero** `into_inner`: every CG buffer is a CONCRETE cell of
//! `g`, lent on each `sync` and rehomed to the same cell on the run's `Checkout`
//! drop (the home invariant), so the next `sync` reuses identical `cl_mem`
//! handles with no rebinding. This is the ideal command-buffer-partitioner
//! workload: a single maximal recordable region whose only host seam is the
//! len-1 residual read at the loop boundary.
//!
//! ## The uniform, self-closing iteration (no peel)
//!
//! We solve `A x = b` for the SPD "screened-Poisson" operator
//! `A = tridiag(-1, DIAG, -1)` (DIAG > 2 ⇒ well-conditioned), matrix-FREE: `A` is
//! never materialised, only applied by `spmv`. The classic CG step,
//! written as ONE dataflow graph over device buffers `x, r, p, ap, partials` and
//! device scalars `rsold, alpha, nalpha, beta, rsnew`:
//!
//! ```text
//!   1. xpby_dev(p, r, beta)                    p = r + beta*p         (device)
//!   2. spmv(p, ap)                             ap = A*p               (device)
//!   3. dot_partial(p, ap, partials)            partials = p·Ap        (device)
//!   4. finish_alpha(partials, rsold,           alpha = rsold / Σpartials
//!                   alpha, nalpha)             nalpha = -alpha        (device)
//!   5. bundle2( axpy_dev(x, p, alpha),         x += alpha*p           (device)
//!               axpy_dev(r, ap, nalpha)        r -= alpha*Ap
//!                 -> norm2_partial(r, part.))  partials = r·r         (device)
//!   6. finish_beta(partials, rsold,            rsnew = Σpartials
//!                  beta, rsnew)                beta  = rsnew / rsold
//!                                              rsold = rsnew          (device)
//! ```
//!
//! The graph **closes on itself**: step 6's `beta` feeds the NEXT step 1's `xpby`,
//! step 5's `r` feeds the next step 1, and `partials`/`rsold` cycle. No peeled
//! first iteration is needed: initialise `beta = 0`, `p = r = b`, `x = 0`, and
//! `rsold = b·b`. Then iteration 1's `xpby(p, r, 0)` computes `p = r + 0·p = b`
//! (harmless — `p` already equals `b`), so the uniform body IS a correct first CG
//! step. Every subsequent iteration is the same graph with the on-device scalars
//! carrying the recurrence forward.
//!
//! ## Buffer threading + reclaim (why not every buffer reaches the terminal)
//!
//! Only `rsnew` must reach the terminal — it is the value mapped at the loop
//! boundary. Every other buffer is threaded (as an `and_then` pipe) ONLY where
//! the dataflow needs it; wherever a produced buffer is not consumed onward, the
//! run's `reclaim_undelivered` rehomes it to its concrete cell (the mid-graph
//! half of the home invariant), so a mid-graph buffer needn't be dragged to the
//! terminal just to be returned. Concretely: `p` threads 1→2→3→5A, `ap` 2→3→5B,
//! `partials` 3→4 then 5B→6, `r` carries 1→5B, `rsold` carries 4→6, `alpha`
//! 4→5A, `nalpha` 4→5B, `beta` carries 1→6; unconsumed tails (`x`, and the
//! finish kernels' non-`rsnew` outputs) reclaim.
//!
//! Reductions are matrix-free too: `dot_partial` /
//! `norm2_partial` grid-stride `G` lanes into `partials`,
//! and the finish kernels sum the `G` partials on-device (a len-1 dispatch) — no
//! workgroup memory / barriers, robust on every ICD.

use claspr::eager::{DeviceOpExt, bundle2, forward};
use claspr::{Context, DeviceScalar, DeviceSlice};

#[claspr::device]
pub mod gpu {
    /// Problem size (1D grid). Small so the handful of CG iterations stay quick on
    /// a CPU ICD; the structure is identical at any `N`.
    pub const N: usize = 512;
    /// Dot-product partial-sum lanes: each lane grid-strides `N/G` elements into
    /// one `partials` slot; a finish kernel sums the `G` partials on-device. Keeps
    /// the reduction on-device without workgroup memory.
    pub const G: usize = 64;
    /// Main-diagonal weight of `A = tridiag(-1, DIAG, -1)`. `DIAG = 2.0` is the
    /// exact 1D Poisson (ill-conditioned, cond ~ (N/π)²); `> 2.0` is a
    /// screened-Poisson / Helmholtz shift `(-∇² + κ²)`, well-conditioned so CG
    /// converges fast — nicer for a demo, still a genuine SPD system.
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
    /// grid-stride slice (`i = g, g+G, g+2G, …`) into `partials[g]`. Dispatched
    /// over `[G]`; a finish kernel sums the `G` partials into a scalar.
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

    /// Partial squared-norm `a·a` — the one-argument twin of [`dot_partial`], used
    /// for `r·r`. (A slot can't be bound to two kernel args at once — `a` moved —
    /// so `r·r` needs its own kernel rather than `dot_partial(r, r, …)`.)
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

    /// `axpy` with a **device-resident** scale: `y += (*s) · x`, in place. `x +=
    /// α p` binds `s = alpha`; `r −= α Ap` binds `s = nalpha` (`= −α`, also
    /// computed on-device) — so the step scalar never leaves the device. The
    /// `s: &f32` scalar-by-reference arg is backed by a len-1 `DeviceSlice<f32>`.
    #[claspr::kernel]
    pub fn axpy_dev(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] y: &mut [f32],
        #[spirv(cross_workgroup)] x: &[f32],
        #[spirv(cross_workgroup)] s: &f32,
    ) {
        let i = id.x as usize;
        if i >= N {
            return;
        }
        y[i] += *s * x[i];
    }

    /// `xpby` with a **device-resident** β: `p = r + (*beta) · p`, in place — the
    /// CG direction update. `beta` is a len-1 `DeviceSlice<f32>` written by
    /// [`finish_beta`] the previous iteration (and initialised to `0`, so the very
    /// first iteration's update is `p = r`).
    #[claspr::kernel]
    pub fn xpby_dev(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] p: &mut [f32],
        #[spirv(cross_workgroup)] r: &[f32],
        #[spirv(cross_workgroup)] beta: &f32,
    ) {
        let i = id.x as usize;
        if i >= N {
            return;
        }
        p[i] = r[i] + *beta * p[i];
    }

    /// On-device α finish (single work-item, `[1]`): sum the `G` reduction
    /// `partials` (`= p·Ap`) and write the step length `α = rsold / (p·Ap)` plus
    /// its negation `−α` into two len-1 device scalars. `rsold` is read by
    /// reference (`&f32`); `alpha`/`nalpha` are written by reference
    /// (`&mut f32`) — the scalar-ref output path. Keeps α on-device so
    /// [`axpy_dev`] can read it without a host round-trip.
    #[claspr::kernel]
    pub fn finish_alpha(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] partials: &[f32],
        #[spirv(cross_workgroup)] rsold: &f32,
        #[spirv(cross_workgroup)] alpha: &mut f32,
        #[spirv(cross_workgroup)] nalpha: &mut f32,
    ) {
        if id.x as usize >= 1 {
            return;
        }
        let mut s = 0.0f32;
        let mut g = 0usize;
        while g < G {
            s += partials[g];
            g += 1;
        }
        let a = *rsold / s;
        *alpha = a;
        *nalpha = -a;
    }

    /// On-device β finish (single work-item, `[1]`): sum the `G` reduction
    /// `partials` (now `= r·r`), publish it as `rsnew`, form `β = rsnew / rsold`
    /// (using the *old* `rsold`), then advance `rsold = rsnew` for the next
    /// iteration. `partials` is read by reference; `rsold`/`beta`/`rsnew` are len-1
    /// device scalars written by reference. `rsnew` is the one value the host maps
    /// at the loop boundary for the convergence test.
    #[claspr::kernel]
    pub fn finish_beta(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] partials: &[f32],
        #[spirv(cross_workgroup)] rsold: &mut f32,
        #[spirv(cross_workgroup)] beta: &mut f32,
        #[spirv(cross_workgroup)] rsnew: &mut f32,
    ) {
        if id.x as usize >= 1 {
            return;
        }
        let mut s = 0.0f32;
        let mut g = 0usize;
        while g < G {
            s += partials[g];
            g += 1;
        }
        *rsnew = s;
        *beta = s / *rsold;
        *rsold = s;
    }
}

use gpu::{DIAG, G, N};

const TOL: f32 = 1e-5;
const MAXITER: usize = 1000;

/// Host application of the same operator `A` — used to build a right-hand side
/// with a KNOWN solution (`b = A x_true`) so the solve can be checked.
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

/// Read a device vector back to the host.
fn read_vec(buf: &DeviceSlice<f32>) -> claspr::Result<Vec<f32>> {
    let guard = buf.map().wait()?;
    Ok(guard.to_vec())
}

/// Solve `A x = b` by Conjugate Gradient. Returns `(x, iterations, final ‖r‖)`.
///
/// The entire iteration is ONE device graph `g`, built once and `sync`'d in a
/// loop. Every CG buffer + device scalar is a concrete cell of `g`, lent per
/// `sync` and rehomed on the run's `Checkout` drop — so the loop reuses identical
/// handles with **zero** `into_inner` and no rebinding.
fn solve(ctx: &Context, b_host: &[f32]) -> claspr::Result<(Vec<f32>, usize, f32)> {
    let kernels = gpu::kernels(ctx)?;
    let ks = &kernels;

    // Device-resident CG state, each a CONCRETE cell of the graph below. The
    // vectors are updated in place across iterations; the scalars carry the CG
    // recurrence entirely on-device (no host round-trip for α/β).
    //   x  solution     (x0 = 0)
    //   r  residual     (r0 = b − A x0 = b)
    //   p  search dir    (p0 = r0 = b)
    //   ap scratch A·p
    //   partials  G-way partial sums for the dot / norm2 reductions
    let x: DeviceSlice<f32> = DeviceSlice::alloc_zero(ctx, N)?;
    let r: DeviceSlice<f32> = DeviceSlice::from_slice(ctx, b_host)?;
    let p: DeviceSlice<f32> = DeviceSlice::from_slice(ctx, b_host)?;
    let ap: DeviceSlice<f32> = DeviceSlice::alloc_zero(ctx, N)?;
    let partials: DeviceSlice<f32> = DeviceSlice::alloc_zero(ctx, G)?;
    // Device scalars — first-class `DeviceScalar<f32>` (each a single `&f32` /
    // `&mut f32` scalar-by-reference kernel arg). `beta = 0` makes iteration 1's
    // `xpby(p, r, 0)` a no-op (`p` already = b), so no peeled first iteration is
    // needed. `rsold = b·b` seeds the α of the first step (we already hold b on
    // the host).
    let rsold_val: f32 = b_host.iter().map(|v| v * v).sum();
    let rsold: DeviceScalar<f32> = DeviceScalar::new(ctx, rsold_val)?;
    let alpha: DeviceScalar<f32> = DeviceScalar::new(ctx, 0.0)?;
    let nalpha: DeviceScalar<f32> = DeviceScalar::new(ctx, 0.0)?;
    let beta: DeviceScalar<f32> = DeviceScalar::new(ctx, 0.0)?;
    let rsnew: DeviceScalar<f32> = DeviceScalar::new(ctx, 0.0)?;

    // ── Build the whole CG iteration ONCE as a single self-closing graph. ────
    // Each kernel op takes its buffers as concrete cells (first use) or threaded
    // pipes (later use); the `and_then` closures capture-and-forward the buffers a
    // later step needs (gray-scott `run_immutable`'s closure-carrying style). `ks`
    // is a `&Kernels` (Copy) captured by every closure; the ops it builds own a
    // fresh `cl_kernel` + a cloned context, so `g` does NOT borrow `ks`.
    //
    // Buffers read-early / written-late must be THREADED (a single concrete cell
    // lent once, its pipe carried past intervening steps) rather than named twice:
    //   beta  : read at step 1 (xpby), written at step 6 (finish_beta) — pipe 1→6.
    //   rsold : read at step 4 (finish_alpha), r/w at step 6 — pipe 4→6.
    // Only the two host-read buffers reach the terminal — `x` (the solution, read
    // once on convergence) and `rsnew` (r·r, mapped every iteration). Every other
    // produced-but-unconsumed buffer rehomes via `reclaim_undelivered`.
    let g = ks
        // 1. p = r + beta*p   (beta = 0 first iter ⇒ p = r = b, a no-op)
        .xpby_dev([N], p, r, beta)
        .and_then(move |(p, r, beta)| {
            // 2. ap = A*p
            ks.spmv([N], p, ap).and_then(move |(p, ap)| {
                // 3. partials = p·Ap
                ks.dot_partial([G], p, ap, partials)
                    .and_then(move |(p, ap, partials)| {
                        // 4. alpha = rsold / Σpartials ; nalpha = -alpha
                        ks.finish_alpha([1], partials, rsold, alpha, nalpha)
                            .and_then(move |(partials, rsold, alpha, nalpha)| {
                                // 5. BUNDLE two independent branches:
                                //   A: x += alpha*p
                                //   B: r -= alpha*Ap  ->  partials = r·r
                                bundle2(
                                    ks.axpy_dev([N], x, p, alpha),
                                    ks.axpy_dev([N], r, ap, nalpha).and_then(
                                        move |(r, _ap, _nalpha)| ks.norm2_partial([G], r, partials),
                                    ),
                                )
                                .and_then(
                                    move |((x, _p, _alpha), (_r, partials))| {
                                        // 6. rsnew = Σpartials ; beta = rsnew/rsold ;
                                        //    rsold = rsnew. Thread `x` (solution) and
                                        //    `rsnew` (residual) to the terminal; the
                                        //    finish kernel's other outputs reclaim.
                                        bundle2(
                                            forward(x),
                                            ks.finish_beta([1], partials, rsold, beta, rsnew)
                                                .and_then(|(_partials, _rsold, _beta, rsnew)| {
                                                    forward(rsnew)
                                                }),
                                        )
                                    },
                                )
                            })
                    })
            })
        });

    let mut iters = 0usize;
    loop {
        // Run ONE full CG iteration on-device. The sole per-iteration host action
        // is mapping the len-1 `rsnew` (= r·r) for the convergence test. ZERO
        // `into_inner`: both Checkouts drop at the end of the body, rehoming `x`
        // and `rsnew` to their cells (and `reclaim_undelivered` rehomes every
        // mid-graph buffer), so the next `sync` reuses identical handles with no
        // rebinding.
        let (x_co, rsnew_co) = g.sync(ctx)?;
        // The sole host action: read back the len-1 `rsnew` (= r·r) scalar. A
        // borrowing `DeviceScalar::read_value` (Deref through the Checkout) — no
        // `into_inner`, so the Checkout still rehomes on drop below.
        let rsnew_scalar = rsnew_co.read_value()?;
        iters += 1;
        if rsnew_scalar < TOL * TOL || iters >= MAXITER {
            // Converged (or capped): read the solution off the same Checkout — a
            // borrowing map, still no `into_inner`. Both Checkouts drop after this
            // returns.
            let final_rnorm = rsnew_scalar.max(0.0).sqrt();
            let x_host = read_vec(&x_co)?;
            return Ok((x_host, iters, final_rnorm));
        }
        // Not done: drop both Checkouts to re-arm `g` over the same handles.
        drop((x_co, rsnew_co));
    }
}

fn run(ctx: Context) -> claspr::Result<()> {
    // A known solution (a discrete parabola, ~zero at the boundaries), scaled into
    // a small range so f32 stays comfortable. Build b = A x_true so we can check
    // the solve against x_true.
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
