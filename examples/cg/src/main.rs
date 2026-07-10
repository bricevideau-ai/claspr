//! claspr Tier 2: **matrix-free Conjugate Gradient** — ONE self-closing graph
//! replayed as the CG loop body, with the α/β reduction **parametrized behind a
//! [`ReduceStrategy`]** so CG becomes a comparison sample: the SAME math, the SAME
//! self-closing loop, computing α/β two interchangeable ways that a
//! command-buffer partitioner sees as two DIFFERENT shapes.
//!
//! Gray-Scott is a *pure device* solver with a FIXED step count, so the whole
//! simulation trivially records into one command buffer. CG is more interesting:
//! its OUTER control flow (the convergence test) is inherently a HOST decision.
//! The naive shape chops each iteration into three device regions separated by two
//! host cuts — because α and β (`rsold/pAp`, `rsnew/rsold`) were computed on the
//! HOST from downloaded reductions, forcing three `sync`s per iteration and a
//! fistful of `into_inner`s to shuttle buffers across the seams.
//!
//! ## Two strategies, one algorithm — the whole point
//!
//! Both strategies run the identical device reduction (`dot_partial` /
//! `norm2_partial` grid-stride the vectors into a `G`-lane `partials` array) and
//! differ ONLY in WHERE α/β are *finished* from those partials — which is exactly
//! what changes the command-buffer shape:
//!
//! - **[`AllDevice`]** (the default / primary, the guide's readable worked
//!   example): two tiny single-work-item **finish KERNELS** (`finish_alpha` /
//!   `finish_beta`) consume the `partials` + residual scalars and write α, −α, β,
//!   and the running `rsold`/`rsnew` straight into len-1 device buffers (the
//!   `&f32` / `&mut f32` scalar-by-reference kernel-arg path). The step scalars
//!   never touch the host, so the ENTIRE iteration graph `g` is ONE fully
//!   command-buffer-able region — a single maximal recordable graph whose only
//!   host seam is the len-1 residual read at the loop boundary.
//!
//! - **[`AndThenHost`](HostSeam)**: the device reduction still runs, but α/β are
//!   finished in an [`and_then_host`](claspr::eager::DeviceOpExt::and_then_host)
//!   closure that READS the mapped `partials` and WRITES the α/−α/rsold/rsnew
//!   [`DeviceScalar`]s through their `&mut f32` host views. This puts *interpreted
//!   host cuts* INSIDE the iteration graph — the MIXED shape a partitioner must
//!   discover (record the device spans between the cuts, interpret the cuts). It
//!   is a GENUINE self-closing graph replayed in the SAME `loop { g.sync() }` —
//!   NOT a rebuild-per-iteration workaround: the α seam bundles the `partials`
//!   read-view together with the α/−α scalar write-views into ONE host seam whose
//!   every branch re-homes across replays (the mid-graph multi-home seam), so the
//!   scalars written on-host feed the next device kernels over stable handles.
//!
//! Both converge IDENTICALLY (same iters, same `‖r‖`, same solution) — the host
//! seam changes WHERE the reduction finishes, not the math. That identity is the
//! proof they are the same algorithm, and it is asserted as a test
//! (`both_strategies_converge_identically`).
//!
//! ## The self-closing loop (both strategies)
//!
//! ```text
//!   loop {
//!       let rsnew = g.sync(ctx)?;          // run one whole CG iteration
//!       done = *map(rsnew) < TOL*TOL;      // the ONLY loop-boundary host read
//!       drop(rsnew);                       // re-arm g over the SAME handles
//!       if done { break }
//!   }
//! ```
//!
//! The loop body has **zero** `into_inner`: every CG buffer + device scalar is a
//! CONCRETE cell of `g` (or, in the host-seam strategy, a `lift`ed cell that
//! re-homes just the same), lent on each `sync` and rehomed to the same cell on
//! the run's `Checkout` drop (the home invariant), so the next `sync` reuses
//! identical `cl_mem` handles with no rebinding.
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

use claspr::eager::{DeviceOp, DeviceOpExt, bundle2, bundle4, lift};
use claspr::{Context, DeviceScalar, DeviceSlice, Pipe};

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

/// The CG **state scalars**, each a device-resident [`DeviceScalar<f32>`] carrying
/// the recurrence. Bundled so the two strategies receive the SAME handles and the
/// solver reads them back uniformly. `alpha`/`nalpha` feed the step-5 axpys;
/// `rsold` carries into the β finish; `beta` feeds the next step-1 `xpby`; `rsnew`
/// (= r·r) is the value mapped at the loop boundary for the convergence test.
struct Scalars {
    rsold: DeviceScalar<f32>,
    alpha: DeviceScalar<f32>,
    nalpha: DeviceScalar<f32>,
    beta: DeviceScalar<f32>,
    rsnew: DeviceScalar<f32>,
}

/// How α and β are **finished** from the `G`-lane reduction `partials` — the ONE
/// axis on which the two CG variants differ. Everything else (the device
/// reduction, the axpys, the self-closing loop) is shared. A strategy supplies two
/// subgraph builders; the solver ([`solve_with`]) is generic over it.
///
/// Each builder is handed the buffers/scalars it needs (as concrete cells or
/// threaded pipes) and returns a subgraph whose [`DeviceOp::Output`] /
/// [`DeviceOp::Handle`] shape is FIXED across strategies, so the solver body that
/// composes them is identical. See [`AllDevice`] (finish kernels — one recordable
/// region) and [`HostSeam`] (an `and_then_host` cut — the mixed shape).
///
/// The two shapes a command-buffer partitioner sees:
/// - [`AllDevice`]: no host seam anywhere in `g` ⇒ ONE maximal recordable region.
/// - [`HostSeam`]: two interpreted host cuts inside `g` ⇒ device spans to record
///   between the cuts, host closures to interpret at the cuts.
trait ReduceStrategy {
    /// Whether this strategy introduces an `and_then_host` cut into the iteration
    /// graph. `false` for [`AllDevice`] (one recordable region — the property the
    /// self-closing all-device path must NOT regress); `true` for [`HostSeam`]. The
    /// solver `debug_assert`s the built graph's
    /// [`contains_host_seam`](claspr::eager::DeviceOp::contains_host_seam) matches
    /// this — so an accidental seam in the all-device path (or a lost seam in the
    /// host path) is caught on every run.
    const HAS_HOST_SEAM: bool;

    /// Build the **α finish**: read `partials` (= p·Ap) and `rsold`, produce α and
    /// −α as `Pipe<DeviceScalar<f32>>`s the step-5 axpys read. All three scalars
    /// arrive CONCRETE (first use). `rsold` is threaded onward (β needs the *old*
    /// value), and `partials` is threaded onward too (step 5's `norm2` reuses the
    /// SAME buffer for r·r), so the handle is
    /// `(Pipe<alpha>, Pipe<nalpha>, Pipe<rsold>, Pipe<partials>)`.
    #[allow(clippy::type_complexity)]
    fn compute_alpha(
        &self,
        ks: &gpu::Kernels,
        partials: Pipe<DeviceSlice<f32>>,
        rsold: DeviceScalar<f32>,
        alpha: DeviceScalar<f32>,
        nalpha: DeviceScalar<f32>,
    ) -> impl DeviceOp<
        Output = (
            DeviceScalar<f32>,
            DeviceScalar<f32>,
            DeviceScalar<f32>,
            DeviceSlice<f32>,
        ),
        Handle = (
            Pipe<DeviceScalar<f32>>,
            Pipe<DeviceScalar<f32>>,
            Pipe<DeviceScalar<f32>>,
            Pipe<DeviceSlice<f32>>,
        ),
        Checkouts = (
            claspr::Checkout<DeviceScalar<f32>>,
            claspr::Checkout<DeviceScalar<f32>>,
            claspr::Checkout<DeviceScalar<f32>>,
            claspr::Checkout<DeviceSlice<f32>>,
        ),
    >;

    /// Build the **β finish**: consume `partials` (now = r·r) and `rsold`, publish
    /// `rsnew = Σpartials`, form `beta = rsnew / rsold`, advance `rsold = rsnew`.
    /// `rsold` and `beta` arrive as THREADED pipes (`rsold` from the α finish,
    /// `beta` from step 1's `xpby`) — the physical cells whose homes the finish must
    /// re-home; `rsnew` arrives CONCRETE (first use). Only `rsnew` is threaded to
    /// the terminal (the loop-boundary read); `beta`/`rsold` cycle and re-home via
    /// reclaim, so the handle is just `Pipe<rsnew>`.
    fn compute_beta(
        &self,
        ks: &gpu::Kernels,
        partials: Pipe<DeviceSlice<f32>>,
        rsold: Pipe<DeviceScalar<f32>>,
        beta: Pipe<DeviceScalar<f32>>,
        rsnew: DeviceScalar<f32>,
    ) -> impl DeviceOp<
        Output = DeviceScalar<f32>,
        Handle = Pipe<DeviceScalar<f32>>,
        Checkouts = claspr::Checkout<DeviceScalar<f32>>,
    >;
}

/// **Strategy 1 — all on-device (default / primary).** α and β are finished by two
/// single-work-item KERNELS (`finish_alpha` / `finish_beta`) that sum the `G`
/// partials and write the step scalars straight into their len-1 device buffers.
/// No host seam anywhere ⇒ the whole iteration graph is ONE recordable region.
/// This is the readable worked example; keep it simple.
struct AllDevice;

impl ReduceStrategy for AllDevice {
    const HAS_HOST_SEAM: bool = false;

    fn compute_alpha(
        &self,
        ks: &gpu::Kernels,
        partials: Pipe<DeviceSlice<f32>>,
        rsold: DeviceScalar<f32>,
        alpha: DeviceScalar<f32>,
        nalpha: DeviceScalar<f32>,
    ) -> impl DeviceOp<
        Output = (
            DeviceScalar<f32>,
            DeviceScalar<f32>,
            DeviceScalar<f32>,
            DeviceSlice<f32>,
        ),
        Handle = (
            Pipe<DeviceScalar<f32>>,
            Pipe<DeviceScalar<f32>>,
            Pipe<DeviceScalar<f32>>,
            Pipe<DeviceSlice<f32>>,
        ),
        Checkouts = (
            claspr::Checkout<DeviceScalar<f32>>,
            claspr::Checkout<DeviceScalar<f32>>,
            claspr::Checkout<DeviceScalar<f32>>,
            claspr::Checkout<DeviceSlice<f32>>,
        ),
    > {
        // finish_alpha(partials, rsold, alpha, nalpha) → (partials, rsold, alpha,
        // nalpha); re-expose (alpha, nalpha, rsold, partials) as the fixed 4-tuple
        // handle. alpha = rsold / Σpartials; partials threads to step 5's norm2.
        ks.finish_alpha([1], partials, rsold, alpha, nalpha)
            .and_then(|(partials, rsold, alpha, nalpha)| bundle4(alpha, nalpha, rsold, partials))
    }

    fn compute_beta(
        &self,
        ks: &gpu::Kernels,
        partials: Pipe<DeviceSlice<f32>>,
        rsold: Pipe<DeviceScalar<f32>>,
        beta: Pipe<DeviceScalar<f32>>,
        rsnew: DeviceScalar<f32>,
    ) -> impl DeviceOp<
        Output = DeviceScalar<f32>,
        Handle = Pipe<DeviceScalar<f32>>,
        Checkouts = claspr::Checkout<DeviceScalar<f32>>,
    > {
        // finish_beta(partials, rsold, beta, rsnew): rsnew = Σpartials; beta =
        // rsnew/rsold; rsold = rsnew. Thread only rsnew onward (beta/rsold reclaim).
        // The `and_then` SELECTS rsnew out of the 4-tuple handle; a bare `Pipe` is
        // itself a `DeviceOp`, so no `forward` wrap is needed.
        ks.finish_beta([1], partials, rsold, beta, rsnew)
            .and_then(|(_partials, _rsold, _beta, rsnew)| rsnew)
    }
}

/// **Strategy 2 — host-seam finish.** The device reduction still fills `partials`,
/// but α/β are finished in an [`and_then_host`](claspr::eager::DeviceOpExt::and_then_host)
/// closure that reads the mapped `partials` and writes the step scalars through
/// their `&mut f32` views. This is the MIXED shape: interpreted host cuts inside
/// the self-closing iteration graph. The seam **bundles** the `partials` read-view
/// with the scalar write-views into ONE mid-graph multi-home seam — every branch
/// re-homes across replays, so the scalars written on-host feed the next device
/// kernels over stable handles (NO per-scalar seams, NO `into_inner`).
struct HostSeam;

impl ReduceStrategy for HostSeam {
    const HAS_HOST_SEAM: bool = true;

    fn compute_alpha(
        &self,
        _ks: &gpu::Kernels,
        partials: Pipe<DeviceSlice<f32>>,
        rsold: DeviceScalar<f32>,
        alpha: DeviceScalar<f32>,
        nalpha: DeviceScalar<f32>,
    ) -> impl DeviceOp<
        Output = (
            DeviceScalar<f32>,
            DeviceScalar<f32>,
            DeviceScalar<f32>,
            DeviceSlice<f32>,
        ),
        Handle = (
            Pipe<DeviceScalar<f32>>,
            Pipe<DeviceScalar<f32>>,
            Pipe<DeviceScalar<f32>>,
            Pipe<DeviceSlice<f32>>,
        ),
        Checkouts = (
            claspr::Checkout<DeviceScalar<f32>>,
            claspr::Checkout<DeviceScalar<f32>>,
            claspr::Checkout<DeviceScalar<f32>>,
            claspr::Checkout<DeviceSlice<f32>>,
        ),
    > {
        // Bundle the partials read-view (a threaded pipe) + the three scalars
        // (lifted concrete cells — `lift` re-homes across replays) into ONE seam.
        // The closure sums partials and writes alpha = rsold/Σ, nalpha = −alpha,
        // reading rsold and writing alpha/nalpha through their `&mut f32` views;
        // rsold is threaded onward unchanged (β needs the OLD value). Every branch
        // re-homes (the mid-graph multi-home seam), so alpha/nalpha feed step-5's
        // axpys over stable handles and the whole graph replays.
        bundle4(partials, lift(alpha), lift(nalpha), lift(rsold))
            .and_then_host(
                |(part, alpha, nalpha, rsold): (&mut [f32], &mut f32, &mut f32, &mut f32)| {
                    let sum: f32 = part.iter().sum();
                    let a = *rsold / sum;
                    *alpha = a;
                    *nalpha = -a;
                    Ok(())
                },
            )
            // Re-expose (alpha, nalpha, rsold, partials) as the fixed 4-tuple handle
            // the solver expects — partials threads to step 5's norm2 (same buffer).
            .and_then(|(part, alpha, nalpha, rsold)| bundle4(alpha, nalpha, rsold, part))
    }

    fn compute_beta(
        &self,
        _ks: &gpu::Kernels,
        partials: Pipe<DeviceSlice<f32>>,
        rsold: Pipe<DeviceScalar<f32>>,
        beta: Pipe<DeviceScalar<f32>>,
        rsnew: DeviceScalar<f32>,
    ) -> impl DeviceOp<
        Output = DeviceScalar<f32>,
        Handle = Pipe<DeviceScalar<f32>>,
        Checkouts = claspr::Checkout<DeviceScalar<f32>>,
    > {
        // Same bundle-into-seam shape for β: sum partials (= r·r) → rsnew, form
        // beta = rsnew/rsold (OLD rsold), then advance rsold = rsnew. Writes rsnew,
        // beta, rsold through their `&mut f32` views; reads partials. `rsold`/`beta`
        // arrive as threaded pipes — a bare `Pipe` is itself a `DeviceOp`, so it
        // drops straight into the bundle (its home rides through and re-arms). Thread
        // only rsnew onward (beta/rsold reclaim + re-home for the next iteration).
        bundle4(partials, lift(rsnew), beta, rsold)
            .and_then_host(
                |(part, rsnew, beta, rsold): (&mut [f32], &mut f32, &mut f32, &mut f32)| {
                    let sum: f32 = part.iter().sum();
                    *rsnew = sum;
                    *beta = sum / *rsold;
                    *rsold = sum;
                    Ok(())
                },
            )
            .and_then(|(_part, rsnew, _beta, _rsold)| rsnew)
    }
}

/// Solve `A x = b` by Conjugate Gradient with the given reduction `strategy`.
/// Returns `(x, iterations, final ‖r‖)`.
///
/// The entire iteration is ONE device graph `g`, built once and `sync`'d in a
/// loop. Every CG buffer + device scalar is a concrete (or `lift`ed) cell of `g`,
/// lent per `sync` and rehomed on the run's `Checkout` drop — so the loop reuses
/// identical handles with **zero** `into_inner` and no rebinding, for BOTH
/// strategies. Only the α/β finish subgraphs (built by `strategy`) differ.
fn solve_with<S: ReduceStrategy>(
    ctx: &Context,
    b_host: &[f32],
    strategy: &S,
) -> claspr::Result<(Vec<f32>, usize, f32)> {
    let kernels = gpu::kernels(ctx)?;
    let ks = &kernels;

    // Device-resident CG state, each a CONCRETE cell of the graph below. The
    // vectors are updated in place across iterations.
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
    // Device scalars — first-class `DeviceScalar<f32>`. `beta = 0` makes iteration
    // 1's `xpby(p, r, 0)` a no-op (`p` already = b), so no peeled first iteration
    // is needed. `rsold = b·b` seeds the α of the first step.
    let rsold_val: f32 = b_host.iter().map(|v| v * v).sum();
    let s = Scalars {
        rsold: DeviceScalar::new(ctx, rsold_val)?,
        alpha: DeviceScalar::new(ctx, 0.0)?,
        nalpha: DeviceScalar::new(ctx, 0.0)?,
        beta: DeviceScalar::new(ctx, 0.0)?,
        rsnew: DeviceScalar::new(ctx, 0.0)?,
    };
    let Scalars {
        rsold,
        alpha,
        nalpha,
        beta,
        rsnew,
    } = s;

    // ── Build the whole CG iteration ONCE as a single self-closing graph. ────
    // Steps 1-3 + step 5 are shared device kernels; steps 4 and 6 (the α/β finish)
    // are the `strategy`'s subgraphs — a finish KERNEL (all-device) or an
    // `and_then_host` cut (host-seam), both with the SAME output/handle shape, so
    // this composition is identical either way. `ks` is captured by every closure;
    // the ops it builds own a fresh `cl_kernel` + a cloned context, so `g` does NOT
    // borrow `ks`. Only `x` (solution) and `rsnew` (residual) reach the terminal;
    // every other buffer re-homes via `reclaim_undelivered`.
    let g = ks
        // 1. p = r + beta*p   (beta = 0 first iter ⇒ p = r = b, a no-op)
        .xpby_dev([N], p, r, beta)
        .and_then(move |(p, r, beta)| {
            // 2. ap = A*p
            ks.spmv([N], p, ap).and_then(move |(p, ap)| {
                // 3. partials = p·Ap
                ks.dot_partial([G], p, ap, partials)
                    .and_then(move |(p, ap, partials)| {
                        // 4. STRATEGY: alpha = rsold / Σpartials ; nalpha = -alpha.
                        //    Threads (alpha, nalpha, rsold, partials) onward.
                        strategy
                            .compute_alpha(ks, partials, rsold, alpha, nalpha)
                            .and_then(move |(alpha, nalpha, rsold, partials)| {
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
                                        // 6. STRATEGY: rsnew = Σpartials ; beta =
                                        //    rsnew/rsold ; rsold = rsnew. Thread `x`
                                        //    (solution) and `rsnew` (residual) to the
                                        //    terminal; other outputs reclaim.
                                        bundle2(
                                            x,
                                            strategy.compute_beta(ks, partials, rsold, beta, rsnew),
                                        )
                                    },
                                )
                            })
                    })
            })
        });

    // STRUCTURAL GUARD: the all-device graph must contain NO host seam (one
    // recordable region — the property that keeps the all-device path self-closing
    // / fully command-buffer-able); the host-seam graph must contain one (the mixed
    // shape). `contains_host_seam` walks the whole built graph, so an accidental
    // seam in the all-device path — or a lost seam in the host path — trips this on
    // every solve (a hard assert, so it fires in release too).
    assert_eq!(
        g.contains_host_seam(),
        S::HAS_HOST_SEAM,
        "strategy host-seam shape regressed: contains_host_seam() != HAS_HOST_SEAM"
    );

    let mut iters = 0usize;
    loop {
        // Run ONE full CG iteration. The sole per-iteration host action at the loop
        // boundary is mapping the len-1 `rsnew` (= r·r) for the convergence test.
        // ZERO `into_inner`: both Checkouts drop at the end of the body, rehoming
        // `x` and `rsnew` (and `reclaim_undelivered` rehomes every mid-graph buffer
        // AND, for the host-seam strategy, every lifted scalar), so the next `sync`
        // reuses identical handles with no rebinding.
        let (x_co, rsnew_co) = g.sync(ctx)?;
        // Read back the len-1 `rsnew` (= r·r) scalar. A borrowing `read_value`
        // (Deref through the Checkout) — no `into_inner`, so the Checkout still
        // rehomes on drop below.
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

/// Build the test problem: a known solution `x_true` (a discrete parabola, ~zero
/// at the boundaries, small range so f32 stays comfortable) and its right-hand
/// side `b = A x_true`, so a solve can be checked against `x_true`.
fn test_problem() -> (Vec<f32>, Vec<f32>) {
    let x_true: Vec<f32> = (0..N)
        .map(|i| ((i + 1) as f32) * ((N - i) as f32) / (N as f32 * N as f32))
        .collect();
    let b = host_poisson(&x_true);
    (x_true, b)
}

/// Solve with one strategy, print + assert convergence, and return
/// `(iters, ‖r‖, max|x − x_true|)` so callers can compare strategies.
fn run_strategy<S: ReduceStrategy>(
    ctx: &Context,
    label: &str,
    strategy: &S,
) -> claspr::Result<(usize, f32, f32)> {
    let (x_true, b) = test_problem();
    let (x, iters, rnorm) = solve_with(ctx, &b, strategy)?;
    let err = x
        .iter()
        .zip(&x_true)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "cg[{label}]: N={N}, DIAG={DIAG} → converged in {iters} iters, |r|={rnorm:e}, \
         max|x - x_true|={err:e}"
    );
    assert!(
        rnorm < TOL,
        "CG[{label}] did not reach tolerance (|r|={rnorm:e})"
    );
    assert!(err < 1e-3, "CG[{label}] solution error too large: {err:e}");
    Ok((iters, rnorm, err))
}

fn run(ctx: Context) -> claspr::Result<()> {
    // Run BOTH strategies — the comparison sample. Same math, two command-buffer
    // shapes: `AllDevice` (one recordable region) and `HostSeam` (interpreted cuts).
    run_strategy(&ctx, "all-device", &AllDevice)?;
    run_strategy(&ctx, "host-seam", &HostSeam)?;
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

    /// The primary all-device strategy converges to the known solution.
    #[test]
    fn cg_converges_to_known_solution() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        run_strategy(&ctx, "all-device", &AllDevice).expect("CG all-device solve");
    }

    /// The host-seam strategy converges to the known solution too.
    #[test]
    fn cg_host_seam_converges() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        run_strategy(&ctx, "host-seam", &HostSeam).expect("CG host-seam solve");
    }

    /// **The proof they are the same algorithm.** Both strategies compute the SAME
    /// math (α/β from the same reduction), differing only in WHERE the reduction
    /// finishes — so they must converge IDENTICALLY: same iteration count, same
    /// final ‖r‖, same solution error. Anything else is a strategy bug.
    #[test]
    fn both_strategies_converge_identically() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        let (i_dev, r_dev, e_dev) =
            run_strategy(&ctx, "all-device", &AllDevice).expect("all-device");
        let (i_host, r_host, e_host) =
            run_strategy(&ctx, "host-seam", &HostSeam).expect("host-seam");
        // Identical iteration count — same recurrence, same convergence test.
        assert_eq!(
            i_dev, i_host,
            "iteration counts differ: all-device={i_dev}, host-seam={i_host}"
        );
        // Same finish arithmetic (device single-work-item sum vs host `iter().sum()`
        // are both left-to-right f32 sums over the SAME G partials), so ‖r‖ and the
        // solution error match to a hair — a tight tolerance catches any real drift.
        assert!(
            (r_dev - r_host).abs() <= 1e-6 * r_dev.max(r_host).max(1e-6),
            "final ‖r‖ differ: all-device={r_dev:e}, host-seam={r_host:e}"
        );
        assert!(
            (e_dev - e_host).abs() <= 1e-6 * e_dev.max(e_host).max(1e-6),
            "solution error differs: all-device={e_dev:e}, host-seam={e_host:e}"
        );
    }

    /// **The all-device path stays self-closing / fully recordable.** Its built
    /// graph must contain NO `and_then_host` cut — one maximal command-buffer-able
    /// region. The host-seam path, by contrast, MUST contain the interpreted cuts.
    /// `solve_with` `assert_eq!`s `contains_host_seam() == HAS_HOST_SEAM` on the
    /// whole built graph every solve; this test makes that structural contract
    /// explicit and also inspects the node `description()` so a regression (an
    /// accidental host seam sneaking into the all-device graph, or the host strategy
    /// losing its seam) is caught with a readable diff. Runs the solve too (via the
    /// internal assert) — SKIP without a device.
    #[test]
    fn all_device_is_self_closing_host_seam_is_mixed() {
        // The strategy const is the source of truth the solver asserts against.
        const { assert!(!AllDevice::HAS_HOST_SEAM, "all-device must be seam-free") };
        const { assert!(HostSeam::HAS_HOST_SEAM, "host-seam must declare its seam") };

        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        // A solve runs the in-`solve_with` structural assert (contains_host_seam ==
        // HAS_HOST_SEAM) over the fully-built graph for each strategy — the actual
        // guard. If the all-device graph grew a host seam, this solve would panic.
        run_strategy(&ctx, "all-device", &AllDevice).expect("all-device solve");
        run_strategy(&ctx, "host-seam", &HostSeam).expect("host-seam solve");
    }
}
