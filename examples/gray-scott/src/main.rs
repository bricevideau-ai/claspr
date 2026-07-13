//! # Gray-Scott reaction-diffusion — *a claspr graph is a reusable meta-kernel*
//!
//! This is the flagship demonstration of the claspr Tier 2 idea: **a device
//! graph is not a one-shot dispatch, it is a meta-kernel you define ONCE and
//! then replay, ping-pong, and reparameterize without ever rebuilding it.**
//! Gray-Scott is the perfect vehicle — it is a tight iterated stencil that
//! wants to run for thousands of identical steps, double-buffering two fields,
//! reading a handful of shared scalar parameters, and (the punchline) producing
//! a visibly *different* pattern the moment you retune those parameters mid-run.
//!
//! ## TWO execution strategies for the SAME computation (the thesis)
//!
//! The headline claim — *graphs as meta-kernels* — is proven here by running the
//! **identical** Gray-Scott arithmetic through **two different graph shapes**,
//! and asserting they agree to the bit. Both drive the same three device kernels
//! (`laplacian` ×2 + `combine`); they differ only in how double-buffering is
//! wired into the graph:
//!
//! - [`run_swap`] — the **mutable-swap** strategy. Build a graph for ONE step,
//!   then re-bind its four field slots *crossed* between steps (`mutate_call`)
//!   so the buffers ping-pong. The graph's buffer *handles change every step*;
//!   the topology is one step and the rotation lives in per-step rebinding.
//!   Replayed once per step (`STEPS` syncs).
//!
//! - [`run_immutable`] — the **immutable / unroll-by-2** strategy. Bake the
//!   buffer rotation into the graph TOPOLOGY: compose the per-step subgraph WITH
//!   ITSELF (step-k then step-k+1). Double-buffering rotates roles every step
//!   (step k: read A write B; step k+1: read B write A), so after *two* steps
//!   you are back to "current = A" — the two-step graph has **fixed buffer
//!   roles/handles** and needs **NO per-step rebinding at all**. Its outer input
//!   slots are bound ONCE (`call`) and it is replayed `STEPS/2` times via plain
//!   `sync()`; the A↔B rotation is entirely internal, threaded as pipes. This is
//!   the shape that would bake into a single immutable command buffer once
//!   step-c (command-buffer caching) lands: no mutation between replays.
//!
//! Both start from the same initial condition and the same `(F, k)`, run the
//! same number of steps, and their final `V` fields must be **bit-identical** —
//! same arithmetic, same order, just a different graph geometry. That equality
//! (asserted in the `#[test]`) is the proof that *unroll-by-period replay* and
//! *mutable-swap replay* are two spellings of one meta-kernel. `main` runs both
//! and prints whether they agree; each writes its own final frame.
//!
//! ## The computation
//!
//! Two scalar fields `U` and `V` live on a 2D periodic grid and co-evolve by
//!
//! ```text
//!   U' = U + dt * ( Du*Lap(U) - U*V*V + F*(1 - U) )
//!   V' = V + dt * ( Dv*Lap(V) + U*V*V - (F + k)*V )
//! ```
//!
//! where `Lap` is the 5-point Laplacian with wrap-around (periodic) boundaries.
//! Started from `U=1, V=0` with a seeded central square, the fields
//! self-organize into spots / stripes / mazes depending on `(F, k)`.
//!
//! ## A REAL multi-kernel graph (the meta-kernel is three dispatches)
//!
//! Rather than fuse the whole step into one dispatch, the update is
//! operator-FACTORED into two reusable device kernels:
//!
//! - `gpu::laplacian` — reads ONE field, writes its 5-point periodic Laplacian
//!   into a scratch buffer. It is dispatched **TWICE per step** — once for `U`
//!   (→ `lap_u`) and once for `V` (→ `lap_v`). **Same kernel, two sites.**
//! - `gpu::combine` — reads `(u_in, v_in, lap_u, lap_v)` and writes
//!   `(u_out, v_out)` via the reaction terms, applying the diffusion + feed/kill.
//!
//! So the per-step graph `g` is a genuine **three-dispatch DAG** — `lap_u`,
//! `lap_v`, then `combine` — and two of its nodes are the *same* kernel. This is
//! numerically identical to a fused single-dispatch step: it is just factoring
//! the Laplacian into a temp pass (NOT ODE operator-splitting — `combine` still
//! advances BOTH fields together from the same-step inputs).
//!
//! ## How this sample exercises the slot machinery (the whole point)
//!
//! The host builds this three-dispatch update graph `g` exactly once, then drives
//! the entire simulation by re-`sync()`-ing that one graph. Concretely:
//!
//! 1. **Reuse** — `g` is built ONCE (three chained dispatches), then
//!    `g.sync(&ctx)` runs in a `for` loop for thousands of steps; the graph is
//!    never rebuilt. Defining the meta-kernel once and replaying it cheaply is
//!    precisely what step-c command-buffer caching will accelerate: today each
//!    `sync` re-validates + re-enqueues, tomorrow the cached command buffer just
//!    re-submits.
//!
//! 2. **Shared launch slot fanned across all three sites** — `slot!(Grid)`
//!    (`Tag::Value = LaunchSpec`) is placed in the grid position of ALL THREE
//!    dispatches. ONE `bind(Grid(LaunchSpec::from([W, H])))` fans out and fills
//!    every site — the canonical shared-launch-slot demo (`slot_generalization`
//!    test 7), here across a real three-node graph. The dispatch is genuinely
//!    **2D** (`[W, H]`): the kernels index by `id.x` (column) / `id.y` (row).
//!
//! 3. **Pipe threading between dispatches** — the field BUFFERS are move-only
//!    (`DeviceSlice`, not `Clone`), so a single buffer slot can't fan out to two
//!    sites. Instead each `laplacian` dispatch is multi-output
//!    `(field_in, lap_out)`, and `combine` reads BOTH of its outputs **threaded
//!    as pipes** via `and_then`: the field itself (passed through unchanged) and
//!    its Laplacian. So `slot!(UIn)`/`slot!(VIn)` sit at ONE site each (their lap
//!    dispatch), and the Laplacian scratch buffers need no slot tags at all —
//!    both edges are carried by the pipe threading, not by extra slots.
//!
//! 4. **Double-buffering** — `U` and `V` each ping-pong between two device
//!    buffers. After a step we re-bind the four output Checkouts *crossed* in ONE
//!    [`mutate_call`](claspr::DeviceOpExt::mutate_call), binding each Checkout
//!    DIRECTLY into a slot: binding a Checkout SEVERS its source home (`Lent →
//!    Severed`) and the target slot ADOPTS the buffer — so the swap is the crossing
//!    in one line, no manual `into_inner()`. This step's `*Out` becomes next step's
//!    `*In`, and the now-stale `*In` becomes next step's scratch `*Out`. The
//!    crossed re-bind MUST go through the `mutate` verb — a plain `bind` on a
//!    severed slot is `Error::SlotSevered` (the canonical ping-pong rule from
//!    `tests/tier2/tests/double_buffering.rs`, here doubled across the two field
//!    pairs and issued as one 4-tuple `mutate_call`).
//!
//! 5. **Scalar slots** — `F` and `k` are `slot!` SCALAR slots (non-resource,
//!    `Copy`, value-equality, never handed back as Checkouts). Bound once up
//!    front, they are READ (not consumed) on every replay, so they persist
//!    across all the steps for free. `Du`/`Dv` and `dt` and the grid size are
//!    compile-time constants of the meta-kernel (see below).
//!
//! 6. **Mutate-to-reconfigure (the meta-kernel proof)** — after the first phase
//!    we `mutate_bind(F(..))` / `mutate_bind(K(..))` to a *different* `(F, k)`
//!    regime and keep stepping the SAME graph. The very same meta-kernel now
//!    grows a different pattern. We emit a PPM frame before and after so the
//!    reconfiguration is visible: **the graph is a kernel you reparameterize
//!    without rebuilding it.**
//!
//! ## The arity budget (why Du/Dv are consts)
//!
//! The OpenCL `KernelArgs` tuple impls top out at arity 8. `combine` spends ALL
//! EIGHT on what genuinely varies per step or per regime: the six buffers
//! (`u_in, v_in, lap_u, lap_v, u_out, v_out`) plus the two reaction scalars
//! `F`, `k`. There is no room left to also pass `Du`/`Dv`, so the diffusion
//! constants — which are fixed for the whole run anyway — are baked in as
//! `gpu::DU`/`gpu::DV` compile-time consts, alongside `gpu::DT` and the grid
//! size `gpu::W`/`gpu::H`. `laplacian` is comfortably under the ceiling (its
//! kernel args are just `field_in` + `lap_out`; the grid rides the launch slot).
//!
//! ## Output
//!
//! The **swap** variant writes three PPM frames — `gray-scott-early.ppm`,
//! `gray-scott-late.ppm`, `gray-scott-reconfigured.ppm` — showing pattern
//! emergence then the mid-run retune. The **immutable** variant writes
//! `gray-scott-immutable.ppm` (its final `V` at the phase-1 regime). `V` is
//! colorized on the HOST after read-back (a blue→teal→white ramp); no image
//! kernel is involved.
//!
//! Run with
//! `OCL_ICD_VENDORS=$HOME/local/pocl/etc/OpenCL/vendors cargo run -p gray-scott-example`.
//! It runs BOTH variants and prints whether the swap and immutable final `V`
//! fields agree. The `#[test]` at the bottom is the smoke test (short sim,
//! asserts the field stays bounded, NaN-free, and actually evolved away from the
//! initial condition) PLUS the swap-vs-immutable equality proof.

use claspr::eager::DeviceOpExt;
use claspr::{Context, DeviceSlice, LaunchSpec};
use claspr::{slot, slots};

#[claspr::device]
mod gpu {
    /// Grid dimensions, time step, and diffusion rates are compile-time
    /// constants of the meta-kernel. The OpenCL `KernelArgs` tuple impls top out
    /// at arity 8, and `combine` spends all eight on what actually VARIES at
    /// runtime — the six field/scratch buffers plus the two scalar slots `F`,
    /// `k`. So the rest is baked in: `dt = 1.0` is the standard Gray-Scott step,
    /// `DU`/`DV` are the (fixed) diffusion rates, and `W`/`H` the grid extent.
    /// Host code mirrors `W`/`H` with its own constants.
    pub const W: u32 = 256;
    pub const H: u32 = 256;
    pub const DT: f32 = 1.0;
    pub const DU: f32 = 0.16;
    pub const DV: f32 = 0.08;

    /// Periodic (wrap-around) neighbor index along one axis: `coord ± 1`
    /// modulo `extent`, computed without signed arithmetic so it lowers
    /// cleanly to the kernel target. Used by the Laplacian for both axes.
    /// Pure — host and kernel both call it.
    pub fn wrap(coord: u32, delta_is_plus: bool, extent: u32) -> u32 {
        if delta_is_plus {
            let c = coord + 1;
            if c == extent { 0 } else { c }
        } else if coord == 0 {
            extent - 1
        } else {
            coord - 1
        }
    }

    /// 5-point Laplacian of a flat `W*H` field at cell `(x, y)` with periodic
    /// boundaries: `left + right + up + down - 4*center`. Pure helper called by
    /// the `laplacian` kernel.
    pub fn laplacian_at(field: &[f32], x: u32, y: u32, w: u32, h: u32) -> f32 {
        let idx = |xx: u32, yy: u32| (yy * w + xx) as usize;
        let center = field[idx(x, y)];
        let left = field[idx(wrap(x, false, w), y)];
        let right = field[idx(wrap(x, true, w), y)];
        let up = field[idx(x, wrap(y, false, h))];
        let down = field[idx(x, wrap(y, true, h))];
        left + right + up + down - 4.0 * center
    }

    /// SHARED Laplacian pass. Reads one field, writes its 5-point periodic
    /// Laplacian into a scratch buffer — one value per cell. Dispatched TWICE per
    /// step (once for `U` → `lap_u`, once for `V` → `lap_v`): **same kernel, two
    /// sites**, sharing one `slot!(Grid)` launch slot.
    ///
    /// Kernel args are just `(field_in, lap_out)` — well under the arity-8
    /// ceiling — because the grid extent rides the launch slot, not an arg.
    #[claspr::kernel]
    pub fn laplacian(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] field_in: &[f32],
        #[spirv(cross_workgroup)] lap_out: &mut [f32],
    ) {
        let x = id.x as u32;
        let y = id.y as u32;
        if x >= W || y >= H {
            return;
        }
        let i = (y * W + x) as usize;
        lap_out[i] = laplacian_at(field_in, x, y, W, H);
    }

    /// COMBINE pass — the reaction + diffusion update. Reads the current fields
    /// `(u_in, v_in)` plus their precomputed Laplacians `(lap_u, lap_v)` and
    /// writes the next fields `(u_out, v_out)`. This single dispatch advances
    /// BOTH fields from the same-step inputs (so it is numerically identical to a
    /// fused step). The two scalars `(feed_rate, kill_rate)` are scalar slots —
    /// bound once, re-read each replay, and `mutate_bind`-ed mid-run to
    /// reconfigure the meta-kernel.
    ///
    /// Six buffer args + two scalar args = 8 = the `KernelArgs` ceiling exactly,
    /// which is why `Du`/`Dv` are compile-time consts rather than args.
    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn combine(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] u_in: &[f32],
        #[spirv(cross_workgroup)] v_in: &[f32],
        #[spirv(cross_workgroup)] lap_u: &[f32],
        #[spirv(cross_workgroup)] lap_v: &[f32],
        #[spirv(cross_workgroup)] u_out: &mut [f32],
        #[spirv(cross_workgroup)] v_out: &mut [f32],
        feed_rate: f32,
        kill_rate: f32,
    ) {
        let x = id.x as u32;
        let y = id.y as u32;
        if x >= W || y >= H {
            return;
        }
        let i = (y * W + x) as usize;

        let u = u_in[i];
        let v = v_in[i];
        let uvv = u * v * v;

        let u_next = u + DT * (DU * lap_u[i] - uvv + feed_rate * (1.0 - u));
        let v_next = v + DT * (DV * lap_v[i] + uvv - (feed_rate + kill_rate) * v);

        u_out[i] = u_next;
        v_out[i] = v_next;
    }
}

// ── Simulation parameters ────────────────────────────────────────────────────

const W: usize = 256;
const H: usize = 256;
const N: usize = W * H;

// Phase-1 reaction regime, then the mid-run retune for phase 2. Du/Dv live as
// compile-time constants in the device module (`gpu::DU`/`gpu::DV`) because the
// `combine` kernel-arg tuple is already at the arity-8 ceiling (6 buffers + F/k).
const F1: f32 = 0.060;
const K1: f32 = 0.062;
const F2: f32 = 0.034; // "mazes/worms" regime — visibly different texture
const K2: f32 = 0.056;

const STEPS_EARLY: usize = 800; // first frame
const STEPS_PHASE1: usize = 4000; // total steps of phase 1 (regime F1/K1)
const STEPS_PHASE2: usize = 4000; // steps after the mid-run reconfigure (F2/K2)

// ── Double-buffer + scalar slot tags ─────────────────────────────────────────

slots! {
    // The launch slot: ONE tag, fanned across all THREE dispatch sites (lap_u,
    // lap_v, combine). Tag::Value = LaunchSpec (Copy, geometry-equality).
    Grid: LaunchSpec,
    // The two field pairs, each ping-ponging between two device buffers. UIn/VIn
    // are each shared across two sites (their lap dispatch + combine).
    UIn:  DeviceSlice<f32>,
    UOut: DeviceSlice<f32>,
    VIn:  DeviceSlice<f32>,
    VOut: DeviceSlice<f32>,
    // Scalar parameter slots — bound once, re-read each replay; both are
    // mutate_bind-ed mid-run to reconfigure the meta-kernel.
    F:  f32,
    K:  f32,
}

/// Allocate a device field and seed it from a host `Vec<f32>`.
fn seeded(ctx: &Context, data: Vec<f32>) -> claspr::Result<DeviceSlice<f32>> {
    let buf = DeviceSlice::<f32>::alloc_zero(ctx, data.len())?;
    buf.write(data).wait()
}

/// Host-side initial condition: `U = 1` everywhere, `V = 0`, with a small
/// central square seeded `U = 0.5, V = 0.25`. A tiny DETERMINISTIC perturbation
/// (derived from the cell index, NOT `Math::random`) breaks the symmetry so the
/// pattern is interesting rather than perfectly radial.
fn initial_fields() -> (Vec<f32>, Vec<f32>) {
    let mut u = vec![1.0f32; N];
    let mut v = vec![0.0f32; N];
    let r = 12; // half-side of the central seed square
    let (cx, cy) = (W / 2, H / 2);
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            if x >= cx - r && x < cx + r && y >= cy - r && y < cy + r {
                // Deterministic per-cell jitter in [-0.02, 0.02].
                let jitter = (((i.wrapping_mul(2654435761)) & 0xff) as f32 / 255.0 - 0.5) * 0.04;
                u[i] = 0.5 + jitter;
                v[i] = 0.25 + jitter;
            }
        }
    }
    (u, v)
}

/// Map a `V` field to RGBA8 with a simple blue→teal→white ramp. `V` for
/// Gray-Scott sits roughly in `[0, 0.4]`; we normalize against the running max.
fn colorize_v(v: &[f32]) -> Vec<u8> {
    let vmax = v.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let mut out = Vec::with_capacity(N * 4);
    for &val in v {
        let t = (val / vmax).clamp(0.0, 1.0);
        // blue (low) → teal (mid) → white (high)
        let r = (t * t * 255.0) as u8;
        let g = (t.sqrt() * 255.0) as u8;
        let b = (40.0 + 215.0 * (1.0 - t)) as u8;
        out.extend_from_slice(&[r, g, b, 255]);
    }
    out
}

/// Read back a device field into a host `Vec` **without consuming it** — we are
/// mid-ping-pong and must keep the buffer alive. The consuming `read(self)`
/// would sever it, so instead we take a borrowing read-map (`map().wait()`,
/// which derefs to `&[f32]`) and copy out. Works on a live
/// `Checkout<DeviceSlice<f32>>` via its `Deref` to `DeviceSlice`.
fn read_field(field: &DeviceSlice<f32>) -> claspr::Result<Vec<f32>> {
    let guard = field.map().wait()?;
    Ok(guard.to_vec())
}

/// **Strategy 1: mutable-swap replay.** Build a graph for ONE step, then
/// ping-pong its four field slots crossed each step via `mutate_call`. The
/// graph's buffer handles change every step; the rotation lives in per-step
/// rebinding. Returns the final `V` field (host) for the smoke test + equality
/// proof, and writes the PPM frames as it goes when `write_frames` is set.
///
/// This is also the showcase for mid-run reconfigure: with `steps_phase2 > 0` it
/// `mutate_bind`s a new `(F, k)` regime between the two phases and keeps stepping
/// the SAME graph. Pass `steps_phase2 = 0` for a plain single-regime run (what
/// the equality proof compares against [`run_immutable`]).
fn run_swap(
    ctx: &Context,
    grid_w: usize,
    grid_h: usize,
    steps_phase1: usize,
    steps_phase2: usize,
    early_at: usize,
    write_frames: bool,
) -> claspr::Result<Vec<f32>> {
    assert_eq!(
        grid_w * grid_h,
        N,
        "this helper is fixed to the W*H constants"
    );
    let ks = gpu::kernels(ctx)?;

    let (u0, v0) = initial_fields();

    // The four ping-pong buffers: U/V each get an "A" (initial / current) and a
    // "B" (scratch / next) device buffer. The B buffers start zeroed; `combine`
    // overwrites them.
    let u_a = seeded(ctx, u0)?;
    let u_b = seeded(ctx, vec![0.0f32; N])?;
    let v_a = seeded(ctx, v0)?;
    let v_b = seeded(ctx, vec![0.0f32; N])?;

    // Laplacian scratch buffers — FIXED scratch (not ping-ponged): each step's
    // laplacian dispatch overwrites them, and they feed `combine` as pipes. They
    // are allocated here (outside the graph builder closures, which can't use `?`)
    // and moved into the two laplacian dispatch sites as concrete out buffers.
    let lap_u_buf = seeded(ctx, vec![0.0f32; N])?;
    let lap_v_buf = seeded(ctx, vec![0.0f32; N])?;

    // ── Build the per-step update graph ONCE. This is the meta-kernel. ───────
    // A genuine THREE-dispatch DAG:
    //   1. laplacian(Grid, UIn → lap_u scratch)     ─┐ same kernel,
    //   2. laplacian(Grid, VIn → lap_v scratch)     ─┘ two sites
    //   3. combine(Grid, UIn, VIn, lap_u, lap_v → UOut, VOut; F, k)
    //
    // `slot!(Grid)` appears at ALL THREE sites — one `bind(Grid(..))` fans out
    // and fills every dispatch's launch extent (the shared launch slot; Grid's
    // value is a `Copy` `LaunchSpec`, so it clones into every cell). The field
    // BUFFERS are move-only (`DeviceSlice`, not `Clone`), so they cannot fan out
    // to two sites; instead each `laplacian` dispatch is multi-output
    // `(field_in, lap_out)` and we thread BOTH its outputs forward as PIPES — the
    // field buffer (`field_in`, passed through unchanged) AND its Laplacian
    // (`lap_out`) flow into `combine`. So `slot!(UIn)`/`slot!(VIn)` live at ONE
    // site each (their lap dispatch), and the Laplacian scratch buffers need no
    // slot tags at all — both are carried by `and_then`'s pipe threading.
    // Build the DAG, then fold ALL SEVEN set-once binds into ONE consuming chain.
    // `bind` is consuming + infallible now (`bind(self, arg) -> Self`), so there
    // is no `?` and no separate statement per slot — the whole graph, with every
    // set-once slot filled, is the value of `g`. Any bind error is DEFERRED and
    // surfaced at `sync` (sticky/poison: rebuild to recover).
    //
    // Scalar slots (F/K) are read — not consumed — on every replay, so they
    // persist across all steps for free. F/K are the phase-1 regime we reconfigure
    // mid-run. `Grid` is the shared launch slot: ONE bind fills its cell at ALL
    // THREE dispatch sites (genuinely 2D — `[W, H]`). UIn/VIn/UOut/VOut are the
    // step-0 buffer roles; UIn/VIn each fan out to their lap dispatch AND combine
    // from this one bind. (Du/Dv are compile-time consts in the device module.)
    let g = ks
        .laplacian(slot!(Grid), slot!(UIn), lap_u_buf)
        .and_then(move |(u_in, lap_u)| {
            // Inner dispatch: V's Laplacian. We capture the U dispatch's outputs —
            // `u_in` (the field buffer, threaded through unchanged) and `lap_u`
            // (its Laplacian) — so `combine` can read both.
            ks.laplacian(slot!(Grid), slot!(VIn), lap_v_buf)
                .and_then(move |(v_in, lap_v)| {
                    ks.combine(
                        slot!(Grid),
                        u_in,  // pipe: U field, threaded through the lap_u dispatch
                        v_in,  // pipe: V field, threaded through the lap_v dispatch
                        lap_u, // pipe: U-Laplacian scratch from dispatch 1
                        lap_v, // pipe: V-Laplacian scratch from dispatch 2
                        slot!(UOut),
                        slot!(VOut),
                        slot!(F),
                        slot!(K),
                    )
                })
        })
        .bind(F(F1))
        .bind(K(K1))
        .bind(Grid(LaunchSpec::from([grid_w, grid_h])))
        .bind(UIn(u_a))
        .bind(VIn(v_a))
        .bind(UOut(u_b))
        .bind(VOut(v_b));

    let total = steps_phase1 + steps_phase2;

    // The three frames, captured IN the step body at their due step indices —
    // straight off the freshly-written `*Out` buffer, before the swap. No extra
    // sync, no peeled iteration: every frame falls out of a normal step.
    let mut early_v: Option<Vec<f32>> = None; // after `early_at` steps
    let mut late_v: Option<Vec<f32>> = None; // end of phase 1 (F1/K1)
    let mut final_v: Vec<f32> = Vec::new(); // end of phase 2 (F2/K2)

    // ── The single step body, shared by BOTH phases (no peeling). ───────────
    // Each call: sync ONCE (advances exactly this step), captures a frame if due,
    // then swaps the buffers for the NEXT step. The swap is skipped on the very
    // last step so nothing is over-run. `combine` is the terminal node, so `sync`
    // yields ITS outputs: (u_in, v_in, lap_u, lap_v, u_out, v_out). We only
    // ping-pong the four FIELD buffers; the two lap scratch pipes are internal
    // (freshly recomputed each step) and just dropped each iteration.
    //
    // Phase 1 and phase 2 call this SAME closure — the only difference between
    // the phases is the F/k slots, reconfigured by a `mutate_bind` between the
    // two loops. No iteration is hoisted out; both phases share this body.
    let mut step = |step_idx: usize| -> claspr::Result<()> {
        let (u_in_co, v_in_co, _lap_u_co, _lap_v_co, u_out_co, v_out_co) = g.sync(ctx)?;

        // Frame capture — straight from the freshly-written `*Out` (this step's
        // result), before we swap it away. The three frames land at distinct
        // step indices, all on the SAME normal step (no special-cased iteration).
        if write_frames {
            if step_idx + 1 == early_at {
                early_v = Some(read_field(&v_out_co)?);
                println!("step {}/{total}: captured early frame", step_idx + 1);
            }
            if step_idx + 1 == steps_phase1 {
                late_v = Some(read_field(&v_out_co)?);
            }
            if step_idx + 1 == total {
                final_v = read_field(&v_out_co)?;
            }
            if (step_idx + 1).is_multiple_of(1000) {
                println!("step {}/{total}", step_idx + 1);
            }
        } else if step_idx + 1 == total {
            // The smoke test needs the final V even when not writing frames.
            final_v = read_field(&v_out_co)?;
        }

        // SWAP for the next step — unless this was the final step. Binding a
        // `Checkout` into a slot SEVERS its source home (Lent → Severed) and the
        // target slot ADOPTS the buffer, so the swap is the crossing read DIRECTLY:
        // the freshly written `*Out` Checkout becomes next step's `*In`, the stale
        // `*In` Checkout becomes next step's scratch `*Out` — no manual
        // `into_inner()`. Crossed re-bind into Severed slots ⇒ the `mutate` verb;
        // ONE `mutate_call` rebinds all four field slots together, in one line.
        if step_idx + 1 < total {
            g.mutate_call((UIn(u_out_co), VIn(v_out_co), UOut(u_in_co), VOut(v_in_co)))?;
        }
        Ok(())
    };

    // ── Phase 1: replay the meta-kernel, ping-ponging each step. ────────────
    for s in 0..steps_phase1 {
        step(s)?;
    }

    // ── MID-RUN RECONFIGURE. Same graph, different reaction regime. ─────────
    // No rebuild: just `mutate_bind` the two scalar slots that define the regime.
    // The very next `sync` runs the SAME meta-kernel at the new (F, k), and the
    // pattern morphs.
    g.mutate_bind(F(F2))?;
    g.mutate_bind(K(K2))?;
    if write_frames {
        println!("reconfigured F/k: ({F1}, {K1}) -> ({F2}, {K2}) — same graph, no rebuild");
    }

    // ── Phase 2: identical replay (SAME step body), new parameters. ─────────
    for s in steps_phase1..total {
        step(s)?;
    }

    // Write the three frames captured during the loops.
    if write_frames {
        if let Some(ev) = early_v {
            claspr::write_ppm_rgba8("gray-scott-early.ppm", W as u32, H as u32, &colorize_v(&ev))?;
            println!("wrote frame gray-scott-early.ppm");
        }
        if let Some(lv) = late_v {
            claspr::write_ppm_rgba8("gray-scott-late.ppm", W as u32, H as u32, &colorize_v(&lv))?;
            println!("wrote frame gray-scott-late.ppm (end of phase 1, F={F1}, k={K1})");
        }
        claspr::write_ppm_rgba8(
            "gray-scott-reconfigured.ppm",
            W as u32,
            H as u32,
            &colorize_v(&final_v),
        )?;
        println!("wrote frame gray-scott-reconfigured.ppm (after retune, F={F2}, k={K2})");
    }

    Ok(final_v)
}

/// **Strategy 2: immutable / unroll-by-2 replay.** Same three kernels, same
/// arithmetic — but the double-buffer rotation is baked into the graph TOPOLOGY
/// instead of per-step rebinding, giving a graph with FIXED buffer handles that
/// replays with ZERO mutation between iterations.
///
/// ## Why unroll-by-2 fixes the handles
///
/// Double-buffering rotates roles every step: step k reads A writes B, step k+1
/// reads B writes A. A ONE-step graph therefore has handles that change each
/// step (which is exactly why [`run_swap`] must `mutate_call` the four field
/// slots crossed between steps). But compose the per-step subgraph WITH ITSELF —
/// step-k THEN step-k+1 — and after those two steps "current" is back in A. The
/// **two-step** graph has fixed buffer roles: its inputs are `(A_u, A_v)` and
/// after running it "current" is again `(A_u, A_v)`. So it can be replayed
/// `steps/2` times over the SAME handles with no rebinding at all — the shape
/// that bakes cleanly into one immutable command buffer once step-c lands.
///
/// ## How the rotation is threaded (pipes, not slots)
///
/// The two-step graph is the per-step DAG composed with itself. Step 1's
/// `combine` is 6-output — it threads ALL its buffer args forward as pipes,
/// including its read-only inputs `(u_a, v_a)` passed through unchanged. So step
/// 2 reads step 1's OUTPUT buffers `(u_b, v_b)` as its fields and writes back
/// into step 1's INPUT buffers `(u_a, v_a)` — both arrive as pipes off step 1's
/// `combine`, needing NO slots. Net over the pair: A→B then B→A = identity on
/// roles.
///
/// ## The composition model: a curried, bind-by-name meta-kernel
///
/// The per-step subgraph is captured ONCE as a pair of curried closures over the
/// consuming set-once `claspr::eager` verb (`call`) and the unified `Tag(value)`/
/// `Tag(pipe)` slot constructor:
///
/// - **`get_meta_kernel(ks, lap_u, lap_v)`** builds the raw three-dispatch DAG
///   (`lap_u`, `lap_v`, `combine`) with ALL SEVEN input slots left OPEN
///   (`Grid`, `F`, `K`, `UIn`, `VIn`, `UOut`, `VOut`). It then `.and_then`s a
///   **4-of-6 output TRIM** that re-exposes just the four FIELD buffers
///   `(UIn, VIn, UOut, VOut)` as a clean `bundle4` Handle — dropping the two
///   lap-scratch pipes the composition never reads, so the compose closure
///   destructures a tidy `|(u_a, v_a, u_b, v_b)|` and not a 6-tuple with
///   `_lap` placeholders.
/// - **`curried_kernel(ks, lap_u, lap_v)`** partially binds the invariants that
///   never rotate — the launch `Grid` and the reaction scalars `F`/`K` — via
///   set-once `call`, leaving ONLY the four field slots open for the step-specific
///   `call` at the call site.
///
/// `step` is then two `curried_kernel` calls composed:
///
/// - STEP 1 binds the four field slots to CONCRETE buffers by value:
///   `call((UIn(u_a), VIn(v_a), UOut(u_b), VOut(v_b)))` — read A, write B.
/// - STEP 2, inside the `and_then`, wires the SAME four slots to step 1's output
///   PIPES with the rotation VISIBLE in the arg list:
///   `call((UIn(u_b), VIn(v_b), UOut(u_a), VOut(v_a)))` — read B, write back
///   into A. The SAME tag constructor fed a pipe (`Tag(pipe)`) installs
///   `SlotState::FedByPipe`, so each slot DRAINS its upstream pipe every run and
///   re-arms on the next replay — no separate `feed` verb.
///
/// Set-once `call` is CONSUMING + INFALLIBLE: it returns the owned graph (so it is
/// usable as the bare `U` inside an `and_then` closure) and DEFERS any bind
/// error to `sync`'s readiness check (sticky/poison — rebuild to recover). There
/// is no per-step rebinding: the pair is bound once at build and replayed
/// `steps/2` times via plain `sync()`.
///
/// ## Lap scratch: two independent sets
///
/// The two steps use SEPARATE lap scratch buffers (`lap_*1` for step 1,
/// `lap_*2` for step 2). Reusing one set across the two steps within a single
/// graph would make step 2's lap WRITE the very buffer step 1's `combine` READS,
/// a write-after-read hazard the pure-dataflow (pipe) graph does not serialize —
/// so two sets keep the two steps' Laplacians independent and correct. They are
/// concrete buffers passed by arg into each `get_meta_kernel` call, never rebound.
///
/// The gray-scott per-step meta-kernel as a **named, reusable, testable** subgraph.
///
/// This was a local closure inside `run_immutable` — and it *had* to be, because a
/// closure's graph type is unnameable, so a hand-written `-> impl DeviceOp<..>` return
/// needed the full `Output`/`Handle`/`Checkouts` shape spelled out, and the noise
/// pushed it back inline. `-> impl `[`Subgraph`](claspr::eager::Subgraph)`<O>` fixes
/// that: `O` is the four field buffers, and the one bound pins the canonical
/// [`OutputShape`](claspr::eager::OutputShape) handle/checkouts + `FromCheckout` — so
/// callers still destructure a clean `|(u_in, v_in, u_out, v_out)|` and compose it
/// onward, with no where-clause here. Now it can be reused across sites, aliased, and
/// unit-tested in isolation (see `meta_kernel_builds_and_runs`), which a closure can't.
///
/// Builds the raw three-dispatch DAG with ALL SEVEN slots OPEN (`Grid`/`UIn`/`VIn`/
/// `UOut`/`VOut`/`F`/`K`), then TRIMS `combine`'s six-output handle to the four field
/// buffers via `bundle4`. Takes `ks` + the two lap-scratch buffers by arg; the built
/// graph does NOT borrow `ks` (the launchers clone the context internally).
fn gray_scott_step(
    ks: &gpu::Kernels,
    lap_u: DeviceSlice<f32>,
    lap_v: DeviceSlice<f32>,
) -> impl claspr::eager::Subgraph<(
    DeviceSlice<f32>,
    DeviceSlice<f32>,
    DeviceSlice<f32>,
    DeviceSlice<f32>,
)> + use<> {
    // `+ use<>` (edition-2024 precise capturing): the built graph owns everything it
    // needs — each launcher clones its `Kernel`/context by value — so it does NOT
    // borrow `ks`. Without it, 2024's RPIT auto-captures `ks`'s lifetime and the
    // step-2 call inside the `and_then` move-closure trips E0515.
    use claspr::eager::bundle4;
    ks.laplacian(slot!(Grid), slot!(UIn), lap_u)
        .and_then(move |(u_in, lap_u_pipe)| {
            ks.laplacian(slot!(Grid), slot!(VIn), lap_v)
                .and_then(move |(v_in, lap_v_pipe)| {
                    ks.combine(
                        slot!(Grid),
                        u_in,       // pipe: U field (read)
                        v_in,       // pipe: V field (read)
                        lap_u_pipe, // pipe: U-Laplacian
                        lap_v_pipe, // pipe: V-Laplacian
                        slot!(UOut),
                        slot!(VOut),
                        slot!(F),
                        slot!(K),
                    )
                    // 4-of-6 TRIM: re-expose ONLY the four FIELD pipes so the caller
                    // sees `(u_in, v_in, u_out, v_out)`, not the lap-scratch placeholders.
                    .and_then(|(u_in, v_in, _lap_u, _lap_v, u_out, v_out)| {
                        bundle4(u_in, v_in, u_out, v_out)
                    })
                })
        })
}

/// `steps` must be EVEN (the unroll period is 2). Returns the final `V` field.
fn run_immutable(
    ctx: &Context,
    grid_w: usize,
    grid_h: usize,
    steps: usize,
    feed_rate: f32,
    kill_rate: f32,
    write_frame: bool,
) -> claspr::Result<Vec<f32>> {
    assert_eq!(
        grid_w * grid_h,
        N,
        "this helper is fixed to the W*H constants"
    );
    assert_eq!(steps % 2, 0, "unroll-by-2 requires an even step count");
    let ks = gpu::kernels(ctx)?;

    let (u0, v0) = initial_fields();

    // Two device buffers per field. A holds the initial / period-boundary state;
    // B is the intermediate (step-1 output). Over the two-step graph the roles
    // are identity: read A → write B → read B → write A. Minted up front and
    // MOVED into the compose closure — the closures must NOT borrow `ctx` (which
    // `sync(&ctx)` needs); the kernel ops clone the context internally, so the
    // graph they build owns everything it needs.
    let u_a = seeded(ctx, u0)?;
    let u_b = seeded(ctx, vec![0.0f32; N])?;
    let v_a = seeded(ctx, v0)?;
    let v_b = seeded(ctx, vec![0.0f32; N])?;

    // Two INDEPENDENT lap scratch sets — one per unrolled step (see doc above).
    let lap_u1 = seeded(ctx, vec![0.0f32; N])?;
    let lap_v1 = seeded(ctx, vec![0.0f32; N])?;
    let lap_u2 = seeded(ctx, vec![0.0f32; N])?;
    let lap_v2 = seeded(ctx, vec![0.0f32; N])?;

    // ── Build the TWO-step meta-kernel from the NAMED `gray_scott_step` fn. ──
    //   Each step: a fresh `gray_scott_step(...)` (reusable `impl Subgraph<..>`), then
    //   a set-once `call` of the INVARIANTS (`Grid`, `F`, `K`), then a `call` of the
    //   four field slots.
    //   STEP 1 (read A, write B): value-bind the four field slots to concrete bufs.
    //   STEP 2 (read B, write A): feed the same four slots from step-1's output
    //     pipes — the crossed rotation is VISIBLE in the `Tag(pipe)` args.
    // Over the pair the buffer roles are identity (A→B→A), so the graph is bound
    // ONCE at build and replays with NO per-step rebinding. A small `invariants`
    // helper keeps the repeated `(Grid, F, K)` currying tuple in one place.
    let invariants = || {
        (
            F(feed_rate),
            K(kill_rate),
            Grid(LaunchSpec::from([grid_w, grid_h])),
        )
    };
    let g = gray_scott_step(&ks, lap_u1, lap_v1)
        .call(invariants())
        .call((UIn(u_a), VIn(v_a), UOut(u_b), VOut(v_b)))
        .and_then(move |(u_a, v_a, u_b, v_b)| {
            gray_scott_step(&ks, lap_u2, lap_v2)
                .call(invariants())
                .call((
                    UIn(u_b),  // read B
                    VIn(v_b),  // read B
                    UOut(u_a), // write back into A
                    VOut(v_a), // write back into A
                ))
        });

    // ── Replay the TWO-step graph steps/2 times via plain sync(). ───────────
    // NO per-step rebinding: each `sync` runs both unrolled steps (A→B→A), and
    // the returned Checkouts drop at end of iteration, re-arming every lent
    // concrete cell to the SAME buffer — so the next `sync` reuses the identical
    // handles. This is the whole point: an immutable graph replayed by period.
    let pairs = steps / 2;
    let mut final_v: Vec<f32> = Vec::new();
    for p in 0..pairs {
        // Step 2's trimmed `bundle4` is terminal: sync yields its FOUR field
        // Checkouts (u_in, v_in, u_out, v_out) = (u_b, v_b, u_a, v_a) — the last
        // two are the fresh A fields (this pair's result). Read V from the A
        // output (`v_out` = v_a) before drop.
        let (_u_b_co, _v_b_co, _u_a_co, v_a_co) = g.sync(ctx)?;
        if p + 1 == pairs {
            final_v = read_field(&v_a_co)?;
        }
        // Checkouts drop here → re-arm the graph over the same handles. Next
        // sync replays with zero rebinding.
    }

    if write_frame {
        claspr::write_ppm_rgba8(
            "gray-scott-immutable.ppm",
            W as u32,
            H as u32,
            &colorize_v(&final_v),
        )?;
        println!(
            "wrote frame gray-scott-immutable.ppm ({steps} steps @ F={feed_rate}, k={kill_rate}, \
             unroll-by-2, {pairs} replays, no rebinding)"
        );
    }

    Ok(final_v)
}

fn main() -> claspr::Result<()> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return Ok(());
        }
    };

    // ── Variant 1: the mutable-swap showcase (with a mid-run reconfigure). ───
    println!(
        "gray-scott [swap]: {W}x{H} grid (2D dispatch), {} phase-1 + {} phase-2 steps; the \
         per-step THREE-dispatch graph (lap_u, lap_v, combine) is built ONCE and replayed, \
         ping-ponging four field slots each step (meta-kernel, mutable-swap replay).",
        STEPS_PHASE1, STEPS_PHASE2
    );
    run_swap(&ctx, W, H, STEPS_PHASE1, STEPS_PHASE2, STEPS_EARLY, true)?;

    // ── Variant 2: the immutable / unroll-by-2 form (same math, fixed graph). ─
    // Run it at the SAME phase-1 regime for the SAME number of steps as swap's
    // phase 1, so the two are directly comparable — then prove they agree.
    println!(
        "gray-scott [immutable]: SAME computation via a TWO-step graph (step-k THEN step-k+1) \
         whose buffer roles are IDENTITY over the pair — bound ONCE with call, replayed {}/2 \
         times via plain sync() with ZERO per-step rebinding (unroll-by-period replay).",
        STEPS_PHASE1
    );
    let imm_v = run_immutable(&ctx, W, H, STEPS_PHASE1, F1, K1, true)?;

    // The equality proof at full scale: a single-regime swap run of the SAME
    // steps/params must match the immutable run to the bit.
    let swap_v = run_swap(&ctx, W, H, STEPS_PHASE1, 0, STEPS_PHASE1 + 1, false)?;
    let agree = swap_v == imm_v;
    println!(
        "gray-scott: swap vs immutable final V over {} steps @ (F={F1}, k={K1}): {}",
        STEPS_PHASE1,
        if agree {
            "BIT-IDENTICAL — the two graph shapes are one meta-kernel".to_string()
        } else {
            let max_abs = swap_v
                .iter()
                .zip(&imm_v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            format!("differ (max |Δ| = {max_abs:e})")
        }
    );

    println!("gray-scott: done");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: a SHORT sim must stay bounded, NaN-free, and actually evolve
    /// away from the initial condition (V grows from ~0). Skips silently with no
    /// device.
    #[test]
    fn gray_scott_evolves_and_stays_sane() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };

        // Reuse the full constants (grid is fixed to W*H in `run_swap`), but only
        // a handful of steps in each phase — enough to move V off zero.
        let final_v = run_swap(&ctx, W, H, 120, 120, 60, false).expect("run gray-scott smoke");

        let sum_v: f64 = final_v.iter().map(|&x| x as f64).sum();
        let any_nan = final_v.iter().any(|x| x.is_nan());
        let in_bounds = final_v.iter().all(|&x| (-0.5..=1.5).contains(&x));

        assert!(!any_nan, "V field must be NaN-free");
        assert!(
            in_bounds,
            "V field must stay in a sane range (~[0,1]); got min={:?} max={:?}",
            final_v.iter().cloned().fold(f32::INFINITY, f32::min),
            final_v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
        assert!(
            sum_v > 1.0,
            "V must grow from the ~0 initial condition (the reaction ran); sum_v={sum_v}"
        );
    }

    /// The payoff of naming the per-step meta-kernel: `gray_scott_step` — a former
    /// inline closure, now a reusable `impl Subgraph<O>` — can be built, bound, and
    /// run **in isolation**, with none of `run_immutable`/`run_swap`'s double-buffer
    /// scaffolding. (A closure's unnameable graph type can't be returned from a fn,
    /// so this unit test was impossible before.) One step from the seeded IC must
    /// stay finite, in range, and actually change the V field.
    #[test]
    fn meta_kernel_builds_and_runs() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        let ks = gpu::kernels(&ctx).expect("load kernels");
        let (u0, v0) = initial_fields();
        let v0_before = v0.clone();

        // Build the NAMED subgraph standalone; bind all seven slots (three invariants
        // then the four fields) via two set-once `call`s; run ONE step.
        let g = gray_scott_step(
            &ks,
            seeded(&ctx, vec![0.0f32; N]).expect("lap_u"),
            seeded(&ctx, vec![0.0f32; N]).expect("lap_v"),
        )
        .call((F(0.037), K(0.06), Grid(LaunchSpec::from([W, H]))))
        .call((
            UIn(seeded(&ctx, u0).expect("u_in")),
            VIn(seeded(&ctx, v0).expect("v_in")),
            UOut(seeded(&ctx, vec![0.0f32; N]).expect("u_out")),
            VOut(seeded(&ctx, vec![0.0f32; N]).expect("v_out")),
        ));
        let (_u_in, _v_in, _u_out, v_out) = g.sync(&ctx).expect("sync one step");
        let v = read_field(&v_out).expect("read v_out");

        assert!(!v.iter().any(|x| x.is_nan()), "V must be NaN-free");
        assert!(
            v.iter().all(|&x| (-0.5..=1.5).contains(&x)),
            "V must stay in a sane range after one step"
        );
        assert!(v != v0_before, "one step must actually change the V field");
    }

    /// **The thesis proof.** The two graph shapes — mutable-swap replay
    /// ([`run_swap`], one-step graph re-bound crossed each step) and immutable
    /// unroll-by-2 replay ([`run_immutable`], two-step graph, fixed handles, no
    /// rebinding) — are the SAME arithmetic in the SAME order. From the SAME
    /// initial condition, SAME `(F, k)`, and SAME (even) step count, their final
    /// `V` fields must be **bit-identical**. That equality is the proof that
    /// unroll-by-period replay == mutable-swap replay: one meta-kernel, two
    /// execution strategies.
    /// Pure-host reference implementation of the same simulation — the CPU
    /// golden. Jacobi double-buffer, one step = (compute both Laplacians from the
    /// current fields, then the reaction+diffusion update writing the next
    /// fields), matching `gpu::combine` term-for-term and in the same float
    /// evaluation order. Returns the final `V` field, like `run_swap` /
    /// `run_immutable`.
    ///
    /// This is NOT expected to be *bit*-identical to the GPU (FMA contraction and
    /// op-fusion differ between the OpenCL device and host f32), so callers must
    /// compare with a tolerance. Its purpose is to break the symmetry of the
    /// swap-vs-immutable equality: when the two GPU strategies disagree, whichever
    /// one drifts from this golden by more than ordinary CPU/GPU f32 error is the
    /// incorrect one.
    fn cpu_reference(steps: usize, feed: f32, kill: f32) -> Vec<f32> {
        use super::gpu::{DT, DU, DV, laplacian_at};
        let (mut u, mut v) = initial_fields();
        let mut u_next = vec![0.0f32; N];
        let mut v_next = vec![0.0f32; N];
        for _ in 0..steps {
            for y in 0..H as u32 {
                for x in 0..W as u32 {
                    let i = (y * W as u32 + x) as usize;
                    let lap_u = laplacian_at(&u, x, y, W as u32, H as u32);
                    let lap_v = laplacian_at(&v, x, y, W as u32, H as u32);
                    let uu = u[i];
                    let vv = v[i];
                    let uvv = uu * vv * vv;
                    u_next[i] = uu + DT * (DU * lap_u - uvv + feed * (1.0 - uu));
                    v_next[i] = vv + DT * (DV * lap_v + uvv - (feed + kill) * vv);
                }
            }
            std::mem::swap(&mut u, &mut u_next);
            std::mem::swap(&mut v, &mut v_next);
        }
        v
    }

    #[test]
    fn swap_and_immutable_agree_bit_for_bit() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };

        // A short EVEN-length single-regime sim under both strategies.
        const STEPS: usize = 120;

        // Swap: single regime (phase2 = 0), no frames, no reconfigure.
        let swap_v = run_swap(&ctx, W, H, STEPS, 0, STEPS + 1, false).expect("run swap variant");
        // Immutable: same steps / params, no frame.
        let imm_v = run_immutable(&ctx, W, H, STEPS, F1, K1, false).expect("run immutable variant");

        // CPU golden — the ground truth for which strategy is physically correct.
        // Compared with a tolerance (CPU/GPU f32 are not bit-identical). This
        // DIAGNOSES the swap-vs-immutable race: the strategy that drifts from the
        // golden is the buggy one. Reported before the strict bit-equality gate so
        // the verdict is visible even while that gate fails.
        let gold_v = cpu_reference(STEPS, F1, K1);
        let dmax = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        };
        let swap_vs_gold = dmax(&swap_v, &gold_v);
        let imm_vs_gold = dmax(&imm_v, &gold_v);
        eprintln!(
            "CPU-golden diagnosis: max|swap-gold|={swap_vs_gold:e}  \
             max|imm-gold|={imm_vs_gold:e}  (STEPS={STEPS})"
        );
        // Ordinary CPU/GPU f32 drift over {STEPS} steps of this system stays small;
        // a dropped-dependency race produces a structurally larger error. Measured
        // separation on the per-op path (no cl_khr_command_buffer): rusticl/llvmpipe
        // = 0, intel-legacy Iris = 2.68e-7 (genuine FMA/rounding drift, deterministic,
        // swap == imm). A raced immutable CB (pocl) = 1.4e-2..4.4e-1, nondeterministic.
        // 1e-5 sits ~37x above the largest legitimate drift and ~1400x below the
        // smallest race — clears real rounding, still catches a subtle dropped dep.
        const GOLDEN_TOL: f32 = 1.0e-5;
        assert!(
            swap_vs_gold < GOLDEN_TOL,
            "swap strategy diverges from CPU golden: max|Δ|={swap_vs_gold:e} (tol {GOLDEN_TOL:e})"
        );
        assert!(
            imm_vs_gold < GOLDEN_TOL,
            "immutable strategy diverges from CPU golden: max|Δ|={imm_vs_gold:e} (tol {GOLDEN_TOL:e})"
        );

        // Both must have evolved off the ~0 initial V and be NaN-free.
        assert!(
            !swap_v.iter().any(|x| x.is_nan()),
            "swap V must be NaN-free"
        );
        assert!(
            !imm_v.iter().any(|x| x.is_nan()),
            "immutable V must be NaN-free"
        );
        let swap_sum: f64 = swap_v.iter().map(|&x| x as f64).sum();
        let imm_sum: f64 = imm_v.iter().map(|&x| x as f64).sum();
        assert!(swap_sum > 1.0, "swap V must evolve from ~0; sum={swap_sum}");
        assert!(
            imm_sum > 1.0,
            "immutable V must evolve from ~0; sum={imm_sum}"
        );

        // THE equality: same math, same order, different graph geometry.
        assert_eq!(
            swap_v.len(),
            imm_v.len(),
            "both variants produce a W*H field"
        );
        let max_abs = swap_v
            .iter()
            .zip(&imm_v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            swap_v, imm_v,
            "swap and immutable final V must be BIT-IDENTICAL (same arithmetic, same order); \
             max |Δ| = {max_abs:e}"
        );
    }
}
