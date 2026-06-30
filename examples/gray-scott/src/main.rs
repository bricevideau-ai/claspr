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
//! - [`gpu::laplacian`] — reads ONE field, writes its 5-point periodic Laplacian
//!   into a scratch buffer. It is dispatched **TWICE per step** — once for `U`
//!   (→ `lap_u`) and once for `V` (→ `lap_v`). **Same kernel, two sites.**
//! - [`gpu::combine`] — reads `(u_in, v_in, lap_u, lap_v)` and writes
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
//!    buffers. After a step we `into_inner()` all four output Checkouts to KEEP
//!    their buffers (severing the slots, `Lent → Severed`), then re-bind them
//!    *crossed* in ONE [`mutate_call`](claspr::DeviceOpExt::mutate_call): this
//!    step's `*Out` becomes next step's `*In`, and the now-stale `*In` becomes
//!    next step's scratch `*Out`. The crossed re-bind MUST go through the
//!    `mutate` verb — a plain `bind` on a severed slot is `Error::SlotSevered`
//!    (the canonical ping-pong rule from `tests/tier2/tests/double_buffering.rs`,
//!    here doubled across the two field pairs and issued as one 4-tuple
//!    `mutate_call`).
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
//! Writes three PPM frames — `gray-scott-early.ppm`, `gray-scott-late.ppm`,
//! `gray-scott-reconfigured.ppm` — showing pattern emergence then the mid-run
//! retune. `V` is colorized on the HOST after `download` (a blue→teal→white
//! ramp); no image kernel is involved.
//!
//! Run with
//! `OCL_ICD_VENDORS=$HOME/local/pocl/etc/OpenCL/vendors cargo run -p gray-scott-example`.
//! The `#[test]` at the bottom is the smoke test (short sim, asserts the field
//! stays bounded, is NaN-free, and actually evolved away from the initial
//! condition).

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
    /// the [`laplacian`] kernel.
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
    /// fused step). The two scalars `(feed, kill)` are scalar slots — bound once,
    /// re-read each replay, and `mutate_bind`-ed mid-run to reconfigure the
    /// meta-kernel.
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
        feed: f32,
        kill: f32,
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

        let u_next = u + DT * (DU * lap_u[i] - uvv + feed * (1.0 - u));
        let v_next = v + DT * (DV * lap_v[i] + uvv - (feed + kill) * v);

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

/// Run the whole simulation. Returns the final `V` field (host) for the smoke
/// test, and writes the PPM frames as it goes when `write_frames` is set.
fn run(
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
        });

    // Scalar slots bound ONCE — read (not consumed) on every replay, so they
    // persist across all steps for free. F/K are the phase-1 regime we will
    // reconfigure mid-run. (Du/Dv are compile-time consts in the device module.)
    g.bind(F(F1))?;
    g.bind(K(K1))?;

    // The shared launch slot: ONE bind fills the Grid cell at ALL THREE dispatch
    // sites. The dispatch is genuinely 2D — `[W, H]`.
    g.bind(Grid(LaunchSpec::from([grid_w, grid_h])))?;

    // Step 0: bind the initial buffer roles (set-once `bind` on virgin slots).
    // UIn/VIn each fan out to their lap dispatch AND combine from this one bind.
    g.bind(UIn(u_a))?;
    g.bind(VIn(v_a))?;
    g.bind(UOut(u_b))?;
    g.bind(VOut(v_b))?;

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

        // SWAP for the next step — unless this was the final step. `into_inner`
        // keeps each buffer AND severs its slot (Lent → Severed). The freshly
        // written `*Out` becomes next step's `*In`; the stale `*In` becomes next
        // step's scratch `*Out`. Crossed re-bind ⇒ MUST be the `mutate` verb;
        // ONE `mutate_call` rebinds all four field slots together.
        if step_idx + 1 < total {
            let next_u_in = u_out_co.into_inner();
            let next_v_in = v_out_co.into_inner();
            let next_u_out = u_in_co.into_inner();
            let next_v_out = v_in_co.into_inner();
            g.mutate_call((
                UIn(next_u_in),
                VIn(next_v_in),
                UOut(next_u_out),
                VOut(next_v_out),
            ))?;
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

fn main() -> claspr::Result<()> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return Ok(());
        }
    };
    println!(
        "gray-scott: {W}x{H} grid (2D dispatch), {} phase-1 + {} phase-2 steps; the per-step \
         THREE-dispatch graph (lap_u, lap_v, combine) is built ONCE and replayed (meta-kernel).",
        STEPS_PHASE1, STEPS_PHASE2
    );
    run(&ctx, W, H, STEPS_PHASE1, STEPS_PHASE2, STEPS_EARLY, true)?;
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

        // Reuse the full constants (grid is fixed to W*H in `run`), but only a
        // handful of steps in each phase — enough to move V off zero.
        let final_v = run(&ctx, W, H, 120, 120, 60, false).expect("run gray-scott smoke");

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
}
