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
//! ## How this sample exercises the slot machinery (the whole point)
//!
//! One fused kernel — [`gpu::gray_scott_step`] — reads `(u_in, v_in)` and
//! writes `(u_out, v_out)` in a single dispatch, with `F`, `k`, `Du`, `Dv` as
//! **scalar arguments** (grid size + `dt` are compile-time constants of the
//! meta-kernel, so the runtime kernel-arg tuple stays within the arity-8
//! `KernelArgs` ceiling). The host builds the per-step update graph `g` exactly
//! once, then drives the entire simulation by re-`sync()`-ing that one graph.
//! Concretely:
//!
//! 1. **Reuse** — `g = ks.gray_scott_step([W*H], slot!(UIn), slot!(VIn),
//!    slot!(UOut), slot!(VOut), slot!(F), slot!(K), …)` is built ONCE, then
//!    `g.sync(&ctx)` runs in a `for` loop for thousands of steps. Defining the
//!    meta-kernel once and replaying it cheaply is precisely what step-c
//!    command-buffer caching will accelerate: today each `sync` re-validates +
//!    re-enqueues, tomorrow the cached command buffer just re-submits.
//!
//! 2. **Double-buffering** — `U` and `V` each ping-pong between two device
//!    buffers. After a step we `into_inner()` all four output Checkouts to KEEP
//!    their buffers (severing the slots, `Lent → Severed`), then `mutate_bind`
//!    them *crossed*: this step's `*Out` becomes next step's `*In`, and the now
//!    stale `*In` becomes next step's scratch `*Out`. The crossed re-bind MUST
//!    be `mutate_bind` — a plain `bind` on a severed slot is `Error::SlotSevered`
//!    (the canonical ping-pong rule from `tests/tier2/tests/double_buffering.rs`,
//!    here doubled across the two field pairs).
//!
//! 3. **Scalar slots** — `F`, `k`, `Du`, `Dv` are `slot!` SCALAR slots
//!    (non-resource, `Copy`, value-equality, never handed back as Checkouts).
//!    Bound once up front, they are READ (not consumed) on every replay, so they
//!    persist across all four-thousand steps for free.
//!
//! 4. **Mutate-to-reconfigure (the meta-kernel proof)** — after the first phase
//!    we `mutate_bind(F(..))` / `mutate_bind(K(..))` to a *different* `(F, k)`
//!    regime and keep stepping the SAME graph. The very same meta-kernel now
//!    grows a different pattern. We emit a PPM frame before and after so the
//!    reconfiguration is visible: **the graph is a kernel you reparameterize
//!    without rebuilding it.**
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
use claspr::{Context, DeviceSlice};
use claspr::{slot, slots};

#[claspr::device]
mod gpu {
    /// Grid dimensions + time step are compile-time constants of the
    /// meta-kernel. The OpenCL `KernelArgs` tuple impls top out at arity 8, so
    /// we spend the eight argument slots on what actually VARIES at runtime —
    /// the four ping-pong field buffers plus the four scalar slots `F`, `k`,
    /// `Du`, `Dv` — and bake the rest in. `dt = 1.0` is the standard
    /// Gray-Scott step. Host code mirrors `W`/`H` with its own constants.
    pub const W: u32 = 256;
    pub const H: u32 = 256;
    pub const DT: f32 = 1.0;

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
    /// boundaries: `left + right + up + down - 4*center`. Pure helper shared by
    /// both field updates inside the fused kernel.
    pub fn laplacian(field: &[f32], x: u32, y: u32, w: u32, h: u32) -> f32 {
        let idx = |xx: u32, yy: u32| (yy * w + xx) as usize;
        let center = field[idx(x, y)];
        let left = field[idx(wrap(x, false, w), y)];
        let right = field[idx(wrap(x, true, w), y)];
        let up = field[idx(x, wrap(y, false, h))];
        let down = field[idx(x, wrap(y, true, h))];
        left + right + up + down - 4.0 * center
    }

    /// One fused Gray-Scott update step over the whole grid.
    ///
    /// Reads the current fields `(u_in, v_in)` and writes the next fields
    /// `(u_out, v_out)` — a single dispatch advances BOTH fields, which is why
    /// the host only ever ping-pongs two buffer *pairs* and binds one set of
    /// scalar parameters. The four slices are the double-buffer slots; the four
    /// scalars `(feed, kill, du, dv)` are scalar slots (bound once, re-read
    /// every replay; `feed`/`kill` are also `mutate_bind`-ed mid-run).
    #[claspr::kernel]
    pub fn gray_scott_step(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] u_in: &[f32],
        #[spirv(cross_workgroup)] v_in: &[f32],
        #[spirv(cross_workgroup)] u_out: &mut [f32],
        #[spirv(cross_workgroup)] v_out: &mut [f32],
        feed: f32,
        kill: f32,
        du: f32,
        dv: f32,
    ) {
        let gid = id.x as u32;
        let n = W * H;
        if gid >= n {
            return;
        }
        let x = gid % W;
        let y = gid / W;
        let i = gid as usize;

        let u = u_in[i];
        let v = v_in[i];
        let uvv = u * v * v;

        let lap_u = laplacian(u_in, x, y, W, H);
        let lap_v = laplacian(v_in, x, y, W, H);

        let du_next = u + DT * (du * lap_u - uvv + feed * (1.0 - u));
        let dv_next = v + DT * (dv * lap_v + uvv - (feed + kill) * v);

        u_out[i] = du_next;
        v_out[i] = dv_next;
    }
}

// ── Simulation parameters ────────────────────────────────────────────────────

const W: usize = 256;
const H: usize = 256;
const N: usize = W * H;

// Diffusion constants (held fixed across the whole run; still bound through
// scalar slots to show the mechanism). The time step `dt` and grid size live as
// compile-time constants in the device module (`gpu::DT`/`gpu::W`/`gpu::H`) so
// the runtime kernel-arg tuple stays within the arity-8 `KernelArgs` ceiling —
// the eight slots are spent on what actually varies (4 fields + F/k/Du/Dv).
const DU: f32 = 0.16;
const DV: f32 = 0.08;

// Phase-1 reaction regime, then the mid-run retune for phase 2.
const F1: f32 = 0.060;
const K1: f32 = 0.062;
const F2: f32 = 0.034; // "mazes/worms" regime — visibly different texture
const K2: f32 = 0.056;

const STEPS_EARLY: usize = 800; // first frame
const STEPS_PHASE1: usize = 4000; // total steps of phase 1 (regime F1/K1)
const STEPS_PHASE2: usize = 4000; // steps after the mid-run reconfigure (F2/K2)

// ── Double-buffer + scalar slot tags ─────────────────────────────────────────

slots! {
    // The two field pairs, each ping-ponging between two device buffers.
    UIn:  DeviceSlice<f32>,
    UOut: DeviceSlice<f32>,
    VIn:  DeviceSlice<f32>,
    VOut: DeviceSlice<f32>,
    // Scalar parameter slots — bound once, re-read each replay; F and K are the
    // ones we mutate_bind mid-run to reconfigure the meta-kernel.
    F:  f32,
    K:  f32,
    Du: f32,
    Dv: f32,
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
    // "B" (scratch / next) device buffer. The B buffers start zeroed; the kernel
    // overwrites them.
    let u_a = seeded(ctx, u0)?;
    let u_b = seeded(ctx, vec![0.0f32; N])?;
    let v_a = seeded(ctx, v0)?;
    let v_b = seeded(ctx, vec![0.0f32; N])?;

    // ── Build the per-step update graph ONCE. This is the meta-kernel. ───────
    // Four buffer slots (the double-buffer pairs) + four scalar slots. Every
    // subsequent step is just a `sync()` over THIS graph — never rebuilt.
    let g = ks.gray_scott_step(
        [N],
        slot!(UIn),
        slot!(VIn),
        slot!(UOut),
        slot!(VOut),
        slot!(F),
        slot!(K),
        slot!(Du),
        slot!(Dv),
    );

    // Scalar slots bound ONCE — read (not consumed) on every replay, so they
    // persist across all steps for free. Diffusion stays fixed; F/K are the
    // phase-1 regime we will reconfigure mid-run.
    g.bind(Du(DU))?;
    g.bind(Dv(DV))?;
    g.bind(F(F1))?;
    g.bind(K(K1))?;

    // Step 0: bind the initial buffer roles (set-once `bind` on virgin slots).
    g.bind(UIn(u_a))?;
    g.bind(VIn(v_a))?;
    g.bind(UOut(u_b))?;
    g.bind(VOut(v_b))?;

    let (mut u_in_co, mut v_in_co, mut u_out_co, mut v_out_co) = g.sync(ctx)?;

    // `early` frame from the field as it stands after `early_at` steps.
    let mut early_v: Option<Vec<f32>> = None;

    // Helper: advance ONE step by ping-ponging the four buffers (crossed
    // `mutate_bind`) and re-syncing the SAME graph.
    let mut step = 1usize; // step 0 ran above
    let total_phase1 = steps_phase1;

    // ── Phase 1 loop: replay the meta-kernel, ping-ponging each step. ───────
    while step < total_phase1 {
        // SWAP. `into_inner` keeps each buffer AND severs its slot. The freshly
        // written `*Out` becomes next step's `*In`; the stale `*In` becomes next
        // step's scratch `*Out`. Crossed re-bind ⇒ MUST be `mutate_bind`.
        let next_u_in = u_out_co.into_inner();
        let next_v_in = v_out_co.into_inner();
        let next_u_out = u_in_co.into_inner();
        let next_v_out = v_in_co.into_inner();

        g.mutate_bind(UIn(next_u_in))?;
        g.mutate_bind(VIn(next_v_in))?;
        g.mutate_bind(UOut(next_u_out))?;
        g.mutate_bind(VOut(next_v_out))?;

        let next = g.sync(ctx)?;
        u_in_co = next.0;
        v_in_co = next.1;
        u_out_co = next.2;
        v_out_co = next.3;
        step += 1;

        if step == early_at && write_frames {
            // The latest result lives in the `*Out` buffers (last written).
            early_v = Some(read_field(&v_out_co)?);
            println!(
                "step {step}/{}: captured early frame",
                total_phase1 + steps_phase2
            );
        }
        if write_frames && step.is_multiple_of(1000) {
            println!("step {step}/{}", total_phase1 + steps_phase2);
        }
    }

    if write_frames {
        if let Some(ev) = early_v {
            let rgba = colorize_v(&ev);
            claspr::write_ppm_rgba8("gray-scott-early.ppm", W as u32, H as u32, &rgba)?;
            println!("wrote frame gray-scott-early.ppm");
        }
        let late_v = read_field(&v_out_co)?;
        let rgba = colorize_v(&late_v);
        claspr::write_ppm_rgba8("gray-scott-late.ppm", W as u32, H as u32, &rgba)?;
        println!("wrote frame gray-scott-late.ppm (end of phase 1, F={F1}, k={K1})");
    }

    // ── MID-RUN RECONFIGURE. Same graph, different reaction regime. ─────────
    // No rebuild: just `mutate_bind` the two scalar slots that define the
    // regime. The very next `sync` runs the SAME meta-kernel at the new (F, k),
    // and the pattern morphs.
    g.mutate_bind(F(F2))?;
    g.mutate_bind(K(K2))?;
    if write_frames {
        println!("reconfigured F/k: ({F1}, {K1}) -> ({F2}, {K2}) — same graph, no rebuild");
    }

    // ── Phase 2 loop: identical replay, new parameters. ─────────────────────
    let target = total_phase1 + steps_phase2;
    while step < target {
        let next_u_in = u_out_co.into_inner();
        let next_v_in = v_out_co.into_inner();
        let next_u_out = u_in_co.into_inner();
        let next_v_out = v_in_co.into_inner();

        g.mutate_bind(UIn(next_u_in))?;
        g.mutate_bind(VIn(next_v_in))?;
        g.mutate_bind(UOut(next_u_out))?;
        g.mutate_bind(VOut(next_v_out))?;

        let next = g.sync(ctx)?;
        u_in_co = next.0;
        v_in_co = next.1;
        u_out_co = next.2;
        v_out_co = next.3;
        step += 1;
        if write_frames && step.is_multiple_of(1000) {
            println!("step {step}/{target}");
        }
    }

    let final_v = read_field(&v_out_co)?;
    if write_frames {
        let rgba = colorize_v(&final_v);
        claspr::write_ppm_rgba8("gray-scott-reconfigured.ppm", W as u32, H as u32, &rgba)?;
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
        "gray-scott: {W}x{H} grid, {} phase-1 + {} phase-2 steps; the per-step graph is built \
         ONCE and replayed (meta-kernel).",
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
