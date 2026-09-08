//! **Command-buffer record-churn bench** — how many `cl_khr_command_buffer`s a
//! replayed Tier-2 graph CREATES, and what a `sync` costs, across three replay
//! shapes.
//!
//! Motivation (`NOTES.md`, "Graph mutation invalidates the recorded CB wholesale"):
//! the miniWeather port measured a graph re-recording its CB on EVERY replay when
//! scalar slots are `mutate_*`-ed between syncs — 30 steps produced 30
//! `clCreateCommandBufferKHR` and ~16k `clSetKernelArg`, ~30% of wall. A correct
//! keyed cache should record ONE CB per distinct binding state and then replay.
//!
//! The three workloads are the shapes that matter:
//!
//! | workload           | mutation between syncs        | models          |
//! |--------------------|-------------------------------|-----------------|
//! | `control`          | none                          | the good case   |
//! | `scalar-alternate` | one scalar slot, period 2     | miniWeather     |
//! | `buffer-pingpong`  | two buffer slots crossed      | gray-scott swap |
//!
//! `control` must stay at 1 record; the other two are the regression. Run:
//!
//! ```sh
//! OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors \
//!   cargo run -p claspr-tier2-tests --example cb_bench --release
//! ```

use claspr::eager::{DeviceOp, DeviceOpExt, bundle2, fill, forward};
use claspr::record::cb_records_total;
use claspr::{Context, DeviceSlice, slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;
use std::time::Instant;

slots! {
    Factor: u32,
    Src: DeviceSlice<u32>,
    Dst: DeviceSlice<u32>,
}

/// Buffer length, overridable via `CB_BENCH_N`. This bench measures HOST cost
/// (record churn + walk overhead), so device time must not dominate and mask it —
/// sweep this to find the regime where it doesn't. The port's 100x50 grid over 24
/// dispatches puts each dispatch in the low-thousands of elements.
fn n() -> usize {
    std::env::var("CB_BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024)
}
/// Replay count per workload. Large enough that a per-replay re-record is obvious
/// against a one-off record, small enough to stay quick on a CPU ICD.
/// Replays per workload, overridable via `CB_BENCH_ITERS`.
///
/// This used to be capped near 400 because sustained replay of one cached CB hit
/// a pocl race (`CL_INVALID_OPERATION` at a random iteration — the
/// `cb_replay_limit` example). Fixed upstream-ready in pocl
/// ("Command-buffers: retire the submission before publishing CL_COMPLETE"), and
/// 5000 replays run clean against a pocl carrying it. An OLDER pocl will still
/// abort the longer workloads at random — that is the runtime, not this bench.
fn iters() -> usize {
    std::env::var("CB_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000)
}
/// Dispatch count for the `-deep` workload — matched to the miniWeather trace in
/// `NOTES.md` (24 dispatches re-recorded per step). Re-record cost scales with
/// commands-per-CB, so the shallow 3-command workloads badly UNDER-state the win;
/// this is the shape the 30%-of-wall figure came from.
const DEEP: usize = 24;

/// One workload's measurement: how many CBs it created and what each sync cost.
struct Report {
    name: &'static str,
    records: u64,
    total: std::time::Duration,
    per_sync_us: Vec<f64>,
}

impl Report {
    fn print(&self) {
        let mut s = self.per_sync_us.clone();
        s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in timings"));
        let median = s[s.len() / 2];
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        let p95 = s[(s.len() as f64 * 0.95) as usize % s.len()];
        println!(
            "{:<18} records={:<5} wall={:>8.1}ms  per-sync mean={:>8.1}us \
             median={:>8.1}us p95={:>8.1}us",
            self.name,
            self.records,
            self.total.as_secs_f64() * 1e3,
            mean,
            median,
            p95,
        );
    }
}

/// Run `body` `ITERS` times, timing each call and bracketing the whole loop with a
/// [`cb_records_total`] delta. `body` takes the iteration index so a workload can
/// alternate its binding state.
fn measure(
    name: &'static str,
    mut body: impl FnMut(usize) -> claspr::Result<()>,
) -> claspr::Result<Report> {
    // One warm-up OUTSIDE the measured window: the first sync records the CB and
    // pays first-touch driver costs, which would otherwise swamp the median.
    body(0)?;

    let before = cb_records_total();
    let t0 = Instant::now();
    let mut per_sync_us = Vec::with_capacity(iters());
    for i in 0..iters() {
        let t = Instant::now();
        body(i)?;
        per_sync_us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let total = t0.elapsed();
    Ok(Report {
        name,
        records: cb_records_total() - before,
        total,
        per_sync_us,
    })
}

/// **control** — a weight-3 all-device chain replayed with NO mutation. The cached
/// CB stays valid for every replay, so this must record exactly once (in the
/// warm-up) and zero times inside the measured window. Any non-zero count here is a
/// straight cache bug, independent of the keying work.
fn control(ctx: &Context) -> claspr::Result<Report> {
    let ks = kernels::kernels(ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    let g = fill(buf, 2u32)
        .and_then(|b| ks.scale_u32([n()], b, 5u32))
        .and_then(|b| ks.scale_u32([n()], b, 1u32));
    measure("control", |_| {
        let co = g.sync(ctx)?;
        drop(co);
        Ok(())
    })
}

/// **scalar-alternate** — the miniWeather shape: one `u32` scalar slot flipped
/// between two values each sync. Only the baked scalar changes; the graph topology
/// and every `cl_mem` are identical, so a keyed cache should settle at TWO records
/// (one per value) and replay thereafter.
fn scalar_alternate(ctx: &Context) -> claspr::Result<Report> {
    let ks = kernels::kernels(ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    let g = fill(buf, 2u32)
        .and_then(|b| ks.scale_u32([n()], b, slot!(Factor)))
        .and_then(|b| ks.scale_u32([n()], b, 1u32));
    g.mutate_bind(Factor(3u32))?;
    measure("scalar-alternate", |i| {
        // Flip BEFORE the sync so every measured sync sees a freshly mutated slot.
        g.mutate_bind(Factor(if i % 2 == 0 { 3u32 } else { 4u32 }))?;
        let co = g.sync(ctx)?;
        drop(co);
        Ok(())
    })
}

/// **buffer-pingpong** — the gray-scott `run_swap` shape: two buffer slots crossed
/// each sync by binding each Checkout into the OTHER slot. The bound `cl_mem`s
/// alternate with period 2, so the state space is again two — but the mutation is a
/// buffer identity change, not a scalar, which is the second tier the keyed cache
/// has to cover.
fn buffer_pingpong(ctx: &Context) -> claspr::Result<Report> {
    let ks = kernels::kernels(ctx)?;
    let a = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    let b = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    let g = bundle2(
        ks.scale_u32([n()], slot!(Src), 1u32),
        ks.scale_u32([n()], slot!(Dst), 2u32),
    );
    g.mutate_call((Src(a), Dst(b)))?;
    measure("buffer-pingpong", |_| {
        let (src_co, dst_co) = g.sync(ctx)?;
        // Cross the checkouts: each slot ADOPTS the other's buffer (Lent → Severed
        // → Bound), exactly as gray-scott's `run_swap` does.
        g.mutate_call((Src(dst_co), Dst(src_co)))?;
        Ok(())
    })
}

/// Stack `.and_then(scale(slot!(Factor)))` once per extra token, on top of `$head`.
///
/// A `for` loop cannot build this: `and_then` is EAGER and returns a new type per
/// link, so a 24-deep chain is 24 distinct types. Token repetition is the only way
/// to get miniWeather's dispatch count without writing it out longhand. The closures
/// run at BUILD time (no `FnOnce` stored), so borrowing `$ks` across all of them is
/// fine.
///
/// (`fan_out` is the homogeneous alternative and now binds slots correctly, but a
/// CHAIN is the shape miniWeather actually has — 24 dependent dispatches, not 24
/// independent ones — so the chain is what this models.)
macro_rules! stack_scale {
    ($head:expr, $ks:ident $(, $tok:tt)*) => {
        $head $(
            .and_then(|b| { let _ = stringify!($tok); $ks.scale_u32([n()], b, slot!(Factor)) })
        )*
    };
}

/// **scalar-alternate-deep** — `scalar_alternate` at miniWeather's dispatch count.
/// A [`DEEP`]-command chain whose every kernel reads the SAME scalar slot (a scalar
/// `bind` fans out to every matching cell), so one `mutate_bind` dirties all 24
/// recorded commands — the exact shape that cost 30% of wall in the port. Same
/// two-state alternation, so a keyed cache should still settle at two records; what
/// differs from the shallow case is how much each avoided record is WORTH.
fn scalar_alternate_deep(ctx: &Context) -> claspr::Result<Report> {
    let ks = kernels::kernels(ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    // 1 fill + 23 scales = DEEP commands.
    let g = stack_scale!(
        fill(buf, 1u32),
        ks,
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        i,
        j,
        k,
        l,
        m,
        n,
        o,
        p,
        q,
        r,
        s,
        t,
        u,
        v,
        w
    );
    assert_eq!(
        g.cbable_weight(),
        DEEP,
        "deep chain should be DEEP commands"
    );
    g.mutate_bind(Factor(1u32))?;
    measure("scalar-alt-deep", |i| {
        g.mutate_bind(Factor(if i % 2 == 0 { 1u32 } else { 2u32 }))?;
        let co = g.sync(ctx)?;
        drop(co);
        Ok(())
    })
}

/// **control-deep** — the SAME [`DEEP`]-command graph as `scalar_alternate_deep`,
/// bound once and never mutated. This is the isolating control: `scalar-alt-deep`
/// minus `control-deep` is the per-sync cost of re-recording 24 commands, with
/// device work, graph depth and walk cost all held constant. Without this pair the
/// deep numbers only say "more commands is slower", which is not the claim.
fn control_deep(ctx: &Context) -> claspr::Result<Report> {
    let ks = kernels::kernels(ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    let g = stack_scale!(
        fill(buf, 1u32),
        ks,
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        i,
        j,
        k,
        l,
        m,
        n,
        o,
        p,
        q,
        r,
        s,
        t,
        u,
        v,
        w
    );
    g.mutate_bind(Factor(1u32))?;
    measure("control-deep", |_| {
        let co = g.sync(ctx)?;
        drop(co);
        Ok(())
    })
}

/// **scalar-rebind-same** — `mutate_bind` every sync, but always to the SAME value.
/// Isolates the two halves of the mutating workloads' overhead: the `mutate_bind`
/// walk itself (which a keyed cache CANNOT remove — it still has to run) versus the
/// re-record it triggers (which the cache CAN). If this costs the same as
/// `scalar-alternate`, today's invalidation is firing on a no-op rebind, and a
/// digest comparison would skip it for free.
fn scalar_rebind_same(ctx: &Context) -> claspr::Result<Report> {
    let ks = kernels::kernels(ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    let g = fill(buf, 2u32)
        .and_then(|b| ks.scale_u32([n()], b, slot!(Factor)))
        .and_then(|b| ks.scale_u32([n()], b, 1u32));
    g.mutate_bind(Factor(3u32))?;
    measure("scalar-rebind-same", |_| {
        g.mutate_bind(Factor(3u32))?;
        let co = g.sync(ctx)?;
        drop(co);
        Ok(())
    })
}

/// A no-op host seam: segments the graph without touching the data.
fn noop_seam(_v: &mut [u32]) -> claspr::Result<()> {
    Ok(())
}

/// Stack `.and_then(scale).and_then_host(noop_seam)` once per token — a chain that
/// alternates device work and host seams, so no single subtree is wholly addable.
macro_rules! stack_seam {
    ($head:expr, $ks:ident $(, $tok:tt)*) => {
        $head $(
            .and_then(|b| { let _ = stringify!($tok); $ks.scale_u32([n()], b, 1u32) })
            .and_then_host(noop_seam)
        )*
    };
}

/// **seam-chain** — the shape the predicate memo actually targets. A LEFT-deep chain
/// (`a.and_then(b).and_then(c)` nests to the left) alternating device kernels with
/// host seams. No subtree is wholly `cb_addable`, so the walk stays in `CbWalk::Off`
/// and re-runs the whole-subtree `cb_addable` / `contains_host_seam` recursions at
/// every level it descends — the O(n²) case. The fully-addable workloads above never
/// reach it: their root opens one boundary and every nested test short-circuits on
/// `Off`. No mutation here, so this isolates walk cost from record cost entirely.
fn seam_chain(ctx: &Context) -> claspr::Result<Report> {
    let ks = kernels::kernels(ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    let g = stack_seam!(fill(buf, 1u32), ks, a, b, c, d, e, f, g, h);
    measure("seam-chain", |_| {
        let co = g.sync(ctx)?;
        drop(co);
        Ok(())
    })
}

/// Stack `.and_then(forward)` once per token — zero-command passthrough nodes.
macro_rules! stack_forward {
    ($head:expr $(, $tok:tt)*) => {
        $head $( .and_then(|p| { let _ = stringify!($tok); forward(p) }) )*
    };
}

/// **walk-lean / walk-heavy** — the pair that isolates the REPLAY WALK.
///
/// Both graphs record the SAME two commands (a fill and a scale) over the same
/// buffer, so their device work, their command-buffer contents and their
/// per-sync enqueue/wait costs are identical. They differ only in how many
/// zero-command `forward` passthrough nodes the walk has to traverse: 0 vs 24.
///
/// So `(walk_heavy - walk_lean) / 24` is the marginal cost of one node in the
/// `LendOnly` replay walk, measured rather than inferred — which is the number
/// that decides whether skipping the walk on a cache hit is worth doing.
fn walk_lean(ctx: &Context) -> claspr::Result<Report> {
    let ks = kernels::kernels(ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    let g = fill(buf, 1u32).and_then(|b| ks.scale_u32([n()], b, 1u32));
    measure("walk-lean", |_| {
        drop(g.sync(ctx)?);
        Ok(())
    })
}

fn walk_heavy(ctx: &Context) -> claspr::Result<Report> {
    let ks = kernels::kernels(ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(ctx, n())?;
    // 24 passthroughs between the two commands: same CB, 24 more nodes to walk.
    let g = stack_forward!(
        fill(buf, 1u32),
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        i,
        j,
        k,
        l,
        m,
        n2,
        o,
        p,
        q,
        r,
        s,
        t,
        u,
        v,
        w,
        x
    )
    .and_then(|b| ks.scale_u32([n()], b, 1u32));
    measure("walk-heavy", |_| {
        drop(g.sync(ctx)?);
        Ok(())
    })
}

fn main() -> claspr::Result<()> {
    let Some(ctx) = ctx() else { return Ok(()) };
    println!(
        "device: {}  cl_khr_command_buffer={}",
        ctx.device().name().unwrap_or_else(|_| "?".into()),
        ctx.has_cl_khr_command_buffer(),
    );
    if !ctx.has_cl_khr_command_buffer() {
        eprintln!("SKIP: no cl_khr_command_buffer — nothing to measure");
        return Ok(());
    }
    println!("N={} iters={}\n", n(), iters());
    // Run and PRINT one at a time: a workload that errors should not take the
    // other six's numbers down with it (and the failure should name itself).
    type Workload = (&'static str, fn(&Context) -> claspr::Result<Report>);
    const WORKLOADS: &[Workload] = &[
        ("control", control),
        ("scalar-alternate", scalar_alternate),
        ("scalar-rebind-same", scalar_rebind_same),
        ("buffer-pingpong", buffer_pingpong),
        ("control-deep", control_deep),
        ("scalar-alt-deep", scalar_alternate_deep),
        ("seam-chain", seam_chain),
        ("walk-lean", walk_lean),
        ("walk-heavy", walk_heavy),
    ];
    // `CB_BENCH_ONLY=<substring>` restricts the run to matching workloads.
    let only = std::env::var("CB_BENCH_ONLY").unwrap_or_default();
    let mut failed = false;
    for (name, f) in WORKLOADS.iter().filter(|(n, _)| n.contains(&only)) {
        match f(&ctx) {
            Ok(r) => r.print(),
            Err(e) => {
                println!("{name:<18} FAILED: {e:?}");
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}
