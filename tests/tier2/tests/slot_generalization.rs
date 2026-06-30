//! Generalized typed slots — beyond today's buffer/image kernel-arg slots, a
//! `slot!(Tag)` now also fills:
//!
//! - **(A) SCALAR args** — `slot!(Factor)` in a `factor: u32` position. A scalar
//!   slot is NOT a resource: it has no `cl_mem`, is `Copy`, is never handed back
//!   (no `Checkout`/lend/sever). It rides the SIMPLE two-state cell
//!   (`Unbound`/`Bound`) with bind-idempotency by **value** equality, and at
//!   execute it is READ (cloned) — so a bound scalar persists across replays.
//! - **(B) LAUNCH args** — `slot!(Grid)` in the grid position, `Tag::Value =
//!   LaunchSpec`. Same non-resource two-state path; re-dispatch the SAME graph at
//!   a different extent across replays without rebuilding it.
//! - **(C) SHARED slots** — one tag at MANY positions, filled by ONE `bind`. A
//!   clone-able value (scalar / `LaunchSpec` / `Arc<DeviceSlice>`) fans out into
//!   every matching cell; a move-only single-site buffer slot still moves once
//!   (unchanged — see the regression test at the end + the `graph_slots` suite).
//!
//! Each test asserts real device behavior (skips silently with no OpenCL device).

use claspr::eager::{DeviceOpExt, download};
use claspr::{Context, DeviceSlice, Error, LaunchSpec};
use claspr::{slot, slots};
use claspr_test_kernels::kernels;

const N: usize = 64;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

slots! {
    // Scalar slots (Tag::Value is the scalar type — Copy, value-equality).
    Factor: u32,
    K: u32,
    // Launch slot (Tag::Value = LaunchSpec — Copy, geometry-equality).
    Grid: claspr::LaunchSpec,
    // Buffers used as bind targets / shared operands.
    Buf: DeviceSlice<u32>,
    BufA: DeviceSlice<u32>,
    BufB: DeviceSlice<u32>,
    // Read-only shared buffers fanned across sites via Arc::clone.
    Shared: std::sync::Arc<DeviceSlice<u32>>,
    SharedB: std::sync::Arc<DeviceSlice<u32>>,
    OutA: DeviceSlice<u32>,
    OutB: DeviceSlice<u32>,
    // A move-only single-site buffer slot (regression — must NOT require Clone).
    MoveOnly: DeviceSlice<u32>,
    // Zero-match `bind` regression: a tag the graph DOES use vs one it does NOT.
    Present: u32,
    Absent: u32,
}

/// Extract the [`Error`] from a `bind`/`mutate_bind` result, asserting it failed.
/// (The Ok arm is `&Op`, which is not `Debug`.)
fn bind_err<G>(r: claspr::Result<&G>, msg: &str) -> Error {
    match r {
        Ok(_) => panic!("{msg}"),
        Err(e) => e,
    }
}

/// Allocate + fill a `DeviceSlice<u32>` of `N` elements with `v`.
fn seeded(ctx: &Context, v: u32) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::alloc_zero(ctx, N)
        .expect("alloc")
        .fill(v)
        .wait()
        .expect("seed")
}

// ── (A) SCALAR-arg slots ────────────────────────────────────────────────────

/// (1) **Scalar slot bind.** `scale_u32([N], buf, slot!(Factor))`; binding the
/// factor with `bind(Factor(2))` scales ×2, and re-running over the re-armed graph
/// is idempotent in the factor (the scalar is READ, not consumed, so the second
/// run sees the SAME factor — `scale` compounds the buffer, not the factor).
#[test]
fn scalar_slot_bind_then_sync() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // First, a download-terminated graph to assert the bound factor scales ×2.
    let dl = ks
        .scale_u32([N], slot!(Buf), slot!(Factor))
        .and_then(download);
    let out = dl
        .bind(Buf(seeded(&ctx, 3)))
        .expect("bind buf")
        .bind(Factor(2u32))
        .expect("bind factor")
        .sync(&ctx)
        .expect("sync");
    assert!(
        out.iter().all(|&v| v == 6),
        "3 * 2 = 6, got {:?}",
        &out[..8]
    );

    // Now an in-place graph re-run: drop the first Checkout UNREAD to re-arm the
    // buffer slot; the factor slot is READ again (still 2), so 3 -> 6 -> 12. The
    // final 12 is only correct if the scalar factor persisted across the replay.
    let g = ks.scale_u32([N], slot!(Buf), slot!(Factor));
    let co1 = g
        .bind(Buf(seeded(&ctx, 3)))
        .expect("bind buf")
        .bind(Factor(2u32))
        .expect("bind factor")
        .sync(&ctx)
        .expect("run 1");
    drop(co1); // re-arm (Lent -> Bound), no read/sever

    let co2 = g.sync(&ctx).expect("re-run over re-armed graph");
    let mut r2 = vec![0u32; N];
    co2.read(&mut r2).wait().expect("read 2");
    assert!(
        r2.iter().all(|&v| v == 12),
        "re-read scalar factor 2 over compounded buffer 3 -> 6 -> 12, got {:?}",
        &r2[..8]
    );
}

/// (2) **Scalar mutate across replays** — the meta-kernel-ish reuse. The SAME
/// graph runs with a DIFFERENT scalar each pass via `mutate_bind(Factor(f))`,
/// proving a launch's by-value config is reconfigurable without rebuilding.
#[test]
fn scalar_slot_mutate_across_replays() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // scale -> download: each pass mints a fresh buffer + a fresh factor.
    let g = ks
        .scale_u32([N], slot!(Buf), slot!(Factor))
        .and_then(download);

    // First an explicit mutate to 3, asserting ×3.
    let out = g
        .mutate_bind(Buf(seeded(&ctx, 5)))
        .expect("mutate buf")
        .mutate_bind(Factor(3u32))
        .expect("mutate factor")
        .sync(&ctx)
        .expect("sync ×3");
    assert!(
        out.iter().all(|&v| v == 15),
        "5 * 3 = 15, got {:?}",
        &out[..8]
    );

    // Then the loop: same graph, a different scalar each run.
    for f in [2u32, 3u32, 4u32] {
        let out = g
            .mutate_bind(Buf(seeded(&ctx, 10)))
            .expect("mutate buf in loop")
            .mutate_bind(Factor(f))
            .expect("mutate factor in loop")
            .sync(&ctx)
            .expect("loop sync");
        let want = 10 * f;
        assert!(
            out.iter().all(|&v| v == want),
            "loop f={f}: 10 * {f} = {want}, got {:?}",
            &out[..8]
        );
    }
}

/// (3) **Scalar bind idempotency + conflict.** `bind(Factor(2))` twice is a clean
/// no-op (value equality); `bind(Factor(9))` over a Bound `Factor(2)` is
/// `SlotConflict`; `mutate_bind(Factor(9))` then changes it.
#[test]
fn scalar_slot_idempotent_and_conflict() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks
        .scale_u32([N], slot!(Buf), slot!(Factor))
        .and_then(download);

    // Bind the same value twice — idempotent, not a conflict.
    g.bind(Factor(2u32)).expect("first scalar bind");
    g.bind(Factor(2u32))
        .expect("second bind of the SAME scalar value is an idempotent no-op");

    // A different value via `bind` → SlotConflict (set-once contract, by value).
    let err = bind_err(
        g.bind(Factor(9u32)),
        "different-value scalar bind must conflict",
    );
    assert!(
        matches!(err, Error::SlotConflict(n) if n.contains("Factor")),
        "expected SlotConflict naming Factor, got {err:?}"
    );

    // `mutate_bind` overwrites to 9; the new factor drives the result.
    let out = g
        .bind(Buf(seeded(&ctx, 2)))
        .expect("bind buf")
        .mutate_bind(Factor(9u32))
        .expect("mutate_bind changes a bound scalar")
        .sync(&ctx)
        .expect("sync after mutate");
    assert!(
        out.iter().all(|&v| v == 18),
        "mutated factor drives result: 2 * 9 = 18, got {:?}",
        &out[..8]
    );
}

/// (4) **Scalar completeness.** An unbound scalar slot makes `sync` return
/// `SlotUnbound`, naming the tag — exactly like an unbound buffer slot.
#[test]
fn scalar_slot_unbound_sync_errors() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Buffer bound, factor NOT — the scalar slot is the only hole.
    let g = ks
        .scale_u32([N], slot!(Buf), slot!(Factor))
        .and_then(download);
    g.bind(Buf(seeded(&ctx, 3))).expect("bind buf");

    let err = g
        .sync(&ctx)
        .expect_err("unbound scalar slot must error at sync");
    assert!(
        matches!(err, Error::SlotUnbound(n) if n.contains("Factor")),
        "expected SlotUnbound naming Factor, got {err:?}"
    );

    // Bind it → the same graph runs.
    let out = g
        .bind(Factor(2u32))
        .expect("bind factor")
        .sync(&ctx)
        .expect("now complete");
    assert!(
        out.iter().all(|&v| v == 6),
        "3 * 2 = 6, got {:?}",
        &out[..8]
    );
}

// ── (B) LAUNCH-arg slots ────────────────────────────────────────────────────

/// (5) **Launch-arg slot.** `global_id_u32(slot!(Grid), data)` writes `gid.x` into
/// `data[gid.x]`, so ONLY the dispatched prefix is written. Binding two different
/// grids across replays (`[N]` then `[N/2]`) makes the written prefix length track
/// the bound grid — observable proof the dispatched extent is the slot's value.
#[test]
fn launch_slot_redispatch_at_different_grid() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    const SENTINEL: u32 = 0xDEAD_BEEF;
    let g = ks.global_id_u32(slot!(Grid), slot!(Buf));

    // Pass 1: dispatch at full [N] over a sentinel-seeded buffer.
    let co = g
        .mutate_bind(Grid(LaunchSpec::from([N])))
        .expect("bind grid [N]")
        .mutate_bind(Buf(seeded(&ctx, SENTINEL)))
        .expect("bind buf")
        .sync(&ctx)
        .expect("sync [N]");
    let mut r = vec![0u32; N];
    co.read(&mut r).wait().expect("read pass1");
    // Every index in 0..N was written to its own gid.
    assert!(
        (0..N).all(|i| r[i] == i as u32),
        "grid [N] writes the whole buffer to its gids, got {:?}",
        &r[..8]
    );

    // Pass 2: same graph, re-mutate the grid to [N/2] over a fresh sentinel buffer.
    let half = N / 2;
    let co2 = g
        .mutate_bind(Grid(LaunchSpec::from([half])))
        .expect("re-bind grid [N/2]")
        .mutate_bind(Buf(seeded(&ctx, SENTINEL)))
        .expect("re-bind buf")
        .sync(&ctx)
        .expect("sync [N/2]");
    let mut r2 = vec![0u32; N];
    co2.read(&mut r2).wait().expect("read pass2");
    // Prefix 0..N/2 written; the suffix N/2..N keeps the sentinel — the dispatched
    // extent shrank with the bound grid.
    assert!(
        (0..half).all(|i| r2[i] == i as u32),
        "grid [N/2] writes the prefix to its gids, got {:?}",
        &r2[..8]
    );
    assert!(
        (half..N).all(|i| r2[i] == SENTINEL),
        "grid [N/2] leaves the suffix UNwritten (sentinel), got idx {}={}",
        half,
        r2[half]
    );
}

// ── (C) SHARED slots — one tag, many positions, one bind fills ALL ───────────

/// (6) **Shared scalar slot (one tag, two sites).** Two `scale_u32` stages chained
/// via `and_then`, BOTH reading `slot!(K)` in their scalar position. ONE
/// `bind(K(v))` fills BOTH — proven by the compounded result (×v then ×v).
#[test]
fn shared_scalar_slot_fans_out() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // scale(buf, K) -> scale(same buf, K) -> download. The buffer threads through
    // (the first stage's Checkout output feeds the second), and BOTH stages read K.
    let g = ks
        .scale_u32([N], slot!(Buf), slot!(K))
        .and_then(|buf| ks.scale_u32([N], buf, slot!(K)))
        .and_then(download);

    // ONE bind of K fills the scalar slot at BOTH dispatch sites.
    let out = g
        .bind(Buf(seeded(&ctx, 2)))
        .expect("bind buf")
        .bind(K(3u32))
        .expect("one bind fills both K sites")
        .sync(&ctx)
        .expect("sync");
    // 2 * 3 * 3 = 18 — only correct if BOTH sites saw K=3.
    assert!(
        out.iter().all(|&v| v == 18),
        "shared K=3 at two sites: 2*3*3 = 18, got {:?}",
        &out[..8]
    );
}

/// (7) **Shared launch slot (one tag, two dispatch sites)** — the motivating case.
/// Two `global_id_u32` dispatches in one graph, both via `slot!(Grid)`. One
/// `bind(Grid(g))` sets BOTH; both dispatch at the bound extent — and we surface
/// BOTH sites' output buffers from the SINGLE fanned graph and assert each.
///
/// The composition is the NATURAL `bundle2(siteA, siteB)` — the two dispatch
/// sites are the bundle's two branches directly, slots and all. This relies on
/// `Bundle*::bind_slots` recursing into EVERY branch (mirroring
/// `AndThen::bind_slots`): a `slot!(Grid)` placed *inside* a bundle branch IS now
/// reached by `bind`, so one fan-out `bind(Grid(..))` fills both branches' grid
/// cells. (Before that fix the bundle inherited the no-op default `bind_slots`,
/// the slots stayed unbound, and `sync` errored `SlotUnbound` — which is exactly
/// what `slot_in_bundle_branch_is_bound` below now regression-guards.) The
/// bundle's `gather_checkouts` surfaces both branches' buffers as a
/// `(Checkout, Checkout)` tuple directly.
#[test]
fn shared_launch_slot_fans_out() {
    use claspr::eager::bundle2;

    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    const SENTINEL: u32 = 0xDEAD_BEEF;
    let half = N / 2;

    // Two dispatch sites sharing ONE grid slot, as the two branches of a bundle.
    // The fan-out binder reaches both branches (Bundle::bind_slots recurses), so
    // one `bind(Grid(..))` fills both sites; the bundle terminal yields BOTH
    // buffers as a `(Checkout, Checkout)` tuple.
    let g = bundle2(
        ks.global_id_u32(slot!(Grid), slot!(BufA)),
        ks.global_id_u32(slot!(Grid), slot!(BufB)),
    );

    // ONE bind of Grid fills BOTH sites at [N/2]; the bundle delivers both outputs.
    let (a_co, b_co) = g
        .bind(Grid(LaunchSpec::from([half])))
        .expect("one bind fills both Grid sites")
        .bind(BufA(seeded(&ctx, SENTINEL)))
        .expect("bind A")
        .bind(BufB(seeded(&ctx, SENTINEL)))
        .expect("bind B")
        .sync(&ctx)
        .expect("sync shared grid");

    // Both buffers come straight out of the one fanned graph — no proxy re-run.
    let mut ra = vec![0u32; N];
    let mut rb = vec![0u32; N];
    a_co.read(&mut ra).wait().expect("read A");
    b_co.read(&mut rb).wait().expect("read B");

    // Site A dispatched at [N/2]: prefix written to its gids, suffix kept sentinel.
    assert!(
        (0..half).all(|i| ra[i] == i as u32),
        "shared grid: site A prefix written, got {:?}",
        &ra[..8]
    );
    assert!(
        (half..N).all(|i| ra[i] == SENTINEL),
        "shared grid: site A suffix untouched (so it dispatched at the bound [N/2])"
    );

    // Site B dispatched at [N/2] too: same prefix/suffix shape from the same bind.
    assert!(
        (0..half).all(|i| rb[i] == i as u32),
        "shared grid: site B prefix written, got {:?}",
        &rb[..8]
    );
    assert!(
        (half..N).all(|i| rb[i] == SENTINEL),
        "shared grid: site B suffix untouched (so it dispatched at the bound [N/2])"
    );

    // Belt-and-braces: both sites took the SAME bound grid, so their buffers match.
    assert_eq!(
        ra, rb,
        "both Grid sites dispatched identically at the one bound grid"
    );
}

/// (8) **Shared move-only buffer via Arc.** `Tag::Value = Arc<DeviceSlice>` used
/// read-only at TWO kernel sites; ONE `bind(Shared(arc))` fans the SAME `cl_mem`
/// out (Arc::clone) to both. Each site adds the shared operand into its own out —
/// and we surface BOTH sites' out buffers from the SINGLE fanned graph and assert
/// each == 12, so each site provably saw BOTH shared operands.
///
/// Composition is the NATURAL `bundle2(siteA, siteB)` now that
/// `Bundle*::bind_slots` recurses into every branch (like test 7). `add_u32` is
/// multi-output (`(a, b, out)`), so each branch ends on `.and_then(|(_, _, out)|
/// forward(out))` to project just its out buffer — keeping each branch
/// single-output so the bundle's Checkouts are a clean `(Checkout, Checkout)` over
/// the two out buffers. The shared-operand `slot!`s live *inside* the bundle
/// branches; the fan-out binder reaches them through the bundle recursion.
#[test]
fn shared_arc_buffer_fans_out() {
    use claspr::eager::{bundle2, forward};

    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    use std::sync::Arc;
    // Two `add` sites, each reading the SAME two Arc operands (Shared=7, SharedB=5)
    // and writing its own out buffer. `add_u32`'s two read operands accept
    // `Arc<DeviceSlice>`, so both operand slots are clone-able → one bind each fans
    // out across both branches. Each branch projects its out via `forward` so the
    // bundle delivers both outs as a `(Checkout, Checkout)` tuple.
    let shared = Arc::new(seeded(&ctx, 7));
    let shared_b = Arc::new(seeded(&ctx, 5));

    let g = bundle2(
        ks.add_u32([N], slot!(Shared), slot!(SharedB), slot!(OutA))
            .and_then(|(_a, _b, oa)| forward(oa)),
        ks.add_u32([N], slot!(Shared), slot!(SharedB), slot!(OutB))
            .and_then(|(_a, _b, ob)| forward(ob)),
    );

    // ONE bind of each Arc operand fills BOTH add sites (Arc::clone fan-out); the
    // bundle delivers both outs as a `(Checkout, Checkout)` tuple.
    let (out_a, out_b) = g
        .bind(Shared(Arc::clone(&shared)))
        .expect("one bind fills both Shared sites")
        .bind(SharedB(Arc::clone(&shared_b)))
        .expect("one bind fills both SharedB sites")
        .bind(OutA(seeded(&ctx, 0)))
        .expect("bind outA")
        .bind(OutB(seeded(&ctx, 0)))
        .expect("bind outB")
        .sync(&ctx)
        .expect("sync two-site");

    // Both out buffers come straight out of the one fanned graph — no proxy re-run.
    let mut ra = vec![0u32; N];
    let mut rb = vec![0u32; N];
    out_a.read(&mut ra).wait().expect("read A");
    out_b.read(&mut rb).wait().expect("read B");

    // Site A computed 7 + 5 = 12 — only possible if BOTH its operand slots were
    // filled by the single bind that also fed site B.
    assert!(
        ra.iter().all(|&v| v == 12),
        "shared Arc operands fanned to site A: 7 + 5 = 12, got {:?}",
        &ra[..8]
    );
    // Site B likewise — both shared operands reached the terminal site.
    assert!(
        rb.iter().all(|&v| v == 12),
        "shared Arc operands fanned to site B: 7 + 5 = 12, got {:?}",
        &rb[..8]
    );
}

/// (9) **Regression: move-only single-site buffer slot.** `Tag::Value =
/// DeviceSlice` (NOT Clone) still works exactly as before — the shared-slot
/// generalisation must NOT force `Clone` on move-only buffers nor break the
/// take-once move path. (The full move-only matrix lives in `graph_slots`; this
/// guards the specific "didn't regress" property in the generalized file.)
#[test]
fn move_only_single_site_buffer_slot_unchanged() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // A bare DeviceSlice slot (move-only) at a single in-place site, plain scalar
    // factor (NOT a slot) — the exact pre-generalisation shape.
    let g = ks.scale_u32([N], slot!(MoveOnly), 2u32);

    let co1 = g
        .bind(MoveOnly(seeded(&ctx, 4)))
        .expect("move-only single-site bind still works")
        .sync(&ctx)
        .expect("run 1");
    drop(co1); // re-arm (Lent -> Bound), no read/sever

    // Re-runs over the re-armed move-only slot compound (4 -> 8 -> 16), proving the
    // four-state resource machine still drives it.
    let co2 = g.sync(&ctx).expect("re-run move-only");
    let mut r2 = vec![0u32; N];
    co2.read(&mut r2).wait().expect("read 2");
    assert!(
        r2.iter().all(|&v| v == 16),
        "in-place move-only slot compounds 4 -> 8 -> 16, got {:?}",
        &r2[..8]
    );
}

// ── (D) Bundle branches carry slots (Bug 1) + zero-match bind errors (Bug 2) ──

/// (10) **Regression: a `slot!` inside a `bundle2` branch IS bound.** This is the
/// direct Bug-1 guard: before `Bundle*::bind_slots` recursed into its branches, a
/// `slot!(Buf)` / `slot!(Factor)` placed inside a bundle branch was unreachable by
/// `bind` (the bundle inherited the no-op default), so this graph would error
/// `SlotUnbound` at `sync`. With the fix one `bind` of each tag reaches into the
/// branch and the graph runs correctly. Branch B is a trivial second branch so the
/// bundle is a *real* bundle (≥2 branches) while branch A holds the slots.
#[test]
fn slot_in_bundle_branch_is_bound() {
    use claspr::eager::bundle2;

    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Branch A: scale a slot buffer by a slot factor. Branch B: an independent
    // global_id write (its own concrete buffer, no slots) so the bundle has two
    // real branches and A's slots live strictly *inside* a bundle branch.
    let g = bundle2(
        ks.scale_u32([N], slot!(Buf), slot!(Factor)),
        ks.global_id_u32([N], seeded(&ctx, 0)),
    );

    // One bind of each tag must reach INTO branch A through the bundle.
    let (a_co, _b_co) = g
        .bind(Buf(seeded(&ctx, 3)))
        .expect("bind Buf reaches into the bundle branch")
        .bind(Factor(4u32))
        .expect("bind Factor reaches into the bundle branch")
        .sync(&ctx)
        .expect("bundle-branch slots all bound → sync runs (would SlotUnbound before the fix)");

    let mut ra = vec![0u32; N];
    a_co.read(&mut ra).wait().expect("read A");
    assert!(
        ra.iter().all(|&v| v == 12),
        "branch-A slots bound through the bundle: 3 * 4 = 12, got {:?}",
        &ra[..8]
    );
}

/// (11) **Regression: a `bind` of a tag matching ZERO cells is a hard error.**
/// The graph uses only `slot!(Present)`; `bind(Absent(..))` matches no cell, so —
/// per the AT-LEAST-ONE rule — it returns `Error::SlotNoSuchTag` naming `Absent`
/// (rather than silently succeeding and only surfacing as `SlotUnbound` at sync).
/// `bind(Present(..))` still succeeds (it matches its one cell).
#[test]
fn bind_absent_tag_errors() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // A graph whose ONLY scalar slot is `Present` (the buffer is concrete).
    let g = ks.scale_u32([N], seeded(&ctx, 1), slot!(Present));

    // Binding a tag the graph never uses → SlotNoSuchTag naming Absent.
    let err = bind_err(
        g.bind(Absent(7u32)),
        "binding a tag absent from the graph must hard-error",
    );
    assert!(
        matches!(err, Error::SlotNoSuchTag(n) if n.contains("Absent")),
        "expected SlotNoSuchTag naming Absent, got {err:?}"
    );

    // The tag that IS present still binds cleanly.
    g.bind(Present(2u32))
        .expect("binding the present tag still succeeds");
}

/// (12) **Fan-out across two bundle branches, one bind.** The SAME `slot!(K)`
/// appears in BOTH `bundle2` branches; one `bind(K(v))` (a clone-able scalar →
/// fan-out) fills both — proving the bundle recursion visits EVERY branch, not
/// just the first. Each branch scales its own seeded buffer by K; both outputs
/// reflect the single bound K.
#[test]
fn fan_out_across_bundle_branches() {
    use claspr::eager::bundle2;

    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Both branches read slot!(K); each has its own concrete buffer seeded
    // differently (5 and 6) so the two outputs are distinct yet both ×K.
    let g = bundle2(
        ks.scale_u32([N], seeded(&ctx, 5), slot!(K)),
        ks.scale_u32([N], seeded(&ctx, 6), slot!(K)),
    );

    // ONE bind of K must fan out into BOTH branches (no early stop at branch 1).
    let (a_co, b_co) = g
        .bind(K(3u32))
        .expect("one bind of a fan-out scalar fills K in BOTH bundle branches")
        .sync(&ctx)
        .expect("both branches' K bound → sync runs");

    let mut ra = vec![0u32; N];
    let mut rb = vec![0u32; N];
    a_co.read(&mut ra).wait().expect("read A");
    b_co.read(&mut rb).wait().expect("read B");
    assert!(
        ra.iter().all(|&v| v == 15),
        "branch A: 5 * 3 = 15 (got K from the single fan-out bind), got {:?}",
        &ra[..8]
    );
    assert!(
        rb.iter().all(|&v| v == 18),
        "branch B: 6 * 3 = 18 (got the SAME K from the single bind), got {:?}",
        &rb[..8]
    );
}
