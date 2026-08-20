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

use claspr::eager::{DeviceOpExt, download, write};
use claspr::{DeviceSlice, Error, LaunchSpec};
use claspr::{slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::{N, ctx, handle_of, seeded};

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
    // Sever-and-adopt: a source slot (yields a Checkout) and a distinct target slot
    // the Checkout is bound INTO.
    Src: DeviceSlice<u32>,
    Dst: DeviceSlice<u32>,
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
        .bind(Factor(2u32))
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
    // Set-once binds (consuming) folded into `g`; re-`sync`'d by `&` below.
    let g = ks
        .scale_u32([N], slot!(Buf), slot!(Factor))
        .bind(Buf(seeded(&ctx, 3)))
        .bind(Factor(2u32));
    let co1 = g.sync(&ctx).expect("run 1");
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
/// no-op (value equality); a set-once `bind(Factor(9))` over a Bound `Factor(2)` is
/// `SlotConflict`, now RECORDED and surfaced DEFERRED at `sync`; `mutate_bind(Factor(9))`
/// (fluent, EAGER) then changes it.
#[test]
fn scalar_slot_idempotent_and_conflict() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // The duplicate `bind(Factor(2))` must leave the bound VALUE unchanged (still 2),
    // not silently replaced by an equal-looking one. Prove it by RUNNING a graph whose
    // Factor is bound-then-idempotently-rebound to 2 (folded into one consuming
    // chain): the result must reflect factor 2 (5 * 2 = 10).
    let idem_out = ks
        .scale_u32([N], slot!(Buf), slot!(Factor))
        .and_then(download)
        .bind(Factor(2u32))
        .bind(Factor(2u32)) // idempotent no-op (same value)
        .bind(Buf(seeded(&ctx, 5)))
        .sync(&ctx)
        .expect("sync idempotent-value check");
    assert!(
        idem_out.iter().all(|&v| v == 10),
        "idempotent scalar rebind kept Factor=2 (unchanged): 5 * 2 = 10, got {:?}",
        &idem_out[..8]
    );

    // A different value via a set-once `bind` → `SlotConflict`. It is recorded and
    // surfaces DEFERRED at `sync`: the conflicting bind leaves the cell `Bound` to the
    // OLD value, so `check_ready` (state-first) sees it satisfiable and drains the
    // recorded conflict. Buf is bound too so the graph is otherwise complete.
    let conflict = ks
        .scale_u32([N], slot!(Buf), slot!(Factor))
        .and_then(download)
        .bind(Factor(2u32)) // Factor := Bound(2)
        .bind(Buf(seeded(&ctx, 2)))
        .bind(Factor(9u32)); // CONFLICT: set-once onto Bound(2) → recorded
    match conflict.sync(&ctx) {
        Ok(_) => panic!("different-value scalar set-once bind must fail at sync (deferred)"),
        Err(Error::SlotConflict(n)) => assert!(
            n.contains("Factor"),
            "expected SlotConflict naming Factor, got {n:?}"
        ),
        Err(other) => panic!("expected deferred SlotConflict, got {other:?}"),
    }

    // `mutate_bind` (fluent, EAGER) overwrites a bound scalar to 9; the new factor
    // drives the result.
    let g = ks
        .scale_u32([N], slot!(Buf), slot!(Factor))
        .and_then(download)
        .bind(Buf(seeded(&ctx, 2)))
        .bind(Factor(2u32)); // Factor := Bound(2)
    let out = g
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

    // Buffer bound, factor NOT — the scalar slot is the only hole. Set-once bind
    // (consuming) folded into `g`; `g` is re-`sync`'d / bound by `&`+move below.
    let g = ks
        .scale_u32([N], slot!(Buf), slot!(Factor))
        .and_then(download)
        .bind(Buf(seeded(&ctx, 3)));

    let err = g
        .sync(&ctx)
        .expect_err("unbound scalar slot must error at sync");
    assert!(
        matches!(err, Error::SlotUnbound(n) if n.contains("Factor")),
        "expected SlotUnbound naming Factor, got {err:?}"
    );

    // Bind it → the same graph runs.
    let out = g.bind(Factor(2u32)).sync(&ctx).expect("now complete");
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
        .bind(K(3u32)) // one bind fills both K sites
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
        .bind(Grid(LaunchSpec::from([half]))) // one bind fills both Grid sites
        .bind(BufA(seeded(&ctx, SENTINEL)))
        .bind(BufB(seeded(&ctx, SENTINEL)))
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
        .bind(Shared(Arc::clone(&shared))) // one bind fills both Shared sites
        .bind(SharedB(Arc::clone(&shared_b))) // one bind fills both SharedB sites
        .bind(OutA(seeded(&ctx, 0)))
        .bind(OutB(seeded(&ctx, 0)))
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
    // factor (NOT a slot) — the exact pre-generalisation shape. Set-once bind
    // (consuming) folded into `g`; re-`sync`'d by `&` below.
    let g = ks
        .scale_u32([N], slot!(MoveOnly), 2u32)
        .bind(MoveOnly(seeded(&ctx, 4)));
    let co1 = g.sync(&ctx).expect("run 1");
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
        .bind(Buf(seeded(&ctx, 3))) // reaches into the bundle branch
        .bind(Factor(4u32)) // reaches into the bundle branch
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
/// The graph uses only `slot!(Present)`; a set-once `bind(Absent(..))` matches no
/// cell, so — per the AT-LEAST-ONE rule — it RECORDS `Error::SlotNoSuchTag` naming
/// `Absent` (consuming, infallible) which surfaces DEFERRED at `sync` (rather than
/// silently succeeding). We bind the real `Present` tag too so the graph is otherwise
/// satisfiable and `check_ready` drains the recorded absent-tag error (an absent tag
/// has no cell of its own, so it is recorded onto the first real slot's sink).
/// `bind(Present(..))` alone still runs cleanly.
#[test]
fn bind_absent_tag_errors() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Bind the real `Present` scalar (satisfiable) PLUS a tag the graph never uses.
    // The absent-tag `SlotNoSuchTag(Absent)` is recorded and surfaces at `sync`.
    let g = ks
        .scale_u32([N], seeded(&ctx, 1), slot!(Present))
        .bind(Present(2u32))
        .bind(Absent(7u32));
    match g.sync(&ctx) {
        Ok(_) => panic!("binding a tag absent from the graph must fail at sync (deferred)"),
        Err(Error::SlotNoSuchTag(n)) => {
            // The diagnostic is the CLEAN tag ident — exactly `Absent`, with no
            // internal `<KeyMarker>` source suffix leaking into user-facing text
            // (review issue S3).
            assert_eq!(n, "Absent", "SlotNoSuchTag should name exactly `Absent`");
            assert!(
                !n.contains("KeyMarker"),
                "no `KeyMarker` in slot error: {n:?}"
            );
            assert!(!n.contains('<'), "no generic suffix in slot error: {n:?}");
        }
        Err(other) => panic!("expected deferred SlotNoSuchTag naming Absent, got {other:?}"),
    }

    // The tag that IS present binds and runs cleanly on its own.
    let _present_co = ks
        .scale_u32([N], seeded(&ctx, 1), slot!(Present))
        .bind(Present(2u32))
        .sync(&ctx)
        .expect("binding the present tag still runs");
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
        .bind(K(3u32)) // one fan-out bind fills K in BOTH bundle branches
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

// ── (E) Bind a Checkout into a slot — sever source, adopt into target ─────────

/// (13) **Bind a `Checkout` into a slot — sever-and-adopt.** A `Checkout` produced
/// by one graph's run is bound DIRECTLY into a slot of another graph (no manual
/// `into_inner`). Binding a Checkout:
/// - **severs** the Checkout's SOURCE home (`Lent → Severed`) — so a later set-once
///   `bind` on that source slot is `Error::SlotSevered`, and
/// - hands the raw buffer to the TARGET slot, which **adopts** it normally.
///
/// We assert all three: the source slot is severed, the target ran with the adopted
/// buffer producing correct data, and the buffer is the SAME `cl_mem` object (the
/// Checkout's buffer was moved, not copied).
#[test]
fn bind_checkout_into_slot_severs_source_adopts_target() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // SOURCE graph: scale `slot!(Src)` by 2. One run yields a Checkout<DeviceSlice>
    // whose home is the Src slot (Lent while the Checkout is alive). Set-once bind
    // (consuming) folded in; `src_graph` is re-`bind`'d (deferred error) below.
    let src_graph = ks
        .scale_u32([N], slot!(Src), 2u32)
        .bind(Src(seeded(&ctx, 5))); // 5*2=10
    let co = src_graph.sync(&ctx).expect("source run");
    let src_handle = handle_of(&*co); // identity of the buffer the Checkout holds

    // TARGET graph: scale `slot!(Dst)` by 3. Bind the Checkout DIRECTLY into Dst —
    // this severs Src's home and Dst adopts the buffer (no `into_inner` at the call).
    let dst_graph = ks.scale_u32([N], slot!(Dst), 3u32);
    let dst_co = dst_graph
        .bind(Dst(co)) // <-- Checkout bound into a slot: sever + adopt
        .sync(&ctx)
        .expect("target run");

    // (a) The SOURCE slot is now Severed: a plain set-once `bind` on it fails closed
    //     at `sync`. `bind` is infallible now, so the `SlotSevered` is RECORDED, but a
    //     bind onto a Severed cell leaves it Severed, so `check_ready` (state-first)
    //     reports the completeness `SlotUnbound` before draining the sink. Both are
    //     fail-closed catches — the point is Src was severed by the Checkout bind.
    let err = src_graph
        .bind(Src(seeded(&ctx, 1)))
        .sync(&ctx)
        .expect_err("Src must be Severed after its Checkout was bound away");
    assert!(
        matches!(&err, Error::SlotUnbound(n) | Error::SlotSevered(n) if n.contains("Src")),
        "expected state-first SlotUnbound / recorded SlotSevered naming Src, got {err:?}"
    );

    // (b) The TARGET ran with the adopted buffer: 10 * 3 = 30, and it is the SAME
    //     cl_mem the Checkout carried (moved, not copied).
    assert_eq!(
        handle_of(&*dst_co),
        src_handle,
        "Dst must hold the SAME buffer the Checkout carried (adopt = move, not copy)"
    );
    let mut r = vec![0u32; N];
    dst_co.read(&mut r).wait().expect("read Dst result");
    assert!(
        r.iter().all(|&v| v == 30),
        "adopted buffer (was 10) scaled by 3 = 30, got {:?}",
        &r[..8]
    );
}

// ── (D) Buffer slots in NON-KERNEL positions (fill/write/download/copy/…) ─────
//
// `slot!(Tag)` is documented to plug into `download`/`fill`/`write`/copy sources, not
// just kernel args — but every non-kernel leaf except the buffer copy used to skip
// `DeviceOp::bind_slots`, so a slot in those positions type-checked yet failed at
// runtime (`SlotNoSuchTag`). Every fill/write/download/transfer/acquire/image leaf now
// exposes its input(s) to the binder. These two exercise the `write` and `download`
// positions AND a re-run (correct rearming) — the class the kernel-arg tests above
// never reached.

/// A buffer slot in a `write()` position: `write(Buf, [5]) -> scale(*2) -> download`.
/// Bind `Buf`, run (5*2=10), then RE-RUN over the re-armed graph with a fresh `Buf`.
#[test]
fn buffer_slot_in_write_position_binds_and_rearms() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ks = &ks;

    let g = write(slot!(Buf).into_slot_input(), vec![5u32; N])
        .and_then(|b| ks.scale_u32([N], b, 2u32))
        .and_then(download);

    g.mutate_bind(Buf(seeded(&ctx, 0)))
        .expect("bind Buf in a write position");
    let co1 = g.sync(&ctx).expect("run 1");
    assert!(
        co1.iter().all(|&v| v == 10),
        "write 5 * 2 = 10, got {:?}",
        &co1[..8]
    );
    drop(co1);

    // Fresh Buf over the re-armed graph — proves the write-position slot rehomed and
    // rebinds cleanly (correct rearming), not just the first bind.
    g.mutate_bind(Buf(seeded(&ctx, 0))).expect("re-bind Buf");
    let co2 = g.sync(&ctx).expect("run 2 over re-armed graph");
    assert!(
        co2.iter().all(|&v| v == 10),
        "re-run write 5 * 2 = 10, got {:?}",
        &co2[..8]
    );
    drop(co2);
}

/// A buffer slot in a `download()` position: `download(Buf)` reads the bound buffer
/// straight back, and a mutate to a differently-seeded buffer reads the new content.
#[test]
fn buffer_slot_in_download_position_binds_and_rearms() {
    let Some(ctx) = ctx() else { return };

    let g = download(slot!(Buf).into_slot_input());

    g.mutate_bind(Buf(seeded(&ctx, 7)))
        .expect("bind Buf in a download position");
    let co = g.sync(&ctx).expect("run 1");
    assert!(
        co.iter().all(|&v| v == 7),
        "download Buf=7, got {:?}",
        &co[..8]
    );
    drop(co);

    g.mutate_bind(Buf(seeded(&ctx, 42)))
        .expect("re-bind Buf=42");
    let co = g.sync(&ctx).expect("run 2");
    assert!(
        co.iter().all(|&v| v == 42),
        "download re-bound Buf=42, got {:?}",
        &co[..8]
    );
    drop(co);
}
