//! Typed-slots + verb-2×2 proof test (step (b): `slots!` / `slot!` /
//! `g.bind` / `g.mutate_bind` / `g.call`).
//!
//! A reusable graph can carry **unbound typed holes** — `slot!(Tag)` — that plug
//! into the same positions a concrete buffer does (here: kernel buffer args).
//! Binding a slot folds a `TypeId → resource` binding into the graph's tri-state
//! slot cells. The set-binding verb 2×2 governs what happens at bind time:
//!
//! |               | `bind` (set-once)        | `mutate_bind` (set/change) |
//! |---------------|--------------------------|----------------------------|
//! | Unbound       | fill                     | fill                       |
//! | Bound, `==`   | no-op (idempotent)       | overwrite                  |
//! | Bound, `≠`    | `Err(SlotConflict)`      | overwrite                  |
//! | Lent          | `Err(SlotCheckedOut)`    | `Err(SlotCheckedOut)`      |
//! | Severed       | `Err(SlotSevered)`       | fill (re-arm)              |
//!
//! Equality is **buffer-handle identity** (`SlotEq`), not byte-equal contents.
//! Completeness (every slot bound) is enforced only at `sync` (runtime): an
//! unbound (or severed) slot is `Error::SlotUnbound`. After a run the Checkout
//! returns the buffer to its slot cell (re-arm `Lent → Bound`), so a bound graph
//! re-runs; `into_inner` severs it (`Lent → Severed`) — a state a set-once `bind`
//! rejects (`SlotSevered`) and only `mutate_bind` re-arms. These tests lock the
//! matrix plus the kept step-(b) properties:
//!
//! (a) `slot!` + `g.bind(Tag(v))?.sync()` produces correct data.
//! (b) order-free — chained `bind`s AND the tuple `call((A,B,Out))`.
//! (c) re-run a bound graph twice (the slot re-arms like a concrete cell).
//! (d) an unbound slot makes `sync` return the "slot unbound" `Err`.
//! (e) the verb 2×2: idempotent bind, conflict, mutate, checked-out, sever
//!     (`bind` after sever rejects with `SlotSevered`; `mutate_bind` re-arms).

use claspr::eager::{DeviceOpExt, download};
use claspr::{Context, DeviceSlice, Error};
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

// Tags for the slots used below. Each carries one fixed buffer type (compile-time);
// the tag type is the runtime identity key.
slots! {
    Buf: DeviceSlice<u32>,
    A: DeviceSlice<u32>,
    B: DeviceSlice<u32>,
    Out: DeviceSlice<u32>,
    // A shared read-only operand: an `Arc<DeviceSlice>` so two `Arc::clone`s name
    // the SAME buffer object — the only way to hand `bind` the identical handle
    // twice (a bare `DeviceSlice` isn't `Clone`). Used by the idempotency test.
    SharedA: std::sync::Arc<DeviceSlice<u32>>,
}

/// Extract the [`Error`] from a `bind`/`mutate_bind`/`call` result, asserting it
/// failed. The Ok arm is `&Op` (the graph), which is NOT `Debug`, so the usual
/// `expect_err` doesn't apply — drop the Ok value and pull the error out.
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

/// (a) `slot!(Buf)` in a kernel arg position; `g.bind(Buf(b))?.sync()` runs the
/// graph over the bound buffer and produces the expected data.
#[test]
fn slot_bind_then_sync_produces_data() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // scale(slot, 2) -> download. The slot is the in-place scale target.
    let g = ks.scale_u32([N], slot!(Buf), 2u32).and_then(download);

    let b = seeded(&ctx, 3); // 3 * 2 = 6
    let out = g
        .bind(Buf(b))
        .expect("bind")
        .sync(&ctx)
        .expect("bound sync");
    assert!(
        out.iter().all(|&v| v == 6),
        "scale(slot=3, 2) should be 6, got {:?}",
        &out[..8]
    );
}

/// (b) Binding is **order-free**: chained `bind`s in either order, AND the
/// turbofish-free tuple `call((A, B, Out))`, produce the same result.
#[test]
fn slot_bind_is_order_free() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // out = a + b, with all three operands as slots. `add_u32` is multi-output:
    // sync yields (Checkout<a>, Checkout<b>, Checkout<out>). Read `out`.
    let forward = ks.add_u32([N], slot!(A), slot!(B), slot!(Out));
    let (_a, _b, out_co) = forward
        .bind(A(seeded(&ctx, 2)))
        .expect("bind A")
        .bind(B(seeded(&ctx, 5)))
        .expect("bind B")
        .bind(Out(seeded(&ctx, 0)))
        .expect("bind Out")
        .sync(&ctx)
        .expect("forward-order sync");
    let mut r1 = vec![0u32; N];
    out_co.read(&mut r1).wait().expect("read forward");
    assert!(r1.iter().all(|&v| v == 7), "2 + 5 = 7, got {:?}", &r1[..8]);

    // Same graph shape, reverse bind order — identical result.
    let reverse = ks.add_u32([N], slot!(A), slot!(B), slot!(Out));
    let (_a, _b, out_co) = reverse
        .bind(Out(seeded(&ctx, 0)))
        .expect("bind Out")
        .bind(B(seeded(&ctx, 5)))
        .expect("bind B")
        .bind(A(seeded(&ctx, 2)))
        .expect("bind A")
        .sync(&ctx)
        .expect("reverse-order sync");
    let mut r2 = vec![0u32; N];
    out_co.read(&mut r2).wait().expect("read reverse");
    assert!(
        r2.iter().all(|&v| v == 7),
        "reverse bind order must match: 2 + 5 = 7, got {:?}",
        &r2[..8]
    );

    // Tuple `call((A, B, Out))` — one turbofish-free multi-fill, same result.
    let tupled = ks.add_u32([N], slot!(A), slot!(B), slot!(Out));
    let (_a, _b, out_co) = tupled
        .call((A(seeded(&ctx, 2)), B(seeded(&ctx, 5)), Out(seeded(&ctx, 0))))
        .expect("call")
        .sync(&ctx)
        .expect("call sync");
    let mut r3 = vec![0u32; N];
    out_co.read(&mut r3).wait().expect("read call");
    assert!(
        r3.iter().all(|&v| v == 7),
        "call((A,B,Out)) must match: 2 + 5 = 7, got {:?}",
        &r3[..8]
    );
}

/// (c) A bound graph is **re-runnable**: after a run, dropping the Checkout
/// returns the buffer to its slot cell (re-arm `Lent → Bound`), so a second
/// `sync` runs again. `scale` is in place, so it compounds over the same buffer
/// (3 -> 6 -> 12), proving the slot cell carried the buffer across runs.
#[test]
fn bound_slot_graph_reruns() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // No download: the output IS the buffer, so its Checkout re-arms the slot.
    let g = ks.scale_u32([N], slot!(Buf), 2u32);

    let b = seeded(&ctx, 3);
    // Run 1: 3 -> 6. Drop the Checkout to re-arm the slot.
    let co1 = g.bind(Buf(b)).expect("bind").sync(&ctx).expect("run 1");
    drop(co1);

    // Run 2 (already bound, slot re-armed): 6 -> 12, in place.
    let co2 = g.sync(&ctx).expect("run 2 over re-armed slot");
    let mut rb = vec![0u32; N];
    co2.read(&mut rb).wait().expect("read");
    assert!(
        rb.iter().all(|&v| v == 12),
        "in-place scale compounds over the re-armed slot buffer (3->6->12), got {:?}",
        &rb[..8]
    );
}

/// (d) Completeness is checked at `sync`: an UNBOUND slot makes `sync` return
/// `Error::SlotUnbound`, naming the tag's type.
#[test]
fn unbound_slot_sync_errors() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Never bound — the slot stays `Unbound`.
    let g = ks.scale_u32([N], slot!(Buf), 2u32).and_then(download);

    let err = g.sync(&ctx).expect_err("unbound slot must error at sync");
    match err {
        Error::SlotUnbound(name) => {
            // Clean tag ident only — exactly `Buf`, with no internal `<KeyMarker>`
            // source suffix leaking into user-facing text (review issue S3). This
            // slot name flows from `SlotHandle::new` (the primary leak site).
            assert_eq!(name, "Buf", "unbound-slot error should name exactly `Buf`");
            assert!(
                !name.contains("KeyMarker") && !name.contains('<'),
                "no KeyMarker / generic suffix in slot error, got {name:?}"
            );
        }
        other => panic!("expected Error::SlotUnbound, got {other:?}"),
    }

    // The same graph runs once bound — proves the slot, not the graph, was at fault.
    let out = g
        .bind(Buf(seeded(&ctx, 4)))
        .expect("bind")
        .sync(&ctx)
        .expect("now bound");
    assert!(
        out.iter().all(|&v| v == 8),
        "4 * 2 = 8, got {:?}",
        &out[..8]
    );
}

/// (e1) `bind` is **idempotent on an equal binding**: binding the SAME buffer
/// object (handle identity) twice is a clean no-op (no conflict), and the graph
/// still produces correct data. Uses an `Arc<DeviceSlice>` slot so two
/// `Arc::clone`s name the identical underlying buffer.
#[test]
fn bind_same_buffer_is_idempotent() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    use claspr::{MemRef, RecordableBuffer};
    use std::sync::Arc;

    /// Raw `cl_mem`/SVM pointer as a `usize`, for `==` identity across the run.
    fn handle_of<B: RecordableBuffer>(b: &B) -> usize {
        match b.record_handle().mem {
            MemRef::Buffer(m) => m as usize,
            MemRef::Svm(p) => p as usize,
        }
    }

    // out = sharedA + b. `a` is a read-only arg (accepts `Arc<DeviceSlice>`).
    let g = ks.add_u32([N], slot!(SharedA), slot!(B), slot!(Out));

    let shared = Arc::new(seeded(&ctx, 2));
    let h_shared = handle_of(&*shared); // the FIRST-bound buffer's identity.
    // First bind, then bind the SAME buffer again (a second `Arc::clone`): equal
    // handle → `bind` is idempotent, NOT a conflict.
    g.bind(SharedA(Arc::clone(&shared))).expect("first bind");
    g.bind(SharedA(Arc::clone(&shared)))
        .expect("second bind of the SAME buffer is an idempotent no-op");

    let (a_co, _b, out_co) = g
        .bind(B(seeded(&ctx, 5)))
        .expect("bind B")
        .bind(Out(seeded(&ctx, 0)))
        .expect("bind Out")
        .sync(&ctx)
        .expect("sync after idempotent rebind");

    // The duplicate `bind` must leave the binding UNCHANGED — the run's `a` operand
    // is the EXACT buffer the FIRST bind provided (same cl_mem), not a silently
    // re-installed equal one. (`a_co` derefs to `Arc<DeviceSlice>` → `DeviceSlice`.)
    assert_eq!(
        handle_of(&**a_co),
        h_shared,
        "idempotent rebind must keep the FIRST-bound buffer (unchanged), not replace it"
    );

    let mut r = vec![0u32; N];
    out_co.read(&mut r).wait().expect("read");
    assert!(
        r.iter().all(|&v| v == 7),
        "idempotent rebind still computes 2 + 5 = 7 over the SAME shared buffer, got {:?}",
        &r[..8]
    );
}

/// (e2) `bind` on a Bound slot with a **different** buffer is `Err(SlotConflict)`;
/// `mutate_bind` changes it; the new buffer drives the result.
#[test]
fn bind_conflict_then_mutate() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks.scale_u32([N], slot!(Buf), 2u32).and_then(download);

    // Bind to buffer a (seeded 3). Slot is now Bound(a) (download consumes the
    // value, but the slot's cell is re-armed only on Checkout drop — so bind the
    // slot WITHOUT running to keep it Bound for the conflict check).
    g.bind(Buf(seeded(&ctx, 3))).expect("first bind");

    // A different buffer via `bind` → conflict (set-once contract).
    let err = bind_err(
        g.bind(Buf(seeded(&ctx, 10))),
        "different-value bind must conflict",
    );
    match err {
        Error::SlotConflict(name) => assert!(
            name.contains("Buf"),
            "conflict should name the tag, got {name:?}"
        ),
        other => panic!("expected SlotConflict, got {other:?}"),
    }

    // `mutate_bind` overwrites the slot to the new buffer (seeded 10) — allowed.
    let out = g
        .mutate_bind(Buf(seeded(&ctx, 10)))
        .expect("mutate_bind changes a bound slot")
        .sync(&ctx)
        .expect("run after mutate");
    assert!(
        out.iter().all(|&v| v == 20),
        "mutate_bind's buffer drives the result: 10 * 2 = 20, got {:?}",
        &out[..8]
    );
}

/// (e3) `mutate_bind` on an **Unbound** slot fills it (no prior bind needed) — so
/// the loop idiom `for v in vs { mutate_bind, sync, assert }` works without
/// peeling the first iteration.
#[test]
fn mutate_bind_fills_unbound_in_loop() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks.scale_u32([N], slot!(Buf), 2u32).and_then(download);

    // First iteration: slot is Unbound; `mutate_bind` fills it. Subsequent
    // iterations: the slot was severed-back-to-Unbound by `into_inner` (download
    // mints a host Vec, so the slot buffer is checked out as part of... actually
    // download produces a Vec, the scale's buffer is the slot's value; its
    // Checkout re-arms the slot on drop). We re-bind a fresh buffer each pass with
    // `mutate_bind` (overwrite/fill), proving it never requires a prior `bind`.
    for v in [3u32, 4u32] {
        let out = g
            .mutate_bind(Buf(seeded(&ctx, v)))
            .expect("mutate_bind fills/overwrites without a prior bind")
            .sync(&ctx)
            .expect("loop sync");
        let want = v * 2;
        assert!(
            out.iter().all(|&x| x == want),
            "loop pass v={v}: {v} * 2 = {want}, got {:?}",
            &out[..8]
        );
    }
}

/// (e4) **Binding while checked out is a hard error.** Use an in-place op with no
/// download so the slot buffer IS the checked-out output. While the Checkout is
/// live, both `bind` and `mutate_bind` to a different buffer are
/// `Err(SlotCheckedOut)`; after `drop(co)` the slot re-arms and a rebind works.
#[test]
fn bind_while_checked_out_errors() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // In-place scale, NO download: the Checkout holds the slot's buffer (Lent).
    let g = ks.scale_u32([N], slot!(Buf), 2u32);

    let a = seeded(&ctx, 3);
    let co = g.bind(Buf(a)).expect("bind a").sync(&ctx).expect("run");
    // `co` is live → the slot is `Lent`.

    // Both verbs reject a re-bind while the buffer is in the caller's hands.
    let e_bind = bind_err(
        g.bind(Buf(seeded(&ctx, 9))),
        "bind while checked out must error",
    );
    assert!(
        matches!(e_bind, Error::SlotCheckedOut(n) if n.contains("Buf")),
        "expected SlotCheckedOut from bind, got {e_bind:?}"
    );
    let e_mut = bind_err(
        g.mutate_bind(Buf(seeded(&ctx, 9))),
        "mutate_bind while checked out must error",
    );
    assert!(
        matches!(e_mut, Error::SlotCheckedOut(n) if n.contains("Buf")),
        "expected SlotCheckedOut from mutate_bind, got {e_mut:?}"
    );

    // Drop the Checkout → slot re-arms (`Lent → Bound`). Now `mutate_bind` to a
    // fresh buffer is allowed and drives the result.
    drop(co);
    let out = g
        .mutate_bind(Buf(seeded(&ctx, 5)))
        .expect("rebind after drop works")
        .sync(&ctx)
        .expect("run after rebind");
    let mut rb = vec![0u32; N];
    out.read(&mut rb).wait().expect("read");
    assert!(
        rb.iter().all(|&v| v == 10),
        "rebound buffer after drop: 5 * 2 = 10, got {:?}",
        &rb[..8]
    );
}

/// (e5) **`into_inner` severs the slot to `Severed`.** After a run, taking the
/// value out (rather than re-arming) leaves the slot SEVERED — empty, but NOT
/// virgin: it was once bound and the caller deliberately kept its value. So the
/// set-once `bind` of a DIFFERENT buffer is now `Err(SlotSevered)` (re-providing a
/// buffer is a CHANGE, not a first declaration); only `mutate_bind` may re-arm it,
/// and the NEW buffer then drives the result.
///
/// (This replaces the old `into_inner_severs_slot_to_unbound`, which asserted a
/// plain `bind` after sever SUCCEEDS — the bug the 4th `Severed` state fixes.)
#[test]
fn into_inner_severs_slot_then_bind_rejected_mutate_rearms() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks.scale_u32([N], slot!(Buf), 2u32);

    let a = seeded(&ctx, 3);
    let co = g.bind(Buf(a)).expect("bind a").sync(&ctx).expect("run");
    // Sever: keep the value, leave the slot `Severed` (NOT `Lent`, NOT re-armed,
    // NOT virgin-`Unbound`).
    let kept = co.into_inner();
    drop(kept);

    // (a) A set-once `bind` of a DIFFERENT buffer must now REJECT — the slot is
    // Severed, so re-providing a buffer is a change, not a first declaration.
    let err = bind_err(
        g.bind(Buf(seeded(&ctx, 99))),
        "bind after sever must error (slot is Severed, not virgin)",
    );
    assert!(
        matches!(err, Error::SlotSevered(n) if n.contains("Buf")),
        "expected SlotSevered from bind after sever, got {err:?}"
    );

    // (b) `mutate_bind` of a DIFFERENT buffer re-arms the severed slot, and the
    // NEW buffer drives the result.
    let out = g
        .mutate_bind(Buf(seeded(&ctx, 7)))
        .expect("mutate_bind re-arms a severed slot")
        .sync(&ctx)
        .expect("run after re-arm");
    let mut rb = vec![0u32; N];
    out.read(&mut rb).wait().expect("read");
    assert!(
        rb.iter().all(|&v| v == 14),
        "post-sever mutate_bind's buffer drives the result: 7 * 2 = 14, got {:?}",
        &rb[..8]
    );
}

/// (e5b) **Regressions guarding the new `Severed` state.** Two properties that the
/// 4th state must NOT have broken or conflated:
/// - A **virgin** (`Unbound`) slot still accepts a plain set-once `bind` (adding
///   `Severed` must not turn the virgin-fill path into a rejection).
/// - A **severed** slot with NO new bind, re-sync'd, surfaces as
///   `Error::SlotUnbound` (a severed slot has nothing to lend — it must ERROR, not
///   silently run an empty slot).
#[test]
fn virgin_bind_ok_and_severed_resync_without_rebind_errors() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks.scale_u32([N], slot!(Buf), 2u32);

    // Virgin path: a plain `bind` on a never-bound slot fills it and runs.
    let co = g
        .bind(Buf(seeded(&ctx, 4)))
        .expect("virgin slot accepts a plain bind")
        .sync(&ctx)
        .expect("run over virgin-bound slot");
    // Sever it (keep the value), leaving the slot `Severed`.
    let kept = co.into_inner();
    drop(kept);

    // Severed + NO rebind: lending a severed slot has nothing to hand the run, so
    // a re-sync must ERROR (SlotUnbound), not silently run.
    match g.sync(&ctx) {
        Ok(_) => panic!("re-sync of a severed slot with no rebind must error"),
        Err(Error::SlotUnbound(n)) => assert!(
            n.contains("Buf"),
            "expected SlotUnbound naming Buf, got name {n:?}"
        ),
        Err(other) => panic!("expected SlotUnbound on severed-no-rebind sync, got {other:?}"),
    }
}

/// (e6) A **conflicting element inside a `call`** errors (the multi-fill inherits
/// the set-once contract per element).
#[test]
fn call_with_conflicting_element_errors() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks.add_u32([N], slot!(A), slot!(B), slot!(Out));

    // Pre-bind A to one buffer (no run, so A stays Bound).
    g.bind(A(seeded(&ctx, 1))).expect("pre-bind A");

    // A `call` whose A element conflicts with the existing A binding errors;
    // B/Out elements bind fine, but the conflicting A stops the fold.
    let err = bind_err(
        g.call((A(seeded(&ctx, 2)), B(seeded(&ctx, 5)), Out(seeded(&ctx, 0)))),
        "call with a conflicting element must error",
    );
    assert!(
        matches!(err, Error::SlotConflict(n) if n.contains("A")),
        "expected SlotConflict on A, got {err:?}"
    );
}

// ── B2: all-or-nothing (probe before sever) ─────────────────────────────────
//
// `call`/`mutate_call` are two-phase: PHASE 1 severs EVERY `Checkout` source
// (`into_value`), PHASE 2 folds. Before the phase-0 probe, a failure in phase 2 —
// or even a zero-match on the FIRST element — had ALREADY severed all sources.
// The probe (a read-only dry run over the SAME `bind_slots` walk, run BEFORE any
// `into_value`) makes the covered failures all-or-nothing: an absent tag, an
// externally-checked-out target, and a `Set`-onto-severed slot now return their
// error having severed / mutated NOTHING.
//
// The oracle for "a source was NOT severed": a `Checkout` sourced from a slot
// graph re-arms its slot on DROP (`Lent → Bound`). When the probe rejects, the
// tuple's Checkouts (moved into `mutate_call`) drop un-severed at the failed
// call's exit, so their source graphs re-`sync` cleanly. Had they been severed
// (`Lent → Severed`), the source's next `sync` would be `Err(SlotUnbound)`.

/// A single-output, re-armable source graph: `scale_u32(slot!(TAG), 1)` in place.
/// `sync` yields ONE `Checkout` whose home is the TAG slot's cell — dropping it
/// re-arms the slot, `into_inner`/binding-it-elsewhere severs it. `factor = 1`
/// leaves the data unchanged so the buffer's contents stay meaningful across the
/// sever-and-adopt swap.
macro_rules! scale_src {
    ($ks:expr, $tag:ident) => {
        $ks.scale_u32([N], slot!($tag), 1u32)
    };
}

/// B2 (1): a `mutate_call` whose FIRST element's tag is ABSENT from the target
/// graph must error `SlotNoSuchTag` WITHOUT severing ANY source. Both source
/// graphs must still `sync` afterward (their slots were re-armed on the dropped
/// Checkouts, NOT severed). This is the egregious "sever everything then
/// SlotNoSuchTag" bug the probe closes.
#[test]
fn call_absent_tag_does_not_sever_other_sources() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Two independent source graphs producing co_a (home = ga's A slot) and
    // co_b (home = gb's B slot); both slots are `Lent` while the Checkouts live.
    let ga = scale_src!(ks, A);
    let gb = scale_src!(ks, B);
    let co_a = ga
        .call((A(seeded(&ctx, 7)),))
        .expect("bind A")
        .sync(&ctx)
        .expect("ga sync");
    let co_b = gb
        .call((B(seeded(&ctx, 9)),))
        .expect("bind B")
        .sync(&ctx)
        .expect("gb sync");

    // Target graph carries `Buf` — NEITHER A NOR B. `mutate_call((A(co_a),
    // B(co_b)))` must reject on the FIRST (absent) element, A.
    let g = ks.scale_u32([N], slot!(Buf), 2u32);
    let err = bind_err(
        g.mutate_call((A(co_a), B(co_b))),
        "mutate_call with an absent first tag must error",
    );
    assert!(
        matches!(err, Error::SlotNoSuchTag(n) if n.contains("A")),
        "expected SlotNoSuchTag(A), got {err:?}"
    );

    // The proof: NEITHER source was severed. Both source graphs re-`sync`
    // cleanly (their slots re-armed `Lent → Bound` on the dropped Checkouts).
    // A severed slot would make these `Err(SlotUnbound)`.
    let co_a2 = ga
        .sync(&ctx)
        .expect("ga re-sync — A slot must be re-armed, not severed");
    let co_b2 = gb
        .sync(&ctx)
        .expect("gb re-sync — B slot must be re-armed, not severed");
    drop(co_a2);
    drop(co_b2);
}

/// B2 (2): a `mutate_call` whose element targets a slot that is CHECKED OUT by an
/// UNRELATED live `Checkout` (NOT one of this tuple's elements) must error
/// `SlotCheckedOut` WITHOUT severing the OTHER tuple element's source. The probe
/// distinguishes "Lent by a tuple Checkout (a crossed swap → OK)" from "Lent by
/// an external Checkout (→ SlotCheckedOut)" via the tuple's severable-cell ids.
#[test]
fn call_checked_out_element_does_not_sever_others() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Target graph with two slots A and B; sync once with both bound. Keep the
    // A Checkout ALIVE (external) so A stays `Lent`; re-arm B (drop its Checkout).
    let g = ks.add_u32([N], slot!(A), slot!(B), slot!(Out));
    let (a_live, b_co, out_co) = g
        .call((A(seeded(&ctx, 1)), B(seeded(&ctx, 2)), Out(seeded(&ctx, 0))))
        .expect("initial bind")
        .sync(&ctx)
        .expect("initial sync");
    drop(b_co); // B re-arms (Lent → Bound); `a_live` stays alive → A stays Lent.
    drop(out_co); // Out re-arms too.

    // An independent source graph producing co_b (home = gb's B slot), Lent.
    let gb = scale_src!(ks, B);
    let co_b = gb
        .call((B(seeded(&ctx, 5)),))
        .expect("bind gb B")
        .sync(&ctx)
        .expect("gb sync");

    // `mutate_call((A(seeded), B(co_b)))` on g: A is Lent by the EXTERNAL
    // `a_live` (its cell id is NOT among the tuple's severable cells), so the
    // probe rejects with SlotCheckedOut(A) BEFORE phase 1 severs co_b.
    let err = bind_err(
        g.mutate_call((A(seeded(&ctx, 3)), B(co_b))),
        "mutate_call onto an externally-checked-out slot must error",
    );
    assert!(
        matches!(err, Error::SlotCheckedOut(n) if n.contains("A")),
        "expected SlotCheckedOut(A) from the external checkout, got {err:?}"
    );

    // The proof: co_b's source (gb) was NOT severed — it re-`sync`s cleanly.
    let co_b2 = gb
        .sync(&ctx)
        .expect("gb re-sync — B slot must be re-armed, not severed");
    drop(co_b2);
    drop(a_live);
}

/// B2 (3): a focused restatement of the crossed-swap oracle — `mutate_call`'s
/// crossed rebind over two slots that are BOTH `Lent`, each by a Checkout IN THE
/// SAME tuple, succeeds (the probe recognises the swap and passes it). This is the
/// case the probe must NOT over-reject; `double_buffering` exercises the full
/// ping-pong, this locks the single crossing in isolation.
#[test]
fn mutate_call_crossed_swap_over_lent_by_tuple_checkout_succeeds() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // add_u32(In, Ones, Out): out = in + ones. Two data slots (In, Out) plus a
    // shared operand (B reused as "ones").
    let g = ks.add_u32([N], slot!(A), slot!(B), slot!(Out));
    let (in_co, ones_co, out_co) = g
        .call((
            A(seeded(&ctx, 10)),
            B(seeded(&ctx, 1)),
            Out(seeded(&ctx, 0)),
        ))
        .expect("bind step0")
        .sync(&ctx)
        .expect("sync step0");
    // Now A and Out slots are Lent (held by in_co / out_co — both in the swap
    // tuple below). Re-arm the shared operand.
    drop(ones_co);

    // THE CROSSED SWAP over Lent-by-tuple-checkout: A adopts the old Out, Out
    // adopts the old In. Both targets are `Lent`; the probe sees each target's
    // cell id IS in the tuple's severable set (the crossing Checkouts sever them
    // in phase 1) → OK, not SlotCheckedOut.
    g.mutate_call((A(out_co), Out(in_co)))
        .expect("crossed mutate_call over Lent-by-tuple-checkout must succeed");

    let (_a2, _ones2, out2) = g.sync(&ctx).expect("sync after swap");
    let mut result = vec![0u32; N];
    out2.read(&mut result).wait().expect("read post-swap");
    // Step 0 computed out = 10 + 1 = 11. After the swap A = old Out (=11),
    // Out = old In (=10); step 1 computes out = 11 + 1 = 12.
    assert!(
        result.iter().all(|&v| v == 12),
        "post-swap add should be 11 + 1 = 12, got {:?}",
        &result[..8]
    );
}
