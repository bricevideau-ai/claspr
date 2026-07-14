//! Engine-level proof for the unified `Tag(value)`/`Tag(pipe)` constructor, the
//! consuming set-once `bind` / `call` verbs, and the [`SlotState::FedByPipe`] slot
//! state.
//!
//! Three interlocking pieces are exercised here:
//!
//! - **`bind` / `call`** — CONSUMING, INFALLIBLE, mixed value-or-feed set-once binds.
//!   They return the OWNED graph (so they chain and compose inside `and_then`) and
//!   DEFER every bind error to `sync` (an absent / unbound tag surfaces there as
//!   `SlotUnbound`/`SlotNoSuchTag`, a conflicting set-once as `SlotConflict`, with
//!   nothing enqueued — the atomicity guarantee). The fluent, EAGER-error verbs are
//!   now `mutate_bind` / `mutate_call` (the reuse-loop set/change verbs); the
//!   set-once verbs no longer have an eager form.
//! - **the unified tag constructor** — `Tag(value)` binds a slot by value; the SAME
//!   `Tag(pipe)` (fed a `Pipe` instead of a value) WIRES a `slot!(Tag)` to an UPSTREAM
//!   pipe (a build-time `Handle`), installing `SlotState::FedByPipe` so the slot reads
//!   whatever the upstream produced each run. The fed slot resolves DEFERRED (drains
//!   the pipe at run time) and RE-ARMS every replay (the upstream refills the pipe).
//!   There is no separate `feed(Tag, pipe)` verb — the pipe source IS the tag ctor.
//! - **currying** — `call` binds ONLY the tags in its tuple, leaving the rest open
//!   for a later `bind` / `call` (bind a subset now, the rest later).
//!
//! Uses the portable `add_u32` (3-output) / `scale_u32` (in-place, single-output)
//! test kernels — NOT gray-scott.

use claspr::eager::{DeviceOpExt, Pipe, download, upload};
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

/// Allocate + fill a `DeviceSlice<u32>` of `N` elements with `v`.
fn seeded(ctx: &Context, v: u32) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::alloc_zero(ctx, N)
        .expect("alloc")
        .fill(v)
        .wait()
        .expect("seed")
}

// Tags for the slot positions used below.
slots! {
    A: DeviceSlice<u32>,
    B: DeviceSlice<u32>,
    Out: DeviceSlice<u32>,
    Buf: DeviceSlice<u32>,
    // A tag the graph never declares — used to prove `call` DEFERS an absent
    // tag to sync instead of erroring eagerly.
    Absent: DeviceSlice<u32>,
    // Downstream slot fed from an upstream pipe.
    Dst: DeviceSlice<u32>,
}

/// (1) `call` binds a graph's value slots, `sync` produces the correct data,
/// and the owned return chains further.
#[test]
fn call_binds_and_syncs() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // out = a + b, all three as value slots, bound in ONE consuming call.
    // The owned graph is then `.and_then(download)`-chained (proving owned-return).
    let g = ks
        .add_u32([N], slot!(A), slot!(B), slot!(Out))
        .call((A(seeded(&ctx, 2)), B(seeded(&ctx, 5)), Out(seeded(&ctx, 0))))
        // `add_u32`'s Handle is a 3-tuple of pipes; download the `out` pipe.
        .and_then(|(_a, _b, out)| download(out));

    let out = g.sync(&ctx).expect("call sync");
    assert!(
        out.iter().all(|&v| v == 7),
        "call((A=2,B=5,Out=0)) then add: 2 + 5 = 7, got {:?}",
        &out[..8]
    );
}

/// (2) `call` an ABSENT tag → NO eager error (it is consuming + infallible); the
/// error is DEFERRED to `sync`, with NOTHING enqueued.
///
/// NOTE: this graph is ALSO left with B/Out unbound, so `sync` has two deferred
/// reasons to fail — the RECORDED `SlotNoSuchTag(Absent)` (record-don't-drop, the
/// silent-swallow fix) and the completeness `SlotUnbound` for B/Out. Either is a
/// correct "fails closed at sync, nothing ran" catch; the recorded absent-tag error
/// (the more precise diagnosis of the typo) is what surfaces. Before the fix the
/// absent tag was silently dropped.
#[test]
fn call_defers_absent_tag_to_sync() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Bind only `Absent` (a tag NOT in the graph) plus real A — leave B/Out unbound.
    // `call` RECORDS the SlotNoSuchTag(Absent) into the deferred sink and cannot
    // bind B/Out; it is infallible, so NOTHING errors HERE (deferred to sync).
    let g = ks
        .add_u32([N], slot!(A), slot!(B), slot!(Out))
        .call((A(seeded(&ctx, 2)), Absent(seeded(&ctx, 9))));

    // The deferred catch: sync fails closed (nothing enqueued). The recorded
    // absent-tag error surfaces (a completeness SlotUnbound would also be acceptable).
    match g.sync(&ctx) {
        Ok(_) => panic!("sync must fail: Absent recorded + B/Out never bound (deferred)"),
        Err(Error::SlotNoSuchTag(_)) => { /* expected: recorded absent-tag catch */ }
        Err(Error::SlotUnbound(_)) => { /* also acceptable: completeness catch */ }
        Err(e) => panic!("expected SlotNoSuchTag/SlotUnbound at sync, got {e:?}"),
    }

    // The single-tag `bind` form of the SAME absent tag likewise DEFERS (infallible):
    // build it, then assert `sync` surfaces the recorded SlotNoSuchTag.
    let g2 = ks
        .add_u32([N], slot!(A), slot!(B), slot!(Out))
        .bind(A(seeded(&ctx, 2)))
        .bind(Absent(seeded(&ctx, 9)));
    match g2.sync(&ctx) {
        Ok(_) => panic!("sync must fail: Absent recorded via bind + B/Out unbound (deferred)"),
        Err(Error::SlotNoSuchTag(_)) => { /* expected: recorded absent-tag catch */ }
        Err(Error::SlotUnbound(_)) => { /* also acceptable: completeness catch */ }
        Err(e) => panic!("expected SlotNoSuchTag/SlotUnbound at sync, got {e:?}"),
    }
}

/// (3) Wire a downstream slot to an UPSTREAM pipe via the `Tag(pipe)` constructor; the
/// slot resolves from the pipe, and re-running RE-ARMS the fed slot (same result each
/// replay).
///
/// The upstream is a MINT (`upload` re-seeds a fresh buffer each run) then a ×2
/// scale, producing a buffer into its Handle pipe; the downstream `scale_u32(slot!
/// (Dst), 3)` reads that pipe (FedByPipe) and scales again. scale is in-place, so the
/// fed pipe carries the upstream buffer THROUGH: the whole graph computes 5*2*3 = 30.
/// Because the upstream RE-SEEDS each run (no compounding), re-running gives the SAME
/// result — the fed slot re-arms every replay, proving the FedByPipe re-arm property.
#[test]
fn feed_slot_from_pipe_resolves_deferred_and_rearms() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Upstream: mint fresh 5s each run, scale ×2 -> 10. Feed its output pipe into a
    // SEPARATELY-built downstream subgraph's `slot!(Dst)` hole, which scales ×3 -> 30.
    let g = upload(vec![5u32; N])
        .and_then(|buf| ks.scale_u32([N], buf, 2u32)) // upstream: 5 -> 10 (re-seeded)
        .and_then(|up_pipe: Pipe<DeviceSlice<u32>>| {
            // Downstream reads the upstream buffer via the fed slot, scales ×3.
            ks.scale_u32([N], slot!(Dst), 3u32)
                .call((Dst(up_pipe),)) // Dst := FedByPipe(up_pipe)
                .and_then(download)
        });

    // Run 1: 5 * 2 * 3 = 30.
    let r1 = g.sync(&ctx).expect("feed run 1");
    assert!(
        r1.iter().all(|&v| v == 30),
        "feed: 5 * 2 * 3 = 30, got {:?}",
        &r1[..8]
    );
    drop(r1);

    // Run 2: the fed slot RE-ARMS (upstream refills its pipe each run) — identical
    // result (the upstream mint re-seeds, so no compounding).
    let r2 = g.sync(&ctx).expect("feed run 2 (re-arm)");
    assert!(
        r2.iter().all(|&v| v == 30),
        "feed re-arm: replay must MATCH run 1 (30), got {:?}",
        &r2[..8]
    );
}

/// (4) A `FedByPipe` slot passes `check_ready` (it is satisfied-by-upstream, NOT
/// `SlotUnbound`) even though it is not `Bound`. If the FedByPipe check_ready arm
/// were wrong (treating it like Unbound), the atomicity pre-pass in `sync` would
/// reject the graph before running. A clean `sync` proves check_ready accepted it.
#[test]
fn feed_slot_check_ready_ok() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Downstream Dst is fed; there is NO value bind on Dst. Only a passing
    // check_ready on the FedByPipe slot lets sync proceed.
    let g = ks
        .scale_u32([N], slot!(Buf), 4u32)
        .call((Buf(seeded(&ctx, 1)),)) // 1 -> 4
        .and_then(|up_pipe: Pipe<DeviceSlice<u32>>| {
            ks.scale_u32([N], slot!(Dst), 1u32) // ×1: pass-through, isolates readiness
                .call((Dst(up_pipe),))
                .and_then(download)
        });

    let out = g.sync(&ctx).expect("FedByPipe slot must pass check_ready");
    assert!(
        out.iter().all(|&v| v == 4),
        "feed check_ready: 1 * 4 * 1 = 4, got {:?}",
        &out[..8]
    );
}

/// (5) The double-buffer CROSSED feed: two downstream slots each fed from a distinct
/// upstream pipe, minimally. Two independent upstream scales produce two pipes; a
/// downstream `add_u32(slot!(A), slot!(B), slot!(Out))` reads BOTH via feeds (A ←
/// pipe_x, B ← pipe_y), plus a value-bound Out. Proves two FedByPipe slots coexist
/// and both re-arm on replay.
#[test]
fn crossed_feed_swap() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Two independent MINT upstreams (upload re-seeds each run, so no compounding):
    // x = 3*2 = 6, y = 4*5 = 20. Fed into add's A and B. out = A + B = 6 + 20 = 26,
    // into the value-bound Out. `Out`'s seed is minted up front and MOVED into the
    // inner closure (the closures must not borrow `ctx`, which `sync(&ctx)` needs).
    let seed_out = seeded(&ctx, 0);
    let g = upload(vec![3u32; N])
        .and_then(|b| ks.scale_u32([N], b, 2u32)) // pipe_x: 3 -> 6 (re-seeded)
        .and_then(move |pipe_x: Pipe<DeviceSlice<u32>>| {
            upload(vec![4u32; N])
                .and_then(|b| ks.scale_u32([N], b, 5u32)) // pipe_y: 4 -> 20 (re-seeded)
                .and_then(move |pipe_y: Pipe<DeviceSlice<u32>>| {
                    // Downstream add reads A ← pipe_x, B ← pipe_y; Out value-bound.
                    // `add_u32`'s Handle is a 3-tuple of pipes; destructure to the
                    // `out` pipe and download it.
                    ks.add_u32([N], slot!(A), slot!(B), slot!(Out))
                        .call((A(pipe_x), B(pipe_y), Out(seed_out)))
                        .and_then(|(_a, _b, out)| download(out))
                })
        });

    // Run 1.
    let r1 = g.sync(&ctx).expect("crossed feed run 1");
    assert!(
        r1.iter().all(|&v| v == 26),
        "crossed feed: (3*2) + (4*5) = 26, got {:?}",
        &r1[..8]
    );
    drop(r1);

    // Run 2 — both fed slots re-arm; identical result.
    let r2 = g.sync(&ctx).expect("crossed feed run 2 (re-arm both)");
    assert!(
        r2.iter().all(|&v| v == 26),
        "crossed feed re-arm: replay must MATCH (26), got {:?}",
        &r2[..8]
    );
}

/// (7) SILENT-SWALLOW REGRESSION — a CONFLICTING set-once `call` onto an
/// already-`Bound` slot must be RECORDED and surfaced at `sync` (record-don't-drop),
/// NOT dropped so the graph runs with the OLD value.
///
/// `call` is set-once (folds through `bind`). Binding `Buf` once leaves it
/// `Bound`; a SECOND `call` of a DIFFERENT buffer onto the same slot is a
/// `SlotConflict`. Before the fix that error was `let _ =`-dropped and, because the
/// cell stayed `Bound` to the first value, `check_ready` passed and the graph RAN
/// with the OLD value — the conflicting bind vanished silently. Now the deferred sink
/// carries the `SlotConflict` and `sync` fails closed with NOTHING enqueued.
#[test]
fn call_conflict_surfaces_at_sync_not_silent() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // scale_u32(slot!(Buf), 4): Buf is a single in-place slot. Bind it to a buffer of
    // 1s (valid), then CONFLICT-bind it to a buffer of 9s via a second set-once
    // call. The graph is otherwise complete (a satisfiable slot), so the OLD
    // silent-swallow path would have run 1*4 = 4.
    let g = ks
        .scale_u32([N], slot!(Buf), 4u32)
        .bind(Buf(seeded(&ctx, 1))) // Buf := Bound(1s)
        .bind(Buf(seeded(&ctx, 9))) // CONFLICT: set-once onto Bound → recorded
        .and_then(download);

    match g.sync(&ctx) {
        Ok(out) => panic!(
            "sync must FAIL on the conflicting set-once bind (silent-swallow bug); \
             instead it ran and produced {:?}",
            &out[..8]
        ),
        // The deferred-recorded conflict (or a deferred variant) surfaces here.
        Err(Error::SlotConflict(_)) => { /* expected: recorded, surfaced at sync */ }
        Err(e) => panic!("expected SlotConflict at sync, got {e:?}"),
    }
}

/// (7b) STICKY / POISON — a recorded deferred error POISONS the graph: a SECOND
/// `sync` WITHOUT rebinding re-reports the SAME error (check_ready PEEKS, does not
/// drain), and a freshly-rebuilt graph works (rebuild is the recovery). This is the
/// contract that closes the former report-once wart.
#[test]
fn deferred_error_is_sticky_rebuild_recovers() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Build an errored graph (conflicting set-once bind), as in (7).
    let g = ks
        .scale_u32([N], slot!(Buf), 4u32)
        .bind(Buf(seeded(&ctx, 1)))
        .bind(Buf(seeded(&ctx, 9))) // CONFLICT → recorded, poisons g
        .and_then(download);

    // First sync fails closed.
    match g.sync(&ctx) {
        Err(Error::SlotConflict(_)) => {}
        other => panic!("sync 1 must fail with SlotConflict, got {other:?}"),
    }
    // STICKY: a second sync WITHOUT rebinding re-reports the SAME error (peek, not
    // pop). The old report-once (pop) path would have found an empty sink and run.
    match g.sync(&ctx) {
        Err(Error::SlotConflict(_)) => { /* expected: poison is sticky */ }
        Ok(out) => panic!(
            "sync 2 must STILL fail (sticky poison); instead it ran and produced {:?}",
            &out[..8]
        ),
        Err(e) => panic!("sync 2 must re-report SlotConflict, got {e:?}"),
    }

    // RECOVERY = rebuild a fresh graph (empty sinks). This one is valid and runs.
    let fresh = ks
        .scale_u32([N], slot!(Buf), 4u32)
        .bind(Buf(seeded(&ctx, 3)))
        .and_then(download);
    let out = fresh.sync(&ctx).expect("rebuilt graph recovers");
    assert!(
        out.iter().all(|&v| v == 12),
        "rebuild recovers: 3 * 4 = 12, got {:?}",
        &out[..8]
    );
}

/// (7c) A FAILED `mutate_*` does NOT poison the graph — the fluent mutate verbs fail
/// EAGERLY at the call site and never touch the deferred-error sink, so the graph
/// stays reusable. Contrast (7b): only the infallible `bind`/`call` path records.
#[test]
fn failed_mutate_does_not_poison() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks
        .scale_u32([N], slot!(Buf), 4u32)
        .bind(Buf(seeded(&ctx, 2)));

    // A mutate of an ABSENT tag errors EAGERLY (fluent, &self) and returns the error
    // at the call site — nothing is recorded into any sink. (`mutate_bind` returns
    // `Result<&Op>` whose `Op` isn't `Debug`, so map to the error before matching.)
    match g.mutate_bind(Absent(seeded(&ctx, 9))).map(|_| ()) {
        Err(Error::SlotNoSuchTag(_)) => { /* eager, at the call site */ }
        other => panic!("mutate_bind of an absent tag must error eagerly, got {other:?}"),
    }

    // The graph is UNPOISONED: it still syncs correctly (Buf is validly bound to 2s).
    let out = g
        .and_then(download)
        .sync(&ctx)
        .expect("graph unpoisoned after failed mutate");
    assert!(
        out.iter().all(|&v| v == 8),
        "unpoisoned graph runs: 2 * 4 = 8, got {:?}",
        &out[..8]
    );
}

/// (8) SILENT-SWALLOW REGRESSION — an ABSENT/typo'd tag in `call` while the
/// REAL slots ARE satisfiable must be RECORDED and surfaced at `sync` as
/// `SlotNoSuchTag`, NOT dropped so the graph silently runs.
///
/// Before the fix, an absent tag was dropped and — if every real slot happened to be
/// bound — `check_ready` passed and the graph RAN, hiding the typo. The absent tag
/// has NO cell of its own, so it is recorded onto the first real slot's sink and
/// drained by `check_ready`.
#[test]
fn call_absent_tag_surfaces_when_real_slots_satisfied() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // add_u32(A, B, Out): bind ALL THREE real slots (fully satisfiable) PLUS a typo'd
    // Absent tag in the SAME call. The old path would drop Absent and run 2+5=7.
    let g = ks
        .add_u32([N], slot!(A), slot!(B), slot!(Out))
        .call((
            A(seeded(&ctx, 2)),
            B(seeded(&ctx, 5)),
            Out(seeded(&ctx, 0)),
            Absent(seeded(&ctx, 9)), // absent: recorded as SlotNoSuchTag
        ))
        .and_then(|(_a, _b, out)| download(out));

    match g.sync(&ctx) {
        Ok(out) => panic!(
            "sync must FAIL on the absent tag even though the real slots are bound \
             (silent-swallow bug); instead it ran and produced {:?}",
            &out[..8]
        ),
        Err(Error::SlotNoSuchTag(_)) => { /* expected: recorded, surfaced at sync */ }
        Err(e) => panic!("expected SlotNoSuchTag at sync, got {e:?}"),
    }
}

/// (9) REGRESSION — a fully VALID `call` still syncs correctly after the
/// record-don't-drop change (the sink stays empty, so `check_ready` is unaffected).
#[test]
fn call_valid_still_syncs_after_fix() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks
        .add_u32([N], slot!(A), slot!(B), slot!(Out))
        .call((A(seeded(&ctx, 3)), B(seeded(&ctx, 4)), Out(seeded(&ctx, 0))))
        .and_then(|(_a, _b, out)| download(out));

    // Run twice: proves the empty-sink reuse path is intact (re-sync unchanged).
    let r1 = g.sync(&ctx).expect("valid call sync 1");
    assert!(r1.iter().all(|&v| v == 7), "3 + 4 = 7, got {:?}", &r1[..8]);
    drop(r1);
    let r2 = g.sync(&ctx).expect("valid call sync 2 (re-arm)");
    assert!(
        r2.iter().all(|&v| v == 7),
        "re-sync must match: 7, got {:?}",
        &r2[..8]
    );
}

/// (6) `bind` currying: bind a SUBSET of the graph's slots via `bind`, then
/// bind the REST via a second `call`, then `sync` — correct data. Proves
/// call-is-partial (unbound slots left open for a later bind).
#[test]
fn bind_currying() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Bind A now (single-tag `bind`), then B + Out later (`call`). All three needed
    // by sync; the partial first bind leaves B/Out open, filled by the second.
    let g = ks
        .add_u32([N], slot!(A), slot!(B), slot!(Out))
        .bind(A(seeded(&ctx, 10))) // partial: only A
        .call((B(seeded(&ctx, 20)), Out(seeded(&ctx, 0)))) // the rest
        // `add_u32`'s Handle is a 3-tuple of pipes; download the `out` pipe.
        .and_then(|(_a, _b, out)| download(out));

    let out = g.sync(&ctx).expect("curried sync");
    assert!(
        out.iter().all(|&v| v == 30),
        "bind(A=10) then call(B=20,Out): 10 + 20 = 30, got {:?}",
        &out[..8]
    );
}
