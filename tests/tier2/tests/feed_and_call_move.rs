//! Engine-level proof for the `feed` / `call_move` / `bind_move` verbs and the
//! [`SlotState::FedByPipe`] slot state (promoted from spike #194).
//!
//! Three interlocking pieces are exercised here:
//!
//! - **`call_move`** — a CONSUMING, INFALLIBLE, mixed value-or-feed bind. It returns
//!   the OWNED graph (so it chains and composes inside `and_then`) and DEFERS every
//!   bind error to `sync` (an absent / unbound tag surfaces there as
//!   `SlotUnbound`/`SlotNoSuchTag`, with nothing enqueued — the atomicity guarantee).
//!   Contrast the fluent `call`, which errors EAGERLY.
//! - **`feed`** — wire a `slot!(Tag)` to an UPSTREAM pipe (a build-time `Handle`), so
//!   the slot reads whatever the upstream produced each run, installing
//!   `SlotState::FedByPipe`. The fed slot resolves DEFERRED (drains the pipe at run
//!   time) and RE-ARMS every replay (the upstream refills the pipe).
//! - **`bind_move`** — `call_move` used for currying (bind a subset now, the rest
//!   later); it IS `call_move` under a currying-flavoured name.
//!
//! Uses the portable `add_u32` (3-output) / `scale_u32` (in-place, single-output)
//! test kernels — NOT gray-scott.

use claspr::eager::{DeviceOpExt, Pipe, download, feed, upload};
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
    // A tag the graph never declares — used to prove `call_move` DEFERS an absent
    // tag to sync instead of erroring eagerly.
    Absent: DeviceSlice<u32>,
    // Downstream slot fed from an upstream pipe.
    Dst: DeviceSlice<u32>,
}

/// (1) `call_move` binds a graph's value slots, `sync` produces the correct data,
/// and the owned return chains further.
#[test]
fn call_move_binds_and_syncs() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // out = a + b, all three as value slots, bound in ONE consuming call_move.
    // The owned graph is then `.and_then(download)`-chained (proving owned-return).
    let g = ks
        .add_u32([N], slot!(A), slot!(B), slot!(Out))
        .call_move((A(seeded(&ctx, 2)), B(seeded(&ctx, 5)), Out(seeded(&ctx, 0))))
        // `add_u32`'s Handle is a 3-tuple of pipes; download the `out` pipe.
        .and_then(|(_a, _b, out)| download(out));

    let out = g.sync(&ctx).expect("call_move sync");
    assert!(
        out.iter().all(|&v| v == 7),
        "call_move((A=2,B=5,Out=0)) then add: 2 + 5 = 7, got {:?}",
        &out[..8]
    );
}

/// (2) `call_move` an ABSENT tag → NO eager error (it is infallible); the error is
/// DEFERRED to `sync` (`SlotUnbound` for the still-unbound real slots), with
/// NOTHING enqueued. Contrast the fluent `call`, which errors eagerly.
#[test]
fn call_move_defers_absent_tag_to_sync() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Bind only `Absent` (a tag NOT in the graph) plus real A — leave B/Out unbound.
    // call_move drops both the SlotNoSuchTag(Absent) and cannot bind B/Out; it is
    // infallible, so NOTHING errors here.
    let g = ks
        .add_u32([N], slot!(A), slot!(B), slot!(Out))
        .call_move((A(seeded(&ctx, 2)), Absent(seeded(&ctx, 9))));

    // The deferred catch: sync errors on the first still-unbound real slot.
    match g.sync(&ctx) {
        Ok(_) => panic!("sync must fail: B/Out were never bound (deferred error)"),
        Err(Error::SlotUnbound(_)) => { /* expected: deferred completeness catch */ }
        Err(e) => panic!("expected SlotUnbound at sync, got {e:?}"),
    }

    // And the fluent `call` errors EAGERLY on the same absent tag (the contrast).
    let g2 = ks.add_u32([N], slot!(A), slot!(B), slot!(Out));
    match g2.call((A(seeded(&ctx, 2)), Absent(seeded(&ctx, 9)))) {
        Ok(_) => panic!("fluent call must error eagerly on an absent tag"),
        Err(Error::SlotNoSuchTag(_)) => { /* expected: eager error */ }
        Err(e) => panic!("expected eager SlotNoSuchTag from call, got {e:?}"),
    }
}

/// (3) Wire a downstream slot to an UPSTREAM pipe via `feed`; the slot resolves from
/// the pipe, and re-running the graph RE-ARMS the fed slot (same result each replay).
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
                .call_move((feed(Dst, up_pipe),)) // Dst := FedByPipe(up_pipe)
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
        .call_move((Buf(seeded(&ctx, 1)),)) // 1 -> 4
        .and_then(|up_pipe: Pipe<DeviceSlice<u32>>| {
            ks.scale_u32([N], slot!(Dst), 1u32) // ×1: pass-through, isolates readiness
                .call_move((feed(Dst, up_pipe),))
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
                        .call_move((feed(A, pipe_x), feed(B, pipe_y), Out(seed_out)))
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

/// (6) `bind_move` currying: bind a SUBSET of the graph's slots via `bind_move`, then
/// bind the REST via a second `call_move`, then `sync` — correct data. Proves
/// call_move-is-partial (unbound slots left open for a later bind).
#[test]
fn bind_move_currying() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Bind A now (bind_move), then B + Out later (call_move). All three needed by
    // sync; the partial first bind leaves B/Out open, filled by the second.
    let g = ks
        .add_u32([N], slot!(A), slot!(B), slot!(Out))
        .bind_move((A(seeded(&ctx, 10)),)) // partial: only A
        .call_move((B(seeded(&ctx, 20)), Out(seeded(&ctx, 0)))) // the rest
        // `add_u32`'s Handle is a 3-tuple of pipes; download the `out` pipe.
        .and_then(|(_a, _b, out)| download(out));

    let out = g.sync(&ctx).expect("curried sync");
    assert!(
        out.iter().all(|&v| v == 30),
        "bind_move(A=10) then call_move(B=20,Out): 10 + 20 = 30, got {:?}",
        &out[..8]
    );
}
