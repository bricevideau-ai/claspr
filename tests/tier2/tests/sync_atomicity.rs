//! `sync` / `wait_on` ATOMICITY — a failed `sync` leaves the graph UNCHANGED and
//! re-runnable.
//!
//! ## The bug this locks down
//!
//! `sync` → `wait_on` walks the graph depth-first, and each leaf's `execute`
//! BOTH lends its input buffer (`Bound → Lent`) AND enqueues its device work in
//! the SAME call. So if a LATER node has an unsatisfiable input (empty concrete
//! cell / unbound-or-lent slot / unbound scalar), the EARLIER nodes have already
//! lent their buffers — and the failing `sync` strands those earlier cells `Lent`
//! with no `Checkout` to re-arm them. A retry then spuriously reports them busy.
//!
//! ## The fix
//!
//! A read-only `check_ready` pre-pass walks EVERY input cell of the whole graph
//! before any device work is enqueued. If any input is unsatisfiable, `sync`
//! returns that error having LENT nothing and ENQUEUED nothing — so the graph is
//! untouched and re-runnable. The error is the SAME variant/message the late
//! (execute-time) check produced, so existing "unbound slot ⇒ `SlotUnbound`"
//! tests are unaffected.
//!
//! These tests assert:
//!  1. an unbound slot in a LATER node leaves the EARLIER bound buffer untouched
//!     (same handle on the recovered run — never left `Lent`);
//!  2. a checked-out (`Lent`) slot in a later node is atomic — earlier cells
//!     untouched, graph recovers once the live `Checkout` drops;
//!  3. a fully-ready multi-node graph runs normally and produces correct data;
//!  4. the early-caught error is `SlotUnbound`, identical to the old execute-time
//!     failure.

use claspr::eager::{DeviceOpExt, bundle2};
use claspr::record::{MemRef, RecordableBuffer};
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

slots! {
    Late: DeviceSlice<u32>,
    Missing: DeviceSlice<u32>,
}

/// Allocate + fill a `DeviceSlice<u32>` of `N` elements with `v`.
fn seeded(ctx: &Context, v: u32) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::alloc_zero(ctx, N)
        .expect("alloc")
        .fill(v)
        .wait()
        .expect("seed")
}

/// The raw buffer-identity pointer behind a `DeviceSlice` — for asserting the
/// SAME `cl_mem` survived a failed-then-retried `sync` (it was never lent away).
fn handle_of(buf: &DeviceSlice<u32>) -> usize {
    match buf.record_handle().mem {
        MemRef::Buffer(mem) => mem as usize,
        MemRef::Svm(p) => p as usize,
    }
}

/// (1) An UNBOUND slot in a LATER node makes `sync` error WITHOUT touching the
/// EARLIER (bound concrete) node's buffer: it is never lent, so binding the
/// missing slot and re-`sync`ing succeeds AND the early buffer comes back with the
/// SAME handle (proving it was never left `Lent` by the failed attempt).
#[test]
fn unbound_slot_in_later_node_leaves_earlier_cells_untouched() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // EARLY branch: a CONCRETE bound buffer, scaled ×1 (in-place identity → its
    // Checkout returns the same buffer). LATE branch: an UNBOUND slot.
    let early = seeded(&ctx, 7);
    let early_handle = handle_of(&early);

    let g = bundle2(
        ks.scale_u32([N], early, 1u32),
        ks.scale_u32([N], slot!(Missing), 2u32),
    );

    // First sync: the later slot is unbound → `SlotUnbound`. With the atomicity
    // pre-pass this fires BEFORE the early branch lends its concrete buffer.
    let err = g
        .sync(&ctx)
        .expect_err("unbound later slot must error at sync");
    assert!(
        matches!(err, Error::SlotUnbound(n) if n.contains("Missing")),
        "expected SlotUnbound naming Missing, got {err:?}"
    );

    // Recover: bind the missing slot. If the failed attempt had left the early
    // concrete buffer `Lent` (cell emptied with no Checkout to re-arm it), THIS
    // `sync` would spuriously fail "graph busy". It must succeed.
    let (early_co, late_co) = g
        .bind(Missing(seeded(&ctx, 5)))
        .expect("bind Missing")
        .sync(&ctx)
        .expect("recovered sync — early buffer must NOT have been left Lent");

    // The early buffer (×1) is unchanged data (7) AND the SAME handle: the failed
    // first attempt never lent it.
    let early_buf: &DeviceSlice<u32> = &early_co;
    assert_eq!(
        handle_of(early_buf),
        early_handle,
        "early buffer must be the SAME cl_mem — never lent by the failed sync"
    );
    let mut e = vec![0u32; N];
    early_co.read(&mut e).wait().expect("read early");
    assert!(e.iter().all(|&v| v == 7), "early ×1 = 7, got {:?}", &e[..8]);

    // The late branch produced correct data (5 × 2 = 10).
    let mut l = vec![0u32; N];
    late_co.read(&mut l).wait().expect("read late");
    assert!(
        l.iter().all(|&v| v == 10),
        "late 5×2 = 10, got {:?}",
        &l[..8]
    );
}

/// (2) A CHECKED-OUT (`Lent`) slot is atomic: `check_ready` reports it as
/// `SlotUnbound` (a `Lent` slot has nothing to lend) WITHOUT side effects, and the
/// graph recovers as soon as the live `Checkout` drops. This is the read-only
/// mirror of `resolve_home`/`lend_slot`'s Lent → `SlotUnbound` rule.
///
/// An in-place slot scale is the terminal, so run 1's `Checkout` HOLDS the slot's
/// buffer (`Lent`). While it is alive a re-`sync` must error; `check_ready` catches
/// it before any work is enqueued (so a graph carrying additional ready nodes would
/// not have them lent — see scenario (1) for the earlier-untouched proof). After
/// the `Checkout` drops the slot re-arms `Lent → Bound` and the graph re-runs with
/// the SAME buffer handle (it was never severed or stranded by the failed attempt).
#[test]
fn checked_out_cell_in_later_node_is_atomic() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // In-place slot scale ×1, NO download: the Checkout holds the slot's buffer.
    let g = ks.scale_u32([N], slot!(Late), 1u32);

    let bound = seeded(&ctx, 4);
    let bound_handle = handle_of(&bound);

    // Run 1: bind, sync → a live Checkout (its slot buffer is now `Lent`).
    let co1 = g
        .bind(Late(bound))
        .expect("bind Late")
        .sync(&ctx)
        .expect("run 1");

    // Run 2 (slot still checked out → `Lent`). `check_ready` must error
    // `SlotUnbound` atomically — nothing is lent or enqueued.
    let err = g
        .sync(&ctx)
        .expect_err("a checked-out (Lent) slot must error at sync");
    assert!(
        matches!(err, Error::SlotUnbound(n) if n.contains("Late")),
        "expected SlotUnbound naming Late, got {err:?}"
    );

    // Recover: drop the live Checkout → the slot re-arms `Lent → Bound`.
    drop(co1);

    // Run 3: the slot is Bound again; `sync` runs. The buffer is the SAME handle —
    // the failed attempt neither severed nor re-lent it.
    let co3 = g.sync(&ctx).expect(
        "recovered sync — the slot must re-arm after the live Checkout drops, not stay stranded",
    );
    let co3_buf: &DeviceSlice<u32> = &co3;
    assert_eq!(
        handle_of(co3_buf),
        bound_handle,
        "slot buffer must be the SAME cl_mem across the failed attempt"
    );
    let mut l = vec![0u32; N];
    co3.read(&mut l).wait().expect("read run 3");
    assert!(
        l.iter().all(|&v| v == 4),
        "×1 (twice) = 4, got {:?}",
        &l[..8]
    );
}

/// (3) A fully-bound multi-node graph runs normally — `check_ready` passes and
/// there is no regression: every branch produces correct data.
#[test]
fn ready_graph_runs_normally() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Multi-node: a bundle of two in-place scales over distinct concrete buffers,
    // plus a downstream chain on one branch is not needed — the bundle itself is a
    // two-leaf graph whose check_ready recurses both branches.
    let a = seeded(&ctx, 2);
    let b = seeded(&ctx, 5);
    let g = bundle2(ks.scale_u32([N], a, 3u32), ks.scale_u32([N], b, 4u32));

    let (a_co, b_co) = g.sync(&ctx).expect("ready graph must sync");
    let mut ra = vec![0u32; N];
    a_co.read(&mut ra).wait().expect("read a");
    assert!(ra.iter().all(|&v| v == 6), "2×3 = 6, got {:?}", &ra[..8]);
    let mut rb = vec![0u32; N];
    b_co.read(&mut rb).wait().expect("read b");
    assert!(rb.iter().all(|&v| v == 20), "5×4 = 20, got {:?}", &rb[..8]);
}

/// (4) The error a failed `check_ready` produces is the SAME variant/message the
/// old execute-time failure produced — `SlotUnbound`, naming the tag — so a
/// single-node unbound-slot `sync` still surfaces exactly `SlotUnbound`.
#[test]
fn early_caught_error_is_identical_slot_unbound() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks.scale_u32([N], slot!(Missing), 2u32);
    let err = g.sync(&ctx).expect_err("unbound slot must error at sync");
    match err {
        Error::SlotUnbound(name) => assert!(
            name.contains("Missing"),
            "SlotUnbound should name the tag (`Missing`), got {name:?}"
        ),
        other => panic!("expected Error::SlotUnbound, got {other:?}"),
    }
}
