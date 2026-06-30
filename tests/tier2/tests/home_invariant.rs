//! Home-on-payload invariant — RED spec for the reusable-graph engine fix.
//!
//! THE INVARIANT being specified here (most of it NOT yet implemented):
//!
//! > A buffer with a home is never destroyed by the graph; it is *returned*
//! > ("rehomed") to its origin cell so the graph re-runs with a STABLE
//! > underlying `cl_mem` handle across replays. **"Homeless is never
//! > legitimate."**
//!
//! Consequences each test below pins down:
//! - A consuming terminal (`download`) reads the device buffer into a host
//!   `Vec` but must RETURN the device buffer to its home (NOT release it). On
//!   replay the same `cl_mem` handle is reused. (TODAY `Download::execute` uses
//!   `resolve` — discarding the home — then drops `buf`, releasing the cl_mem.
//!   See `eager.rs:3452`. THE bug.)
//! - `upload()`-minted buffers also acquire a persistent home (stable handle
//!   across runs). Reseed-by-access-mode: ReadOnly/Frozen seed once (upload
//!   skipped on replay), WriteOnly never seeds, ReadWrite re-seeds the SAME
//!   data each run into the SAME handle. (TODAY `Upload::execute`,
//!   `eager.rs:3401`, mints a FRESH `from_slice` buffer each run → new handle,
//!   no home.)
//! - `into_inner()` (user keeps the value) is the ONLY path that empties /
//!   severs a cell.
//! - A homed buffer dropped mid-graph (produced, never delivered) returns to
//!   the graph.
//!
//! Scenarios that FAIL today are marked `#[ignore = "RED: <reason>"]`; the
//! ignored set IS the spec the implementer must turn on. Scenarios that pass
//! today are left active (green regression). Each asserts BOTH data correctness
//! AND the handle invariant.
//!
//! Handle identity is read via the public `RecordableBuffer::record_handle()`
//! (re-exported from the crate root) — `BufHandle.mem` is a `MemRef::Buffer(cl_mem)`
//! whose raw pointer is the stable identity key. No new accessor was needed.

use claspr::eager::{DeviceOpExt, download, eager_copy_to, upload, upload_as};
use claspr::{Context, DeviceSlice, MemRef, ReadOnly, RecordableBuffer, WriteOnly};
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

/// The stable identity of a buffer's backing memory: the raw `cl_mem` (or SVM)
/// pointer as a `usize`, for `==` identity comparison across runs. Reads through
/// the public `RecordableBuffer::record_handle()` — works on a bare
/// `DeviceSlice` and (via `Deref`) on a live `Checkout<DeviceSlice>`.
fn handle_of<B: RecordableBuffer>(b: &B) -> usize {
    match b.record_handle().mem {
        MemRef::Buffer(m) => m as usize,
        MemRef::Svm(p) => p as usize,
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

// Slot tags for the slot-positioned scenarios.
slots! {
    Buf: DeviceSlice<u32>,
    Src: DeviceSlice<u32>,
    Dst: DeviceSlice<u32>,
}

// ───────────────────────────────────────────────────────────────────────────
// 1. upload(ReadOnly) → kernel → download, ×3.
//    Invariant: handle stable across runs; data correct each run; (future)
//    upload work skipped after run 1.
//    A ReadOnly buffer can only be a kernel READ operand (it isn't
//    `KernelWritable`), so the kernel here is `add_u32(ro, ro, out)` — the
//    ReadOnly upload buffer is the seed-once operand whose handle we pin; `out`
//    holds 3+3 = 6.
//    RED: upload mints a fresh `from_slice` buffer each run (new handle, no
//    home, eager.rs:3401), so the ReadOnly operand's handle is NOT stable across
//    runs; seed-skip-on-replay is not implemented.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn upload_readonly_kernel_download_x3_stable_handle() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // upload(ReadOnly, [3;N]) is a read operand `a` to add_u32(a, b, out); b is a
    // second ReadOnly [3;N], out holds 3+3 = 6. No download in the probe so the
    // operand Checkout exposes the upload buffer's handle.
    let mk_b = || DeviceSlice::<u32, ReadOnly>::from_slice(&ctx, &[3u32; N]).expect("b operand");
    let probe = upload_as(vec![3u32; N], ReadOnly).and_then(|ro| {
        let out = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("out");
        ks.add_u32([N], ro, mk_b(), out)
    });
    let (ro_co, _b_co, _out_co) = probe.sync(&ctx).expect("probe run 1");
    let h0 = handle_of(&*ro_co);
    drop((ro_co, _b_co, _out_co));
    // Re-run the SAME graph: the ReadOnly upload buffer must keep its handle.
    let (ro_co2, _b, _o) = probe.sync(&ctx).expect("probe run 2");
    assert_eq!(
        handle_of(&*ro_co2),
        h0,
        "upload(ReadOnly) buffer must keep a STABLE cl_mem across runs"
    );
    drop((ro_co2, _b, _o));

    // Data correctness through a consuming terminal, three runs, no compounding.
    let g = upload_as(vec![3u32; N], ReadOnly)
        .and_then(|ro| {
            let out = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("out");
            ks.add_u32([N], ro, mk_b(), out)
        })
        .and_then(|(_a, _b, out)| download(out));
    for run in 0..3 {
        let out = g.sync(&ctx).unwrap_or_else(|e| panic!("run {run}: {e}"));
        assert!(
            out.iter().all(|&v| v == 6),
            "run {run}: upload(ReadOnly) seed-once → 3+3 = 6, got {:?}",
            &out[..8]
        );
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 2. user-alloc → scale (in place) → download, ×3.
//    Invariant: buffer rehomed, handle stable, data idempotent (same result
//    each run, NOT compounding).
//    RED: download releases the buffer (no home returned), so run 2 finds the
//    concrete cell empty → "graph busy". And the handle can't be stable because
//    the buffer is gone after run 1.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn user_alloc_scale_download_x3_rehomed_stable() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let buf = seeded(&ctx, 5);
    let _h0 = handle_of(&buf); // the handle the rehomed buffer MUST keep each run.

    // scale(buf, 1) in place -> download. factor 1 makes the result idempotent
    // even though scale itself compounds — so a stable result across 3 runs proves
    // the SAME buffer (same handle) is returned by download and reused (NOT
    // re-allocated, NOT compounding). The download terminal yields a Vec, so the
    // kernel buffer's handle can't be read directly here — its stability is what
    // makes runs 2/3 succeed at all (a released buffer → empty cell → busy error).
    let g = ks.scale_u32([N], buf, 1u32).and_then(download);

    for run in 0..3 {
        let out = g
            .sync(&ctx)
            .unwrap_or_else(|e| panic!("run {run}: graph must re-arm via rehome: {e}"));
        assert!(
            out.iter().all(|&v| v == 5),
            "run {run}: idempotent scale(×1) over the rehomed buffer → 5, got {:?}",
            &out[..8]
        );
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 3. upload(ReadWrite) → scale → download, ×3.
//    Invariant: handle stable, contents RE-SEEDED each run (result identical
//    each run — proves reset, no compounding).
//    RED: upload mints a fresh buffer each run (new handle); download releases.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn upload_readwrite_scale_download_x3_reseed_stable() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Default upload marker is ReadWrite. Re-seeds 3 each run; ×5 = 15 each run.
    let probe = upload(vec![3u32; N]).and_then(|b| ks.scale_u32([N], b, 5u32));
    // Keep c1 ALIVE (do not consume it) through the next sync. The persistent
    // upload home is LENT while c1 holds it, so a second sync is graph-BUSY — the
    // invariant in action (the buffer is not re-minted; it lives in exactly one
    // place, currently c1's hands).
    let c1 = probe.sync(&ctx).expect("probe 1");
    let h0 = handle_of(&*c1);

    // A second sync while c1 is alive must be graph-busy (the home is lent), NOT a
    // fresh re-mint with a distinct handle.
    assert!(
        probe.sync(&ctx).is_err(),
        "upload(ReadWrite) home is lent while c1 is alive → second sync must be busy"
    );

    // Drop c1 → the buffer rehomes to the persistent upload cell. A re-run must
    // reuse the SAME stable handle (re-seeded into the SAME cl_mem, not re-minted).
    drop(c1);
    let c2 = probe.sync(&ctx).expect("probe 2 after c1 dropped");
    assert_eq!(
        handle_of(&*c2),
        h0,
        "upload(ReadWrite) must re-seed into the SAME stable handle (not re-mint)"
    );
    drop(c2);

    // And through the full Vec-producing consuming terminal.
    let g = upload(vec![3u32; N])
        .and_then(|b| ks.scale_u32([N], b, 5u32))
        .and_then(download);
    for run in 0..3 {
        let out = g.sync(&ctx).unwrap_or_else(|e| panic!("run {run}: {e}"));
        assert!(out.iter().all(|&v| v == 15), "run {run}: reseed → 15");
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 4. upload(WriteOnly) → kernel-writes → download, ×3.
//    Invariant: handle stable; result depends only on kernel, not prior
//    contents (WriteOnly never seeds — kernel overwrites).
//
//    RED(no-API): a `WriteOnly` buffer cannot be ANY of today's kernel slice
//    args. `WriteOnly` is `KernelWritable` but NOT `KernelReadable`
//    (access.rs:208-211); a read arg `&[T]` needs `KernelReadable`
//    (launch.rs:160) and a `&mut [T]` arg needs `KernelReadable + KernelWritable`
//    (launch.rs:176 — rust-gpu's `&mut [T]` always permits reads). So neither
//    `fill_u32(&mut)` nor `add_u32(&[])` accepts a WriteOnly buffer; WriteOnly is
//    images-only for buffers (access.rs §"marker set"). The body is written the
//    way it SHOULD read once a write-only buffer kernel-arg path exists, but is
//    neutralized so the file compiles. Once that exists, drop the `return` and the
//    `_` aliases and re-enable. Also still RED on the engine side: upload re-mints
//    each run (eager.rs:3401), no home, no seed-skip.
// ───────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "RED(no-API, DEFERRED): the no-seed home path is unreachable today. \
            WriteOnly is not a usable buffer kernel arg (access.rs:208-211, \
            launch.rs:160/176 — KernelWritable-only, &mut [T] needs both); the \
            intended exercise via a write-only IMAGE kernel is blocked because \
            image kernels are one-shot/consuming, not reusable DeviceOps \
            (eager.rs:801-802; borrowed image views aren't 'static, so no home/ \
            cell). Re-enable once image kernels gain a reusable DeviceOp path. \
            ALSO engine-RED: upload re-mints each run with no home (eager.rs:3401)."]
fn upload_writeonly_kernel_download_x3_stable() {
    let Some(ctx) = ctx() else { return };
    let _ks = kernels::kernels(&ctx).expect("load kernels");
    let _ = WriteOnly; // witness the marker the scenario needs.

    // RED(no-API): the following is the intended shape; it does NOT compile today
    // because `WriteOnly` is not a `KernelSliceRead[Write]Arg`. Re-enable when a
    // write-only buffer kernel-arg path lands.
    /*
    // fill_u32 overwrites every element with 0xAB regardless of prior contents —
    // the WriteOnly buffer's seed never matters. Use a sentinel seed to prove it.
    let probe = upload_as(vec![1u32; N], WriteOnly).and_then(|b| _ks.fill_u32([N], b, 0xABu32));
    let c1 = probe.sync(&ctx).expect("probe 1");
    let h0 = handle_of(&*c1);
    let mut r1 = vec![0u32; N];
    c1.read(&mut r1).wait().expect("read 1");
    assert!(r1.iter().all(|&v| v == 0xAB), "kernel writes 0xAB");

    let c2 = probe.sync(&ctx).expect("probe 2");
    assert_eq!(handle_of(&*c2), h0, "WriteOnly buffer keeps a stable handle");
    let mut r2 = vec![0u32; N];
    c2.read(&mut r2).wait().expect("read 2");
    assert!(
        r2.iter().all(|&v| v == 0xAB),
        "result depends only on the kernel, not prior contents, got {:?}",
        &r2[..8]
    );

    let g = upload_as(vec![1u32; N], WriteOnly)
        .and_then(|b| _ks.fill_u32([N], b, 0xABu32))
        .and_then(download);
    for run in 0..3 {
        let out = g.sync(&ctx).unwrap_or_else(|e| panic!("run {run}: {e}"));
        assert!(out.iter().all(|&v| v == 0xAB), "run {run}: 0xAB");
    }
    */
    panic!("RED(no-API): WriteOnly buffer kernel-arg path does not exist yet");
}

// ───────────────────────────────────────────────────────────────────────────
// 5. user-alloc → download directly, ×2.
//    Invariant: after run 1 the buffer is rehomed (same handle), run 2 works
//    without a fresh alloc.
//    RED: download discards the home (eager.rs:3452) → run 2 finds the cell
//    empty → "graph busy".
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn user_alloc_download_directly_x2_rehomed() {
    let Some(ctx) = ctx() else { return };

    let buf = seeded(&ctx, 9);

    // Pure download of a user buffer. The buffer must be RETURNED to its cell on
    // the run's Checkout drop so a second sync reads the SAME buffer again.
    let g = download::<u32, _>(buf);

    let out1 = g.sync(&ctx).expect("download run 1");
    assert!(out1.iter().all(|&v| v == 9), "run 1 reads 9");
    drop(out1); // re-arm via rehome (NOT release).

    let out2 = g
        .sync(&ctx)
        .expect("download run 2 must reuse the rehomed buffer (no fresh alloc)");
    assert!(
        out2.iter().all(|&v| v == 9),
        "run 2 reads the SAME buffer: 9"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// 6. copy with slot src AND slot dst, ×2.
//    Invariant: both rehome (handles stable) — the copy-slot gap.
//
//    RED(no-API) + RED(engine): two layers.
//    - no-API: `eager_copy_to(slot!(Src), slot!(Dst))` does not type-check — a
//      `slot!` is a `SlotHandle<Tag>`, which is not `CopyTo<_>` (only the buffer
//      families are, copy.rs). A slot can only sit in a KERNEL-arg position today,
//      not a copy operand position. The intended shape is written below but
//      neutralized so the file compiles.
//    - engine: even once a slot is accepted as a copy operand, `CopyTo2::execute`
//      (eager.rs:5348) only threads homes for CONCRETE src/dst (via
//      `return_cell` + `CopyHome`); a slot's `return_cell()` is `None`, so its
//      `SlotHome` is never threaded (eager.rs:5339-5347 comment) → the slot stays
//      `Lent` after run 1 → second sync is graph-busy. This is THE copy-slot gap.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn copy_slot_src_and_slot_dst_x2_both_rehome() {
    let Some(ctx) = ctx() else { return };

    let data: Vec<u32> = (0..N as u32).collect();
    let _src = DeviceSlice::<u32>::from_slice(&ctx, &data).expect("src");
    let _dst = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("dst");

    let hs = handle_of(&_src);
    let hd = handle_of(&_dst);

    // Both copy operands are SLOTS, bound before the run. `bind` returns `&Self`,
    // so the copy op must live in a `let` (it can't be a temporary).
    let g = eager_copy_to(slot!(Src), slot!(Dst));
    g.bind(Src(_src)).expect("bind Src");
    g.bind(Dst(_dst)).expect("bind Dst");

    {
        let (co_s, co_d) = g.sync(&ctx).expect("copy run 1");
        assert_eq!(handle_of(&*co_s), hs, "slot src handle stable run 1");
        assert_eq!(handle_of(&*co_d), hd, "slot dst handle stable run 1");
    } // both Checkouts drop → both slots must re-arm Lent → Bound.

    let (co_s, co_d) = g
        .sync(&ctx)
        .expect("copy run 2 (both slots must re-arm, not stay Lent)");
    assert_eq!(handle_of(&*co_s), hs, "slot src handle stable run 2");
    assert_eq!(handle_of(&*co_d), hd, "slot dst handle stable run 2");
    let mut out = vec![0u32; N];
    co_d.read(&mut out).wait().expect("read dst run2");
    assert_eq!(out, data, "run 2 dst == src");
}

// ───────────────────────────────────────────────────────────────────────────
// 7. cross-graph hand-off.
//    g produces a buffer; co = g.bind(...).sync(); v = download(co); v.sync().
//    Invariant: g is left re-armable (slot severed to Unbound, so a fresh
//    g.bind(Buf(other)).sync() works) AND v's Vec is correct (b's scaled data).
//    This PASSES today: feeding a Checkout to `download` severs the slot
//    (into_inner → SlotHome::sever → Lent→Unbound), so a fresh bind works.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn cross_graph_handoff_severs_and_rearms() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // g: scale a slot buffer by 2, in place (no download → output IS the buffer).
    let g = ks.scale_u32([N], slot!(Buf), 2u32);

    let b = seeded(&ctx, 4); // 4*2 = 8
    let co = g.bind(Buf(b)).expect("bind").sync(&ctx).expect("g run");

    // Hand `co` to a second graph that downloads it. Feeding a Checkout as a
    // `download` input severs g's slot (into_inner → Lent→Unbound).
    let v = download(co);
    let out = v.sync(&ctx).expect("download co");
    assert!(
        out.iter().all(|&x| x == 8),
        "downloaded co holds b's scaled data: 4*2 = 8, got {:?}",
        &out[..8]
    );

    // g's slot was SEVERED → a fresh bind of a DIFFERENT buffer succeeds.
    let other = seeded(&ctx, 5); // 5*2 = 10
    let co2 = g
        .bind(Buf(other))
        .expect("g re-armable: fresh bind after sever")
        .sync(&ctx)
        .expect("g re-run");
    let mut rb = vec![0u32; N];
    co2.read(&mut rb).wait().expect("read");
    assert!(rb.iter().all(|&x| x == 10), "re-bound buffer: 5*2 = 10");
}

// ───────────────────────────────────────────────────────────────────────────
// 8. cross-graph as kernel arg.
//    Feed co (a produced buffer) as a kernel arg to a second graph, run it,
//    then re-bind the first graph's slot and re-run. Both produce correct data.
//    PASSES today: feeding co as a kernel arg severs g1's slot; g1 re-arms via
//    a fresh bind.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn cross_graph_as_kernel_arg() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // g1: scale a slot buffer by 3 in place.
    let g1 = ks.scale_u32([N], slot!(Buf), 3u32);
    let b = seeded(&ctx, 2); // 2*3 = 6
    let co = g1
        .bind(Buf(b))
        .expect("bind g1")
        .sync(&ctx)
        .expect("g1 run");

    // g2: feed `co` (holding [6;N]) as a kernel arg — scale by 2 → 12.
    let g2 = ks.scale_u32([N], co, 2u32);
    let out2 = g2.sync(&ctx).expect("g2 run");
    let mut r2 = vec![0u32; N];
    out2.read(&mut r2).wait().expect("read g2");
    assert!(
        r2.iter().all(|&x| x == 12),
        "g2: 6*2 = 12, got {:?}",
        &r2[..8]
    );

    // g1's slot was severed by feeding co into g2 → re-bind + re-run works.
    let other = seeded(&ctx, 7); // 7*3 = 21
    let co3 = g1
        .bind(Buf(other))
        .expect("g1 re-armable after its buffer was consumed by g2")
        .sync(&ctx)
        .expect("g1 re-run");
    let mut r3 = vec![0u32; N];
    co3.read(&mut r3).wait().expect("read g1 rerun");
    assert!(r3.iter().all(|&x| x == 21), "g1 re-run: 7*3 = 21");
}

// ───────────────────────────────────────────────────────────────────────────
// 9. slot rehome vs sever.
//    (a) bind→sync→drop(checkout)→re-sync reuses same handle (rehome).
//    (b) bind→sync→into_inner→bind(different)→sync works and uses the NEW
//        buffer (sever path).
//    PASSES today (this is the locked slot re-arm/sever behaviour; we add the
//    handle-identity assertion on top).
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn slot_rehome_vs_sever_handle_identity() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks.scale_u32([N], slot!(Buf), 2u32);

    // (a) rehome: drop the Checkout → slot re-arms Lent→Bound, SAME handle.
    let b = seeded(&ctx, 3);
    let h0 = handle_of(&b);
    let co1 = g.bind(Buf(b)).expect("bind").sync(&ctx).expect("run a1");
    assert_eq!(handle_of(&*co1), h0, "lent buffer keeps its handle in-run");
    drop(co1); // rehome.
    let co2 = g.sync(&ctx).expect("run a2 over re-armed slot");
    assert_eq!(
        handle_of(&*co2),
        h0,
        "rehome path: re-armed slot reuses the SAME cl_mem handle"
    );
    drop(co2);

    // (b) sever: into_inner keeps the value → slot Unbound; a fresh bind of a
    //     DIFFERENT buffer is used (its handle drives the run), and the result
    //     follows it.
    let kept = g.sync(&ctx).expect("run b0").into_inner();
    let kept_h = handle_of(&kept);
    assert_eq!(kept_h, h0, "into_inner returns the same buffer object");
    drop(kept);

    // NOTE: do NOT assert `other`'s handle differs from `h0` — the runtime may
    // RECYCLE a just-freed cl_mem address, so a fresh alloc can legitimately land
    // on the old handle. A raw handle is a reliable identity key only among LIVE
    // buffers; here we read `h_other` from the still-live `other` and check the
    // run uses exactly it.
    let other = seeded(&ctx, 9);
    let h_other = handle_of(&other);
    let co3 = g
        .bind(Buf(other))
        .expect("fresh bind after sever")
        .sync(&ctx)
        .expect("run b1");
    assert_eq!(
        handle_of(&*co3),
        h_other,
        "sever path: the NEW (live) buffer's handle drives the run"
    );
    let mut rb = vec![0u32; N];
    co3.read(&mut rb).wait().expect("read");
    // The 'other' buffer (seeded 9) was scaled ×2 once in run b1 → 18.
    assert!(rb.iter().all(|&v| v == 18), "new buffer: 9*2 = 18");
}

// ───────────────────────────────────────────────────────────────────────────
// 10. homed buffer dropped mid-graph (produced, never delivered).
//     produce a buffer (user-alloc → scale), get the Checkout, drop it WITHOUT
//     into_inner → the graph is re-armable with the SAME handle (rehome on
//     undelivered drop).
//     PASSES today for a CONCRETE-head in-place op: the concrete cell is the
//     home, Checkout-drop rehomes it, the handle is stable. (We add the handle
//     assertion; the concrete-head re-arm itself is locked by graph_reuse.)
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn homed_buffer_dropped_mid_graph_rearms_same_handle() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let buf = seeded(&ctx, 1);
    let h0 = handle_of(&buf);

    // In-place scale ×1 (idempotent) so re-runs over the same buffer are stable.
    let g = ks.scale_u32([N], buf, 1u32);

    // Run once, capture the handle, then DROP the Checkout without into_inner.
    let co = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*co), h0, "run 1 lent the original buffer");
    drop(co); // undelivered → rehome to the concrete cell.

    // Re-armable with the SAME handle.
    let co2 = g.sync(&ctx).expect("run 2 over rehomed buffer");
    assert_eq!(
        handle_of(&*co2),
        h0,
        "undelivered drop rehomes the buffer with a STABLE handle"
    );
    let mut rb = vec![0u32; N];
    co2.read(&mut rb).wait().expect("read");
    assert!(
        rb.iter().all(|&v| v == 1),
        "idempotent ×1 over stable buffer"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// 11. multi-output copy independence.
//     copy producing (Checkout<Src>, Checkout<Dst>); drop one, into_inner the
//     other; assert src re-armed (handle stable, re-sync works) and dst severed
//     (independent).
//     RED: with CONCRETE src/dst this re-arms today, BUT the asymmetric
//     "into_inner one side severs ONLY that side, the other re-arms" requires
//     the per-output home to be independent AND the src cell to re-arm with a
//     stable handle while the dst cell is severed — combined with a re-sync that
//     re-allocates the severed dst. A concrete severed cell goes empty, so a
//     re-sync of the severed side is "busy" unless the dst is re-bound. We
//     express it the way it SHOULD read and ignore until the per-side
//     sever/re-arm + re-alloc story is settled.
// ───────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "DEFERRED: per-side independent homes now work (src re-arms, dst severs \
            independently — the home-invariant core landed), but re-SYNCing after \
            severing a CONCRETE copy dst requires re-ALLOCATING that side. A bare \
            concrete `Cell` can't distinguish 'severed (→ re-alloc)' from 'lent / \
            busy (→ error)' — that needs the tri-state disambiguation the upload \
            cell got, extended to user copy cells. Out of scope for the home \
            invariant; the per-side independence it asserts is otherwise satisfied."]
fn multi_output_copy_independence() {
    let Some(ctx) = ctx() else { return };

    let data: Vec<u32> = (0..N as u32).collect();
    let src = DeviceSlice::<u32>::from_slice(&ctx, &data).expect("src");
    let dst = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("dst");
    let hs = handle_of(&src);

    let g = eager_copy_to(src, dst);

    let (co_src, co_dst) = g.sync(&ctx).expect("copy run 1");
    assert_eq!(handle_of(&*co_src), hs, "src handle stable run 1");
    // Drop src (re-arm), into_inner dst (sever + keep).
    drop(co_src);
    let kept_dst = co_dst.into_inner();
    let mut out = vec![0u32; N];
    kept_dst.read(&mut out).wait().expect("read kept dst");
    assert_eq!(out, data, "kept dst holds the copied data");

    // src must be re-armed with a STABLE handle; dst was severed (independent).
    // A re-sync exercises src's re-arm. The severed dst cell is empty, so the
    // engine must re-allocate it (the spec for a severed copy side) rather than
    // error busy.
    let (co_src2, _co_dst2) = g
        .sync(&ctx)
        .expect("copy run 2: src re-armed, dst re-allocated after sever");
    assert_eq!(
        handle_of(&*co_src2),
        hs,
        "src side independent: re-armed with the same handle after dst severed"
    );
}
