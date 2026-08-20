//! Home-on-payload invariant — the test catalog that locks it in.
//!
//! THE INVARIANT pinned down here:
//!
//! > A buffer with a home is never destroyed by the graph; it is *returned*
//! > ("rehomed") to its origin cell so the graph re-runs with a STABLE
//! > underlying `cl_mem` handle across replays. **"Homeless is never
//! > legitimate."**
//!
//! Consequences each test below pins down:
//! - A consuming terminal (`download`) reads the device buffer into a host
//!   `Vec` but RETURNS the device buffer to its home (NOT release it). On
//!   replay the same `cl_mem` handle is reused.
//! - `upload()`-minted buffers also acquire a persistent home (stable handle
//!   across runs). Reseed-by-access-mode: ReadOnly/Frozen seed once (upload
//!   skipped on replay), WriteOnly never seeds, ReadWrite re-seeds the SAME
//!   data each run into the SAME handle.
//! - `into_inner()` (user keeps the value) is the ONLY path that empties /
//!   severs a cell. For a SLOT it lands in `Severed` (not virgin `Unbound`): a
//!   later set-once `bind` is `SlotSevered` and only `mutate_bind` re-arms it.
//! - A homed buffer dropped mid-graph (produced, never delivered) returns to
//!   the graph.
//!
//! Each scenario asserts BOTH data correctness AND the handle invariant.
//!
//! Handle identity is read via the public `RecordableBuffer::record_handle()`
//! (re-exported from the crate root) — `BufHandle.mem` is a `MemRef::Buffer(cl_mem)`
//! whose raw pointer is the stable identity key.

use claspr::eager::{
    DeviceOpExt, bundle2, download, eager_copy_to, fan_out, fill, upload, upload_as,
};
use claspr::image::format::R32Uint;
use claspr::{DeviceSlice, Error, Image2D, ReadOnly, WriteOnly};
use claspr::{slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::{N, ctx, handle_of, seeded};

// Slot tags for the slot-positioned scenarios.
slots! {
    Buf: DeviceSlice<u32>,
    Src: DeviceSlice<u32>,
    Dst: DeviceSlice<u32>,
}

// ───────────────────────────────────────────────────────────────────────────
// 1. upload(ReadOnly) → kernel → download, ×3.
//    Invariant: handle stable across runs; data correct each run; upload work
//    skipped after run 1.
//    A ReadOnly buffer can only be a kernel READ operand (it isn't
//    `KernelWritable`), so the kernel here is `add_u32(ro, ro, out)` — the
//    ReadOnly upload buffer is the seed-once operand whose handle we pin; `out`
//    holds 3+3 = 6.
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
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn user_alloc_scale_download_x3_rehomed_stable() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let buf = seeded(&ctx, 5);
    let h0 = handle_of(&buf); // the handle the rehomed buffer MUST keep each run.

    // scale(buf, 1) in place, NO download — the terminal IS the buffer, so its
    // Checkout exposes the (rehomed) handle each run. factor 1 makes the result
    // idempotent even though scale itself compounds, so a stable result across 3
    // runs proves the SAME buffer is returned and reused (NOT re-allocated). We
    // ALSO capture the handle before the graph consumes it and assert, each run,
    // that the run lent EXACTLY that cl_mem back (rehome, not re-mint) — reading
    // the contents through the still-lent Checkout (a borrowing read), then
    // dropping it to rehome for the next run.
    let g = ks.scale_u32([N], buf, 1u32);

    for run in 0..3 {
        let co = g
            .sync(&ctx)
            .unwrap_or_else(|e| panic!("run {run}: graph must re-arm via rehome: {e}"));
        assert_eq!(
            handle_of(&*co),
            h0,
            "run {run}: rehomed buffer must keep a STABLE cl_mem across runs (not re-mint)"
        );
        // Borrowing read (map, not the consuming `read`) so `co` survives to be
        // dropped → rehomed for the next run.
        let guard = co.map().wait().expect("map read back");
        assert!(
            guard.iter().all(|&v| v == 5),
            "run {run}: idempotent scale(×1) over the rehomed buffer → 5, got {:?}",
            &guard[..8]
        );
        drop(guard);
        drop(co); // rehome the buffer to its concrete cell for the next run.
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 3. upload(ReadWrite) → scale → download, ×3.
//    Invariant: handle stable, contents RE-SEEDED each run (result identical
//    each run — proves reset, no compounding).
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
    // fresh re-mint with a distinct handle. The upload cell is a concrete-style
    // cell, so the busy path is `Error::NotSupported` whose message names the lent
    // upload buffer ("already lent" / "the graph is busy") — NOT a slot error.
    let err = probe
        .sync(&ctx)
        .expect_err("upload(ReadWrite) home is lent while c1 is alive → second sync must be busy");
    assert!(
        matches!(&err, Error::NotSupported(m)
            if m.contains("already lent") && m.contains("busy")),
        "expected NotSupported busy (upload buffer already lent), got {err:?}"
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
// 4. WRITE-ONLY image kernel (no-seed home path), ×3.
//    Invariant: the image is allocated ONCE, never seeded, and a write-only
//    image kernel (`dim2_uint::fill_pattern`) FULLY OVERWRITES it each run; its
//    `cl_mem` handle is STABLE across 3 runs (rehomed, not re-minted) AND the
//    data is correct each run.
//
//    Why an IMAGE, not a buffer: a `WriteOnly` BUFFER cannot be a kernel slice
//    arg (`WriteOnly` is `KernelWritable` but NOT `KernelReadable`,
//    access.rs:208-211; `&[T]` needs Readable and `&mut [T]` needs both,
//    launch.rs:160/176). A `WriteOnly` IMAGE is the canonical write-only case —
//    `Image2D<WriteOnly, _>` impls `KernelImage2DWriteArg`, matching the kernel's
//    `image_access="write_only"` qualifier. This is exactly the no-seed home
//    path: there is no upload, the image's contents are entirely kernel-produced,
//    so the only thing the reusable graph must guarantee is a stable rehomed
//    handle across replays.
//
//    Image args ride the same `Input`/cell/`Checkout` lend-and-return path as
//    slice args, so the write-only image kernel Op is a reusable `DeviceOp`: the
//    allocated-once image is lent from its concrete cell each run and rehomed on
//    the run's `Checkout` drop, keeping a stable `cl_mem`.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn upload_writeonly_kernel_download_x3_stable() {
    let Some(ctx) = ctx() else { return };
    if !ctx.device().cl3().image_support().unwrap_or(false) {
        eprintln!("SKIP: device has no image support");
        return;
    }
    let _ = WriteOnly; // witness the marker the scenario exercises.
    let iks = claspr_test_image_kernels::dim2_uint::kernels(&ctx).expect("load image kernels");

    const W: u32 = 16;
    const H: u32 = 8;

    // Allocate the write-only image ONCE; it is never seeded. The kernel writes
    // `(x + y*W)` at every pixel, so the result depends only on the kernel.
    let img = Image2D::<WriteOnly, R32Uint>::alloc(&ctx, W, H).expect("alloc image");
    let h0 = handle_of(&img); // the cl_mem the rehomed image MUST keep each run.

    // The reusable graph: fill_pattern over the lent image. `g.sync()` runs it and
    // hands back a `Checkout<Image2D<…>>`; dropping the Checkout rehomes the image
    // to its concrete cell so the next run re-lends the SAME handle.
    let g = iks.fill_pattern([W as usize, H as usize], img, W, H);

    for run in 0..3 {
        let co = g
            .sync(&ctx)
            .unwrap_or_else(|e| panic!("run {run}: write-only image graph must re-arm: {e}"));
        assert_eq!(
            handle_of(&*co),
            h0,
            "run {run}: write-only image must keep a STABLE cl_mem across runs (rehome, not re-mint)"
        );
        // Data correctness each run: read back through a fresh download terminal on
        // the (still-lent) image, then drop `co` to rehome it for the next run.
        let pixels: Vec<u32> = co.read_alloc().wait().expect("read back");
        for y in 0..H {
            for x in 0..W {
                let got = pixels[(y * W + x) as usize];
                let want = x + y * W;
                assert_eq!(
                    got, want,
                    "run {run}: pixel ({x},{y}): got {got}, want {want}"
                );
            }
        }
        drop(co); // rehome the image to its concrete cell (re-arm), NOT release.
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 5. user-alloc → buffer-terminal op, ×2.
//    Invariant: after run 1 the buffer is rehomed (same handle), run 2 works
//    without a fresh alloc — proven by capturing the buffer's handle before the
//    graph consumes it and asserting the SAME cl_mem is lent back on BOTH runs.
//
//    A pure `download` terminal yields a host `Vec`, so the device handle is not
//    observable through it. To PROVE the stable-handle claim (not just
//    re-runnability + contents) this uses `fill(buf, 9)` — a kernel-free op over
//    a bare user buffer whose terminal IS the buffer (a Checkout), so its handle
//    is readable each run. `fill(9)` on an already-9 buffer is idempotent, and we
//    read contents via a borrowing `map` so the Checkout survives to rehome.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn user_alloc_download_directly_x2_rehomed() {
    let Some(ctx) = ctx() else { return };

    let buf = seeded(&ctx, 9);
    let h0 = handle_of(&buf); // the handle the rehomed buffer MUST keep each run.

    // Idempotent fill over a user buffer. The buffer must be RETURNED to its cell
    // on the run's Checkout drop so a second sync lends the SAME buffer again.
    let g = fill(buf, 9u32);

    let co1 = g.sync(&ctx).expect("run 1");
    assert_eq!(handle_of(&*co1), h0, "run 1 lends the original buffer");
    let guard1 = co1.map().wait().expect("map run 1");
    assert!(guard1.iter().all(|&v| v == 9), "run 1 reads 9");
    drop(guard1);
    drop(co1); // re-arm via rehome (NOT release).

    let co2 = g
        .sync(&ctx)
        .expect("run 2 must reuse the rehomed buffer (no fresh alloc)");
    assert_eq!(
        handle_of(&*co2),
        h0,
        "run 2 lends the SAME rehomed cl_mem (stable handle, not re-minted)"
    );
    let guard2 = co2.map().wait().expect("map run 2");
    assert!(
        guard2.iter().all(|&v| v == 9),
        "run 2 reads the SAME buffer: 9"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// 6. copy with slot src AND slot dst, ×2.
//    Invariant: both operands are SLOTS; both rehome (handles stable) across the
//    two runs.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn copy_slot_src_and_slot_dst_x2_both_rehome() {
    let Some(ctx) = ctx() else { return };

    let data: Vec<u32> = (0..N as u32).collect();
    let _src = DeviceSlice::<u32>::from_slice(&ctx, &data).expect("src");
    let _dst = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("dst");

    let hs = handle_of(&_src);
    let hd = handle_of(&_dst);

    // Both copy operands are SLOTS, set-once bound (consuming) before the run and
    // folded into `g`; `g` is then re-`sync`'d by `&` below.
    let g = eager_copy_to(slot!(Src), slot!(Dst))
        .bind(Src(_src))
        .bind(Dst(_dst));

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
// 7. cross-graph hand-off via `download` — LENDS and returns (NOT sever).
//    g produces a buffer; co = g.bind(...).sync(); download(co).sync().
//    Feeding a Checkout forward as a borrow input LENDS it: g's slot home rides
//    into the second graph. `download` CONSUMES the value into a host Vec, but the
//    device buffer is RETURNED to g's slot (via `rehome_consumed`) during the
//    download run — so right after the download `sync()` returns, g re-runs via a
//    plain `sync()` (NO `mutate_bind`) over the SAME handle. This is the corrected
//    behaviour — the old test asserted the buggy sever path (into_inner).
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn cross_graph_handoff_lends_and_returns() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // g: scale a slot buffer by 1 (idempotent), in place. Idempotent so a re-run
    // over the SAME returned buffer is stable (proves rehome, not compounding).
    let b = seeded(&ctx, 8);
    let h0 = handle_of(&b);
    // Set-once bind (consuming) folded into `g`; re-`sync`'d by `&` below.
    let g = ks.scale_u32([N], slot!(Buf), 1u32).bind(Buf(b));
    let co = g.sync(&ctx).expect("g run");

    // Hand `co` to a second graph that downloads it (consumes the value into a Vec,
    // but the device buffer's home — g's `Lent` slot — rides the lend and is
    // returned by the download).
    let v = download(co);
    let out = v.sync(&ctx).expect("download co");
    assert!(
        out.iter().all(|&x| x == 8),
        "downloaded co holds b's scaled data: 8*1 = 8, got {:?}",
        &out[..8]
    );

    // The device buffer was RETURNED to g's slot by the download (NOT severed) →
    // g re-runs via a plain `sync()` (NO mutate_bind), over the SAME handle.
    let co2 = g
        .sync(&ctx)
        .expect("g re-armable via plain sync after the lent buffer was returned by download");
    assert_eq!(
        handle_of(&*co2),
        h0,
        "lent buffer returned to g's slot with a STABLE handle (not severed/re-minted)"
    );
    let mut rb = vec![0u32; N];
    co2.read(&mut rb).wait().expect("read");
    assert!(rb.iter().all(|&x| x == 8), "re-armed buffer: 8*1 = 8");
}

// ───────────────────────────────────────────────────────────────────────────
// 8. cross-graph as kernel arg — LENDS and returns (NOT sever).
//    Feed co (a produced buffer) as a KERNEL ARG to a second graph whose terminal
//    is the BUFFER itself (a `Checkout<DeviceSlice>`). g1's slot home rides into
//    g2's result. While g2's result is ALIVE, g1 is BUSY (its slot is still
//    `Lent`); a plain DROP of g2's result RETURNS the buffer to g1's slot,
//    re-arming it for a plain `sync()` (NO mutate_bind). This is the corrected
//    behaviour — the old test asserted the buggy sever path.
//
//    (Data correctness over g1's buffer is covered by scenario 7's download and by
//    `checkout_lend_transitive`; here we pin the busy-while-held / return-on-drop
//    lifetime, which requires a non-consuming terminal — a plain DROP, not the
//    severing `read`/`into_inner`.)
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn cross_graph_as_kernel_arg_lends_and_returns() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // g1: scale a slot buffer by 1 (idempotent) in place. Set-once bind (consuming)
    // folded into `g1`; `g1` is re-`sync`'d by `&` below.
    let b = seeded(&ctx, 6);
    let h0 = handle_of(&b);
    let g1 = ks.scale_u32([N], slot!(Buf), 1u32).bind(Buf(b));
    let co = g1.sync(&ctx).expect("g1 run");

    // g2: feed `co` as a kernel arg — scale by 2 (the terminal is the BUFFER, a
    // Checkout). This LENDS g1's slot buffer into g2; g1 stays `Lent`/busy while
    // g2's result is alive.
    let g2 = ks.scale_u32([N], co, 2u32);
    let out2 = g2.sync(&ctx).expect("g2 run");
    assert_eq!(
        handle_of(&*out2),
        h0,
        "g2 ran over g1's LENT buffer (same handle), not a fresh one"
    );

    // g1 is busy (lent into g2) while out2 is alive → a second sync must error.
    // g1's home is a SLOT (Buf), currently `Lent`, so the busy path surfaces as
    // `Error::SlotUnbound` naming the tag (its message covers the Lent/busy case).
    let err = g1
        .sync(&ctx)
        .expect_err("g1 is lent into g2 while g2's result is alive → busy, NOT a silent re-run");
    assert!(
        matches!(&err, Error::SlotUnbound(n) if n.contains("Buf")),
        "expected SlotUnbound(Buf) for the lent slot, got {err:?}"
    );

    // Plain DROP of g2's result (NOT into_inner/read) → the lent buffer RETURNS to
    // g1's slot, re-arming it.
    drop(out2);

    // g1 re-runs via a plain `sync()` (NO mutate_bind), over the SAME handle.
    let co3 = g1
        .sync(&ctx)
        .expect("g1 re-armable via plain sync after the lent buffer returned from g2");
    assert_eq!(
        handle_of(&*co3),
        h0,
        "lent buffer returned to g1's slot with a STABLE handle (not severed)"
    );
    let mut r3 = vec![0u32; N];
    co3.read(&mut r3).wait().expect("read g1 rerun");
    // g2 scaled the SAME buffer ×2 IN PLACE (6 → 12) and it returned to g1 with
    // those contents; g1's re-run scales ×1 (idempotent) → still 12. This also
    // confirms g2 computed correctly over g1's lent buffer.
    assert!(
        r3.iter().all(|&x| x == 12),
        "g1 re-run over returned buffer: 6*2*1 = 12"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// 8a. cross-graph as a COPY OPERAND — LENDS and returns (NOT sever).
//     Same story as 8, but the Checkout is fed as the SRC of an
//     `eager_copy_to(co, dst)` rather than as a kernel arg. A copy operand
//     Checkout must LEND (identical semantics to the kernel-arg path): while B's
//     copy result is alive, A is BUSY (its slot is still `Lent`); dropping B's
//     result RETURNS the buffer to A's slot, re-arming it for a plain `sync()`
//     (NO mutate_bind), over the SAME handle. Before this fix the copy-operand
//     path called `into_inner()` (severed A) — the OPPOSITE of the kernel-arg
//     path under the same `eager_copy_to(co, …)` / `kernel(co, …)` surface
//     syntax. The copy also computes correctly over A's lent buffer.
//     (`into_inner_still_severs` below covers the contrast — the explicit sever.)
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn copy_operand_checkout_lends_and_returns() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // Graph A: scale a slot buffer by 2 in place → yields a Checkout `co`. Set-once
    // bind (consuming) folded into `ga`; re-`sync`'d by `&` below.
    let b = seeded(&ctx, 3); // 3*2 = 6
    let h0 = handle_of(&b);
    let ga = ks.scale_u32([N], slot!(Buf), 2u32).bind(Buf(b));
    let co = ga.sync(&ctx).expect("A run");

    // Graph B: feed `co` as the SRC of a copy into a fresh dst. This LENDS A's
    // buffer into B; A stays `Lent`/busy while B's copy result is alive.
    let dst = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("dst alloc");
    let (co_src, co_dst) = eager_copy_to(co, dst).sync(&ctx).expect("B copy run");

    // (c) the copy ran over A's LENT buffer (same handle on the src side), not a
    // fresh one, and produced correct data (dst == A's buffer contents = 6).
    assert_eq!(
        handle_of(&*co_src),
        h0,
        "copy ran over A's LENT buffer (same handle), not a copy of a fresh one"
    );
    let mut out = vec![0u32; N];
    co_dst.read(&mut out).wait().expect("read dst");
    assert!(
        out.iter().all(|&x| x == 6),
        "copy produced A's data: 3*2 = 6"
    );

    // (a) A is busy (lent into B) while B's src result is alive → a second
    // A.sync() must ERROR — A was LENT, not severed. A's home is a SLOT (Buf),
    // currently `Lent`, so the busy path is `Error::SlotUnbound` naming the tag.
    let err = ga
        .sync(&ctx)
        .expect_err("A is lent as a copy operand while B's result is alive → busy");
    assert!(
        matches!(&err, Error::SlotUnbound(n) if n.contains("Buf")),
        "expected SlotUnbound(Buf) for the lent slot, got {err:?}"
    );

    // (b) Plain DROP of B's result (NOT into_inner) RETURNS the lent buffer to
    // A's slot, re-arming it — A re-runs via a PLAIN sync() (NO mutate_bind),
    // over the SAME handle.
    drop(co_src);
    let co3 = ga
        .sync(&ctx)
        .expect("A re-armable via plain sync after the lent copy operand returned");
    assert_eq!(
        handle_of(&*co3),
        h0,
        "lent copy-operand buffer returned to A's slot with a STABLE handle (not severed)"
    );
    let mut r3 = vec![0u32; N];
    co3.read(&mut r3).wait().expect("read A rerun");
    // A's buffer returned holding 6 (the copy is non-mutating on its src); A's
    // re-run scales ×2 → 12.
    assert!(
        r3.iter().all(|&x| x == 12),
        "A re-run over returned buffer: (3*2)*2 = 12"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// 8b. into_inner STILL severs (the explicit take-it-out verb is unchanged).
//     Contrast with 7/8: only the IMPLICIT feed-as-input path lends; calling
//     `into_inner()` explicitly and feeding the RAW buffer still severs the slot
//     (Lent → Severed), so a plain `bind` is `SlotSevered` and only `mutate_bind`
//     re-arms it.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn into_inner_still_severs() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let b = seeded(&ctx, 4); // 4*2 = 8
    // Set-once bind (consuming) folded into `g`; `g` is re-armed via `mutate_bind`
    // (`&self`) and re-`sync`'d below.
    let g = ks.scale_u32([N], slot!(Buf), 2u32).bind(Buf(b));
    let co = g.sync(&ctx).expect("g run");

    // Explicitly TAKE the buffer out — this severs g's slot (Lent → Severed).
    let raw = co.into_inner();
    let mut rb = vec![0u32; N];
    raw.read(&mut rb).wait().expect("read raw");
    assert!(rb.iter().all(|&x| x == 8), "kept buffer holds 4*2 = 8");

    // g's slot is SEVERED → a plain (consuming, infallible) `bind` records
    // `SlotSevered` and it surfaces DEFERRED at `sync` (nothing enqueued). Prove
    // that on a CLONE-shaped separate probe graph seeded the same way, since testing
    // it on `g` itself would consume `g` (and we still want to prove `mutate_bind`
    // re-arms the very same slot below).
    {
        let probe_b = seeded(&ctx, 4);
        let probe = ks.scale_u32([N], slot!(Buf), 2u32).bind(Buf(probe_b));
        let probe_co = probe.sync(&ctx).expect("probe run");
        let _kept = probe_co.into_inner(); // sever the probe's slot
        let err = probe
            .bind(Buf(seeded(&ctx, 1)))
            .sync(&ctx)
            .expect_err("plain bind on a severed slot must error at sync");
        // A set-once bind onto a Severed cell records `SlotSevered` but leaves the cell
        // Severed, so `check_ready` (state-first) reports the completeness `SlotUnbound`
        // for that still-empty slot before draining the sink. Both are fail-closed.
        assert!(
            matches!(&err, Error::SlotUnbound(n) | Error::SlotSevered(n) if n.contains("Buf")),
            "expected deferred SlotSevered / state-first SlotUnbound naming Buf, got {err:?}"
        );
    }

    // Only `mutate_bind` re-arms the severed slot — proven on `g` itself.
    let other = seeded(&ctx, 5); // 5*2 = 10
    let co2 = g
        .mutate_bind(Buf(other))
        .expect("g re-armable after into_inner via mutate_bind")
        .sync(&ctx)
        .expect("g re-run");
    let mut r2 = vec![0u32; N];
    co2.read(&mut r2).wait().expect("read");
    assert!(r2.iter().all(|&x| x == 10), "re-bound buffer: 5*2 = 10");
}

// ───────────────────────────────────────────────────────────────────────────
// 8c. transitive lend chain: A → B → C.
//     Feed A's Checkout as a kernel arg to B, then B's Checkout as a kernel arg
//     to C. A's slot home must ride the WHOLE chain and return to A only at the
//     FINAL drop (C's result). A is busy until then; afterwards A re-runs via a
//     plain `sync()` over the same handle.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn checkout_lend_transitive() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // A: scale a slot buffer by 1 (idempotent) in place. Set-once bind (consuming)
    // folded into `g_a`; re-`sync`'d by `&` below.
    let b = seeded(&ctx, 3);
    let h0 = handle_of(&b);
    let g_a = ks.scale_u32([N], slot!(Buf), 1u32).bind(Buf(b));
    let co_a = g_a.sync(&ctx).expect("A run");

    // B: scale A's buffer by 2 → 6. C: scale B's result by 5 → 30.
    let co_b = ks.scale_u32([N], co_a, 2u32).sync(&ctx).expect("B run");
    let co_c = ks.scale_u32([N], co_b, 5u32).sync(&ctx).expect("C run");

    // A is still busy: its home rode A → B → C and hasn't returned yet. A's home is
    // a SLOT (Buf), currently `Lent`, so the busy path is `Error::SlotUnbound`.
    let err = g_a
        .sync(&ctx)
        .expect_err("A is lent through the whole chain while C's result is alive → busy");
    assert!(
        matches!(&err, Error::SlotUnbound(n) if n.contains("Buf")),
        "expected SlotUnbound(Buf) for the lent slot, got {err:?}"
    );

    // Feed co_c forward ONE more hop into a download (D): the home rides A→B→C→D
    // and D's download RETURNS the buffer all the way back to A's slot, while also
    // reading the data out for the correctness check.
    let rc = download(co_c).sync(&ctx).expect("download C result");
    assert!(
        rc.iter().all(|&x| x == 30),
        "C: 3*1*2*5 = 30, got {:?}",
        &rc[..8]
    );

    // The buffer has returned to A → A re-runs via a plain `sync()` (NO mutate_bind)
    // over the SAME handle.
    let co_a2 = g_a
        .sync(&ctx)
        .expect("A re-armable via plain sync after the transitive chain returned the buffer");
    assert_eq!(
        handle_of(&*co_a2),
        h0,
        "buffer returned to A's slot with a STABLE handle after the transitive chain"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// 9. slot rehome vs sever.
//    (a) bind→sync→drop(checkout)→re-sync reuses same handle (rehome).
//    (b) bind→sync→into_inner→mutate_bind(different)→sync works and uses the NEW
//        buffer (sever path). After sever the slot is `Severed`, so re-arming is
//        the `mutate_bind`-only path (a plain `bind` there is `SlotSevered`).
//    Locked slot re-arm/sever behaviour; we add the handle-identity assertion on
//    top.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn slot_rehome_vs_sever_handle_identity() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // (a) rehome: drop the Checkout → slot re-arms Lent→Bound, SAME handle.
    // Set-once bind (consuming) folded into `g`; re-`sync`'d / `mutate_bind`'d by `&`.
    let b = seeded(&ctx, 3);
    let h0 = handle_of(&b);
    let g = ks.scale_u32([N], slot!(Buf), 2u32).bind(Buf(b));
    let co1 = g.sync(&ctx).expect("run a1");
    assert_eq!(handle_of(&*co1), h0, "lent buffer keeps its handle in-run");
    drop(co1); // rehome.
    let co2 = g.sync(&ctx).expect("run a2 over re-armed slot");
    assert_eq!(
        handle_of(&*co2),
        h0,
        "rehome path: re-armed slot reuses the SAME cl_mem handle"
    );
    drop(co2);

    // (b) sever: into_inner keeps the value → slot Severed; re-arm via mutate_bind
    //     of a DIFFERENT buffer (its handle drives the run), and the result
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
        .mutate_bind(Buf(other))
        .expect("re-arm after sever via mutate_bind")
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
// 11. multi-output copy independence — through the SLOT path.
//     A multi-output copy `(Checkout<Src>, Checkout<Dst>)` whose dst is a
//     `slot!(Dst)` (so the 4-state slot machine governs it). Drop the src
//     Checkout (rehome → src re-arms with a STABLE handle) and `into_inner` the
//     dst Checkout (→ the dst slot is `Severed`). Assert the two sides are
//     INDEPENDENT:
//       - the severed dst rejects a plain `bind` (`SlotSevered`) but re-arms via
//         `mutate_bind` of a NEW dst, and a re-sync then copies into it;
//       - src is re-armed (same handle) across that re-sync — it was untouched by
//         the dst sever.
//
//     Why the dst is routed through a `slot!`: with a concrete copy dst, a
//     severed cell goes empty and a bare `Cell` can't tell "severed (→ re-alloc)"
//     from "lent / busy (→ error)". Routing the dst through a `slot!` gives
//     exactly that disambiguation (the `Severed` state), and the
//     `eager_copy_to(src, slot!(Dst))` copy-slot path threads its `SlotHome`.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn multi_output_copy_independence() {
    let Some(ctx) = ctx() else { return };

    let data: Vec<u32> = (0..N as u32).collect();
    let src = DeviceSlice::<u32>::from_slice(&ctx, &data).expect("src");
    let dst = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("dst");
    let hs = handle_of(&src);
    let hd = handle_of(&dst);

    // Concrete src + SLOT dst: the dst is governed by the 4-state slot machine.
    // Set-once bind (consuming) folded into `g`; re-`sync`'d / `mutate_bind`'d by `&`.
    let g = eager_copy_to(src, slot!(Dst)).bind(Dst(dst));

    let (co_src, co_dst) = g.sync(&ctx).expect("copy run 1");
    assert_eq!(handle_of(&*co_src), hs, "src handle stable run 1");
    assert_eq!(handle_of(&*co_dst), hd, "dst (slot) handle stable run 1");

    // Drop src (rehome → re-arm), into_inner dst (sever its slot + keep value).
    drop(co_src);
    let kept_dst = co_dst.into_inner();
    let mut out = vec![0u32; N];
    // `read` consumes the kept dst buffer (releasing it).
    kept_dst.read(&mut out).wait().expect("read kept dst");
    assert_eq!(out, data, "kept dst holds the copied data");

    // The dst slot is SEVERED → a plain (consuming, infallible) `bind` records
    // `SlotSevered`, surfacing DEFERRED at `sync` (nothing enqueued). Prove it on a
    // separate probe graph shaped the same way (testing it on `g` would consume `g`,
    // and we still want to prove `mutate_bind` re-arms `g`'s very slot below).
    {
        let psrc = DeviceSlice::<u32>::from_slice(&ctx, &data).expect("psrc");
        let pdst = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("pdst");
        let probe = eager_copy_to(psrc, slot!(Dst)).bind(Dst(pdst));
        let (ps, pd) = probe.sync(&ctx).expect("probe copy run");
        let _kept = pd.into_inner(); // sever the probe's dst slot
        drop(ps); // re-arm the concrete src (so ONLY the severed Dst is exercised)
        let err = probe
            .bind(Dst(DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("d2")))
            .sync(&ctx)
            .expect_err("plain bind on a severed copy-dst slot must error at sync");
        // Set-once bind onto a Severed cell records `SlotSevered` but leaves the cell
        // Severed, so `check_ready` (state-first) reports the completeness `SlotUnbound`
        // before draining the sink. Both are fail-closed.
        assert!(
            matches!(&err, Error::SlotUnbound(n) | Error::SlotSevered(n) if n.contains("Dst")),
            "expected deferred SlotSevered / state-first SlotUnbound naming Dst, got {err:?}"
        );
    }

    // Re-arm the severed dst with a NEW buffer via mutate_bind, then re-sync. src
    // must re-arm with its STABLE handle (independent of the dst sever); the new
    // dst is copied into and holds src's data.
    let dst2 = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("dst2");
    let hd2 = handle_of(&dst2);
    g.mutate_bind(Dst(dst2))
        .expect("mutate_bind re-arms the severed copy-dst slot");

    let (co_src2, co_dst2) = g
        .sync(&ctx)
        .expect("copy run 2: src re-armed, dst re-armed via mutate_bind");
    assert_eq!(
        handle_of(&*co_src2),
        hs,
        "src side independent: re-armed with the same handle after dst severed"
    );
    assert_eq!(
        handle_of(&*co_dst2),
        hd2,
        "dst side: the mutate_bound NEW buffer drives run 2"
    );
    let mut out2 = vec![0u32; N];
    co_dst2.read(&mut out2).wait().expect("read dst2");
    assert_eq!(out2, data, "run 2 copied src into the re-armed dst");
}

// ───────────────────────────────────────────────────────────────────────────
// 14. bundle2 over TWO caller-owned buffers, ×2 (S5 re-arm fix).
//     A REUSABLE bundle whose two branches each `fill` a distinct caller-owned
//     DeviceSlice in place. Each branch is single-output, so its lent buffer's
//     home threads through the branch pipe (`collect_home`/`put_home`) into its
//     per-element `Checkout`. Dropping the two Checkouts must return BOTH buffers
//     to their concrete cells (re-arm), so a SECOND `sync` of the same graph
//     succeeds with the SAME two handles. Before the fix the bundle dropped each
//     branch's home (`collect` + `put`), so run 2 found both cells empty →
//     spurious graph-busy.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn bundle2_caller_buffers_rearm_x2_stable_handles() {
    let Some(ctx) = ctx() else { return };

    // Two caller-owned buffers; pin their handles.
    let buf_a = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("buf_a");
    let buf_b = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("buf_b");
    let ha = handle_of(&buf_a);
    let hb = handle_of(&buf_b);

    // bundle2(fill(buf_a, 7), fill(buf_b, 9)) — both branches in place over a
    // caller-owned buffer. factor-free fill is idempotent, so the contents are
    // stable across runs and only the re-arm/handle invariant is under test.
    let g = bundle2(fill(buf_a, 7u32), fill(buf_b, 9u32));

    // Run 1.
    let (co_a, co_b) = g.sync(&ctx).expect("bundle run 1");
    assert_eq!(handle_of(&*co_a), ha, "run 1: branch a keeps its handle");
    assert_eq!(handle_of(&*co_b), hb, "run 1: branch b keeps its handle");
    // Borrowing reads (`map`, not the consuming `read`) so the Checkouts survive
    // to be dropped → rehomed. The buffers are still lent here.
    {
        let ga = co_a.map().wait().expect("map a run 1");
        let gb = co_b.map().wait().expect("map b run 1");
        assert!(ga.iter().all(|&v| v == 7), "run 1: branch a filled with 7");
        assert!(gb.iter().all(|&v| v == 9), "run 1: branch b filled with 9");
    }
    // Drop both Checkouts → BOTH buffers must re-arm their cells for run 2.
    drop((co_a, co_b));

    // Run 2 — the crux: the SAME graph re-runs only if both homes fired.
    let (co_a2, co_b2) = g
        .sync(&ctx)
        .expect("bundle run 2: both caller buffers must re-arm via their homes");
    assert_eq!(
        handle_of(&*co_a2),
        ha,
        "run 2: branch a re-armed with the SAME cl_mem"
    );
    assert_eq!(
        handle_of(&*co_b2),
        hb,
        "run 2: branch b re-armed with the SAME cl_mem"
    );
    {
        let ga2 = co_a2.map().wait().expect("map a run 2");
        let gb2 = co_b2.map().wait().expect("map b run 2");
        assert!(
            ga2.iter().all(|&v| v == 7),
            "run 2: idempotent fill(7) over the re-armed buffer a"
        );
        assert!(
            gb2.iter().all(|&v| v == 9),
            "run 2: idempotent fill(9) over the re-armed buffer b"
        );
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 15. fan_out over caller-owned buffers RE-ARMS per-branch, x2, stable handles
//     (#214 — the dynamic-`Vec` analog of #14's bundle re-arm). `FanOut`'s
//     terminal `Checkouts` is a `Vec<U::Checkouts>` — one `Checkout` per branch,
//     each threading its OWN return home — so dropping the Vec returns every
//     caller buffer to its cell and the SAME fan-out replays with stable `cl_mem`
//     handles, exactly like `bundle!` at fixed arity. (Was pinned as a documented
//     limitation `fan_out_caller_buffers_do_not_rearm`; FLIPPED here now that the
//     per-branch gather is implemented.)
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn fan_out_caller_buffers_rearm_x2_stable_handles() {
    let Some(ctx) = ctx() else { return };

    let buf_a = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("buf_a");
    let buf_b = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("buf_b");
    let ha = handle_of(&buf_a);
    let hb = handle_of(&buf_b);

    // Two in-place fill branches over caller-owned buffers, homogeneous → fan_out.
    // Idempotent fill, so contents are stable across runs; the re-arm/handle
    // invariant is what's under test.
    let g = fan_out(vec![buf_a, buf_b], |b| fill(b, 5u32));

    // Run 1: two branches → two Checkouts, each keeping its own handle.
    let cos = g.sync(&ctx).expect("fan_out run 1");
    assert_eq!(cos.len(), 2, "two branches → two Checkouts");
    assert_eq!(handle_of(&*cos[0]), ha, "run 1: branch 0 keeps its handle");
    assert_eq!(handle_of(&*cos[1]), hb, "run 1: branch 1 keeps its handle");
    {
        let g0 = cos[0].map().wait().expect("map branch 0 run 1");
        let g1 = cos[1].map().wait().expect("map branch 1 run 1");
        assert!(g0.iter().all(|&v| v == 5), "run 1: branch 0 filled with 5");
        assert!(g1.iter().all(|&v| v == 5), "run 1: branch 1 filled with 5");
    }
    // Drop the Checkouts → BOTH branch buffers re-arm their cells (each Checkout
    // carries its own home now).
    drop(cos);

    // Run 2 — the crux: the SAME fan-out re-runs only if both branch homes fired,
    // over the SAME cl_mem handles.
    let cos2 = g
        .sync(&ctx)
        .expect("fan_out run 2: both caller buffers must re-arm via their homes");
    assert_eq!(cos2.len(), 2);
    assert_eq!(
        handle_of(&*cos2[0]),
        ha,
        "run 2: branch 0 re-armed with the SAME cl_mem"
    );
    assert_eq!(
        handle_of(&*cos2[1]),
        hb,
        "run 2: branch 1 re-armed with the SAME cl_mem"
    );
    {
        let g0 = cos2[0].map().wait().expect("map branch 0 run 2");
        let g1 = cos2[1].map().wait().expect("map branch 1 run 2");
        assert!(
            g0.iter().all(|&v| v == 5),
            "run 2: idempotent fill(5) over the re-armed buffer 0"
        );
        assert!(
            g1.iter().all(|&v| v == 5),
            "run 2: idempotent fill(5) over the re-armed buffer 1"
        );
    }
}
