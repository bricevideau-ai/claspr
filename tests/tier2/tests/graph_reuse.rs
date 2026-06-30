//! Reusable-graph proof test (step (a): own-the-buffers reuse + multi-output).
//!
//! The op-tree `g` IS the reusable graph: `g.sync(&ctx)` returns a `Checkout`
//! over the run's output; reading happens through the `Checkout`, and on its
//! drop any LENT concrete buffer returns to its cell (re-arming `g`). These
//! tests lock the four correctness properties of the model:
//!
//! 1. **Idempotent reseed** — a mint-and-consume graph (`upload→scale→download`)
//!    gives the SAME result on every `sync`; the `upload` op re-seeds its buffer
//!    each run (it does NOT compound).
//! 2. **Multi-output** — a multi-output kernel yields a TUPLE of `Checkout`s
//!    (`(Checkout<A>, Checkout<B>, Checkout<Out>)`), each independently readable /
//!    `into_inner`'d / returned-on-drop.
//! 3. **`into_inner`** — permanently extracts the output buffer.
//! 4. **Graph busy** — a second `sync` while a `Checkout` holds a still-lent
//!    concrete buffer errors at runtime.
//! 5. **Per-output re-arm (`add(a, b, out)`)** — the home-in-pipe provenance
//!    re-arms EVERY same-typed concrete cell, which the old single-guard /
//!    type-match heuristic could not.

use claspr::Context;
use claspr::DeviceSlice;
use claspr::eager::{DeviceOpExt, download, upload};
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

/// Property 1: a mint-and-consume graph is **idempotent** — `sync`'d twice it
/// gives the same result both runs (the `upload` op re-seeds its buffer each
/// run; the mutable buffer does NOT compound).
#[test]
fn reused_graph_is_idempotent() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // upload(vec![1; N]) -> scale ×2 -> download. Expect all 2s on BOTH runs.
    let g = upload(vec![1u32; N])
        .and_then(|buf| ks.scale_u32([N], buf, 2u32))
        .and_then(download);

    let run1 = g.sync(&ctx).expect("run 1");
    assert!(
        run1.iter().all(|&v| v == 2),
        "run 1 expected all 2, got {:?}",
        &run1[..8]
    );
    drop(run1); // release the run (re-arm g)

    let run2 = g.sync(&ctx).expect("run 2");
    assert!(
        run2.iter().all(|&v| v == 2),
        "run 2 must MATCH run 1 (idempotent reseed, not compounding), got {:?}",
        &run2[..8]
    );
}

/// Property 1b: the same with an explicit re-bound (`let` rebind) between runs,
/// and a third run, to make the "no compounding" guarantee unmistakable.
#[test]
fn reused_graph_three_runs_no_compounding() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = upload(vec![3u32; N])
        .and_then(|buf| ks.scale_u32([N], buf, 5u32)) // 3 -> 15 each run
        .and_then(download);

    for run in 0..3 {
        let out = g.sync(&ctx).unwrap_or_else(|e| panic!("run {run}: {e}"));
        assert!(
            out.iter().all(|&v| v == 15),
            "run {run}: expected all 15, got {:?}",
            &out[..8]
        );
    }
}

/// Property 2: a multi-output kernel yields a TUPLE of `Checkout`s — one per
/// output — each independently readable. The transparency layer lets a
/// `Checkout<DeviceSlice>` be `.read()` directly (no explicit `into_inner`).
#[test]
fn multi_output_checkout_reads_each_output() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let a = DeviceSlice::<u32>::alloc_zero(&ctx, N)
        .expect("a")
        .fill(3u32)
        .wait()
        .expect("seed a");
    let b = DeviceSlice::<u32>::alloc_zero(&ctx, N)
        .expect("b")
        .fill(4u32)
        .wait()
        .expect("seed b");
    let out = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("out");

    // add_u32(a, b, out) -> Checkouts = (Checkout<a>, Checkout<b>, Checkout<out>).
    let g = ks.add_u32([N], a, b, out);
    let (a_co, _b_co, out_co) = g.sync(&ctx).expect("sync add");

    // Transparency: `read` works DIRECTLY on a `Checkout<DeviceSlice>` (it severs
    // the return and reads) — no `.into_inner()` needed.
    let mut out_rb = vec![0u32; N];
    out_co.read(&mut out_rb).wait().expect("read out");
    assert!(
        out_rb.iter().all(|&v| v == 7),
        "out should be a+b = 7, got {:?}",
        &out_rb[..8]
    );

    // Each input is its own independent Checkout — read one without touching the
    // others.
    let mut a_rb = vec![0u32; N];
    a_co.read(&mut a_rb).wait().expect("read a");
    assert!(a_rb.iter().all(|&v| v == 3), "a still 3");
}

/// Property 5: a same-typed multi-buffer kernel `add(a, b, out)` re-arms EVERY
/// concrete cell on Checkout drop — the case the OLD type-match heuristic could
/// NOT handle (three `DeviceSlice<u32>` lent inputs are ambiguous by type alone,
/// so the heuristic re-armed none → second `sync` would error "busy"). With
/// home-in-pipe each output carries its own home, so all three re-arm and the
/// graph is runnable repeatedly.
#[test]
fn add_three_buffers_each_rearms() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let a = DeviceSlice::<u32>::alloc_zero(&ctx, N)
        .expect("a")
        .fill(2u32)
        .wait()
        .expect("seed a");
    let b = DeviceSlice::<u32>::alloc_zero(&ctx, N)
        .expect("b")
        .fill(5u32)
        .wait()
        .expect("seed b");
    let out = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("out");

    // out = a + b = 7. a, b are read-only inputs; out is written in place. All
    // three are lent concrete `DeviceSlice<u32>` cells.
    let g = ks.add_u32([N], a, b, out);

    // Run twice, each time DROPPING all three Checkouts (return-on-drop) so every
    // cell re-arms. If any of the three same-typed cells failed to re-arm, the
    // NEXT `sync` would error "busy". Three clean runs prove all three re-arm.
    for run in 0..3 {
        let checkouts = g
            .sync(&ctx)
            .unwrap_or_else(|e| panic!("run {run}: graph must re-arm all 3 cells: {e}"));
        // Drop all three Checkouts → a, b, out cells all re-armed.
        drop(checkouts);
    }

    // Final run: read `out` to confirm the value is correct over the re-armed
    // buffers (add is idempotent: 2 + 5 = 7 every run).
    let (_a, _b, out_co) = g.sync(&ctx).expect("final run");
    let mut rb = vec![0u32; N];
    out_co.read(&mut rb).wait().expect("read out");
    assert!(
        rb.iter().all(|&v| v == 7),
        "out should be a+b = 7 over re-armed buffers, got {:?}",
        &rb[..8]
    );
}

/// Property 3: `into_inner()` permanently extracts a buffer from the Checkout.
#[test]
fn into_inner_extracts_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // A concrete buffer is LENT into the graph; `into_inner` keeps it.
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N)
        .expect("alloc")
        .fill(6u32)
        .wait()
        .expect("seed");

    let g = ks.scale_u32([N], buf, 2u32); // 6 -> 12, in place

    let scaled: DeviceSlice<u32> = g.sync(&ctx).expect("sync").into_inner();
    // The extracted buffer is fully usable (it was severed from g).
    let mut rb = vec![0u32; N];
    scaled.read(&mut rb).wait().expect("read");
    assert!(
        rb.iter().all(|&v| v == 12),
        "extracted buffer should hold 12, got {:?}",
        &rb[..8]
    );
}

/// Property 4: a second `sync` while a `Checkout` holds a still-lent concrete
/// buffer errors at runtime ("graph busy"). The graph lends `buf`; while the
/// first Checkout is alive the cell is empty, so a second `sync` fails fast.
#[test]
fn second_sync_while_checked_out_is_busy_error() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N)
        .expect("alloc")
        .fill(1u32)
        .wait()
        .expect("seed");

    let g = ks.scale_u32([N], buf, 2u32);

    let live = g.sync(&ctx).expect("first sync");
    // Second sync while `live` is alive: the lent buffer's cell is empty.
    let busy = g.sync(&ctx);
    assert!(
        busy.is_err(),
        "second sync while a Checkout is alive must error (graph busy)"
    );

    // After the first Checkout drops, the buffer returns to its cell and the
    // graph is runnable again.
    drop(live);
    let again = g.sync(&ctx).expect("graph re-armed after Checkout drop");
    let mut rb = vec![0u32; N];
    again
        .into_inner()
        .read(&mut rb)
        .wait()
        .expect("read after re-arm");
    // Single-output scale: the buffer was re-armed with its prior (scaled)
    // contents — note the SECOND run scales again (2 -> 4). This documents that a
    // concrete in-place buffer compounds across runs (it is the same buffer); use
    // an `upload` reseed for idempotence (property 1).
    assert!(rb.iter().all(|&v| v == 4), "got {:?}", &rb[..8]);
}

/// A concrete buffer downloaded into a host `Vec` is REHOMED, not released —
/// "homeless is never legitimate". `download` reads the device buffer into a Vec
/// but returns the device buffer to its origin cell, so a reused `download` graph
/// re-runs over the SAME backing handle. (Pre-invariant, `download` discarded the
/// home and dropped the buffer, leaving the cell empty → second sync errored. This
/// test was updated to the new, correct rehome-on-consume behaviour.)
#[test]
fn concrete_consumed_by_download_is_rehomed() {
    let Some(ctx) = ctx() else { return };

    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N)
        .expect("alloc")
        .fill(9u32)
        .wait()
        .expect("seed");

    // A graph that downloads buf. The device buffer is rehomed on the run's drop
    // (its output is a host Vec; the buffer itself returns to its concrete cell).
    let g = download::<u32, _>(buf);
    let out = g.sync(&ctx).expect("download once");
    assert!(out.iter().all(|&v| v == 9));
    drop(out);

    // The buffer was RETURNED to its cell, so a second run reads the SAME buffer
    // again (no fresh alloc, no "busy").
    let again = g
        .sync(&ctx)
        .expect("a downloaded concrete buffer re-arms via rehome; second sync works");
    assert!(
        again.iter().all(|&v| v == 9),
        "second download reads the rehomed buffer: still 9"
    );
}
