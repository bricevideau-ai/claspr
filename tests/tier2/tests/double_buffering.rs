//! Double-buffering (ping-pong) — the canonical integration test for the slot
//! `mutate_bind` verb.
//!
//! ## The pattern
//!
//! Iterative ping-pong over two device buffers A and B. A graph reads one buffer
//! and writes the *other*; each step swaps their roles so this step reads what
//! the last step wrote. This is the classic stencil / Jacobi shape (read the old
//! grid, write the new one, then flip). Here the per-step transform is
//! `out[i] = in[i] + 1`, realised with `add_u32(in, ones, out)` against a
//! PERSISTENT `ones` operand (a third buffer of all-`1`s that stays fixed across
//! every step). After K steps the live result buffer holds `initial + K`.
//!
//! ## Why this is THE `mutate_bind` test
//!
//! The crossed re-bind between steps is `mutate_bind`, NOT `bind`, and that is the
//! whole point. After a step we call `into_inner()` on both the In and Out
//! Checkouts to KEEP their buffers (we need them for the next step) — and
//! `into_inner` *severs* the slot (`Lent → Severed`). A set-once `bind` on a
//! `Severed` slot is `Error::SlotSevered`; only `mutate_bind` re-arms it. So the
//! swap is only expressible via `mutate_bind` — `bind` cannot ping-pong. The
//! second test (`..._plain_bind_after_sever_rejected`) pins exactly that: a plain
//! `bind` after the sever returns `Err(SlotSevered)`.
//!
//! ## The constant `ones` operand
//!
//! `add_u32` is a 3-arg / 3-output kernel: `sync` yields `(in_co, ones_co,
//! out_co)`. The `ones` slot is bound once up front and never changes. Each step
//! we simply `drop(ones_co)`, which re-arms its slot (`Lent → Bound`) with the
//! SAME buffer — no rebind needed, and the next `sync` reuses it. Only the In/Out
//! pair is severed-and-swapped.
//!
//! ## Handle stability (exactly two buffers)
//!
//! A correct ping-pong recycles exactly the SAME two `cl_mem` objects across all
//! steps — the buffers swap roles but nothing is re-allocated per step. We capture
//! `handle_of(A)` and `handle_of(B)` up front and assert that on every step each
//! of the in/out handles is drawn from the fixed pair `{hA, hB}` and the two are
//! distinct from each other — together those two checks pin the ping-pong to
//! recycling exactly those two `cl_mem`s (no per-step alloc). (Both A and B stay
//! live across the loop via `into_inner`, so their handles are stable identities —
//! no cl_mem recycling confound.) This runs entirely on the existing `sync()`
//! reuse path; no command-buffer backend is involved.

use claspr::eager::DeviceOpExt;
use claspr::{Context, DeviceSlice, Error, MemRef, RecordableBuffer};
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

// Tags for the three slot positions of `add_u32(In, Ones, Out)`.
slots! {
    In: DeviceSlice<u32>,
    Ones: DeviceSlice<u32>,
    Out: DeviceSlice<u32>,
}

/// The core ping-pong test. `K` iterations of `out = in + 1` with a crossed
/// `mutate_bind` swap each step. Asserts BOTH:
///
/// - **Correctness**: after K steps the live result buffer holds `initial + K`.
/// - **Handle stability / exactly-two-buffers**: across every step the in/out
///   handles are drawn from exactly `{hA, hB}` and are distinct — the two
///   `cl_mem` objects are recycled, never re-allocated.
#[test]
fn double_buffer_ping_pong_computes_and_handles_stable() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    const INITIAL: u32 = 10;
    const K: usize = 4;

    // The two ping-pong buffers and the fixed `ones` operand.
    let a = seeded(&ctx, INITIAL); // step 0 reads A
    let b = seeded(&ctx, 0); // step 0 writes B (overwritten by the kernel)
    let ones = seeded(&ctx, 1); // f(x) = x + 1

    // The two cl_mem identities the ping-pong must recycle for all K steps.
    let ha = handle_of(&a);
    let hb = handle_of(&b);
    assert_ne!(ha, hb, "A and B must be distinct buffers to begin with");

    // out = In + Ones. Three slots; `add_u32` is 3-output.
    let g = ks.add_u32([N], slot!(In), slot!(Ones), slot!(Out));

    // Step 0: read A, write B. `bind` (set-once) on the virgin slots.
    g.bind(In(a)).expect("bind In=A");
    g.bind(Ones(ones)).expect("bind Ones");
    g.bind(Out(b)).expect("bind Out=B");

    let (mut in_co, mut ones_co, mut out_co) = g.sync(&ctx).expect("step 0 sync");

    // After step 0: B = A + 1 = INITIAL + 1. The freshly-written buffer is `out`.
    // Track which buffer currently holds the latest result (for the final read).
    let mut steps_done = 1usize;

    for step in 0..K {
        // Handle invariant for THIS step: in/out are exactly the two buffers,
        // distinct from each other.
        let h_in = handle_of(&*in_co);
        let h_out = handle_of(&*out_co);
        assert!(
            (h_in == ha || h_in == hb) && (h_out == ha || h_out == hb),
            "step {step}: in/out handles must be drawn from the two ping-pong \
             buffers {{hA,hB}} (in={h_in:#x}, out={h_out:#x}, hA={ha:#x}, hB={hb:#x})"
        );
        assert_ne!(
            h_in, h_out,
            "step {step}: ping-pong must read and write DIFFERENT buffers"
        );

        if step + 1 == K {
            break; // step 0 already ran above; we run K syncs total.
        }

        // SWAP. `into_inner` keeps each buffer AND severs its slot (Lent →
        // Severed): the prior `out` (latest result) becomes the next `in`, and the
        // prior `in` (now stale) becomes the next scratch `out`.
        let next_in = out_co.into_inner();
        let next_out = in_co.into_inner();
        // The `ones` operand never changes: dropping its Checkout re-arms the slot
        // (Lent → Bound) with the SAME buffer, so the next sync reuses it.
        drop(ones_co);

        // The crossed re-bind. MUST be `mutate_bind`: both slots are `Severed`, so
        // a set-once `bind` here would be `Error::SlotSevered`.
        g.mutate_bind(In(next_in)).expect("mutate_bind In (swap)");
        g.mutate_bind(Out(next_out))
            .expect("mutate_bind Out (swap)");

        let next = g.sync(&ctx).expect("swap-step sync");
        in_co = next.0;
        ones_co = next.1;
        out_co = next.2;
        steps_done += 1;
    }

    assert_eq!(steps_done, K, "ran exactly K steps");

    // The live result is in `out_co` (the buffer last written). After K steps of
    // `+1` over INITIAL: INITIAL + K.
    let want = INITIAL + K as u32;
    let mut result = vec![0u32; N];
    out_co.read(&mut result).wait().expect("read final result");
    assert!(
        result.iter().all(|&v| v == want),
        "after K={K} ping-pong steps of +1 over {INITIAL}, every cell must be \
         {want}; got {:?}",
        &result[..8]
    );
}

/// The **one-line swap**: identical ping-pong to the test above, but the crossed
/// re-bind binds the `Checkout`s DIRECTLY into the slots — `mutate_call((In(out_co),
/// Out(in_co)))` — with NO manual `into_inner()`. Binding a `Checkout` into a slot
/// severs the Checkout's source home (`Lent → Severed`) and the target slot adopts
/// the buffer, so the four-line extract-then-rebind collapses to one crossing.
///
/// Proves the ergonomic end-to-end: it computes the SAME `initial + K` AND recycles
/// the SAME two `cl_mem` objects (handles stay the fixed `{hA, hB}` pair).
#[test]
fn double_buffer_one_line_swap_computes_and_handles_stable() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    const INITIAL: u32 = 10;
    const K: usize = 4;

    let a = seeded(&ctx, INITIAL);
    let b = seeded(&ctx, 0);
    let ones = seeded(&ctx, 1);

    let ha = handle_of(&a);
    let hb = handle_of(&b);
    assert_ne!(ha, hb, "A and B must be distinct buffers to begin with");

    let g = ks.add_u32([N], slot!(In), slot!(Ones), slot!(Out));

    // Step 0: set-once on the virgin slots.
    g.call((In(a), Ones(ones), Out(b))).expect("call step 0");

    let (mut in_co, mut ones_co, mut out_co) = g.sync(&ctx).expect("step 0 sync");
    let mut steps_done = 1usize;

    for step in 0..K {
        let h_in = handle_of(&*in_co);
        let h_out = handle_of(&*out_co);
        assert!(
            (h_in == ha || h_in == hb) && (h_out == ha || h_out == hb),
            "step {step}: in/out handles must be drawn from {{hA,hB}} \
             (in={h_in:#x}, out={h_out:#x}, hA={ha:#x}, hB={hb:#x})"
        );
        assert_ne!(
            h_in, h_out,
            "step {step}: ping-pong must read and write DIFFERENT buffers"
        );

        if step + 1 == K {
            break;
        }

        // The `ones` operand never changes: drop re-arms its slot (Lent → Bound).
        drop(ones_co);

        // THE ONE-LINE CROSSING. Bind the Checkouts directly: each severs its
        // source slot (Lent → Severed) and the target slot adopts the buffer. The
        // prior `out` becomes the next `in`; the prior `in` becomes the next `out`.
        // `mutate_call` because both targets are Severed (a set-once `call` would be
        // `Error::SlotSevered`).
        g.mutate_call((In(out_co), Out(in_co)))
            .expect("one-line crossed mutate_call");

        let next = g.sync(&ctx).expect("swap-step sync");
        in_co = next.0;
        ones_co = next.1;
        out_co = next.2;
        steps_done += 1;
    }

    assert_eq!(steps_done, K, "ran exactly K steps");

    let want = INITIAL + K as u32;
    let mut result = vec![0u32; N];
    out_co.read(&mut result).wait().expect("read final result");
    assert!(
        result.iter().all(|&v| v == want),
        "after K={K} one-line-swap steps of +1 over {INITIAL}, every cell must be \
         {want}; got {:?}",
        &result[..8]
    );
}

/// Locks WHY `mutate_bind` is required: after one step + `into_inner` of both
/// In/Out, a PLAIN `bind` (set-once) on either slot returns `Err(SlotSevered)`.
/// This proves the ping-pong swap genuinely needs `mutate_bind` — `bind` cannot
/// re-arm a severed slot, so it cannot express the crossed re-bind.
#[test]
fn double_buffer_plain_bind_after_sever_rejected() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let a = seeded(&ctx, 10);
    let b = seeded(&ctx, 0);
    let ones = seeded(&ctx, 1);

    let g = ks.add_u32([N], slot!(In), slot!(Ones), slot!(Out));
    g.bind(In(a)).expect("bind In=A");
    g.bind(Ones(ones)).expect("bind Ones");
    g.bind(Out(b)).expect("bind Out=B");

    let (in_co, ones_co, out_co) = g.sync(&ctx).expect("step 0 sync");

    // Sever both In and Out (keep the buffers, as the ping-pong loop does).
    let kept_in = in_co.into_inner();
    let kept_out = out_co.into_inner();
    drop(ones_co);

    // A plain set-once `bind` of the crossed buffers must REJECT: the slots are
    // `Severed`, not virgin. Re-providing a buffer is a CHANGE, not a first
    // declaration. The Ok arm is `&Op` (not `Debug`), so match by hand.
    match g.bind(In(kept_out)) {
        Ok(_) => panic!("plain bind on a severed In slot must error (needs mutate_bind)"),
        Err(Error::SlotSevered(name)) => assert!(
            name.contains("In"),
            "SlotSevered should name the tag `In`, got {name:?}"
        ),
        Err(other) => panic!("expected Error::SlotSevered, got {other:?}"),
    }

    // And the same for Out — both legs of the swap need `mutate_bind`.
    match g.bind(Out(kept_in)) {
        Ok(_) => panic!("plain bind on a severed Out slot must error (needs mutate_bind)"),
        Err(Error::SlotSevered(name)) => assert!(
            name.contains("Out"),
            "SlotSevered should name the tag `Out`, got {name:?}"
        ),
        Err(other) => panic!("expected Error::SlotSevered, got {other:?}"),
    }
}
