//! Typed-slots proof test (step (b): `slots!` / `slot!` / `g.call(Tag(v))`).
//!
//! A reusable graph can carry **unbound typed holes** — `slot!(Tag)` — that plug
//! into the same positions a concrete buffer does (here: kernel buffer args).
//! `g.call(Tag(value))` folds a `TypeId → resource` binding into the graph's slot
//! cells (order-free, curryable, partial). Completeness is enforced only at
//! `sync` (runtime): an unbound slot is `Error::SlotUnbound`. After a run the
//! Checkout returns the buffer to its slot cell (re-arm), so a bound graph
//! re-runs. These tests lock the five properties from the step-(b) spec:
//!
//! (a) `slot!` + `g.call(Tag(v)).sync()` produces correct data.
//! (b) order-free — both bind orders give the same result.
//! (c) re-run a bound graph twice (the slot re-arms like a concrete cell).
//! (d) an unbound slot makes `sync` return the "slot unbound" `Err`.
//! (e) a second `call(Tag(other))` rebinds — the new buffer's result wins.

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
}

/// Allocate + fill a `DeviceSlice<u32>` of `N` elements with `v`.
fn seeded(ctx: &Context, v: u32) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::alloc_zero(ctx, N)
        .expect("alloc")
        .fill(v)
        .wait()
        .expect("seed")
}

/// (a) `slot!(Buf)` in a kernel arg position; `g.call(Buf(b)).sync()` runs the
/// graph over the bound buffer and produces the expected data.
#[test]
fn slot_bind_then_sync_produces_data() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // scale(slot, 2) -> download. The slot is the in-place scale target.
    let g = ks.scale_u32([N], slot!(Buf), 2u32).and_then(download);

    let b = seeded(&ctx, 3); // 3 * 2 = 6
    let out = g.call(Buf(b)).sync(&ctx).expect("bound sync");
    assert!(
        out.iter().all(|&v| v == 6),
        "scale(slot=3, 2) should be 6, got {:?}",
        &out[..8]
    );
}

/// (b) Binding is **order-free**: `call(A).call(B).call(Out)` and the reverse
/// order produce the same result. Each `call` carries one tag, folded
/// independently.
#[test]
fn slot_bind_is_order_free() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // out = a + b, with all three operands as slots. `add_u32` is multi-output:
    // sync yields (Checkout<a>, Checkout<b>, Checkout<out>). Read `out`.
    let forward = ks.add_u32([N], slot!(A), slot!(B), slot!(Out));
    let (_a, _b, out_co) = forward
        .call(A(seeded(&ctx, 2)))
        .call(B(seeded(&ctx, 5)))
        .call(Out(seeded(&ctx, 0)))
        .sync(&ctx)
        .expect("forward-order sync");
    let mut r1 = vec![0u32; N];
    out_co.read(&mut r1).wait().expect("read forward");
    assert!(r1.iter().all(|&v| v == 7), "2 + 5 = 7, got {:?}", &r1[..8]);

    // Same graph shape, reverse bind order — identical result.
    let reverse = ks.add_u32([N], slot!(A), slot!(B), slot!(Out));
    let (_a, _b, out_co) = reverse
        .call(Out(seeded(&ctx, 0)))
        .call(B(seeded(&ctx, 5)))
        .call(A(seeded(&ctx, 2)))
        .sync(&ctx)
        .expect("reverse-order sync");
    let mut r2 = vec![0u32; N];
    out_co.read(&mut r2).wait().expect("read reverse");
    assert!(
        r2.iter().all(|&v| v == 7),
        "reverse bind order must match: 2 + 5 = 7, got {:?}",
        &r2[..8]
    );
}

/// (c) A bound graph is **re-runnable**: after a run, dropping the Checkout
/// returns the buffer to its slot cell (re-arm), so a second `sync` runs again.
/// `scale` is in place, so it compounds over the same buffer (3 -> 6 -> 12),
/// proving the slot cell carried the buffer across runs.
#[test]
fn bound_slot_graph_reruns() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    // No download: the output IS the buffer, so its Checkout re-arms the slot.
    let g = ks.scale_u32([N], slot!(Buf), 2u32);

    let b = seeded(&ctx, 3);
    // Run 1: 3 -> 6. Drop the Checkout to re-arm the slot.
    let co1 = g.call(Buf(b)).sync(&ctx).expect("run 1");
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

    // Never `call`'d — the slot stays empty.
    let g = ks.scale_u32([N], slot!(Buf), 2u32).and_then(download);

    let err = g.sync(&ctx).expect_err("unbound slot must error at sync");
    match err {
        Error::SlotUnbound(name) => {
            assert!(
                name.contains("Buf"),
                "unbound-slot error should name the tag (`Buf`), got {name:?}"
            );
        }
        other => panic!("expected Error::SlotUnbound, got {other:?}"),
    }

    // The same graph runs once bound — proves the slot, not the graph, was at fault.
    let out = g.call(Buf(seeded(&ctx, 4))).sync(&ctx).expect("now bound");
    assert!(
        out.iter().all(|&v| v == 8),
        "4 * 2 = 8, got {:?}",
        &out[..8]
    );
}

/// (e) A second `call(Tag(other))` **rebinds** the slot to a different buffer; the
/// new buffer's data drives the result (the previous binding is replaced).
#[test]
fn rebind_slot_uses_new_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let g = ks.scale_u32([N], slot!(Buf), 2u32).and_then(download);

    // First binding: 3 -> 6.
    let first = g.call(Buf(seeded(&ctx, 3))).sync(&ctx).expect("first bind");
    assert!(
        first.iter().all(|&v| v == 6),
        "first buffer: 3 * 2 = 6, got {:?}",
        &first[..8]
    );
    drop(first);

    // Rebind to a DIFFERENT buffer (seeded 10): 10 -> 20. The first buffer (whose
    // consumed-by-download cell is empty anyway) is replaced by this call.
    let second = g.call(Buf(seeded(&ctx, 10))).sync(&ctx).expect("rebind");
    assert!(
        second.iter().all(|&v| v == 20),
        "rebound buffer: 10 * 2 = 20, got {:?}",
        &second[..8]
    );
}
