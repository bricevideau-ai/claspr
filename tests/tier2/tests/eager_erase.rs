//! `.erase()` / `DeviceDynOp` — transparent type erasure.
//!
//! `.and_then` erases the chain it extends, so a chain's type nests per
//! combinator, not per link (it used to nest per link and hit the default
//! `recursion_limit = 128` at ~110 links, E0275 on `AndThen<…>: Send`). Explicit
//! `.erase()` remains for unifying conditional arms. These tests pin that a long
//! chain compiles and runs, and that erasure is TRANSPARENT: slots inside still bind and
//! re-bind, commands inside still record into one command buffer, per-output
//! handles survive, and the graph's description is unchanged.
//!
//! **Parity guard.** `DeviceDynOp` forwards each `DeviceOp` member by hand
//! through a private mirror trait; an unforwarded member with a default body
//! would compile and silently run the default for every erased op. The
//! `parity_*` tests build each graph twice — plain and erased — and compare
//! every member they can observe, before and after running. When you add a
//! `DeviceOp` member, add it to `assert_static_parity` (or the run-time checks).

use claspr::DeviceSlice;
use claspr::eager::{DeviceOp, DeviceOpExt, bundle2, download, fill};
use claspr::{slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::{ctx, homed_cb};

slots! {
    Factor: u32,
}

const N: usize = 64;

/// Sixteen `x = y + b` links over a 3-buffer caravan, rotating the roles so each
/// link adds `b` (= 1) to the running total: after L links the out buffer holds L.
macro_rules! sixteen_links {
    ($g:expr, $ks:ident) => {
        $g.and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
            .and_then(move |(x, b, y)| $ks.add_u32([N], y, b, x))
    };
}

/// 1 + 8×16 = 129 three-output kernel links: past the ~110-link ceiling the
/// chain had before `and_then` erased by default. The explicit per-segment
/// `.erase()` calls below are now redundant (kept to exercise them). Built and
/// run on a default-sized (2 MiB) spawned thread — the stack a `cargo test`
/// thread gets — to pin the debug-build stack cost too.
#[test]
fn long_chain_compiles_and_runs_at_default_limit() {
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let Some(ctx) = ctx() else { return };
            let ks = kernels::kernels(&ctx).expect("load kernels");
            let ks = &ks;
            let x = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("x");
            let b = DeviceSlice::<u32>::from_vec(&ctx, vec![1u32; N]).expect("b");
            let y = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("y");

            let g = ks.add_u32([N], x, b, y).erase();
            let g = sixteen_links!(g, ks).erase();
            let g = sixteen_links!(g, ks).erase();
            let g = sixteen_links!(g, ks).erase();
            let g = sixteen_links!(g, ks).erase();
            let g = sixteen_links!(g, ks).erase();
            let g = sixteen_links!(g, ks).erase();
            let g = sixteen_links!(g, ks).erase();
            let g = sixteen_links!(g, ks).erase();
            // The erased chain still hands downstream all three per-output pipes.
            let g = g.and_then(|(_x, _b, y)| download(y));

            let out = g.sync(&ctx).expect("sync");
            assert!(out.iter().all(|&v| v == 129), "129 links: {:?}", &out[..4]);
        })
        .expect("spawn")
        .join()
        .expect("erased 129-link chain ran on a 2 MiB stack");
}

/// A slot INSIDE an erased segment binds, re-binds, and invalidates the command
/// buffer it was recorded into — the erased op forwards `bind_slots`,
/// `apply_binding_delta` and the CB cache to the op it wraps.
#[test]
fn erased_segment_forwards_slots_and_command_buffers() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    // fill 1 -> ×Factor (erased segment) -> ×2.
    let g = fill(buf, 1u32)
        .and_then(|b| ks.scale_u32([N], b, slot!(Factor)))
        .erase()
        .and_then(|b| ks.scale_u32([N], b, 2u32));

    for factor in [3u32, 5, 5, 7] {
        g.mutate_bind(Factor(factor)).expect("bind factor");
        let co = g.sync(&ctx).expect("sync");
        let got = co.map().wait().expect("read");
        assert!(
            got.iter().all(|&v| v == 2 * factor),
            "factor {factor}: {:?}",
            &got[..4]
        );
        drop(got);
        drop(co);
        if ctx.has_cl_khr_command_buffer() {
            assert!(homed_cb(&g), "erased chain still records a command buffer");
        }
    }
}

/// Erasure adds no node and changes no structure: the description of an erased
/// chain equals the unerased one's.
#[test]
fn erasure_is_invisible_to_description() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let mk = || DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    let plain = fill(mk(), 1u32)
        .and_then(|b| ks.scale_u32([N], b, 3u32))
        .and_then(|b| ks.scale_u32([N], b, 2u32));
    let erased = fill(mk(), 1u32)
        .and_then(|b| ks.scale_u32([N], b, 3u32))
        .erase()
        .and_then(|b| ks.scale_u32([N], b, 2u32))
        .erase();

    assert_eq!(plain.description(), erased.description());
    assert_eq!(plain.cbable_weight(), erased.cbable_weight());
}

// ── Parity: erased vs. plain twins, member by member ────────────────────────

/// Every `DeviceOp` member observable without running, compared between a plain
/// graph and its erased twin. `dump_graph`'s pipe cell ids differ between twins
/// (each build mints fresh pipes), so the structural projection drops them.
fn assert_static_parity<P: DeviceOp, E: DeviceOp>(plain: &P, erased: &E) {
    type Node = (usize, String, usize, usize, bool, bool);
    fn structure<O: DeviceOp>(op: &O) -> Vec<Node> {
        let mut nodes = Vec::new();
        op.dump_graph(0, &mut nodes);
        nodes
            .into_iter()
            .map(|n| {
                let (outs, ins) = (n.out_cells.len(), n.in_cells.len());
                (n.depth, n.name, outs, ins, n.cb_addable, n.seam)
            })
            .collect()
    }
    fn cb_ids<O: DeviceOp>(op: &O) -> usize {
        let mut ids = Vec::new();
        op.collect_cb_ids(&mut ids);
        ids.len()
    }
    assert_eq!(plain.description(), erased.description(), "describe");
    assert_eq!(plain.node_label(), erased.node_label(), "node_label");
    assert_eq!(structure(plain), structure(erased), "dump_graph");
    assert_eq!(
        plain.output_pipe().is_some(),
        erased.output_pipe().is_some(),
        "output_pipe"
    );
    assert_eq!(
        plain.check_ready().is_ok(),
        erased.check_ready().is_ok(),
        "check_ready"
    );
    assert_eq!(
        plain.contains_host_seam(),
        erased.contains_host_seam(),
        "contains_host_seam"
    );
    assert_eq!(plain.cb_addable(), erased.cb_addable(), "cb_addable");
    assert_eq!(
        plain.cbable_weight(),
        erased.cbable_weight(),
        "cbable_weight"
    );
    assert_eq!(
        plain.cb_spine_head_addable(),
        erased.cb_spine_head_addable(),
        "cb_spine_head_addable"
    );
    assert_eq!(
        plain.cb_cache().is_some(),
        erased.cb_cache().is_some(),
        "cb_cache"
    );
    assert_eq!(cb_ids(plain), cb_ids(erased), "collect_cb_ids");
}

/// A CB-recordable kernel chain with a slot: `bind_slots`, `check_ready` (unbound
/// → error on both), the CB members, and `apply_binding_delta` via `mutate_bind`.
#[test]
fn parity_slot_kernel_chain() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let mk = || DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    let plain = fill(mk(), 1u32)
        .and_then(|b| ks.scale_u32([N], b, slot!(Factor)))
        .and_then(|b| ks.scale_u32([N], b, 2u32));
    let erased = fill(mk(), 1u32)
        .and_then(|b| ks.scale_u32([N], b, slot!(Factor)))
        .erase()
        .and_then(|b| ks.scale_u32([N], b, 2u32))
        .erase();
    assert_static_parity(&plain, &erased);
    assert!(
        erased.check_ready().is_err(),
        "unbound slot must fail check_ready"
    );

    for factor in [3u32, 3, 4] {
        plain.mutate_bind(Factor(factor)).expect("bind plain");
        erased.mutate_bind(Factor(factor)).expect("bind erased");
        assert_eq!(
            homed_cb(&plain),
            homed_cb(&erased),
            "apply_binding_delta after bind"
        );
        let p = plain
            .sync(&ctx)
            .expect("plain")
            .map()
            .wait()
            .expect("read")
            .to_vec();
        let e = erased
            .sync(&ctx)
            .expect("erased")
            .map()
            .wait()
            .expect("read")
            .to_vec();
        assert_eq!(p, e, "factor {factor}");
        assert!(
            e.iter().all(|&v| v == 2 * factor),
            "factor {factor}: {:?}",
            &e[..4]
        );
        assert_eq!(homed_cb(&plain), homed_cb(&erased), "cb_cache after run");
        assert_static_parity(&plain, &erased);
    }
}

/// A chain whose SOURCE is addable but whose `next` holds a host seam (the CG
/// span shape): `contains_host_seam`, `cb_addable` false while
/// `cb_spine_head_addable` is true (the span opens at the fill and closes at the
/// seam), and `cb_restamp` (the kernel after the seam waits on the CB before it).
#[test]
fn parity_host_seam_chain() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ks = &ks;
    let mk = || DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let seam = |s: &mut [u32]| -> claspr::Result<()> {
        s.iter_mut().for_each(|v| *v += 1);
        Ok(())
    };

    // fill 1 → ×3 → ×5 → host +1 → ×2  =  (15 + 1) × 2 = 32.
    let plain = fill(mk(), 1u32)
        .and_then(|b| {
            ks.scale_u32([N], b, 3u32)
                .and_then(|b| ks.scale_u32([N], b, 5u32))
                .and_then_host(seam)
        })
        .and_then(|b| ks.scale_u32([N], b, 2u32));
    let erased = fill(mk(), 1u32)
        .and_then(|b| {
            ks.scale_u32([N], b, 3u32)
                .and_then(|b| ks.scale_u32([N], b, 5u32))
                .and_then_host(seam)
                .erase()
        })
        .erase()
        .and_then(|b| ks.scale_u32([N], b, 2u32))
        .erase();
    assert_static_parity(&plain, &erased);
    assert!(erased.contains_host_seam());
    assert!(
        !erased.cb_addable() && erased.cb_spine_head_addable(),
        "span head"
    );

    for _ in 0..2 {
        let p = plain
            .sync(&ctx)
            .expect("plain")
            .map()
            .wait()
            .expect("read")
            .to_vec();
        let e = erased
            .sync(&ctx)
            .expect("erased")
            .map()
            .wait()
            .expect("read")
            .to_vec();
        assert_eq!(p, e);
        assert!(e.iter().all(|&v| v == 32), "{:?}", &e[..4]);
        assert_static_parity(&plain, &erased);
    }
}

/// A multi-output join: per-element handles, `gather_checkouts` (one `Checkout`
/// per output) and `reclaim_undelivered` (the discarded `a`/`b` outputs of the
/// final kernel go home, so the twins replay identically).
#[test]
fn parity_multi_output_join() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let ks = &ks;
    let mk = || DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");

    let (pa, pb, pout) = (mk(), mk(), mk());
    let plain =
        bundle2(fill(pa, 3u32), fill(pb, 4u32)).and_then(move |(a, b)| ks.add_u32([N], a, b, pout));
    let (ea, eb, eout) = (mk(), mk(), mk());
    let erased = bundle2(fill(ea, 3u32), fill(eb, 4u32))
        .erase()
        .and_then(move |(a, b)| ks.add_u32([N], a, b, eout))
        .erase();
    assert_static_parity(&plain, &erased);

    for _ in 0..2 {
        let (_, _, po) = plain.sync(&ctx).expect("plain");
        let (_, _, eo) = erased.sync(&ctx).expect("erased");
        let (p, e) = (
            po.map().wait().expect("read").to_vec(),
            eo.map().wait().expect("read").to_vec(),
        );
        assert_eq!(p, e);
        assert!(e.iter().all(|&v| v == 7), "{:?}", &e[..4]);
        assert_static_parity(&plain, &erased);
    }
}
