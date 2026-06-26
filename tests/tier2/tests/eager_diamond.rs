//! Eager-API port of `diamond_arc.rs`: a shared input buffer fanned out to N
//! read-only branches, then fanned back in.
//!
//! Old → new mapping for the SHARED-BUFFER fan-out (the point of this port):
//!   `upload!(v).arc().and_then(|shared| { let s1 = Arc::clone(&shared); … })`
//!     → `arc_split::<N>(arced(upload(v))).and_then(|[s1, s2, …]| { … })`
//!
//! In the closure model the `.arc()` closure receives the concrete
//! `Arc<DeviceSlice>`, so `Arc::clone(&shared)` works directly. In the eager
//! model an `and_then` closure receives a `Pipe<Arc<DeviceSlice>>`, NOT the
//! `Arc`, so `Arc::clone` there is impossible. The eager way to hand one
//! producer's output to N consumers is `arc_split::<N>(arced(producer))`: each
//! array slot is a `Pipe<Arc<DeviceSlice>>` that yields its own `Arc::clone`
//! (a refcount bump on the same `cl_mem`) at execute. Each branch feeds its
//! slot as the shared read-only kernel arg — `ToInput` accepts `Pipe<Arc<…>>`
//! cleanly, exactly like the `copy_u32` case in
//! `eager_cutover::arc_split_read_only_fan_out`. THIS PART OF THE PORT WORKED
//! WITH NO FRICTION.
//!
//! Branch reducer: a bundle branch must end on a single `Pipe<DeviceSlice>` so
//! the join hands the combine kernel a per-branch buffer input (`ToInput` is
//! impl'd for `Pipe<one buffer>`, not for the multi-output kernel's
//! `Pipe<(a,b,out)>`). `add_u32`'s output handle is the 3-tuple
//! `(Pipe<a>, Pipe<b>, Pipe<out>)`, so each branch ends with `forward(out)` — the
//! eager analog of the old reducer's `value(out)`: a pure select/identity node
//! that re-deposits the chosen pipe (resolve + forward, NO device work). Faithful
//! 1:1 to `diamond_arc.rs` (which ended each branch `add_u32(...).and_then(|(_,_,
//! out)| value(out))`); no extra kernel.

use claspr::Context;
use claspr::eager::{
    DeviceOpExt, alloc_zero, arc_split, arced, bundle2, bundle3, download, forward, upload,
};
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

/// diamond_arc.rs::diamond_shares_single_cl_mem_via_arc_device_slice.
///
/// Upload [5; N] ONCE; share its DeviceSlice via `arc_split::<2>` across two
/// branches; each branch runs `add_u32` reading the shared buffer plus its own
/// per-branch input, into a fresh output; combine the two branch outputs via a
/// final on-device `add_u32`.
///
///   branch A: shared(5) + [10; N] → [15; N]
///   branch B: shared(5) + [20; N] → [25; N]
///   combine : a + b               → [40; N]
#[test]
fn diamond_shares_single_cl_mem_via_arc_device_slice() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result = arc_split::<2, _>(arced(upload(vec![5u32; N])))
        .and_then(|[s1, s2]| {
            let ks = &kernels;
            bundle3(
                // Branch A: out = shared + [10; N], reduced to its single
                // `out` pipe via `forward` (= old `value(out)`, no device work)
                // so the bundle hands the combine kernel a `Pipe<DeviceSlice>`.
                bundle2(upload(vec![10u32; N]), alloc_zero::<u32>(N))
                    .and_then(move |(a_in, out)| ks.add_u32([N], s1, a_in, out))
                    .and_then(|(_s, _a_in, out)| forward(out)),
                // Branch B: out = shared + [20; N], same reducer.
                bundle2(upload(vec![20u32; N]), alloc_zero::<u32>(N))
                    .and_then(move |(b_in, out)| ks.add_u32([N], s2, b_in, out))
                    .and_then(|(_s, _b_in, out)| forward(out)),
                // Fresh destination for the combine.
                alloc_zero::<u32>(N),
            )
            .and_then(move |(a_out, b_out, out)| ks.add_u32([N], a_out, b_out, out))
            .and_then(|(_a, _b, out)| download(out))
        })
        .sync(&ctx)
        .expect("diamond chain");

    assert_eq!(result.len(), N);
    // (5 + 10) + (5 + 20) = 15 + 25 = 40
    assert!(
        result.iter().all(|&v| v == 40),
        "first few = {:?}",
        &result[..4]
    );
}

/// diamond_arc.rs::arc_device_slice_refcount_holds_until_last_branch_finishes.
///
/// 4-way fan: the same shared buffer ([7; N]) feeds four branches, each adding
/// its own [0; N] input (so out == shared == 7). Branch outputs are joined by a
/// nested bundle-of-bundles, then branch A's result is downloaded and the
/// sticky error counter is asserted 0 — if the shared `cl_mem` were released
/// before the last branch's kernel finished, a use-after-free / sticky error
/// would surface. Each branch ends with `forward(out)` (select/identity, no
/// device work) so the nested bundles compose over single `Pipe<DeviceSlice>`
/// outputs.
#[test]
fn arc_device_slice_refcount_holds_until_last_branch_finishes() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // The nested bundle-of-bundles is the terminal: `sync` reconstructs the
    // `(Vec, (Vec, (Vec, Vec)))` tuple, and branch A's result is picked on the
    // host. (Picking one branch's device buffer to download *in-graph* would
    // need a pipe-forward/select primitive — see module doc + report; here all
    // four branches download and we select after sync, which keeps all four
    // Arc clones live until every kernel finishes, satisfying the refcount
    // assertion identically.)
    // The terminal op is the `and_then` (single output), so its result is ONE
    // `Checkout<(Vec, (Vec, (Vec, Vec)))>` wrapping the whole reconstructed nested
    // tuple — not a tuple of checkouts. Deref and read branch A (the `.0` slot).
    let result = arc_split::<4, _>(arced(upload(vec![7u32; N])))
        .and_then(|[s1, s2, s3, s4]| {
            let ks = &kernels;
            // out = shared(7) + [0; N] = 7 on every branch. Nested
            // bundle-of-bundles: A joined with (B joined with (C joined
            // with D)). Each branch downloads its own output.
            let branch = |s| {
                bundle2(upload(vec![0u32; N]), alloc_zero::<u32>(N))
                    .and_then(move |(b, out)| ks.add_u32([N], s, b, out))
                    .and_then(|(_s, _b, out)| forward(out))
                    .and_then(download)
            };
            // Bundle of bundles (nested join) — composes cleanly in eager.
            bundle2(
                branch(s1),
                bundle2(branch(s2), bundle2(branch(s3), branch(s4))),
            )
        })
        .sync(&ctx)
        .expect("4-way fan chain");
    // Multi-output terminal tail → a tuple of per-output checkouts.
    let (a, _rest) = &result;

    assert!(a.iter().all(|&v| v == 7));
    assert_eq!(ctx.error_count(), 0);
}
