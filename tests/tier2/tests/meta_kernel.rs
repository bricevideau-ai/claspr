//! `#[meta_kernel]` — author a generic reusable-graph function without the
//! subgraph-signature plumbing (generics + `Fn` bounds + `DeviceOp<Output/Handle/
//! Checkouts>` + `FromCheckout`). Each `subgraph!(Fn(inputs..) -> Output)` parameter
//! is rewritten to a fresh closure generic bounded `Fn(inputs..) -> __Out` with
//! `__Out: Subgraph<Output>`. This guards the macro end-to-end on the simplest shape
//! (one subgraph, single-buffer output); `examples/cg` covers the multi-subgraph
//! tuple-output case.

use claspr::eager::{DeviceOpExt, download};
use claspr::{Checkout, Context, DeviceSlice};
use claspr_test_kernels::kernels;

const N: usize = 64;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            None
        }
    }
}

fn seeded(ctx: &Context, v: u32) -> DeviceSlice<u32> {
    DeviceSlice::<u32>::alloc_zero(ctx, N)
        .expect("alloc")
        .fill(v)
        .wait()
        .expect("seed")
}

/// A meta-kernel: apply a caller-supplied one-buffer subgraph, then download. No
/// hand-written generics or `where`-clause — the `subgraph!` marker declares the
/// closure's inputs and output, and the macro generates the rest. The `.and_then(
/// download)` compose only typechecks because the generated `Subgraph<DeviceSlice<u32>>`
/// bound pins the closure result's `Handle` to the canonical `Pipe<DeviceSlice<u32>>`.
#[claspr::meta_kernel]
fn run_one(
    ctx: &Context,
    input: DeviceSlice<u32>,
    build: subgraph!(Fn(&kernels::Kernels, DeviceSlice<u32>) -> DeviceSlice<u32>),
) -> claspr::Result<Vec<u32>> {
    let ks = kernels::kernels(ctx)?;
    build(&ks, input)
        .and_then(download)
        .sync(ctx)
        .map(Checkout::into_inner)
}

#[test]
fn meta_kernel_runs_a_one_buffer_subgraph() {
    let Some(ctx) = ctx() else { return };

    // The subgraph closure builds `scale(*3)`; its input/output types are inferred from
    // the `subgraph!` marker's generated `Fn` bound (no annotations at the call site).
    let out = run_one(&ctx, seeded(&ctx, 4), |ks, b| ks.scale_u32([N], b, 3u32))
        .expect("run meta-kernel");
    assert!(
        out.iter().all(|&v| v == 12),
        "4 * 3 = 12, got {:?}",
        &out[..8]
    );

    // A DIFFERENT subgraph through the SAME meta-kernel — proving it is generic over
    // the subgraph, not specialized to one.
    let out = run_one(&ctx, seeded(&ctx, 5), |ks, b| ks.scale_u32([N], b, 2u32))
        .expect("run meta-kernel 2");
    assert!(
        out.iter().all(|&v| v == 10),
        "5 * 2 = 10, got {:?}",
        &out[..8]
    );
}
