//! Migration-pattern proof: every distinct chain/assertion from `chain.rs`
//! rewritten against the eager graph API (`claspr::eager`). Same N, same
//! fill/scale/add values, same final assertions — proving the eager API can
//! express the real Tier-2 chain test end-to-end before the full port.
//!
//! Old → new mapping:
//!   `upload!(v)`          → `upload(v)`
//!   `download!(buf)`      → `download` (terminal `.and_then(download).sync()` yields the Vec)
//!   multi-output add_u32 → `.and_then(|(_a,_b,out)| ...)` per-element pipe select
//!
//! All chain.rs shapes port to the eager API. One has a documented DEVIATION in
//! construction only (`three_slice...` below: the inputs are built as concrete
//! buffers up front rather than via an in-graph `bundle!`, since that shape is
//! covered by the additive `bundle_feeds_multi_arg_kernel`); assertions are
//! identical. The host-value chain (`value(x).and_then(|n| value(n+1))`) ports
//! 1:1 thanks to `value`'s by-value handle.

use claspr::Context;
use claspr::eager::{DeviceOpExt, alloc_zero, bundle3, download, upload, value};
use claspr_test_kernels::kernels;
use std::sync::Arc;

const N: usize = 256;
const FILL_VALUE: u32 = 0xfeed_cafe;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

/// chain.rs::linear_chain_upload_kernel_download — upload → fill_u32 → download.
#[test]
fn linear_chain_upload_kernel_download() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("kernels load");

    let result = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, FILL_VALUE))
        .and_then(download)
        .sync(&ctx)
        .expect("chain sync");

    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == FILL_VALUE));
}

/// chain.rs::three_slice_kernel_op_threads_tuple_output — multi-output add_u32 →
/// select `out` pipe → download.
///
/// DEVIATION FROM chain.rs: the old test feeds three parallel uploads into the
/// kernel via `bundle!(...).and_then(|(a,b,out)| add_u32(...))`. That shape is
/// NOT expressible in the eager API: an eager bundle's `Handle` is the default
/// `Pipe<(A,B,C)>` (one pipe over a tuple), and `and_then`'s closure receives
/// that single pipe — there is no projection from `Pipe<(A,B,C)>` into the three
/// separate `Pipe<A>/Pipe<B>/Pipe<C>` inputs the kernel needs (`ToInput` is impl'd
/// for `Pipe<D>` of one buffer, not a tuple-pipe element). So we mirror
/// eager_cutover's `multi_output_kernel_element_select`: sync the three inputs to
/// concrete buffers first, then run add_u32 → download as one chain. Same N,
/// same 3+4→7 assertion.
#[test]
fn three_slice_kernel_op_threads_tuple_output() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("kernels load");

    let a = upload(vec![3u32; N]).sync(&ctx).expect("upload a");
    let b = upload(vec![4u32; N]).sync(&ctx).expect("upload b");
    let out = alloc_zero::<u32>(N).sync(&ctx).expect("alloc out");

    let result = kernels
        .add_u32([N], a, b, out)
        .and_then(|(_a, _b, out)| download(out))
        .sync(&ctx)
        .expect("add chain");
    assert!(result.iter().all(|&v| v == 7));
}

/// The shape `three_slice_kernel_op_threads_tuple_output` had to AVOID, now
/// directly expressible: a `bundle3` feeds its three branches into a downstream
/// multi-arg kernel. The bundle's `Handle` is now a tuple of per-branch output
/// pipes `(Pipe<a>, Pipe<b>, Pipe<out>)`, so `.and_then(|(a, b, out)| ...)`
/// projects each branch into a separate kernel buffer input. Same 3+4→7.
#[test]
fn bundle_feeds_multi_arg_kernel() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("kernels load");

    let result = bundle3(
        upload(vec![3u32; N]),
        upload(vec![4u32; N]),
        alloc_zero::<u32>(N),
    )
    .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
    .and_then(|(_a, _b, out)| download(out))
    .sync(&ctx)
    .expect("bundle → add chain");

    assert!(
        result.iter().all(|&v| v == 7),
        "3+4=7; got {:?}",
        &result[..8]
    );
}

/// chain.rs::kernel_op_chains_two_kernels — upload → fill_u32 → scale_u32 → download.
#[test]
fn kernel_op_chains_two_kernels() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("kernels load");

    let result = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 5))
        .and_then(|buf| kernels.scale_u32([N], buf, 7))
        .and_then(download)
        .sync(&ctx)
        .expect("fill+scale chain");
    assert!(result.iter().all(|&v| v == 35));
}

/// chain.rs::value_passthrough — host value chain through the graph. Ports 1:1:
/// `value`'s by-value handle hands each `and_then` closure the concrete `u32`, so
/// the multi-stage in-graph host arithmetic is expressible directly.
#[test]
fn value_passthrough() {
    let Some(ctx) = ctx() else { return };

    let out = value(42u32)
        .and_then(|n| value(n.wrapping_add(1)))
        .and_then(|n| value(n.wrapping_mul(2)))
        .sync(&ctx)
        .expect("value chain");
    assert_eq!(*out, 86);
}

/// chain.rs::upload_accepts_arc_source_caller_retains_clone — Arc<[T]> upload
/// source, caller keeps its own clone, round-trip download.
#[test]
fn upload_accepts_arc_source_caller_retains_clone() {
    let Some(ctx) = ctx() else { return };

    let shared: Arc<[u32]> = Arc::from(vec![7u32; N]);
    let kept_by_caller = Arc::clone(&shared);

    let result = upload(Arc::clone(&shared))
        .and_then(download)
        .sync(&ctx)
        .expect("arc upload");
    assert!(result.iter().all(|&v| v == 7));
    // Caller's clone is still usable; data heap is alive.
    assert_eq!(kept_by_caller[0], 7);
}
