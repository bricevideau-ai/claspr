//! Diamond fan-out + fan-in where the shared input is held by a
//! single `cl_mem`, not duplicated per branch.
//!
//! `upload(vec).arc()` produces `Arc<DeviceSlice<T>>`; each branch
//! gets an `Arc::clone` of the same handle, and the typed launcher
//! accepts it via the new `KernelSliceArg<T>` impl for
//! `Arc<DeviceSlice<T>>`. One device allocation, one upload — N
//! branches read from the same `cl_mem` via OpenCL's refcounted
//! buffer semantics.
//!
//! Original spike scenario 3 modelled this with a fake `.arc()` on
//! its mock DeviceSlice; the pre-rebase translation worked around
//! the missing primitive by uploading the host `Arc<[T]>` twice
//! (one fresh `cl_mem` per branch). This test pins the real
//! one-cl_mem-shared shape.

use claspr::Context;
use claspr_async::{DeviceOperation, bundle, device_slice_alloc, download, upload, value};
use claspr_test_kernels::kernels;
use std::sync::Arc;

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

#[test]
fn diamond_shares_single_cl_mem_via_arc_device_slice() {
    // Upload [5; N] ONCE; share its DeviceSlice via Arc across two
    // branches; each branch runs add_u32 reading from the shared
    // buffer plus its own [0; N] output; combine via add_u32 again.
    //
    // Per-branch: out = shared + per_branch_input.
    //   branch A: per_branch_input = [10; N] → out = [15; N]
    //   branch B: per_branch_input = [20; N] → out = [25; N]
    // Combined: a + b = [40; N].
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let result: Vec<u32> = upload(vec![5u32; N])
        .arc()
        .and_then(move |shared: Arc<claspr::DeviceSlice<u32>>| {
            let s1 = Arc::clone(&shared);
            let s2 = Arc::clone(&shared);
            // add_u32(a, b, out): reads a and b, writes out. So the
            // per-branch inputs ([10; N] / [20; N]) need real
            // uploads; the output buffer is just a fresh alloc.
            bundle!(
                // Branch A.
                bundle!(upload(vec![10u32; N]), device_slice_alloc::<u32>(N)).and_then(
                    move |(a_in, out)| {
                        kernels_ref
                            .add_u32([N], s1, a_in, out)
                            .and_then(|(_s1, _a_in, out)| value(out))
                    }
                ),
                // Branch B.
                bundle!(upload(vec![20u32; N]), device_slice_alloc::<u32>(N)).and_then(
                    move |(b_in, out)| {
                        kernels_ref
                            .add_u32([N], s2, b_in, out)
                            .and_then(|(_s2, _b_in, out)| value(out))
                    }
                ),
            )
            .and_then(move |(a_out, b_out)| {
                // Final combine: write a_out + b_out into a fresh
                // alloc. add_u32 takes (a, b, out).
                bundle!(value(a_out), value(b_out), device_slice_alloc::<u32>(N)).and_then(
                    move |(a, b, out)| {
                        kernels_ref
                            .add_u32([N], a, b, out)
                            .and_then(|(_a, _b, out)| value(out))
                    },
                )
            })
            .and_then(download)
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

#[test]
fn arc_device_slice_refcount_holds_until_last_branch_finishes() {
    // Stress the Arc<DeviceSlice<T>> lifecycle: build the chain,
    // sync, assert error_count is 0. If the cl_mem were released
    // prematurely (e.g. before the last branch's kernel finished),
    // we'd either get an OpenCL error from a use-after-free or the
    // sticky-error counter would catch it.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    // 4-way fan: same shared buffer feeds 4 branches. `b` needs
    // real data (kernel reads it); `out` just needs an alloc.
    let result: Vec<u32> = upload(vec![7u32; N])
        .arc()
        .and_then(move |shared: Arc<claspr::DeviceSlice<u32>>| {
            let s1 = Arc::clone(&shared);
            let s2 = Arc::clone(&shared);
            let s3 = Arc::clone(&shared);
            let s4 = Arc::clone(&shared);
            bundle!(
                bundle!(upload(vec![0u32; N]), device_slice_alloc::<u32>(N)).and_then(
                    move |(b, out)| kernels_ref
                        .add_u32([N], s1, b, out)
                        .and_then(|(_, _, out)| value(out))
                ),
                bundle!(upload(vec![0u32; N]), device_slice_alloc::<u32>(N)).and_then(
                    move |(b, out)| kernels_ref
                        .add_u32([N], s2, b, out)
                        .and_then(|(_, _, out)| value(out))
                ),
                bundle!(upload(vec![0u32; N]), device_slice_alloc::<u32>(N)).and_then(
                    move |(b, out)| kernels_ref
                        .add_u32([N], s3, b, out)
                        .and_then(|(_, _, out)| value(out))
                ),
                bundle!(upload(vec![0u32; N]), device_slice_alloc::<u32>(N)).and_then(
                    move |(b, out)| kernels_ref
                        .add_u32([N], s4, b, out)
                        .and_then(|(_, _, out)| value(out))
                ),
            )
            .and_then(|(a, _b, _c, _d)| download(a))
        })
        .sync(&ctx)
        .expect("4-way fan chain");

    assert!(result.iter().all(|&v| v == 7));
    assert_eq!(ctx.error_count(), 0);
}
