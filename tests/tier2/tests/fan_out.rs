//! `fan_out(inputs, |input| op_for(input))` — N-ary parallel
//! composition where every child has the same Op type.
//!
//! Where `bundle!` is heterogeneous (different child Op types), fan_out
//! is the data-parallel shape: tile up a problem, apply the same
//! transform to each tile, collect results in order.

use claspr::Context;
use claspr_async::{DeviceOperation, FanOutExt, download, fan_out, upload, value};
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

#[test]
fn fan_out_preserves_input_order() {
    let Some(ctx) = ctx() else { return };
    let inputs: Vec<u32> = (0..8).collect();
    let outputs: Vec<u32> = fan_out(inputs.clone(), |n| value(n.wrapping_mul(10)))
        .sync(&ctx)
        .expect("fan_out");
    assert_eq!(outputs, vec![0, 10, 20, 30, 40, 50, 60, 70]);
}

#[test]
fn fan_out_over_empty_yields_empty_vec() {
    let Some(ctx) = ctx() else { return };
    let outputs: Vec<u32> = fan_out(Vec::<u32>::new(), value)
        .sync(&ctx)
        .expect("fan_out");
    assert!(outputs.is_empty());
}

#[test]
fn fan_out_of_kernel_ops_runs_each_branch() {
    // 4 independent upload→fill→download pipelines via fan_out.
    // Result vec is [Vec<u32>; 4]; each tile holds its expected fill.
    //
    // `&kernels` is captured by the FnMut fan_out closure (Copy reference),
    // then re-captured by the inner per-branch `and_then` closure. This
    // is the canonical pattern when chains need a shared Kernels.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;
    let fill_values: Vec<u32> = vec![100, 200, 300, 400];
    let outputs: Vec<Vec<u32>> = fan_out(fill_values.clone(), move |val| {
        upload(vec![0u32; N])
            .and_then(move |buf| kernels_ref.fill_u32([N], buf, val))
            .and_then(download)
    })
    .sync(&ctx)
    .expect("fan_out chain");
    for (out, expected) in outputs.iter().zip(fill_values.iter()) {
        assert!(out.iter().all(|&v| v == *expected));
    }
}

#[test]
fn vec_method_form_matches_free_fn() {
    // The `Vec::fan_out(op)` method form should produce the same chain
    // as `fan_out(vec, op)`. Run both, assert outputs match.
    let Some(ctx) = ctx() else { return };
    let inputs: Vec<u32> = (0..6).collect();
    let via_free: Vec<u32> = fan_out(inputs.clone(), |n| value(n.wrapping_add(1)))
        .sync(&ctx)
        .expect("fan_out free");
    let via_method: Vec<u32> = inputs
        .clone()
        .fan_out(|n| value(n.wrapping_add(1)))
        .sync(&ctx)
        .expect("fan_out method");
    assert_eq!(via_free, via_method);
    assert_eq!(via_method, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn fan_out_propagates_child_error() {
    // Inject a failing child via and_then_host returning Err.
    use claspr::Error;
    use claspr_async::DeviceOperationHostExt;
    let Some(ctx) = ctx() else { return };
    let err = fan_out(vec![1u32, 2, 3], |n| {
        value(n).and_then_host(|_n: u32| {
            // Closure error surfaces as Error::OpenCl(ClError(neg))
            // through the user-event signal — the specific variant
            // is lost in the async boundary in v1.
            Err::<(), _>(Error::Build {
                log: "injected failure".to_string(),
            })
        })
    })
    .sync(&ctx)
    .expect_err("fan_out should surface child error");
    assert!(matches!(err, Error::OpenCl(_)), "got {err:?}");
}
