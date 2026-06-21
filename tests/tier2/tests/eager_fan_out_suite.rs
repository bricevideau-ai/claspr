//! Eager-API port of `fan_out.rs`: `fan_out(inputs, |input| op)` — N-ary
//! homogeneous (data-parallel) composition.
//!
//! Old → new mapping:
//!   `fan_out(v, op)`    → `fan_out(v, op)` (same free-fn signature)
//!   `upload!(v)`        → `upload::<T, claspr::ReadWrite, _>(v)`
//!   `download!(buf)`    → `download`
//!   `.and_then_host(f)` → `.and_then_host(f)` (host seam; `u32` is Mappable,
//!                          so the View is the scalar by value — same closure)
//!
//! Same N, same values, same assertions as `fan_out.rs`.

use claspr::Context;
use claspr::eager::{EagerOpExt, download, fan_out, upload, value};
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
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;
    let fill_values: Vec<u32> = vec![100, 200, 300, 400];
    let outputs: Vec<Vec<u32>> = fan_out(fill_values.clone(), move |val| {
        upload::<u32, claspr::ReadWrite, _>(vec![0u32; N])
            .and_then(move |buf| kernels_ref.fill_u32([N], buf, val))
            .and_then(download)
    })
    .sync(&ctx)
    .expect("fan_out chain");
    for (out, expected) in outputs.iter().zip(fill_values.iter()) {
        assert!(out.iter().all(|&v| v == *expected));
    }
}

// BLOCKED: vec_method_form_matches_free_fn — needs a `FanOutExt` method form
// (`Vec::fan_out(op)`) on the eager API. `claspr::eager` exposes only the free
// `fan_out(vec, op)` fn; there is no `Vec::fan_out` method to compare against,
// so the "method form == free fn" equivalence this test asserts is not
// expressible. Reproducing both halves via the free fn would make the
// equivalence trivially true and test nothing — left blocked rather than
// fake-passed.

#[test]
fn fan_out_propagates_child_error() {
    // Inject a failing child via and_then_host returning Err. `u32` is
    // Mappable in the eager host seam (View is the scalar by value), so the
    // closure receives `n: u32` exactly as in the closure-layer original.
    use claspr::Error;
    let Some(ctx) = ctx() else { return };
    let err = fan_out(vec![1u32, 2, 3], |n| {
        value(n).and_then_host(|_n: u32| {
            // Closure error surfaces at the terminal as the original variant.
            Err(Error::Build {
                log: "injected failure".to_string(),
            })
        })
    })
    .sync(&ctx)
    .expect_err("fan_out should surface child error");
    assert!(
        matches!(&err, Error::Build { log } if log == "injected failure"),
        "got {err:?}",
    );
}
