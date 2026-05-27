//! Sub-step-2 coverage: `bundle!` (arities 2/3) and `fan_out`.
//!
//! Validates that heterogeneous-parallel and homogeneous-parallel
//! compositions both run end-to-end through the chain. We don't try
//! to measure actual on-device overlap here — pocl + rusticl handle
//! OOO scheduling per their own policies and a microbenchmark would
//! be flaky. The goal is correctness: each child runs once, outputs
//! arrive in declaration order, the chain finishes.

use claspr::Context;
use claspr_async::{DeviceOperation, bundle, download, fan_out, upload, value};
use claspr_test_kernels::kernels;

const N: usize = 128;

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
fn bundle2_pure_values() {
    let Some(ctx) = ctx() else {
        return;
    };
    let (a, b) = bundle!(value(11u32), value(22u32))
        .sync(&ctx)
        .expect("bundle2");
    assert_eq!((a, b), (11, 22));
}

#[test]
fn bundle3_pure_values() {
    let Some(ctx) = ctx() else {
        return;
    };
    let (a, b, c) = bundle!(value("one"), value(2u32), value(3.0f32))
        .sync(&ctx)
        .expect("bundle3");
    assert_eq!(a, "one");
    assert_eq!(b, 2);
    assert_eq!(c, 3.0);
}

#[test]
fn bundle2_two_kernels_on_distinct_buffers() {
    // Run fill_u32 on two independent buffers via Bundle2; each writes
    // its own value; both downloaded back. No data dependency between
    // the two branches.
    let Some(ctx) = ctx() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let left = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 0xAA))
        .and_then(download);
    let right = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 0xBB))
        .and_then(download);

    let (l, r) = bundle!(left, right).sync(&ctx).expect("bundle2 kernels");
    assert!(l.iter().all(|&v| v == 0xAA), "left branch");
    assert!(r.iter().all(|&v| v == 0xBB), "right branch");
}

#[test]
fn fan_out_homogeneous_kernels() {
    // Four independent fill_u32 calls, each writing a distinct value.
    let Some(ctx) = ctx() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let values: Vec<u32> = (0..4).collect();
    let outs: Vec<Vec<u32>> = fan_out(values.clone(), move |v| {
        upload(vec![0u32; N])
            .and_then(move |buf| kernels_ref.fill_u32([N], buf, v))
            .and_then(download)
    })
    .sync(&ctx)
    .expect("fan_out");

    assert_eq!(outs.len(), values.len());
    for (i, out) in outs.iter().enumerate() {
        assert!(
            out.iter().all(|&v| v == i as u32),
            "branch {i}: expected all {i}",
        );
    }
}
