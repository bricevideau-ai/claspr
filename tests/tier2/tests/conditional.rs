//! Spike scenario 9 — conditional graph via `DynOp` type erasure.
//!
//! `if`/`match` arms produce different concrete op types — without
//! erasure, Rust would reject the expression with a type-mismatch
//! error. `DynOp::new` boxes each arm so they share `Output` and the
//! enclosing chain can compose normally.

use claspr::Context;
use claspr_async::{DeviceOperation, DeviceOperationHostExt, DynOp, download, upload, value};
use claspr_test_kernels::kernels;

const N: usize = 32;

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
fn dyn_op_lets_if_arms_have_different_concrete_types() {
    let Some(ctx) = ctx() else { return };
    // Different op shapes per branch: one is a bare `value`, the other
    // is a multi-stage chain. They share Output = u32.
    let cond = true;
    let chain: DynOp<u32> = if cond {
        DynOp::new(value(7u32))
    } else {
        DynOp::new(value(0u32).and_then(|n| value(n.wrapping_add(100))))
    };
    let result = chain.sync(&ctx).expect("dyn_op");
    assert_eq!(result, 7);
}

#[test]
fn dyn_op_wraps_simple_value() {
    let Some(ctx) = ctx() else { return };
    let chain: DynOp<u32> = DynOp::new(value(42u32));
    let v = chain.sync(&ctx).expect("sync");
    assert_eq!(v, 42);
}

#[test]
fn dyn_op_wraps_value_chain() {
    let Some(ctx) = ctx() else { return };
    let chain: DynOp<u32> = DynOp::new(value(1u32).and_then(|n| value(n + 41)));
    let v = chain.sync(&ctx).expect("sync");
    assert_eq!(v, 42);
}

#[test]
fn dyn_op_wraps_upload_download() {
    // No kernel, just data transfer.
    let Some(ctx) = ctx() else { return };
    let chain: DynOp<Vec<u32>> = DynOp::new(upload(vec![7u32; N]).and_then(download));
    let v = chain.sync(&ctx).expect("sync");
    assert!(v.iter().all(|&x| x == 7));
}

#[test]
fn baseline_kernel_chain_without_dynop() {
    // Sanity baseline: the exact same chain shape outside DynOp.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let a = upload(vec![3u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 9))
        .and_then(download)
        .and_then_host(|v| Ok(v.iter().sum::<u32>()))
        .sync(&ctx)
        .expect("baseline");
    assert_eq!(a, 9 * N as u32);
}

#[test]
fn dyn_op_wraps_bare_kernel_op() {
    // Smallest DynOp+kernel chain: no upload/download, just construct
    // a buffer manually and wrap the kernel call. Tests whether
    // wrapping the per-kernel Op alone triggers the crash.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    use claspr::DeviceSlice;
    let buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");
    let chain: DynOp<DeviceSlice<u32>> = DynOp::new(kernels.fill_u32([N], buf, 5));
    let buf = chain.sync(&ctx).expect("sync");
    let mut out = vec![0u32; N];
    buf.read(&ctx, &mut out).wait().expect("read");
    assert!(out.iter().all(|&v| v == 5));
}

#[test]
fn dyn_op_minimal_kernel_chain() {
    // Single DynOp wrapping a chain that touches a kernel.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let chain: DynOp<u32> = DynOp::new(
        upload(vec![3u32; N])
            .and_then(|buf| kernels.fill_u32([N], buf, 9))
            .and_then(download)
            .and_then_host(|v| Ok(v.iter().sum::<u32>())),
    );
    let a = chain.sync(&ctx).expect("a");
    assert_eq!(a, 9 * N as u32);
}

#[test]
fn dyn_op_picks_branch_with_or_without_kernel() {
    // The actual conditional shape: two arms produce different
    // concrete chain types — one runs the kernel, the other lifts a
    // literal — and DynOp erases them so a single `let chain = ...`
    // can hold either.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let make = |use_kernel: bool| -> DynOp<u32> {
        if use_kernel {
            DynOp::new(
                upload(vec![3u32; N])
                    .and_then(move |buf| kernels_ref.fill_u32([N], buf, 9))
                    .and_then(download)
                    .and_then_host(|v| Ok(v.iter().sum::<u32>())),
            )
        } else {
            DynOp::new(value(0u32))
        }
    };
    assert_eq!(make(true).sync(&ctx).expect("a"), 9 * N as u32);
    assert_eq!(make(false).sync(&ctx).expect("b"), 0);
}
