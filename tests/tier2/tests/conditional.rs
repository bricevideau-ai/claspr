//! Spike scenario 9 — conditional graph via `DynOp` type erasure.
//!
//! `if`/`match` arms produce different concrete op types — without
//! erasure, Rust would reject the expression with a type-mismatch
//! error. `DynOp::new` boxes each arm so they share `Output` and the
//! enclosing chain can compose normally.

use claspr::Context;
use claspr_async::{DeviceOperation, DeviceOperationHostExt, DynOp, download, upload, value};
use claspr_test_kernels::kernels;
use std::sync::{Arc, Mutex};

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
    // and_then_host's closure returns Result<()> by design (async
    // submit-vs-completion gap — see module docs); the reduction
    // value flows out via Arc<Mutex<_>> as usual.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let sum_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&sum_cell);
    let _final_buf = upload(vec![3u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 9))
        .and_then_host(move |slice: &mut [u32]| {
            *cell.lock().unwrap() = slice.iter().sum();
            Ok(())
        })
        .sync(&ctx)
        .expect("baseline");
    assert_eq!(*sum_cell.lock().unwrap(), 9 * N as u32);
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
    // Single DynOp wrapping a chain that touches a kernel. The
    // chain's Output is the buffer; the sum is captured via
    // Arc<Mutex<_>> (the canonical side-effect channel for
    // host-closure-computed values).
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let sum_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&sum_cell);

    let chain: DynOp<claspr::DeviceSlice<u32>> = DynOp::new(
        upload(vec![3u32; N])
            .and_then(|buf| kernels.fill_u32([N], buf, 9))
            .and_then_host(move |slice: &mut [u32]| {
                *cell.lock().unwrap() = slice.iter().sum();
                Ok(())
            }),
    );
    let _buf = chain.sync(&ctx).expect("a");
    assert_eq!(*sum_cell.lock().unwrap(), 9 * N as u32);
}

#[test]
fn dyn_op_picks_branch_with_or_without_kernel() {
    // The actual conditional shape: two arms produce different
    // concrete chain types — one runs the kernel and captures sum
    // via cell, the other lifts a literal 0 — and DynOp erases them.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;
    let sum_cell = Arc::new(Mutex::new(0u32));

    let make = |use_kernel: bool| -> DynOp<u32> {
        if use_kernel {
            let cell = Arc::clone(&sum_cell);
            DynOp::new(
                upload(vec![3u32; N])
                    .and_then(move |buf| kernels_ref.fill_u32([N], buf, 9))
                    .and_then_host(move |slice: &mut [u32]| {
                        *cell.lock().unwrap() = slice.iter().sum();
                        Ok(())
                    })
                    .and_then(|_buf| value(0u32)),
            )
        } else {
            DynOp::new(value(0u32))
        }
    };
    assert_eq!(make(true).sync(&ctx).expect("a"), 0);
    assert_eq!(*sum_cell.lock().unwrap(), 9 * N as u32);
    *sum_cell.lock().unwrap() = 0;
    assert_eq!(make(false).sync(&ctx).expect("b"), 0);
    assert_eq!(*sum_cell.lock().unwrap(), 0);
}

#[test]
fn non_taken_branch_closure_does_not_fire() {
    // Pin the laziness guarantee: an `if`/`else` that builds one of
    // two `DynOp<T>`s never constructs the not-taken arm. That arm
    // here contains a closure that would panic; if branch selection
    // were eager (e.g. some future refactor pre-built both arms), the
    // test would panic. Picks the safe arm; assertion is "sync
    // returns the safe arm's value, no panic." Belt-and-suspenders
    // for the basic Rust short-circuit on top of DynOp's wrapper.
    let Some(ctx) = ctx() else { return };
    let make = |take_safe: bool| -> DynOp<u32> {
        if take_safe {
            DynOp::new(value(7u32))
        } else {
            DynOp::new(
                value(())
                    .and_then_host(|()| -> claspr::Result<()> {
                        panic!("non-taken branch fired — DynOp construction must be lazy");
                    })
                    .and_then(|()| value(0u32)),
            )
        }
    };
    assert_eq!(make(true).sync(&ctx).expect("safe arm"), 7);
}
