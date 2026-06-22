//! Eager-API port of `conditional.rs`: conditional graphs via `DeviceDynOp` type
//! erasure.
//!
//! `conditional.rs` is built entirely around `DynOp` — a boxed, type-erased op so
//! that `if`/`match` arms producing DIFFERENT concrete op types can share one
//! `Output` and compose. The eager analog is `DeviceDynOp<'op, T>`
//! (`claspr::eager::DeviceDynOp`): an object-safe `ErasedDeviceOp` shim boxed into a
//! single-output `DeviceOp`. All seven `DynOp` tests now port; only the
//! host-scalar `value(n).and_then(|n| value(n+1))` shape deviates (eager `and_then`
//! hands a `Pipe<T>`, not the host value — same deviation as
//! `eager_chain.rs::value_passthrough`; the value is computed up front).
//!
//!   `DynOp::new(op)`    → `DeviceDynOp::new(op)`
//!   `upload!(v)`        → `upload::<u32, claspr::ReadWrite, _>(v)`
//!   `download!(buf)`    → `download`
//!   `.and_then_host(f)` → `.and_then_host(f)` (DeviceSlice View is `&mut [u32]`)

use claspr::Context;
use claspr::eager::{DeviceDynOp, DeviceOpExt, download, upload, value};
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

/// conditional.rs::dyn_op_lets_if_arms_have_different_concrete_types — the core
/// reason `DeviceDynOp` exists. Two arms produce different concrete op types
/// (a bare `value` vs a multi-stage chain), unified on `Output = u32` by erasure.
#[test]
fn dyn_op_lets_if_arms_have_different_concrete_types() {
    let Some(ctx) = ctx() else { return };
    let cond = true;
    let chain: DeviceDynOp<u32> = if cond {
        // Arm 1: a bare `value`. Concrete type `Value<u32>`.
        DeviceDynOp::new(value(7u32))
    } else {
        // Arm 2: a value-chain of a DIFFERENT concrete type, both `Output = u32`.
        // `value`'s handle is by-value, so the host-scalar transform the original
        // used (`value(0).and_then(|n| value(n + 100))`) ports faithfully — `n` is
        // the `u32`, computed at build.
        DeviceDynOp::new(value(0u32).and_then(|n| value(n + 100)))
    };
    let result = chain.sync(&ctx).expect("dyn_op");
    assert_eq!(result, 7);
}

/// conditional.rs::dyn_op_wraps_simple_value — erase a bare `value`.
#[test]
fn dyn_op_wraps_simple_value() {
    let Some(ctx) = ctx() else { return };
    let chain: DeviceDynOp<u32> = DeviceDynOp::new(value(42u32));
    let v = chain.sync(&ctx).expect("sync");
    assert_eq!(v, 42);
}

/// conditional.rs::dyn_op_wraps_value_chain — erase a `value`-chain.
///
/// Ports faithfully: `value`'s by-value handle lets the host-scalar transform
/// `value(1).and_then(|n| value(n + 41))` run in-graph (`n` is the `u32`).
#[test]
fn dyn_op_wraps_value_chain() {
    let Some(ctx) = ctx() else { return };
    let chain: DeviceDynOp<u32> = DeviceDynOp::new(value(1u32).and_then(|n| value(n + 41)));
    let v = chain.sync(&ctx).expect("sync");
    assert_eq!(v, 42);
}

/// conditional.rs::dyn_op_wraps_upload_download — erase a transfer chain.
#[test]
fn dyn_op_wraps_upload_download() {
    let Some(ctx) = ctx() else { return };
    let chain: DeviceDynOp<Vec<u32>> =
        DeviceDynOp::new(upload::<u32, claspr::ReadWrite, _>(vec![7u32; N]).and_then(download));
    let v = chain.sync(&ctx).expect("sync");
    assert!(v.iter().all(|&x| x == 7));
}

/// conditional.rs::baseline_kernel_chain_without_dynop — the one test using no
/// erasure, ported 1:1. and_then_host's closure returns Result<()>; the reduction
/// flows out via Arc<Mutex<_>>.
#[test]
fn baseline_kernel_chain_without_dynop() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let sum_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&sum_cell);
    let _final_buf = upload::<u32, claspr::ReadWrite, _>(vec![3u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 9))
        .and_then_host(move |slice: &mut [u32]| {
            *cell.lock().unwrap() = slice.iter().sum();
            Ok(())
        })
        .sync(&ctx)
        .expect("baseline");
    assert_eq!(*sum_cell.lock().unwrap(), 9 * N as u32);
}

/// conditional.rs::dyn_op_wraps_bare_kernel_op — erase a bare kernel op. The
/// boxed op borrows `&kernels`, so the `DeviceDynOp` carries that lifetime.
#[test]
fn dyn_op_wraps_bare_kernel_op() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    use claspr::DeviceSlice;
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let chain: DeviceDynOp<DeviceSlice<u32>> = DeviceDynOp::new(kernels.fill_u32([N], buf, 5));
    let buf = chain.sync(&ctx).expect("sync");
    let mut out = vec![0u32; N];
    buf.read(&mut out).wait().expect("read");
    assert!(out.iter().all(|&v| v == 5));
}

/// conditional.rs::dyn_op_minimal_kernel_chain — erase a kernel+host chain; the
/// sum is captured via Arc<Mutex<_>>.
#[test]
fn dyn_op_minimal_kernel_chain() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let sum_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&sum_cell);

    let chain: DeviceDynOp<claspr::DeviceSlice<u32>> = DeviceDynOp::new(
        upload::<u32, claspr::ReadWrite, _>(vec![3u32; N])
            .and_then(|buf| kernels.fill_u32([N], buf, 9))
            .and_then_host(move |slice: &mut [u32]| {
                *cell.lock().unwrap() = slice.iter().sum();
                Ok(())
            }),
    );
    let _buf = chain.sync(&ctx).expect("a");
    assert_eq!(*sum_cell.lock().unwrap(), 9 * N as u32);
}

/// conditional.rs::dyn_op_picks_branch_with_or_without_kernel — the actual
/// conditional shape: two arms of different concrete chain types (one runs the
/// kernel + captures the sum, the other lifts a literal 0), erased by `DeviceDynOp`.
#[test]
fn dyn_op_picks_branch_with_or_without_kernel() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;
    let sum_cell = Arc::new(Mutex::new(0u32));

    let make = |use_kernel: bool| -> DeviceDynOp<u32> {
        if use_kernel {
            let cell = Arc::clone(&sum_cell);
            DeviceDynOp::new(
                upload::<u32, claspr::ReadWrite, _>(vec![3u32; N])
                    .and_then(move |buf| kernels_ref.fill_u32([N], buf, 9))
                    .and_then_host(move |slice: &mut [u32]| {
                        *cell.lock().unwrap() = slice.iter().sum();
                        Ok(())
                    })
                    .and_then(|_buf| value(0u32)),
            )
        } else {
            DeviceDynOp::new(value(0u32))
        }
    };
    assert_eq!(make(true).sync(&ctx).expect("a"), 0);
    assert_eq!(*sum_cell.lock().unwrap(), 9 * N as u32);
    *sum_cell.lock().unwrap() = 0;
    assert_eq!(make(false).sync(&ctx).expect("b"), 0);
    assert_eq!(*sum_cell.lock().unwrap(), 0);
}

/// conditional.rs::non_taken_branch_closure_does_not_fire — the laziness
/// guarantee: an `if`/`else` that builds one of two `DeviceDynOp<T>`s never
/// constructs the not-taken arm. The unsafe arm's host closure would panic; if
/// branch selection were eager (both arms pre-built) this would panic. Picks the
/// safe arm; asserts no panic + the safe value.
#[test]
fn non_taken_branch_closure_does_not_fire() {
    let Some(ctx) = ctx() else { return };
    let make = |take_safe: bool| -> DeviceDynOp<u32> {
        if take_safe {
            DeviceDynOp::new(value(7u32))
        } else {
            DeviceDynOp::new(
                upload::<u32, claspr::ReadWrite, _>(vec![0u32; N])
                    .and_then_host(|_slice: &mut [u32]| -> claspr::Result<()> {
                        panic!("non-taken branch fired — DeviceDynOp construction must be lazy");
                    })
                    .and_then(|_buf| value(0u32)),
            )
        }
    };
    assert_eq!(make(true).sync(&ctx).expect("safe arm"), 7);
}

/// DeviceDynOp over a MULTI-OUTPUT inner op. `bundle2` has `Output = (u32, u32)`;
/// erasing it loses the per-branch build-time handle but keeps the reconstructed
/// tuple `Output` (via the inner op's `collect`). Pins that erasure works across
/// arity — both `if` arms are different multi-output concrete types.
#[test]
fn dyn_op_erases_multi_output_op() {
    use claspr::eager::bundle2;
    let Some(ctx) = ctx() else { return };

    let pick = |left: bool| -> DeviceDynOp<(u32, u32)> {
        if left {
            DeviceDynOp::new(bundle2(value(1u32), value(2u32)))
        } else {
            DeviceDynOp::new(bundle2(value(3u32).and_then(value), value(4u32)))
        }
    };
    assert_eq!(pick(true).sync(&ctx).expect("left"), (1, 2));
    assert_eq!(pick(false).sync(&ctx).expect("right"), (3, 4));
}
