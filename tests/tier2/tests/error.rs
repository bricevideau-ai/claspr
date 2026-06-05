//! Error short-circuiting: when any op in a chain returns Err, the
//! chain stops at that op and surfaces the error from the terminal
//! (`.sync()` or `.run().await`). Later ops must not run.

use claspr::{Context, Error};
use claspr_async::{DeviceOperation, DeviceOperationHostExt, upload, value};
use claspr_test_kernels::kernels;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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

// The host-error slot on `ExecutionContext` carries the original
// Rust `Error` variant across the `and_then_host` user-event
// boundary, so terminals surface the closure's exact variant rather
// than the `Error::OpenCl(-1)` cascade.

#[test]
fn and_then_host_error_stops_chain_immediately() {
    let Some(ctx) = ctx() else { return };
    let counter = Arc::new(AtomicUsize::new(0));
    let c2 = Arc::clone(&counter);

    let result = value(1u32)
        .and_then_host(|_| {
            Err::<(), _>(Error::Build {
                log: "abort".to_string(),
            })
        })
        .and_then_host(move |_n| {
            // Must NOT run — upstream user event was set to negative,
            // worker short-circuits on the source-event wait.
            c2.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .sync(&ctx);

    assert!(
        matches!(&result, Err(Error::Build { log }) if log == "abort"),
        "got {result:?}",
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "downstream closure ran despite upstream error"
    );
}

#[test]
fn error_after_some_device_work_still_propagates() {
    // Upload succeeds → kernel succeeds → host closure errs. Chain
    // surfaces the error from the kernel's downstream waiter.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result = upload!(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 5))
        .and_then_host(|_slice: &mut [u32]| {
            Err::<(), _>(Error::Build {
                log: "post-kernel abort".to_string(),
            })
        })
        .sync(&ctx);
    let err = result.expect_err("expected error");
    assert!(
        matches!(&err, Error::Build { log } if log == "post-kernel abort"),
        "got {err:?}",
    );
}

#[test]
fn nested_chain_error_does_not_skip_outer_terminator() {
    // Verify that an Err inside an and_then-closure-returned chain
    // is still observed by the outer .sync().
    let Some(ctx) = ctx() else { return };
    let result = value(0u32)
        .and_then(|_| {
            value(0u32).and_then_host(|_| {
                Err::<(), _>(Error::Build {
                    log: "nested".to_string(),
                })
            })
        })
        .sync(&ctx);
    assert!(
        matches!(&result, Err(Error::Build { log }) if log == "nested"),
        "got {result:?}",
    );
}
