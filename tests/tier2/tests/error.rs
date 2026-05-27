//! Error short-circuiting: when any op in a chain returns Err, the
//! chain stops at that op and surfaces the error from the terminal
//! (`.sync()` or `.run().await`). Later ops must not run.

use claspr::{Context, Error};
use claspr_async::{DeviceOperation, DeviceOperationHostExt, download, upload, value};
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

#[test]
fn and_then_host_error_stops_chain_immediately() {
    let Some(ctx) = ctx() else { return };
    let counter = Arc::new(AtomicUsize::new(0));
    let c2 = Arc::clone(&counter);

    let result = value(1u32)
        .and_then_host(|_| {
            Err::<u32, _>(Error::Build {
                log: "abort".to_string(),
            })
        })
        .and_then_host(move |n| {
            // Must NOT run — chain errored above.
            c2.fetch_add(1, Ordering::SeqCst);
            Ok(n)
        })
        .sync(&ctx);

    assert!(
        matches!(result, Err(Error::Build { ref log }) if log == "abort"),
        "got {result:?}"
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "downstream closure ran despite upstream error"
    );
}

#[test]
fn error_after_some_device_work_still_propagates() {
    // Upload succeeds → kernel succeeds → host closure errs. The
    // chain returns the host error. Earlier device work has already
    // run, but the terminator surfaces the failure path correctly.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 5))
        .and_then(download)
        .and_then_host(|_vec| {
            Err::<Vec<u32>, _>(Error::Build {
                log: "post-download abort".to_string(),
            })
        })
        .sync(&ctx);
    assert!(
        matches!(result, Err(Error::Build { ref log }) if log == "post-download abort"),
        "got {result:?}"
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
                Err::<u32, _>(Error::Build {
                    log: "nested".to_string(),
                })
            })
        })
        .sync(&ctx);
    assert!(
        matches!(result, Err(Error::Build { ref log }) if log == "nested"),
        "got {result:?}"
    );
}
