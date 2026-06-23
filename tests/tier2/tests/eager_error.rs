//! Eager-API port of `error.rs`: when any op in a chain returns Err, the chain
//! stops at that op and surfaces the error from the terminal (`.sync()`). Later
//! ops must not run.
//!
//! Old → new mapping:
//!   `value(x)`            → `value(x)` (eager `value`; scalar/`()` is `Mappable`,
//!                            so the host seam's "view" is the value by-copy)
//!   `upload!(v)`          → `upload(v)`
//!   `.and_then_host(|_|…)`→ same method on `DeviceOpExt`; closure `Err` surfaces
//!                            at the terminal via `?` (see eager_cutover
//!                            `eager_and_then_host_error_propagates`).
//!
//! All three test fns port 1:1 — same N, error variants, and the
//! downstream-closure-must-not-run counter assertion. The eager host seam
//! propagates the closure's exact `Error` variant (no `OpenCl(-1)` cascade),
//! and short-circuits before the downstream `and_then_host` builds/executes.

use claspr::eager::{DeviceOpExt, upload, value};
use claspr::{Context, Error};
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
            Err::<(), _>(Error::Build {
                log: "abort".to_string(),
            })
        })
        .and_then_host(move |_n| {
            // Must NOT run — the upstream host seam returned Err, so the chain
            // short-circuits before this closure is reached.
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
    // Upload succeeds → kernel succeeds → host closure errs. Chain surfaces the
    // error from the host seam.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result = upload(vec![0u32; N])
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
    // An Err inside an and_then-closure-returned chain is still observed by the
    // outer `.sync()`.
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
