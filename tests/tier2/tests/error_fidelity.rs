//! Coverage for the host-error slot on `ExecutionContext` — the
//! mechanism that preserves the original Rust `Error` variant across
//! the `and_then_host` user-event boundary. Complements the variant
//! assertions in `error.rs` / `host_and_profile.rs` with three cases
//! they don't reach: closure panics, the async (`.run`) terminal, and
//! concurrent failures in a `bundle!` (first-writer-wins).

use claspr::{Context, Error};
use claspr_async::{DeviceOperation, DeviceOperationHostExt, bundle, value};
use futures::executor::block_on;

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
fn panic_in_host_closure_surfaces_host_panic() {
    // `catch_unwind` in the worker converts a closure panic into
    // `Error::HostPanic(payload)`. The payload is the format-string
    // text (extracted via `downcast_ref::<&str>() / String`).
    let Some(ctx) = ctx() else { return };
    let err = value(())
        .and_then_host(|()| -> claspr::Result<()> { panic!("boom-{}", 42) })
        .sync(&ctx)
        .expect_err("expected panic-converted error");
    assert!(
        matches!(&err, Error::HostPanic(msg) if msg.contains("boom-42")),
        "got {err:?}",
    );
}

#[test]
fn async_terminal_run_also_delivers_rich_variant() {
    // Same rich-variant guarantee for the `.run(&ctx).await` path —
    // ChainFuture::poll prefers the stashed slot over the marker's
    // negative status, mirroring `.sync()`'s contract.
    let Some(ctx) = ctx() else { return };
    let chain = value(()).and_then_host(|()| -> claspr::Result<()> { Err(Error::SvmNotAvailable) });
    let err = block_on(chain.run(&ctx)).expect_err("expected error");
    assert!(matches!(err, Error::SvmNotAvailable), "got {err:?}");
}

#[test]
fn first_writer_wins_when_bundle_branches_both_fail() {
    // Two parallel `and_then_host` branches, both returning Err with
    // distinct variants. Exactly one stashes (first-writer-wins); the
    // other is shadowed. Either branch winning is acceptable — what
    // matters is that the surfaced error is one of the two rich
    // variants, never the `Error::OpenCl(-1)` cascade.
    let Some(ctx) = ctx() else { return };
    let left = value(()).and_then_host(|()| -> claspr::Result<()> {
        Err(Error::Build {
            log: "left-arm".to_string(),
        })
    });
    let right = value(()).and_then_host(|()| -> claspr::Result<()> { Err(Error::SvmNotAvailable) });
    let err = bundle!(left, right).sync(&ctx).expect_err("expected error");
    let acceptable = matches!(&err, Error::Build { log } if log == "left-arm")
        || matches!(err, Error::SvmNotAvailable);
    assert!(
        acceptable,
        "expected one of the two rich branch errors, got {err:?}",
    );
}
