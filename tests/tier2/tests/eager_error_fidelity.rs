//! Eager-API port of `error_fidelity.rs`. Three cases beyond the variant
//! assertions in `eager_error.rs` / `eager_host_and_profile.rs`: closure
//! panics, the async (`.run`) terminal, and concurrent failures in a bundle
//! (first-writer-wins). All three port 1:1.
//!
//! Old → new mapping:
//!   `value(x)`        → `value(x)` (eager)
//!   `bundle!(l, r)`   → `bundle2(l, r)`
//!   `.and_then_host`  → same method on `EagerOpExt`
//!   `.run(&ctx)`      → same async terminal on `EagerOpExt` (async-events)
//!
//! All three exercise the host-error slot: the eager host seam (`run_host_seam`)
//! runs the closure on a worker thread under `catch_unwind`, stashing the rich
//! `Error` variant (or `HostPanic`) before signalling its user event negative.
//! Both terminals (`sync` + `run`) prefer the stashed variant over the
//! `OpenCl(-1)` cascade.

use claspr::eager::{EagerOpExt, bundle2, value};
use claspr::{Context, Error};
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

// A panic inside an `and_then_host` closure is caught by the eager host seam
// (`run_host_seam`'s `catch_unwind`) and surfaced as `Error::HostPanic(msg)`
// at the terminal — not an unwind of the caller, and not the `OpenCl(-1)`
// cascade. The payload is the FORMATTED panic text (a format-string panic, to
// prove interpolated runtime content survives the `catch_unwind` → `HostPanic`
// conversion, not just a static literal).
#[test]
fn panic_in_host_closure_surfaces_host_panic() {
    let Some(ctx) = ctx() else { return };
    let err = value(())
        .and_then_host(|()| -> claspr::Result<()> { panic!("boom-{}", 42) })
        .sync(&ctx)
        .expect_err("expected error");
    match err {
        Error::HostPanic(msg) => assert!(msg.contains("boom-42"), "msg was {msg:?}"),
        other => panic!("expected HostPanic, got {other:?}"),
    }
}

// async_terminal_run_also_delivers_rich_variant — the same rich-variant
// guarantee for the `.run(&ctx).await` path. The eager async terminal exists
// (`EagerChainFuture`); its poll prefers the stashed host-error slot over the
// marker's cascade, mirroring `.sync()`'s contract.
#[test]
fn async_terminal_run_also_delivers_rich_variant() {
    let Some(ctx) = ctx() else { return };
    let chain = value(()).and_then_host(|()| -> claspr::Result<()> { Err(Error::SvmNotAvailable) });
    let err = block_on(chain.run(&ctx)).expect_err("expected error");
    assert!(matches!(err, Error::SvmNotAvailable), "got {err:?}");
}

#[test]
fn first_writer_wins_when_bundle_branches_both_fail() {
    // Two parallel `and_then_host` branches, both returning Err with distinct
    // variants. Whichever surfaces, it must be one of the two rich variants,
    // never the `Error::OpenCl(-1)` cascade.
    let Some(ctx) = ctx() else { return };
    let left = value(()).and_then_host(|()| -> claspr::Result<()> {
        Err(Error::Build {
            log: "left-arm".to_string(),
        })
    });
    let right = value(()).and_then_host(|()| -> claspr::Result<()> { Err(Error::SvmNotAvailable) });
    let err = bundle2(left, right).sync(&ctx).expect_err("expected error");
    let acceptable = matches!(&err, Error::Build { log } if log == "left-arm")
        || matches!(err, Error::SvmNotAvailable);
    assert!(
        acceptable,
        "expected one of the two rich branch errors, got {err:?}",
    );
}
