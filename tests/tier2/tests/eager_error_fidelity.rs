//! Eager-API port of `error_fidelity.rs`. Three cases beyond the variant
//! assertions in `eager_error.rs` / `eager_host_and_profile.rs`: closure
//! panics, the async (`.run`) terminal, and concurrent failures in a bundle
//! (first-writer-wins).
//!
//! Old → new mapping:
//!   `value(x)`        → `value(x)` (eager)
//!   `bundle!(l, r)`   → `bundle2(l, r)`
//!   `.and_then_host`  → same method on `EagerOpExt`
//!
//! ONE of the three cases still hits a KNOWN eager gap and is BLOCKED:
//!   - async `.run().await`: the eager API has no async terminal (only `.sync`).
//!
//! The panic-in-closure case now ports: the eager host seam (`run_host_seam`)
//! wraps the closure in `catch_unwind`, converting a panic to
//! `Error::HostPanic`. The bundle first-writer-wins case ports 1:1.

use claspr::eager::{EagerOpExt, bundle2, value};
use claspr::{Context, Error};

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
// cascade. The message carries the panic literal.
#[test]
fn panic_in_host_closure_surfaces_host_panic() {
    let Some(ctx) = ctx() else { return };
    let err = value(())
        .and_then_host(|()| -> claspr::Result<()> { panic!("boom") })
        .sync(&ctx)
        .expect_err("expected error");
    match err {
        Error::HostPanic(msg) => assert!(msg.contains("boom"), "msg was {msg:?}"),
        other => panic!("expected HostPanic, got {other:?}"),
    }
}

// BLOCKED: async `.run().await` terminal — eager has no async/`ChainFuture`
// terminal (only `.sync()`). Original `async_terminal_run_also_delivers_rich_variant`
// needs an eager async terminal primitive.

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
