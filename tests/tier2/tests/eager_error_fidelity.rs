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
//! TWO of the three cases hit KNOWN eager gaps and are BLOCKED (see comments):
//!   - panic-in-closure: the eager host seam (`run_host_seam`) runs the closure
//!     WITHOUT `catch_unwind`, so a panic unwinds the thread rather than being
//!     converted to `Error::HostPanic`. There is no eager equivalent.
//!   - async `.run().await`: the eager API has no async terminal (only `.sync`).
//!
//! The third case (bundle first-writer-wins) ports 1:1.

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

// BLOCKED: panic→HostPanic conversion — eager host seam (`run_host_seam`)
// invokes the closure without `catch_unwind`, so a closure panic unwinds
// instead of surfacing as `Error::HostPanic`. Needs a catch_unwind wrapper in
// the eager host seam (deliberate non-goal of the eager port today). Original
// `panic_in_host_closure_surfaces_host_panic` cannot be expressed.

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
