//! Eager-API port of `arc_split.rs`: shared-input fan-out + host reductions.
//!
//! The originals split a HOST value (`value(vec).arc().split::<N>()`) to N
//! branches and reduce each on the host (`arc.iter().sum()`, `.product()`,
//! `.len()`). With `value`'s BY-VALUE handle, the eager idiom expresses this
//! directly: `value(vec)` hands the closure the `Vec` (not a pipe), so the
//! reductions compute in-graph — and because `value` is `Clone`, a single
//! `value(vec)` feeds all N branches without an explicit `Arc` (the closure
//! borrows the vec to build each branch). So these port WITHOUT `arc_split` —
//! the eager `arc_split` op is the DEVICE-buffer Arc fan-out tool (each clone a
//! read-only kernel arg), exercised in `eager_cutover::arc_split_read_only_fan_out`
//! and `eager_diamond.rs`; host-value reduction is a different shape that
//! by-value `value` covers without it.

use claspr::eager::{EagerOpExt, bundle3, value};
use claspr::{Context, Error};

#[test]
fn arc_split_into_three_branches_share_value() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    // value(vec) by-value handle → the closure gets the Vec; one source feeds
    // three host reductions (sum / product / len), bundled into a tuple.
    let chain = value(vec![1u32, 2, 3, 4]).and_then(|v| {
        bundle3(
            value(v.iter().sum::<u32>()),
            value(v.iter().product::<u32>()),
            value(v.len() as u32),
        )
    });

    let (sum, product, len) = chain.sync(&ctx).expect("arc-split chain");
    assert_eq!(sum, 10);
    assert_eq!(product, 24);
    assert_eq!(len, 4);
}

#[test]
fn arc_split_propagates_branch_error() {
    // If one fan-out branch errors mid-chain, the joining bundle terminal must
    // surface it. Branch A reduces the shared value; branch B injects an error
    // via an `and_then_host` over a Mappable `()` (the eager host seam preserves
    // the rich Rust variant, so we match it directly).
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let chain = value(vec![1u32, 2, 3]).and_then(|v| {
        let sum = v.iter().sum::<u32>();
        bundle3(
            value(sum),
            value(()).and_then_host(|()| -> claspr::Result<()> {
                Err(Error::Build {
                    log: "branch B aborted".to_string(),
                })
            }),
            value(0u32),
        )
    });

    let err = chain.sync(&ctx).expect_err("branch B errored");
    assert!(
        matches!(&err, Error::Build { log } if log == "branch B aborted"),
        "got {err:?}",
    );
}

#[test]
fn arc_split_single_does_not_panic() {
    // Edge case: a single branch over a non-numeric host value. `value(String)`
    // hands the closure the `String` by value, so `s.len()` computes in-graph.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let chain = value("only-input".to_string()).and_then(|s| value(s.len()));
    assert_eq!(chain.sync(&ctx).expect("single branch"), 10);
}
