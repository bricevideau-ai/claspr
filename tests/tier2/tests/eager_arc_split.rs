//! Eager-API port of `arc_split.rs`: the shared-read-only-input fan-out pattern.
//!
//! The old `.arc().split::<N>()` existed to share ONE immutable input across N
//! branches via cheap `Arc::clone`s (a refcount bump, not a copy) — model
//! weights, a look-up table, etc. read by several flows. The eager `arc_split`
//! primitive captures that for DEVICE resources (`arc_split(arced(upload(v)))`,
//! one `cl_mem` refcounted into N read-only kernel-arg branches — see
//! `eager_cutover::arc_split_read_only_fan_out` and `eager_diamond.rs`). It does
//! NOT apply to host values: an eager `arc_split` branch is a `Pipe<T>`, so a
//! host reduction (`arc.iter().sum()`) can't run on it at build, and "sharing" a
//! host `Vec` has no device-resource meaning anyway.
//!
//! For a host value the faithful spelling keeps the Arc-clone semantics with std
//! `Arc` + `value`'s by-value handle: `Arc::new(v)` once, `value(Arc::clone(&v))`
//! per branch (a refcount bump, NOT a `v.clone()` copy — the data is read-only),
//! and each branch reduces the `Arc` directly in its `and_then` closure (the
//! by-value handle hands it the `Arc`, which derefs to the value). This mirrors
//! old `arc.split::<3>()` → 3 `Arc::clone`s exactly.

use claspr::eager::{DeviceOpExt, bundle2, bundle3, value};
use claspr::{Context, Error};
use std::sync::Arc;

/// arc_split.rs::arc_split_into_three_branches_share_value — one shared
/// read-only `Arc<Vec>` fanned to three branches, each computing a DIFFERENT
/// reduction over the WHOLE value (sum / product / len). Cheap `Arc::clone` per
/// branch; the by-value handle hands each closure the `Arc` to reduce.
#[test]
fn arc_split_into_three_branches_share_value() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let shared = Arc::new(vec![1u32, 2, 3, 4]);
    let (sum, product, len) = bundle3(
        value(Arc::clone(&shared)).and_then(|a| value(a.iter().sum::<u32>())),
        value(Arc::clone(&shared)).and_then(|a| value(a.iter().product::<u32>())),
        value(shared).and_then(|a| value(a.len() as u32)),
    )
    .sync(&ctx)
    .expect("arc-split chain");
    assert_eq!(*sum, 10);
    assert_eq!(*product, 24);
    assert_eq!(*len, 4);
}

/// arc_split.rs::arc_split_propagates_branch_error — one fan-out branch errors
/// mid-chain; the joining bundle terminal must surface the original rich variant
/// (not the `OpenCl(-1)` cascade). Both branches take an `Arc::clone` of the
/// shared value; branch B ignores it and aborts via `and_then_host`.
#[test]
fn arc_split_propagates_branch_error() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let shared = Arc::new(vec![1u32, 2, 3]);
    let chain = bundle2(
        value(Arc::clone(&shared)).and_then(|a| value(a.iter().sum::<u32>())),
        value(shared)
            .and_then(|_a| value(()))
            .and_then_host(|()| -> claspr::Result<()> {
                Err(Error::Build {
                    log: "branch B aborted".to_string(),
                })
            })
            // after `and_then_host` the `()` flows as a `Pipe<()>` (its handle is
            // the default pipe, not by-value), so the closure binds the pipe.
            .and_then(|_p| value(0u32)),
    );

    let err = chain.sync(&ctx).expect_err("branch B errored");
    assert!(
        matches!(&err, Error::Build { log } if log == "branch B aborted"),
        "got {err:?}",
    );
}

/// arc_split.rs::arc_split_single_does_not_panic — the N=1 edge case: a single
/// branch over one shared `Arc` (the old `split::<1>()` no-op clone). A
/// non-numeric value (`Arc<String>`) reduced via `.len()`.
#[test]
fn arc_split_single_does_not_panic() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let shared = Arc::new("only-input".to_string());
    let chain = value(shared).and_then(|s| value(s.len()));
    assert_eq!(*chain.sync(&ctx).expect("single branch"), 10);
}
