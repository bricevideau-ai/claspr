//! `OutputShape` — the canonical `Handle`/`Checkouts` shape of a `DeviceOp::Output`.
//!
//! The whole point of the trait (and the `examples/cg` `solve_with` where-clause it
//! shrinks) is the invariant: for EVERY op, `Output: OutputShape<Handle = Handle,
//! Checkouts = Checkouts>`. If a `DeviceOp` impl ever grows a non-canonical `Handle`
//! or `Checkouts` (or an `OutputShape` impl drifts), `require_canonical` below stops
//! compiling — a compile-time guard, exercised at runtime here on a leaf and a tuple.

use claspr::eager::{DeviceOp, DeviceOpExt, OutputShape, bundle2, fill};
use claspr::{Context, DeviceSlice};

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            None
        }
    }
}

/// Compile-time proof that `O`'s `Handle`/`Checkouts` ARE the `OutputShape` of its
/// `Output`. Passing a real op through it type-checks ONLY if the invariant holds.
fn require_canonical<O>(op: O) -> O
where
    O: DeviceOp,
    O::Output: OutputShape<Handle = O::Handle, Checkouts = O::Checkouts>,
{
    op
}

#[test]
fn output_shape_is_canonical_for_leaf_and_tuple() {
    let Some(ctx) = ctx() else { return };

    // Leaf: Fill has Output = DeviceSlice, so OutputShape::Handle must be its Handle.
    let a = DeviceSlice::<u32>::alloc_zero(&ctx, 8).expect("alloc a");
    let leaf = require_canonical(fill(a, 7u32));

    // Tuple: bundle2's Output = (DeviceSlice, DeviceSlice); OutputShape::Handle is the
    // element-wise tuple of pipes — exactly the bundle's Handle.
    let b = DeviceSlice::<u32>::alloc_zero(&ctx, 8).expect("alloc b");
    let c = DeviceSlice::<u32>::alloc_zero(&ctx, 8).expect("alloc c");
    let tup = require_canonical(bundle2(fill(b, 2u32), fill(c, 3u32)));

    // Run both so the guard isn't only a type-check: results confirm the ops are real.
    let _filled = leaf.wait().expect("run leaf");

    let (bo, co) = tup.sync(&ctx).expect("run tuple");
    let bv = bo.map().wait().expect("read b");
    let cv = co.map().wait().expect("read c");
    assert!(
        bv.iter().all(|&v| v == 2),
        "bundle branch 0 filled 2: {:?}",
        &bv[..4]
    );
    assert!(
        cv.iter().all(|&v| v == 3),
        "bundle branch 1 filled 3: {:?}",
        &cv[..4]
    );
}
