//! `OpShape` — one type-level descriptor per op from which `Output`, the
//! downstream `HandleOf<Op>` and the terminal `CheckoutsOf<Op>` are all projected.
//!
//! Handle/checkouts agreement is now structural (both are projections of the one
//! `DeviceOp::Shape`), so what is left to guard is that each op advertises the
//! RIGHT shape: `require_shape::<S, _>` stops compiling if an op's `Shape` drifts.
//! The `value` case is the reason the shape can't be keyed on `Output` alone — a
//! `u32` output hands downstream BY VALUE there, where a kernel would hand a pipe.

use claspr::eager::{ByValue, DeviceOp, DeviceOpExt, bundle2, fill, value};
use claspr::{Checkout, DeviceSlice, OpShape, Pipe, Single};
use claspr_test_support::ctx;

/// Compile-time proof that `op`'s shape IS `S`.
fn require_shape<S, O: DeviceOp<Shape = S>>(op: O) -> O {
    op
}

/// Compile-time proof that `op`'s downstream handle is `H` and its terminal
/// result is `C` — both read off the one `Shape`.
fn projects<H, C, O>(_: &O)
where
    O: DeviceOp,
    O::Shape: OpShape<Handle = H, Checkouts = C>,
{
}

#[test]
fn op_shapes_project_handle_and_checkouts() {
    let Some(ctx) = ctx() else { return };
    type Buf = DeviceSlice<u32>;

    // Leaf: one pipe downstream, one checkout at the terminal.
    let a = Buf::alloc_zero(&ctx, 8).expect("alloc a");
    let leaf = require_shape::<Single<Buf>, _>(fill(a, 7u32));
    projects::<Pipe<Buf>, Checkout<Buf>, _>(&leaf);

    // Tuple: element-wise, for handles and checkouts alike.
    let b = Buf::alloc_zero(&ctx, 8).expect("alloc b");
    let c = Buf::alloc_zero(&ctx, 8).expect("alloc c");
    let tup = require_shape::<(Single<Buf>, Single<Buf>), _>(bundle2(fill(b, 2u32), fill(c, 3u32)));
    projects::<(Pipe<Buf>, Pipe<Buf>), (Checkout<Buf>, Checkout<Buf>), _>(&tup);

    // By value: same `Output` type as a pipe-handled op would have, different shape.
    let v = require_shape::<ByValue<u32>, _>(value(5u32));
    projects::<u32, Checkout<u32>, _>(&v);

    // Run them so the guard isn't only a type-check: results confirm the ops are real.
    let _filled = leaf.wait().expect("run leaf");
    assert_eq!(*v.sync(&ctx).expect("run value"), 5);

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
