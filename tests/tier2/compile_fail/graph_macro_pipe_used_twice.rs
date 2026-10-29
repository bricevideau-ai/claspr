//@no-rustfix
//! `graph!` is syntax over nested `and_then` closures, so it inherits their
//! ownership checks: a pipe bound by one step moves into the closure of the
//! step that uses it, and using it again in a later step is a use-after-move.
//!
//! The `no-rustfix` directive above stops ui_test from applying rustc's
//! `.clone()` suggestion, whose span lies partly in the macro's definition.

use claspr::prelude::*;
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let ks = kernels::kernels(&ctx).unwrap();
    let buf = |v: u32| DeviceSlice::<u32>::from_slice(&ctx, &[v; 4]).unwrap();
    let _ = graph! {
        let (a, b, out) = ks.add_u32([4usize], buf(1), buf(2), buf(0));
        let (_a, _b, _out) = ks.add_u32([4usize], a, b, out);
        // `a` was moved into the previous step: using it again must not compile.
        ks.scale_u32([4usize], a, 2u32)
    };
}
