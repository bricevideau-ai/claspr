//! Eager-API method-form fan-out: `vec.fan_out(|i| value(i))` via
//! [`EagerFanOutExt`] must match the free-fn [`eager::fan_out`] result.
//!
//! Mirrors the old `fan_out.rs` `vec_method_form_matches_free_fn` shape.

use claspr::Context;
use claspr::eager::{EagerFanOutExt, EagerOpExt, fan_out, value};

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
fn vec_method_form_matches_free_fn() {
    let Some(ctx) = ctx() else { return };

    let inputs: Vec<u32> = (0..8).collect();

    // Free-fn form.
    let free: Vec<u32> = fan_out(inputs.clone(), |i| value(i.wrapping_mul(2)))
        .sync(&ctx)
        .expect("free-fn fan_out");

    // Method form — delegates to the same free fn.
    let method: Vec<u32> = inputs
        .clone()
        .fan_out(|i| value(i.wrapping_mul(2)))
        .sync(&ctx)
        .expect("method fan_out");

    assert_eq!(free, method);
    assert_eq!(method, vec![0, 2, 4, 6, 8, 10, 12, 14]);
}
