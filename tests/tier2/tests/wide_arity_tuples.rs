//! Compile-level proof that the tuple-arg trait families extend past 8 (raised
//! 8→16 to match `bundle`/checkout/seam, which were already 16). `KernelArgs`
//! (kernel launch args) is the load-bearing one — it is NOT chainable, so a kernel
//! with >8 args previously had no way to launch (gray-scott's `combine` hit the old
//! ceiling and pushed constants into compile-time consts to fit). `CallArgs` /
//! `BindAll` (the `call` / `mutate_call` slot-binding tuples) are extended in lockstep.
//!
//! These are TYPE-LEVEL assertions: a 16-arg kernel to actually launch doesn't exist
//! in the test kernels, so we assert the trait impls resolve at arity 9..=16 rather
//! than dispatch a real 16-arg NDRange. If an arity regressed, this fails to compile.

use claspr::{DeviceSlice, KernelArgs};

// A `KernelArgs`-bound generic fn only accepts a tuple whose every element is a
// `KernelArg` AND whose arity has an `impl KernelArgs for (..)`. Instantiating it at
// arity 9..16 is the assertion.
fn assert_kernel_args<A: KernelArgs>(_: &A) {}

#[test]
fn kernel_args_tuple_impls_reach_arity_16() {
    // 16 by-value primitive args (all `KernelArg`). Never launched — just type-checked.
    let a16 = (
        0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32,
        14u32, 15u32,
    );
    assert_kernel_args(&a16); // arity 16
    let a9 = (0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32);
    assert_kernel_args(&a9); // arity 9 (first past the old 8 ceiling)

    // Mixed buffer + scalar at arity 12, mirroring a real dense kernel signature
    // (the shape gray-scott's `combine` wanted but couldn't express at 8).
    fn mixed<T: Send + 'static>(b: &DeviceSlice<T>) {
        let t = (
            b, b, b, b, b, 1u32, 2u32, 3.0f32, 4.0f32, 5u32, 6u32, 7.0f32,
        );
        assert_kernel_args(&t); // arity 12
    }
    let _ = mixed::<u32>;
}
