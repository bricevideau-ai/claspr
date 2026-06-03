//! Compile-fail surface for claspr's image trait dispatch.
//!
//! Each fixture in `tests/tier1/compile_fail/image/` deliberately
//! passes the wrong host-side `Image2D<A, F>` to a kernel and is
//! expected to fail with a trait-bound error. The kernel side
//! lives in `claspr-test-image-kernels`; this file just wires
//! up `trybuild` to assert that the host call site rejects the
//! bad input.
//!
//! Coverage:
//!
//! - `family_mismatch_*` — kernel says `type=u32` (`Uint` family);
//!   host passes a `Float`/`Sint` format. Should fail at the
//!   `KernelImage2DWriteArg<Uint>` bound.
//! - `access_*` — kernel says `&Image` (read-only access qualifier);
//!   host passes a `WriteOnly` image (which only impls the write
//!   trait variant). Or vice-versa.
//!
//! Run via `cargo test -p claspr-tier1-tests --test image_compile_fail`.
//! No OpenCL device needed — these are pure compile checks.

#[test]
fn image_compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("compile_fail/image/*.rs");
}
