//! Compile-fail surface for claspr's image trait dispatch, driven
//! by [`ui_test`] via the shared harness in
//! `claspr_test_support::ui` (rlib discovery, rustc wiring, and the
//! trybuild-vs-ui_test rationale live there).
//!
//! Each fixture in `tests/tier1/compile_fail/image/` deliberately
//! passes the wrong host-side `Image2D<A, F>` (or related image
//! type) to a kernel and is expected to fail with a trait-bound
//! error. The kernel side lives in `claspr-test-image-kernels`;
//! this file just invokes rustc per fixture and diffs the captured
//! stderr against the golden `.stderr` files committed alongside.
//!
//! Coverage:
//!
//! - `family_mismatch_*` — kernel says `type=u32` (`Uint` family);
//!   host passes a `Float`/`Sint` format. Should fail at the
//!   `KernelImage2DWriteArg<Uint>` bound.
//! - `access_*` — kernel says `&Image` (read-only access qualifier);
//!   host passes a `WriteOnly` image (which only impls the write
//!   trait variant). Or vice-versa.
//! - `dim_mismatch_*` — host's `Image<N>D<…>` doesn't match the
//!   dim the kernel declared. Wrong trait family entirely.
//! - `view_*` — `Image1DBufferView` access/lifetime checks.
//!
//! `compile_fail/macro` + `compile_pass/macro` cover macro-level
//! diagnostics (no kernel crate needed — a `claspr::kernels!`
//! invocation exercises `expand_kernel` at host compile time, so
//! only the `claspr` extern is required): the >16 runtime-arg
//! friendly error, and the 16-arg boundary as still-valid.
//!
//! ## Running and re-blessing
//!
//! ```text
//! cargo test -p claspr-tier1-tests --test image_compile_fail
//! cargo test -p claspr-tier1-tests --test image_compile_fail -- --bless
//! ```
//!
//! No OpenCL device needed — these are pure compile-time checks.

use claspr_test_support::ui::{Mode, Result, run_compile_tests};

fn main() -> Result<()> {
    run_compile_tests(
        &["claspr", "claspr_test_image_kernels"],
        "cargo test -p claspr-tier1-tests --test image_compile_fail -- --bless",
        &[
            ("compile_fail/image", Mode::Fail),
            ("compile_pass/image", Mode::Pass),
            ("compile_fail/macro", Mode::Fail),
            ("compile_pass/macro", Mode::Pass),
        ],
    )
}
