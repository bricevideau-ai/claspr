//! Sanity-only fixture for the trybuild test alongside the
//! `compile_fail/` fixtures.
//!
//! ## Why this file exists
//!
//! trybuild has a known intermittent false-success bug in its bulk
//! `cargo check --bins --keep-going` mode (upstream issues #299,
//! #286, #242) where some fixtures' compile errors are dropped from
//! the JSON diagnostic stream and the runner reports them as
//! "succeeded when expected to fail". The bulk mode is triggered
//! only when *all* fixtures in a single `TestCases` invocation are
//! `compile_fail`. Adding even one `pass` fixture flips
//! `has_pass = true` inside trybuild and routes every fixture
//! through the per-bin reliable code path instead.
//!
//! This file is therefore a workaround marker — its content just
//! needs to compile. The accompanying compile_fail fixtures still
//! carry the actual coverage we want from the surface.

use claspr::{image::format::R32Uint, Context, WriteOnly};

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = claspr_test_image_kernels::dim2_uint::kernels(&ctx).unwrap();
    let img = claspr::Image2D::<WriteOnly, R32Uint>::alloc(&ctx, 4, 4).unwrap();
    let _ = kernels.fill_pattern([4usize, 4usize], img, 4u32, 4u32);
}
