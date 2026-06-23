//! Smoke fixture — confirms the compile-fail harness can actually build a
//! valid unified-API `DeviceOp` chain. Doesn't run; ui_test only checks
//! rustc's exit status.
//!
//! Harness-integrity guard: if the `--extern`/`-L` wiring in
//! `safety_compile_fail.rs` were misconfigured, *every* fixture would fail to
//! compile and the compile-fail suite would pass for the wrong reason. This
//! `Mode::Pass` fixture catches that — a known-good chain must compile.

use claspr::eager::{upload, DeviceOp};
use claspr::DeviceSlice;

#[allow(dead_code)]
fn build_chain() -> impl DeviceOp<Output = DeviceSlice<u32>> {
    upload(vec![1u32, 2, 3])
}

fn main() {}
