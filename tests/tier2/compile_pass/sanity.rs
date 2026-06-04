//! Smoke fixture — confirms the compile-fail harness can actually
//! build a Tier 2 chain. Doesn't run; ui_test only checks rustc's
//! exit status.

use claspr::DeviceSlice;
use claspr_async::{upload, DeviceOperation};

#[allow(dead_code)]
fn build_chain() -> impl DeviceOperation<Output = DeviceSlice<u32>> {
    upload(vec![1u32, 2, 3])
}

fn main() {}
