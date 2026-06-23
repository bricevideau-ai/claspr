//! `DeviceSlice::fill` requires `M: Fillable`. `Frozen` is kernel-RO +
//! host-RO, so it is NOT `Fillable` — building a `fill` op on a `Frozen`
//! buffer must be rejected at compile time.
//!
//! Unified-API guard: the fill verb is now a `DeviceOp` (returns the eager
//! `Fill` node), but the marker bound is unchanged — this is the
//! reunification-era restatement of the deleted `buffer_ops_fill_on_frozen`
//! fixture.

use claspr::{Context, DeviceSlice, Frozen};

fn main() {
    let ctx = Context::any().unwrap();
    let buf: DeviceSlice<u32, Frozen> = DeviceSlice::from_slice(&ctx, &[0u32, 1, 2, 3]).unwrap();
    // `Frozen` lacks `Fillable`; the `fill` bound must reject.
    let _ = buf.fill(7u32);
}
