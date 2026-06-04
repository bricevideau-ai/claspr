//! `device_slice_fill` requires `M: KernelWritable`.
//! `ReadOnly` is kernel-RO — fill must reject.

use claspr::{Context, DeviceSlice, ReadOnly};
use claspr_async::device_slice_fill;

fn main() {
    let ctx = Context::any().unwrap();
    let buf: DeviceSlice<u32, ReadOnly> =
        DeviceSlice::from_slice(&ctx, &[0u32, 1, 2, 3]).unwrap();
    // ReadOnly lacks KernelWritable; the wrapper's bound must reject.
    let _ = device_slice_fill(buf, 7u32);
}
