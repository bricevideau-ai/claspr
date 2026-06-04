//! `device_slice_write` requires `M: HostWritable`.
//! `HostReadOnly` is host-RO + kernel-RW — write must reject.

use claspr::{Context, DeviceSlice, HostReadOnly};
use claspr_async::device_slice_write;

fn main() {
    let ctx = Context::any().unwrap();
    let buf: DeviceSlice<u32, HostReadOnly> =
        DeviceSlice::from_slice(&ctx, &[0u32, 1, 2, 3]).unwrap();
    // HostReadOnly lacks HostWritable; the wrapper's bound must reject.
    let _ = device_slice_write(buf, vec![7u32, 8, 9, 10]);
}
