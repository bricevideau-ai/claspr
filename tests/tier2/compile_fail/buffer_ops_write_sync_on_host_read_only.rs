//! `DeviceSlice::write_sync` (the blocking BORROWING upload verb) mirrors
//! `DeviceSlice::write`'s marker bound: it requires `M: HostWritable`. Being
//! blocking + borrowing doesn't relax the access contract — a host-read-only
//! buffer (`HostReadOnly` = host-RO + kernel-RW) must still reject the write at
//! compile time, exactly like the async `write`
//! (`buffer_ops_write_on_host_read_only`).

use claspr::{Context, DeviceSlice, HostReadOnly};

fn main() {
    let ctx = Context::any().unwrap();
    let mut buf: DeviceSlice<u32, HostReadOnly> =
        DeviceSlice::from_slice(&ctx, &[0u32, 1, 2, 3]).unwrap();
    // `HostReadOnly` lacks `HostWritable`; the `write_sync` bound must reject.
    let _ = buf.write_sync(&[7u32, 8, 9, 10]);
}
