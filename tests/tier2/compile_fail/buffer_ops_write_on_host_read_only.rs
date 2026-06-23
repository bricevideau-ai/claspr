//! `DeviceSlice::write` requires `M: HostWritable` (the impl block at
//! claspr/src/buffer.rs). `HostReadOnly` is host-RO + kernel-RW, so it is NOT
//! `HostWritable` — building a `write` op on a `HostReadOnly` buffer must be
//! rejected at compile time.
//!
//! Unified-API restatement of the deleted `buffer_ops_write_on_host_read_only`
//! fixture: the free `claspr_async::device_slice_write` wrapper is gone; the
//! write verb is now the `DeviceSlice::write` method (returns the eager
//! `WriteDevice` node), but the marker bound is unchanged.

use claspr::{Context, DeviceSlice, HostReadOnly};

fn main() {
    let ctx = Context::any().unwrap();
    let buf: DeviceSlice<u32, HostReadOnly> =
        DeviceSlice::from_slice(&ctx, &[0u32, 1, 2, 3]).unwrap();
    // `HostReadOnly` lacks `HostWritable`; the `write` bound must reject.
    let _ = buf.write(vec![7u32, 8, 9, 10]);
}
