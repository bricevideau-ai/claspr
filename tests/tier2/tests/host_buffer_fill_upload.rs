//! `host_buffer_filled` / `host_buffer_upload` + the `host_buffer!`
//! macro — host-pinned analog of `device_slice_filled` / `_upload` /
//! `device_slice!`.
//!
//! HostBuffer storage is host-accessible, so fill / upload are pure
//! host memcpy operations through the persistent map (no clEnqueue,
//! no event tracking). The OpenCL runtime *should* handle host↔device
//! coherency at the next kernel-launch boundary that consumes the
//! buffer, but rusticl is strict about explicit unmap/remap around
//! kernel access — HostBuffer's current "persistent map for the whole
//! lifetime" design doesn't honour that. So these tests verify the
//! primitives produce host-visible-correct buffers (via Deref) but
//! intentionally do NOT chain through a kernel that consumes the
//! buffer — that's a separate HostBuffer coherency story.

use claspr::{Buffer, Context};
use claspr_async::{DeviceOperation, host_buffer, host_buffer_filled, host_buffer_upload};

const N: usize = 64;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

#[test]
fn host_buffer_filled_produces_host_visible_pattern() {
    // Alloc + slice-fill via DerefMut. Read back through Deref.
    let Some(ctx) = ctx() else { return };
    let buf = host_buffer_filled(42u32, N).sync(&ctx).expect("filled");
    assert_eq!(buf.len(), N);
    assert!(buf.iter().all(|&v| v == 42));
}

#[test]
fn host_buffer_upload_carries_host_literal() {
    // Alloc + copy_from_slice via the from_slice path. Round-trips
    // exactly.
    let Some(ctx) = ctx() else { return };
    let src = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let buf = host_buffer_upload::<u32, _>(src.clone())
        .sync(&ctx)
        .expect("upload");
    assert_eq!(&buf[..], &src[..]);
}

#[test]
fn macro_host_buffer_repeat_arm() {
    let Some(ctx) = ctx() else { return };
    let buf = host_buffer![7u32; N].sync(&ctx).expect("macro repeat");
    assert!(buf.iter().all(|&v| v == 7));
}

#[test]
fn macro_host_buffer_literal_arm() {
    let Some(ctx) = ctx() else { return };
    let buf = host_buffer![100u32, 200, 300, 400]
        .sync(&ctx)
        .expect("macro literal");
    assert_eq!(&buf[..], &[100u32, 200, 300, 400]);
}
