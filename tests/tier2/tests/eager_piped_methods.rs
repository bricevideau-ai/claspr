//! Piped-buffer verb methods (reunification stage 6, §7): a buffer produced
//! *upstream* in the graph — a `Pipe<buffer>`, the build-time handle handed to an
//! `and_then` closure — supports the same verbs as a concrete `DeviceSlice`
//! (`write` / `read` / `fill` / `copy_to`), and the pipe-uninit types support
//! `write` / `fill`. Each delegates to the matching eager free fn, so a piped
//! buffer reads as a buffer: `device_alloc_uninit(n).and_then(|u| u.write(data))`.
//!
//! These tests fail to COMPILE if the inherent `Pipe<...>` methods are missing,
//! and fail to PASS if they don't delegate to the right enqueue.

use claspr::Context;
use claspr::eager::{DeviceOpExt, alloc_zero, device_alloc_uninit, download, upload};

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

/// `device_alloc_uninit(n).and_then(|u| u.write(data))` — the piped uninit
/// buffer's `.write` verb transitions it to init by uploading host data.
#[test]
fn piped_uninit_write_method() {
    let Some(ctx) = ctx() else { return };
    let data: Vec<u32> = (0..N as u32).collect();
    let result = device_alloc_uninit(N)
        .and_then(|u| u.write(data.clone()))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc_uninit + piped write + download");
    assert_eq!(*result, data);
}

/// `device_alloc_uninit(n).and_then(|u| u.fill(v))` — the piped uninit buffer's
/// `.fill` verb fills every slot (transitioning to init).
#[test]
fn piped_uninit_fill_method() {
    let Some(ctx) = ctx() else { return };
    let result = device_alloc_uninit(N)
        .and_then(|u| u.fill(7u32))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc_uninit + piped fill + download");
    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 7));
}

/// `alloc_zero(N).and_then(|buf| buf.fill(v))` — the piped (init) buffer's
/// `.fill` verb refills an already-allocated buffer.
#[test]
fn piped_init_fill_method() {
    let Some(ctx) = ctx() else { return };
    let result = alloc_zero::<u32>(N)
        .and_then(|buf| buf.fill(13u32))
        .and_then(download)
        .sync(&ctx)
        .expect("alloc_zero + piped fill + download");
    assert!(result.iter().all(|&v| v == 13));
}

/// `upload(v).and_then(|buf| buf.write(other))` — the piped (init) buffer's
/// `.write` verb overwrites the uploaded contents host-side.
#[test]
fn piped_init_write_method() {
    let Some(ctx) = ctx() else { return };
    let result = upload(vec![0u32; N])
        .and_then(|buf| buf.write(vec![42u32; N]))
        .and_then(download)
        .sync(&ctx)
        .expect("upload + piped write + download");
    assert!(result.iter().all(|&v| v == 42));
}

/// `upload(v).and_then(|buf| buf.read())` — the piped buffer's `.read` verb
/// downloads to a fresh Vec, equivalent to the free `download`.
#[test]
fn piped_read_method() {
    let Some(ctx) = ctx() else { return };
    let data: Vec<u32> = (100..100 + N as u32).collect();
    let result = upload(data.clone())
        .and_then(|buf| buf.read())
        .sync(&ctx)
        .expect("upload + piped read");
    assert_eq!(*result, data);
}

/// `upload(a).and_then(|src| src.copy_to(dst))` — the piped buffer's `.copy_to`
/// verb does a device-to-device copy into a second buffer, yielding `(src, dst)`.
#[test]
fn piped_copy_to_method() {
    let Some(ctx) = ctx() else { return };
    let data: Vec<u32> = (0..N as u32).collect();
    // Allocate the destination concretely, then copy the uploaded src into it.
    // The piped `.copy_to` wants a concrete `DeviceSlice`, so `into_inner` the dst.
    let dst = alloc_zero::<u32>(N)
        .wait_on(&ctx)
        .expect("dst alloc")
        .into_inner();
    // Terminal is the `and_then` (single output): one `Checkout<(src, dst)>`.
    // `into_inner` to own the pair so `dst` can feed the downstream `download`.
    let (_src, dst) = upload(data.clone())
        .and_then(|src| src.copy_to(dst))
        .sync(&ctx)
        .expect("upload + piped copy_to")
        .into_inner();
    // The copy landed in dst: download it and compare to the original data.
    let back = download(dst).sync(&ctx).expect("download dst");
    assert_eq!(*back, data);
}
