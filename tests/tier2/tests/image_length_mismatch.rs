//! Regression tests for review finding #3: image `write`/`read` length mismatch
//! must surface as `Err(LengthMismatch)` at the terminal — NOT an `assert!` panic
//! — and the length check must live off the constructor so the ops stay
//! Tier-2-composable (an `and_then` closure returns `U: DeviceOp`, not
//! `Result<U>`, so a fallible constructor would bar mid-graph use).
//!
//! Before the fix: `image_write_op`/`image_write_bytes_op` panicked via
//! `assert_eq!`; `image_read_op` returned `Result<ImageRead>` at construction
//! (so `.and_then(|img| img.read(dst))` didn't compile without `.unwrap()`).
//! After: both are infallible constructors that validate in `check_ready`
//! (the atomicity pre-pass) + an `execute` backstop.

use claspr::image::format::R32Uint;
use claspr::{Context, Error, Image2D, ReadWrite};

fn ctx() -> Option<Context> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return None;
        }
    };
    if !ctx.device().cl3().image_support().unwrap_or(false) {
        eprintln!("SKIP: device has no image support");
        return None;
    }
    Some(ctx)
}

const W: u32 = 8;
const H: u32 = 4; // 32 pixels

/// `write` with too-few pixels returns `Err(LengthMismatch)` at the terminal
/// instead of panicking. (Was an `assert_eq!` in `image_write_op`.)
#[test]
fn write_wrong_pixel_count_errors_not_panics() {
    let Some(ctx) = ctx() else { return };
    let img = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc");
    let wrong = vec![0u32; (W * H) as usize - 1]; // one short

    let result = img.write(&wrong).wait();
    match result {
        Err(Error::LengthMismatch { src, dst }) => {
            assert_eq!(src, (W * H) as usize);
            assert_eq!(dst, (W * H) as usize - 1);
        }
        Ok(_) => panic!("expected LengthMismatch, write succeeded"),
        Err(e) => panic!("expected LengthMismatch, got {e:?}"),
    }
}

/// `write_bytes` with the wrong byte count returns `Err(LengthMismatch)`.
#[test]
fn write_bytes_wrong_len_errors_not_panics() {
    let Some(ctx) = ctx() else { return };
    let img = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc");
    let expected_bytes = (W * H) as usize * std::mem::size_of::<u32>();
    let wrong = vec![0u8; expected_bytes + 4]; // one pixel too many

    match img.write_bytes(&wrong).wait() {
        Err(Error::LengthMismatch { src, dst }) => {
            assert_eq!(src, expected_bytes);
            assert_eq!(dst, expected_bytes + 4);
        }
        Ok(_) => panic!("expected LengthMismatch, write_bytes succeeded"),
        Err(e) => panic!("expected LengthMismatch, got {e:?}"),
    }
}

/// `read` into a too-large dst returns `Err(LengthMismatch)` at the terminal.
/// (Previously a construction-time `Result`; now surfaces via `check_ready`.)
#[test]
fn read_wrong_dst_len_errors() {
    let Some(ctx) = ctx() else { return };
    let img = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc");
    let mut too_big = vec![0u32; (W * H) as usize + 5];

    match img.read(&mut too_big).wait() {
        Err(Error::LengthMismatch { src, dst }) => {
            assert_eq!(src, (W * H) as usize);
            assert_eq!(dst, (W * H) as usize + 5);
        }
        Ok(_) => panic!("expected LengthMismatch, read succeeded"),
        Err(e) => panic!("expected LengthMismatch, got {e:?}"),
    }
}

/// A correctly-sized read/write round-trips cleanly (the length checks in
/// `check_ready` are no-ops on the matching path). Also the compile-level proof
/// of the Tier-2 gap the fix closes: `read`/`write` now return a bare
/// `ImageRead`/`ImageWrite` (a `DeviceOp`), not `Result<_>`, so they can appear
/// in an `and_then` closure (which must yield `U: DeviceOp`). If either verb
/// still returned `Result`, this file would not compile.
#[test]
fn matching_lengths_round_trip_and_ops_are_bare_device_ops() {
    let Some(ctx) = ctx() else { return };
    let pixels: Vec<u32> = (0..(W * H)).map(|i| 0xF00D_0000 | i).collect();
    let img = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc");

    // `write` yields a bare `ImageWrite` (would not compile as `Result` here).
    let write_op = img.write(&pixels);
    let img = write_op.wait().expect("write");

    // `read` yields a bare `ImageRead` (the closed Tier-2 gap).
    let mut out = vec![0u32; (W * H) as usize];
    let read_op = img.read(&mut out);
    read_op.wait().expect("read");
    assert_eq!(out, pixels);
}

/// An image→image `copy_to` across mismatched dims now errors client-side
/// (`InvalidArgument`) in the atomicity pre-pass, instead of relying on a bare
/// driver `CL_INVALID_*` at enqueue.
#[test]
fn copy_to_mismatched_dims_errors_client_side() {
    let Some(ctx) = ctx() else { return };
    let src = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc src");
    let dst = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W * 2, H).expect("alloc dst");

    match src.copy_to(dst).wait() {
        Err(Error::InvalidArgument(msg)) => assert!(msg.contains("dimensions differ")),
        Ok(_) => panic!("expected InvalidArgument, copy succeeded"),
        Err(e) => panic!("expected InvalidArgument, got {e:?}"),
    }
}
