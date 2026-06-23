//! Coverage for the image transfer ops added in the late-bind /
//! builder refactor — `image.copy_to(...)`, `image.fill(...)`,
//! `image.write(...).submit(...)` (non-blocking variants), and the
//! caller-supplied-dst `image.read(&mut dst)` form.
//!
//! The existing `image_dispatch.rs` test covers the dim/format/access
//! matrix end-to-end via kernels; this file isolates the new
//! transfer primitives without going through kernels at all (so a
//! breakage in `clEnqueueCopyImage`/`clEnqueueFillImage` wiring
//! surfaces as a test failure here without ambiguity about
//! kernel-side state).

use claspr::{
    Context, Image2D, ReadOnly, ReadWrite, WriteOnly,
    image::format::{R32Float, R32G32B32A32Uint, R32Uint},
};

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

/// `image.copy_to(&mut dst)` propagates pixels through
/// `clEnqueueCopyImage`. Writes a known pattern into src, copies,
/// reads dst, confirms equality.
#[test]
fn image2d_copy_to_propagates_pixels() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 8;
    const H: u32 = 4;

    let src = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc src");
    let dst = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc dst");

    let pixels: Vec<u32> = (0..(W * H)).map(|i| 0xCAFE_0000 | i).collect();
    let src = src.write(&pixels).wait().expect("write src");

    let (_src, dst) = src.copy_to(dst).wait().expect("copy src→dst");

    let got: Vec<u32> = dst.read_alloc().wait().expect("read dst");
    assert_eq!(got, pixels, "copy_to should propagate every pixel");
}

/// `image.fill([v; 4])` writes the same 4-component pattern to
/// every pixel via `clEnqueueFillImage`. Uses a 4-channel format
/// (`R32G32B32A32Uint`) so the pattern lands in all four channels
/// of every pixel without spec ambiguity about how the runtime
/// truncates `[T; 4]` to fewer channels.
#[test]
fn image2d_fill_writes_pattern_to_every_pixel() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 2;
    let pattern: [u32; 4] = [10, 20, 30, 40];

    let img = Image2D::<ReadWrite, R32G32B32A32Uint>::alloc(&ctx, W, H).expect("alloc");
    let img = img.fill(pattern).wait().expect("fill");

    let got: Vec<[u32; 4]> = img.read_alloc().wait().expect("read");
    assert_eq!(got.len(), (W as usize) * (H as usize));
    assert!(
        got.iter().all(|&px| px == pattern),
        "fill should land pattern in every pixel; got first {:?}",
        got.first(),
    );
}

/// `image.fill(...)` for a Float-family format. Confirms the
/// fill-pattern dispatch lines up with the format's
/// `SampledTypeFamily` (`Float`).
#[test]
fn image2d_fill_float_format_round_trips() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 6;
    const H: u32 = 3;
    let pattern: [f32; 4] = [1.5, 2.5, 3.5, 4.5];

    let img = Image2D::<ReadWrite, R32Float>::alloc(&ctx, W, H).expect("alloc");
    let img = img.fill(pattern).wait().expect("fill");

    // R32Float is single-channel — only the first component lands.
    let got: Vec<f32> = img.read_alloc().wait().expect("read");
    assert_eq!(got.len(), (W as usize) * (H as usize));
    assert!(
        got.iter().all(|&v| v == 1.5_f32),
        "single-channel fill should land pattern[0] in every pixel",
    );
}

/// `image.read(&mut dst)` (caller-supplied destination) works
/// symmetrically to `buf.read(&mut dst)`. Confirms the
/// non-allocating shape is functional and length-checked.
#[test]
fn image2d_read_into_caller_dst() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 8;
    const H: u32 = 4;

    let img = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc");
    let pixels: Vec<u32> = (0..(W * H)).map(|i| 0xBEEF_0000 | i).collect();
    let img = img.write(&pixels).wait().expect("write");

    let mut got = vec![0u32; (W as usize) * (H as usize)];
    img.read(&mut got).expect("read op").wait().expect("wait");
    assert_eq!(got, pixels);
}

/// `image.read(&mut dst)` returns `Error::LengthMismatch` when
/// dst's length doesn't match the image's pixel count — same
/// shape as `buf.read`'s length check.
#[test]
fn image2d_read_length_mismatch_errors() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 4;

    let img = Image2D::<ReadOnly, R32Uint>::alloc(&ctx, W, H).expect("alloc");

    let mut wrong_size = vec![0u32; 8]; // expected 16
    let err = match img.read(&mut wrong_size) {
        Ok(_) => panic!("expected LengthMismatch, got Ok"),
        Err(e) => e,
    };
    assert!(
        matches!(err, claspr::Error::LengthMismatch { src: 16, dst: 8 }),
        "expected LengthMismatch, got {err:?}",
    );
}

/// `image.write(...).submit()?` enqueues the write
/// non-blocking and returns an `Event`. Waiting on the event
/// surfaces completion; the data must be live until then (here,
/// `pixels` outlives the event).
#[test]
fn image2d_write_submit_returns_event() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 4;

    let img = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc");
    let pixels: Vec<u32> = (1..=(W * H)).collect();
    // Non-blocking submit returns the (rebindable) image plus the write event.
    let (img, event) = img.write(&pixels).submit().expect("submit write");
    event.wait().expect("wait write");

    let got: Vec<u32> = img.read_alloc().wait().expect("read");
    assert_eq!(got, pixels);
}

/// Same shape for `read(...).submit()?` — non-blocking
/// download path, caller waits on the event before reading the dst.
#[test]
fn image2d_read_submit_returns_event() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 4;

    let img = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc");
    let pixels: Vec<u32> = (100..(100 + W * H)).collect();
    let img = img.write(&pixels).wait().expect("write");

    let mut got = vec![0u32; (W as usize) * (H as usize)];
    {
        let op = img.read(&mut got).expect("read op");
        // Non-blocking submit returns the (rebindable) image plus the read event.
        let (_img, event) = op.submit().expect("submit read");
        event.wait().expect("wait read");
    }
    assert_eq!(got, pixels);
}

/// `image.write_bytes(...)` writes raw bytes (no pixel-type
/// round-trip). Useful for byte-oriented sources like image-file
/// loaders.
#[test]
fn image2d_write_bytes_and_read_bytes_round_trip() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 2;
    let pixel_bytes = std::mem::size_of::<u32>();
    let byte_count = (W as usize) * (H as usize) * pixel_bytes;

    let img = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc");
    let raw: Vec<u8> = (0..byte_count as u8).collect();
    let img = img.write_bytes(&raw).wait().expect("write bytes");

    let got = img.read_bytes_alloc().wait().expect("read bytes");
    assert_eq!(got, raw);
}

/// Two writes to the same image, ordered via the move-out form: each verb
/// consumes the image and rebinds it, so the second write is enqueued on the
/// same context queue after the first. Confirms the folded image verbs sequence
/// correctly (the eager-graph replacement for the old `.after(&event)` modifier).
#[test]
fn image2d_write_after_event() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 4;

    let img = Image2D::<ReadWrite, R32Uint>::alloc(&ctx, W, H).expect("alloc");

    let first: Vec<u32> = vec![1u32; (W as usize) * (H as usize)];
    // Non-blocking first write; wait on its event before the second so the
    // ordering is explicit (mirrors the former `.after(&ev)`).
    let (img, ev) = img.write(&first).submit().expect("first write");
    ev.wait().expect("wait first write");

    let second: Vec<u32> = vec![2u32; (W as usize) * (H as usize)];
    let img = img.write(&second).wait().expect("second write after first");

    let got: Vec<u32> = img.read_alloc().wait().expect("read");
    assert!(got.iter().all(|&v| v == 2), "second write should win");
}

/// `image.write()` returning a `WriteOnly` access marker still
/// composes — the access marker is type-state, the transfer ops
/// don't gate on it directly. Confirms that the marker change
/// (WriteOnly host-side ↔ kernel-only WriteOnly) doesn't refuse
/// host-side write or read paths.
#[test]
fn image2d_write_only_marker_still_writes() {
    let Some(ctx) = ctx() else { return };
    const W: u32 = 4;
    const H: u32 = 4;

    let img = Image2D::<WriteOnly, R32Uint>::alloc(&ctx, W, H).expect("alloc");
    let pixels: Vec<u32> = vec![42u32; (W as usize) * (H as usize)];
    let img = img.write(&pixels).wait().expect("write");

    // Reading back from a WriteOnly host marker: today the API
    // permits it (the marker gates kernel-side access, not
    // host-side I/O). If a future change adds host-access marker
    // gating to image transfers, this test will start failing
    // and is the signal to either update the test or document
    // the new gating.
    let got: Vec<u32> = img.read_alloc().wait().expect("read");
    assert!(got.iter().all(|&v| v == 42));
}
