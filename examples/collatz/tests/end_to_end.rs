//! Stage-3 single-source end-to-end test.
//!
//! No standalone kernel crate, no manual signature declarations — the
//! kernel function lives in `examples/collatz/src/kernels.rs` and
//! drives both the host wrapper (via `#[claspr::kernel]`) and the
//! kernel-side compilation (via `claspr_build::compile_from_host`).
//!
//! Skips silently if no OpenCL device is reachable.

use claspr::Context;
use collatz_example::compiled::Kernels;

/// Well-known Collatz sequence lengths (1-indexed input → length to
/// reach 1). OEIS A006577.
const CHECKS: &[(u32, u32)] = &[(1, 0), (2, 1), (3, 7), (4, 2), (27, 111)];

#[test]
fn collatz_single_source() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP collatz_single_source: no OpenCL device ({e})");
            return;
        }
    };

    let kernels = Kernels::load(&ctx).expect("Kernels::load");

    let n: usize = 1024;
    let mut data: Vec<u32> = (1..=n as u32).collect();

    let buf = ctx.upload(&data).expect("upload");
    kernels
        .collatz_kernel(&ctx, [n], &buf)
        .expect("launch collatz_kernel");
    ctx.download(&buf, &mut data).expect("download");

    for &(input, expected) in CHECKS {
        let idx = (input - 1) as usize;
        assert_eq!(
            data[idx], expected,
            "collatz({input}) = {} (expected {expected})",
            data[idx],
        );
    }
}
