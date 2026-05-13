//! End-to-end smoke test driven by the build-time generated module.
//!
//! The interesting bit relative to claspr's earlier
//! `tests/collatz.rs`: there's no `claspr::compile(...)` call at
//! runtime. The kernel is compiled to SPIR-V at *build* time by
//! `claspr-build` (see `build.rs`), and the resulting `Kernels` struct
//! is `include!()`d in `src/lib.rs`. The test only does the
//! load → upload → launch → download → assert dance.
//!
//! Skips silently if no OpenCL device is reachable.

use claspr::Context;
use collatz_example::Kernels;

/// Well-known Collatz sequence lengths (1-indexed input → length to
/// reach 1). OEIS A006577.
const CHECKS: &[(u32, u32)] = &[(1, 0), (2, 1), (3, 7), (4, 2), (27, 111)];

#[test]
fn collatz_via_generated_module() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP collatz_via_generated_module: no OpenCL device ({e})");
            return;
        }
    };

    let kernels = Kernels::load(&ctx).expect("Kernels::load");

    let n: usize = 1024;
    let mut data: Vec<u32> = (1..=n as u32).collect();

    let buf = ctx.upload(&data).expect("upload");
    ctx.launch(&kernels.collatz_kernel, [n], (&buf,))
        .expect("launch");
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
