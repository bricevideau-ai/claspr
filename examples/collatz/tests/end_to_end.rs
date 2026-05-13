//! End-to-end smoke test driven by the build-time generated module
//! plus a `#[claspr::kernel]` stub.
//!
//! What's new vs. the stage-2 form:
//!
//! - The build script no longer declares the kernel signature with
//!   `.kernel(...)` — it just compiles the kernel crate.
//! - A body-less stub function in this test file, marked with
//!   `#[claspr::kernel]`, mirrors the kernel-crate signature exactly
//!   (kernel-style `&mut [u32]`, builtin params with `#[spirv(...)]`).
//!   The proc-macro turns it into a host launch wrapper that takes
//!   `&claspr::Context`, `&claspr::Kernel`, an `impl IntoLaunchSpec`,
//!   and the kernel-supplied buffers as `&claspr::DeviceSlice<T>`.
//!
//! Skips silently if no OpenCL device is reachable.

use claspr::Context;
use collatz_example::Kernels;

/// Stub mirrors the collatz kernel signature. The body is discarded
/// by the proc-macro; we leave it empty here to match the
/// "documentation-only" style stage-3-v1 settles on.
#[claspr::kernel]
fn collatz_kernel(
    #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
) {
}

/// Well-known Collatz sequence lengths (1-indexed input → length to
/// reach 1). OEIS A006577.
const CHECKS: &[(u32, u32)] = &[(1, 0), (2, 1), (3, 7), (4, 2), (27, 111)];

#[test]
fn collatz_via_proc_macro_stub() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP collatz_via_proc_macro_stub: no OpenCL device ({e})");
            return;
        }
    };

    let kernels = Kernels::load(&ctx).expect("Kernels::load");

    let n: usize = 1024;
    let mut data: Vec<u32> = (1..=n as u32).collect();

    let buf = ctx.upload(&data).expect("upload");
    collatz_kernel(&ctx, &kernels.collatz_kernel, [n], &buf).expect("launch collatz_kernel");
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
