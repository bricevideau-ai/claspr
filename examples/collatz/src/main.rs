//! End-to-end **single-file** claspr example.
//!
//! The kernel function, the device-side helper it calls, the
//! `mod compiled` import of the build-script-generated SPIR-V, the
//! host launch, and the test all live in this one file. The same
//! source serves both:
//!
//! - the **host** compilation, which expands `#[claspr::kernel]` into
//!   a typed launch method on `compiled::Kernels`, treats
//!   `#[claspr::device]` as a no-op marker, and ignores everything
//!   else as ordinary host code; and
//! - the **kernel** compilation, which is driven by `build.rs`'s
//!   `claspr_build::compile_from_host("src/main.rs")` call —
//!   `claspr-build` keeps only items marked `#[claspr::kernel]` or
//!   `#[claspr::device]`, translates `#[claspr::kernel]` to
//!   `#[spirv(kernel)]`, and strips the `#[claspr::device]` marker.
//!   `use claspr::*`, `mod compiled`, `fn main`, etc. never reach
//!   the kernel crate.
//!
//! Run with `cargo run -p collatz-example`. The `#[test]` at the
//! bottom turns this into the project's smoke test.

use claspr::Context;

#[allow(dead_code)] // SPV_BYTES + ENTRY_POINTS are exposed but not used in this demo.
mod compiled {
    include!(concat!(env!("OUT_DIR"), "/collatz_kernels.rs"));
}

/// Length of the Collatz sequence for `n` (1-indexed input → number
/// of steps to reach 1), or `None` on overflow / zero input.
///
/// Marked `#[claspr::device]` so `claspr-build` pulls this into the
/// kernel sub-crate alongside the entry point that calls it. The
/// marker doesn't restrict host-side use — `run` below uses the same
/// function to validate the kernel's output against an
/// independently-computed reference, which is the stronger
/// "single-source" claim: one definition serves both the device
/// computation and the host validator.
#[claspr::device]
fn collatz(mut n: u32) -> Option<u32> {
    let mut i = 0;
    if n == 0 {
        return None;
    }
    while n != 1 {
        n = if n.is_multiple_of(2) {
            n / 2
        } else {
            if n >= 0x5555_5555 {
                return None;
            }
            3 * n + 1
        };
        i += 1;
    }
    Some(i)
}

#[claspr::kernel(kernels = crate::compiled::Kernels)]
pub fn collatz_kernel(
    #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
) {
    let index = _id.x;
    data[index] = collatz(data[index]).unwrap_or(u32::MAX);
}

const N: usize = 1024;

fn run() -> claspr::Result<bool> {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return Ok(false);
        }
    };

    let kernels = compiled::Kernels::load(&ctx)?;

    let inputs: Vec<u32> = (1..=N as u32).collect();
    let mut device_results = inputs.clone();
    let buf = ctx.upload(&device_results)?;
    kernels.collatz_kernel(&ctx, [N], &buf)?;
    ctx.download(&buf, &mut device_results)?;

    // Validate the kernel's output element-by-element against the
    // host-side `collatz` implementation. Same function, two callers.
    for (i, (&input, &device)) in inputs.iter().zip(&device_results).enumerate() {
        let host = collatz(input).unwrap_or(u32::MAX);
        assert_eq!(
            device, host,
            "device/host mismatch at index {i} (input {input}): device={device}, host={host}",
        );
    }
    Ok(true)
}

fn main() -> claspr::Result<()> {
    if run()? {
        println!("collatz: device/host agreement on {N} elements");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn collatz_single_source() {
        // `run` returns false (without panicking) when there's no
        // OpenCL device, which we want the test to treat as a skip
        // rather than a failure. Errors during the run itself unwrap
        // and fail the test loudly.
        let _ran = run().expect("run collatz");
    }
}
