//! End-to-end **single-file** claspr example using a `#[claspr::device]`
//! module.
//!
//! The whole device side — kernel entry point + helper function +
//! whatever future state shows up (constants, static configs, etc.) —
//! lives inside one `mod gpu { ... }` tagged with `#[claspr::device]`.
//! The build script lifts everything inside that module into the
//! generated kernel sub-crate; the host sees the module normally and
//! can call into it (`gpu::collatz(...)`) for validation against the
//! kernel output.
//!
//! Items at the top level of this file (use claspr::*, `mod compiled`,
//! `fn main`, `#[cfg(test)] mod tests`) stay host-only — `claspr-build`
//! drops anything that isn't a `#[claspr::kernel]` / `#[claspr::device]`
//! item.
//!
//! Run with `cargo run -p collatz-example`. The `#[test]` at the
//! bottom turns this into the project's smoke test.

use claspr::Context;

#[allow(dead_code)] // SPV_BYTES + ENTRY_POINTS are exposed but not used in this demo.
mod compiled {
    include!(concat!(env!("OUT_DIR"), "/collatz_kernels.rs"));
}

#[claspr::device]
mod gpu {
    // `spirv` is the proc-macro that recognises `#[spirv(kernel)]` and
    // `#[spirv(<builtin>)]` attributes on the kernel side. Cfg-gated
    // because the host crate doesn't have spirv-std as a dep — and
    // it doesn't need to: the host compilation never sees those
    // attributes (the `#[claspr::kernel]` proc-macro discards the
    // builtin params + replaces the function with its impl block
    // before name resolution touches them).
    #[cfg(target_arch = "spirv")]
    use spirv_std::spirv;

    /// Length of the Collatz sequence for `n` (1-indexed input → number
    /// of steps to reach 1), or `None` on overflow / zero input. Pure
    /// Rust — both the kernel body (per-element step) and the host
    /// validator below call into this.
    pub fn collatz(mut n: u32) -> Option<u32> {
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

    // Validate every kernel output against the host-side `collatz`
    // implementation lifted from inside the device module. Same
    // function, two callers.
    for (i, (&input, &device)) in inputs.iter().zip(&device_results).enumerate() {
        let host = gpu::collatz(input).unwrap_or(u32::MAX);
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
        let _ran = run().expect("run collatz");
    }
}
