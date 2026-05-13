//! End-to-end smoke test for the claspr runtime.
//!
//! Compiles the `collatz` kernel crate (sibling workspace member),
//! runs it through the full claspr API surface — `compile()` →
//! `Context::new()` → `kernel_from_spv` → `upload`/`launch`/`download`
//! — and checks a handful of well-known Collatz sequence lengths
//! against the kernel's output.
//!
//! Skips silently if no OpenCL device is reachable so the test passes
//! on machines without an OpenCL runtime (e.g. minimal CI images).

use claspr::{Context, compile};
use std::path::Path;

const KERNEL_NAME: &str = "collatz_kernel";

/// Well-known Collatz sequence lengths (1-indexed input, length of the
/// sequence to reach 1). Sourced from OEIS A006577.
const CHECKS: &[(u32, u32)] = &[(1, 0), (2, 1), (3, 7), (4, 2), (27, 111)];

#[test]
fn collatz_end_to_end() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP collatz_end_to_end: no OpenCL device ({e})");
            return;
        }
    };

    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/collatz");
    let module = compile(&kernel_crate)
        .opencl12()
        .build()
        .expect("compile collatz kernel");
    assert!(
        module.entry_points.iter().any(|e| e == KERNEL_NAME),
        "compiled module is missing the {KERNEL_NAME} entry point: {:?}",
        module.entry_points,
    );

    let kernel = ctx
        .kernel_from_spv(&module.spv_bytes, KERNEL_NAME)
        .expect("create kernel");

    // 1024 elements is plenty to cover every check input and keep the
    // test fast.
    let n: usize = 1024;
    let mut data: Vec<u32> = (1..=n as u32).collect();

    let buf = ctx.upload(&data).expect("upload");
    ctx.launch(&kernel, [n], (&buf,)).expect("launch");
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
