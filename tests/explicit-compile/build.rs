//! Build script for the `claspr-explicit-compile-test` crate.
//!
//! Compiles the kernel sub-crate at `kernel/` to SPIR-V via the
//! explicit `claspr_build::compile(...).write_to(...)` path — the
//! path that *doesn't* go through `compile_from_host` (which is
//! what the rest of claspr's tests exercise). The generated file
//! at `OUT_DIR/kernels.rs` exposes only `SPV_BYTES` and
//! `ENTRY_POINTS`; the host-side `Kernels` surface is declared
//! separately by the `claspr::kernels!` macro in `src/lib.rs`.

use std::path::{Path, PathBuf};

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let kernel_crate = manifest.join("kernel");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is always set for build scripts");
    let out_path = PathBuf::from(out_dir).join("kernels.rs");

    claspr_build::compile(&kernel_crate)
        .opencl12()
        .write_to(&out_path)
        .expect("compile explicit-compile-test kernels to SPIR-V");

    // Second compile of the SAME kernel crate with a different feature
    // set. spirv-builder reuses one build location per kernel crate, so
    // this overwrites the first build's `.spv` — the generated files must
    // each embed a frozen copy or both would silently alias to this
    // (last-written) variant. `alias_regression` in src/lib.rs asserts
    // the two blobs differ.
    let alt_out_path = out_path.with_file_name("kernels_alt.rs");
    claspr_build::compile(&kernel_crate)
        .opencl12()
        .with(|sb| sb.shader_crate_features(["alt".to_string()]))
        .write_to(&alt_out_path)
        .expect("compile explicit-compile-test kernels (alt feature) to SPIR-V");
}
