//! Compiles the example's kernel sub-crate to SPIR-V so the main
//! binary has a self-contained `SPV_BYTES` to introspect without
//! the user having to supply one. Mirrors `tests/explicit-compile/`'s
//! shape.

use std::path::{Path, PathBuf};

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let kernel_crate = manifest.join("kernel");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is always set for build scripts");
    let out_path = PathBuf::from(out_dir).join("kernels.rs");

    // `claspr-build` defaults to `SpirvMetadata::NameVariables` —
    // emits `OpName` for kernel-arg interface variables so the
    // names round-trip through `clGetKernelArgInfo`. No explicit
    // opt-in needed; this example used to call
    // `.spirv_metadata(NameVariables)` back when the default was
    // `None`, but since the default flipped (kept here as a
    // historical note) the bare `compile().opencl12()` produces
    // named SPIR-V automatically.
    claspr_build::compile(&kernel_crate)
        .opencl12()
        .write_to(&out_path)
        .expect("compile spv-introspect-example demo kernels");
}
