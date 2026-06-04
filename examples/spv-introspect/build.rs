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

    // Ask spirv-builder to emit `OpName` instructions in the
    // SPIR-V so `clGetKernelArgInfo`'s name field has something
    // to recover. spirv-builder's default `SpirvMetadata::None`
    // strips all names — which is why every ICD reports `<empty>`
    // until you opt in. `Full` adds `OpLine` debug info too
    // (bigger binary); `NameVariables` is the minimal opt-in that
    // gives us OpNames for kernel-arg interface variables.
    claspr_build::compile(&kernel_crate)
        .opencl12()
        .with(|sb| sb.spirv_metadata(claspr_build::SpirvMetadata::NameVariables))
        .write_to(&out_path)
        .expect("compile spv-introspect-example demo kernels");
}
