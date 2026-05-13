use std::path::{Path, PathBuf};

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("collatz_kernels.rs");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels.rs");

    // Single-source mode: read kernel functions out of the host crate's
    // own source file. `claspr-build` translates #[claspr::kernel] to
    // #[spirv(kernel)], generates a kernel sub-crate under OUT_DIR, and
    // compiles via rust-gpu. Whatever entry points it finds become
    // public fields on the generated `Kernels` struct; the host launch
    // wrappers are emitted by the `#[claspr::kernel]` proc-macro on
    // the same source, so `.kernel(...)` declarations aren't needed
    // here any more.
    claspr_build::compile_from_host(&src)
        .opencl12()
        .write_to(&out_path)
        .expect("compile collatz kernel from host source");
}
