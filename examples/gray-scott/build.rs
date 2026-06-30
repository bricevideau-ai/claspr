use std::path::Path;

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");

    // Single-source mode: read the `#[claspr::device] mod gpu` kernel out of
    // this crate's own `src/main.rs`. `claspr-build` lifts the device module
    // into a generated sub-crate, compiles it with spirv-builder, and writes
    // `OUT_DIR/gpu.rs`; the matching `#[claspr::device]` macro on the host
    // side `include!`s that exact path. Identical shape to collatz/build.rs.
    claspr_build::compile_from_host(&src)
        .opencl12()
        .write()
        .expect("compile gray-scott kernel from host source");
}
