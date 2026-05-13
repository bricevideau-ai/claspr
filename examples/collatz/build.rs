use std::path::Path;

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");

    // Single-source mode: read kernel functions out of the host
    // crate's own source file. `claspr-build` writes one
    // `OUT_DIR/<modname>.rs` per `#[claspr::device]` module it finds
    // — the matching `#[claspr::device]` proc-macro on the host side
    // includes from the same path, so module name is the only
    // coupling.
    claspr_build::compile_from_host(&src)
        .opencl12()
        .write()
        .expect("compile collatz kernel from host source");
}
