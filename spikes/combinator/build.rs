use std::path::Path;

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    claspr_build::compile_from_host(&src)
        .opencl12()
        .write()
        .expect("compile combinator-spike kernels from host source");
}
