use std::path::Path;

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    claspr_build::compile_from_host(&src)
        .image()
        .write()
        .expect("compile claspr-test-image-kernels from src/lib.rs");
}
