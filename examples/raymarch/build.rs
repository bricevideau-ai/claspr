use std::path::{Path, PathBuf};

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("raymarch_kernels.rs");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");

    claspr_build::compile_from_host(&src)
        .image()
        .write_to(&out_path)
        .expect("compile raymarch kernel from host source");
}
