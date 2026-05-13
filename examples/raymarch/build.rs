use std::path::{Path, PathBuf};

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    // Convention: write to `OUT_DIR/kernels.rs`, matching the
    // include synthesised by `#[claspr::device]`.
    let out_path = PathBuf::from(out_dir).join("kernels.rs");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");

    claspr_build::compile_from_host(&src)
        .image()
        .write_to(&out_path)
        .expect("compile raymarch kernel from host source");
}
