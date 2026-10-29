fn main() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    claspr_build::compile_from_host(&src)
        .opencl12()
        .write()
        .expect("compile llama2 kernels from src/main.rs");
}
