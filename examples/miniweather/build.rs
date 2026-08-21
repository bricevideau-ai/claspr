fn main() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    // No `.with_f64()`: the f64 stamp adds `Float64` automatically and the
    // f32 stamp must build without fp64 permission (portability guard).
    claspr_build::compile_from_host(&src)
        .opencl12()
        .write()
        .expect("compile miniweather kernels from src/main.rs");
}
