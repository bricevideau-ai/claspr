fn main() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    // No `.with_f64()` here — the f64 stamp adds the `Float64`
    // capability automatically, and the f32 stamp must build WITHOUT
    // permission to use f64 so accidental widening is a build error.
    claspr_build::compile_from_host(&src)
        .opencl12()
        .write()
        .expect("compile claspr-test-instantiate from src/lib.rs");
}
