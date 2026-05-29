use std::path::Path;

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    claspr_build::compile_from_host(&src)
        .opencl12()
        // `Float64` capability so `fill_f64` / `scale_f64` survive
        // spirv-builder. Runtime tests skip if the device doesn't
        // advertise it.
        .with_f64()
        .write()
        .expect("compile claspr-test-kernels from src/lib.rs");
}
