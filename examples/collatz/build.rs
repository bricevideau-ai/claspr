use std::path::{Path, PathBuf};

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("collatz_kernels.rs");
    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/collatz");

    claspr_build::compile(&kernel_crate)
        .opencl12()
        .kernel("collatz_kernel", &[("data", "&::claspr::DeviceSlice<u32>")])
        .write_to(&out_path)
        .expect("compile collatz kernel");
}
