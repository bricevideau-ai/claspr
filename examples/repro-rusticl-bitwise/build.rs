use claspr_build::ShaderPanicStrategy;
use std::path::Path;

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    // Use the unreachable strategy instead of DebugPrintfThenExit
    // (the default that `.opencl12()` would set). With UB-via-
    // unreachable, `panic!` lowers to OpUnreachable and the printf
    // abort scaffolding is gone — strips a lot of SPIR-V from each
    // kernel and lets us see whether the rusticl crash and the
    // separate spirv-opt-perf-pass crash are both downstream of the
    // panic infrastructure.
    claspr_build::compile_from_host(&src)
        .target_env("spirv-unknown-opencl1.2")
        .panic_strategy(ShaderPanicStrategy::UNSOUND_DO_NOT_USE_UndefinedBehaviorViaUnreachable)
        .write()
        .expect("compile bitwise reproducer kernels");
}
