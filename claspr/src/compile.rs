//! Thin builder around [`spirv_builder::SpirvBuilder`].
//!
//! Replaces the four `compile_kernel*` variants in
//! `rust-gpu-opencl-samples/runner/src/main.rs` with one builder + a
//! handful of named presets. The underlying `SpirvBuilder` is exposed
//! via [`CompileBuilder::with`] for cases this wrapper doesn't cover.

use crate::Result;
use spirv_builder::{Capability, CompileResult, ShaderPanicStrategy, SpirvBuilder};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// A compiled SPIR-V module ready to feed to [`Context::build_program`].
///
/// [`Context::build_program`]: crate::context::Context::build_program
#[derive(Debug)]
pub struct CompiledModule {
    /// Raw SPIR-V bytes (the contents of the file rust-gpu wrote).
    pub spv_bytes: Vec<u8>,
    /// Wall-clock time spent in [`SpirvBuilder::build`].
    pub compile_time: Duration,
    /// Path on disk to the SPIR-V binary (kept around for diagnostics).
    pub spv_path: PathBuf,
    /// Entry-point names exposed by the module.
    pub entry_points: Vec<String>,
}

/// Builder for compiling a kernel crate to OpenCL SPIR-V.
///
/// Most users get away with one of the named presets:
///
/// ```ignore
/// claspr::compile("kernels/collatz").opencl12().build()?;
/// claspr::compile("kernels/nbody").opencl12().with_f64().build()?;
/// claspr::compile("kernels/reduce").opencl20_groups().build()?;
/// claspr::compile("kernels/raymarch").image().build()?;
/// ```
///
/// For settings claspr doesn't expose (custom target dirs, multi-module
/// output, etc.), pass a closure to [`CompileBuilder::with`]:
///
/// ```ignore
/// claspr::compile("kernels/foo")
///     .opencl12()
///     .with(|sb| sb.print_metadata(spirv_builder::MetadataPrintout::Full))
///     .build()?;
/// ```
pub struct CompileBuilder {
    inner: SpirvBuilder,
    crate_path: PathBuf,
}

/// Start a [`CompileBuilder`] for the kernel crate at `path`.
///
/// Defaults to the `spirv-unknown-opencl1.2` target with no extra
/// capabilities or panic strategy. Chain a preset (e.g. `opencl12()`)
/// to apply the matching defaults.
pub fn compile(path: impl AsRef<Path>) -> CompileBuilder {
    CompileBuilder::new(path)
}

impl CompileBuilder {
    /// Equivalent to the free function [`compile`].
    pub fn new(path: impl AsRef<Path>) -> Self {
        let crate_path = path.as_ref().to_path_buf();
        Self {
            inner: SpirvBuilder::new(&crate_path, "spirv-unknown-opencl1.2"),
            crate_path,
        }
    }

    /// Set the SPIR-V target environment string passed to rust-gpu
    /// (e.g. `"spirv-unknown-opencl2.0"`).
    ///
    /// Switching target rebuilds the underlying [`SpirvBuilder`], so
    /// call this **before** [`capability`], [`panic_strategy`], or
    /// [`with`] — settings applied earlier are dropped.
    ///
    /// [`capability`]: Self::capability
    /// [`panic_strategy`]: Self::panic_strategy
    /// [`with`]: Self::with
    pub fn target_env(mut self, target: impl Into<String>) -> Self {
        self.inner = SpirvBuilder::new(&self.crate_path, target);
        self
    }

    /// Add a SPIR-V capability the kernel needs (e.g. `Capability::Float64`).
    pub fn capability(mut self, cap: Capability) -> Self {
        self.inner = self.inner.capability(cap);
        self
    }

    /// Set the panic strategy used by SPIR-T to lower `panic!`/`abort`.
    pub fn panic_strategy(mut self, strategy: ShaderPanicStrategy) -> Self {
        self.inner = self.inner.shader_panic_strategy(strategy);
        self
    }

    /// Escape hatch — apply an arbitrary closure to the underlying
    /// [`SpirvBuilder`] for settings claspr doesn't wrap. The closure
    /// must return the modified `SpirvBuilder`.
    pub fn with(mut self, f: impl FnOnce(SpirvBuilder) -> SpirvBuilder) -> Self {
        self.inner = f(self.inner);
        self
    }

    /// Preset for OpenCL 1.2 compute kernels with `panic!` lowered to
    /// `printf` + early return (visible on stock OpenCL runtimes).
    ///
    /// Equivalent to:
    /// ```ignore
    /// .target_env("spirv-unknown-opencl1.2")
    /// .panic_strategy(ShaderPanicStrategy::DebugPrintfThenExit {
    ///     print_inputs: true,
    ///     print_backtrace: true,
    /// })
    /// ```
    pub fn opencl12(self) -> Self {
        self.target_env("spirv-unknown-opencl1.2").panic_strategy(
            ShaderPanicStrategy::DebugPrintfThenExit {
                print_inputs: true,
                print_backtrace: true,
            },
        )
    }

    /// Preset for OpenCL 2.0 kernels that use the `Groups` capability
    /// (subgroup ops, work-group collectives) — typically reduction-
    /// style kernels with workgroup barriers.
    ///
    /// Uses [`ShaderPanicStrategy::UNSOUND_DO_NOT_USE_UndefinedBehaviorViaUnreachable`]
    /// because barrier-using kernels can deadlock if the
    /// `DebugPrintfThenExit` strategy diverges work items at a barrier
    /// (PoCL #2156).
    pub fn opencl20_groups(self) -> Self {
        self.target_env("spirv-unknown-opencl2.0")
            .capability(Capability::Groups)
            .panic_strategy(ShaderPanicStrategy::UNSOUND_DO_NOT_USE_UndefinedBehaviorViaUnreachable)
    }

    /// Preset for image kernels — OpenCL 1.2 target, no panic strategy.
    ///
    /// Image kernels typically don't need panic diagnostics (they
    /// don't bounds-check) and the codegen auto-adds `ImageBasic` when
    /// it sees an `Image!` parameter.
    pub fn image(self) -> Self {
        self.target_env("spirv-unknown-opencl1.2")
    }

    /// Convenience shortcut: add the `Float64` capability for kernels
    /// that use `f64`.
    pub fn with_f64(self) -> Self {
        self.capability(Capability::Float64)
    }

    /// Run rust-gpu and return the compiled module along with the build
    /// duration. Errors propagate as the boxed claspr error type.
    pub fn build(self) -> Result<CompiledModule> {
        let start = Instant::now();
        let result: CompileResult = self.inner.build()?;
        let compile_time = start.elapsed();
        let spv_path = result.module.unwrap_single().to_path_buf();
        let spv_bytes = std::fs::read(&spv_path)?;
        Ok(CompiledModule {
            spv_bytes,
            compile_time,
            spv_path,
            entry_points: result.entry_points,
        })
    }
}
