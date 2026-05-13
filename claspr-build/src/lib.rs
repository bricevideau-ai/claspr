//! Build-script helper for [claspr].
//!
//! Compiles a rust-gpu kernel crate to OpenCL SPIR-V at build time
//! (via `spirv-builder`) and emits a generated Rust module that
//! embeds the SPIR-V bytes and pre-built kernels into a typed
//! `Kernels` struct.
//!
//! Use from a downstream crate's `build.rs`:
//!
//! ```ignore
//! // build.rs
//! fn main() {
//!     let out_dir = std::env::var("OUT_DIR").unwrap();
//!     let out_path = std::path::PathBuf::from(out_dir).join("collatz_kernels.rs");
//!     let kernel_crate = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
//!         .join("../../kernels/collatz");
//!     claspr_build::compile(&kernel_crate)
//!         .opencl12()
//!         .write_to(&out_path)
//!         .unwrap();
//! }
//! ```
//!
//! Then in your library/binary:
//!
//! ```ignore
//! mod kernels {
//!     include!(concat!(env!("OUT_DIR"), "/collatz_kernels.rs"));
//! }
//!
//! let ctx = claspr::Context::new()?;
//! let kernels = kernels::Kernels::load(&ctx)?;
//! ctx.launch(&kernels.collatz_kernel, [n], (&buf,))?;
//! ```
//!
//! ## Status
//!
//! This is the **stage 2 sketch** — only SPIR-V bytes + kernel objects
//! are generated; per-kernel typed launch wrappers (the headline
//! feature of stage 2) come in a follow-up commit, after we have
//! collatz running through this build-time path so we can iterate on
//! reflection on real SPIR-V.
//!
//! [claspr]: https://github.com/bricevideau-ai/claspr

use spirv_builder::{Capability, CompileResult, ShaderPanicStrategy, SpirvBuilder};
use std::error::Error;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Boxed-error result alias used by all [`claspr_build`] entry points.
pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync + 'static>>;

/// Builder for compiling a kernel crate at build time and emitting
/// generated Rust source.
///
/// The interface mirrors `claspr::compile` — the methods (and named
/// presets `opencl12` / `opencl20_groups` / `image` / `with_f64`) carry
/// the same meaning. The terminal call is [`write_to`] rather than
/// `build`, since the build-script use case wants a file written into
/// `OUT_DIR`, not bytes returned in memory.
///
/// [`write_to`]: Self::write_to
pub struct CompileBuilder {
    inner: SpirvBuilder,
    crate_path: PathBuf,
}

/// Start a [`CompileBuilder`] for the kernel crate at `path`.
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
    /// call this **before** `capability` / `panic_strategy` / `with`.
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

    /// Escape hatch for settings claspr-build doesn't wrap.
    pub fn with(mut self, f: impl FnOnce(SpirvBuilder) -> SpirvBuilder) -> Self {
        self.inner = f(self.inner);
        self
    }

    /// Preset — OpenCL 1.2 with `panic!` lowered to printf-then-exit.
    pub fn opencl12(self) -> Self {
        self.target_env("spirv-unknown-opencl1.2").panic_strategy(
            ShaderPanicStrategy::DebugPrintfThenExit {
                print_inputs: true,
                print_backtrace: true,
            },
        )
    }

    /// Preset — OpenCL 2.0 + `Groups` capability for subgroup / workgroup
    /// collective kernels with barriers (uses the UB-via-unreachable
    /// panic strategy to avoid divergence at barriers).
    pub fn opencl20_groups(self) -> Self {
        self.target_env("spirv-unknown-opencl2.0")
            .capability(Capability::Groups)
            .panic_strategy(ShaderPanicStrategy::UNSOUND_DO_NOT_USE_UndefinedBehaviorViaUnreachable)
    }

    /// Preset — image kernels: OpenCL 1.2 target, no panic strategy.
    pub fn image(self) -> Self {
        self.target_env("spirv-unknown-opencl1.2")
    }

    /// Convenience — add the `Float64` capability.
    pub fn with_f64(self) -> Self {
        self.capability(Capability::Float64)
    }

    /// Compile the kernel crate, then write a generated Rust source
    /// file to `out_path` containing:
    ///
    /// - `SPV_BYTES: &[u8]` — the SPIR-V module, embedded via `include_bytes!`
    /// - `ENTRY_POINTS: &[&str]` — the entry-point names rust-gpu reported
    /// - `Kernels { ... }` — a struct holding one [`claspr::Kernel`] per
    ///   entry point, constructed via `Kernels::load(&ctx)`
    ///
    /// Also emits `cargo:rerun-if-changed=` lines for the kernel
    /// crate's `Cargo.toml` and `src/` directory so cargo recompiles
    /// when the kernel changes. Call this from a `build.rs`.
    ///
    /// [`claspr::Kernel`]: https://docs.rs/claspr
    pub fn write_to(self, out_path: impl AsRef<Path>) -> Result<()> {
        let out_path = out_path.as_ref();

        // cargo:rerun-if-changed for the kernel crate so build script
        // re-runs when the kernel source changes.
        emit_rerun_if_changed(&self.crate_path);

        let crate_path = self.crate_path.clone();
        let result: CompileResult = self.inner.build()?;
        let spv_path = result.module.unwrap_single().to_path_buf();

        let generated = generate_module_source(&spv_path, &result.entry_points, &crate_path)?;

        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(out_path)?;
        file.write_all(generated.as_bytes())?;
        Ok(())
    }
}

fn emit_rerun_if_changed(crate_path: &Path) {
    println!("cargo:rerun-if-changed={}/Cargo.toml", crate_path.display());
    println!("cargo:rerun-if-changed={}/src", crate_path.display());
}

fn generate_module_source(
    spv_path: &Path,
    entry_points: &[String],
    crate_path: &Path,
) -> Result<String> {
    let mut s = String::new();
    writeln!(
        s,
        "// Generated by claspr-build for {}.\n// Do not edit by hand.\n",
        crate_path.display()
    )?;
    writeln!(
        s,
        "/// Raw SPIR-V bytes for the kernel module, embedded at build time."
    )?;
    writeln!(
        s,
        "pub const SPV_BYTES: &[u8] = include_bytes!({:?});\n",
        spv_path
    )?;
    writeln!(s, "/// Entry-point names exposed by this module.")?;
    writeln!(s, "pub const ENTRY_POINTS: &[&str] = &[")?;
    for ep in entry_points {
        writeln!(s, "    {ep:?},")?;
    }
    writeln!(s, "];\n")?;

    writeln!(
        s,
        "/// All kernels in this module, pre-built once at startup and held"
    )?;
    writeln!(s, "/// for repeated launch.")?;
    writeln!(s, "pub struct Kernels {{")?;
    for ep in entry_points {
        let field = sanitize_field_name(ep);
        writeln!(s, "    pub {field}: ::claspr::Kernel,")?;
    }
    writeln!(s, "}}\n")?;

    writeln!(s, "impl Kernels {{")?;
    writeln!(
        s,
        "    /// Build the program from the embedded SPIR-V and look up every entry point."
    )?;
    writeln!(
        s,
        "    pub fn load(ctx: &::claspr::Context) -> ::claspr::Result<Self> {{"
    )?;
    writeln!(s, "        let program = ctx.build_program(SPV_BYTES)?;")?;
    writeln!(s, "        Ok(Self {{")?;
    for ep in entry_points {
        let field = sanitize_field_name(ep);
        writeln!(s, "            {field}: ctx.kernel(&program, {ep:?})?,")?;
    }
    writeln!(s, "        }})")?;
    writeln!(s, "    }}")?;
    writeln!(s, "}}")?;

    Ok(s)
}

/// Convert an OpenCL entry-point name into a valid Rust field
/// identifier. Today this is a passthrough — entry points emitted by
/// rust-gpu are already valid Rust identifiers — but reserved here so
/// future renames (mangled C++-style names, leading-digit guards) can
/// be slotted in without changing the call sites.
fn sanitize_field_name(name: &str) -> String {
    name.to_string()
}
