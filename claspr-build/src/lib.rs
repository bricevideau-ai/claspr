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
//!         .kernel("collatz_kernel", &[("data", "&::claspr::DeviceSlice<u32>")])
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
//! kernels.collatz_kernel(&ctx, [n], &buf)?;
//! ```
//!
//! ## Typed launch wrappers
//!
//! For each kernel the build script declares via [`CompileBuilder::kernel`],
//! the generated module emits a typed launch method on `Kernels` so the
//! call site reads:
//!
//! ```ignore
//! kernels.collatz_kernel(&ctx, [n], &buf)?;
//! ```
//!
//! instead of the raw [`claspr::Context::launch`] form. We don't reflect
//! the SPIR-V to discover signatures because the long-term plan
//! (stage 3 proc-macro) will know them from the kernel function
//! definition — explicit declaration here is the same shape the
//! proc-macro will emit.
//!
//! [`claspr::Context::launch`]: https://docs.rs/claspr
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
    kernels: Vec<KernelDecl>,
}

/// A declared kernel entry point + its host-side launch signature.
///
/// Built up via [`CompileBuilder::kernel`] — see that method for usage.
struct KernelDecl {
    name: String,
    /// `(arg_name, arg_type)` pairs, in source declaration order. Both
    /// fields are spliced verbatim into the generated wrapper, so they
    /// must be valid Rust syntax in the position they're written
    /// (paths can be absolute, e.g. `&::claspr::DeviceSlice<u32>`).
    params: Vec<(String, String)>,
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
            kernels: Vec::new(),
        }
    }

    /// Declare a kernel entry point and its host-side launch signature.
    ///
    /// `name` must match the SPIR-V entry-point name (cross-checked
    /// against `spirv-builder`'s reported entry points at [`write_to`]
    /// time — a typo will fail the build with a clear error). `params`
    /// is a list of `(arg_name, arg_type)` pairs as written in Rust
    /// source.
    ///
    /// ```ignore
    /// claspr_build::compile("kernels/collatz")
    ///     .opencl12()
    ///     .kernel("collatz_kernel", &[("data", "&::claspr::DeviceSlice<u32>")])
    ///     .write_to(&out_path)?;
    /// ```
    ///
    /// Both `arg_name` and `arg_type` are spliced into the generated
    /// wrapper verbatim — `arg_name` becomes a parameter name and a
    /// launch-tuple element, `arg_type` becomes the parameter type.
    /// Use absolute paths (`::claspr::...`) so the wrapper compiles
    /// regardless of what's in scope at the include site.
    ///
    /// Multiple `.kernel(...)` calls are supported for modules with
    /// more than one entry point. Entry points compiled by
    /// `spirv-builder` but **not** declared here still appear as
    /// [`claspr::Kernel`] fields on the `Kernels` struct, just without
    /// a typed launch method — call sites can still launch them via
    /// [`claspr::Context::launch`] directly.
    ///
    /// [`write_to`]: Self::write_to
    /// [`claspr::Kernel`]: https://docs.rs/claspr
    /// [`claspr::Context::launch`]: https://docs.rs/claspr
    pub fn kernel(mut self, name: &str, params: &[(&str, &str)]) -> Self {
        self.kernels.push(KernelDecl {
            name: name.to_string(),
            params: params
                .iter()
                .map(|(n, t)| ((*n).to_string(), (*t).to_string()))
                .collect(),
        });
        self
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
    ///   entry point, constructed via `Kernels::load(&ctx)`. Entry
    ///   points declared via [`kernel`] also get a typed launch method
    ///   on the struct.
    ///
    /// Also emits `cargo:rerun-if-changed=` lines for the kernel
    /// crate's `Cargo.toml` and `src/` directory so cargo recompiles
    /// when the kernel changes. Call this from a `build.rs`.
    ///
    /// Errors if any [`kernel`] declaration names an entry point that
    /// isn't present in the compiled SPIR-V module — typo detection.
    ///
    /// [`kernel`]: Self::kernel
    /// [`claspr::Kernel`]: https://docs.rs/claspr
    pub fn write_to(self, out_path: impl AsRef<Path>) -> Result<()> {
        let out_path = out_path.as_ref();

        // cargo:rerun-if-changed for the kernel crate so build script
        // re-runs when the kernel source changes.
        emit_rerun_if_changed(&self.crate_path);

        let crate_path = self.crate_path.clone();
        let kernels = self.kernels;
        let result: CompileResult = self.inner.build()?;
        let spv_path = result.module.unwrap_single().to_path_buf();

        // Cross-check: every declared kernel must exist in the module.
        for decl in &kernels {
            if !result.entry_points.iter().any(|ep| ep == &decl.name) {
                return Err(format!(
                    "kernel declaration {:?} does not match any entry point in the compiled module \
                     (entry points: {:?})",
                    decl.name, result.entry_points,
                )
                .into());
            }
        }

        let generated =
            generate_module_source(&spv_path, &result.entry_points, &kernels, &crate_path)?;

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
    kernels: &[KernelDecl],
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
        // Field is private when the kernel has a typed launch method
        // (avoids `kernels.foo` field vs `kernels.foo(...)` confusion);
        // public otherwise so callers can still launch via
        // `ctx.launch(&kernels.foo, ...)`.
        let vis = if has_decl(kernels, ep) { "" } else { "pub " };
        writeln!(s, "    {vis}{field}: ::claspr::Kernel,")?;
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

    // Typed launch methods, one per declared kernel.
    for decl in kernels {
        let field = sanitize_field_name(&decl.name);
        let params_sig: String = decl
            .params
            .iter()
            .map(|(n, t)| format!(",\n        {n}: {t}"))
            .collect();
        let tuple_args: String = decl
            .params
            .iter()
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>()
            .join(", ");
        // Trailing comma needed for single-element tuples.
        let tuple_lit = if decl.params.len() == 1 {
            format!("({tuple_args},)")
        } else {
            format!("({tuple_args})")
        };

        writeln!(s)?;
        writeln!(
            s,
            "    /// Launch the `{}` kernel with typed arguments.",
            decl.name
        )?;
        writeln!(s, "    pub fn {field}(")?;
        writeln!(s, "        &self,")?;
        writeln!(s, "        ctx: &::claspr::Context,")?;
        writeln!(
            s,
            "        grid: impl ::claspr::IntoLaunchSpec{params_sig},"
        )?;
        writeln!(s, "    ) -> ::claspr::Result<::claspr::Event> {{")?;
        writeln!(s, "        ctx.launch(&self.{field}, grid, {tuple_lit})")?;
        writeln!(s, "    }}")?;
    }

    writeln!(s, "}}")?;

    Ok(s)
}

fn has_decl(kernels: &[KernelDecl], name: &str) -> bool {
    kernels.iter().any(|k| k.name == name)
}

/// Convert an OpenCL entry-point name into a valid Rust field
/// identifier. Today this is a passthrough — entry points emitted by
/// rust-gpu are already valid Rust identifiers — but reserved here so
/// future renames (mangled C++-style names, leading-digit guards) can
/// be slotted in without changing the call sites.
fn sanitize_field_name(name: &str) -> String {
    name.to_string()
}
