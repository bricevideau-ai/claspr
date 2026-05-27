//! Build-script helper for [claspr].
//!
//! Compiles a rust-gpu kernel crate to OpenCL SPIR-V at build time
//! (via `spirv-builder`) and emits a generated Rust module that
//! embeds the SPIR-V bytes and pre-built kernels into a typed
//! `Kernels` struct.
//!
//! Two entry points, picked by where the kernel source lives:
//!
//! - [`compile_from_host`] — single-source mode. Kernel functions
//!   live alongside host code in the host crate's own source file,
//!   wrapped in `#[claspr::device] mod <name> { ... }`. The build
//!   script extracts each device module into a generated kernel
//!   sub-crate, compiles via rust-gpu, and writes
//!   `OUT_DIR/<name>.rs`. The matching `#[claspr::device]` proc-macro
//!   on the host side `include!()`s that file. This is what both
//!   in-tree examples (collatz, raymarch) use.
//! - [`compile`] — explicit mode. The kernel lives in a separate
//!   crate; the build script names entry points and their typed
//!   launch signatures by hand. Useful when the kernel sources are
//!   shared between projects or maintained independently of any
//!   single host crate.
//!
//! ## Single-source: `compile_from_host`
//!
//! ```ignore
//! // build.rs
//! fn main() {
//!     let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
//!     claspr_build::compile_from_host(&src).opencl12().write().unwrap();
//! }
//! ```
//!
//! ```ignore
//! // src/main.rs
//! use claspr::Context;
//!
//! #[claspr::device]
//! mod gpu {
//!     #[cfg(target_arch = "spirv")]
//!     use spirv_std::spirv;
//!
//!     #[claspr::kernel]
//!     pub fn collatz_kernel(
//!         #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
//!         #[spirv(cross_workgroup)] data: &mut [u32],
//!     ) { /* ... */ }
//! }
//!
//! fn main() -> claspr::Result<()> {
//!     let ctx = Context::new()?;
//!     let kernels = gpu::kernels(&ctx)?;
//!     // kernels.collatz_kernel(&ctx, [n], &buf).wait()?;
//!     Ok(())
//! }
//! ```
//!
//! ## Explicit: `compile` + manual kernel declarations
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
//! ```ignore
//! mod kernels {
//!     include!(concat!(env!("OUT_DIR"), "/collatz_kernels.rs"));
//! }
//!
//! let ctx = claspr::Context::new()?;
//! let kernels = kernels::Kernels::load(&ctx)?;
//! kernels.collatz_kernel(&ctx, [n], &buf).wait()?;
//! ```
//!
//! ## Typed launch wrappers
//!
//! Both flows produce a `Kernels` struct with one field per entry
//! point and a `Kernels::load(&ctx)` constructor. Single-source mode
//! gets the typed launch methods from the [`#[claspr::kernel]`][kernel-macro]
//! proc-macro (which sees the kernel's signature directly). Explicit
//! mode gets them from each [`CompileBuilder::kernel`] call's
//! `(name, type)` pairs. Either way the call site reads:
//!
//! ```ignore
//! kernels.collatz_kernel(&ctx, [n], &buf).wait()?;
//! ```
//!
//! and the returned [`LaunchOp`] supports `.wait()`, `.submit()`,
//! `.await`, plus `.profiled(...)` / `.after(...)` modifiers.
//!
//! [`LaunchOp`]: https://docs.rs/claspr
//! [kernel-macro]: https://docs.rs/claspr_macros
//! [claspr]: https://github.com/bricevideau-ai/claspr

use quote::ToTokens;
use spirv_builder::{CompileResult, SpirvBuilder};
use std::error::Error;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

// Re-exported so build scripts that need to pass extra
// capabilities / a non-default panic strategy don't have to add
// spirv-builder as a separate build-dependency.
pub use spirv_builder::{Capability, ShaderPanicStrategy};

/// Boxed-error result alias used by all [`claspr_build`] entry points.
pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync + 'static>>;

/// Deferred-construction settings for [`SpirvBuilder`].
///
/// Both [`CompileBuilder`] and [`HostBuilder`] embed one and apply it
/// to a fresh [`SpirvBuilder`] at terminal-call time. Call order
/// doesn't matter — `target_env` no longer rebuilds anything mid-chain.
#[derive(Default)]
struct SpirvBuilderSettings {
    target_env: String,
    capabilities: Vec<Capability>,
    panic_strategy: Option<ShaderPanicStrategy>,
    customizers: Vec<Box<dyn Fn(SpirvBuilder) -> SpirvBuilder>>,
}

impl SpirvBuilderSettings {
    fn new() -> Self {
        Self {
            target_env: "spirv-unknown-opencl1.2".to_string(),
            ..Self::default()
        }
    }

    /// Build a fresh [`SpirvBuilder`] for `crate_path` with these
    /// settings + customizers applied.
    ///
    /// Takes `&self` so [`HostBuilder::write`] can call this once per
    /// device module without consuming the settings.
    fn apply_to(&self, crate_path: &Path) -> SpirvBuilder {
        let mut sb = SpirvBuilder::new(crate_path, &self.target_env);
        for cap in &self.capabilities {
            sb = sb.capability(*cap);
        }
        if let Some(ps) = self.panic_strategy {
            sb = sb.shader_panic_strategy(ps);
        }
        for f in &self.customizers {
            sb = f(sb);
        }
        sb
    }
}

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
    settings: SpirvBuilderSettings,
    crate_path: PathBuf,
    kernels: Vec<KernelDecl>,
}

/// A declared kernel entry point + its host-side launch signature.
///
/// Built up via [`CompileBuilder::kernel`] — see that method for usage.
/// The `params` are currently unused by `emit_kernels_file` (typed
/// launch methods are emitted by the `#[claspr::kernel]` proc-macro,
/// not the build script), but kept on the struct so the
/// `.kernel(name, params)` builder method remains stable for callers.
#[allow(dead_code)]
struct KernelDecl {
    name: String,
    params: Vec<(String, String)>,
}

/// Start a [`CompileBuilder`] for the kernel crate at `path`.
pub fn compile(path: impl AsRef<Path>) -> CompileBuilder {
    CompileBuilder::new(path)
}

impl CompileBuilder {
    /// Equivalent to the free function [`compile`].
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            settings: SpirvBuilderSettings::new(),
            crate_path: path.as_ref().to_path_buf(),
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
    /// (e.g. `"spirv-unknown-opencl2.0"`). Call order is irrelevant —
    /// settings accumulate and apply when [`build`][Self::build] runs.
    pub fn target_env(mut self, target: impl Into<String>) -> Self {
        self.settings.target_env = target.into();
        self
    }

    /// Add a SPIR-V capability the kernel needs (e.g. `Capability::Float64`).
    pub fn capability(mut self, cap: Capability) -> Self {
        self.settings.capabilities.push(cap);
        self
    }

    /// Set the panic strategy used by SPIR-T to lower `panic!`/`abort`.
    pub fn panic_strategy(mut self, strategy: ShaderPanicStrategy) -> Self {
        self.settings.panic_strategy = Some(strategy);
        self
    }

    /// Escape hatch for settings claspr-build doesn't wrap. Multiple
    /// `with` calls accumulate; closures fire in call order at
    /// terminal-call time, after the inherent setters and presets
    /// have been applied.
    pub fn with(mut self, f: impl Fn(SpirvBuilder) -> SpirvBuilder + 'static) -> Self {
        self.settings.customizers.push(Box::new(f));
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

        let result: CompileResult = self.settings.apply_to(&self.crate_path).build()?;
        let spv_path = result.module.unwrap_single().to_path_buf();
        let kernels = self.kernels;

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
            generate_module_source(&spv_path, &result.entry_points, &kernels, &self.crate_path)?;

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
    // Per-item `#[allow(dead_code)]` — `include!()` doesn't allow an
    // inner attribute at the top of the file, so we tag each
    // typically-unused public item directly. SPV_BYTES /
    // ENTRY_POINTS are exposed for inspection but most callers go
    // through `Kernels::load(&ctx)` and never touch them.
    writeln!(
        s,
        "/// Raw SPIR-V bytes for the kernel module, embedded at build time."
    )?;
    writeln!(s, "#[allow(dead_code)]")?;
    writeln!(
        s,
        "pub const SPV_BYTES: &[u8] = include_bytes!({:?});\n",
        spv_path
    )?;
    writeln!(s, "/// Entry-point names exposed by this module.")?;
    writeln!(s, "#[allow(dead_code)]")?;
    writeln!(s, "pub const ENTRY_POINTS: &[&str] = &[")?;
    for ep in entry_points {
        writeln!(s, "    {ep:?},")?;
    }
    writeln!(s, "];\n")?;

    // The Kernels struct just holds the built Program. Per-launch
    // `clCreateKernel` (done by the proc-macro-emitted launch methods)
    // hands each launch its own private `cl_kernel` handle — no
    // shared mutable state across launches means no `clSetKernelArg`
    // race, so we don't need an `unsafe impl Sync for Kernels {}`
    // hack. The Op the proc-macro emits owns its kernel and has no
    // lifetime tie back to Kernels.
    writeln!(
        s,
        "/// Built program for this module. Per-launch kernel handles are"
    )?;
    writeln!(
        s,
        "/// created via `clCreateKernel` inside the proc-macro-emitted"
    )?;
    writeln!(
        s,
        "/// launch methods (or via [`Kernels::kernel`] for the untyped path)."
    )?;
    writeln!(s, "pub struct Kernels {{")?;
    writeln!(s, "    #[doc(hidden)]")?;
    writeln!(s, "    pub __claspr_program: ::claspr::Program,")?;
    writeln!(s, "}}\n")?;

    writeln!(s, "impl Kernels {{")?;
    writeln!(
        s,
        "    /// Build the program from the embedded SPIR-V and validate"
    )?;
    writeln!(
        s,
        "    /// every entry point exists by attempting one `clCreateKernel`"
    )?;
    writeln!(
        s,
        "    /// per entry-point name (the handle is dropped immediately)."
    )?;
    writeln!(
        s,
        "    /// After `load` returns Ok, per-launch `clCreateKernel` calls"
    )?;
    writeln!(
        s,
        "    /// can only fail with `CL_OUT_OF_RESOURCES` / `CL_OUT_OF_HOST_MEMORY`."
    )?;
    writeln!(
        s,
        "    pub fn load(ctx: &::claspr::Context) -> ::claspr::Result<Self> {{"
    )?;
    writeln!(s, "        let program = ctx.build_program(SPV_BYTES)?;")?;
    writeln!(s, "        for ep in ENTRY_POINTS {{")?;
    writeln!(s, "            let _ = ctx.kernel(&program, ep)?;")?;
    writeln!(s, "        }}")?;
    writeln!(s, "        Ok(Self {{ __claspr_program: program }})")?;
    writeln!(s, "    }}")?;
    writeln!(s)?;
    writeln!(
        s,
        "    /// Get a fresh `cl_kernel` handle for `name`. Panics if the"
    )?;
    writeln!(
        s,
        "    /// runtime is out of resources — `load` validated every entry"
    )?;
    writeln!(
        s,
        "    /// point's existence, so the only remaining failure mode is OOM."
    )?;
    writeln!(
        s,
        "    /// Used internally by proc-macro-emitted launch methods; exposed"
    )?;
    writeln!(
        s,
        "    /// for the rare case where you need a raw kernel by name (e.g."
    )?;
    writeln!(s, "    /// `ctx.launch(&kernels.kernel(\"foo\"), ...)`).")?;
    writeln!(
        s,
        "    pub fn kernel(&self, name: &str) -> ::claspr::Kernel {{"
    )?;
    writeln!(
        s,
        "        ::claspr::Kernel::create(&self.__claspr_program, name)"
    )?;
    writeln!(
        s,
        "            .unwrap_or_else(|e| panic!(\"clCreateKernel({{name:?}}) failed: {{e:?}}\"))"
    )?;
    writeln!(s, "    }}")?;
    writeln!(s, "}}")?;

    // The compile() path's KernelDecl-driven typed launch methods are
    // no longer emitted here — switching every emission to match the
    // proc-macro's per-call clCreateKernel + Op + DeviceOperation shape
    // would mean re-implementing the macro inside the build script.
    // No example or test exercises the compile()+KernelDecl path, so
    // we leave typed-launch emission to the proc-macro and treat the
    // `kernels` argument as a no-op for now.
    let _ = kernels;

    Ok(s)
}

// ── compile_from_host: single-source kernel extraction ────────────────

/// Build kernel SPIR-V from a *host-crate* source file containing
/// `#[claspr::kernel]`-marked functions.
///
/// This is the "single source" mode — kernel function bodies live in
/// the host crate (next to where they're called from), and `claspr-build`
/// extracts them at build time into a generated kernel sub-crate that
/// `spirv-builder` then compiles. The host crate's own compilation of
/// the same file goes through the [`#[claspr::kernel]`][kernel-macro]
/// proc-macro, which emits a host launch wrapper from the same source.
///
/// The whole source file is copied into the generated kernel crate
/// (modulo translating `#[claspr::kernel]` → `#[spirv(kernel)]`), so
/// device-side helper functions can sit alongside entry points without
/// any extra annotation.
///
/// ```ignore
/// // build.rs
/// claspr_build::compile_from_host("src/kernels.rs")
///     .opencl12()
///     .write_to(format!("{}/kernels.rs", std::env::var("OUT_DIR").unwrap()))?;
/// ```
///
/// [kernel-macro]: https://docs.rs/claspr_macros
pub fn compile_from_host(src_file: impl AsRef<Path>) -> HostBuilder {
    HostBuilder::new(src_file)
}

/// Builder for [`compile_from_host`].
///
/// The preset / capability / panic-strategy methods carry the same
/// semantics as on [`CompileBuilder`]; the terminal call is
/// [`write_to`] just like the explicit-kernel-crate flow.
///
/// [`write_to`]: Self::write_to
pub struct HostBuilder {
    settings: SpirvBuilderSettings,
    src_file: PathBuf,
}

impl HostBuilder {
    fn new(src_file: impl AsRef<Path>) -> Self {
        Self {
            settings: SpirvBuilderSettings::new(),
            src_file: src_file.as_ref().to_path_buf(),
        }
    }

    /// Set the SPIR-V target environment string.
    pub fn target_env(mut self, target: impl Into<String>) -> Self {
        self.settings.target_env = target.into();
        self
    }

    /// Add a SPIR-V capability the kernels need.
    pub fn capability(mut self, cap: Capability) -> Self {
        self.settings.capabilities.push(cap);
        self
    }

    /// Set the panic strategy.
    pub fn panic_strategy(mut self, strategy: ShaderPanicStrategy) -> Self {
        self.settings.panic_strategy = Some(strategy);
        self
    }

    /// Escape hatch for settings claspr-build doesn't wrap. Multiple
    /// `with` calls accumulate; closures fire in call order at build
    /// time, after the inherent setters and presets have been
    /// applied. Each device module's [`SpirvBuilder`] gets the
    /// customizations independently.
    pub fn with(mut self, f: impl Fn(SpirvBuilder) -> SpirvBuilder + 'static) -> Self {
        self.settings.customizers.push(Box::new(f));
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
    /// collective kernels with barriers.
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

    /// Discover every `#[claspr::device] mod <name>` in the host
    /// source, generate one kernel sub-crate per module, compile each
    /// via rust-gpu, and write the corresponding [`Kernels`-style
    /// module][module-shape] to `OUT_DIR/<name>.rs`. The
    /// `#[claspr::device]` proc-macro on each module includes from
    /// the matching `<name>.rs`, so module name is the only piece of
    /// coupling between this side and the host source.
    ///
    /// Multiple `#[claspr::device]` modules in one source file are
    /// fine — each gets its own kernel sub-crate, SPV blob, and
    /// generated `Kernels`. Top-level `#[claspr::kernel]` /
    /// `#[claspr::device]` items outside any device module are
    /// rejected as an error: organise kernel code into a module so
    /// the per-module file naming has something to key off.
    ///
    /// Emits `cargo:rerun-if-changed=` for the source file so changes
    /// trigger a rebuild.
    ///
    /// [module-shape]: CompileBuilder::write_to
    pub fn write(self) -> Result<()> {
        println!("cargo:rerun-if-changed={}", self.src_file.display());

        let source = std::fs::read_to_string(&self.src_file)
            .map_err(|e| format!("read {}: {}", self.src_file.display(), e))?;
        let parsed: syn::File = syn::parse_str(&source)
            .map_err(|e| format!("parse {}: {}", self.src_file.display(), e))?;

        // Reject top-level marked items — they have no module name to
        // key the output filename against.
        for item in &parsed.items {
            if let syn::Item::Fn(f) = item
                && (has_any_claspr_marker(&f.attrs))
            {
                return Err(format!(
                    "claspr::kernel / claspr::device on top-level fn `{}` is unsupported by \
                     compile_from_host — wrap kernel code in `#[claspr::device] mod <name> {{ ... }}` \
                     so the generated file can be named after the module",
                    f.sig.ident,
                )
                .into());
            }
        }

        let device_mods: Vec<&syn::ItemMod> = parsed
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Mod(m) if has_any_claspr_device_attr(&m.attrs) => Some(m),
                _ => None,
            })
            .collect();
        if device_mods.is_empty() {
            return Err(format!(
                "no #[claspr::device] mod found in {} — at least one is required",
                self.src_file.display(),
            )
            .into());
        }

        let out_dir = std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?;
        let out_dir = PathBuf::from(&out_dir);

        for device_mod in device_mods {
            self.compile_one_module(device_mod, &out_dir)?;
        }
        Ok(())
    }

    /// Build one device module into its own kernel sub-crate.
    fn compile_one_module(&self, device_mod: &syn::ItemMod, out_dir: &Path) -> Result<()> {
        let mod_name = device_mod.ident.to_string();

        // Lift the module body into a fresh syn::File representing the
        // kernel sub-crate's lib.rs contents. Submodules declared with
        // `mod foo;` (no inline body) are followed using rustc's
        // standard file-resolution rules, so the user can split a
        // device module across files.
        //
        // The "module directory" for the device module's body is
        // `<src_file_dir>/<mod_name>/` — same convention rustc uses
        // for inline modules at the crate root.
        let src_dir = self
            .src_file
            .parent()
            .ok_or("source file has no parent directory")?;
        let device_mod_dir = src_dir.join(&mod_name);
        let raw_items = device_mod
            .content
            .as_ref()
            .map(|(_, items)| items.clone())
            .unwrap_or_default();
        let lifted_items = translate_and_inline(raw_items, &device_mod_dir)?;
        let lifted_file = syn::File {
            shebang: None,
            attrs: vec![],
            items: lifted_items,
        };

        // Materialise per-module kernel sub-crate. Distinct dir per
        // module so multiple modules don't clobber each other.
        let crate_dir = out_dir.join(format!("claspr_kernel_{mod_name}"));
        std::fs::create_dir_all(crate_dir.join("src"))?;
        write_generated_cargo_toml(&crate_dir)?;
        write_generated_lib_rs(&crate_dir, &lifted_file)?;
        // Seed the sub-crate with the host workspace's Cargo.lock so
        // shared transitive deps (notably `glam`) resolve to the same
        // versions the host built against. Without this, the sub-crate
        // is its own independent workspace and would re-resolve fresh
        // against crates.io. That can pick up new releases that gate
        // previously-always-available items behind features — e.g.
        // glam 0.33 moved `UVec2/UVec3/UVec4/IVec*` behind the
        // `integer-types` feature, which `spirv-std`'s
        // `opencl-kernel-support` branch doesn't request, breaking
        // its build.
        seed_lockfile_from_host(src_dir, &crate_dir);

        // Compile via spirv-builder.
        let result: CompileResult = self.settings.apply_to(&crate_dir).build()?;
        let spv_path = result.module.unwrap_single().to_path_buf();

        // Emit the Kernels module to OUT_DIR/<mod_name>.rs — this
        // is what `#[claspr::device] mod <name>` includes!() from.
        let module_out_path = out_dir.join(format!("{mod_name}.rs"));
        let generated = generate_module_source(&spv_path, &result.entry_points, &[], &crate_dir)?;
        std::fs::write(&module_out_path, generated)?;
        Ok(())
    }
}

/// Translate items lifted out of a `#[claspr::device]` module body
/// (or a submodule within), recursively inlining external module
/// declarations (`mod foo;` whose body lives in a separate file).
///
/// `dir` is the directory the *current* scope's `mod foo;` declarations
/// resolve against, following rustc's normal rules:
/// - For an inline module nested inside the device module, push the
///   inline module's name onto the parent dir.
/// - For an external module loaded from `<dir>/<name>.rs`, the
///   sub-modules of that file resolve against `<dir>/<name>/`.
/// - For an external module loaded from `<dir>/<name>/mod.rs`, same
///   resolution dir as the file.
///
/// Function items get their `#[claspr::kernel]` attrs translated to
/// `#[spirv(kernel)]`; everything else (use, const, static, struct,
/// type def, …) passes through verbatim.
fn translate_and_inline(items: Vec<syn::Item>, dir: &Path) -> Result<Vec<syn::Item>> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            syn::Item::Fn(mut f) => {
                translate_fn_attrs(&mut f.attrs);
                out.push(syn::Item::Fn(f));
            }
            syn::Item::Mod(mut m) => {
                if let Some((brace, inner)) = m.content {
                    // Inline module — recurse into its body, with
                    // the dir extended by the module's name.
                    let sub_dir = dir.join(m.ident.to_string());
                    let inlined = translate_and_inline(inner, &sub_dir)?;
                    m.content = Some((brace, inlined));
                    out.push(syn::Item::Mod(m));
                } else {
                    // External module declaration `mod foo;` — find
                    // the file and recurse on its items.
                    let name = m.ident.to_string();
                    let (path, sub_dir) = resolve_module_file(dir, &name)?;
                    println!("cargo:rerun-if-changed={}", path.display());
                    let source = std::fs::read_to_string(&path)
                        .map_err(|e| format!("read {}: {}", path.display(), e))?;
                    let parsed: syn::File = syn::parse_str(&source)
                        .map_err(|e| format!("parse {}: {}", path.display(), e))?;
                    let inlined = translate_and_inline(parsed.items, &sub_dir)?;
                    m.content = Some((syn::token::Brace::default(), inlined));
                    m.semi = None;
                    out.push(syn::Item::Mod(m));
                }
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Resolve `mod <name>;` against `dir` using rustc's file-naming
/// rules. Returns `(file_path, dir_for_that_module's_submodules)`.
fn resolve_module_file(dir: &Path, name: &str) -> Result<(PathBuf, PathBuf)> {
    let as_file = dir.join(format!("{name}.rs"));
    let as_dir_mod = dir.join(name).join("mod.rs");
    if as_file.exists() {
        Ok((as_file, dir.join(name)))
    } else if as_dir_mod.exists() {
        Ok((as_dir_mod, dir.join(name)))
    } else {
        Err(format!(
            "could not resolve `mod {name};` — looked for {} and {}. \
             #[path = \"...\"] overrides aren't supported yet.",
            as_file.display(),
            as_dir_mod.display(),
        )
        .into())
    }
}

fn has_any_claspr_marker(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| is_claspr_kernel_attr(a) || is_claspr_device_attr(a))
}

fn has_any_claspr_device_attr(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(is_claspr_device_attr)
}

fn translate_fn_attrs(attrs: &mut Vec<syn::Attribute>) {
    let mut out = Vec::with_capacity(attrs.len());
    for attr in attrs.drain(..) {
        if is_claspr_kernel_attr(&attr) {
            out.push(syn::parse_quote!(#[spirv(kernel)]));
        } else if is_claspr_device_attr(&attr) {
            // Pure marker — drop with no replacement.
        } else {
            out.push(attr);
        }
    }
    *attrs = out;
}

fn is_claspr_kernel_attr(attr: &syn::Attribute) -> bool {
    attr_path_matches(attr, "kernel")
}

fn is_claspr_device_attr(attr: &syn::Attribute) -> bool {
    attr_path_matches(attr, "device")
}

fn attr_path_matches(attr: &syn::Attribute, name: &str) -> bool {
    let segs: Vec<String> = attr
        .path()
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    match segs.as_slice() {
        [single] => single == name,
        [first, second] => first == "claspr" && second == name,
        _ => false,
    }
}

/// Best-effort: walk up from `src_dir` looking for a `Cargo.lock` (the
/// host workspace's), and copy it into the generated kernel sub-crate
/// so cargo reuses the host's version pins for transitive deps. Silent
/// on failure — the build still works if the lock is missing, it just
/// resolves fresh and may pick newer (potentially-incompatible) deps.
fn seed_lockfile_from_host(src_dir: &Path, crate_dir: &Path) {
    let mut probe = src_dir.to_path_buf();
    for _ in 0..16 {
        let candidate = probe.join("Cargo.lock");
        if candidate.exists() {
            let _ = std::fs::copy(&candidate, crate_dir.join("Cargo.lock"));
            return;
        }
        if !probe.pop() {
            return;
        }
    }
}

fn write_generated_cargo_toml(crate_dir: &Path) -> Result<()> {
    // Hardcoded for the bricevideau-ai/rust-gpu opencl-kernel-support
    // branch used throughout this workspace. Future iterations should
    // either inherit from the host workspace or take the dep specs as
    // a builder parameter.
    // Empty `[workspace]` table makes this a standalone workspace —
    // it lives under the host's `target/` so cargo would otherwise
    // try to associate it with the host workspace and fail.
    let cargo_toml = r#"[package]
name = "claspr_generated_kernels"
version = "0.0.0"
edition = "2024"

[workspace]

[lib]
crate-type = ["dylib"]

[dependencies]
spirv-std = { git = "https://github.com/bricevideau-ai/rust-gpu.git", branch = "opencl-kernel-support" }
glam = { version = ">=0.30.8", default-features = false }
# Always-available extras for kernel code: `num-complex` for Complex
# arithmetic. Tiny no_std crate; pulling it unconditionally beats
# requiring every user to extend the generated Cargo.toml just to
# `use num_complex::Complex32;`.
num-complex = { version = "0.4", default-features = false }
"#;
    std::fs::write(crate_dir.join("Cargo.toml"), cargo_toml)?;
    Ok(())
}

fn write_generated_lib_rs(crate_dir: &Path, file: &syn::File) -> Result<()> {
    // Preamble injected by claspr-build. Beyond the no_std attribute,
    // we inject `use spirv_std::spirv;` because claspr-build's own
    // translation emits `#[spirv(kernel)]` (rewritten from
    // `#[claspr::kernel]`) — so `spirv` MUST be in scope for any
    // claspr kernel module to compile, and the user never writes
    // `spirv` directly. The crate-level `#[allow(unused_imports)]`
    // covers degenerate cases (e.g. a device module with only
    // helper fns and no kernel entry points).
    //
    // Other spirv-std names users do reference directly — `Image` for
    // image kernel param types, `cl::Float3` etc. for vector
    // arithmetic, `opencl_std` for math intrinsics, `num_traits::Float`
    // for the libm intercept on bare `f32` — are imported by the
    // user alongside the kernel, since they're situational and the
    // user's import makes the dep visible at the call site.
    let mut s = String::new();
    s.push_str("#![cfg_attr(target_arch = \"spirv\", no_std)]\n");
    s.push_str("#![allow(unused_imports)]\n\n");
    s.push_str("use spirv_std::spirv;\n\n");
    for item in &file.items {
        s.push_str(&item.to_token_stream().to_string());
        s.push_str("\n\n");
    }
    std::fs::write(crate_dir.join("src/lib.rs"), s)?;
    Ok(())
}
