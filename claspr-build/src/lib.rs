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
//!     // kernels.collatz_kernel(&ctx, [n], &buf)?;
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
//! kernels.collatz_kernel(&ctx, [n], &buf)?;
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
//! kernels.collatz_kernel(&ctx, [n], &buf)?;
//! ```
//!
//! instead of the raw [`claspr::Context::launch`] form.
//!
//! [`claspr::Context::launch`]: https://docs.rs/claspr
//! [kernel-macro]: https://docs.rs/claspr_macros
//! [claspr]: https://github.com/bricevideau-ai/claspr

use quote::ToTokens;
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
    src_file: PathBuf,
    target_env: String,
    capabilities: Vec<Capability>,
    panic_strategy: Option<ShaderPanicStrategy>,
}

impl HostBuilder {
    fn new(src_file: impl AsRef<Path>) -> Self {
        Self {
            src_file: src_file.as_ref().to_path_buf(),
            target_env: "spirv-unknown-opencl1.2".to_string(),
            capabilities: Vec::new(),
            panic_strategy: None,
        }
    }

    /// Set the SPIR-V target environment string.
    pub fn target_env(mut self, target: impl Into<String>) -> Self {
        self.target_env = target.into();
        self
    }

    /// Add a SPIR-V capability the kernels need.
    pub fn capability(mut self, cap: Capability) -> Self {
        self.capabilities.push(cap);
        self
    }

    /// Set the panic strategy.
    pub fn panic_strategy(mut self, strategy: ShaderPanicStrategy) -> Self {
        self.panic_strategy = Some(strategy);
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

        // Lift the module body into a fresh syn::File representing
        // the kernel sub-crate's lib.rs contents.
        let lifted_items: Vec<syn::Item> = device_mod
            .content
            .as_ref()
            .map(|(_, items)| items.iter().cloned().map(translate_lifted_item).collect())
            .unwrap_or_default();
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

        // Compile via spirv-builder.
        let mut sb = SpirvBuilder::new(&crate_dir, &self.target_env);
        for cap in &self.capabilities {
            sb = sb.capability(*cap);
        }
        if let Some(ps) = self.panic_strategy {
            sb = sb.shader_panic_strategy(ps);
        }
        let result: CompileResult = sb.build()?;
        let spv_path = result.module.unwrap_single().to_path_buf();

        // Emit the Kernels module to OUT_DIR/<mod_name>.rs — this
        // is what `#[claspr::device] mod <name>` includes!() from.
        let module_out_path = out_dir.join(format!("{mod_name}.rs"));
        let generated = generate_module_source(&spv_path, &result.entry_points, &[], &crate_dir)?;
        std::fs::write(&module_out_path, generated)?;
        Ok(())
    }
}

/// Translate an item that's been lifted out of a `#[claspr::device]`
/// module. Functions get their `#[claspr::kernel]` attrs translated;
/// other item kinds (use, const, static, struct, mod, …) pass
/// through verbatim.
fn translate_lifted_item(item: syn::Item) -> syn::Item {
    match item {
        syn::Item::Fn(mut f) => {
            translate_fn_attrs(&mut f.attrs);
            syn::Item::Fn(f)
        }
        other => other,
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
"#;
    std::fs::write(crate_dir.join("Cargo.toml"), cargo_toml)?;
    Ok(())
}

fn write_generated_lib_rs(crate_dir: &Path, file: &syn::File) -> Result<()> {
    // Bare-minimum preamble — just the no_std attribute. The user's
    // `#[claspr::device]` module is expected to bring its own `use`
    // statements (`use spirv_std::{glam, spirv};` etc.), which carry
    // through to the generated kernel crate verbatim. Per-fn
    // `#[claspr::device]` callers without a wrapping module pay the
    // price of declaring their own `use` statements at module level
    // alongside the function — same source file, same scope.
    let mut s = String::new();
    s.push_str("#![cfg_attr(target_arch = \"spirv\", no_std)]\n\n");
    for item in &file.items {
        s.push_str(&item.to_token_stream().to_string());
        s.push_str("\n\n");
    }
    std::fs::write(crate_dir.join("src/lib.rs"), s)?;
    Ok(())
}
