# claspr — repo orientation

Single-source OpenCL with [rust-gpu](https://github.com/Rust-GPU/rust-gpu). Kernel + host code in one Rust file (or split across files), build script extracts the kernel side into a generated sub-crate compiled by rust-gpu, proc-macro emits a typed launch wrapper. Built on top of [`bricevideau-ai/rust-gpu`](https://github.com/bricevideau-ai/rust-gpu) branch `opencl-kernel-support`.

## Workspace

```
claspr/                       runtime helpers — Context, DeviceSlice,
                              KernelArgs, Image2DRgba8. Host-only,
                              no spirv-builder dep.
claspr-build/                 build-script library — compile() and
                              compile_from_host(). Owns the kernel-
                              crate generation + spirv-builder
                              invocation + module-following.
claspr-macros/                proc-macros — #[kernel] and #[device].
examples/collatz/             single-file demo (slice + scalar args).
examples/raymarch/            multi-file demo (src/main.rs + src/gpu/
                              scene.rs + src/gpu/shading.rs).
examples/mandelbrot-kernel/   kernel-as-library — Mandelbrot image kernel.
examples/sobel-kernel/        kernel-as-library — Sobel edge detector.
examples/image-pipeline/      bin that depends on both kernel libraries
                              above and chains them.
```

## How the single-source pipeline fits together

User writes one source file (e.g. `examples/collatz/src/main.rs`). It contains:

- **Top-level host code**: `use claspr::*`, `fn main`, optional `#[cfg(test)] mod tests`. No `mod compiled` — the device macro injects the include for you.
- **`#[claspr::device] mod gpu { ... }`** — the device side, in a single tagged module. Inside (user-written): kernel-only `use` statements (cfg-gated to `target_arch = "spirv"` if the host doesn't depend on those crates), `const`s, helper `fn`s, optional `mod foo;` declarations to split the module across files, and one or more `#[claspr::kernel]` entry points (defaults to `kernels = Kernels` — the relative-path `Kernels` resolves to the one the macro injects below). Inside (macro-injected, at the end of the module body): `include!(concat!(env!("OUT_DIR"), "/<modname>.rs"))` (brings `Kernels` + `Kernels::load` + `SPV_BYTES` + `ENTRY_POINTS` in) and a `pub fn kernels(ctx) -> Result<Kernels>` convenience wrapper.
- The user does *not* need to import `spirv` from spirv-std — `claspr-build`'s preamble injects `use spirv_std::spirv;` because every translated `#[claspr::kernel]` becomes `#[spirv(kernel)]` and the `spirv` proc-macro must be in scope. Anything else (`Image`, `cl::*`, `opencl_std`, `num_traits::Float`, …) the user imports themselves.
- Calling code reads `let kernels = gpu::kernels(&ctx)?;` then `kernels.collatz_kernel(&ctx, ...)`. Multiple `#[claspr::device]` modules in the same file each scope their own `Kernels`/`kernels()` — no collisions.
- The build script writes one `OUT_DIR/<modname>.rs` per device module it finds — the macro's injected include matches the module ident, so module name is the only piece of coupling between the build-script side and the host source. Top-level `#[claspr::kernel]` / `#[claspr::device]` items outside any module are rejected: organise kernel code into a module so the per-module file naming has something to key off.

Two compilation paths run on the same source:

1. **Host build** (cargo's normal flow). The proc-macros do the heavy lifting:
   - `#[claspr::device]` on a fn → `#[allow(dead_code, unused_imports)] <fn>` (no semantic change beyond the warning suppression).
   - `#[claspr::device]` on a mod → re-emits the user's module with two extra items appended *inside* the body: an `include!(concat!(env!("OUT_DIR"), "/<modname>.rs"))` (brings `Kernels`/`Kernels::load` into the module's scope) and a `pub fn kernels(&ctx) -> Result<Kernels>` convenience wrapper. The whole module is wrapped in `#[allow(dead_code, unused_imports)]`.
   - `#[claspr::kernel(kernels = path)]` (path defaults to `Kernels` — relative; resolves to the device module's local `Kernels`) parses the kernel-style fn signature, drops `#[spirv(<builtin>)]` params, translates `#[spirv(cross_workgroup)] &mut [T]` → `&claspr::DeviceSlice<T>` and `&Image!(...)` → `&claspr::Image2DRgba8`, then emits `impl path { fn name(&self, ctx, grid, args...) }`. The impl ends up inside the same module, attached to the same `Kernels` struct the include brought in. The original kernel body is discarded on the host side.
2. **Kernel build** (driven by `examples/<name>/build.rs` calling `claspr_build::compile_from_host(src).opencl12().write()`):
   - Reads the source file, parses with syn.
   - Discovers every top-level `Item::Mod` with `#[claspr::device]`. Top-level `Item::Fn` with `#[claspr::kernel]` / `#[claspr::device]` (no enclosing module) is an error — there's no module name to use for the output file.
   - For each device module, lifts its body into a fresh kernel sub-crate at `OUT_DIR/claspr_kernel_<modname>/`. `mod foo;` declarations inside the body are followed using rustc's standard file-resolution rules (`<dir>/<name>.rs` then `<dir>/<name>/mod.rs`); `cargo:rerun-if-changed` is emitted for each followed file. Inside the lifted body, translates `#[claspr::kernel(...)]` → `#[spirv(kernel)]` and strips `#[claspr::device]`. Wrapper preamble injects `#![cfg_attr(target_arch = "spirv", no_std)]`, `#![allow(unused_imports)]`, and `use spirv_std::spirv;`.
   - Writes a `Cargo.toml` for the sub-crate (with `[workspace]` so cargo doesn't try to attach it to the host workspace; spirv-std + glam deps hardcoded at the rust-gpu branch we depend on).
   - Runs `SpirvBuilder` on each sub-crate, then writes the `Kernels { ... }` struct (one `pub` field per entry point + `Kernels::load(&Context)`) to `OUT_DIR/<modname>.rs`. The matching `#[claspr::device]` on the host side `include!()`s this exact path.

## Library-crate composition

A claspr kernel can be packaged as its own library crate (`pub mod gpu`, build.rs of its own), and consumed from a host binary via `lib_name::kernels(&ctx)?.entry_point(...)`. The host binary needs no build.rs; each kernel library carries its own. Two patterns:

- **Pure kernel library** (mandelbrot-kernel, sobel-kernel today): exposes only typed launch handles. spirv-std imports inside the library's device module are cfg-gated to `target_arch = "spirv"` so consumers don't pay for spirv-std as a transitive dep. Helpers can't reference device-only types directly (the library's host build won't have those types in scope).
- **Mixed kernel + host library** (raymarch is binary-shaped today; could be a library): exposes typed launch handles AND host-callable helpers (e.g., `pixel_color` for validation). Needs spirv-std as a *regular* host dep so types like `cl::Float3` are in scope on both sides. Cost: consumers pull spirv-std transitively. Benefit: full host parity.

## Key files

- `claspr-build/src/lib.rs` — both `compile()` (separate kernel crate) and `compile_from_host()` (single-source extraction) live here. The translation logic (`translate_and_inline`, `resolve_module_file`, `is_claspr_kernel_attr`, `is_claspr_device_attr`) is the most-likely-to-change surface as new kernel patterns surface; multi-file resolution rules also live there.
- `claspr-macros/src/lib.rs` — `#[kernel]` + `#[device]`. The kernel signature → host wrapper translation lives in `classify_param`, `translate_cross_workgroup_ty`, `is_image_param`. The device-on-mod injection (include! + `kernels()` fn) lives in the `device` proc-macro body.
- `claspr/src/launch.rs` — `KernelArg` / `KernelArgs` / `LaunchSpec` / `IntoLaunchSpec`. Tuple impls are macro-emitted up to arity 8.
- `examples/raymarch/src/main.rs` + `src/gpu/scene.rs` + `src/gpu/shading.rs` — multi-file device-module reference. Cross-file `use super::scene::...` works because the build script preserves module structure during inlining.
- `examples/image-pipeline/src/main.rs` — library-composition reference. Pulls two kernel libraries and chains their launches.

## Conventions

- **Toolchain**: pinned at `nightly-2026-04-11` (rust-gpu's pinned nightly) via `rust-toolchain.toml`. Same channel as `bricevideau-ai/rust-gpu opencl-kernel-support` and `rust-gpu-opencl-samples`. Bump in lockstep with rust-gpu.
- **License**: BSD-3-Clause, copyright `Brice Videau, Argonne National Laboratory`. One `LICENSE` file at root, no per-file headers.
- **Lint before push**: `cargo fmt --all` + `cargo clippy --all-targets -- -D warnings` + `cargo doc --no-deps --workspace` with `RUSTDOCFLAGS=-D warnings`. CI (`.github/workflows/ci.yaml`) runs the same checks plus rustfmt on `tests/*/compile_fail/*.rs` (which `cargo fmt --all` skips) and the integration test suites on rusticl/llvmpipe.
- **No CHANGELOG / release notes** — too early.

## Build / test / run

All examples need an OpenCL runtime. On this machine, pocl is at `~/.local`:

```bash
# Whole-workspace check
cargo build
cargo clippy --all-targets -- -D warnings

# Run an example
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo run -p collatz-example
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo run -p raymarch-example      # writes raymarch.ppm
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo run -p image-pipeline        # writes image-pipeline.ppm

# Tests (skip silently if no OpenCL device)
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo test -p collatz-example
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo test -p raymarch-example
```

On macOS the system OpenCL framework picks up automatically; no `OCL_ICD_VENDORS` needed (the Rust opencl3 binding goes through Apple's `OpenCL.framework`).

## Common gotchas

- **Multi-file device modules need `#![feature(proc_macro_hygiene)]`** at the user's crate root. `mod foo;` (file modules) inside a proc-macro's input is gated on nightly (rust-lang/rust#54727). The proc-macro can't auto-inject this — feature attrs only apply at the crate root, and macros can't reach there. Single-file device modules don't need the gate.
- **Library-crate spirv-std imports must be cfg-gated** if the library doesn't pull spirv-std as a regular host dep. The proc-macro discards builtin params and kernel bodies before host name resolution touches the spirv-std names, so `#[cfg(target_arch = "spirv")] use spirv_std::{...}` is enough — host doesn't need them. Helpers that take device-only parameter types (`fn foo(img: &Image!(...))`) can't be host-callable in this pattern; restructure to take primitive args, or switch to the "mixed kernel + host library" pattern (regular spirv-std host dep).
- **Builtin param types are never name-resolved on host.** The `_id: ::glam::USizeVec3` in a `#[spirv(global_invocation_id)]` param doesn't require glam to be a host dep — the proc-macro discards the whole parameter before name resolution touches the type.
- **Generated kernel `Cargo.toml` must have `[workspace]`** — without it, cargo tries to associate the OUT_DIR-located crate with the host workspace and refuses (`current package believes it's in a workspace when it's not`).
- **`Kernel` field name == method name** on `Kernels`. The generated impl block adds a method named after the kernel; the field is also named after the kernel (private). Rust disambiguates by syntax (`kernels.foo` is the field, `kernels.foo(...)` is the method). Don't try to make them different.
- **`librustc_codegen_spirv.so` collision warning** when running `cargo build --workspace`: cargo issue #6313 (workspace-level dylib build-dep collision). Building one example at a time (`cargo build -p collatz-example`) is silent. Forward-looking warning, not currently a hard error.

## Backlog (deferred)

- **Auto-dependency-tracking codegen** — walk the call graph from each `#[claspr::kernel]` body, transitively pull in only the local fns it actually references. Would eliminate per-fn `#[claspr::device]` markers entirely. Module-level marker would still be useful as the "everything in here" boundary. Non-trivial: needs body parsing + cross-module call resolution + local-vs-external distinction.
- **Auto-enable rust-gpu capabilities** — codegen is currently inconsistent about which `Capability::*` it auto-emits versus what build scripts must declare. Move toward auto-adding everything the codegen actually needs and let device-side `clBuildProgram` rejection handle "this device can't run it." Will likely shrink `CompileBuilder::capability` / `with_f64` etc.
- **Cargo-features-driven `auto()` build.rs** — move OpenCL version + remaining capability flags to Cargo features so the build.rs collapses to `claspr_build::auto()`.
- **`#[path = "..."]` attribute support in multi-file resolution** — currently errors with a clear message.
- **`cargo:warning=` for missing `proc_macro_hygiene` feature gate** — claspr-build could detect file modules in device-module bodies and check if the crate's inner attrs include the gate; warn at build time before the rustc error fires.
- **Inherit `spirv-std` / `glam` dep specs from the host workspace** rather than hardcoding the bricevideau-ai/rust-gpu branch in the generated `Cargo.toml`.
- **More samples through single-source**: subgroup ops + workgroup memory (reduce sample), f64 (nbody), printf, sampler-based image reads. Each will likely surface a small claspr-build / proc-macro extension.
- **`cargo claspr` subcommand** to eliminate build.rs entirely (NVlabs/cuda-oxide-style). Doesn't port cleanly because rust-gpu is a whole-crate codegen backend (no per-item dispatch); workflow friction (plain cargo stops working, IDE tooling breaks) likely makes it not worth it.

## Sibling repos

- `bricevideau-ai/rust-gpu` (`opencl-kernel-support` branch) — the rust-gpu fork with the kernel target. claspr-build's generated kernel `Cargo.toml` pins this.
- `bricevideau-ai/rust-gpu-opencl-samples` — the standalone-kernel-crate samples. Most patterns claspr generalises came from there.
