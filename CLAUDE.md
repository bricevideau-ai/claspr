# claspr — repo orientation

Single-source OpenCL with [rust-gpu](https://github.com/Rust-GPU/rust-gpu). Kernel + host code in one Rust file; build script extracts the kernel side into a generated sub-crate compiled by rust-gpu, proc-macro emits a typed launch wrapper. Built on top of [`bricevideau-ai/rust-gpu`](https://github.com/bricevideau-ai/rust-gpu) branch `opencl-kernel-support`.

## Workspace

```
claspr/                runtime helpers (Context, DeviceSlice, KernelArgs, Image2DRgba8, compile())
claspr-build/          build-script library — compile() / compile_from_host()
claspr-macros/         proc-macros — #[kernel(kernels = ...)] and #[device]
examples/collatz/      single-file demo (slice + scalar args)
examples/raymarch/     larger demo (consts, helpers, image kernel)
```

## How the single-source pipeline fits together

User writes one source file (e.g. `examples/collatz/src/main.rs`). It contains:

- **Top-level host code**: `use claspr::*`, `fn main`, optional `#[cfg(test)] mod tests`. No `mod compiled` — that's owned by the device module now.
- **`#[claspr::device] mod gpu { ... }`** — the device side, in a single tagged module. Inside (user-written): kernel-only `use` statements (cfg-gated to `target_arch = "spirv"` if the host doesn't depend on those crates), `const`s, helper `fn`s, and one or more `#[claspr::kernel]` entry points (defaults to `kernels = Kernels` — the relative-path `Kernels` resolves to the one the macro injects below). Inside (macro-injected, at the end of the module body): `include!(concat!(env!("OUT_DIR"), "/kernels.rs"));` (brings `Kernels` + `Kernels::load` + `SPV_BYTES` + `ENTRY_POINTS` in) and a `pub fn kernels(ctx) -> Result<Kernels>` convenience wrapper. The user does *not* need to import `spirv` or `Image` from spirv-std — claspr-build's preamble already provides them on the kernel side.
- Calling code reads `let kernels = gpu::kernels(&ctx)?;` then `kernels.collatz_kernel(&ctx, ...)`. Multiple `#[claspr::device]` modules in the same file each scope their own `Kernels`/`kernels()` — no collisions.
- The build script writes one `OUT_DIR/<modname>.rs` per device module it finds — the macro's injected include matches the module ident, so module name is the only piece of coupling between the build-script side and the host source. Top-level `#[claspr::kernel]` / `#[claspr::device]` items outside any module are rejected: organise kernel code into a module so the per-module file naming has something to key off.

Two compilation paths run on the same source:

1. **Host build** (cargo's normal flow). The proc-macros do the heavy lifting:
   - `#[claspr::device]` on a fn → `#[allow(dead_code, unused_imports)] <fn>` (no semantic change beyond the warning suppression).
   - `#[claspr::device]` on a mod → re-emits the user's module with two extra items appended *inside* the body: an `include!(concat!(env!("OUT_DIR"), "/kernels.rs"))` (brings `Kernels`/`Kernels::load` into the module's scope) and a `pub fn kernels(&ctx) -> Result<Kernels>` convenience wrapper. The whole module is wrapped in `#[allow(dead_code, unused_imports)]`.
   - `#[claspr::kernel(kernels = path)]` (path defaults to `Kernels` — relative; resolves to the device module's local `Kernels`) parses the kernel-style fn signature, drops `#[spirv(<builtin>)]` params, translates `#[spirv(cross_workgroup)] &mut [T]` → `&claspr::DeviceSlice<T>` and `&Image!(...)` → `&claspr::Image2DRgba8`, then emits `impl path { fn name(&self, ctx, grid, args...) }`. The impl ends up inside the same module, attached to the same `Kernels` struct the include brought in. The original kernel body is discarded on the host side.
2. **Kernel build** (driven by `examples/<name>/build.rs` calling `claspr_build::compile_from_host(src).opencl12().write()`):
   - Reads the source file, parses with syn.
   - Discovers every top-level `Item::Mod` with `#[claspr::device]`. Top-level `Item::Fn` with `#[claspr::kernel]` / `#[claspr::device]` (no enclosing module) is an error — there's no module name to use for the output file.
   - For each device module, lifts its body verbatim into a fresh kernel sub-crate at `OUT_DIR/claspr_kernel_<modname>/`. Inside the lifted body, translates `#[claspr::kernel(...)]` → `#[spirv(kernel)]` and strips `#[claspr::device]`. Wrapper preamble is just `#![cfg_attr(target_arch = "spirv", no_std)]` — user's `use` lines come along.
   - Writes a `Cargo.toml` for the sub-crate (with `[workspace]` so cargo doesn't try to attach it to the host workspace; spirv-std + glam deps hardcoded at the rust-gpu branch we depend on).
   - Runs `SpirvBuilder` on each sub-crate, then writes the `Kernels { ... }` struct (one `pub` field per entry point + `Kernels::load(&Context)`) to `OUT_DIR/<modname>.rs`. The matching `#[claspr::device]` on the host side `include!()`s this exact path.

## Key files

- `claspr-build/src/lib.rs` — both `compile()` (separate kernel crate) and `compile_from_host()` (single-source extraction) are here. The translation logic (`translate_for_kernel_crate`, `translate_lifted_item`, `is_claspr_kernel_attr`, `is_claspr_device_attr`) is the most-likely-to-change surface as new kernel patterns surface.
- `claspr-macros/src/lib.rs` — `#[kernel]` + `#[device]`. The kernel signature → host wrapper translation lives in `classify_param` and `translate_cross_workgroup_ty` / `is_image_param`.
- `claspr/src/launch.rs` — `KernelArg` / `KernelArgs` / `LaunchSpec` / `IntoLaunchSpec`. Tuple impls are macro-emitted up to arity 8.
- `claspr/src/compile.rs` — runtime SpirvBuilder wrapper. `claspr-build` duplicates this code (intentional for now — we'd need to factor out a `claspr-compile` crate to dedupe, deferred).
- `examples/raymarch/src/main.rs` — the larger single-source example, useful as a working reference for patterns: cfg-gated `use spirv_std::num_traits::Float`, image kernels, host-side validation via `gpu::pixel_color`.

## Conventions

- **Toolchain**: pinned at `nightly-2026-04-11` (rust-gpu's pinned nightly) via `rust-toolchain.toml`. Same channel as `bricevideau-ai/rust-gpu opencl-kernel-support` and `rust-gpu-opencl-samples`. Bump in lockstep with rust-gpu.
- **License**: BSD-3-Clause, copyright `Brice Videau, Argonne National Laboratory`. One `LICENSE` file at root, no per-file headers.
- **Lint before push**: `cargo fmt --all` + `cargo clippy --all-targets -- -D warnings`. CI not yet wired.
- **No CHANGELOG / release notes** — too early.

## Build / test / run

All examples need an OpenCL runtime. On this machine, pocl is at `~/.local`:

```bash
# Whole-workspace check
cargo build
cargo clippy --all-targets -- -D warnings

# Run an example
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo run -p collatz-example
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo run -p raymarch-example   # writes raymarch.ppm

# Tests (skip silently if no OpenCL device)
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo test -p collatz-example
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo test -p raymarch-example
```

On macOS the system OpenCL framework picks up automatically; no `OCL_ICD_VENDORS` needed (the Rust opencl3 binding goes through Apple's `OpenCL.framework`).

## Common gotchas

- **Don't put host-only `use` statements outside a `#[claspr::device]` mod and inside something the build script lifts.** The whole-file filter only sees the markers — anything with no marker at top level is host-only and dropped from the kernel crate. If you put `use claspr::Context` inside `mod gpu { ... }`, it'll get copied into the kernel crate and fail to resolve.
- **`use spirv_std::spirv;` inside a device module** is what makes `#[spirv(kernel)]` resolve in the *generated kernel crate*. Currently we cfg-gate it to `target_arch = "spirv"` so the host build doesn't need spirv-std as a dep — only required if the host crate doesn't already depend on spirv-std for other reasons (e.g. for `cl::Float3` host arms in raymarch).
- **Builtin param types are never name-resolved on host.** The `_id: ::glam::USizeVec3` in a `#[spirv(global_invocation_id)]` param doesn't require glam to be a host dep — the proc-macro discards the whole parameter before name resolution touches the type. Same trick gets us out of needing glam as a host dep when only the kernel uses it.
- **Generated kernel `Cargo.toml` must have `[workspace]`** — without it, cargo tries to associate the OUT_DIR-located crate with the host workspace and refuses (`current package believes it's in a workspace when it's not`).
- **`Kernel` field name == method name** on `Kernels`. The generated impl block adds a method named after the kernel; the field is also named after the kernel. Rust disambiguates by syntax (`kernels.foo` is the field, `kernels.foo(...)` is the method). Don't try to make them different.

## Backlog (deferred)

- **Auto-dependency-tracking codegen** — walk the call graph from each `#[claspr::kernel]` body, transitively pull in only the local fns it actually references. Would eliminate per-fn `#[claspr::device]` markers entirely. Module-level marker would still be useful as the "everything in here" boundary. Non-trivial: needs body parsing + cross-module call resolution + local-vs-external distinction.
- **Inherit `spirv-std` / `glam` dep specs from the host workspace** rather than hardcoding the bricevideau-ai/rust-gpu branch in the generated `Cargo.toml`. Probably read from the host's `Cargo.toml` at build time.
- **Image format dispatch in the proc-macro** — today every `&Image!(...)` maps to `&Image2DRgba8`. Need to read the macro's tokens (`type=u32, sampled=false`) and dispatch when other formats land.
- **Refactor `claspr::compile` and `claspr_build::compile_*`** to share the SpirvBuilder wrapping. Currently duplicated — fine for now, but extract a `claspr-compile` crate when the duplication grows.
- **More samples through single-source**: subgroup ops + workgroup memory (reduce sample), f64 (nbody), printf, sampler-based image reads. Each will likely surface a small claspr-build / proc-macro extension.

## Sibling repos

- `bricevideau-ai/rust-gpu` (`opencl-kernel-support` branch) — the rust-gpu fork with the kernel target. claspr-build's generated kernel `Cargo.toml` pins this.
- `bricevideau-ai/rust-gpu-opencl-samples` — the standalone-kernel-crate samples. Most patterns claspr generalises came from there; the raytracer port in `examples/raymarch/` mirrors the original `kernels/raymarch/` source.
