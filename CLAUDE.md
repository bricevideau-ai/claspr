# claspr — repo orientation

Single-source OpenCL with [rust-gpu](https://github.com/Rust-GPU/rust-gpu). Kernel + host code in one Rust file (or split across files), build script extracts the kernel side into a generated sub-crate compiled by rust-gpu, proc-macro emits a typed launch wrapper. Built on top of [`bricevideau-ai/rust-gpu`](https://github.com/bricevideau-ai/rust-gpu) branch `opencl-kernel-support`.

**Before changing runtime code, read `ARCHITECTURE.md`** — the layered map + a "to change X, read Y" index, so you load one subsystem (Tier 1 / op-launch / Tier 2 eager graph / CB recording) instead of the whole tree. This file (`CLAUDE.md`) covers the single-source *compilation pipeline*; `ARCHITECTURE.md` covers the *runtime library*.

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
examples/two-device/          multi-device routing (on_device_at /
                              transfer_to_device_at).
examples/async-pipeline/      .run().await async-terminal demo.
examples/batch-inference/     Tier 2 device-graph batch demo.
examples/gray-scott/          reusable-graph flagship — reaction-diffusion
                              with typed slots + bind-by-name meta-kernel
                              (run_swap mutable-replay vs run_immutable
                              curried compose, proven bit-identical).
examples/spv-introspect/      SPIR-V introspection helper demo.
examples/miniweather/         instantiate flagship — miniWeather in ONE file
                              at two precisions (device module stamped
                              f64/f32; 24-dispatch Tier-2 replay graph;
                              self-validating conservation tests).

tests/kernels, tests/image-kernels, tests/explicit-compile,
tests/instantiate, tests/tier1, tests/tier2
                           integration-test crates (also workspace members).
                           tests/instantiate is the single-source showcase
                           for #[claspr::device(instantiate(...))].
```

## How the single-source pipeline fits together

User writes one source file (e.g. `examples/collatz/src/main.rs`). It contains:

- **Top-level host code**: `use claspr::*`, `fn main`, optional `#[cfg(test)] mod tests`. No `mod compiled` — the device macro injects the include for you.
- **`#[claspr::device] mod gpu { ... }`** — the device side, in a single tagged module. Inside (user-written): kernel-only `use` statements (cfg-gated to `target_arch = "spirv"` if the host doesn't depend on those crates), `const`s, helper `fn`s, optional `mod foo;` declarations to split the module across files, and one or more `#[claspr::kernel]` entry points (defaults to `kernels = Kernels` — the relative-path `Kernels` resolves to the one the macro injects below). Inside (macro-injected, at the end of the module body): `include!(concat!(env!("OUT_DIR"), "/<modname>.rs"))` (brings `Kernels` + `SPV_BYTES` + `ENTRY_POINTS` + `Kernels::{bind,load_from,load}` in) and a `pub fn kernels(ctx) -> Result<Kernels>` convenience wrapper (which calls `Kernels::load_from(ctx, SPV_BYTES)`).
- The user does *not* need to import `spirv` from spirv-std — `claspr-build`'s preamble injects `use spirv_std::spirv;` because every translated `#[claspr::kernel]` becomes `#[spirv(kernel)]` and the `spirv` proc-macro must be in scope. Anything else (`Image`, `cl::*`, `opencl_std`, `num_traits::Float`, …) the user imports themselves.
- Calling code reads `let kernels = gpu::kernels(&ctx)?;` then `kernels.collatz_kernel([N], buf, ...).wait()?` (the typed launcher carries the context; no `&ctx` argument, and a terminal like `.wait()` / `.submit()` runs it). Multiple `#[claspr::device]` modules in the same file each scope their own `Kernels`/`kernels()` — no collisions.
- **`#[claspr::device(instantiate(Real = [f64, f32]))]`** stamps the module once
  per listed scalar type: the body is written against the placeholder (`Real`),
  each stamp becomes a nested host sub-module named after its type (`gpu::f64`,
  `gpu::f32`) with its own independently compiled SPIR-V + `Kernels` + launchers
  + `kernels()`. Both sides inject `pub type Real = <ty>;` (claspr-build into the
  stamped kernel crate, the device macro into each host stamp module) — nothing
  is textually substituted. The f64 stamp gets `Capability::Float64`
  automatically; other stamps build without fp64 permission so accidental
  widening errors out. Kernel sub-crates/outputs are named `<mod>__<ty>`.
  Showcase + tests: `tests/instantiate`. Not yet done: generated trait unifying
  the stamps (generic host drivers), per-stamp vector-family aliases, `slots!`
  generic value types.
- The build script writes one `OUT_DIR/<modname>.rs` per device module it finds — the macro's injected include matches the module ident, so module name is the only piece of coupling between the build-script side and the host source. Top-level `#[claspr::kernel]` / `#[claspr::device]` items outside any module are rejected: organise kernel code into a module so the per-module file naming has something to key off.

Two compilation paths run on the same source:

1. **Host build** (cargo's normal flow). The proc-macros do the heavy lifting:
   - `#[claspr::device]` on a fn → `#[allow(dead_code, unused_imports)] <fn>` (no semantic change beyond the warning suppression).
   - `#[claspr::device]` on a mod → re-emits the user's module with two extra items appended *inside* the body: an `include!(concat!(env!("OUT_DIR"), "/<modname>.rs"))` (brings `Kernels`/`Kernels::load` into the module's scope) and a `pub fn kernels(&ctx) -> Result<Kernels>` convenience wrapper. The whole module is wrapped in `#[allow(dead_code, unused_imports)]`.
   - `#[claspr::kernel(kernels = path)]` (path defaults to `Kernels` — relative; resolves to the device module's local `Kernels`) parses the kernel-style fn signature, drops `#[spirv(<builtin>)]` params, and translates the device-pointer/image params into *generic* host args: `#[spirv(cross_workgroup)] &mut [T]` → a generic bound by `KernelSliceRead[Write]Arg<T>` (accepts `DeviceSlice`/`MappedSlice`/`USMSlice` by value), and `&Image!(...)` → a generic bound by the matching `KernelImage<dim>D<Access>Arg<family>` trait (1D/2D/3D/Buffer, arrayed variants). It then emits `impl path { fn name(&self, grid, args...) }` (the launcher carries the context). The impl ends up inside the same module, attached to the same `Kernels` struct the include brought in. The original kernel body is discarded on the host side.
2. **Kernel build** (driven by `examples/<name>/build.rs` calling `claspr_build::compile_from_host(src).opencl12().write()`):
   - Reads the source file, parses with syn.
   - Discovers every top-level `Item::Mod` with `#[claspr::device]`. Top-level `Item::Fn` with `#[claspr::kernel]` / `#[claspr::device]` (no enclosing module) is an error — there's no module name to use for the output file.
   - For each device module, lifts its body into a fresh kernel sub-crate at `OUT_DIR/claspr_kernel_<modname>/`. `mod foo;` declarations inside the body are followed using rustc's standard file-resolution rules (`<dir>/<name>.rs` then `<dir>/<name>/mod.rs`); `cargo:rerun-if-changed` is emitted for each followed file. Inside the lifted body, translates `#[claspr::kernel(...)]` → `#[spirv(kernel)]` and strips `#[claspr::device]`. Wrapper preamble injects `#![cfg_attr(target_arch = "spirv", no_std)]`, `#![allow(unused_imports)]`, and `use spirv_std::spirv;`.
   - Writes a `Cargo.toml` for the sub-crate (with `[workspace]` so cargo doesn't try to attach it to the host workspace; spirv-std + glam deps hardcoded at the rust-gpu branch we depend on).
   - Runs `SpirvBuilder` on each sub-crate, then writes the `Kernels` struct (holding the built program + context; constructed via `bind`/`load_from`/`load`) plus one typed launcher method per entry point, to `OUT_DIR/<modname>.rs`. The matching `#[claspr::device]` on the host side `include!()`s this exact path.

## Library-crate composition

A claspr kernel can be packaged as its own library crate (`pub mod gpu`, build.rs of its own), and consumed from a host binary via `lib_name::kernels(&ctx)?.entry_point(...)`. The host binary needs no build.rs; each kernel library carries its own. Two patterns:

- **Pure kernel library** (mandelbrot-kernel, sobel-kernel today): exposes only typed launch handles. spirv-std imports inside the library's device module are cfg-gated to `target_arch = "spirv"` so consumers don't pay for spirv-std as a transitive dep. Helpers can't reference device-only types directly (the library's host build won't have those types in scope).
- **Mixed kernel + host library** (raymarch is binary-shaped today; could be a library): exposes typed launch handles AND host-callable helpers (e.g., `pixel_color` for validation). Needs spirv-std as a *regular* host dep so types like `cl::Float3` are in scope on both sides. Cost: consumers pull spirv-std transitively. Benefit: full host parity.

## Tier 2: the reusable device-operation graph (`eager.rs` + `eager/`)

Every op implements the one `DeviceOp` trait, so it runs standalone (`.wait()`) or
composes into an **eager, closure-free** graph via `.and_then` / `bundle!` / `fan_out`,
with terminals `.sync(&ctx)` / `.run(&ctx)`. Typed slots (`slots!` / `slot!`) make a
graph reusable: build once, re-bind, replay. **The full model — the closed 4-verb
`bind`/`call`/`mutate_bind`/`mutate_call` set, the 5-state `SlotState`, the home
invariant, the unified `Tag(value)`/`Tag(pipe)` constructor — lives in `ARCHITECTURE.md`
and the `eager.rs` module `//!` docs; this file does not restate it.**

**Before writing a graph-using example, read the "Writing reusable-graph host code"
usage guide in the `eager.rs` module `//!` docs** (the lend/rehome/reclaim lifecycle,
`Checkout` deref+`map`, `into_inner`-vs-`drop`, the build-once-`sync`-in-a-loop replay
idiom, device scalars). Its worked examples: `examples/gray-scott` (reusable graph —
`run_swap` mutable-replay vs `run_immutable` curried compose, proven bit-identical) and
`examples/cg` (matrix-free CG as a self-closing single-graph loop with device-resident
α/β scalars).

## Key files

- `claspr/src/eager.rs` + `eager/` — the Tier 2 graph engine, split by concern: `eager.rs` holds the core (`DeviceOp`/`DeviceOpExt`, the edge/home model, terminals, the closed 4-verb `bind`/`call`/`mutate_bind`/`mutate_call` set), and the submodules hold `combinators.rs` (AndThen/Bundle/FanOut/…), `leaves.rs` (fill/copy/transfer leaf ops), `slots.rs` (`SlotState`/`SlotBinder`), `cb.rs` (command-buffer recording). See `ARCHITECTURE.md` for the "to change X, read Y" map. Most active surface for graph work.
- `claspr/src/tier2_macros.rs` — the `slots!` / `slot!` macros, including the per-tag `Tag`, `IntoBound`, and `CallArg` impls (value + `Checkout` + `Pipe` sources) each tag emits.
- `claspr-build/src/lib.rs` — both `compile()` (separate kernel crate) and `compile_from_host()` (single-source extraction) live here. The translation logic (`translate_and_inline`, `resolve_module_file`, `is_claspr_kernel_attr`, `is_claspr_device_attr`) is the most-likely-to-change surface as new kernel patterns surface; multi-file resolution rules also live there.
- `claspr-macros/src/lib.rs` — `#[kernel]` + `#[device]`. The kernel signature → host wrapper translation lives in `classify_param`, `classify_image_param`, `slice_element_ty` (plus `parse_image_tokens`/`read_image_access_attr` for images). The device-on-mod injection (include! + `kernels()` fn) lives in the `device` proc-macro body.
- `claspr/src/launch.rs` — `KernelArg` / `KernelArgs` / `LaunchSpec` / `IntoLaunchSpec`. Tuple impls are macro-emitted up to arity 16.
- `examples/raymarch/src/main.rs` + `src/gpu/scene.rs` + `src/gpu/shading.rs` — multi-file device-module reference. Cross-file `use super::scene::...` works because the build script preserves module structure during inlining.
- `examples/image-pipeline/src/main.rs` — library-composition reference. Pulls two kernel libraries and chains their launches.

## Conventions

- **Toolchain**: pinned at `nightly-2026-05-22` (rust-gpu's pinned nightly) via `rust-toolchain.toml`. Same channel as `bricevideau-ai/rust-gpu opencl-kernel-support` and `rust-gpu-opencl-samples`. Bump in lockstep with rust-gpu.
- **License**: BSD-3-Clause, copyright `Brice Videau, Argonne National Laboratory`. One `LICENSE` file at root, no per-file headers.
- **Lint before push**: `cargo fmt --all` + `cargo clippy --all-targets -- -D warnings` + `cargo doc --no-deps --workspace` with `RUSTDOCFLAGS=-D warnings`. CI (`.github/workflows/ci.yaml`) runs the same checks plus rustfmt on `tests/*/compile_fail/*.rs` (which `cargo fmt --all` skips) and the integration test suites on rusticl/llvmpipe.
- **No CHANGELOG / release notes** — too early.

## Build / test / run

All examples need an OpenCL runtime. pocl's prefix is per-machine: the
Mac/Linux laptop installs to `~/.local` (paths below); the **Intel Linux box
installs to `~/local`** (so use `OCL_ICD_VENDORS=$HOME/local/pocl/etc/OpenCL/vendors`
there). On that box, leaving the `~/.local` path set silently falls back to the
system ICDs in `/etc/OpenCL/vendors` (which run but can flakily SIGABRT in
driver teardown) — point at `~/local/pocl` instead:

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
- **`librustc_codegen_spirv.so` collision warning** when running `cargo build --workspace`: cargo issue #6313 (workspace-level dylib build-dep collision). Building one example at a time (`cargo build -p collatz-example`) is silent. Forward-looking warning, not currently a hard error.

## Backlog (deferred)

- **Auto-dependency-tracking codegen** — walk the call graph from each `#[claspr::kernel]` body, transitively pull in only the local fns it actually references. Would eliminate per-fn `#[claspr::device]` markers entirely. Module-level marker would still be useful as the "everything in here" boundary. Non-trivial: needs body parsing + cross-module call resolution + local-vs-external distinction.
- **Auto-enable rust-gpu capabilities** — codegen is currently inconsistent about which `Capability::*` it auto-emits versus what build scripts must declare. Move toward auto-adding everything the codegen actually needs and let device-side `clBuildProgram` rejection handle "this device can't run it." Will likely shrink `CompileBuilder::capability` / `with_f64` etc.
- **Cargo-features-driven `auto()` build.rs** — move OpenCL version + remaining capability flags to Cargo features so the build.rs collapses to `claspr_build::auto()`.
- **`#[path = "..."]` attribute support in multi-file resolution** — currently errors with a clear message.
- **`cargo:warning=` for missing `proc_macro_hygiene` feature gate** — claspr-build could detect file modules in device-module bodies and check if the crate's inner attrs include the gate; warn at build time before the rustc error fires.
- **More samples through single-source**: subgroup ops + workgroup memory (reduce sample), f64 (nbody), printf, sampler-based image reads. Each will likely surface a small claspr-build / proc-macro extension.
- **`cargo claspr` subcommand** to eliminate build.rs entirely (NVlabs/cuda-oxide-style). Doesn't port cleanly because rust-gpu is a whole-crate codegen backend (no per-item dispatch); workflow friction (plain cargo stops working, IDE tooling breaks) likely makes it not worth it.

## Sibling repos

- `bricevideau-ai/rust-gpu` (`opencl-kernel-support` branch) — the rust-gpu fork with the kernel target. claspr-build's generated kernel `Cargo.toml` pins this.
- `bricevideau-ai/rust-gpu-opencl-samples` — the standalone-kernel-crate samples. Most patterns claspr generalises came from there.

## Inter-session notes

`NOTES.md` is the single rolling doc for active work, deferred items, and unresolved concerns across sessions. **Do not spawn new planning docs** (no `IMPLEMENTATION-PLAN.md`, `REVIEW-2026-XX.md`, `DESIGN-NOTES.md` etc.) — append to the matching section of `NOTES.md` instead and prune as items resolve. If a section grows past ~30 lines, that's a signal to either ship the work or summarize.

Three lanes:

- **Ongoing work** (active / deferred / concerns) → `NOTES.md`.
- **Point-in-time snapshots** (a specific deep review, a one-off audit, the rationale behind a particular change) → commit message of whatever ships as a result. Git log is the canonical history.
- **Persistent rules / cross-session preferences** (the user's lint cadence, naming conventions, workflow quirks) → auto-memory (`~/.claude/projects/-home-claudecode-projects/memory/`), not here.

The `Backlog (deferred)` section above is the long-horizon "someday" list that's stable across sessions; `NOTES.md` is for things actively in flight or recently deferred with a revisit-trigger.
