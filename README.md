# claspr

Single-source OpenCL with [rust-gpu](https://github.com/Rust-GPU/rust-gpu) — kernel and host code in one Rust source file, with type-safe kernel launches.

> **Status:** working end to end. The single-source pipeline (`#[claspr::device] mod` + `#[claspr::kernel]`) drives every example on pocl 7.2-pre / aarch64 and on rusticl-on-llvmpipe in CI. Two-tier API: synchronous launches (Tier 1, in `claspr`) and lazy combinator chains (Tier 2, in `claspr-async`). Supported today: single-file kernel modules, multi-file device modules (via `mod foo;` declarations), library-crate composition (each kernel published as its own crate, consumed from a host binary), coarse-grain SVM (`MappedSlice`) and fine-grain-system SVM (`USMSlice`, host `Vec<T>` straight into kernels) as first-class kernel args. APIs are still volatile — early iteration; see [Limitations](#limitations).

## What it looks like

```rust
use claspr::{Context, DeviceSlice};

#[claspr::device]
mod gpu {
    /// Pure Rust — runs on the device (called from the kernel)
    /// and on the host (called from the validator below).
    pub fn collatz(mut n: u32) -> Option<u32> {
        let mut i = 0;
        if n == 0 { return None; }
        while n != 1 {
            n = if n.is_multiple_of(2) { n / 2 } else { 3 * n + 1 };
            i += 1;
        }
        Some(i)
    }

    #[claspr::kernel]
    pub fn collatz_kernel(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
    ) {
        let i = _id.x;
        data[i] = collatz(data[i]).unwrap_or(u32::MAX);
    }
}

fn main() -> claspr::Result<()> {
    let ctx = Context::any()?;
    let kernels = gpu::kernels(&ctx)?;

    let mut data: Vec<u32> = (1..=1024).collect();
    let mut buf = DeviceSlice::alloc(&ctx, data.len())?;
    buf.write(&ctx, &data).wait()?;
    let buf = kernels.collatz_kernel([data.len()], buf).wait(&ctx)?;
    buf.read(&ctx, &mut data).wait()?;

    // Validate every element against the host implementation —
    // same `collatz` definition, two callers.
    for (i, &device) in data.iter().enumerate() {
        assert_eq!(device, gpu::collatz(i as u32 + 1).unwrap_or(u32::MAX));
    }
    Ok(())
}
```

…and a build script:

```rust
// build.rs
fn main() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    claspr_build::compile_from_host(&src).opencl12().write().unwrap();
}
```

That's the whole thing. The kernel function lives once, in the `#[claspr::device] mod`. claspr-build extracts the module's body into a generated kernel sub-crate compiled by rust-gpu; for each device module `<name>` it finds, output is written to `OUT_DIR/<name>.rs`. `#[claspr::device]` on the matching host module injects an `include!()` of that file *inside* the module (so `Kernels` ends up scoped to `gpu::Kernels`) plus a `pub fn kernels(&ctx) -> Result<Kernels>` convenience wrapper; the `#[claspr::kernel]` proc-macro emits a typed launch method on `Kernels` for each function. The host can also call `gpu::collatz(...)` for validation — same source, two consumers. Multiple `#[claspr::device]` modules in the same file each map to their own `OUT_DIR/<name>.rs` and own `Kernels` — no collisions.

The typed launcher (`kernels.collatz_kernel(...)`) takes the slice arg by value and returns it from `.wait(...)`, so you can keep using the buffer after the launch without unsafe-borrowing across the device/host boundary. The slice param is generic over `KernelSliceArg<T>` — `DeviceSlice<T>`, `MappedSlice<T>` (coarse-grain SVM), and `USMSlice<T>` (fine-grain-system SVM over a host `Vec<T>`) all flow through the same call.

## Other modes: pre-compiled and external SPIR-V

Single-source mode above is the headline, but `claspr::kernels!`
decouples the typed host API from where the SPIR-V comes from.
Two further entry points cover the cases where the kernel isn't
authored next to its caller:

### Pre-compiled SPIR-V from a separate kernel crate

When the kernel lives in its own crate (e.g., a Rust GPU library
you want to vendor, or a build that wants the SPIR-V cached
separately from host changes), drive `claspr-build` from the host's
`build.rs` and bind the resulting bytes at runtime:

```rust
// build.rs — compiles ./kernel/ to SPIR-V, writes SPV_BYTES + ENTRY_POINTS.
fn main() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .join("kernels.rs");
    claspr_build::compile("kernel").opencl12().write_to(&out).unwrap();
}
```

```rust
// src/lib.rs — declare the host-side typed surface near the call site.
mod generated { include!(concat!(env!("OUT_DIR"), "/kernels.rs")); }

claspr::kernels! {
    pub mod gpu {
        fn fill_u32(
            #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
            #[spirv(cross_workgroup)] data: &mut [u32],
            value: u32,
        );
    }
}

let kernels = gpu::Kernels::load_from(&ctx, generated::SPV_BYTES)?;
let buf = kernels.fill_u32([N], buf, 0xdead_beefu32).wait(&ctx)?;
```

See `tests/explicit-compile/` for the canonical reference shape
(build script, `kernels!` declaration, three round-trip tests).

### External SPIR-V (clang, downloaded blobs, runtime codegen)

When the SPIR-V comes from outside the Rust ecosystem — clang's
`-target spirv64`, a downloaded blob, or a runtime code-generator —
skip `claspr-build` entirely. Declare the typed surface with
`claspr::kernels!` and feed it the bytes (or a pre-built `Program`):

```rust
claspr::kernels! {
    pub mod gpu {
        fn fill_u32(
            #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
            #[spirv(cross_workgroup)] data: &mut [u32],
            value: u32,
        );
    }
}

let bytes = std::fs::read("kernels.spv")?;          // or clang, or HTTP fetch
let kernels = gpu::Kernels::load_from(&ctx, &bytes)?;
// Or, if you want to share the built program across surfaces:
let program = ctx.build_program(&bytes)?;
let kernels = gpu::Kernels::bind(program)?;
```

The trade-off vs single-source: `kernels!` signatures live in the
host crate and must match the kernel's parameter list manually —
no proc-macro discovery, no build-time kernel-arg-info validation
against the source. In exchange you get to consume SPIR-V from any
toolchain. `tests/explicit-compile/` exercises both `load_from` and
`bind` end-to-end.

## Composing kernels: the Tier 2 async chain

The same `kernels.foo(...)` method that returns a Tier 1 Op also implements `DeviceOperation`, so it composes into a lazy chain in `claspr-async`:

```rust
use claspr_async::{DeviceOperation, download, upload};

let result: Vec<u32> = upload(input)
    .and_then(|buf| kernels.linear([N], buf, W1, B1))
    .and_then(|buf| kernels.relu_threshold([N], buf, THRESHOLD))
    .and_then(|buf| kernels.linear([N], buf, W2, B2))
    .and_then(download)
    .sync(&ctx)?;
```

`.sync(&ctx)` enqueues the whole chain on the per-device out-of-order queue, the OpenCL runtime overlaps stages, and host-side work (via `.and_then_host` / `.and_then_host_with_context`) slots in without serialising through the submitting thread. `.run(&ctx)` returns a `Future` for the same chain. Other combinators: `bundle!(a, b, c)` for heterogeneous parallel composition, `items.fan_out(|i| op)` for N-way homogeneous parallelism, `DynOp<T>` for type-erased branches, `.and_then_with_context(|ec, prev| op)` when the next step needs the running context, `.on_device(&dev)` / `transfer_to_device(buf, &dev)` for non-blocking cross-device pipelines, and lazy `device_slice_alloc::<T>(N)` / `mapped_slice_alloc::<T>(N)` / `usm_slice(vec)` so temp buffers materialize at execute time. See `examples/async-pipeline` and `examples/batch-inference`.

## Workspace layout

| Crate | Role |
|-------|------|
| `claspr/` | Runtime helper library (Tier 1): `Context`, `DeviceSlice<T>` / `MappedSlice<T>` / `USMSlice<T>`, `Queue<InOrder/OutOfOrder>` + `Launcher`, `KernelArgs` tuples, `Image2D<A, F>`, `write_ppm_rgba8`, typed `Error` enum, kernel/event/program re-exports from `opencl3`. Re-exports `claspr_macros::{kernel, device}`. Host-only — does *not* depend on `spirv-builder`. |
| `claspr-async/` | Tier 2 lazy combinators: the `DeviceOperation` trait, `value`, `upload` / `download`, lazy `device_slice_alloc` / `mapped_slice_alloc` / `usm_slice`, `.and_then` / `.and_then_with_context` / `.and_then_host` / `.and_then_host_with_context`, `.on_device(&dev)` / `transfer_to_device(buf, &dev)` for cross-device routing, `bundle!` / `fan_out` / `DynOp<T>`. Composes ops into a typed, dependency-threaded graph; terminals are `.sync(&ctx)` (blocking) or `.run(&ctx)` (returns a `Future`). |
| `claspr-build/` | Build-script library — `compile_from_host(src_file)` reads a host source, lifts `#[claspr::kernel]` / `#[claspr::device]` items into a generated kernel sub-crate, compiles via rust-gpu, emits the `Kernels` struct. |
| `claspr-macros/` | Proc-macros: `#[kernel(kernels = path::to::Kernels)]` and `#[device]`. |
| `examples/collatz/` | One-file demo: kernel + host validation in `src/main.rs`. The README quickstart above. |
| `examples/raymarch/` | Multi-file demo: SDF ray-march with sun lighting + soft shadows. Splits the device module across `src/main.rs` + `src/gpu/scene.rs` + `src/gpu/shading.rs`. Writes `raymarch.ppm`. |
| `examples/mandelbrot-kernel/` + `examples/sobel-kernel/` | Two **library** crates each packaging one kernel — demonstrates publishing a claspr kernel as a reusable dependency. |
| `examples/image-pipeline/` | Binary that depends on both kernel libraries above and runs them as a two-stage pipeline (mandelbrot → sobel edge detection). No `build.rs` of its own; each kernel library carries its own. |
| `examples/async-pipeline/` | Tier 2 demo: upload → linear → relu → linear → download as one lazy chain. Inline `#[test]` validates device output against an identical host implementation. |
| `examples/batch-inference/` | Tier 2 fan-out: N independent batches in parallel via `fan_out` + `bundle!`, sharing model weights through `Arc`. |
| `examples/two-device/` | Multi-device API: `Context::for_devices()`, `Queue::on_device()`, cross-queue buffer `copy_to`, plus a sub-device partition fallback so it does something useful even on single-physical-device boxes. No kernel code. |

## Running the examples

```bash
# Single-element collatz; validates kernel output against host implementation
cargo run -p collatz-example

# 1280×720 ray-marched SDF scene → raymarch.ppm
cargo run -p raymarch-example

# Two library crates composed from one binary → image-pipeline.ppm
cargo run -p image-pipeline

# Tier 2 chains
cargo run -p async-pipeline-example
cargo run -p batch-inference-example

# Multi-device walk
cargo run -p two-device-example
```

All need an OpenCL runtime. With pocl installed under `~/.local`:
```bash
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo run -p collatz-example
```
With rusticl on llvmpipe (same setup CI uses):
```bash
OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd RUSTICL_ENABLE=llvmpipe RUSTICL_FEATURES=fp64 cargo test --workspace
```
On macOS, the system OpenCL framework is picked up automatically (no `OCL_ICD_VENDORS` needed).

**CI.** Every push to `main` and every PR runs `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and the full workspace test suite against rusticl-on-llvmpipe (installed via the `kisak-mesa` PPA). See `.github/workflows/ci.yaml`. Badge to follow once the workflow stabilises.

## Three layers

1. **Runtime helper crate (`claspr`)** — generalises the `OclContext` / `DeviceSlice` / `KernelArg` / image-and-ppm helper patterns from [rust-gpu-opencl-samples](https://github.com/bricevideau-ai/rust-gpu-opencl-samples) into a reusable library. Synchronous launch surface with two entry points: the lower-level `LaunchOp::new(...)` builder and the proc-macro-emitted `kernels.foo(...)` typed launchers. Host-only, no rust-gpu deps. Sibling crate `claspr-async` adds a lazy-combinator (Tier 2) surface on top — chains compose because every `kernels.foo(...)` Op implements `DeviceOperation`.
2. **Build-script codegen (`claspr-build`)** — turns a host source file into a generated kernel crate + a `Kernels` struct with one field per entry point. Two flavours: `compile()` for the kernel-crate-as-separate-folder workflow, and `compile_from_host()` for the in-host source extraction that single-source mode uses (with multi-file support via `mod foo;` declarations following rustc's standard file-resolution rules).
3. **Proc-macro frontend (`claspr-macros`)** — `#[claspr::kernel]` emits an `impl Kernels { fn name(...) -> Op<...> }` typed launch method whose signature mirrors the kernel's (each `#[spirv(cross_workgroup)] &mut [T]` becomes a generic `D: KernelSliceArg<T>` slot, builtin-tagged params dropped, image params translated to `Image2D<A, F>`). The emitted Op exposes both Tier 1 terminals (`.wait(&launcher)` / `.submit(&launcher)`) and a `DeviceOperation` impl for Tier 2 chains. `#[claspr::device]` marks individual fns or whole modules; on a module, also injects an `include!()` of the build-script-generated `Kernels` and a `pub fn kernels(&ctx)` convenience wrapper.

## Limitations

- **rust-gpu fork pinned**: depends on `bricevideau-ai/rust-gpu` branch `opencl-kernel-support` (kernel target + everything that goes with it; not yet upstreamed). The generated kernel sub-crate's `Cargo.toml` hardcodes the same branch — should eventually inherit from the host workspace.
- **Image format hardcoded to RGBA8 in the proc-macro**: any `&Image!(...)` param maps to `Image2D<_, R8G8B8A8Uint>`. The runtime side (`claspr::Image2D<A, F>`) is fully generic — only the macro's dispatch is the gap.
- **Filter is opt-in by attribute**: items not marked `#[claspr::kernel]` / `#[claspr::device]` are dropped from the kernel crate. Auto-call-graph extraction (kernel-side helpers picked up automatically from the kernel body's references) is on the eventual TODO.
- **Multi-file device modules need `#![feature(proc_macro_hygiene)]`** at the user's crate root — `mod foo;` (file modules) inside a proc-macro's input is gated on nightly (rust-lang/rust#54727). Single-file modules don't need the feature gate. Can't be auto-injected (crate-level inner attrs only live at the crate root).
- **Limited kernel patterns covered through single source**: collatz (slice + scalar args), raymarch (image kernel + many helpers), mandelbrot + sobel (image generator + image filter, library-crate form), the small linear / relu pipelines in `async-pipeline` and `batch-inference`. fp64 / vector / subgroup ops / sampler-based image reads / custom struct args have not yet been ported through the single-source path — coverage on those lives in rust-gpu's upstream difftest suite.
- **Build-script error messages are still terse** — fine for normal flow, blunt when something genuinely goes wrong.

## Prior art and inspiration

- [rust-gpu](https://github.com/Rust-GPU/rust-gpu) — the SPIR-V codegen backend
- [krnl](https://github.com/charles-r-earp/krnl) — closest analog: proc-macro single-source for Vulkan compute
- [cust](https://github.com/Rust-GPU/Rust-CUDA) — typed launch wrappers for CUDA-Rust
- [rust-gpu-opencl-samples](https://github.com/bricevideau-ai/rust-gpu-opencl-samples) — the runtime patterns claspr generalises

## License

BSD-3-Clause
