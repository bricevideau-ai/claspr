# claspr

Single-source OpenCL with [rust-gpu](https://github.com/Rust-GPU/rust-gpu) — kernel and host code in one Rust source file, with type-safe kernel launches.

> **Status:** working end to end. Single-source mode (`#[claspr::device] mod` + `#[claspr::kernel]`) drives every example on pocl 7.2-pre / aarch64. Supported today: single-file kernel modules, multi-file device modules (via `mod foo;` declarations), library-crate composition (each kernel published as its own crate, consumed from a host binary). The runtime helper layer (`claspr`), build-script codegen (`claspr-build`), and proc-macro frontend (`claspr-macros`) are all live. APIs are still volatile — early iteration; see [Limitations](#limitations).

## What it looks like

```rust
use claspr::Context;

#[claspr::device]
mod gpu {
    #[cfg(target_arch = "spirv")]
    use spirv_std::spirv;

    pub fn collatz(mut n: u32) -> Option<u32> {
        // pure Rust — runs on both the device (called from the kernel)
        // and the host (called from the validator below)
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
    let ctx = Context::new()?;
    let kernels = gpu::kernels(&ctx)?;

    let mut data: Vec<u32> = (1..=1024).collect();
    let buf = ctx.upload(&data)?;
    kernels.collatz_kernel(&ctx, [data.len()], &buf)?;
    ctx.download(&buf, &mut data)?;

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

That's the whole thing. The kernel function lives once, in the `#[claspr::device] mod`. claspr-build extracts the module's body into a generated kernel sub-crate compiled by rust-gpu; for each device module `<name>` it finds, output is written to `OUT_DIR/<name>.rs`. `#[claspr::device]` on the matching host module injects an `include!()` of that file *inside* the module (so `Kernels` ends up scoped to `gpu::Kernels`) plus a `pub fn kernels(&ctx) -> Result<Kernels>` convenience wrapper; the `#[claspr::kernel]` proc-macro emits a typed launch method on `Kernels` for each function (resolving to the local `Kernels` by default). The host can also call `gpu::collatz(...)` for validation — same source, two consumers. Multiple `#[claspr::device]` modules in the same file each map to their own `OUT_DIR/<name>.rs` and own `Kernels` — no collisions.

## Workspace layout

| Crate | Role |
|-------|------|
| `claspr/` | Runtime helper library: `Context`, `DeviceSlice<T>`, `KernelArgs` tuples, `Image2DRgba8`, `write_ppm_rgba8`, kernel/event/program re-exports from `opencl3`. Re-exports `claspr_macros::{kernel, device}`. Host-only — does *not* depend on `spirv-builder`. |
| `claspr-build/` | Build-script library — `compile_from_host(src_file)` reads a host source, lifts `#[claspr::kernel]` / `#[claspr::device]` items into a generated kernel sub-crate, compiles via rust-gpu, emits the `Kernels` struct. |
| `claspr-macros/` | Proc-macros: `#[kernel(kernels = path::to::Kernels)]` (typed launch method on the `Kernels` struct, kernel-style param signature) and `#[device]` (module or fn marker — the build script copies these into the kernel crate). |
| `examples/collatz/` | One-file demo: kernel + host validation in `src/main.rs`. |
| `examples/raymarch/` | Multi-file demo: SDF ray-march with sun lighting + soft shadows. Splits the device module across `src/main.rs` + `src/gpu/scene.rs` + `src/gpu/shading.rs`. Writes `raymarch.ppm`. |
| `examples/mandelbrot-kernel/` + `examples/sobel-kernel/` | Two **library** crates each packaging one kernel — demonstrates publishing a claspr kernel as a reusable dependency. |
| `examples/image-pipeline/` | Binary that depends on both kernel libraries above and runs them as a two-stage pipeline (mandelbrot → sobel edge detection). No `build.rs` of its own; each kernel library carries its own. |

## Running the examples

```bash
# Single-element collatz; validates kernel output against host implementation
cargo run -p collatz-example
# → "collatz: device/host agreement on 1024 elements"

# 1280×720 ray-marched SDF scene → raymarch.ppm
cargo run -p raymarch-example
# → "raymarch: wrote raymarch.ppm (1280x720, ...)"

# Two library crates composed from one binary → image-pipeline.ppm
cargo run -p image-pipeline
# → "image-pipeline: wrote image-pipeline.ppm (1280x720, mandelbrot → sobel via two library crates)"
```

Both need an OpenCL runtime. With pocl installed under `~/.local`:
```bash
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo run -p collatz-example
```
On macOS, the system OpenCL framework is picked up automatically (no `OCL_ICD_VENDORS` needed).

## Three layers

1. **Runtime helper crate (`claspr`)** — generalises the `OclContext` / `DeviceSlice` / `KernelArg` / image-and-ppm helper patterns from [rust-gpu-opencl-samples](https://github.com/bricevideau-ai/rust-gpu-opencl-samples) into a reusable library. Untyped launch surface (`ctx.launch(&kernel, grid, args)`) — useful on its own. Host-only, no rust-gpu deps.
2. **Build-script codegen (`claspr-build`)** — turns a host source file into a generated kernel crate + a `Kernels` struct with one field per entry point. Two flavours: `compile()` for the kernel-crate-as-separate-folder workflow, and `compile_from_host()` for the in-host source extraction that single-source mode uses (with multi-file support via `mod foo;` declarations following rustc's standard file-resolution rules).
3. **Proc-macro frontend (`claspr-macros`)** — `#[claspr::kernel]` emits an `impl Kernels { fn name(...) }` typed launch method whose signature mirrors the kernel's (with `&mut [T]` translated to `&DeviceSlice<T>`, builtin-tagged params dropped, image params translated to `&Image2DRgba8`). `#[claspr::device]` marks individual fns or whole modules; on a module, also injects an `include!()` of the build-script-generated `Kernels` and a `pub fn kernels(&ctx)` convenience wrapper.

## Limitations

- **rust-gpu fork pinned**: depends on `bricevideau-ai/rust-gpu` branch `opencl-kernel-support` (kernel target + everything that goes with it; not yet upstreamed). The generated kernel sub-crate's `Cargo.toml` hardcodes the same branch — should eventually inherit from the host workspace.
- **Image format hardcoded to RGBA8**: the proc-macro maps any `&Image!(...)` param to `&claspr::Image2DRgba8`. Adding `R32f` etc. is a small dispatch but not yet wired.
- **Filter is opt-in by attribute**: items not marked `#[claspr::kernel]` / `#[claspr::device]` are dropped from the kernel crate. Auto-call-graph extraction (kernel-side helpers picked up automatically from the kernel body's references) is on the eventual TODO.
- **Multi-file device modules need `#![feature(proc_macro_hygiene)]`** at the user's crate root — `mod foo;` (file modules) inside a proc-macro's input is gated on nightly (rust-lang/rust#54727). Single-file modules don't need the feature gate. Can't be auto-injected (crate-level inner attrs only live at the crate root).
- **Limited kernel patterns covered**: collatz (slice + scalar args), raymarch (image kernel + many helpers), mandelbrot + sobel (image generator + image filter, library-crate form). Subgroup ops, workgroup memory, sampler reads, custom struct args, f64 — none ported through the single-source path yet.
- **APIs are early**: `Result` is `Box<dyn Error>`, presets / capability handling are minimal, error messages from the build script are blunt.

## Prior art and inspiration

- [rust-gpu](https://github.com/Rust-GPU/rust-gpu) — the SPIR-V codegen backend
- [krnl](https://github.com/charles-r-earp/krnl) — closest analog: proc-macro single-source for Vulkan compute
- [cust](https://github.com/Rust-GPU/Rust-CUDA) — typed launch wrappers for CUDA-Rust
- [rust-gpu-opencl-samples](https://github.com/bricevideau-ai/rust-gpu-opencl-samples) — the runtime patterns claspr generalises

## License

BSD-3-Clause
