# claspr

Single-source OpenCL with [rust-gpu](https://github.com/Rust-GPU/rust-gpu) — kernel and host code in one Rust source file, with type-safe kernel launches.

> **Status:** working end to end. Single-source mode (`#[claspr::device] mod` + `#[claspr::kernel]`) drives both example crates (collatz, raymarch) on pocl 7.2-pre / aarch64. The runtime helper layer (`claspr`), build-script codegen (`claspr-build`), and proc-macro frontend (`claspr-macros`) are all live. APIs are still volatile — early iteration; see [Limitations](#limitations).

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
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels.rs");
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    claspr_build::compile_from_host(&src)
        .opencl12()
        .write_to(&out)
        .unwrap();
}
```

That's the whole thing. The kernel function lives once, in the `#[claspr::device] mod`. claspr-build extracts that module's body into a generated kernel sub-crate compiled by rust-gpu; `#[claspr::device]` injects an `include!()` of the build-script-generated `Kernels` struct *inside* the device module (so it's scoped to `gpu::Kernels`) plus a `pub fn kernels(&ctx) -> Result<Kernels>` convenience wrapper; the proc-macro emits a typed launch method on `Kernels` for each `#[claspr::kernel]` function (resolving to the device module's local `Kernels` by default). The host can also call `gpu::collatz(...)` for validation — same source, two consumers. Multiple `#[claspr::device]` modules in the same file each get their own `Kernels` (no collisions).

## Workspace layout

| Crate | Role |
|-------|------|
| `claspr/` | Runtime helper library: `Context`, `DeviceSlice<T>`, `KernelArgs` tuples, `Image2DRgba8`, the `compile()` builder around `SpirvBuilder`. Re-exports `claspr_macros::{kernel, device}`. |
| `claspr-build/` | Build-script library — `compile_from_host(src_file)` reads a host source, lifts `#[claspr::kernel]` / `#[claspr::device]` items into a generated kernel sub-crate, compiles via rust-gpu, emits the `Kernels` struct. |
| `claspr-macros/` | Proc-macros: `#[kernel(kernels = path::to::Kernels)]` (typed launch method on the `Kernels` struct, kernel-style param signature) and `#[device]` (module or fn marker — the build script copies these into the kernel crate). |
| `examples/collatz/` | One-file demo: kernel + host validation in `src/main.rs`. |
| `examples/raymarch/` | Larger demo: ~21 consts + 9 helpers + 1 image kernel; SDF ray-march with sun lighting + soft shadows. Writes `raymarch.ppm`. |

## Running the examples

```bash
# Single-element collatz; validates kernel output against host implementation
cargo run -p collatz-example
# → "collatz: device/host agreement on 1024 elements"

# 1280×720 ray-marched SDF scene → raymarch.ppm
cargo run -p raymarch-example
# → "raymarch: wrote raymarch.ppm (1280x720, host validation passed at 81 pixels)"
```

Both need an OpenCL runtime. With pocl installed under `~/.local`:
```bash
OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo run -p collatz-example
```
On macOS, the system OpenCL framework is picked up automatically (no `OCL_ICD_VENDORS` needed).

## Three layers, three commits-worth of work

1. **Runtime helper crate (`claspr`)** — generalises the `OclContext` / `DeviceSlice` / `KernelArg` / `compile_kernel*` patterns from [rust-gpu-opencl-samples](https://github.com/bricevideau-ai/rust-gpu-opencl-samples) into a reusable library. Untyped launch surface (`ctx.launch(&kernel, grid, args)`) — useful on its own.
2. **Build-script codegen (`claspr-build`)** — turn a host source file into a generated kernel crate + a `Kernels` struct with one field per entry point. Two flavours: `compile()` for the kernel-crate-as-separate-folder workflow, and `compile_from_host()` for the in-host source extraction that single-source mode uses.
3. **Proc-macro frontend (`claspr-macros`)** — `#[claspr::kernel(kernels = …)]` emits an `impl Kernels { fn name(...) }` typed launch method whose signature mirrors the kernel's (with `&mut [T]` translated to `&DeviceSlice<T>`, builtin-tagged params dropped, etc.). `#[claspr::device]` marks individual fns, or whole modules, for inclusion in the generated kernel sub-crate.

Each layer is usable on its own: a project that only wants the runtime can use `claspr::compile()` without ever touching the build script or proc-macros.

## Limitations

- **rust-gpu fork pinned**: depends on `bricevideau-ai/rust-gpu` branch `opencl-kernel-support` (kernel target + everything that goes with it; not yet upstreamed). The generated kernel sub-crate's `Cargo.toml` hardcodes the same branch — should eventually inherit from the host workspace.
- **Image format hardcoded to RGBA8**: the proc-macro maps any `&Image!(...)` param to `&claspr::Image2DRgba8`. Adding `R32f` etc. is a small dispatch but not yet wired.
- **Filter is opt-in by attribute**: items not marked `#[claspr::kernel]` / `#[claspr::device]` are dropped from the kernel crate. Auto-call-graph extraction (kernel-side helpers picked up automatically from the kernel body's references) is on the eventual TODO.
- **Limited samples**: only collatz (slice + scalar args) and raymarch (image kernel + globals + many helpers) so far. Subgroup ops, workgroup memory, sampler reads, custom struct args, f64 — none ported through the single-source path yet.
- **APIs are early**: `Result` is `Box<dyn Error>`, presets / capability handling are minimal, error messages from the build script are blunt.

## Prior art and inspiration

- [rust-gpu](https://github.com/Rust-GPU/rust-gpu) — the SPIR-V codegen backend
- [krnl](https://github.com/charles-r-earp/krnl) — closest analog: proc-macro single-source for Vulkan compute
- [cust](https://github.com/Rust-GPU/Rust-CUDA) — typed launch wrappers for CUDA-Rust
- [rust-gpu-opencl-samples](https://github.com/bricevideau-ai/rust-gpu-opencl-samples) — the runtime patterns claspr generalises

## License

BSD-3-Clause
