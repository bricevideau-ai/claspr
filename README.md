# claspr

Single-source OpenCL with [rust-gpu](https://github.com/Rust-GPU/rust-gpu) — host and device code in one Rust project, with type-safe kernel launches.

> **Status:** very early scaffolding. Nothing builds yet.

## Why

Today, writing an OpenCL compute kernel in Rust with rust-gpu means:

- A separate `dylib` crate per kernel module, targeted at `spirv-unknown-opencl*`
- A host crate that calls `SpirvBuilder` to compile each kernel crate to SPIR-V bytes
- Hand-written `KernelArg` plumbing to marshal arguments
- Kernel names as strings, arguments untyped at the launch site

All of that boilerplate is mechanical. claspr aims to collapse it into:

```rust
#[claspr::kernel]
fn collatz(
    #[global_invocation_id] id: USizeVec3,
    data: &mut [u32],
) {
    let i = id.x as usize;
    data[i] = collatz_seq_len(data[i]).unwrap_or(u32::MAX);
}

fn main() -> claspr::Result<()> {
    let ctx = claspr::Context::default()?;
    let mut buf = ctx.upload(&data)?;
    collatz.launch(&ctx, [n], &mut buf)?;   // typed, kernel name resolved at compile time
    ctx.download(&buf, &mut data)?;
    Ok(())
}
```

…with the proc-macro splitting the source into a generated device crate (compiled by rust-gpu) and a host-side typed launch wrapper, transparently.

## Roadmap

The goal above is non-trivial. We get there in three stages, each useful on its own:

1. **Runtime helper crate (`claspr`)** — generalises the `OclContext` /
   `DeviceSlice` / `KernelArg` patterns from
   [rust-gpu-opencl-samples](https://github.com/bricevideau-ai/rust-gpu-opencl-samples)
   into a reusable library. Kernels still live in their own `dylib` crate;
   this just removes the per-project host boilerplate. Untyped launch
   surface (kernel name as a string).

2. **Build-time codegen (`claspr-build`)** — a `build.rs` helper that
   runs `SpirvBuilder` on a kernel crate, parses the resulting SPIR-V
   module (entry points, parameter types), and emits a generated module
   of typed launch wrappers. Kernel author writes plain
   `#[spirv(kernel)]`; host author calls `kernels::collatz(&ctx, [n], &mut buf)`
   with full compile-time argument checking. No proc-macros.

3. **Proc-macro single-source (`claspr-macros`)** — `#[claspr::kernel]`
   on a function in the host crate. The macro materializes a generated
   device sub-crate under `target/`, runs (2) on it at build time, and
   exposes the typed wrapper as a callable item with the same name as
   the source function. Host and device code share one source file.

Each layer builds on the previous. The public-facing API at the top of
this README is the stage 3 surface; until then, users call into stages
1 and 2 directly.

## Prior art and inspiration

- [rust-gpu](https://github.com/Rust-GPU/rust-gpu) — the SPIR-V codegen backend
- [krnl](https://github.com/charles-r-earp/krnl) — closest analog: proc-macro single-source for Vulkan compute
- [cust](https://github.com/Rust-GPU/Rust-CUDA) — typed launch wrappers for CUDA-Rust
- [rust-gpu-opencl-samples](https://github.com/bricevideau-ai/rust-gpu-opencl-samples) — the runtime patterns claspr generalises

## License

BSD-3-Clause
