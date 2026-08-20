//! End-to-end coverage for the runtime-codegen flow:
//!
//! 1. `build.rs` runs `claspr_build::compile("kernel").opencl12().write_to(OUT_DIR/kernels.rs)`,
//!    producing a generated file with `SPV_BYTES` + `ENTRY_POINTS`.
//! 2. We `include!` that file and use `claspr::kernels!` to declare
//!    the host-side typed surface (signatures live here, *next to
//!    the call site*).
//! 3. The `#[test]` actually launches the kernel against a rusticl
//!    device and reads the result back.
//!
//! ## Why this exists
//!
//! Prior to this test, the `claspr_build::compile(...)` path was
//! covered only by `cargo check` — nothing exercised
//! `.write_to(...)` end-to-end, and the typed-launcher half had no
//! reachable shape (the old `.kernel(name, params)` API didn't
//! actually generate anything). This test locks in:
//!
//! - The explicit-compile build path produces a `Kernels`-shaped
//!   generated file with correct `SPV_BYTES` + `ENTRY_POINTS`.
//! - The `claspr::kernels!` macro generates a host-side `Kernels`
//!   surface (with `bind` / `load_from` / `kernel(name)` + typed
//!   launchers) that *interoperates* with that generated file.
//! - The typed launcher actually dispatches the kernel correctly
//!   on an OpenCL device.
//!
//! If anyone changes the proc-macro's emission shape vs the build
//! script's emission shape and the two diverge, this test catches
//! it immediately — they share the same `expand_kernel` codegen
//! today, but the test makes drift impossible to introduce silently.

// The generated file from build.rs exposes `SPV_BYTES` and
// `ENTRY_POINTS`. We rebind the SPIR-V bytes to a `const` of our
// own to keep the path local, then use `claspr::kernels!` to
// declare the typed `Kernels` surface.
mod generated {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

// Single-kernel surface declared near the call site. Same shape
// `#[claspr::kernel]` would emit, but the SPIR-V binding is
// deferred to runtime: `gpu::Kernels::load_from(&ctx, &bytes)`
// accepts bytes from anywhere.
claspr::kernels! {
    pub mod gpu {
        fn fill_u32(
            #[spirv(global_invocation_id)] _id: spirv_std::glam::USizeVec3,
            #[spirv(cross_workgroup)] data: &mut [u32],
            value: u32,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claspr::DeviceSlice;
    use claspr_test_support::ctx;

    /// Embedded-bytes flow — `SPV_BYTES` is `include_bytes!`-ed
    /// into the binary by the build-script-generated file.
    /// `Kernels::load_from(&ctx, SPV_BYTES)` is the typical
    /// embedded use case.
    #[test]
    fn embedded_spv_round_trip() {
        let Some(ctx) = ctx() else { return };
        let kernels = gpu::Kernels::load_from(&ctx, generated::SPV_BYTES).expect("load kernels");

        let buf = DeviceSlice::<u32>::alloc_zero(&ctx, 64).expect("alloc");

        let buf = kernels
            .fill_u32([64usize], buf, 0xdead_beefu32)
            .wait()
            .expect("launch");

        let mut got = vec![0u32; 64];
        buf.read(&mut got).wait().expect("readback");

        assert!(
            got.iter().all(|&v| v == 0xdead_beef),
            "fill_u32 should fill every element with the value; got {got:?}",
        );
    }

    /// The runtime-loaded-bytes flow — same `Kernels` surface,
    /// SPIR-V bytes copied through `Vec<u8>` to simulate "loaded
    /// from disk / downloaded / generated at runtime". This is the
    /// shape the explicit-compile path was designed to support.
    #[test]
    fn runtime_supplied_bytes_round_trip() {
        let Some(ctx) = ctx() else { return };
        let bytes: Vec<u8> = generated::SPV_BYTES.to_vec();
        let kernels = gpu::Kernels::load_from(&ctx, &bytes).expect("load kernels");

        let buf = DeviceSlice::<u32>::alloc_zero(&ctx, 8).expect("alloc");
        let seed = vec![1u32; 8];
        let buf = buf.write(seed).wait().expect("seed write");

        let buf = kernels
            .fill_u32([8usize], buf, 7u32)
            .wait()
            .expect("launch");

        let mut got = vec![0u32; 8];
        buf.read(&mut got).wait().expect("readback");

        assert_eq!(got, vec![7u32; 8]);
    }

    /// `Kernels::bind(program)` accepts an already-built program —
    /// useful when the caller wants to keep the `Program` around
    /// or share it across multiple typed surfaces.
    #[test]
    fn bind_with_prebuilt_program() {
        let Some(ctx) = ctx() else { return };
        let program = ctx
            .build_program(generated::SPV_BYTES)
            .expect("build program");
        let kernels = gpu::Kernels::bind(&ctx, program).expect("bind");

        let buf = DeviceSlice::<u32>::alloc_zero(&ctx, 4).expect("alloc");
        let buf = buf.write(vec![1u32; 4]).wait().expect("seed write");

        let buf = kernels.fill_u32([4usize], buf, 42).wait().expect("launch");

        let mut got = vec![0u32; 4];
        buf.read(&mut got).wait().expect("readback");
        assert_eq!(got, vec![42u32; 4]);
    }

    /// `Kernels::ENTRY_POINTS` is built from the `fn` idents in the
    /// `kernels!` invocation. Verify it matches the generated
    /// file's `ENTRY_POINTS` — drift here would mean a missing
    /// kernel slipped through.
    #[test]
    fn entry_points_match_generated() {
        let macro_names: Vec<&'static str> = gpu::Kernels::ENTRY_POINTS.to_vec();
        let generated_names: Vec<&'static str> = generated::ENTRY_POINTS.to_vec();
        assert_eq!(
            macro_names, generated_names,
            "kernels! ENTRY_POINTS must mirror the build-script-generated ENTRY_POINTS",
        );
    }

    /// The build-emitter-generated `Kernels` struct is the legacy
    /// surface: `load(&ctx)` builds + binds the embedded SPIR-V,
    /// `kernel(name)` returns an untyped `claspr::Kernel`. Cover
    /// that path too so it doesn't bit-rot — the
    /// `#[claspr::device]`-emitted `pub fn kernels(ctx)` shim
    /// (rewritten in this same change) consumes the same
    /// `load_from(ctx, SPV_BYTES)` it ends up calling.
    #[test]
    fn legacy_generated_kernels_load_works() {
        let Some(ctx) = ctx() else { return };
        let k = generated::Kernels::load(&ctx).expect("legacy Kernels::load");
        // Untyped escape-hatch: get a raw `cl_kernel` by name, and
        // assert it's actually the kernel we asked for (a bare `let _`
        // here would pass even if the escape hatch handed back
        // garbage).
        let raw = k.kernel("fill_u32");
        assert_eq!(
            raw.function_name().expect("CL_KERNEL_FUNCTION_NAME"),
            "fill_u32"
        );
    }
}
