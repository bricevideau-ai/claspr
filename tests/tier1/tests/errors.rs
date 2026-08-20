//! Real error-path coverage: errors produced by the runtime itself,
//! not manufactured by a test closure.
//!
//! Before this file, `Error::Build` appeared in tests only as a value
//! a host seam returned on purpose, `Error::Io` had zero assertions,
//! and no test ever fed malformed SPIR-V to `build_program` /
//! `load_from`. These pin the actual failure paths.

use claspr::Error;
use claspr_test_support::ctx;

/// Bytes that aren't SPIR-V at all (wrong magic) must fail at
/// `clCreateProgramWithIL` (`Error::OpenCl`) or, on runtimes that
/// defer validation, at build (`Error::Build`). Never `Ok`.
#[test]
fn build_program_rejects_garbage_bytes() {
    let Some(ctx) = ctx() else { return };
    let garbage: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03];
    let err = ctx
        .build_program(garbage)
        .expect_err("garbage bytes must not build");
    match &err {
        Error::OpenCl(_) | Error::Build { .. } => {}
        other => panic!("unexpected error variant for garbage IL: {other:?}"),
    }
    assert!(!err.to_string().is_empty(), "error must render a message");
}

/// A module with a valid SPIR-V header but a corrupted instruction
/// stream must also fail (the runtime's IL parser or its compiler
/// rejects it) — this is the closest analogue to a truncated or
/// bit-rotted .spv file in the wild.
#[test]
fn build_program_rejects_corrupted_module() {
    let Some(ctx) = ctx() else { return };
    let mut bytes = claspr_test_kernels::kernels::SPV_BYTES.to_vec();
    assert!(bytes.len() > 64, "test kernel module unexpectedly small");
    // Keep the 20-byte header (magic/version/generator/bound/schema)
    // valid; scramble the first instruction words after it.
    for b in &mut bytes[24..64] {
        *b ^= 0xFF;
    }
    let res = ctx.build_program(&bytes);
    assert!(res.is_err(), "corrupted instruction stream must not build");
}

/// The same failure through the generated constructor: `load_from`
/// must propagate, not panic.
#[test]
fn kernels_load_from_propagates_build_failure() {
    let Some(ctx) = ctx() else { return };
    let garbage: &[u8] = &[0u8; 16];
    let res = claspr_test_kernels::kernels::Kernels::load_from(&ctx, garbage);
    assert!(res.is_err(), "load_from(garbage) must be Err, not panic");
}

/// `Error::Io` from the one I/O surface (the PPM writer), produced by
/// the runtime's own `From<io::Error>` path. No device needed.
#[test]
fn ppm_write_into_missing_dir_is_io_error() {
    let err = claspr::write_ppm_rgba8("/nonexistent-claspr-test-dir/out.ppm", 2, 2, &[0u8; 16])
        .expect_err("write into a missing directory must fail");
    assert!(
        matches!(err, Error::Io(_)),
        "expected Error::Io, got {err:?}"
    );
}
