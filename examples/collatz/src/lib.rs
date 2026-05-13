//! End-to-end example for claspr's build-time codegen.
//!
//! The build script compiles the sibling `kernels/collatz` crate to
//! OpenCL SPIR-V and writes a generated module to `OUT_DIR`. That
//! module — `Kernels::load(&ctx)` plus the SPV bytes — is included
//! here so downstream code (and integration tests) can
//! `use collatz_example::Kernels`.

include!(concat!(env!("OUT_DIR"), "/collatz_kernels.rs"));
