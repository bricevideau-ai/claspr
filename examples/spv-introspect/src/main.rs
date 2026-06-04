//! Runtime SPIR-V introspection demo.
//!
//! Loads a SPIR-V binary (the embedded demo, or one passed via
//! `argv[1]`), builds an OpenCL program from it, and walks every
//! kernel entry point — for each one, queries
//! `clGetKernelInfo`(`CL_KERNEL_NUM_ARGS`) and
//! `clGetKernelArgInfo`(`*_NAME`, `*_TYPE_NAME`,
//! `*_ADDRESS_QUALIFIER`, `*_ACCESS_QUALIFIER`) and dumps the result
//! as a readable arg table.
//!
//! Showcases the path the recent `claspr::kernels!` reshape opened:
//! a host can *receive* SPIR-V at runtime (read from disk, fetched
//! over the network, etc.) and meaningfully describe its kernels
//! without any compile-time typed wrapper.
//!
//! ## What you'll see across ICDs
//!
//! `clGetKernelArgInfo` requires the program to have been built
//! with `-cl-kernel-arg-info` (we pass it). Even with the flag,
//! what comes back depends on:
//!
//! 1. **The ICD's implementation** of arg-info recovery from
//!    SPIR-V binaries.
//! 2. **What the SPIR-V actually carries** — types are always
//!    encoded in the type system; names live in optional `OpName`
//!    decorations.
//!
//! On the embedded demo (compiled by rust-gpu's OpenCL Kernel
//! target), the observed matrix is:
//!
//! | ICD                  | address | access | type      | name       |
//! |----------------------|---------|--------|-----------|------------|
//! | rusticl/llvmpipe     | ✓       | n/a    | `<n/a>`   | `<n/a>`    |
//! | PoCL 7.2-pre (PR#2166)| ✓      | n/a    | ✓ (`int*`, `long`) | `<empty>` |
//! | Intel NEO (legacy)   | ✓       | n/a    | ✓         | `<empty>`  |
//!
//! Names are universally missing because rust-gpu's Kernel-target
//! emission doesn't currently produce `OpName` instructions for
//! kernel arguments — even PoCL's [PR #2166][pocl-pr], which
//! recovers names from `llvm::Argument`, has nothing to read.
//! Type recovery on PoCL ≥ 7.2 / NEO is real and useful.
//!
//! Note the **doubled argument count** for slice kernels: a Rust
//! `&[u32]` lowers to a `(global int*, private long)` pair at the
//! Kernel-target SPIR-V level (pointer + length). `fill_u32(data: &mut [u32], value: u32)`
//! becomes 3 args; `add_u32(a, b, out)` becomes 6.
//!
//! Switch ICDs via `OCL_ICD_VENDORS=/path/to/<icd>.icd` to compare.
//! The demo treats missing fields as soft errors and prints
//! `<not-available>` / `<empty>` rather than crashing.
//!
//! [pocl-pr]: https://github.com/pocl/pocl/pull/2166
//!
//! ## Usage
//!
//! ```text
//! # Self-contained demo (uses the embedded SPIR-V from build.rs):
//! cargo run --release -p spv-introspect-example
//!
//! # Or introspect any SPIR-V file:
//! cargo run --release -p spv-introspect-example -- path/to/kernel.spv
//! ```

use std::env;
use std::fs;
use std::process::ExitCode;

use claspr::Context;
use opencl3::error_codes::ClError;
use opencl3::kernel::Kernel;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("SKIP: no OpenCL device available");
            return Ok(());
        }
    };

    let (spv, source_label): (Vec<u8>, String) = match env::args().nth(1) {
        Some(path) => {
            let bytes = fs::read(&path).map_err(|e| format!("could not read {path}: {e}"))?;
            (bytes, path)
        }
        None => (generated::SPV_BYTES.to_vec(), "<embedded demo>".to_string()),
    };

    println!(
        "Introspecting SPIR-V module: {source_label} ({} bytes)\n",
        spv.len(),
    );

    // Build with `-cl-kernel-arg-info` so `clGetKernelArgInfo` is
    // allowed to return anything beyond the address qualifier.
    // Some ICDs (NEO) require this flag; others (PoCL) honour it
    // unconditionally. Harmless either way.
    let device = ctx.device().clone();
    let program = opencl3::program::Program::create_and_build_from_il(
        ctx.raw_context(),
        &spv,
        "-cl-kernel-arg-info",
    )
    .map_err(|e| format!("clBuildProgram failed: {e}"))?;

    // `Program::kernel_names` returns the CL standard ";"-joined
    // string. Empty when the program declared no entries.
    let names_str = program.kernel_names();
    if names_str.is_empty() {
        println!("(no kernel entry points found)");
        return Ok(());
    }
    let names: Vec<&str> = names_str.split(';').filter(|s| !s.is_empty()).collect();
    println!(
        "Program: {} (built on device `{}`)\n",
        device.name().unwrap_or_else(|_| "?".into()),
        device.name().unwrap_or_else(|_| "?".into()),
    );
    println!("Entry points ({}):", names.len());
    for name in &names {
        println!("  - {name}");
    }
    println!();

    // Per-kernel arg table.
    for name in &names {
        match Kernel::create(&program, name) {
            Ok(kernel) => print_kernel_info(name, &kernel),
            Err(e) => println!("kernel `{name}`: failed to create handle: {e}"),
        }
        println!();
    }

    Ok(())
}

fn print_kernel_info(name: &str, kernel: &Kernel) {
    println!("kernel `{name}`:");
    let n = match kernel.num_args() {
        Ok(n) => n,
        Err(e) => {
            println!("  <num_args failed: {e}>");
            return;
        }
    };
    if n == 0 {
        println!("  (no arguments)");
        return;
    }
    println!(
        "    #  {:<24}  {:<12}  {:<10}  type",
        "name", "address", "access"
    );
    println!("  {}", "-".repeat(70));
    for i in 0..n {
        let arg_name = soft_get_string(|| kernel.get_arg_name(i));
        let arg_type = soft_get_string(|| kernel.get_arg_type_name(i));
        let address = kernel
            .get_arg_address_qualifier(i)
            .map(address_qualifier_str)
            .unwrap_or_else(|_| "<?>".into());
        let access = kernel
            .get_arg_access_qualifier(i)
            .map(access_qualifier_str)
            .unwrap_or_else(|_| "<?>".into());
        println!(
            "  {:>3}  {:<24}  {:<12}  {:<10}  {}",
            i, arg_name, address, access, arg_type,
        );
    }
}

/// Many ICDs return CL_KERNEL_ARG_INFO_NOT_AVAILABLE when a kernel
/// was built from a SPIR-V binary (without `-cl-kernel-arg-info`,
/// or just because the implementation doesn't recover the info).
/// Surface that as a placeholder rather than failing the whole
/// dump.
fn soft_get_string(f: impl FnOnce() -> std::result::Result<String, ClError>) -> String {
    match f() {
        Ok(s) if !s.is_empty() => s,
        Ok(_) => "<empty>".into(),
        Err(_) => "<not-available>".into(),
    }
}

fn address_qualifier_str(q: opencl3::types::cl_uint) -> String {
    use opencl3::kernel::{
        CL_KERNEL_ARG_ADDRESS_CONSTANT, CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ADDRESS_LOCAL,
        CL_KERNEL_ARG_ADDRESS_PRIVATE,
    };
    match q {
        CL_KERNEL_ARG_ADDRESS_GLOBAL => "global".into(),
        CL_KERNEL_ARG_ADDRESS_LOCAL => "local".into(),
        CL_KERNEL_ARG_ADDRESS_CONSTANT => "constant".into(),
        CL_KERNEL_ARG_ADDRESS_PRIVATE => "private".into(),
        other => format!("?(0x{other:x})"),
    }
}

fn access_qualifier_str(q: opencl3::types::cl_uint) -> String {
    use opencl3::kernel::{
        CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_ACCESS_READ_ONLY, CL_KERNEL_ARG_ACCESS_READ_WRITE,
        CL_KERNEL_ARG_ACCESS_WRITE_ONLY,
    };
    match q {
        CL_KERNEL_ARG_ACCESS_READ_ONLY => "read-only".into(),
        CL_KERNEL_ARG_ACCESS_WRITE_ONLY => "write-only".into(),
        CL_KERNEL_ARG_ACCESS_READ_WRITE => "read-write".into(),
        CL_KERNEL_ARG_ACCESS_NONE => "n/a".into(),
        other => format!("?(0x{other:x})"),
    }
}
