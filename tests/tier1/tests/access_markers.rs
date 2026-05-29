//! Typestate smoke tests for the access marker scheme.
//!
//! Positive cases for each constructor surface that exists today
//! (ReadWrite via `alloc`, Frozen via `from_slice`). Markers without
//! their own ctors (ReadOnly / HostReadOnly / DeviceScratch) will
//! gain coverage when their ctor surface is added — currently the
//! type-level traits are exercised indirectly via the kernel-arg
//! flow on Frozen.
//!
//! Negative cases (e.g. `frozen_buf.write(...)` should be a compile
//! error) are best exercised via compile_fail doc-tests in the
//! `claspr::access` module — easier to keep in sync with the marker
//! definitions than separate trybuild fixtures.

use claspr::{Context, DeviceSlice, Frozen};
use claspr_test_kernels::kernels;

const N: usize = 32;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

#[test]
fn readwrite_default_marker_exercises_full_surface() {
    // Default marker — entire host + kernel surface should be available.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let mut buf: DeviceSlice<u32> = DeviceSlice::alloc(&ctx, N).expect("alloc");
    buf.write(&ctx, &[1u32; N]).wait().expect("write");
    buf.fill(&ctx, 7u32).wait().expect("fill");
    let buf = kernels.scale_u32([N], buf, 3).wait(&ctx).expect("kernel");
    let mut out = vec![0u32; N];
    buf.read(&ctx, &mut out).wait().expect("read");
    assert!(out.iter().all(|&v| v == 21), "7 * 3 = 21");
}

#[test]
fn frozen_threads_through_read_position_kernel_arg() {
    // Frozen impls KernelReadable but not KernelWritable, so it
    // satisfies the &[u32] kernel param's KernelSliceReadArg<u32>
    // bound — kernel reads its bytes successfully.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let frozen: DeviceSlice<u32, Frozen> =
        DeviceSlice::from_slice(&ctx, &[6u32; N]).expect("Frozen alloc");

    // copy_u32: kernel signature is `&[u32] src, &mut [u32] dst`.
    // src is the Frozen buffer — KernelSliceReadArg<u32> works
    // because Frozen impls KernelReadable.
    let dst = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc dst");
    let (_src, dst) = kernels
        .copy_u32([N], frozen, dst)
        .wait(&ctx)
        .expect("kernel with Frozen source");

    let mut host = vec![0u32; N];
    dst.read(&ctx, &mut host).wait().expect("read");
    assert!(host.iter().all(|&v| v == 6));
}

// NOTE: the negative cases below are SUPPOSED to be compile errors.
// They're documented here as `compile_fail` doc-tests so the build
// catches regressions if a marker accidentally gains an impl it
// shouldn't.
//
// (These are not actually run as compile_fail in this file — they
// live in the `claspr::access` module rustdoc where doc-tests can be
// `compile_fail`-annotated. Listed here as the spec to mirror.)
//
// ```compile_fail
// // Frozen buffer in &mut [u32] position: rejected
// let frozen = DeviceSlice::<u32, Frozen>::from_slice(&ctx, &[0; 32]).unwrap();
// kernels.scale_u32([32], frozen, 2);  // ERROR: Frozen doesn't impl KernelWritable
// ```
//
// ```compile_fail
// // Frozen.write(...): rejected
// let mut frozen = DeviceSlice::<u32, Frozen>::from_slice(&ctx, &[0; 32]).unwrap();
// frozen.write(&ctx, &[1u32; 32]);  // ERROR: Frozen doesn't impl HostWritable
// ```
