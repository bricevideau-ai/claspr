//! Typestate smoke tests for the access marker scheme.
//!
//! One positive-flow test per named marker — exercises the ctor +
//! the methods the marker's traits should permit + the kernel-arg
//! direction (read vs read-write) that fits.
//!
//! Negative cases (e.g. `frozen_buf.write(...)` should be a compile
//! error) live as compile_fail doc-tests in the `claspr::access`
//! module — easier to keep in sync with the marker definitions than
//! separate trybuild fixtures.

use claspr::{Context, DeviceScratch, DeviceSlice, Frozen, HostReadOnly, ReadOnly};
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
    buf.write(&[1u32; N]).wait(&ctx).expect("write");
    buf.fill(7u32).wait(&ctx).expect("fill");
    let buf = kernels.scale_u32([N], buf, 3).wait(&ctx).expect("kernel");
    let mut out = vec![0u32; N];
    buf.read(&mut out).wait(&ctx).expect("read");
    assert!(out.iter().all(|&v| v == 21), "7 * 3 = 21");
}

#[test]
fn read_only_kernel_constant_host_can_update_via_write() {
    // ReadOnly: kernel can only read; host has full RW. Construct
    // via from_slice (initial data), pass to kernel as read-only
    // input, host updates via write() between launches.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let ro: DeviceSlice<u32, ReadOnly> =
        DeviceSlice::from_slice(&ctx, &[2u32; N]).expect("ReadOnly alloc");

    // Pass as the read input of copy_u32 (signature: src: &[u32],
    // dst: &mut [u32]). ReadOnly satisfies KernelSliceReadArg.
    let dst = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc dst");
    let (ro, dst) = kernels
        .copy_u32([N], ro, dst)
        .wait(&ctx)
        .expect("kernel with ReadOnly source");
    let mut out = vec![0u32; N];
    dst.read(&mut out).wait(&ctx).expect("read dst");
    assert!(out.iter().all(|&v| v == 2));

    // Host updates the ReadOnly buffer via write() — HostWritable.
    let mut ro = ro;
    ro.write(&[9u32; N]).wait(&ctx).expect("host write");
    // Run again with the updated bytes.
    let (_ro, dst) = kernels
        .copy_u32([N], ro, dst)
        .wait(&ctx)
        .expect("kernel re-launch");
    dst.read(&mut out).wait(&ctx).expect("read dst again");
    assert!(out.iter().all(|&v| v == 9));
}

#[test]
fn host_read_only_kernel_writes_host_inspects() {
    // HostReadOnly: kernel RW, host can only read. Construct via
    // alloc (zero-init via fill — kernel is writable so fill works).
    // Kernel writes; host reads.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let hro: DeviceSlice<u32, HostReadOnly> =
        DeviceSlice::alloc(&ctx, N).expect("HostReadOnly alloc");
    // Kernel writes to it (fill_u32 has `&mut [u32]` data param —
    // HostReadOnly satisfies KernelSliceReadWriteArg).
    let hro = kernels
        .fill_u32([N], hro, 11)
        .wait(&ctx)
        .expect("kernel fill");
    let mut out = vec![0u32; N];
    hro.read(&mut out).wait(&ctx).expect("host read");
    assert!(out.iter().all(|&v| v == 11));
    // Re-bind to silence the unused-warning for now.
    let _ = hro;
}

#[test]
fn device_scratch_kernel_only_no_host_access() {
    // DeviceScratch: pure intermediate. Allocate, kernel writes,
    // kernel reads it back into a HostReadable buffer for verification.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Zero-init via fill — KernelWritable, OK.
    let scratch: DeviceSlice<u32, DeviceScratch> =
        DeviceSlice::alloc(&ctx, N).expect("DeviceScratch alloc");
    // Kernel-only flow: fill the scratch via kernel, then copy_u32
    // out into a ReadWrite buffer the host CAN read.
    let scratch = kernels
        .fill_u32([N], scratch, 13)
        .wait(&ctx)
        .expect("kernel fill scratch");
    let final_buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc out");
    let (_scratch, final_buf) = kernels
        .copy_u32([N], scratch, final_buf)
        .wait(&ctx)
        .expect("kernel copy out of scratch");
    let mut out = vec![0u32; N];
    final_buf.read(&mut out).wait(&ctx).expect("read");
    assert!(out.iter().all(|&v| v == 13));
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
    dst.read(&mut host).wait(&ctx).expect("read");
    assert!(host.iter().all(|&v| v == 6));
}

// Negative cases (Frozen → &mut, Frozen.write, Frozen.fill) live as
// `compile_fail` doc-tests in `claspr::access`. Run with
// `cargo test --doc -p claspr access`.
