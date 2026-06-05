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

use claspr::{
    Context, DeviceScratch, DeviceSlice, DeviceSliceUninit, Frozen, HostReadOnly, ReadOnly,
};
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

    let mut buf: DeviceSlice<u32> = DeviceSlice::alloc_zero(&ctx, N).expect("alloc");
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
    let dst = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc dst");
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
        DeviceSlice::alloc_zero(&ctx, N).expect("HostReadOnly alloc");
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
        DeviceSlice::alloc_zero(&ctx, N).expect("DeviceScratch alloc");
    // Kernel-only flow: fill the scratch via kernel, then copy_u32
    // out into a ReadWrite buffer the host CAN read.
    let scratch = kernels
        .fill_u32([N], scratch, 13)
        .wait(&ctx)
        .expect("kernel fill scratch");
    let final_buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc out");
    let (_scratch, final_buf) = kernels
        .copy_u32([N], scratch, final_buf)
        .wait(&ctx)
        .expect("kernel copy out of scratch");
    let mut out = vec![0u32; N];
    final_buf.read(&mut out).wait(&ctx).expect("read");
    assert!(out.iter().all(|&v| v == 13));
}

#[test]
fn alloc_uninit_returns_wrapper_for_arbitrary_marker() {
    // alloc_uninit is now safe + has no marker bound (returns
    // DeviceSliceUninit). Verify it constructs for markers that
    // alloc_zero rejects (Frozen lacks Fillable) and that the
    // wrapper exposes only len/ctx/Debug — no host read paths.
    let Some(ctx) = ctx() else { return };
    let uninit: DeviceSliceUninit<u32, Frozen> =
        DeviceSlice::alloc_uninit(&ctx, N).expect("Frozen alloc_uninit");
    assert_eq!(uninit.len(), N);
    assert!(!uninit.is_empty());
    let _ = format!("{uninit:?}");
    // No .read() / .download!() exists on the wrapper — type-checked.
}

#[test]
fn alloc_uninit_assume_init_kernel_write_only_pattern() {
    // The intended escape-hatch flow: alloc_uninit + assume_init +
    // pass to a kernel that writes the whole buffer. For HostReadOnly
    // where alloc_zero ALSO works (via device kernel), this is the
    // "skip the redundant fill" path.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let uninit =
        DeviceSlice::<u32, HostReadOnly>::alloc_uninit(&ctx, N).expect("HostReadOnly alloc_uninit");
    // SAFETY: fill_u32 kernel writes every slot before any read.
    let buf = unsafe { uninit.assume_init() };
    let buf = kernels
        .fill_u32([N], buf, 42)
        .wait(&ctx)
        .expect("kernel fill");
    let mut out = vec![0u32; N];
    buf.read(&mut out).wait(&ctx).expect("host read");
    assert!(out.iter().all(|&v| v == 42));
}

#[test]
fn host_read_only_fill_uses_device_kernel_path() {
    // HostReadOnly: not HostWritable → FILL_STRATEGY = DeviceKernel.
    // .fill() under the hood launches claspr_fill_u32, not
    // clEnqueueFillBuffer. Verify by alloc_uninit + explicit fill
    // (skipping the alloc auto-zero) then host-read the bytes.
    let Some(ctx) = ctx() else { return };

    let uninit = DeviceSlice::<u32, HostReadOnly>::alloc_uninit(&ctx, N).expect("alloc_uninit HRO");
    // SAFETY: fill below overwrites every byte before any read.
    let mut buf = unsafe { uninit.assume_init() };
    buf.fill(0xCAFE_BABEu32)
        .wait(&ctx)
        .expect("fill via device kernel");
    let mut out = vec![0u32; N];
    buf.read(&mut out).wait(&ctx).expect("host read");
    assert!(out.iter().all(|&v| v == 0xCAFE_BABE));
}

#[test]
fn device_scratch_fill_uses_device_kernel_path() {
    // DeviceScratch: host-no-access, can't be runtime-filled. The
    // device-kernel dispatch fills it; verify via a kernel copy to
    // a HostReadable buffer.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let uninit =
        DeviceSlice::<u32, DeviceScratch>::alloc_uninit(&ctx, N).expect("alloc_uninit scratch");
    // SAFETY: fill below overwrites every byte before any read.
    let mut scratch = unsafe { uninit.assume_init() };
    scratch
        .fill(0xDEAD_F00Du32)
        .wait(&ctx)
        .expect("fill DeviceScratch via device kernel");
    // Copy out to a host-readable buffer for verification.
    let out_buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc out");
    let (_scratch, out_buf) = kernels
        .copy_u32([N], scratch, out_buf)
        .wait(&ctx)
        .expect("copy out of scratch");
    let mut out = vec![0u32; N];
    out_buf.read(&mut out).wait(&ctx).expect("host read");
    assert!(out.iter().all(|&v| v == 0xDEAD_F00D));
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
    let dst = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc dst");
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
