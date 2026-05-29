//! `DeviceSlice<T, Frozen>` — kernel-RO + host-RO buffer initialized
//! via `CL_MEM_COPY_HOST_PTR` at construction. Verifies the kernel
//! can read it and the bytes match what we baked in.

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
fn frozen_from_slice_round_trip_via_read() {
    // Set the buffer at creation time via from_slice + COPY_HOST_PTR.
    // Read back via clEnqueueReadBuffer — host can still READ a
    // Frozen buffer (just not write). Bytes should match what we
    // baked in.
    let Some(ctx) = ctx() else { return };
    let input_data: Vec<u32> = (0..N as u32).collect();
    let frozen: DeviceSlice<u32, Frozen> =
        DeviceSlice::from_slice(&ctx, &input_data).expect("Frozen alloc");
    assert_eq!(claspr::Buffer::len(&frozen), N);

    let mut out = vec![0u32; N];
    frozen.read(&ctx, &mut out).wait().expect("read frozen");
    assert_eq!(out, input_data, "Frozen bytes should match init data");
}

#[test]
fn frozen_threads_into_kernel_as_read_only_input() {
    // Pass two Frozen buffers as the `&[u32]` (read-only) inputs of
    // `add_u32`; the output goes to a ReadWrite DeviceSlice. The
    // proc-macro (after step 6) emits `KernelSliceReadArg<u32>` for
    // the `&[u32]` params, which Frozen satisfies (Frozen impls
    // KernelReadable). This compiles AND executes; the kernel reads
    // the bytes we baked in via CL_MEM_COPY_HOST_PTR.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let a: DeviceSlice<u32, Frozen> =
        DeviceSlice::from_slice(&ctx, &[3u32; N]).expect("Frozen a");
    let b: DeviceSlice<u32, Frozen> =
        DeviceSlice::from_slice(&ctx, &[5u32; N]).expect("Frozen b");
    let out = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc out");

    let (_a, _b, out) = kernels
        .add_u32([N], a, b, out)
        .wait(&ctx)
        .expect("kernel launch with Frozen inputs");

    let mut host = vec![0u32; N];
    out.read(&ctx, &mut host).wait().expect("read");
    assert!(host.iter().all(|&v| v == 8), "Frozen a + Frozen b = 8 each");
}

#[test]
fn frozen_from_vec_takes_vec_by_value() {
    // Symmetric with from_slice; the Vec is copied into the device
    // buffer at create time via CL_MEM_COPY_HOST_PTR and then can be
    // dropped — the buffer doesn't retain the host pointer.
    let Some(ctx) = ctx() else { return };
    let v: Vec<u32> = vec![42u32; N];
    let frozen: DeviceSlice<u32, Frozen> =
        DeviceSlice::from_vec(&ctx, v).expect("Frozen alloc from Vec");
    assert_eq!(claspr::Buffer::len(&frozen), N);
}
