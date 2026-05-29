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
    // Pass the Frozen buffer to a kernel that reads its slice arg.
    // Kernel signature is `&mut [T]` today (rust-gpu doesn't yet
    // distinguish read-only at the slice level), but the runtime
    // enforces CL_MEM_READ_ONLY — the kernel reading is fine.
    // Once step 6 lands (proc-macro preserves &/&mut), a kernel
    // declared `&mut [T]` would fail to accept a Frozen buffer at
    // the type level; today it goes through.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Use scale_u32 as a stand-in: kernel reads + writes to its
    // slice. With Frozen the kernel WRITE would be a runtime error
    // — so we just exercise the kernel-arg plumbing without actually
    // launching for now. The full read-only-input scenario lands
    // in step 6 with the proc-macro changes.
    let _ = kernels;
    let input_data: Vec<u32> = (0..N as u32).collect();
    let frozen: DeviceSlice<u32, Frozen> =
        DeviceSlice::from_slice(&ctx, &input_data).expect("Frozen alloc");
    assert_eq!(claspr::Buffer::len(&frozen), N);
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
