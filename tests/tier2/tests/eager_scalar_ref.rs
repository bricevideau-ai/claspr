//! Scalar-by-reference kernel args (`#[spirv(cross_workgroup)] &T` /
//! `&mut T`). rust-gpu lowers these to a bare pointer-to-scalar param
//! (no length operand); the claspr proc-macro treats them as
//! reusable-buffer args backed by a length-1 `DeviceSlice<T>`, setting a
//! SINGLE pointer arg (not the slice's pointer+length pair).
//!
//! Two shapes proven:
//!   - `scale_by_ref_u32(data: &mut [u32], factor: &u32)` — a `&T` READ
//!     scalar-ref, interleaved with a slice arg (arg-index alignment).
//!   - `write_scalar_u32(out: &mut u32, val: u32)` — a `&mut T` OUTPUT
//!     scalar-ref that threads to `Output` and is host-readable.

use claspr::Context;
use claspr::DeviceSlice;
use claspr::eager::{DeviceOpExt, download, upload};
use claspr_test_kernels::kernels;

const N: usize = 64;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

/// `&u32` read scalar-ref: a length-1 `DeviceSlice<u32>` bound to the
/// `factor` param scales every element of `data`. Proves the pointer-only
/// arg is set at the right index (the slice arg before it stays intact).
#[test]
fn scale_by_ref_reads_scalar_from_len1_buffer() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let factor = DeviceSlice::<u32>::from_slice(&ctx, &[3u32]).expect("alloc factor");

    let result = upload(vec![7u32; N])
        .and_then(|data| kernels.scale_by_ref_u32([N], data, factor))
        // Both buffer args thread to Output (like a slice arg — read or
        // write); select the scaled `data` pipe, drop `factor`.
        .and_then(|(data, _factor)| download(data))
        .sync(&ctx)
        .expect("scale-by-ref chain");
    // `data[i] *= *factor` → 7 * 3 = 21.
    assert!(result.iter().all(|&v| v == 21), "got {result:?}");
}

/// `&mut u32` output scalar-ref: the kernel writes `val` through a
/// length-1 `DeviceSlice<u32>` bound to `out`. Proves the &mut scalar-ref
/// threads through `Output` and the written value is host-readable.
#[test]
fn write_scalar_threads_to_output() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let out = DeviceSlice::<u32>::alloc_zero(&ctx, 1).expect("alloc out");

    let result = kernels
        .write_scalar_u32([1usize], out, 12345u32)
        .and_then(download)
        .sync(&ctx)
        .expect("write-scalar chain");
    assert_eq!(*result, vec![12345u32]);
}

/// Tier-1 terminal on the same &mut scalar-ref kernel — the launcher's
/// `.wait()` returns the written buffer directly (no graph).
#[test]
fn write_scalar_tier1_wait() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let out = DeviceSlice::<u32>::alloc_zero(&ctx, 1).expect("alloc out");
    let out = kernels
        .write_scalar_u32([1usize], out, 999u32)
        .wait()
        .expect("write_scalar wait");

    let mut host = vec![0u32; 1];
    out.read(&mut host).wait().expect("read");
    assert_eq!(host, vec![999u32]);
}
