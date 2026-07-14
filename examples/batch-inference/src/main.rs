//! claspr Tier 2: N independent batches in parallel via `fan_out`,
//! each consuming the same shared weight tensor — uploaded **once**
//! to the device and shared across all branches via
//! `Arc<DeviceSlice<u32>>`.
//!
//! Pipeline:
//!
//! ```text
//!    upload!(weights) ────→ weights_dev: Arc<DeviceSlice<u32>>   // ONCE
//!
//!    for each batch (in parallel on the OOO queue):
//!      upload!(batch_input) → input_buf
//!                            ↓ Arc::clone(&weights_dev) for read-only kernel arg
//!                            elem_mul: input_buf[i] *= weights[i]
//!                            ↓
//!                            add_bias: input_buf[i] += bias
//!                            ↓
//!                            download → host: sum
//! ```
//!
//! `Arc<DeviceSlice<T>>` impls `KernelSliceReadArg` only (memory
//! `[[arc-deviceslice-readonly]]`) — exactly the right gate for a
//! shared read-only input. The previous version uploaded weights
//! per branch via `upload!(Arc::clone(&host_arc))`, which allocated
//! N distinct device buffers and N × N × 4 bytes of redundant
//! host→device DMA. The share-on-device pattern: one alloc, one
//! DMA, N branches read from the same `cl_mem`.
//!
//! ## When to upload per branch instead
//!
//! **Multi-device** `fan_out`: if each branch routes to a different
//! device (via `transfer_to_device` or `.on_device(...)`), the
//! shared weights need to live in *each* device's memory. The
//! per-branch upload pattern is correct there: each branch uploads
//! into its target device's memory. See the `two-device` example (and
//! the `eager_transfer_to_device` / `eager_on_device_suite` tests) for
//! the multi-device shape.
//!
//! Single-device `fan_out` (this example) has no such constraint
//! — all branches enqueue onto the same OOO queue on the same
//! device, so one shared device buffer suffices.
//!
//! [`fan_out`](claspr::fan_out()) enqueues every branch on the
//! chain's out-of-order queue with independent event chains; a single
//! `clEnqueueMarkerWithWaitList` joins them at the end. The OOO
//! scheduler decides how much to overlap on the device.
//!
//! Verifies every batch's output against a host reference.

use claspr::Context;
use claspr::eager::{DeviceOpExt, download, fan_out, upload};
use std::sync::Arc;

const N: usize = 64;
const BATCHES: usize = 8;
const BIAS: u32 = 100;

#[claspr::device]
pub mod gpu {
    /// Elementwise multiply: `out[i] = a[i] * b[i]`.
    #[claspr::kernel]
    pub fn elem_mul(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] a: &mut [u32],
        #[spirv(cross_workgroup)] b: &[u32],
    ) {
        let i = id.x;
        a[i] = a[i].wrapping_mul(b[i]);
    }

    /// Scalar bias add: `data[i] += bias`.
    #[claspr::kernel]
    pub fn add_bias(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        bias: u32,
    ) {
        let i = id.x;
        data[i] = data[i].wrapping_add(bias);
    }
}

fn host_inference(input: &[u32], weights: &[u32], bias: u32) -> u32 {
    input
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| x.wrapping_mul(w).wrapping_add(bias))
        .sum::<u32>()
}

fn run(ctx: Context) -> claspr::Result<()> {
    let kernels = gpu::kernels(&ctx)?;
    let kernels_ref = &kernels;

    // Host-side weight tensor (1, 2, 3, ..., N). Kept as a Vec so we
    // can also use it for the host-side reference computation below
    // (we'll then upload it ONCE to the device and share that one
    // device buffer across all branches).
    let weights: Vec<u32> = (1..=N as u32).collect();

    // Per-batch inputs: batch k contains [k, k+1, k+2, ...].
    let inputs: Vec<Vec<u32>> = (0..BATCHES)
        .map(|k| (k as u32..k as u32 + N as u32).collect())
        .collect();
    let expected: Vec<u32> = inputs
        .iter()
        .map(|inp| host_inference(inp, &weights, BIAS))
        .collect();

    // **Upload weights ONCE** to a single device buffer, then share
    // across all branches via `Arc<DeviceSlice<u32>>`. Per memory
    // `[[arc-deviceslice-readonly]]`, `Arc<DeviceSlice<T>>` impls
    // `KernelSliceReadArg` only — that's exactly what we want here:
    // every branch's `elem_mul` reads from `weights_dev` in the
    // `&[u32]` (read-only) kernel slot. The previous version did
    // `upload!(weights_clone)` per branch, allocating BATCHES distinct
    // device buffers and BATCHES × N × 4 bytes of redundant
    // host→device DMA. With the share-on-device pattern: one alloc,
    // one DMA, N branches read from the same `cl_mem`.
    let weights_dev: Arc<claspr::DeviceSlice<u32>> =
        Arc::new(upload(weights).sync(&ctx)?.into_inner());

    // The full fan_out chain: each branch uploads ONLY its own input
    // buffer, then runs `elem_mul` against the shared weights, then
    // bias-adds, then downloads. BATCHES branches run concurrently on
    // the OOO queue. Final reduction is a host-side sum after the
    // chain has finished. (Pre-async `and_then_host` could fold this
    // into the chain, but the new signature only does in-place
    // mutation — for pure reductions, host sum after `.sync()` is the
    // natural shape.)
    // `downloaded` stays a `Checkout<Vec<Vec<u32>>>`; only borrowed below.
    let downloaded = fan_out(inputs.clone(), move |input| {
        // Cheap Arc::clone — both pointers refer to the same `cl_mem`.
        let weights_ref = Arc::clone(&weights_dev);
        upload(input)
            .and_then(move |input_buf| {
                // elem_mul: `(a: &mut [u32], b: &[u32])`. `a` =
                // input_buf (consumed + returned with mul applied),
                // `b` = weights_ref (Arc, read-only KernelSliceReadArg).
                kernels_ref.elem_mul([N], input_buf, weights_ref).and_then(
                    move |(input_buf, _weights_arc)| kernels_ref.add_bias([N], input_buf, BIAS),
                )
            })
            .and_then(download)
    })
    .sync(&ctx)?;
    let outputs: Vec<u32> = downloaded.iter().map(|v| v.iter().sum()).collect();

    for (i, (got, want)) in outputs.iter().zip(expected.iter()).enumerate() {
        if got != want {
            panic!("batch {i}: device {got} != host {want}");
        }
    }
    println!(
        "batch-inference: {} batches OK (per-batch sum = {:?})",
        BATCHES, outputs,
    );
    Ok(())
}

fn main() -> claspr::Result<()> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            return Ok(());
        }
    };
    run(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_batches_match_host() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        run(ctx).expect("batch-inference run");
    }
}
