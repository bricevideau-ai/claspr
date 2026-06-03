//! claspr-async: N independent batches in parallel via `fan_out`,
//! each consuming the same shared weight tensor.
//!
//! Pipeline per batch:
//!
//! ```text
//!    upload(batch_input) ─┐
//!    upload(Arc<weights>) ┴→ pointwise: batch[i] *= weight[i]
//!                          → bias add: batch[i] += bias
//!                          → download
//!                          → host: sum
//! ```
//!
//! `weights` is held as `Arc<[u32]>` on the host; each branch uploads
//! its own copy from the shared host data via [`UploadSource::Arc`][as] —
//! the keep-alive callback on the write event drops the Arc when the
//! transfer finishes, so the runtime tracks lifetime automatically.
//!
//! [`fan_out`](claspr_async::fan_out()) enqueues every branch on the chain's out-of-order
//! queue with independent event chains; a single
//! `clEnqueueMarkerWithWaitList` joins them at the end. The OOO
//! scheduler decides how much to overlap on the device.
//!
//! Verifies every batch's output against a host reference.
//!
//! [as]: claspr_async::transfer::UploadSource

use claspr::Context;
use claspr_async::{DeviceOperation, bundle, download, fan_out, upload};
use std::sync::Arc;

const N: usize = 64;
const BATCHES: usize = 8;
const BIAS: u32 = 100;

#[claspr::device]
pub mod gpu {
    /// Elementwise multiply: `out[i] = a[i] * b[i]`.
    #[claspr::kernel]
    pub fn elem_mul(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] a: &mut [u32],
        #[spirv(cross_workgroup)] b: &[u32],
    ) {
        let i = id.x;
        a[i] = a[i].wrapping_mul(b[i]);
    }

    /// Scalar bias add: `data[i] += bias`.
    #[claspr::kernel]
    pub fn add_bias(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
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

    // Host-side shared weights (1, 2, 3, ..., N). Arc<[T]> so every
    // branch's upload borrows the same heap allocation; the runtime
    // releases the Arc when the last keep-alive callback fires.
    let weights: Arc<[u32]> = (1..=N as u32).collect::<Vec<_>>().into();

    // Per-batch inputs: batch k contains [k, k+1, k+2, ...].
    let inputs: Vec<Vec<u32>> = (0..BATCHES)
        .map(|k| (k as u32..k as u32 + N as u32).collect())
        .collect();
    let expected: Vec<u32> = inputs
        .iter()
        .map(|inp| host_inference(inp, &weights, BIAS))
        .collect();

    // The full fan_out chain: each branch is its own independent
    // upload + bundle(weights, input) + kernel + bias + download + sum.
    // BATCHES branches run concurrently on the OOO queue.
    // Each branch downloads to a Vec<u32>; final reduction is a
    // host-side sum after the chain has finished. (Pre-async
    // `and_then_host` could fold this into the chain, but the new
    // signature only does in-place mutation — for pure reductions,
    // host sum after `.sync()` is the natural shape.)
    let downloaded: Vec<Vec<u32>> = fan_out(inputs.clone(), move |input| {
        // `Arc::clone(&weights)` is cheap; both clones share the same
        // host allocation. The keep-alive callback on the write event
        // drops each clone once OpenCL is done copying from it.
        let weights_clone: Arc<[u32]> = Arc::clone(&weights);
        bundle!(upload(input), upload(weights_clone))
            .and_then(move |(input_buf, weight_buf)| {
                // elem_mul takes `(&mut [u32], &[u32])` → both slices
                // flow through as Output (3-tuple? no, 2-tuple of slices).
                // We discard the weight buffer after the mul.
                kernels_ref.elem_mul([N], input_buf, weight_buf).and_then(
                    move |(input_buf, _weight_buf)| kernels_ref.add_bias([N], input_buf, BIAS),
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
