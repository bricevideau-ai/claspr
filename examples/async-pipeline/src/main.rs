//! claspr-async: multi-stage forward pass.
//!
//! Demonstrates the Tier 2 chain shape on a small ML-style pipeline:
//!
//! ```text
//!    upload!(input)
//!      → linear(weight, bias)            // y = weight·x + bias
//!      → relu_threshold(threshold)        // y = max(0, y) above threshold
//!      → linear(weight, bias)             // second linear stage
//!      → download
//!      → host: sum the output             // "loss"
//! ```
//!
//! Each stage takes the buffer by value and returns it, so the chain
//! threads through without per-stage `with_context` boilerplate. The
//! final reduction uses [`and_then_host`][ath] — `and_then` would
//! drop the Vec while the non-blocking read is still writing into it.
//!
//! Verifies the device computation against an identical host
//! implementation.
//!
//! [ath]: claspr_async::DeviceOperationHostExt::and_then_host

use claspr::Context;
use claspr_async::{DeviceOperation, download, upload};

const N: usize = 256;
// Layer 1
const W1: u32 = 3;
const B1: u32 = 10;
// Activation threshold (any element strictly less than this gets zeroed)
const THRESHOLD: u32 = 25;
// Layer 2
const W2: u32 = 2;
const B2: u32 = 1;

#[claspr::device]
pub mod gpu {
    /// Linear stage: `data[i] = data[i] * weight + bias`.
    #[claspr::kernel]
    pub fn linear(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        weight: u32,
        bias: u32,
    ) {
        let i = id.x;
        data[i] = data[i].wrapping_mul(weight).wrapping_add(bias);
    }

    /// Activation: zero out any element below `threshold`.
    #[claspr::kernel]
    pub fn relu_threshold(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        threshold: u32,
    ) {
        let i = id.x;
        if data[i] < threshold {
            data[i] = 0;
        }
    }
}

/// Host reference matching the device pipeline. Returns the loss
/// (sum of post-pipeline values).
fn host_forward(input: &[u32]) -> u32 {
    input
        .iter()
        .map(|&x| {
            let a = x.wrapping_mul(W1).wrapping_add(B1);
            let a = if a < THRESHOLD { 0 } else { a };
            a.wrapping_mul(W2).wrapping_add(B2)
        })
        .sum::<u32>()
}

fn run(ctx: Context) -> claspr::Result<()> {
    let kernels = gpu::kernels(&ctx)?;

    // Input: 1..=N (mixes elements above and below the threshold
    // after layer 1, so the activation actually does something).
    let input: Vec<u32> = (1..=N as u32).collect();
    let expected = host_forward(&input);

    // The whole pipeline as one chain. `.sync(&ctx)` runs it on the
    // per-device default OOO queue; switch to `.run(&ctx).await` to
    // do the same work asynchronously.
    //
    // The reduction is a host sum on the downloaded Vec — under the
    // async `and_then_host`, in-chain reductions go via Arc<Mutex<_>>
    // capture, but for "reduce after pipeline" the cleanest shape is
    // just `.sync()` then sum on the host.
    let downloaded: Vec<u32> = upload!(input)
        .and_then(|buf| kernels.linear([N], buf, W1, B1))
        .and_then(|buf| kernels.relu_threshold([N], buf, THRESHOLD))
        .and_then(|buf| kernels.linear([N], buf, W2, B2))
        .and_then(|buf| download!(buf))
        .sync(&ctx)?;
    let loss: u32 = downloaded.iter().sum();

    println!("async-pipeline: loss = {loss} (host: {expected})");
    assert_eq!(loss, expected, "device/host mismatch");
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
    fn end_to_end_matches_host() {
        let Ok(ctx) = Context::any() else {
            eprintln!("SKIP: no OpenCL device");
            return;
        };
        run(ctx).expect("pipeline run");
    }
}
