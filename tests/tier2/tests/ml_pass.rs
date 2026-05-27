//! ML-style multi-stage chains — spike scenarios 4 + 7.
//!
//! Scenario 4: forward-pass shape with state carried through stages
//! (input → layer1 → layer2 → loss). Validates that intermediate
//! buffers thread through `.and_then` correctly and the final
//! reduction sees the cumulative effect.
//!
//! Scenario 7: multi-producer / single-consumer via `bundle3` joined
//! by a downstream combine kernel. Three independent uploads run in
//! parallel; their outputs feed into a single `add_u32` then a final
//! `scale_u32`.

use claspr::Context;
use claspr_async::{DeviceOperation, DeviceOperationHostExt, bundle, download, upload, value};
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

#[test]
fn forward_pass_threads_buffer_through_three_stages() {
    // Mimics ML forward pass: input → scale by 2 (layer1) → scale by
    // 3 (layer2) → host reduction (loss). Each device stage takes the
    // buffer by value and returns it; the host reduction at the end
    // sees the cumulative scale (1 * 2 * 3 = 6).
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let loss: u32 = upload(vec![1u32; N])
        .and_then(|buf| kernels.scale_u32([N], buf, 2)) // layer1
        .and_then(|buf| kernels.scale_u32([N], buf, 3)) // layer2
        .and_then(download)
        .and_then_host(|out| Ok(out.iter().sum::<u32>())) // loss
        .sync(&ctx)
        .expect("forward pass");
    assert_eq!(loss, 6 * N as u32);
}

#[test]
fn forward_pass_carries_scalar_state_via_value_tuple_repack() {
    // Spike scenario 4's wordy tuple-pack/unpack pattern: thread an
    // out-of-band scalar (e.g. a learning rate, a step counter)
    // through the chain alongside the buffer.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let (final_sum, step) = upload(vec![10u32; N])
        .and_then(|buf| {
            // Pack: device op output + an external scalar travel
            // together as a tuple. Tuple-repack at every stage is the
            // documented cost.
            kernels
                .scale_u32([N], buf, 2)
                .and_then(|buf| value((buf, 1u32)))
        })
        .and_then(|(buf, step)| {
            kernels
                .scale_u32([N], buf, 3)
                .and_then(move |buf| value((buf, step + 1)))
        })
        .and_then(|(buf, step)| {
            download(buf).and_then_host(move |v| Ok((v.iter().sum::<u32>(), step)))
        })
        .sync(&ctx)
        .expect("stateful forward pass");
    assert_eq!(final_sum, 60 * N as u32);
    assert_eq!(step, 2);
}

#[test]
fn mpsc_three_producers_into_single_combine() {
    // Three independent producers (upload + fill) feed a downstream
    // combine kernel (add_u32). Producers run via `bundle!`; combine
    // is sequential. Final stage: scale the combined buffer.
    //
    // Validates that bundle's parallel branches each get their own
    // chain-deps and that the combiner sees all three buffers.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let producers = bundle!(
        upload(vec![0u32; N]).and_then(|buf| kernels.fill_u32([N], buf, 3)),
        upload(vec![0u32; N]).and_then(|buf| kernels.fill_u32([N], buf, 4)),
        upload(vec![0u32; N]),
    );

    let result: Vec<u32> = producers
        .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
        .and_then(|(_a, _b, out)| kernels.scale_u32([N], out, 5))
        .and_then(download)
        .sync(&ctx)
        .expect("mpsc chain");
    // (3 + 4) * 5 = 35
    assert!(result.iter().all(|&v| v == 35));
}
