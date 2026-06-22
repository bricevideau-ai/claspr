//! Eager-API port of `ml_pass.rs`: ML-style multi-stage chains.
//!
//! Old → new mapping:
//!   `upload!(v)`        → `upload::<u32, claspr::ReadWrite, _>(v)`
//!   `download!(buf)`    → `download`
//!   `bundle!(a, b, c)`  → `bundle3(a, b, c)`
//!   `.and_then_host(f)` → `.and_then_host(f)` (DeviceSlice View is `&mut [u32]`)
//!   multi-output add_u32 → `.and_then(|(_a, _b, out)| ...)` per-element pipe
//!
//! Same N, same scale factors, same assertions as `ml_pass.rs`.

use claspr::Context;
use claspr::eager::{EagerOpExt, bundle3, download, upload};
use claspr_test_kernels::kernels;
use std::sync::{Arc, Mutex};

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
    // 3 (layer2) → host reduction (loss). The host reduction at the end
    // sees the cumulative scale (1 * 2 * 3 = 6).
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let loss_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&loss_cell);
    let _final_buf = upload::<u32, claspr::ReadWrite, _>(vec![1u32; N])
        .and_then(|buf| kernels.scale_u32([N], buf, 2)) // layer1
        .and_then(|buf| kernels.scale_u32([N], buf, 3)) // layer2
        .and_then_host(move |slice: &mut [u32]| {
            *cell.lock().unwrap() = slice.iter().sum(); // loss
            Ok(())
        })
        .sync(&ctx)
        .expect("forward pass");
    let loss = *loss_cell.lock().unwrap();
    assert_eq!(loss, 6 * N as u32);
}

// BLOCKED: forward_pass_carries_scalar_state_via_value_tuple_repack — needs the
// host-value-passthrough seam. The original packs the device buffer + an
// out-of-band scalar into `value((buf, step))` at each stage. In eager,
// `and_then` hands a `Pipe<DeviceSlice>` (not the concrete `DeviceSlice`), so
// `value((buf, step))` cannot be constructed in-graph — there is no way to lift
// the upstream buffer *value* into a fresh `value(...)` node alongside a host
// scalar. (The tuple `(DeviceSlice, u32)` IS Mappable, so the final
// `and_then_host(|(slice, _step)| ...)` would type-check; the blocker is the
// per-stage tuple repack, the same `value_passthrough` gap noted in
// `eager_chain.rs`.) See report for the needed primitive.

#[test]
fn mpsc_three_producers_into_single_combine() {
    // Three independent producers (upload + fill) feed a downstream
    // combine kernel (add_u32). Producers run via bundle3; combine is
    // sequential. Final stage: scale the combined buffer.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let producers = bundle3(
        upload::<u32, claspr::ReadWrite, _>(vec![0u32; N])
            .and_then(|buf| kernels.fill_u32([N], buf, 3)),
        upload::<u32, claspr::ReadWrite, _>(vec![0u32; N])
            .and_then(|buf| kernels.fill_u32([N], buf, 4)),
        upload::<u32, claspr::ReadWrite, _>(vec![0u32; N]),
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
