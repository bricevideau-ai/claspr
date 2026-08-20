//! Eager-API port of `ml_pass.rs`: ML-style multi-stage chains.
//!
//! Old → new mapping:
//!   `upload!(v)`        → `upload(v)`
//!   `download!(buf)`    → `download`
//!   `bundle!(a, b, c)`  → `bundle3(a, b, c)`
//!   `.and_then_host(f)` → `.and_then_host(f)` (DeviceSlice View is `&mut [u32]`)
//!   multi-output add_u32 → `.and_then(|(_a, _b, out)| ...)` per-element pipe
//!
//! Same N, same scale factors, same assertions as `ml_pass.rs`.

use claspr::bundle;
use claspr::eager::{DeviceOpExt, bundle3, download, upload, value};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;
use std::sync::{Arc, Mutex};

const N: usize = 64;

#[test]
fn forward_pass_threads_buffer_through_three_stages() {
    // Mimics ML forward pass: input → scale by 2 (layer1) → scale by
    // 3 (layer2) → host reduction (loss). The host reduction at the end
    // sees the cumulative scale (1 * 2 * 3 = 6).
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let loss_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&loss_cell);
    let _final_buf = upload(vec![1u32; N])
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

/// ml_pass.rs::forward_pass_carries_scalar_state_via_value_tuple_repack — thread
/// an out-of-band scalar (step counter) alongside the buffer through the chain.
///
/// PORT NOTE: the original wrote `value((buf, step))` to pack the buffer + scalar
/// into one edge. Eager `value` can't pack a not-yet-resolved buffer pipe inside
/// a tuple, so the eager idiom BUNDLES the two graph members instead: the buffer
/// (a `Pipe<DeviceSlice>`, passed BARE — `Pipe<T>: DeviceOp`, no `forward(..)`)
/// and the scalar (`value(step)`, a BY-VALUE handle). `bundle2` joins them, and
/// the downstream closure receives `(Pipe<DeviceSlice>, u32)` — `step` is a real
/// `u32`, so `step + 1` is computed in-graph at the next stage (NOT hand-tracked;
/// this is the heterogeneous pipe+scalar carry the bundle handle composition
/// enables). The reconstructed `(DeviceSlice, u32)` tuple IS Mappable, so the
/// final `and_then_host(|(slice, _step)| ...)` type-checks. Same kernels, loss,
/// step==2 as the original.
#[test]
fn forward_pass_carries_scalar_state_via_bundle() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let sum_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&sum_cell);
    // The seam is fed by a `bundle2`, so its terminal `Checkouts` is now the
    // source's per-branch tuple `(Checkout<DeviceSlice>, Checkout<u32>)` (#212 —
    // each branch threads its own home), not a collapsed `Checkout<(…, …)>`.
    let (_final_buf, step) = upload(vec![10u32; N])
        .and_then(|buf| {
            // Pack: kernel output (a bare `Pipe<DeviceSlice>`) + the scalar 1.
            bundle!(kernels.scale_u32([N], buf, 2), value(1u32))
        })
        .and_then(|(buf, step)| {
            // `buf` is a `Pipe<DeviceSlice>`, `step` is `u32` (by-value handle) —
            // so `step + 1` computes here, carried in-chain, not hand-tracked.
            bundle!(kernels.scale_u32([N], buf, 3), value(step + 1))
        })
        .and_then_host(move |(slice, _step): (&mut [u32], u32)| {
            *cell.lock().unwrap() = slice.iter().sum();
            Ok(())
        })
        .sync(&ctx)
        .expect("stateful forward pass");
    let final_sum = *sum_cell.lock().unwrap();
    assert_eq!(final_sum, 60 * N as u32); // 10 * 2 * 3 = 60
    // `step` is a `Checkout<u32>`; `Checkout<u32>: PartialEq<u32>` compares direct.
    assert_eq!(step, 2);
}

#[test]
fn mpsc_three_producers_into_single_combine() {
    // Three independent producers (upload + fill) feed a downstream
    // combine kernel (add_u32). Producers run via bundle3; combine is
    // sequential. Final stage: scale the combined buffer.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let producers = bundle3(
        upload(vec![0u32; N]).and_then(|buf| kernels.fill_u32([N], buf, 3)),
        upload(vec![0u32; N]).and_then(|buf| kernels.fill_u32([N], buf, 4)),
        upload(vec![0u32; N]),
    );

    let result = producers
        .and_then(|(a, b, out)| kernels.add_u32([N], a, b, out))
        .and_then(|(_a, _b, out)| kernels.scale_u32([N], out, 5))
        .and_then(download)
        .sync(&ctx)
        .expect("mpsc chain");
    // (3 + 4) * 5 = 35
    assert!(result.iter().all(|&v| v == 35));
}
