//! Eager-API port of `host_and_profile.rs` — the `and_then_host` mid-chain host
//! work cases. The two `.profiled` cases are ported in `eager_profile.rs` (the
//! eager `.profiled` hook is `DeviceProfileExt`); this file keeps only the
//! host-seam half. All four originals are accounted for across the two files.
//!
//! Old → new mapping:
//!   `value(v).and_then(|x| upload!(x))` → `upload(v)`
//!   `.and_then_host(|view|…)`           → same method on `DeviceOpExt`

use claspr::eager::{DeviceOpExt, upload, value};
use claspr::{Context, Device};
use claspr_test_kernels::kernels;
use std::sync::{Arc, Mutex};

const N: usize = 128;

fn ctx(profiling: bool) -> Option<Context> {
    let dev = Device::any().ok()?;
    Context::builder()
        .device(&dev)
        .profiling(profiling)
        .build()
        .ok()
}

// ── and_then_host ────────────────────────────────────────────────────

#[test]
fn and_then_host_sum_between_device_stages() {
    let Some(ctx) = ctx(false) else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let kernels = kernels::kernels(&ctx).expect("load kernels");
    // upload + fill + (host) sum-in-place via mapped view. The closure returns
    // Result<()>; the reduction value flows out via the canonical
    // Arc<Mutex<_>> side-effect channel.
    let sum_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&sum_cell);
    let _final_buf = upload(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 3))
        .and_then_host(move |slice: &mut [u32]| {
            *cell.lock().unwrap() = slice.iter().sum();
            Ok(())
        })
        .sync(&ctx)
        .expect("and_then_host chain");
    let sum = *sum_cell.lock().unwrap();
    assert_eq!(sum, 3 * N as u32);
}

#[test]
fn and_then_host_error_propagates() {
    let Some(ctx) = ctx(false) else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    // Closure returns Err → the eager host seam surfaces the original Rust
    // variant at the terminal (not the OpenCl(-1) cascade).
    let err = value(())
        .and_then_host(|()| -> claspr::Result<()> { Err(claspr::Error::SvmNotAvailable) })
        .sync(&ctx)
        .expect_err("expected error");
    assert!(matches!(err, claspr::Error::SvmNotAvailable), "got {err:?}",);
}

// ── profile ──────────────────────────────────────────────────────────
//
// `profile_chain_fires_callback_when_profiling_on` and
// `profile_chain_errors_when_profiling_off` are ported in `eager_profile.rs`
// (the eager `.profiled` hook is `DeviceProfileExt::profiled`).
