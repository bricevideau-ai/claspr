//! `.and_then_host_with_context` — host work mid-chain with `&Context`
//! access. Same async / worker-thread semantics as `.and_then_host`;
//! only difference is the closure also receives the chain's running
//! `Context` for read-only host-side use (device props, device list,
//! etc.).
//!
//! Where today's idiom for "host work that needs ec inside a chain"
//! would be `with_context(|ec| ...)` — which silently drains deps
//! and discards events — this combinator is the right shape: the
//! worker handles the dep / unmap plumbing, the closure focuses on
//! the host computation.

use claspr::{Context, Error};
use claspr_async::{DeviceOperation, DeviceOperationHostExt, download, upload};
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
fn closure_receives_running_context() {
    // Read the context's device list inside the closure; capture
    // it via Arc<Mutex<_>> so the test can assert on it after
    // the chain completes. Closure runs on a worker thread, so
    // the Vec<String> needs to flow out via shared state.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_c = Arc::clone(&captured);

    let result: Vec<u32> = upload!(vec![3u32; N])
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .and_then_host_with_context(move |context: &Context, _view: &mut [u32]| {
            let names: Vec<String> = context
                .devices()
                .iter()
                .map(|d| d.name().unwrap_or_else(|_| "<unknown>".to_string()))
                .collect();
            *captured_c.lock().unwrap() = names;
            Ok(())
        })
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("chain");

    // Chain output is unaffected by the host closure (view wasn't
    // mutated) — still scale_u32(3, 2) = 6.
    assert!(result.iter().all(|&v| v == 6));
    // Closure ran and observed the context's device list.
    let names = captured.lock().unwrap();
    assert!(
        !names.is_empty(),
        "host closure should have seen at least one device"
    );
}

#[test]
fn closure_can_mutate_view_and_use_context() {
    // Combine: read context info to drive a per-element host
    // computation on the mapped buffer view, mutations commit back
    // to the device via the worker's unmap. Final download asserts
    // the view-side write made it back.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result: Vec<u32> = upload!(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 10))
        .and_then_host_with_context(|context: &Context, view: &mut [u32]| {
            // Use context's device count as a per-element multiplier.
            // (Just to exercise context access; the actual computation
            // doesn't matter — we assert on its result below.)
            let multiplier = context.devices().len() as u32;
            for slot in view.iter_mut() {
                *slot = slot.saturating_add(multiplier);
            }
            Ok(())
        })
        .and_then(|buf| download!(buf))
        .sync(&ctx)
        .expect("chain");

    let expected = 10u32 + ctx.devices().len() as u32;
    assert!(result.iter().all(|&v| v == expected));
}

#[test]
fn closure_err_surfaces_rich_variant_via_host_error_slot() {
    // Same error-fidelity contract as `.and_then_host`: closure
    // returning Err lands in the host-error slot; terminal surfaces
    // the rich variant, not Error::OpenCl(-1).
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let chain = upload!(vec![1u32; N])
        .and_then(|buf| kernels.scale_u32([N], buf, 2))
        .and_then_host_with_context(|_ctx: &Context, _view: &mut [u32]| -> claspr::Result<()> {
            Err(Error::Build {
                log: "ec-aware host abort".to_string(),
            })
        });

    let err = chain.sync(&ctx).expect_err("expected host error");
    assert!(
        matches!(&err, Error::Build { log } if log == "ec-aware host abort"),
        "got {err:?}",
    );
}
