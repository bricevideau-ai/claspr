//! Eager-API port of `and_then_host_with_context.rs`: host work mid-chain with
//! `&Context` access via the eager host seam's `and_then_host_with_context`.
//!
//! Old → new mapping:
//!   `upload!(v)`                                → `upload::<u32, ReadWrite, _>(v)`
//!   `download!(buf)`                            → `.and_then(download)`
//!   `.and_then_host_with_context(|ctx, view|…)` → same method on `EagerOpExt`
//!
//! The eager `and_then_host_with_context` closure receives `(&Context, view)`
//! exactly like the closure layer's, and propagates `Err` via `?` through the
//! terminal (see eager_cutover `eager_and_then_host_error_propagates`). All
//! three test fns port 1:1 — same N, values, and assertions.

use claspr::eager::{EagerOpExt, download, upload};
use claspr::{Context, Error};
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
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_c = Arc::clone(&captured);

    let result: Vec<u32> = upload::<u32, claspr::ReadWrite, _>(vec![3u32; N])
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
        .and_then(download)
        .sync(&ctx)
        .expect("chain");

    // Chain output is unaffected by the host closure (view wasn't mutated) —
    // still scale_u32(3, 2) = 6.
    assert!(result.iter().all(|&v| v == 6));
    let names = captured.lock().unwrap();
    assert!(
        !names.is_empty(),
        "host closure should have seen at least one device"
    );
}

#[test]
fn closure_can_mutate_view_and_use_context() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let result: Vec<u32> = upload::<u32, claspr::ReadWrite, _>(vec![0u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 10))
        .and_then_host_with_context(|context: &Context, view: &mut [u32]| {
            let multiplier = context.devices().len() as u32;
            for slot in view.iter_mut() {
                *slot = slot.saturating_add(multiplier);
            }
            Ok(())
        })
        .and_then(download)
        .sync(&ctx)
        .expect("chain");

    let expected = 10u32 + ctx.devices().len() as u32;
    assert!(result.iter().all(|&v| v == expected));
}

#[test]
fn closure_err_surfaces_rich_variant_via_host_error_slot() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let chain = upload::<u32, claspr::ReadWrite, _>(vec![1u32; N])
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
