//! Eager-API port of `and_then_host_with_context.rs`: host work mid-chain with
//! `&Context` access via the eager host seam's `and_then_host_with_context`.
//!
//! Old → new mapping:
//!   `upload!(v)`                                → `upload(v)`
//!   `download!(buf)`                            → `.and_then(download)`
//!   `.and_then_host_with_context(|ctx, view|…)` → same method on `DeviceOpExt`
//!
//! The eager `and_then_host_with_context` closure receives `(&Context, view)`
//! exactly like the closure layer's, and propagates `Err` via `?` through the
//! terminal (see eager_cutover `eager_and_then_host_error_propagates`). All
//! three test fns port 1:1 — same N, values, and assertions.

use claspr::eager::{DeviceOpExt, download, upload};
use claspr::{Context, DeviceSlice, Error};
use claspr::{slot, slots};
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

    let result = upload(vec![3u32; N])
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

    let result = upload(vec![0u32; N])
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

    let chain = upload(vec![1u32; N])
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

// ── REUSABLE with-context host seam: replay across syncs (#211) ───────

slots! { Buf: DeviceSlice<u32> }

/// The `_with_context` host seam is now `Fn` (Arc-held, cloned per run; the
/// `&Context` is cloned fresh per run too), so a graph containing one REPLAYS.
/// This graph adds `context.devices().len()` to every element each run, so a
/// buffer seeded at 0 holds `k` after run 1 and `2k` after run 2 — proving the
/// closure re-ran on the replay (a one-shot `FnOnce` would have errored on the
/// 2nd `sync`).
#[test]
fn and_then_host_with_context_replays_and_reruns() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let k = ctx.devices().len() as u32;

    // Slot lives behind the seam; `bind` reaches it and the buffer rehomes across
    // replays (the seam forwards `bind_slots` and threads the source's home). The
    // seam adds `k` to every element each run.
    let buf = DeviceSlice::<u32>::from_slice(&ctx, &[0u32; N]).expect("seed buffer");
    let g = kernels
        .scale_u32([N], slot!(Buf), 1u32)
        .and_then_host_with_context(|context: &Context, view: &mut [u32]| {
            let add = context.devices().len() as u32;
            for slot in view.iter_mut() {
                *slot = slot.wrapping_add(add);
            }
            Ok(())
        })
        .bind(Buf(buf));

    // Run 1: 0 + k.
    let co = g.sync(&ctx).expect("with-context run 1");
    {
        let view = co.map().wait().expect("map 1");
        assert!(view.iter().all(|&v| v == k), "run1 got {:?}", &view[..4]);
    }
    drop(co); // rehome for replay

    // Run 2 (replay): k + k = 2k. Proves the seam re-ran with a fresh context.
    let co = g.sync(&ctx).expect("with-context run 2 (replay)");
    {
        let view = co.map().wait().expect("map 2");
        assert!(
            view.iter().all(|&v| v == 2 * k),
            "run2 got {:?}",
            &view[..4]
        );
    }
    drop(co);
}
