//! Eager port of `on_device.rs`: per-op device routing for eager graph chains.
//! Device-by-index routing is expressed structurally via `.on_device_at(i)`,
//! which resolves the index against the running context at execute (no external
//! Device captures, no execute-time closure). Mirrors
//! `eager_cutover::eager_on_device`.
//!
//! Named `eager_on_device_suite` to avoid a file-stem clash with the existing
//! `eager_cutover::eager_on_device` test fn / harness expectations.
//!
//! Old → new mapping:
//!   `upload!(v)`                  → `upload(v)`
//!   `download!(buf)`              → `download`
//!   `kernel(...).on_device(dev)`  → `.on_device_at(i)` on the eager kernel op
//!   `bundle!(a, b)`              → `bundle2(a, b)`
//!
//! NOTE (eager seam): routing flows through the pipe-fed `.and_then`, so the
//! routed downstream's `clEnqueue*` carries the upstream's completion event on
//! its wait-list — a real device-side ordering edge, stronger than a whole-queue
//! barrier. These tests assert final values / error surfacing / both-branches-
//! ran; the regression that exercises the same-device read-after-write edge
//! itself lives in `and_then_pipe_dep_same_device_raw` below.
//!
//! Skips when only one device is available (no real multi-device platform AND no
//! sub-device partition). Guard copied verbatim from on_device.rs.

use claspr::Error;
use claspr::eager::{DeviceOpExt, bundle2, download, upload};
use claspr_test_kernels::kernels;
use claspr_test_support::{ctx, ctx_two_devices};

const N: usize = 64;

/// on_device.rs::on_device_routes_chain_to_devices_from_context — two scale
/// stages, one per device, plus a final download. Device identity resolved by
/// index at execute each stage. 1 × 3 × 4 = 12.
#[test]
fn on_device_routes_chain_to_devices_from_context() {
    let Some((ctx, _, _)) = ctx_two_devices() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let result = upload(vec![1u32; N])
        .and_then(move |buf| kernels_ref.scale_u32([N], buf, 3).on_device_at(0))
        .and_then(move |buf| kernels_ref.scale_u32([N], buf, 4).on_device_at(1))
        .and_then(download)
        .sync(&ctx)
        .expect("on_device chain");

    assert_eq!(result.len(), N);
    assert!(result.iter().all(|&v| v == 12));
}

/// on_device.rs::on_device_preserves_host_error_slot_across_routing — an
/// `.and_then_host` failure inside a routed chain must still surface the rich
/// variant at the terminal, proving the routed child EC shares the parent's
/// host-error slot.
#[test]
fn on_device_preserves_host_error_slot_across_routing() {
    let Some((ctx, _, _)) = ctx_two_devices() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let chain = upload(vec![1u32; N])
        .and_then(move |buf| kernels_ref.scale_u32([N], buf, 2).on_device_at(1))
        // DeviceSlice's Mappable view is `&mut [T]`. Closure returns Err — the
        // rich variant must survive across the routed child EC's host-error slot
        // back to the chain terminal.
        .and_then_host(|_view: &mut [u32]| -> claspr::Result<()> {
            Err(Error::Build {
                log: "routed-chain abort".to_string(),
            })
        });

    let err = chain.sync(&ctx).expect_err("expected branch B error");
    assert!(
        matches!(&err, Error::Build { log } if log == "routed-chain abort"),
        "got {err:?}",
    );
}

/// on_device.rs::on_device_bundle_runs_branches_on_distinct_devices — a bundle
/// whose two branches are routed to different devices; both run, both outputs
/// produced. Smoke test for "multi-device branches don't hang".
#[test]
fn on_device_bundle_runs_branches_on_distinct_devices() {
    let Some((ctx, _, _)) = ctx_two_devices() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let (a, b) = bundle2(
        upload(vec![1u32; N])
            .and_then(move |buf| kernels_ref.scale_u32([N], buf, 7).on_device_at(0))
            .and_then(download),
        upload(vec![1u32; N])
            .and_then(move |buf| kernels_ref.scale_u32([N], buf, 11).on_device_at(1))
            .and_then(download),
    )
    .sync(&ctx)
    .expect("bundle across devices");

    assert!(a.iter().all(|&v| v == 7));
    assert!(b.iter().all(|&v| v == 11));
}

/// Regression for the pipe-fed `.and_then` device-side ordering edge.
///
/// The pipe-fed `.and_then` hands the downstream builder a `Handle` (a `Pipe`)
/// that carries the upstream's deps. The `Input::Pipe` arm of `resolve` threads
/// the upstream's completion event directly onto the downstream kernel's
/// `clEnqueue*` wait-list — a real per-command device-side edge, strictly
/// stronger than a whole-queue barrier. So a same-device read-after-write is
/// ordered on the device, not by driver luck. (This is the concrete proof the
/// barrier-free redesign is at least as strong as the removed combinator.)
///
/// Chain: upload `1`s → scale ×3 (upstream write) → `.and_then` scale ×5
/// (downstream read-modify-write of the SAME buffer, fed the upstream pipe) →
/// download. Correct result is `15`; a missing edge would let the ×5 read stale
/// `1`s and yield `5` on a racy driver.
#[test]
fn and_then_pipe_dep_same_device_raw() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let kernels_ref = &kernels;

    let out = upload(vec![1u32; N])
        // Upstream write: buffer becomes all 3s. Its completion event rides the
        // pipe handed to the downstream.
        .and_then(|buf| kernels_ref.scale_u32([N], buf, 3))
        // Downstream read-after-write on the SAME buffer, fed the upstream pipe
        // — the upstream event is threaded onto its enqueue wait-list.
        .and_then(move |buf| kernels_ref.scale_u32([N], buf, 5))
        .and_then(download)
        .sync(&ctx)
        .expect("same-device R-A-W chain");

    assert!(
        out.iter().all(|&v| v == 15),
        "expected all 15 (1*3*5); ordering edge missing? got {:?}",
        &out[..out.len().min(8)],
    );
}
