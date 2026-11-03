//! A command-buffer build that can't be recorded — a call the driver rejects, a
//! finalize that fails — replays exactly the commands the walk recorded, enqueued
//! directly, instead of re-running the graph (which can't re-supply inputs that
//! already flowed into the build through pipes: it used to fail with
//! `PipeDrained`, or hang). After a driver rejection the context stops recording.
//!
//! The failures are injected (`record::__inject_cb_failure`), so this runs on any
//! device with command buffers; without them nothing records and the tests check
//! plain correctness.

use claspr::eager::{DeviceOp, DeviceOpExt, bundle2, upload};
use claspr::record::__inject_cb_failure;
use claspr_test_kernels::kernels;
use claspr_test_support::{N, ctx};

/// Clears the injection hook when a test ends (pass or fail), so it never
/// leaks into another test run on the same thread.
struct Injected;

impl Injected {
    fn after(calls: u32) -> Self {
        __inject_cb_failure(Some(calls));
        Injected
    }
}

impl Drop for Injected {
    fn drop(&mut self) {
        __inject_cb_failure(None);
    }
}

fn cb_ids<O: DeviceOp>(g: &O) -> Vec<usize> {
    let mut ids = Vec::new();
    g.collect_cb_ids(&mut ids);
    ids
}

fn plus_100(v: &mut [u32]) -> claspr::Result<()> {
    v.iter_mut().for_each(|x| *x += 100);
    Ok(())
}

/// The shape that used to drain a pipe: the kernels' input arrives through a
/// pipe from the `upload` (run outside the command buffer). The second
/// recording call is rejected; the build replays — 1 * 3 * 5 * 7 = 105 — and
/// the graph keeps replaying correctly afterwards.
#[test]
fn a_rejected_command_replays_a_chain_fed_through_a_pipe() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let g = upload(vec![1u32; N])
        .and_then(|b| ks.scale_u32([N], b, 3u32))
        .and_then(|b| ks.scale_u32([N], b, 5u32))
        .and_then(|b| ks.scale_u32([N], b, 7u32));
    let _hook = Injected::after(1);
    for run in 1..=3 {
        let got = g.sync(&ctx).unwrap_or_else(|e| panic!("run {run}: {e:?}"));
        let v = got.map().wait().expect("read");
        assert!(v.iter().all(|&x| x == 105), "run {run}: {:?}", &v[..4]);
    }
}

/// A span closing at a host seam whose finalize fails (the third command-buffer
/// call, after its two kernels): the close point replays, and the host step
/// reads the replayed result — ((1*3*5)+100)*2*7 = 1610.
#[test]
fn a_failed_finalize_at_a_host_seam_replays_the_span() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let g = upload(vec![1u32; N])
        .and_then(|b| ks.scale_u32([N], b, 3u32))
        .and_then(|b| ks.scale_u32([N], b, 5u32))
        .and_then_host(plus_100)
        .and_then(|b| ks.scale_u32([N], b, 2u32))
        .and_then(|b| ks.scale_u32([N], b, 7u32));
    let _hook = Injected::after(2);
    for run in 1..=3 {
        let got = g.sync(&ctx).unwrap_or_else(|e| panic!("run {run}: {e:?}"));
        let v = got.map().wait().expect("read");
        assert!(v.iter().all(|&x| x == 1610), "run {run}: {:?}", &v[..4]);
    }
}

/// A rejected first command in a root bundle of two independent kernels.
#[test]
fn a_rejected_command_in_a_bundle_replays() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let g = upload(vec![2u32; N]).and_then(|a| {
        upload(vec![5u32; N])
            .and_then(move |b| bundle2(ks.scale_u32([N], a, 10u32), ks.scale_u32([N], b, 3u32)))
    });
    let _hook = Injected::after(0);
    for run in 1..=2 {
        let (a, b) = g.sync(&ctx).unwrap_or_else(|e| panic!("run {run}: {e:?}"));
        let (a, b) = (
            a.map().wait().expect("read a"),
            b.map().wait().expect("read b"),
        );
        assert!(a.iter().all(|&x| x == 20), "run {run}: a {:?}", &a[..4]);
        assert!(b.iter().all(|&x| x == 15), "run {run}: b {:?}", &b[..4]);
    }
}

/// After a driver rejection the context stops recording: a later graph runs
/// per-op (no command buffer), still correctly.
#[test]
fn the_context_stops_recording_after_a_driver_failure() {
    let Some(ctx) = ctx() else { return };
    if !ctx.has_cl_khr_command_buffer() {
        return;
    }
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let chain = || {
        upload(vec![1u32; N])
            .and_then(|b| ks.scale_u32([N], b, 3u32))
            .and_then(|b| ks.scale_u32([N], b, 5u32))
    };
    let first = chain();
    {
        let _hook = Injected::after(0);
        let v = first
            .sync(&ctx)
            .expect("first")
            .map()
            .wait()
            .expect("read")
            .to_vec();
        assert!(v.iter().all(|&x| x == 15));
    }
    let later = chain();
    let v = later
        .sync(&ctx)
        .expect("later")
        .map()
        .wait()
        .expect("read")
        .to_vec();
    assert!(v.iter().all(|&x| x == 15));
    assert!(
        cb_ids(&later).is_empty(),
        "recorded after a driver failure: {:?}",
        cb_ids(&later)
    );
}
