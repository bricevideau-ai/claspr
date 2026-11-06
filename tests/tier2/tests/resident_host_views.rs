//! A non-blocking terminal (kernel `submit`, `submit_value_on`) hands a
//! buffer back with the completion event while the kernel may still run.
//! Every host view of it waits for that work: maps of a `DeviceSlice` or a
//! `MappedSlice` (Tier-1 guards and graph host views) and a `USMSlice` deref.
//!
//! The kernels are held back by a user event another thread completes, so
//! these are deterministic: the view is requested while the kernel can't run.
//! They run on the out-of-order queue and the views on the context's default
//! queue, so queue order alone doesn't serialize them.

use claspr::eager::{DeviceOpExt, acquire_device_view_read};
use claspr::{
    DeviceSlice, Event, MappedSlice, SvmLevel, USMSlice, complete_user_event, create_user_event,
};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;
use opencl3::event::{CL_COMPLETE, retain_event};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const N: usize = 1024;
const HELD: Duration = Duration::from_millis(200);
const WAITED: Duration = Duration::from_millis(150);

fn ooo(ctx: &claspr::Context) -> std::sync::Arc<claspr::Queue<claspr::OutOfOrder>> {
    ctx.default_outoforder_queue(ctx.device()).expect("queue")
}

/// A user event gating a kernel, completed from another thread after `HELD`.
fn gate(ctx: &claspr::Context) -> (Event, JoinHandle<()>) {
    let gate = create_user_event(ctx).expect("user event");
    // SAFETY: a second owned handle on the same event (retained for it).
    let dep = unsafe {
        retain_event(gate.get()).expect("retain");
        Event::new(gate.get())
    };
    let release = std::thread::spawn(move || {
        std::thread::sleep(HELD);
        complete_user_event(&gate, CL_COMPLETE).expect("release");
    });
    (dep, release)
}

#[test]
fn device_slice_map_waits_for_its_kernel() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (dep, release) = gate(&ctx);
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let (buf, _event) = ks
        .fill_u32([N], buf, 7)
        .after(dep)
        .submit_on(&*ooo(&ctx))
        .expect("submit");
    let start = Instant::now();
    let view = buf.map().wait().expect("map");
    assert!(
        start.elapsed() >= WAITED,
        "the map didn't wait for the kernel"
    );
    assert!(view.iter().all(|&v| v == 7));
    drop(view);
    release.join().expect("release thread");
}

/// A host write through a map made after `submit` lands after the kernel's.
#[test]
fn device_slice_map_mut_waits_for_its_kernel() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (dep, release) = gate(&ctx);
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let (mut buf, event) = ks
        .fill_u32([N], buf, 7)
        .after(dep)
        .submit_on(&*ooo(&ctx))
        .expect("submit");
    buf.map_mut().wait().expect("map")[0] = 99;
    event.wait().expect("kernel");
    let view = buf.map().wait().expect("map back");
    assert_eq!(
        view[0], 99,
        "the kernel overwrote a host write made after submit"
    );
    assert!(view[1..].iter().all(|&v| v == 7));
    drop(view);
    release.join().expect("release thread");
}

#[test]
fn mapped_slice_map_waits_for_its_kernel() {
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() == SvmLevel::None {
        eprintln!("SKIP: no SVM support");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (dep, release) = gate(&ctx);
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let (buf, _event) = ks
        .fill_u32([N], buf, 7)
        .after(dep)
        .submit_on(&*ooo(&ctx))
        .expect("submit");
    let start = Instant::now();
    let view = buf.map().wait().expect("map");
    assert!(
        start.elapsed() >= WAITED,
        "the map didn't wait for the kernel"
    );
    assert!(view.iter().all(|&v| v == 7));
    drop(view);
    release.join().expect("release thread");
}

/// A graph host view over a buffer a non-blocking `submit` handed back.
#[test]
fn graph_host_view_waits_for_the_kernel_before_it() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let (dep, release) = gate(&ctx);
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let (buf, _event) = ks
        .fill_u32([N], buf, 7)
        .after(dep)
        .submit_on(&*ooo(&ctx))
        .expect("submit");
    let _view = acquire_device_view_read(buf)
        .and_then_host(|v: &[u32]| {
            assert!(v.iter().all(|&x| x == 7), "the view read {}", v[0]);
            Ok(())
        })
        .sync(&ctx)
        .expect("view");
    release.join().expect("release thread");
}

/// A USMSlice returned from a whole graph: its deref waits for the graph.
/// (Its kernels also register their own events; the terminal's record is what
/// covers a command-buffer replay, where no per-launch hook runs.)
#[test]
fn usm_slice_from_a_graph_waits_for_it() {
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() != SvmLevel::FineSystem {
        eprintln!("SKIP: no fine-grain-system SVM");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let queue = ctx.default_outoforder_queue(ctx.device()).expect("queue");
    let (dep, release) = gate(&ctx);
    let buf = USMSlice::<u32>::new(&ctx, vec![1u32; N]).expect("alloc");
    let (buf, _event) = ks
        .scale_u32([N], buf, 3)
        .after(dep)
        .and_then(|b| ks.scale_u32([N], b, 5))
        .submit_value_on(&*queue)
        .expect("submit");
    let start = Instant::now();
    let first = buf[0];
    assert!(
        start.elapsed() >= WAITED,
        "the deref didn't wait for the graph"
    );
    assert_eq!(first, 15);
    assert!(buf.iter().all(|&v| v == 15));
    release.join().expect("release thread");
}
