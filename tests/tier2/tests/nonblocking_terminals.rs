//! The non-blocking terminals never let the device touch host memory that has
//! been freed (or is being read) under it: `submit_on` keeps the graph and its
//! output alive until its event completes, a dropped `run` future releases its
//! output only once the chain completes, and `submit_pending_on` (and the
//! `submit` of host-memory ops) hands the value back as a `Pending`, which only
//! yields it after the work completes — and waits if dropped. A transfer on
//! caller-borrowed memory (`read_into`, image `read`/`write`) is complete when
//! the terminal returns, so even a leaked `Pending` can't outlive it.
//!
//! The device work is held back by a user event, so these are deterministic:
//! the terminal returns while the read is still pending.

use claspr::eager::{DeviceOpExt, download, read_into};
use claspr::image::format::R32Float;
use claspr::{
    DeviceSlice, Event, Image2D, ImageEnqueue, ReadWrite, complete_user_event, create_user_event,
};
use claspr_test_kernels::kernels;
use claspr_test_support::{ctx, ctx_with_images};
use opencl3::event::{CL_COMPLETE, retain_event};
use std::time::{Duration, Instant};

const LEN: usize = 1 << 16;

/// A user event `(gate, dep)`: `dep` gates work, `gate` releases it.
fn gate(ctx: &claspr::Context) -> (Event, Event) {
    let gate = create_user_event(ctx).expect("user event");
    // SAFETY: a second owned handle on the same event (retained for it).
    let dep = unsafe {
        retain_event(gate.get()).expect("retain");
        Event::new(gate.get())
    };
    (gate, dep)
}

/// `submit_on` of a graph ending in a download: the `Vec` the held read targets
/// stays alive past the terminal, so the read lands in it — not in whatever the
/// heap hands out next (it used to: freshly allocated blocks were overwritten).
#[test]
fn submit_on_keeps_a_downloaded_vec_alive() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let queue = ctx.default_outoforder_queue(ctx.device()).expect("queue");
    for round in 0..20 {
        let (gate, dep) = gate(&ctx);
        let buf = DeviceSlice::<u32>::alloc_zero(&ctx, LEN).expect("alloc");
        let g = ks
            .fill_u32([LEN], buf, 0xdead_beef)
            .after(dep)
            .and_then(download);
        let done = g.submit_on(&*queue).expect("submit");
        // Churn the heap: a freed output `Vec` would be handed out here.
        let junk: Vec<Vec<u8>> = (0..64).map(|i| vec![i as u8; 4096]).collect();
        complete_user_event(&gate, CL_COMPLETE).expect("release");
        done.wait().expect("wait");
        for (i, block) in junk.iter().enumerate() {
            assert!(
                block.iter().all(|&b| b == i as u8),
                "round {round}: heap block {i} overwritten"
            );
        }
    }
}

/// Dropping the async `run` future before it resolves (a cancelled `select!`,
/// a timeout) does not free the download's `Vec` under the held read.
#[test]
fn dropping_a_run_future_keeps_the_download_alive() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let queue = ctx.default_outoforder_queue(ctx.device()).expect("queue");
    for round in 0..20 {
        let (gate, dep) = gate(&ctx);
        let buf = DeviceSlice::<u32>::alloc_zero(&ctx, LEN).expect("alloc");
        let future = ks
            .fill_u32([LEN], buf, 0xdead_beef)
            .after(dep)
            .and_then(download)
            .run(&ctx);
        drop(future);
        let junk: Vec<Vec<u8>> = (0..64).map(|i| vec![i as u8; 4096]).collect();
        complete_user_event(&gate, CL_COMPLETE).expect("release");
        queue.finish().expect("finish");
        for (i, block) in junk.iter().enumerate() {
            assert!(
                block.iter().all(|&b| b == i as u8),
                "round {round}: heap block {i} overwritten"
            );
        }
    }
}

/// `submit_pending_on` hands a download's `Vec` back once the read completes.
#[test]
fn submit_pending_on_yields_the_download() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let queue = ctx.default_outoforder_queue(ctx.device()).expect("queue");
    let (gate, dep) = gate(&ctx);
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, LEN).expect("alloc");
    let pending = ks
        .fill_u32([LEN], buf, 7)
        .after(dep)
        .and_then(download)
        .submit_pending_on(&*queue)
        .expect("submit");
    assert!(!pending.is_complete(), "the read is held back");
    complete_user_event(&gate, CL_COMPLETE).expect("release");
    let host = pending.wait().expect("wait");
    assert!(host.iter().all(|&v| v == 7));
}

/// Dropping a `Pending` without waiting blocks until the work completes.
#[test]
fn dropping_a_pending_waits_for_the_work() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let queue = ctx.default_outoforder_queue(ctx.device()).expect("queue");
    let (gate, dep) = gate(&ctx);
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, LEN).expect("alloc");
    let pending = ks
        .fill_u32([LEN], buf, 7)
        .after(dep)
        .and_then(download)
        .submit_pending_on(&*queue)
        .expect("submit");
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        complete_user_event(&gate, CL_COMPLETE).expect("release");
    });
    let start = Instant::now();
    drop(pending);
    assert!(
        start.elapsed() >= Duration::from_millis(150),
        "the drop returned before the read completed"
    );
    release.join().expect("release thread");
}

/// Completes `gate` from another thread after `ms` milliseconds.
fn release_later(gate: Event, ms: u64) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(ms));
        complete_user_event(&gate, CL_COMPLETE).expect("release");
    })
}

/// A `read_into` held back by a gate: `submit_pending_on` returns only once the
/// read into `dst` is done, so leaking the `Pending` (which ends the borrow of
/// `dst` without running its destructor) can't let the device write it later.
#[test]
fn a_leaked_pending_cannot_outlive_a_borrowed_read() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let queue = ctx.default_outoforder_queue(ctx.device()).expect("queue");
    let (gate, dep) = gate(&ctx);
    let release = release_later(gate, 200);
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, LEN).expect("alloc");
    let mut dst = vec![0u32; LEN];
    let start = Instant::now();
    let pending = ks
        .fill_u32([LEN], buf, 7)
        .after(dep)
        .and_then(|b| read_into(b, &mut dst))
        .submit_pending_on(&*queue)
        .expect("submit");
    std::mem::forget(pending);
    assert!(
        start.elapsed() >= Duration::from_millis(150),
        "submit returned before the read into dst completed"
    );
    assert!(dst.iter().all(|&v| v == 7));
    release.join().expect("release thread");
}

const W: u32 = 64;
const H: u32 = 64;
const PIXELS: usize = (W * H) as usize;

/// An in-order queue whose next command waits for a slow kernel. (Image
/// transfers start a graph, so a gate can't hold them back; the queue order
/// does.)
fn queue_behind_slow_kernel(ctx: &claspr::Context) -> claspr::Queue<claspr::InOrder> {
    let ks = kernels::kernels(ctx).expect("load kernels");
    let queue = claspr::Queue::<claspr::InOrder>::new(ctx).expect("in-order queue");
    let alloc = || DeviceSlice::<u32>::alloc_zero(ctx, LEN).expect("alloc");
    ks.slow_pair_u32([LEN], alloc(), alloc(), 1 << 12)
        .submit_on(&queue)
        .expect("slow kernel");
    queue
}

/// Same for an image read into a caller slice.
#[test]
fn a_leaked_pending_cannot_outlive_a_borrowed_image_read() {
    let Some(ctx) = ctx_with_images() else { return };
    let img = Image2D::<ReadWrite, R32Float>::alloc(&ctx, W, H).expect("alloc image");
    let img = img.fill([3.0, 0.0, 0.0, 0.0]).wait().expect("fill");
    let queue = queue_behind_slow_kernel(&ctx);
    let mut dst = vec![-1.0f32; PIXELS];
    let pending = img
        .read(&mut dst)
        .submit_pending_on(&queue)
        .expect("submit");
    std::mem::forget(pending);
    assert!(
        dst.iter().all(|&v| v == 3.0),
        "dst not read when submit returned"
    );
}

/// An image write from a caller slice has read it by the time `submit`
/// returns: overwriting the slice right after leaking the `Pending` (which ends
/// the borrow) does not change what lands in the image.
#[test]
fn a_leaked_pending_cannot_outlive_a_borrowed_image_write() {
    let Some(ctx) = ctx_with_images() else { return };
    let img = Image2D::<ReadWrite, R32Float>::alloc(&ctx, W, H).expect("alloc image");
    // A second handle on the image, to read it back after leaking the write.
    let mem = opencl3::memory::ClMem::get(img.image_ref());
    // SAFETY: `mem` is live; the retain is released by the wrapping `Image`.
    let raw = unsafe {
        opencl3::memory::retain_mem_object(mem).expect("retain");
        opencl3::memory::Image::new(mem)
    };
    let queue = queue_behind_slow_kernel(&ctx);
    let mut src = vec![1.0f32; PIXELS];
    let pending = img.write(&src).submit_pending_on(&queue).expect("submit");
    std::mem::forget(pending);
    src.fill(2.0);
    queue.finish().expect("finish");
    let mut back = vec![0.0f32; PIXELS];
    let (origin, region) = ([0usize; 3], [W as usize, H as usize, 1]);
    // SAFETY: `back` holds the whole image; the read is blocking.
    unsafe {
        queue.raw().enqueue_read_image(
            &raw,
            opencl3::types::CL_BLOCKING,
            origin.as_ptr(),
            region.as_ptr(),
            0,
            0,
            back.as_mut_ptr().cast(),
            &[],
        )
    }
    .expect("read back");
    assert!(
        back.iter().all(|&v| v == 1.0),
        "the write read the overwritten src"
    );
}
