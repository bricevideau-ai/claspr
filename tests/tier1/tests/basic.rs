//! End-to-end Tier 1 launch flow: alloc → write → kernel → read.
//!
//! Validates that the proc-macro-emitted Tier 1 method
//! (`kernels.foo(launcher, grid, &slice, scalar).wait()`) produces
//! the right data for the simplest pipeline shape — what every
//! claspr user starts with.

use claspr::{Context, DeviceSlice};
use claspr_test_kernels::kernels;

const N: usize = 1024;

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
fn fill_kernel_writes_value_to_every_element() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");
    let buf = kernels
        .fill_u32([N], buf, 0xfeed_cafe)
        .wait(&ctx)
        .expect("launch");

    let mut out = vec![0u32; N];
    buf.read(&mut out).wait(&ctx).expect("read");
    assert!(out.iter().all(|&v| v == 0xfeed_cafe));
}

#[test]
fn write_kernel_read_round_trip() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let initial: Vec<u32> = (0..N as u32).collect();
    let mut buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");
    buf.write(&initial).wait(&ctx).expect("write");

    // Scale by 3, then read back.
    let buf = kernels.scale_u32([N], buf, 3).wait(&ctx).expect("scale");

    let mut out = vec![0u32; N];
    buf.read(&mut out).wait(&ctx).expect("read");
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, (i as u32).wrapping_mul(3), "elem {i}");
    }
}

#[test]
fn multi_buffer_kernel_combines_inputs() {
    // add_u32: out[i] = a[i] + b[i]. Exercises 3-slice kernel launch.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let mut a = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc a");
    let mut b = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc b");
    let out = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc out");

    let host_a: Vec<u32> = vec![10; N];
    let host_b: Vec<u32> = vec![32; N];
    a.write(&host_a).wait(&ctx).expect("write a");
    b.write(&host_b).wait(&ctx).expect("write b");

    let (_a, _b, out) = kernels.add_u32([N], a, b, out).wait(&ctx).expect("add");

    let mut host_out = vec![0u32; N];
    out.read(&mut host_out).wait(&ctx).expect("read");
    assert!(host_out.iter().all(|&v| v == 42));
}

#[test]
fn submit_returns_event_for_cross_queue_chaining() {
    // .submit() yields a non-blocking event; passing it to a downstream
    // .after() on a separate launch enforces ordering even with two
    // different queues (here a single queue, but exercises the API).
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");
    let (buf, fill_event) = kernels
        .fill_u32([N], buf, 7)
        .submit(&ctx)
        .expect("submit fill");
    let buf = kernels
        .scale_u32([N], buf, 6)
        .after(fill_event)
        .wait(&ctx)
        .expect("scale after");

    let mut out = vec![0u32; N];
    buf.read(&mut out).wait(&ctx).expect("read");
    assert!(out.iter().all(|&v| v == 42));
}
