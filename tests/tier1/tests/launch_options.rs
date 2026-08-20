//! Kernel launch options — local work-size. (Global offset is not a
//! `LaunchSpec` feature today; if it grows one, `global_id_u32` in
//! the test-kernels crate is the natural probe.)

use claspr::{Context, DeviceSlice};
use claspr_test_kernels::kernels;

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
fn tuple_form_sets_local_size_via_local_invocation_id() {
    // First test that ever exercised the local plumbing. Uses the
    // pre-existing `IntoLaunchSpec for ([usize; 1], [usize; 1])`
    // tuple form. With N=64, L=8, each workgroup of 8 produces
    // local IDs 0..7 — the output should be [0..8] repeating 8x.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    const N: usize = 64;
    const L: usize = 8;
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("alloc");
    let buf = kernels
        .local_id_u32(([N], [L]), buf)
        .wait()
        .expect("launch with local via tuple form");
    let mut out = vec![0u32; N];
    buf.read(&mut out).wait().expect("read");
    for (i, &v) in out.iter().enumerate() {
        let expected = (i % L) as u32;
        assert_eq!(v, expected, "out[{i}] = {v}, expected {expected}");
    }
}
