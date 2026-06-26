//! Reuse-flaw regression: a concrete-buffer `eager_copy_to` sitting in a reused
//! graph must re-arm BOTH its src and dst cells on `Checkout` drop, so a second
//! `sync` succeeds instead of erroring "graph busy".
//!
//! Before the home-in-pipe `Rehome` generalization, `CopyTo2::execute` deposited
//! its outputs with `Pipe::put` (home = `None`), so neither buffer returned to
//! its lending cell on drop and the second `sync` found an empty cell → busy
//! error. The fix threads each input cell as a typed return home:
//!
//! - **SRC** is not retyped by the copy (`CopyOutputs::Src == Src`) → identity
//!   rehome (the cell takes its own buffer back).
//! - **DST** may be retyped: a `DeviceSliceUninit<T, M>` dst comes back as an
//!   `Init` `DeviceSlice<T, M>` (the copy wrote every byte). Returning the Init
//!   buffer to the `Cell<DeviceSliceUninit>` is a SOUND DOWNGRADE — the Init
//!   buffer is the stronger capability — wired via a downgrade rehome that
//!   re-wraps it as `DeviceSliceUninit` before storing.

use claspr::eager::{DeviceOpExt, eager_copy_to};
use claspr::{Context, DeviceSlice};

const N: usize = 16;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

/// Concrete `DeviceSlice` src + dst: `eager_copy_to` run twice (Checkouts
/// dropped between). Both buffers must return to their cells (identity rehome
/// on each side) so the second `sync` is not "busy".
#[test]
fn copy_init_dst_is_reusable() {
    let Some(ctx) = ctx() else { return };

    let data: Vec<u32> = (0..N as u32).collect();
    let src = DeviceSlice::<u32>::from_slice(&ctx, &data).expect("src alloc");
    let dst = DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("dst alloc");

    let g = eager_copy_to(src, dst);

    // Run 1: copy, then drop BOTH Checkouts immediately (re-arm g's two cells).
    {
        let (_co_src, _co_dst) = g.sync(&ctx).expect("first sync");
    } // <- both Checkouts drop here; identity rehome returns both buffers.

    // Run 2: must succeed (graph re-armed) AND produce correct data.
    {
        let (_co_src, co_dst) = g.sync(&ctx).expect("second sync (graph must NOT be busy)");
        let mut out = vec![0u32; N];
        co_dst.read(&mut out).wait().expect("read dst run2");
        assert_eq!(out, data, "run 2 dst == src");
    }
}

/// Concrete `DeviceSliceUninit` dst: the copy retypes it to `Init`. Returning
/// the Init buffer into the `Cell<DeviceSliceUninit>` is the downgrade rehome.
/// Run twice with the Checkouts dropped between; run-2 success proves the
/// downgrade rehome put the (now Init) buffer back into the uninit cell.
#[test]
fn copy_uninit_dst_is_reusable() {
    let Some(ctx) = ctx() else { return };

    let data: Vec<u32> = (0..N as u32).map(|x| x + 100).collect();
    let src = DeviceSlice::<u32>::from_slice(&ctx, &data).expect("src alloc");
    let dst = DeviceSlice::<u32>::alloc_uninit(&ctx, N).expect("uninit dst alloc");

    let g = eager_copy_to(src, dst);

    {
        let (_co_src, _co_dst) = g.sync(&ctx).expect("first sync");
    } // <- downgrade rehome returns the Init dst into the Cell<DeviceSliceUninit>.

    {
        let (_co_src, co_dst) = g
            .sync(&ctx)
            .expect("second sync (uninit dst must downgrade-rehome, not busy)");
        let mut out = vec![0u32; N];
        co_dst.read(&mut out).wait().expect("read dst run2");
        assert_eq!(out, data, "run 2 uninit dst == src");
    }
}
