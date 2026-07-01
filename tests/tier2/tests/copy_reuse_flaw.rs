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
use claspr::{Context, DeviceSlice, MappedSlice, SvmLevel, USMSlice};

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

/// USM (fine-grain-system SVM) copy into a `USMSliceUninit` dst, reused. USM's
/// uninit backing is a `Vec<MaybeUninit<T>>`, so the Init→Uninit downgrade is a
/// same-layout `Vec` reinterpret (`USMSliceUninit::from_init`) rather than a
/// private-field re-wrap — but it IS sound (the safe downgrade direction) and
/// must re-arm just like Device/Mapped. Skips if the device lacks fine-grain SVM.
#[test]
fn usm_copy_uninit_dst_is_reusable() {
    let Some(ctx) = ctx() else { return };
    if ctx.svm_capability() != SvmLevel::FineSystem {
        eprintln!("SKIP: device lacks fine-grain-system SVM (USM)");
        return;
    }

    let data: Vec<u32> = (0..N as u32).map(|x| x + 200).collect();
    let src = MappedSlice::<u32>::from_slice(&ctx, &data).expect("usm src alloc");
    let dst = USMSlice::<u32>::alloc_uninit(&ctx, N).expect("usm uninit dst alloc");

    let g = eager_copy_to(src, dst);

    {
        // Run 1 ALSO verifies the copy actually copied the right bytes: USMSlice
        // (fine-grain-system SVM) `Deref`s to `[T]` directly, so read the dst back
        // through the Checkout and assert it equals src. (`_co_src` unused.)
        let (_co_src, co_dst) = g.sync(&ctx).expect("first sync");
        assert_eq!(
            &co_dst[..],
            &data[..],
            "run 1 USM uninit dst must hold the copied src bytes"
        );
    } // <- USM downgrade rehome returns the Init dst into the Cell<USMSliceUninit>.

    {
        // Run 2 proves reusability (downgrade-rehome, not busy) AND re-copies the
        // right bytes into the re-armed dst.
        let (_co_src, co_dst) = g
            .sync(&ctx)
            .expect("second sync (USM uninit dst must downgrade-rehome, not busy)");
        assert_eq!(
            &co_dst[..],
            &data[..],
            "run 2 USM uninit dst == src after downgrade-rehome"
        );
    }
}
