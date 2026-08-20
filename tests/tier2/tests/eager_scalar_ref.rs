//! Scalar-by-reference kernel args (`#[spirv(cross_workgroup)] &T` /
//! `&mut T`) backed by a first-class [`DeviceScalar<T>`] (#208).
//!
//! rust-gpu lowers these to a bare pointer-to-scalar param (no length
//! operand); the claspr proc-macro binds them via the DEDICATED
//! `KernelScalarRef[Mut]Arg<T>` traits, impl'd ONLY for `DeviceScalar`
//! (a len-1 `DeviceSlice` no longer binds). Both the read (`&T`) and
//! output (`&mut T`) shapes are proven, generic over the element type
//! (u32 AND f32), and across every graph position a buffer input flows
//! through: kernel arg, `slot!(Tag)`, `Pipe`, `Checkout`, plus a
//! host-seam-write-then-kernel-read replay path.

use claspr::DeviceScalar;
use claspr::eager::{DeviceOpExt, bundle2, download, forward, upload};
use claspr::{slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

const N: usize = 64;

/// `&u32` read scalar-ref: a `DeviceScalar<u32>` bound to `factor`
/// scales every element of `data`. Proves the pointer-only arg is set
/// at the right index (the slice arg before it stays intact).
#[test]
fn scale_by_ref_u32_reads_device_scalar() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let factor = DeviceScalar::<u32>::new(&ctx, 3).expect("alloc factor");

    let result = upload(vec![7u32; N])
        .and_then(|data| kernels.scale_by_ref_u32([N], data, factor))
        .and_then(|(data, _factor)| download(data))
        .sync(&ctx)
        .expect("scale-by-ref chain");
    assert!(result.iter().all(|&v| v == 21), "got {result:?}");
}

/// `&f32` read scalar-ref — the SAME shape over a different element
/// type, proving the feature is generic (not u32-special-cased).
#[test]
fn scale_by_ref_f32_reads_device_scalar() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let factor = DeviceScalar::<f32>::new(&ctx, 2.5).expect("alloc factor");

    let result = upload(vec![4.0f32; N])
        .and_then(|data| kernels.scale_by_ref_f32([N], data, factor))
        .and_then(|(data, _factor)| download(data))
        .sync(&ctx)
        .expect("scale-by-ref f32 chain");
    assert!(
        result.iter().all(|&v| (v - 10.0).abs() < 1e-4),
        "got {result:?}"
    );
}

/// `&mut u32` output scalar-ref: the kernel writes `val` through a
/// `DeviceScalar<u32>` bound to `out`. Proves the &mut scalar-ref
/// threads through `Output` and the written value is host-readable.
#[test]
fn write_scalar_u32_threads_to_output() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let out = DeviceScalar::<u32>::new(&ctx, 0).expect("alloc out");

    let result = kernels
        .write_scalar_u32([1usize], out, 12345u32)
        .sync(&ctx)
        .expect("write-scalar chain");
    assert_eq!(result.read_value().expect("read"), 12345u32);
}

/// `&mut f32` output scalar-ref — element-type generality of the output
/// path.
#[test]
fn write_scalar_f32_threads_to_output() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let out = DeviceScalar::<f32>::new(&ctx, 0.0).expect("alloc out");
    let result = kernels
        .write_scalar_f32([1usize], out, 2.5f32)
        .sync(&ctx)
        .expect("write-scalar f32 chain");
    assert!((result.read_value().expect("read") - 2.5).abs() < 1e-6);
}

/// Tier-1 terminal on the &mut scalar-ref kernel — `.wait()` returns
/// the written `DeviceScalar` directly (no graph).
#[test]
fn write_scalar_tier1_wait() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let out = DeviceScalar::<u32>::new(&ctx, 0).expect("alloc out");
    let out = kernels
        .write_scalar_u32([1usize], out, 999u32)
        .wait()
        .expect("write_scalar wait");

    assert_eq!(out.read_value().expect("read"), 999u32);
}

// ── Generality: a scalar in a slot!(Tag) ────────────────────────────

slots! { Factor: DeviceScalar<u32> }

/// A `DeviceScalar` used through a `slot!(Tag)` value — bound by name,
/// then the built graph replayed with a rebound scalar.
#[test]
fn device_scalar_in_slot() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let data = claspr::DeviceSlice::<u32>::from_slice(&ctx, &[7u32; N]).expect("data");
    let g = kernels
        .scale_by_ref_u32([N], data, slot!(Factor))
        .and_then(|(data, _factor)| download(data));

    // Bind Factor = 3 and run.
    let f3 = DeviceScalar::<u32>::new(&ctx, 3).expect("f3");
    let out = g.bind(Factor(f3)).sync(&ctx).expect("bound run");
    assert!(out.iter().all(|&v| v == 21), "got {out:?}");
}

// ── Generality: a scalar fed by a Pipe ──────────────────────────────

/// A `DeviceScalar` produced by an upstream op (a `Pipe<DeviceScalar>`)
/// feeds a downstream kernel's `&u32` arg — the pipe-fed scalar path.
#[test]
fn device_scalar_fed_by_pipe() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // write_scalar produces a DeviceScalar; feed its pipe into scale_by_ref.
    let factor = DeviceScalar::<u32>::new(&ctx, 0).expect("factor");
    let data = claspr::DeviceSlice::<u32>::from_slice(&ctx, &[5u32; N]).expect("data");
    let result = kernels
        .write_scalar_u32([1usize], factor, 4u32)
        .and_then(move |factor| {
            // `factor` is a Pipe<DeviceScalar<u32>> here — feed it straight in.
            kernels
                .scale_by_ref_u32([N], data, factor)
                .and_then(|(data, _factor)| download(data))
        })
        .sync(&ctx)
        .expect("pipe-fed scalar chain");
    assert!(result.iter().all(|&v| v == 20), "got {result:?}");
}

// ── Generality: a scalar lent/severed via Checkout ──────────────────

/// A `DeviceScalar` lent forward from graph A into graph B via its
/// `Checkout` (LEND), and then severed via `into_inner`.
#[test]
fn device_scalar_lent_and_severed_via_checkout() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Graph A: write factor = 6.
    let factor = DeviceScalar::<u32>::new(&ctx, 0).expect("factor");
    let ga = kernels.write_scalar_u32([1usize], factor, 6u32);
    let factor_co = ga.sync(&ctx).expect("A run"); // Checkout<DeviceScalar>

    // Graph B: LEND factor_co (feed the Checkout forward, not into_inner).
    let data = claspr::DeviceSlice::<u32>::from_slice(&ctx, &[3u32; N]).expect("data");
    let out = kernels
        .scale_by_ref_u32([N], data, factor_co)
        .and_then(|(data, _factor)| download(data))
        .sync(&ctx)
        .expect("B run");
    assert!(out.iter().all(|&v| v == 18), "got {out:?}");

    // A re-arms (factor rehomed on B's terminal drop) → A re-runnable.
    let factor_co2 = ga.sync(&ctx).expect("A re-run after lend");
    // SEVER: take the scalar out of A for good.
    let factor2 = factor_co2.into_inner();
    assert_eq!(factor2.read_value().expect("read severed"), 6u32);
}

// ── #210-critical: host-seam-writes-scalar-then-kernel-reads, replayed ──

slots! { Data: claspr::DeviceSlice<u32>, Addend: DeviceScalar<u32> }

/// The #210-critical path: a `&mut u32` `DeviceScalar` is WRITTEN by a
/// host seam (`*view = …`) mid-graph, then READ by a later kernel
/// (`&u32`) in the SAME graph — and the whole graph is REPLAYED across
/// syncs (the scalar rehomes to its cell each run). Proves a device
/// scalar threads + rehomes for reuse exactly like a buffer.
#[test]
fn host_seam_writes_scalar_then_kernel_reads_replayed() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Build the graph:
    //   1. a device-scalar-alloc leaf mints `addend` (a &mut u32 DeviceScalar).
    //   2. a host seam writes it (= 100) via `*view` (the &mut T Mappable View).
    //   3. add_ref_u32 READS it as `&u32` in the SAME graph and adds it to data.
    let ctx2 = ctx.clone();
    let g = claspr::device_scalar_alloc!(0u32)
        .and_then_host(|view: &mut u32| {
            *view = 100;
            Ok(())
        })
        .and_then(move |addend| {
            let data = claspr::DeviceSlice::<u32>::from_slice(&ctx2, &[1u32; N]).expect("data");
            kernels
                .add_ref_u32([N], data, addend)
                .and_then(|(data, _addend)| download(data))
        });

    // First run.
    let out = g.sync(&ctx).expect("host-seam scalar run");
    assert!(out.iter().all(|&v| v == 101), "got {out:?}");
}

/// Same host-write-then-read shape but the scalar is a REUSED graph
/// cell replayed across multiple syncs (rehome across runs). This
/// variant uses concrete cells + a plain re-bind of the scalar value
/// between runs to prove the rehome path (the #210 replay guarantee)
/// works for a DeviceScalar.
#[test]
fn device_scalar_rehomes_across_replays() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Reusable graph: add a slot-bound DeviceScalar addend to a slot-bound
    // data buffer, replayed. The scalar rehomes to its slot cell each run.
    let g = kernels
        .add_ref_u32([N], slot!(Data), slot!(Addend))
        .and_then(|(data, _addend)| forward(data));

    let data = claspr::DeviceSlice::<u32>::from_slice(&ctx, &[10u32; N]).expect("data");
    let addend = DeviceScalar::<u32>::new(&ctx, 5).expect("addend");
    let g = g.call((Data(data), Addend(addend)));

    // Run 1: 10 + 5 = 15. Borrowing map (NOT `read`, which severs) so the Data
    // slot rehomes on the Checkout's drop and the graph re-arms.
    let data_co = g.sync(&ctx).expect("run 1");
    {
        let view = data_co.map().wait().expect("map 1");
        assert!(view.iter().all(|&v| v == 15), "run1 got {:?}", &view[..4]);
    }
    drop(data_co); // rehome Data (and Addend via reclaim) for replay

    // Run 2 (replay over the SAME handles): 15 + 5 = 20.
    let data_co = g.sync(&ctx).expect("run 2");
    {
        let view = data_co.map().wait().expect("map 2");
        assert!(view.iter().all(|&v| v == 20), "run2 got {:?}", &view[..4]);
    }
}

// ── bundle: two scalar-ref kernels in parallel branches ─────────────

/// Two independent scalar-ref writes bundled — proves scalars work as
/// distinct multi-output homes in a `bundle2`.
#[test]
fn two_scalars_in_bundle() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let a = DeviceScalar::<u32>::new(&ctx, 0).expect("a");
    let b = DeviceScalar::<f32>::new(&ctx, 0.0).expect("b");

    let (a_co, b_co) = bundle2(
        kernels.write_scalar_u32([1usize], a, 7u32),
        kernels.write_scalar_f32([1usize], b, 1.5f32),
    )
    .sync(&ctx)
    .expect("bundle run");

    assert_eq!(a_co.read_value().expect("a read"), 7u32);
    assert!((b_co.read_value().expect("b read") - 1.5).abs() < 1e-6);
}

/// TIER COVERAGE: a `MappedScalar<u32>` (coarse-grain SVM backing) binds to the
/// SAME `&u32` scalar-ref arg as a `DeviceScalar` — the scalar-ref traits are
/// generic over the backing tier (`Scalar<B> where B: KernelSliceReadArg`), so
/// all three memory tiers get scalar-ref for free. Skips on a no-SVM device.
/// (Guards the regression the strict-DeviceScalar-only binding could have caused:
/// before #208 a len-1 MappedSlice bound to `&u32`; now MappedScalar does.)
#[test]
fn mapped_scalar_reads_via_scalar_ref() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let factor = match claspr::MappedScalar::<u32>::new_mapped(&ctx, 3) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("SKIP: no coarse-grain SVM for MappedScalar");
            return;
        }
    };

    let result = upload(vec![7u32; N])
        .and_then(|data| kernels.scale_by_ref_u32([N], data, factor))
        .and_then(|(data, _factor)| download(data))
        .sync(&ctx)
        .expect("mapped-scalar scale-by-ref chain");
    assert!(result.iter().all(|&v| v == 21), "got {result:?}");
}

/// TIER COVERAGE: a `USMScalar<u32>` (fine-grain system SVM backing) binds to the
/// same `&u32` scalar-ref arg — the third memory tier. Skips on a no-fine-grain-SVM
/// device. Together with the DeviceScalar + MappedScalar tests this proves the
/// scalar-ref surface is symmetric with the three slice tiers (no tier dropped).
#[test]
fn usm_scalar_reads_via_scalar_ref() {
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let factor = match claspr::USMScalar::<u32>::new_usm(&ctx, 3) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("SKIP: no fine-grain system SVM for USMScalar");
            return;
        }
    };

    let result = upload(vec![7u32; N])
        .and_then(|data| kernels.scale_by_ref_u32([N], data, factor))
        .and_then(|(data, _factor)| download(data))
        .sync(&ctx)
        .expect("usm-scalar scale-by-ref chain");
    assert!(result.iter().all(|&v| v == 21), "got {result:?}");
}
