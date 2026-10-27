//! Library-provided reusable subgraphs: curry, then compose.
//!
//! The pattern documented in the `eager` module guide ("Reusable subgraphs:
//! curry, then compose"). A library declares its slot tags, leaves late-bound
//! inputs as slots, and returns a NAMED erased graph; users curry it with
//! `call`, compose it inside `and_then`, and reuse one curried configuration at
//! several sites through a closure. These tests pin the properties that make
//! the pattern work: instances curried before composing are independent, a slot
//! left open through composition fans out to every instance, instances are
//! storable values, and a run-time number of them composes into one graph that
//! still records a single command buffer.

use claspr::eager::{DeviceOpExt, download};
use claspr_test_kernels::kernels;
use claspr_test_support::{N, cb_records_of, ctx, seeded};

/// Stands in for a library crate: it owns its slot tags and hands out a graph.
mod affine_lib {
    use claspr::eager::{DeviceDynOp, DeviceOpExt};
    use claspr::{DeviceSlice, Single, slot, slots};
    use claspr_test_kernels::kernels::Kernels;
    use claspr_test_support::N;

    slots! {
        AffIn: DeviceSlice<u32>,
        Bias: DeviceSlice<u32>,
        AffOut: DeviceSlice<u32>,
        Factor: u32,
    }

    /// The library's public graph type: named, storable, no bounds to restate.
    pub type Affine = DeviceDynOp<'static, Single<DeviceSlice<u32>>>;

    /// `AffOut = (AffIn + Bias) * Factor`. Every input is a slot; `AffIn` takes
    /// a value or an upstream pipe. The input buffer is never written, so a
    /// composed graph replays to the same result.
    pub fn affine(ks: &Kernels) -> Affine {
        ks.add_u32([N], slot!(AffIn), slot!(Bias), slot!(AffOut))
            .and_then(move |(_a, _b, out)| ks.scale_u32([N], out, slot!(Factor)))
            .erase()
    }
}
use affine_lib::{AffIn, AffOut, Affine, Bias, Factor, affine};

/// The single value every element of a uniform result holds.
fn uniform(v: &[u32]) -> u32 {
    assert!(v.iter().all(|&x| x == v[0]), "non-uniform result");
    v[0]
}

/// One curried configuration reused at two sites through a closure (the
/// template), composed after the user's own stage; replays are stable.
#[test]
fn closure_template_reused_at_two_sites() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let double = |x, bias, out| affine(&ks).call((AffIn(x), Factor(2), Bias(bias), AffOut(out)));
    let g = ks
        .scale_u32([N], seeded(&ctx, 3), 1u32)
        .and_then(|x| double(x, seeded(&ctx, 1), seeded(&ctx, 0)))
        .and_then(|y| double(y, seeded(&ctx, 4), seeded(&ctx, 0)))
        .and_then(download);
    for run in 1..=3 {
        let out = g.sync(&ctx).expect("sync");
        assert_eq!(uniform(&out), ((3 + 1) * 2 + 4) * 2, "run {run}");
    }
}

/// Instances curried BEFORE composing are independent: different settings in
/// one graph.
#[test]
fn instances_curried_before_composing_are_independent() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = ks
        .scale_u32([N], seeded(&ctx, 3), 1u32)
        .and_then(|x| {
            affine(&ks).call((
                AffIn(x),
                Factor(2),
                Bias(seeded(&ctx, 1)),
                AffOut(seeded(&ctx, 0)),
            ))
        })
        .and_then(|y| {
            affine(&ks).call((
                AffIn(y),
                Factor(5),
                Bias(seeded(&ctx, 4)),
                AffOut(seeded(&ctx, 0)),
            ))
        })
        .and_then(download);
    let out = g.sync(&ctx).expect("sync");
    assert_eq!(uniform(&out), ((3 + 1) * 2 + 4) * 5);
}

/// A slot left OPEN through composition is shared: one bind on the composed
/// graph fills it in every instance.
#[test]
fn open_slot_fans_out_across_instances() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = ks
        .scale_u32([N], seeded(&ctx, 3), 1u32)
        .and_then(|x| affine(&ks).call((AffIn(x), Bias(seeded(&ctx, 1)), AffOut(seeded(&ctx, 0)))))
        .and_then(|y| affine(&ks).call((AffIn(y), Bias(seeded(&ctx, 4)), AffOut(seeded(&ctx, 0)))))
        .and_then(download)
        .call((Factor(7),));
    let out = g.sync(&ctx).expect("sync");
    assert_eq!(uniform(&out), ((3 + 1) * 7 + 4) * 7);
}

/// Curried instances are ordinary values: keep them in a struct, run either.
#[test]
fn curried_instances_are_storable() {
    struct Presets {
        double: Affine,
        triple: Affine,
    }
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let preset = |factor| {
        affine(&ks).call((
            AffIn(seeded(&ctx, 3)),
            Factor(factor),
            Bias(seeded(&ctx, 0)),
            AffOut(seeded(&ctx, 0)),
        ))
    };
    let p = Presets {
        double: preset(2),
        triple: preset(3),
    };
    assert_eq!(
        uniform(&p.double.and_then(download).sync(&ctx).expect("sync")),
        6
    );
    assert_eq!(
        uniform(&p.triple.and_then(download).sync(&ctx).expect("sync")),
        9
    );
}

/// A run-time number of instances composes in a plain loop (each step erases
/// back to `Affine`, so the loop variable keeps one type) and the whole chain
/// still records ONE command buffer, replayed on later syncs.
#[test]
fn runtime_length_chain_records_one_command_buffer() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let k: u32 = std::hint::black_box(20);
    let step = |x| {
        affine(&ks).call((
            AffIn(x),
            Factor(2),
            Bias(seeded(&ctx, 0)),
            AffOut(seeded(&ctx, 0)),
        ))
    };
    let mut g: Affine = affine(&ks).call((
        AffIn(seeded(&ctx, 1)),
        Factor(2),
        Bias(seeded(&ctx, 0)),
        AffOut(seeded(&ctx, 0)),
    ));
    for _ in 1..k {
        g = g.and_then(step).erase();
    }
    let g = g.and_then(download);
    for run in 1..=3 {
        assert_eq!(uniform(&g.sync(&ctx).expect("sync")), 1 << k, "run {run}");
    }
    // 1 on a device with cl_khr_command_buffer, 0 without (per-op path).
    assert!(
        cb_records_of(&g) <= 1,
        "recorded {} command buffers",
        cb_records_of(&g)
    );
}
