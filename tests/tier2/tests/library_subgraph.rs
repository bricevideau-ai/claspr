//! Library-provided reusable subgraphs: curry, then compose.
//!
//! The pattern documented in the `eager` module guide ("Reusable subgraphs:
//! curry, then compose"). A library declares its slot tags, leaves late-bound
//! inputs as slots, and returns a NAMED erased graph; users curry it with
//! `call`, compose it inside `and_then`, and reuse one curried configuration at
//! several sites through a closure. These tests pin the properties that make
//! the pattern work: instances curried before composing are independent, a slot
//! left open through composition fans out to every instance, renaming a tag
//! per use site gives an instance its own knob, instances are storable values,
//! and a run-time number of them composes into one graph that still records a
//! single command buffer.

use claspr::eager::{DeviceOpExt, download};
use claspr::{Error, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::{N, cb_records_of, ctx, seeded};

/// Stands in for a library crate: it owns its slot tags and hands out a graph.
mod affine_lib {
    use claspr::eager::{DeviceDynOp, DeviceOpExt};
    use claspr::{DeviceSlice, Single, SlotSet, slot, slots};
    use claspr_test_kernels::kernels::Kernels;
    use claspr_test_support::N;

    slots! {
        AffIn: DeviceSlice<u32>,
        Bias: DeviceSlice<u32>,
        AffOut: DeviceSlice<u32>,
        Factor: u32,
    }

    /// The library's public graph type: named, storable, no bounds to restate.
    /// Its slot set is part of the type, so `sync` is refused until all four are
    /// bound.
    pub type Affine = DeviceDynOp<'static, Single<DeviceSlice<u32>>, Slots>;
    /// The slots of [`Affine`], in the order the library chooses to declare.
    pub type Slots = SlotSet![AffIn, Bias, AffOut, Factor];
    /// A fully curried [`Affine`]: nothing left to bind.
    pub type Curried = DeviceDynOp<'static, Single<DeviceSlice<u32>>>;

    /// `AffOut = (AffIn + Bias) * Factor`. Every input is a slot; `AffIn` takes
    /// a value or an upstream pipe. The input buffer is never written, so a
    /// composed graph replays to the same result.
    pub fn affine(ks: &Kernels) -> Affine {
        ks.add_u32([N], slot!(AffIn), slot!(Bias), slot!(AffOut))
            .and_then(move |(_a, _b, out)| ks.scale_u32([N], out, slot!(Factor)))
            .erase_as::<Slots, _>()
    }
}
use affine_lib::{AffIn, AffOut, Bias, Curried, Factor, affine};

// The USER's per-instance names for the library's `Factor` (see the rename tests).
slots! { LeftF: u32, RightF: u32 }

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
        .call_all((Factor(7),));
    let out = g.sync(&ctx).expect("sync");
    assert_eq!(uniform(&out), ((3 + 1) * 7 + 4) * 7);
}

/// Curried instances are ordinary values: keep them in a struct, run either.
#[test]
fn curried_instances_are_storable() {
    struct Presets {
        double: Curried,
        triple: Curried,
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
/// back to `Curried`, so the loop variable keeps one type) and the whole chain
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
    let mut g: Curried = affine(&ks).call((
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

/// The USER decides sharing per use site: two instances renamed to their own
/// tags are bound independently after composition; a third left alone shares
/// the library's `Factor` (the library never had to guess).
#[test]
fn rename_gives_an_instance_its_own_knob() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let inst = |x, bias, out| (AffIn(x), Bias(bias), AffOut(out));
    let g = ks
        .scale_u32([N], seeded(&ctx, 1), 1u32)
        .and_then(|x| {
            affine(&ks)
                .rename::<Factor, LeftF, _>()
                .call(inst(x, seeded(&ctx, 1), seeded(&ctx, 0)))
        })
        .and_then(|y| {
            affine(&ks).rename::<Factor, RightF, _>().call(inst(
                y,
                seeded(&ctx, 0),
                seeded(&ctx, 0),
            ))
        })
        .and_then(|z| affine(&ks).call(inst(z, seeded(&ctx, 0), seeded(&ctx, 0))))
        .and_then(download)
        .call((LeftF(2), RightF(3), Factor(5)));
    assert_eq!(uniform(&g.sync(&ctx).expect("sync")), (1 + 1) * 2 * 3 * 5);
}

/// After a rename, the old name no longer reaches the instance, and the unbound
/// site is reported under the name the caller has to bind.
#[test]
fn renamed_away_tag_no_longer_reaches_the_instance() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = affine(&ks)
        .rename::<Factor, LeftF, _>()
        .call((
            AffIn(seeded(&ctx, 1)),
            Bias(seeded(&ctx, 0)),
            AffOut(seeded(&ctx, 0)),
        ))
        .and_then(download)
        // Opt out of static tracking: this test exercises the RUN-TIME checks.
        .erase_unchecked();
    // Binding the old name finds no site; the renamed site stays unbound.
    match g.mutate_call((Factor(9),)) {
        Err(err) => assert!(matches!(err, Error::SlotNoSuchTag(_)), "got {err:?}"),
        Ok(_) => panic!("binding `Factor` reached a renamed site"),
    }
    let err = g.sync(&ctx).expect_err("LeftF is unbound");
    assert!(matches!(err, Error::SlotUnbound("LeftF")), "got {err:?}");
}

/// A renamed knob re-binds alone at replay time, and the keyed cache still
/// recognises each state (one command buffer per state on a CB device).
#[test]
fn renamed_knob_rebinds_alone_and_stays_cached() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let inst = |x| (AffIn(x), Bias(seeded(&ctx, 0)), AffOut(seeded(&ctx, 0)));
    let g = ks
        .scale_u32([N], seeded(&ctx, 1), 1u32)
        .and_then(|x| affine(&ks).rename::<Factor, LeftF, _>().call(inst(x)))
        .and_then(|y| affine(&ks).rename::<Factor, RightF, _>().call(inst(y)))
        .and_then(download)
        .call((LeftF(2), RightF(3)));
    for i in 0..8u32 {
        let r = if i % 2 == 0 { 3 } else { 7 };
        g.mutate_call((RightF(r),)).expect("mutate");
        assert_eq!(uniform(&g.sync(&ctx).expect("sync")), 2 * r, "sync {i}");
    }
    assert!(
        cb_records_of(&g) <= 2,
        "recorded {} command buffers",
        cb_records_of(&g)
    );
}
