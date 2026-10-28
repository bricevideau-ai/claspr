//! A graph whose STRUCTURE is only known at run time — the shape of a model
//! replayer that reads a layer list from a file — under compile-time slot sets.
//!
//! The spec is parsed at run time and each line becomes a stage, chosen by a
//! `match`. Weights are concrete buffers fixed at build time (no slots); the
//! per-run inputs are slots shared by however many layers the spec has. The
//! loop variable keeps ONE type because each step re-declares the graph's
//! distinct slot set (`erase_as`), which collapses a tag repeated a run-time
//! number of times; `match` arms with different sets unify the same way.

use claspr::eager::{DeviceDynOp, DeviceOpExt, download, forward};
use claspr::{DeviceSlice, Single, SlotSet, slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::{N, cb_records_of, ctx, seeded};

slots! {
    X: DeviceSlice<u32>,
    Gain: u32,
}

/// The per-run inputs of the model, declared once.
type Inputs = SlotSet![X, Gain];
/// The model graph: one type whatever the spec says.
type Model = DeviceDynOp<'static, Single<DeviceSlice<u32>>, Inputs>;

/// A layer of the toy model, as parsed from one spec line.
enum Layer {
    /// `x *= Gain` — the shared run-time knob.
    Scale,
    /// `x *= k` — a weight fixed at build time.
    Mul(u32),
    /// `x += bias` — a weight BUFFER fixed at build time.
    Bias(u32),
}

fn parse(spec: &str) -> Vec<Layer> {
    spec.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            match (it.next(), it.next().map(|v| v.parse::<u32>().unwrap())) {
                (Some("scale"), None) => Layer::Scale,
                (Some("mul"), Some(k)) => Layer::Mul(k),
                (Some("bias"), Some(v)) => Layer::Bias(v),
                _ => panic!("bad spec line {l:?}"),
            }
        })
        .collect()
}

/// The host reference: run the spec on one element.
fn reference(layers: &[Layer], x: u32, gain: u32) -> u32 {
    layers.iter().fold(x, |x, l| match l {
        Layer::Scale => x.wrapping_mul(gain),
        Layer::Mul(k) => x.wrapping_mul(*k),
        Layer::Bias(v) => x.wrapping_add(*v),
    })
}

#[test]
fn spec_driven_graph_builds_binds_and_replays() {
    let Some(ctx) = ctx() else { return };
    let ks = &kernels::kernels(&ctx).expect("kernels");
    // Run-time spec (stands in for the model file).
    let spec = std::hint::black_box("mul 3\nscale\nbias 1\nscale\nmul 2\nbias 5\nscale\n");
    let layers = parse(spec);

    // Input stage: the model's input buffer is a slot.
    let mut g: Model = ks.scale_u32([N], slot!(X), 1u32).erase_as::<Inputs, _>();
    for layer in &layers {
        g = match *layer {
            Layer::Scale => g
                .and_then(|x| ks.scale_u32([N], x, slot!(Gain)))
                .erase_as::<Inputs, _>(),
            Layer::Mul(k) => g
                .and_then(move |x| ks.scale_u32([N], x, k))
                .erase_as::<Inputs, _>(),
            Layer::Bias(v) => {
                let (w, out) = (seeded(&ctx, v), seeded(&ctx, 0));
                g.and_then(move |x| {
                    ks.add_u32([N], x, w, out)
                        .and_then(|(_x, _w, out)| forward(out))
                })
                .erase_as::<Inputs, _>()
            }
        };
    }
    // Bind the inputs once (forgetting one is a compile error), then replay.
    // The stages work IN PLACE, so each run re-binds a fresh input with its knob.
    let g = g.call((X(seeded(&ctx, 2)), Gain(1))).and_then(download);
    for (run, (x, gain)) in [(2u32, 1u32), (2, 3), (5, 3), (2, 1), (4, 7)]
        .into_iter()
        .enumerate()
    {
        g.mutate_call((X(seeded(&ctx, x)), Gain(gain)))
            .expect("rebind inputs");
        let out = g.sync(&ctx).expect("sync");
        assert!(out.iter().all(|&v| v == out[0]), "non-uniform");
        assert_eq!(out[0], reference(&layers, x, gain), "run {run}");
    }
    // One command buffer per distinct (input cell, gain) state at most.
    assert!(cb_records_of(&g) <= 5, "recorded {}", cb_records_of(&g));
}
