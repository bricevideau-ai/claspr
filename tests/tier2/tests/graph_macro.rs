//! `graph!`: sequential composition as straight-line `let` bindings.
//!
//! The macro is pure syntax over nested `and_then` closures, so these tests pin
//! that (1) a binding stays usable in every later step — the reason the macro
//! exists — (2) the graph it builds is the hand-written nested chain's, node
//! for node, with the same results and command-buffer behaviour, and (3) the
//! patterns a handle is destructured with (single-output pipe, `_`, `..`) work.
//! The ownership side (a pipe used twice is rejected) is the
//! `graph_macro_pipe_used_twice` compile-fail fixture.

use claspr::eager::{DeviceOp, DeviceOpExt, GraphNode, download, graph_edge_table};
use claspr::graph;
use claspr_test_kernels::kernels;
use claspr_test_support::{N, cb_records_of, ctx, seeded};

/// The single value every element of a uniform result holds.
fn uniform(v: &[u32]) -> u32 {
    assert!(v.iter().all(|&x| x == v[0]), "non-uniform result");
    v[0]
}

/// The graph's shape, independent of cell ids: each node's depth and label,
/// and how many real pipe edges connect them.
fn shape(g: &impl DeviceOp) -> (Vec<(usize, String)>, usize) {
    let mut nodes: Vec<GraphNode> = Vec::new();
    g.dump_graph(0, &mut nodes);
    let edges = graph_edge_table(&nodes)
        .iter()
        .map(|(_, consumers)| consumers.len())
        .sum();
    (
        nodes.into_iter().map(|n| (n.depth, n.name)).collect(),
        edges,
    )
}

/// A pipe bound in the first step is used in the third; the result is stable
/// across replays (no input is written).
#[test]
fn earlier_bindings_stay_in_scope() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = graph! {
        let (a, _b, sum) = ks.add_u32([N], seeded(&ctx, 3), seeded(&ctx, 4), seeded(&ctx, 0));
        let sum = ks.scale_u32([N], sum, 2u32);
        let (_a, _sum, out) = ks.add_u32([N], a, sum, seeded(&ctx, 0));
        download(out)
    };
    for run in 1..=3 {
        assert_eq!(
            uniform(&g.sync(&ctx).expect("sync")),
            (3 + 4) * 2 + 3,
            "run {run}"
        );
    }
}

/// The macro builds exactly the nested `and_then` chain written by hand: same
/// nodes and edges, same result, same command-buffer recording.
#[test]
fn builds_the_hand_written_nested_chain() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let (a, b, z, w) = (
        seeded(&ctx, 3),
        seeded(&ctx, 4),
        seeded(&ctx, 0),
        seeded(&ctx, 0),
    );
    let by_macro = graph! {
        let (a, _b, sum) = ks.add_u32([N], a, b, z);
        let sum = ks.scale_u32([N], sum, 2u32);
        let (_a, _sum, out) = ks.add_u32([N], a, sum, w);
        download(out)
    };
    let (a, b, z, w) = (
        seeded(&ctx, 3),
        seeded(&ctx, 4),
        seeded(&ctx, 0),
        seeded(&ctx, 0),
    );
    let by_hand = ks.add_u32([N], a, b, z).and_then(move |(a, _b, sum)| {
        ks.scale_u32([N], sum, 2u32).and_then(move |sum| {
            ks.add_u32([N], a, sum, w)
                .and_then(move |(_a, _sum, out)| download(out))
        })
    });
    assert_eq!(shape(&by_macro), shape(&by_hand));
    for _ in 0..3 {
        assert_eq!(
            uniform(&by_macro.sync(&ctx).expect("sync macro")),
            uniform(&by_hand.sync(&ctx).expect("sync hand"))
        );
    }
    assert_eq!(cb_records_of(&by_macro), cb_records_of(&by_hand));
}

/// Handles destructure with the usual patterns: a single-output op binds its
/// pipe directly, `_` drops a pipe, `..` skips several.
#[test]
fn handle_patterns() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let g = graph! {
        let x = ks.scale_u32([N], seeded(&ctx, 5), 3u32);
        let (_, _, y) = ks.add_u32([N], x, seeded(&ctx, 1), seeded(&ctx, 0));
        let (.., y) = ks.copy_u32([N], y, seeded(&ctx, 0));
        download(y)
    };
    assert_eq!(uniform(&g.sync(&ctx).expect("sync")), 5 * 3 + 1);
}
