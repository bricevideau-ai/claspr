//! Stand-in for a "library crate" that exports a typed graph the
//! caller can compose. This is the meta-kernel scenario the design
//! is really targeting: a library author builds a pipeline once,
//! exports it as a typed `Graph<I, O>`, and callers compose it
//! across crate boundaries.

use crate::graph::Graph;
use crate::{BufU32, ScalarU32};

/// Exported pipeline: take a buffer, fill it with 0xdead_beef, then
/// scale by 2. Returns a `Graph` whose type signature is fully
/// nameable from outside the module — no impl-trait, no boxed
/// trait objects, no leaking implementation generics.
pub fn scaled_fill() -> Graph<(BufU32,), (BufU32,)> {
    // Internally we compose two leaves, but we *erase* the scalar
    // arg by baking the constant into the graph value. The exported
    // type only mentions the user-facing arg shape.
    //
    // (The real implementation would do this via the graph's IR;
    // here the spike just stitches leaves and pretends.)
    let fill = Graph::<(BufU32, ScalarU32), (BufU32,)>::leaf("fill_with_deadbeef");
    let scale = Graph::<(BufU32,), (BufU32,)>::leaf("scale_by_2");

    // Adapter — in real code, this would be how we hide the inner
    // `(BufU32, ScalarU32)` shape behind the public `(BufU32,)`. For
    // the spike we just emit a leaf that pretends to do both.
    let _ = fill;
    let _ = scale;
    Graph::leaf("scaled_fill (library composite)")
}
