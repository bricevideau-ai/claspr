// Spike scaffolding: placeholder types carry &'static str payloads
// for visual debugging that the type-plumbing tests don't read.
#![allow(dead_code)]

//! Spike: validate the `Graph<I, O>` trait system for command-buffer-
//! backed Tier 2 graphs.
//!
//! Tests:
//! - Does the trait shape support typed variadic `.call(a, b, c)` via
//!   the existing `KernelArgs`-style per-arity macro pattern?
//! - Does `and_then` compose `Inputs` / `Outputs` cleanly across
//!   combinators?
//! - Can a "library" function return a typed `Graph<I, O>` and have a
//!   downstream caller compose it via `and_then` without leaking
//!   implementation generics?
//!
//! What this spike deliberately does NOT do (next phase):
//! - Wire to real claspr Tier 2 ops (`DeviceOperation`, kernels).
//! - Touch the `cl_khr_command_buffer` FFI.
//! - Implement `.mutate_call()` or the runtime-decided recording mode.
//!
//! Cross-reference: `NOTES.md` → "Command-buffer-backed graphs (design)".

mod graph;
mod library;

use graph::Graph;
#[cfg(test)]
use graph::Op;

fn main() {
    println!("graph_cb spike — see tests via `cargo test`");
}

// ─── Placeholder data types (stand-ins for DeviceSlice etc.) ────────

#[derive(Debug, Clone, Default)]
struct BufU32(&'static str);

#[derive(Debug, Clone, Default)]
struct BufF32(&'static str);

#[derive(Debug, Clone, Default)]
struct ScalarU32(&'static str);

// ─── Leaf graphs constructed from "ops" ─────────────────────────────
//
// Real claspr would build these out of `DeviceOperation` nodes. Here
// they're just opaque IR strings — the spike isn't testing execution,
// only the type plumbing.

fn fill_u32() -> Graph<(BufU32, ScalarU32), (BufU32,)> {
    Graph::leaf("fill_u32")
}

fn scale_u32() -> Graph<(BufU32, ScalarU32), (BufU32,)> {
    Graph::leaf("scale_u32")
}

/// In-place "double every element" — no scalar arg, takes only the
/// buffer. Composable end-to-end with leaf ops that emit a single buf.
fn double_u32() -> Graph<(BufU32,), (BufU32,)> {
    Graph::leaf("double_u32")
}

fn u32_to_f32() -> Graph<(BufU32,), (BufF32,)> {
    Graph::leaf("u32_to_f32")
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library;

    #[test]
    fn single_call_arity_2() {
        // Spike verifies the types: a leaf graph's `.call(a, b)` must
        // accept the right number + types of args and return an
        // `Op<Outputs>`. Value-level execution is out of scope (the
        // op is a stub that returns Default outputs).
        let g = fill_u32();
        let _op: Op<(BufU32,)> = g.call(BufU32("buf"), ScalarU32("42"));
    }

    #[test]
    fn and_then_composes_inputs_and_outputs() {
        // fill_u32 : (BufU32, ScalarU32) -> (BufU32,)
        // u32_to_f32 : (BufU32,) -> (BufF32,)
        // composed : (BufU32, ScalarU32) -> (BufF32,)
        let pipeline = fill_u32().and_then(u32_to_f32());
        let _op: Op<(BufF32,)> = pipeline.call(BufU32("buf"), ScalarU32("0"));
    }

    #[test]
    fn three_stage_chain() {
        // fill (BufU32, ScalarU32) -> (BufU32,)
        // double (BufU32,) -> (BufU32,)
        // u32_to_f32 (BufU32,) -> (BufF32,)
        // composed (BufU32, ScalarU32) -> (BufF32,)
        let pipeline = fill_u32().and_then(double_u32()).and_then(u32_to_f32());
        let _op: Op<(BufF32,)> = pipeline.call(BufU32("buf"), ScalarU32("3"));
    }

    #[test]
    fn unused_scale_u32_kept_to_exercise_arity_2_inputs_in_isolation() {
        // scale_u32 is composable into longer chains only if upstream
        // produces (BufU32, ScalarU32). Keeping it as a leaf-call test
        // confirms the arity-2 inherent impl works for any leaf.
        let _op: Op<(BufU32,)> = scale_u32().call(BufU32("x"), ScalarU32("2"));
    }

    #[test]
    fn library_returned_graph_composes_with_local_graph() {
        // The library function returns a typed graph the caller can
        // compose. This is THE meta-kernel scenario.
        let g_lib: Graph<(BufU32,), (BufU32,)> = library::scaled_fill();
        let pipeline = g_lib.and_then(u32_to_f32());
        let _op: Op<(BufF32,)> = pipeline.call(BufU32("buf"));
    }

    #[test]
    fn graph_is_cheaply_cloneable() {
        // Pre-condition for `and_then`-reuse (G.and_then(|_| G.clone())).
        let g = fill_u32();
        let g2 = g.clone();
        let _ = g.call(BufU32("a"), ScalarU32("1"));
        let _ = g2.call(BufU32("b"), ScalarU32("2"));
    }

    // ── Compile-fail style checks ──
    //
    // Verified by hand: uncommenting either block produces the
    // expected compile error. We'll migrate these to a real ui_test
    // harness when promoting the design out of spike status.
    //
    // 1. Wrong arity at the call site:
    //
    //     let g = fill_u32(); // expects (BufU32, ScalarU32)
    //     let _ = g.call(BufU32("buf"));
    //
    //    → error[E0061]: this method takes 2 arguments but 1 argument
    //      was supplied
    //
    // 2. Type mismatch in and_then composition:
    //
    //     let _ = fill_u32().and_then(scale_u32());
    //
    //    → error[E0308]: mismatched types — expected
    //      `Graph<(BufU32,), _>`, found `Graph<(BufU32, ScalarU32),
    //      (BufU32,)>`. The diagnostic points at the .and_then call
    //      site and names both the expected (self's Outputs) and
    //      found (next's Inputs) tuple shapes.
}
