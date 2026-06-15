// Spike scaffolding: fake leaf-op fields stand in for real Tier 1 op
// state that the spike's fake impls don't read. Allowed at the crate
// level so the spike stays readable without scattered attrs.
#![allow(dead_code)]

//! Trait-level spike for Option B: extending claspr's `DeviceOperation`
//! with a `RecordableOp` sub-trait so the existing combinator graph
//! (`and_then`, `bundle`, `fan_out`) becomes the IR for command-buffer
//! recording — instead of a parallel closed-enum IR.
//!
//! ## What this spike validates
//!
//! 1. **Trait shape**: `RecordableOp: DeviceOperation` adds a single
//!    `.record()` method mirroring `.execute()`, threading sync points
//!    the way today's `.execute()` threads event deps.
//! 2. **Combinator participation**: `AndThen` / `Bundle2` impl
//!    `RecordableOp` *conditionally* on their children, so recordability
//!    propagates through composition via trait bounds. Verified at depth
//!    via `composition.rs` (5-deep AndThen, 3-level nested Bundle).
//! 3. **Compile-time opt-out**: non-recordable ops (`Upload`,
//!    `OnDevice`, `AndThenHost`) don't impl `RecordableOp`. Trying to
//!    record a chain that contains any of them fails at compile time
//!    with a clean `E0277` naming the offending type — see
//!    `compile_fail_cases.txt` for captured rustc diagnostics.
//!
//! ## What this spike does NOT do (explicit non-goals)
//!
//! - **No user-facing wrapper API.** An earlier pass of this spike had
//!   `Graph<I, O>`, `EagerPipeline<I, O>`, and `GraphBuilder<...>`
//!   wrapper types. That direction was a design dead end — the agreed
//!   final shape (see `NOTES.md` → "Command-buffer-backed graphs")
//!   puts `.call()` / `.mutate_call()` on `DeviceOperation` itself,
//!   with no separate wrapper. The wrapper code has been deleted from
//!   this spike to avoid misleading future readers; the design notes
//!   capture what the wrappers explored without keeping the code.
//! - No real OpenCL FFI (faked `CommandBuffer` is a string-recording
//!   stub).
//! - No real claspr integration.
//!
//! ## Carry-forward
//!
//! The traits + combinator-propagation pattern here is what real
//! claspr's `claspr-async` crate would adopt. The user-facing
//! surface on top (`.call()` / `.mutate_call()`, opt-ins, cache
//! protocol) is a separate piece of work — see `NOTES.md`
//! → "Command-buffer-backed graphs" for the agreed design and the
//! next-slice plan.

mod combinators;
mod composition;
mod device_op;
mod erasure;
mod leaves;
mod opt_outs;

fn main() {
    println!("graph_devop_record spike — see tests via `cargo test`");
}
