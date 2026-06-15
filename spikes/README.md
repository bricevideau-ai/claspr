# spikes/

Throwaway prototypes that validate design ideas before they land in the
actual claspr crates. Each spike is a standalone Cargo workspace (note
the `[workspace]` table in each `Cargo.toml` — they're independent from
the main claspr workspace).

| Spike | Validates | Referenced by |
|---|---|---|
| `combinator/` | Tier 2 combinator API shape (`DeviceOperation`, `and_then`, `bundle!`, `fan_out`, `.arc()`, `.and_then_host`, `HostAccessible`, profiling-via-callback). 16 worked scenarios covering linear chains through cross-device pipelines. | `EXECUTION-MODEL.md` "Two-tier architecture" section |
| `graph_cb/` | The `Graph<I, O>` type-system shape for command-buffer-backed Tier 2 graphs: single struct (not a trait hierarchy), per-arity inherent `.call(...)` via the `KernelArgs`-style macro, `and_then` composition, library-boundary export of typed graphs (the meta-kernel pattern). | `NOTES.md` "Command-buffer-backed graphs" section |
| `graph_devop_record/` | Trait-level spike for command-buffer-backed graphs: validates extending `DeviceOperation` with a `RecordableOp` sub-trait so the existing combinator graph (`and_then`, `bundle`, `fan_out`) becomes the CB-recording IR directly — instead of building a parallel closed-enum IR. 17 passing tests cover trait propagation through nested combinators (5-deep AndThen, 3-level nested Bundle), structural opt-out for `OnDevice` / `AndThenHost` / `Upload`, and the erasure handoff (`erasure.rs`: concrete `RecordableOp` chain → reusable type-erased recorder via a factory closure, with recordability carried across erasure as `Option<record_fn>`). `compile_fail_cases.txt` captures the rustc diagnostics the type system produces for non-recordable chains. Does NOT include a user-facing wrapper API — see the design notes for the agreed shape (`.call()` / `.mutate_call()` on DeviceOperation itself). | `NOTES.md` "Command-buffer-backed graphs" |

Spikes are reference-only — they don't ship and aren't part of the
workspace build. Run with `cargo run` from inside the spike's directory.
