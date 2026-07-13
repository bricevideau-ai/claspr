# claspr — architecture map (read this to orient before changing code)

This is the layered mental model + a "to change X, read Y" index, so you load ONE
subsystem instead of the whole tree. For the single-source compilation pipeline
(`#[claspr::device]` / `#[claspr::kernel]` → build-script kernel extraction → proc-macro
launcher), read `CLAUDE.md` — that's not repeated here. This file is about the **runtime
library** (`claspr/src`).

## The layers (bottom → top)

```
Tier 1 — buffers, images, context, queue        access.rs buffer.rs image.rs
  memory/access markers, DeviceSlice/Mapped/USM   context.rs queue.rs device.rs
  Image2D, host map views                         mapped.rs usm.rs host_view.rs mappable.rs
        │
op / launch — one kernel launch                  op.rs launch.rs
  LaunchOp (clEnqueueNDRangeKernel), KernelArg(s), LaunchSpec
        │
Tier 2 — the eager device-operation graph        eager.rs   (the big one)
  DeviceOp trait, combinators, leaves, slots
        │
CB recording — command-buffer acceleration       record.rs exec_ctx.rs
  CbBuilder/FinalizedCb, CbWalk, sync points
```

Each layer depends only on the ones below it. A change to Tier 2 rarely needs Tier 1
internals (just the public `DeviceSlice`/`Image2D`/`Context` types).

## Tier 2 (`eager.rs`) — the core concepts, in dependency order

Read these five first; they are the "mandatory core" every graph task touches:

1. **`Pipe<T>` / `Input<T>` / `Cell<T>`** — the graph EDGE. `Pipe` is the runtime
   value-storage cell a producer deposits into; `Input` is a consumer's edge (one of
   `Concrete` / `Pipe` / `Slot` / a `FedByPipe` slot). `resolve_home` lends a buffer for
   one run and threads its return home.
2. **The home invariant** — a lent buffer ALWAYS rehomes to its origin cell on
   `Checkout`/payload drop, so `cl_mem` handles stay stable across replays. This is what
   makes a graph reusable AND what lets a command buffer bake a stable handle.
3. **`DeviceOp` trait** — the one node contract. Required: `Output` + `execute` +
   `output_pipe` + `describe`. Everything else is defaulted: the gather machinery
   (`collect`/`gather_checkouts`/`into_output`), slots (`bind_slots`/`check_ready`),
   introspection (`describe`/`dump_graph`), and the **command-buffer capability**
   (`cb_addable`/`cbable_weight`/`cb_cache`/`invalidate_cbs`/`cb_restamp`/…). You can
   ignore the `cb_*` methods unless you're touching CB — they default to "not CB-able".
   `DeviceOpExt` (blanket-impl'd) holds the user verbs: `and_then`/`bundle`/`fan_out`/
   `bind`/`call`/`mutate_bind`/`sync`/`run`.
4. **Slots (reuse)** — `slot!(Tag)` is an unbound hole; `SlotState` is 5-state
   (Unbound→Bound→Lent→Severed, plus FedByPipe). The 4 bind verbs are a closed 2×2:
   set-once `bind`/`call` (consuming, infallible, deferred errors) vs reuse-loop
   `mutate_bind`/`mutate_call` (`&self`, eager errors).
5. **`ExecutionContext`** (`exec_ctx.rs`) — what `execute` receives: the context/queue,
   plus the CB walk state (`CbWalk`) and the record-time maps (sync points, reach).

## CB-as-EXECUTION-MODE (the subtle layer — skip unless touching command buffers)

Recording is a MODE of the `execute` walk, not a separate API. `ExecutionContext::cb()`
returns a 3-state `CbWalk`: `Off` (normal enqueue), `Build` (record into a live
`CbBuilder`), `LendOnly` (replay: lend buffers, add/enqueue nothing). A seam-free device
subtree opens ONE command buffer at a boundary and re-enters `execute` in `Build`/
`LendOnly`. Ordering INSIDE a CB is **sync points** (per-CB, fresh); OUTSIDE it is
`cl_event`s. Key pieces:

- `record.rs` — `CbBuilder` (live recording target), `FinalizedCb` (immutable, replayed
  once per `sync`, RAII-released), `BufHandle`/`MemRef`/`RecordableBuffer` (arg handles).
- `eager.rs` CB helpers (all `cb_*`, one contiguous block): `cb_boundary_execute`/
  `cb_boundary_gather` (open a CB), `cb_leaf_build` (a command leaf's Build prologue —
  external deps + waits + the precise-invalidation reach), `cb_forward_reach` (the
  passthrough twin), the span logic (`cb_should_open_span`/`cb_close_span`).
- **Precise invalidation**: `mutate_bind` of a slot clears only the CBs whose recorded
  commands trace to that slot. The trace is the `cb_reach` substrate (cell → slot-origin
  set, propagated at record time, carried across host seams). See
  `tests/tier2/tests/cb_precise_invalidation.rs`.
- CB acceleration engages only where the driver advertises `cl_khr_command_buffer`
  (locally: pocl). rusticl/intel fall back to per-op `execute` — same results.

## To change X, read Y

| You want to… | Read | Don't need |
|---|---|---|
| add/modify a Tier 2 leaf op (fill/copy/transfer) | the leaf's struct + `impl DeviceOp` in `eager.rs`; `Input`/`Pipe`/home; `cb_leaf_build` if it records | slot internals, other leaves |
| add a kernel arg kind | `claspr-macros/src/lib.rs` (`classify_param`/`classify_image_param`) + the matching `Kernel*Arg` trait in `launch.rs`/`image.rs` | Tier 2 combinators, CB |
| change composition (and_then/bundle/fan_out) | the combinator structs + `DeviceOpExt` in `eager.rs` | leaves, Tier 1 |
| touch slots / bind verbs | the slot machinery in `eager.rs` (`SlotState`/`SlotBinder`/`Input::Slot`/`ScalarInput`) + `tier2_macros.rs` (`slots!`) | CB, leaves |
| touch command-buffer recording | `record.rs` + the `cb_*` block in `eager.rs` + `exec_ctx.rs` (`CbWalk`, sync points, reach) | slot verbs, Tier 1 |
| touch access modes / buffer memory | `access.rs` (markers) + `buffer.rs`/`mapped.rs`/`usm.rs` | Tier 2 |

## Invariants that bite if broken

- **Home invariant** — never destroy a lent buffer; rehome it. Breaks graph reuse + CB
  handle stability.
- **CB ordering** — inside a CB use sync points, never `cl_event`s; `cb_restamp` stamps
  the one CB completion event onto output pipes for downstream consumers.
- **Reach propagation** — a command leaf must `note_slot` its arg origins and propagate
  them onto its output cell (`cb_leaf_build`); a passthrough forwards them
  (`cb_forward_reach`). Miss it and a threaded slot's CB won't invalidate on mutate.
- **`check_ready` atomicity** — `sync`/`wait_on` validate every input cell before any
  enqueue, so a bad bind fails closed with nothing run.

## Worked examples (the best way to learn a pattern)

- `examples/cg` — self-closing single-graph iteration, device-resident scalars.
- `examples/gray-scott` — reusable graph: `run_swap` (mutate-replay) vs `run_immutable`
  (curried slot meta-kernel), proven bit-identical.
- `examples/collatz` (simplest), `examples/image-pipeline` (library composition).
