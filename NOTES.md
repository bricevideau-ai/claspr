# claspr — rolling notes

Single rolling doc for active work, deferred items, and unresolved
concerns. Convention is documented in `CLAUDE.md` → "Inter-session
notes". **Append here; don't spawn new planning docs.** Prune as
items resolve.

---

## Active

### Command-buffer-backed graphs (design, 2026-06-12 exploratory)

Design converged in an exploratory session 2026-06-12. No code yet;
this section captures the agreed shape before we start implementing.

**Goal.** When the target platform supports `cl_khr_command_buffer`,
record a claspr Tier 2 graph (or recordable sub-graph) into a
finalised CB and replay it via a single `clEnqueueCommandBufferKHR`
instead of N independent enqueues per submission. Wins on submission
overhead (one driver entry instead of many) and unlocks
record-once-replay-many for iterative workloads.

**Strategic framing — why this is the priority.** CB-cached replay is
what makes Tier 2 *reusable graphs* an idiom users actually reach for.
Today there's no incentive to factor a compute pipeline into a
returnable `Graph<I, O>` value, since each call would re-walk the
DAG. With caching, `pub fn gemm(...) -> Graph<(...), (...)>` becomes
shaped exactly like a cuFFT/cuDNN/oneMKL "planned operator" —
typed, composable, cheap to invoke repeatedly. That's the abstraction
unlock: library authors can ship pipelines the way they ship
kernels, and consumers can compose those pipelines via `and_then`
across crate boundaries. The implementation decisions below flow
from that — `Graph<I, O>` is a first-class exported type, the trait
surface must not leak generics that block cross-library composition,
and the Arc-shared cache must survive composition (so a meta-kernel
containing two sub-graphs keeps both sub-caches warm across the
composite's invocations).

**Transparency is a hard constraint.** The API is `graph.call(a, b, c)`
— there is NO separate `submit_as_command_buffer` verb. The runtime
picks: CB-replay if the platform supports `cl_khr_command_buffer`
AND the graph is fully recordable AND the call's handles match the
recorded ones; otherwise walk the DAG and re-enqueue (today's Tier 2
path). User code is identical on every platform; the optimization is
silent. The only observable asymmetry is the immutable-CB
handle-mismatch case — see "Cache invariants" below — surfaced as an
explicit error with an actionable message, not as a silent rebuild.

**Implementation surface.**
- `cl3 = 0.13.1` already exposes the full FFI in `cl3::ext::*`
  (`create_/finalize_/enqueue_command_buffer_khr`,
  `command_nd_range_kernel_khr`, `command_copy_buffer_khr`,
  `command_svm_memcpy_khr`, `command_fill_buffer_khr`,
  `command_svm_mem_fill_khr`, image variants,
  `command_barrier_with_wait_list_khr`,
  `get_command_buffer_mutable_dispatch_data`).
- `opencl3 = 0.12.3` ships a safe `CommandBuffer` wrapper at
  `opencl3-0.12.3/src/command_buffer.rs`, gated behind
  `feature = "cl_khr_command_buffer"` (or `"dynamic"`). Claspr's
  workspace dep currently sets no features — would need to enable.

**Recordable surface (what fits in a CB).**
- Recordable: kernel dispatches, fills, D2D copies, image copies,
  barriers. Sync edges via `cl_sync_point_khr`.
- NOT recordable: uploads, downloads, map/unmap (host-visible), and
  any host-decided `conditional`. Those must bracket the CB (or
  split the graph at host cuts).

**Recordability tracking.** Carry a `recordable: bool` on every
`DeviceOperation` at construction time; combinators compose the bit
(`bundle(a, b).recordable = a.recordable && b.recordable`, leaf
kernel/fill/D2D = true, upload/download/map = false). O(1) check at
submit time, no graph re-walk. Subgraph partitioning (find maximal
recordable subtrees + host cut points) is a strict superset for
later; v1 requires `root.recordable == true`.

**Graph-as-typed-callable.** The graph value (not the terminal) is
the cache holder AND a typed callable: `Graph<Inputs, Outputs>` with
a `.call(a, b, c) -> Op<Outputs>` method. Move-semantic Tier 2 flow
stays the same above the line — the cached CB is implementation
detail. Per-arity variadic typing via the `KernelArgs`
macro-emitted-tuple-impls pattern already in `claspr/src/launch.rs`
(prior art). Type-level `Op<Recordable>` witness is a documented
upgrade path; runtime bit gets ~95% of the value without threading a
generic through every op type / combinator / macro.

**Two-tier capability model (clarified 2026-06-12 from spec).**
`cl_khr_command_buffer_mutable_dispatch` gates BOTH the `MUTABLE_KHR`
and `SIMULTANEOUS_USE_KHR` per-CB-creation flags. So the platform
side collapses to two tiers:

- **Tier 0**: `cl_khr_command_buffer` only. `.call()` works cached,
  immutable, one in-flight per graph.
- **Tier 1**: `cl_khr_command_buffer` + `cl_khr_command_buffer_mutable_dispatch`.
  All of Tier 0 plus opt-in flags at CB creation time (`MUTABLE_KHR`
  for `.mutate_call`, `SIMULTANEOUS_USE_KHR` for concurrent replay).

**Per-graph opt-ins via builder.** Mutability and simultaneous-use
are expressed at construction time (per spec: the flags are set at
CB creation), not per-call. Builder methods:

```rust
let g = Graph::new(factory)
    .with_mutable()       // enables .mutate_call()
    .with_simultaneous()  // enables concurrent .call() / fan_out
    .build(&ctx);
```

**`.call(args)` and `.mutate_call(args)` live on `DeviceOperation`
itself, not on a separate wrapper.** Every operation in the chain
(a leaf, an `AndThen`, a `Bundle`, a whole user-built composite)
has them. They return a `CallOp` (itself a `DeviceOperation`)
composable via `.and_then(...)`, runnable via `.sync(&ctx)` /
`.run(&ctx).await`, embeddable in `fan_out(...)`. No `Graph`,
`Cached`, `Pipeline`, or `EagerPipeline` wrapper types — they
were a design dead-end from an earlier pass; everything they did
should live on `DeviceOperation` directly.

**Args are check/update only (option 1).** The closures' captures
are the source of truth for execution; `.call(args)` verifies the
args match what was captured (strict mode), `.mutate_call(args)`
accepts compatible-different args (relaxed mode, with
`clUpdateMutableCommandsKHR` swap on recordable+mutable chains).
The chain isn't reparameterized — the args inform check/update,
not substitution. (Option 2 — true slot substitution via explicit
placeholders or proc-macro closure rewriting — is deferred until
we see a need.)

**`.call()` is composition syntax, NOT a CB-enqueue verb.** The
spec forbids nested CB enqueues (a recorded CB can't `enqueue`
another CB). So `.call()` has to flexibly:
- **Eagerly run the chain** (sync/run terminals on non-cached
  composites or non-recordable chains).
- **Enqueue a cached CB** when the chain has one cached and we're
  outside any outer recording context.
- **Inline the chain's commands into the outer CB recording**
  when we're recording a parent CB that contains this `.call`.
The runtime picks the right path based on context; the user just
writes `op.call(args).and_then(...)`.

| Method | Required opt-in | CB + mutable_dispatch | CB only (no MD) | No CB |
|---|---|---|---|---|
| `.call(a, b, c)` (stable, single in-flight) | none | replay cached CB; error on handle mismatch | replay cached CB; error on handle mismatch | walk DAG, enqueue |
| `.call(a, b, c)` (concurrent, e.g. fan_out) | `.with_simultaneous()` | replay cached CB concurrently | **fall back** to walk DAG | walk DAG |
| `.mutate_call(a, b, c)` | `.with_mutable()` | `clUpdateMutableCommandsKHR` + replay | **fall back** to walk DAG | walk DAG |
| `fan_out(.., \|i\| g.mutate_call(i))` (cached batch) | `.with_mutable().with_simultaneous()` | same CB, per-call updates, concurrent | **fall back** to walk DAG | walk DAG |

The fallback row makes opt-ins *portable*: a graph built with
`.with_mutable().with_simultaneous()` works correctly on every
platform — it just only gets the cached-CB optimization where the
extensions are available. Users never have to check
`device.has_extension(...)`.

**Where simultaneous-use comes in.** Without `.with_simultaneous()`,
N concurrent fan_out branches calling `.mutate_call(...)` on the same
graph would race on the cached CB's destination buffers. The spec
makes this explicit and opt-in (with non-trivial per-submission
overhead — that's why it's not the default). With the opt-in, the
user is asserting their per-call arg updates make destinations
independent, and the spec guarantees concurrent enqueue is well-
defined. The `spikes/graph_devop_record/src/batch_example.rs::cached_simultaneous_fan_out_batch_pattern`
test demonstrates the protocol end-to-end:
`record_count = 1`, `replay_count = BATCHES`,
`update_mutable_count = BATCHES - 1`.

**Recording mode is decided at construction (revised 2026-06-12).**
Earlier the design said "first-touch decides mode" — but the spec
puts `MUTABLE_KHR` and `SIMULTANEOUS_USE_KHR` at CB *creation* time,
not at command-record time. So the cleaner mapping is: opt-ins
(via a method like `.cb_record_mutable()` / `.cb_record_simultaneous()`
or similar — bikeshed TBD) set flags on a `DeviceOperation` before
the first run.

**Future option: heuristic auto-CB.** The runtime already has the
data it needs (`recordable: bool` from the type system + call_count
+ chain length) to decide automatically when materializing a CB is
worth it: recordable AND called > N times with same handles AND
chain bigger than threshold M ops. Could ship the auto-heuristic
as the default and keep the explicit opt-ins for (a) users who want
guaranteed caching from first call, (b) users who need mutable or
simultaneous (those have non-trivial recording overhead so opt-in
makes sense), (c) predictable benchmarking. Auto + opt-in coexist.

**Cache invariants (CB-capable path).**
- Graph holds `Mutex<Option<CachedCB>>`. Cache stores the recorded
  CB, its mode (from the construction-time opt-ins), the canonical
  `cl_mem`/SVM handles it was recorded against, and the queue
  handle.
- `.call(a, b, c)` (graph built without `.with_mutable()`):
  - Cache empty: record immutably for this `&ctx`, store, enqueue.
  - Cache hit (same handles + same queue): replay.
    - If `.with_simultaneous()` opted in: concurrent replays OK.
    - Otherwise: error if a prior Op is still in flight.
  - Handle mismatch: **error** — "graph called with different
    buffers than recorded — pass the same buffers, switch to
    `.mutate_call(...)` (requires `.with_mutable()` at construction),
    or build a fresh graph per call shape." NOT a silent rebuild.
- `.mutate_call(a, b, c)` (graph built with `.with_mutable()`):
  - Cache empty: record mutably, enqueue.
  - Cache hit, same args: replay.
  - Cache hit, different args: `clUpdateMutableCommandsKHR` swap,
    enqueue.
  - Concurrent: gated on `.with_simultaneous()` (same as `.call`).
- Common rules:
  - Different queue: **only legal if the previous Op has been
    awaited/dropped**; then release old CB, re-record for new queue.

**Cache behaviour (CB-incapable path).** Without
`cl_khr_command_buffer` (or with a not-fully-recordable graph),
both `.call()` and `.mutate_call()` walk the DAG and enqueue each
op with its event dependencies — exactly today's Tier 2 path.
There's no recorded state to invalidate; arbitrary call counts and
handle changes are fine. The "one in-flight per graph" invariant
still holds (it's a data-race protection, independent of CB).
Result: the same user code is correct on every platform, faster
where the extension is available.

**One in-flight per graph — conditional on opt-in.** OOO queues mean
naive concurrent replays of a cached CB would race on its buffers.
The spec's resolution: opt into `CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR`
at CB creation (gated on `cl_khr_command_buffer_mutable_dispatch`).
Without the opt-in: invariant is "at most one outstanding
`.call(...)` per graph" — silently allowing concurrent replay would
hide a data race. With `.with_simultaneous()` at construction:
unlimited concurrent replay is permitted; the user is asserting
that their per-call arg updates make destinations independent. This
is exactly the cached-fan_out batch-inference pattern (see "Two-tier
capability model" below).

**Composition consequence: `and_then`-reuse becomes first-class.**
The "one in-flight" rule doesn't lose expressiveness because
`G.and_then(|_| G).and_then(|_| G)` records into one CB with
internal sync-point edges connecting iteration k's tail to k+1's
head. Single `clEnqueueCommandBufferKHR`, three iterations,
OOO scheduler still overlaps *within* each iteration. This is a
new composition pattern only really expressible with CB-backed
graphs (today you'd have to physically rebuild the underlying ops).
Implies graphs must be cheaply re-usable as sub-graphs — `Clone`
via `Arc<Inner>`, cache slot lives in the `Arc`'d inner so clones
share it.

**Test/runtime target.**
- Native: pocl 7.2-pre (`~/local/pocl`) carries
  `cl_khr_command_buffer` 0.9.6 + `mutable_dispatch` + `multi_device`
  + pocl-specific SVM/host-buffer extras. Distro pocl 6.0 has the
  base extension at 0.9.4 + `multi_device`, no `mutable_dispatch`.
- Emulation: [bashbaug/SimpleOpenCLSamples
  `layers/10_cmdbufemu/`](https://github.com/bashbaug/SimpleOpenCLSamples/tree/main/layers/10_cmdbufemu)
  — Apache-2.0, `OPENCL_LAYERS`-style ICD-loader layer. Implements
  `cl_khr_command_buffer` v0.9.8 + `mutable_dispatch` v0.9.5. No
  `multi_device`. Stacks transparently over rusticl + NEO legacy
  (both OpenCL 2.1+, which the layer requires for `clCloneKernel`).
  Proof-of-concept quality (thread-safety, `FINALIZED_KHR` state,
  some error checks not perfect) — use for semantic coverage, not
  perf. Iris Plus via NEO legacy and rusticl/llvmpipe lack the
  extension natively; emulation gets them in.

**What's left to scope before coding.**
- ~~Concrete `Graph` trait shape + `Inputs`/`Outputs` associated
  types + how `and_then` / `bundle` / `fan_out` compose them.~~
  Resolved 2026-06-12 via `spikes/graph_cb/` — see findings below.
- Final names for the two call verbs (`.call()` / `.mutate_call()`
  is the working draft — `.call_mut()` reads as "mutates the
  receiver" which is wrong here, `.call_with()` is vague; settle at
  implementation time).
- Whether per-op profiling needs a different surface inside a CB
  (the extension only exposes whole-CB timestamps, not per-command).
- CI plumbing: enabling `opencl3 features = ["cl_khr_command_buffer"]`
  on the workspace dep, picking up the cmdbufemu .so build, and
  fitting it into the existing rusticl/llvmpipe matrix entry. Likely
  paired with the deferred pocl-7.2 ICD work — see `claspr CI
  deferred` in auto-memory.

**Spike findings (2026-06-12, `spikes/graph_cb/`).** Trait system is
NOT a blocker. Validated shape:

- **Single struct `Graph<I, O>` (no trait hierarchy).** Library
  authors can name the type directly in `pub fn` signatures, every
  combinator returns the same `Graph<I, O>` (just with new type
  params), no `AndThen<Bundle<..>, ..>` type-name explosion. DAG /
  IR is type-erased inside `Arc<GraphInner>` (cheap clone, future
  home for the cached CB slot + recordability bit).
- **`PhantomData<fn(I) -> O>`** carries the type params with the
  right variance (contravariant in I, covariant in O) and clean
  auto-trait behaviour.
- **Per-arity `.call(a, b, c)` via inherent-impl macro** (1..=8,
  same shape as `KernelArgs` in `claspr/src/launch.rs`). Wrong arity
  is a crisp `E0061: this method takes N arguments but M were
  supplied` with a "consider adding" hint.
- **`and_then` as a single generic method**: `fn and_then<O2>(self,
  next: Graph<O, O2>) -> Graph<I, O2>`. Type mismatch is a clean
  `E0308: mismatched types — expected Graph<(BufU32,), _>, found
  Graph<(BufU32, ScalarU32), (BufU32,)>` pointing at the call site
  and naming both expected (self's Outputs) and found (next's
  Inputs) tuple shapes.
- **Library-boundary export works**: a function in a sibling module
  returns `Graph<(BufU32,), (BufU32,)>` and the caller composes it
  via `and_then` with a local graph — no impl-trait, no boxed dyn,
  no leaking implementation generics. This is the meta-kernel
  pattern working end-to-end.
- **`Clone` via `Arc<Inner>`** enables `G.and_then(G.clone())`
  reuse cheaply, as required for the in-CB iteration pattern.

Compile-fail diagnostics captured inline in `spikes/graph_cb/src/main.rs`
(commented; will migrate to a real `ui_test` harness when the design
graduates from spike status). Next phase: wire the `Graph` to real
claspr `DeviceOperation` IR and start on the `cl_khr_command_buffer`
FFI side.

---

## Deferred

### Inherit generated kernel deps from host workspace

`claspr-build`'s generated kernel `Cargo.toml`
(`claspr-build/src/lib.rs`, `write_generated_cargo_toml`) still
hardcodes `spirv-std` and `num-complex` to floating refs. The host
workspace pins them via `Cargo.lock` and `seed_lockfile_from_host`
copies that lock into the kernel sub-crate at build time, so the
current setup is correct *for consumers built inside the claspr
workspace* — but a kernel crate built fresh in some other workspace
would re-resolve against the floating branch ref.

Approach (sketched): walk up from `OUT_DIR` to find the host
`Cargo.lock`, extract the pinned `rev` for spirv-std/num-complex,
write those into the generated TOML. Fallback to today's hardcoded
branch refs if no lockfile found.

**Status (2026-06-11):** the original blocker (rust-gpu's glam
reshuffle) cleared with upstream's `ce16d0bb680` → `762e9d61272`
saga (finalised 2026-06-08); the rebase brought it in via
`4de1a13`. Glam itself is no longer in the generated TOML at all —
spirv-std re-exports it via `pub use glam;` and its default
`glam_0_33` feature enables exactly the type families kernel code
uses (u32/i32/f64/usize/u64 + libm). Kernel code now writes
`spirv_std::glam::USizeVec3` instead of `::glam::USizeVec3`.
Remaining unstarted work: lockfile-walking for spirv-std + num-complex.

### Tier 1 scoped launcher (`ctx.scope(|s| {...})`)

Original `DESIGN-NOTES.md` #4 sketched a SYCL-style scoped launcher
mechanism. Resolved differently — see `git log -- DESIGN-NOTES.md`
or commit `2ba935a` for the actual landing (ops carry ctx + no-arg
`.wait()`/`.submit()`). A real scope object only becomes
worth-it if bundled with profile-region semantics, scope-wide event
tracking, or queue-model-at-boundary defaults — none of which have
ergonomic pressure today.

**Revisit trigger:** a user actually wants any of those scope
extensions.

### Tier 1 capability gaps already covered elsewhere

- `DeviceSlice::map` Tier 1 ✅ shipped 2026-06-09 (commit `311db59`).
- Non-blocking `MappedSlice::map` ✅ shipped same commit.
- Cross-queue SVM Drop race fix ✅ shipped same commit.

---

## Concerns

### Image format dispatch in the proc-macro

Pre-existing item from REVIEW.md 2026-05-28. The proc-macro emits
`&Image2DRgba8` for every `&Image!(...)` kernel param; the runtime
side (`claspr::Image2D<A, F>`) is fully generic over format. The
gap is in the macro's dispatch only. Lives in the README's
Limitations section too.

### Cross-device + Arc-split test coverage is thin

REVIEW.md 2026-05-28 item 7a/7b. `tests/tier2/cross_device.rs` is
~2 tests; `tests/tier2/arc_split.rs` is sparse on assertions.
Marginal — works as documented; doesn't bite anyone today. Worth
incremental hardening when the cross-device path lands more usage.

### Library-crate transitive spirv-std dependency

Library kernel crates (mandelbrot-kernel, sobel-kernel) cfg-gate
spirv-std imports to keep consumers free of the transitive dep.
Helpers that take device-only types (e.g. `cl::Float3`) can't be
host-callable in that pattern; restructure to primitives or switch
to the mixed-host-and-kernel library pattern (regular spirv-std
host dep). Documented as a gotcha in `CLAUDE.md`.

---

## Recent landings (last ~5, prune as new items land)

| Commit | What |
|---|---|
| `2ba935a` | Ops carry ctx → no-arg `.wait()` / `.submit()` shortcut + rename to `.wait_on(&L)` / `.submit_on(&L)` for cross-queue case. 192 call sites migrated. |
| `311db59` | Tier 1 `DeviceSlice::map` + non-blocking `.submit()` terminal on both DeviceSlice + MappedSlice map ops; closes latent cross-queue SVM Drop race. |
| `a9d825a` | README "Other modes" section — pre-compiled + external SPIR-V via `claspr::kernels!`. |
| `f82ffd3` | Sealed `KernelOp` (proc-macro is the only legitimate impl producer). |
| `cc2437d` | Tier 2 alloc macros as sugar over `alloc_uninit + .fill()/.write()`. |
