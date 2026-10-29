# claspr — rolling notes

Single rolling doc for **active work, deferred items, and unresolved concerns**.
Convention (see `CLAUDE.md` → "Inter-session notes"): append here; don't spawn new
planning docs; prune as items resolve. **Resolved history lives in git commit
messages, not here** — `git log` is canonical. For the stable mental model of how
the codebase is laid out, read `ARCHITECTURE.md` (the map for agents/newcomers).

---

## Active

Nothing in flight on `main` as of 2026-10-29. Parked, deliberately unmerged:
branch `typed-slot-sets` (compile-time slot sets, see the DECLINED entry below).
Recent work is in the landings table at the end.

### Partitioned kernel args — prototype on branch `partitioned-args` (2026-10-29)

Inspired by arXiv 2608.13759 ("GPU Offload in Rust", partitioned views). A kernel
param `PerItem<T>` hands each work-item ITS element (`*out = v`, `out.index()`);
claspr-build lowers it to `&mut [T]` + a per-item view (adding a global id if
none), and the host checks at every launch that the grid is 1-D and exactly the
buffer's length (`Error::GridMismatch`, lent inputs returned → retryable). llama2's
7 per-element kernels use it: byte-identical, no device cost (148.6 vs 148.1 tok/s).
`PerItemIn<T>` is the read-only form (inputs read element-wise; the grid check
bounds them too — llama2's `add`/`swiglu`/`rmsnorm` weights use it, no cost).
`PerItemOut<T>` is the write-only form (lowered to `&mut [MaybeUninit<T>]`); via
`KernelSliceWriteArg` it accepts a `DeviceSliceUninit` and outputs the initialised
`DeviceSlice` (home adapter forgets init on return), incl. `DeviceScratch` memory —
the safe version of the documented "kernel-write-only + assume_init" pattern.
**`PerItemOut` writes are compile-time enforced:** `write(self, v)` consumes the
view and returns a `Proof<marker>`; a kernel with `PerItemOut` params must return
`Written` (generated per kernel, one field per output), so a missing output, a
double write, another output's proof, a forged proof or a bare `return;` are all
compile errors, and early returns must prove too (claspr-build wraps the body in a
closure returning `Written`). llama2's embed/rmsnorm/matmul/att_mix outputs use it
over `alloc_uninit` buffers: byte-identical, 1 CB, no cost (151.0 vs 150.9).
Uninitialised `MappedSlice`/`USMSlice` outputs work too. Partitions are N-D: a
1-, 2- or 3-D grid with one work-item per element, work-item `(x, y, z)` owning
element `x + w * (y + h * z)` (`coord()` gives the coordinates); gray-scott's 2-D
`laplacian`/`combine` use it (frames byte-identical). Not done: named handles.

### Decisions — assessed & DECLINED (kept so future sessions don't re-litigate)

- **`meta_kernel!` macro** (#241) — DECLINED. A repo-wide scan found `cg`'s
  `solve_with` is the ONLY generic-subgraph fn; every other reusable-graph pattern
  uses type-inferred CLOSURES with zero signature boilerplate. The macro would add a
  DSL + opaque expansion errors (a cost-of-entry COST) to save ~15 lines at exactly
  one site that `OutputShape` (now `OpShape`) already made readable. **REVISIT TRIGGER: 3+ generic-fn
  subgraph authors accumulate.** Not the same thing as `graph!` (landed 2026-10-29):
  that is straight-line `let` syntax for a sequence of steps, not a DSL for
  generic subgraph signatures, and it does not trip this trigger.
- **Leaf memory-family unification** (`leaf_fill!`/`leaf_write!`) — DECLINED.
  **RE-CONFIRMED 2026-08-21 by measurement**: `leaves.rs` is 19 `DeviceOp` impls;
  18 of them average ~55 lines with 5–11 members, i.e. genuinely irreducible
  per-op work, and they really do differ across verb × memory-family ×
  init-state. A 5-axis muncher would read worse than ten honest structs. What the
  measurement DID surface as open (see Deferred, below). The
  fill/write families vary on ~5 axes (input/output type, in-place-`put_home` vs
  uninit-`put`, cl_mem vs SVM handle + enqueue fn, CB-recordable vs host); a macro
  would trade skimmable duplication for a fiddly muncher — a net loss for the
  cost-of-entry goal. The truly-identical alloc-uninit family WAS collapsed
  (`impl_alloc_uninit!`).
- **Concrete-head terminal boilerplate** (the ~6 `wait`/`submit` pairs) — DECLINED.
  They carry useful per-op doc comments (each op's Tier-1 spelling); a macro would
  drop those or need doc-passthrough (negating the win).
- **`scalar_slot_fed_pipe` golden pinned an `eager.rs` line number** — FIXED
  2026-08-21 (`f8bcc8b`). A `stderr_filter` in `tests/support/src/ui.rs` now
  rewrites `<claspr path>.rs:N:C` to `:LL:CC` in compile-fail goldens; the file
  stays (real signal: which claspr item imposed the bound), only the volatile
  position is blanked. It had cost three spurious re-blesses in one session and
  was latent in five more goldens (launch.rs, buffer.rs, host_view.rs). NB the
  regex spells classes out (`[a-zA-Z0-9_]`, not `\w`) — ui_test's regex build
  has unicode-perl off.
- **`DeviceOp` trait split** (`CommandBufferOp` supertrait) — DECLINED (R1). Would
  cascade `+ CommandBufferOp` bounds through ~7 CB helpers + every combinator (~53
  impl splits, ~60 call sites) with ZERO object-safety benefit (`dyn DeviceOp`
  works unsplit: `DeviceDynOp` boxes one since 2026-10-27, once the gather
  members' vestigial `Self: Sized` gates were dropped). The "CB is a skippable layer" goal is met by the module split instead.
  **RE-CONFIRMED 2026-08-21 on a different axis** (attempt 2 asked about AGENT CONTEXT
  cost, not impl cost — a fair re-open, and it still declines): measured across the 33
  in-tree impls, **45% override at least one `cb_*` member** (`cb_addable` 14,
  `cb_cache` 11, `cbable_weight` 11). CB is therefore NOT a severable minority layer —
  a supertrait would relocate the members without removing the thinking, and nearly
  half the ops would implement both traits. What the measurements DID justify was the
  documentation fix that landed instead (`7d240ac`): the trait is 24 members but the
  median impl overrides **7**, and nothing marked the tiers. The trait's rustdoc now
  carries a measured tier table (1 core / 2 multi-output / 3 slots / 4 CB /
  5 introspection), in-body banners mirror it, and the deepest CB internals are
  `#[doc(hidden)]`. **REVISIT TRIGGER: if CB participation drops well below ~45% (the
  tier-4 defaults become the overwhelming norm), the severable-layer premise becomes
  true and the split is worth re-pricing.**
- **Type-state slots for Tier 2** (graph type carries the unbound-tag set) —
  BUILT, then NOT MERGED (2026-10-29). The old "hard wall" verdict (repeated tags
  need type equality) was wrong in practice: branch **`typed-slot-sets`** (kept on
  origin, rebased on main b94809c, all tests green) tracks slots through every
  op, leaf and combinator. Keys: `DeviceOp::Slots` (type-level list,
  `eager/slot_sets.rs`), launchers derive it from arg types, `bind`/`call`
  `Remove`, terminals gate on `AllBound`; `call_all` binds everything at once
  (repeats free); `erase_as::<SlotSet![..], _>()` declares the set for a partial
  curry of a repeated tag; nodes store children `Bare` with the set on an outer
  `Slotted` wrapper — without that, monomorphization doubled miniweather's build.
  Cost vs main: +0–5% build, +3–7% binary.
  **Why not merged:** it moves errors that the runtime already reports
  deterministically on the first `sync` to compile time, but adds permanent cost —
  user (`erase_as` + raw E0283 when forgotten, `Slots = HNil` in generic code,
  first fill must be `bind`, `Slotted` returns, extra `_` turbofish), contributor
  (`into_bare` on every op; a combinator that skips the bare pattern silently
  brings back the 2× build), and it deletes none of the runtime checks (`Dynamic`,
  `mutate_*`, readiness stay). llama2 ended up slot-free once made fast:
  per-step state belongs on the device; slots are configuration.
  **If revisited: the library-boundary variant** — ops stay untracked, only a
  declared erasure (`DeviceDynOp<'_, S, SlotSet![..]>`, a library's public graph
  type) carries a set that `bind`/`call` consume and `sync` requires empty.
  Typed contracts where they pay, no `Bare`/`Slotted`, no generic bounds; the
  declaration itself is unchecked (runtime catches a wrong one).

---

## Deferred

### Write-only kernel args → graph-internal uninit scratch (`&mut [MaybeUninit<T>]`)

DESIGN-ONLY (Brice thinking; no code). A kernel scratch output (gray-scott's
`laplacian` `lap_out`) is semantically write-only but the macro types every `&mut [T]`
as ReadWrite (`classify_param`), so it can't be a graph-internal auto-allocated (uninit)
buffer. Marker decided (2026-07-03): use stdlib `&mut [MaybeUninit<T>]` (honest across
device/host/software-pass), NOT a bespoke attribute. Probe DONE: rust-gpu compiles
`&mut [MaybeUninit<f32>]` clean today (transparent → plain `f32` + `OpStore`), so the
representation blocker is gone. Remaining: (a) macro recognizes the marker + emits a
`KernelSliceWriteArg` accepting `DeviceSliceUninit`; (b) runtime write-only path +
graph-internal auto-scratch riding the home invariant; (c) OPTIONAL never-read lint
(decidable) — total-coverage is undecidable in general but PROVABLE for the happy path
(auto-allocated grid-sized scratch + total-map kernel `out[gid]=…`, which gray-scott's
lap is). Size derivation: grid-shaped is free (derive from LaunchSpec); non-grid sizes
need an author-declared closure `alloc(slot!(Grid), |g| …)` (start here — no new IR) or a
reified `SizeExpr` (only when inspection/serialization/CB-cache-key forces it). The
`#[auto]` write-only producer output dissolves gray-scott's two-sets aliasing + threading.

### `leaves.rs` orientation + `CopyTo2` (measured 2026-08-21, not started)

Two findings from measuring `eager/leaves.rs` (2,906L, ~1,942 code, 19 `DeviceOp`
impls) while looking for refactors worth doing:

- **Doc density is 11%**, vs 44% in `eager.rs`. This is the file agents WRITE in
  (19 of the engine's 33 op impls) and the least explained. The type names encode a
  verb × memory-family × init-state matrix (`Fill`/`FillDeviceUninit`/
  `FillMappedUninit`/`FillUsmUninit`/`FillMapped`, `WriteDevice`/`WriteDeviceUninit`/
  `WriteMapped`/`WriteMappedUninit`/`WriteUsmUninit`, …) that a newcomer must infer by
  pattern-matching. A ~30-line module `//!` header stating the axes, which
  combinations exist and why some don't, and the member set a new leaf needs, would do
  for `leaves.rs` what the tier table did for `DeviceOp` (`821cf94`) at the same cost
  and risk. NOT a refactor — the impls themselves are fine (see the re-confirmed
  leaf-unification DECLINE above).
- **`CopyTo2<Src, Dst>` is an outlier**: 270L / 17 members, ~5x the file's median and
  ~10% of it. Probably the src × dst family cross-product plus CB recording. Worth
  measuring ON ITS OWN before deciding whether it is irreducible or genuinely tangled;
  everything else in the file is 26–99L.

### Inherit generated kernel deps from host workspace

`claspr-build`'s generated kernel `Cargo.toml` (`write_generated_cargo_toml`) hardcodes
`spirv-std`/`num-complex` to floating branch refs. Correct for consumers built inside the
claspr workspace (`seed_lockfile_from_host` copies the host lock), but a fresh external
build re-resolves against the floating ref. Fix: walk up from `OUT_DIR` to the host
`Cargo.lock`, extract the pinned revs, write them into the generated TOML (fallback to
today's refs). Glam is no longer in the generated TOML (spirv-std re-exports it).

### Command-buffer reuse under mutation: keyed cache landed, mutable dispatch declined

**Keyed LRU landed 2026-09-04** (`CbLru`), **re-keyed 2026-10-27** (`43605dc`): a
graph keeps `current` + up to `CB_LRU_CAP` (4) parked command buffers, each filed
under what its slot CELLS held when recorded (cell id → value digest, so renamed
and erased sites key correctly). Rotation happens at mutate time, not replay time;
rebinding a slot to the value it already holds is free. A pipe-fed slot has no
digest, so a mutate touching one falls back to a hard clear. On the 24-command
miniWeather shape: host API calls −91%, wall-clock 4–8% on PoCL (its recording is
cheap; see "Where per-sync time goes" below). miniweather's natural per-timestep
shape costs +6% over the unrolled one and records 3 CBs per run, not one per step.

**Mutable dispatch** (`cl_khr_command_buffer_mutable_dispatch`, update by-value
scalars/extents in place) **declined for now**: experimental, and PoCL is about the
only driver advertising it. Revisit if that changes.

**What the keyed cache cannot fix** (found with `examples/llama2`, 2026-10-29): a
scalar slot whose value changes EVERY step (a token position) keys a new CB per step,
so nothing is ever replayed (200 tokens → 200 recordings). The fix is in the graph,
not the cache: keep per-step state on the device as data (llama2's `Step` in a
`DeviceScalar`, advanced by its last kernel) → 1 recording per run, +6% on PoCL.
Now in the eager guide ("Per-step state: keep it on the device"), pinned by
`eager_scalar_ref::device_resident_struct_state_advances_per_sync`.

### Generic device-module instantiation — `#[claspr::device(instantiate(Real = [f64, f32]))]`

**LANDED 2026-08-21** (showcase `tests/instantiate`, full-scale `examples/miniweather`:
one `fn run<Real: Width, K: gpu::Kernels<Real>>` drives the 24-dispatch graph at both
widths, bit-identical to the explicit-compile port). Design and history are in the git
log and CLAUDE.md. Still open:
- Tier-1 inherent terminals (`.wait()`/`.submit()`) aren't visible through the
  generated trait's opaque `impl DeviceOp` returns — generic code uses the
  `DeviceOpExt` verbs.
- Kernels with image params (or no buffer output) are left out of the generated
  trait (listed in its docs); call them through the stamp modules. Revisit when an
  instantiated module uses images.
- Per-stamp vector-family aliases (`RealVec3` → `DVec3`/`Vec3`, …) not done. Probe
  answered 2026-08-21: generic `#[rust_gpu::vector]` structs work INSIDE kernels, but
  vector-typed kernel PARAMETERS need the aligned concrete `cl::*` types, so aliases
  remain the bridge for signatures.
- The stamp-module level shifts `super::` paths in the body by one (document if it bites).
- Explicit-compile mode validates only entry-point NAMES at bind; a bind-time
  `CL_KERNEL_NUM_ARGS` count check would catch added/removed params at startup. Do
  NOT build on `clGetKernelArgInfo` for SPIR-V (semantics in flux at the OpenCL WG,
  Brice 2026-08).

### CB region choice: maximal munch optimizes the wrong objective

Raised 2026-09-03, no code. The boundary algorithm (`cb_should_open_boundary`) is
greedy maximal-subtree tiling — textbook maximal munch. For the pure subtree rule that
is actually OPTIMAL for coverage and CB count: the graph is a term tree of combinators,
so every candidate region is convex by construction and the maximal addable subtrees
form a unique antichain any other partition refines. (The general problem — convex
partitioning of a DAG — is NP-hard; claspr dodges it by partitioning syntax, not
dataflow.)

But coverage is not the objective worth maximizing. A bigger tile costs twice:

1. `cl_khr_command_buffer` has no per-command event wait and returns ONE completion
   event, so a CB is a join/fork barrier — the first command waits on the last external
   producer, and every consumer waits on the whole CB. Maximal tiles maximize that.
2. A bigger tile is more to re-record when anything in it is touched. The keyed cache
   above softens this but does not remove it (a cache MISS re-records the whole tile).

So the right objective has a cut cost and a mutation-frequency term; greedy has neither.
Tractable improvement if it ever matters: Aho-Johnson-style DP over the term tree with
state "is this node inside the parent's tile", costing each cut by crossing dependence
edges. Caveat: pipes threading across bundle branches overlay a DAG on the tree, which
may drag it back toward the NP-hard formulation.

Also note `cb_spine_head_addable` (the interior-seam span rule) is NOT subtree-shaped
and is genuinely greedy-suboptimal — it takes the longest addable spine PREFIX, and the
`closed` latch then forces independent post-seam work out of the CB on syntactic
position rather than dataflow reachability.

### Tier-2 papercuts from the same port (small, independent)

- **Arc args come back in the output tuple.** The typed launcher returns Arc buffer args
  like any other buffer arg, so an `and_then` chain over Arc-heavy kernels destructures
  and drops them at every transition (`|(s, t, fx, td, _hd, _hdt)| …`) — the next site
  binds a fresh clone anyway. Harmless but noisy at 48 dispatches. Options: an
  Arc-of-read-only variant that is NOT threaded into the output tuple (it has no home to
  rehome — the clone is the ownership story), or just show the destructure-and-drop
  idiom in the fan-out docs so chain authors know it's expected.

### Erasure leftovers: shape-strict arms (2026-10-26, not started)

`OpShape` + transparent `.erase()` landed; `and_then` now erases the chain it
extends by default, so users never cut long chains by hand. Two loose ends, neither needed by any in-tree code:
- `Seamed<Single<T>>` ≠ `Single<T>` though their projections match, so a
  conditional arm ending in `and_then_host` won't unify with a plain kernel arm.
- The old `DeviceDynOp<T>` unified arms by `Output` alone (collapsing to one
  pipe); the transparent one needs equal SHAPES, so a `value(t)` arm
  (`ByValue`) and a kernel arm (`Single`) no longer mix. A `.collapse()` (own
  pipe, `collect_home` to keep the home, CB-opaque) would restore it.
**REVISIT TRIGGER: the first real graph that needs either.**

### Tier 1 scoped launcher (`ctx.scope(|s| {...})`)

Sketched but resolved differently (ops carry ctx + no-arg `.wait()`/`.submit()`, commit
`2ba935a`). A real scope object only earns its keep bundled with profile-region
semantics / scope-wide event tracking / queue-model-at-boundary defaults. **Revisit
trigger:** a user wants one of those.

---

## Concerns

### Where per-sync time actually goes (profiled 2026-09-08) — NOT the walk

Closes the deferred "should we skip the LendOnly walk on a cache hit?" question
from the keyed-cache work. Answer: **no**, and the reason is not what the plan
assumed. `perf` is unavailable here (`perf_event_paranoid=4`), so this was
measured two other ways.

**The replay walk is a rounding error.** `cb_bench` gained a `walk-lean` /
`walk-heavy` pair: identical two-command CBs over the same buffer, differing only
in 24 zero-command `forward` passthrough nodes. Median per-sync 93.8/91.8/94.5 vs
91.4/97.1/95.3 us over three runs — a ~1 us delta for 24 nodes, inside the noise.
That is **~50 ns per node**. A 100-node graph spends ~5 us of a ~94 us sync in the
walk. Skipping it would buy under 5% while touching the home invariant and the
lend/rehome lifecycle, which is the riskiest part of Tier 2. Not worth it.

**It is pocl scheduler synchronisation.** CPU time exceeds wall time (0.55s vs
0.40s: multi-threaded) and `sys` beats `user` 4:1. `strace -c -f` says **99.16% of
syscall time is futex**. Futex traffic scales with COMMANDS, not syncs:

| workload | commands/sync | futex/sync |
|---|---|---|
| walk-lean | 2 | 69.5 |
| control-deep | 24 | 711.7 |

i.e. **~29 futex ops per command + ~11 per sync**. So on pocl's CPU device a
command buffer avoids the RE-RECORD but not the per-command dispatch handshake —
each command still goes to the worker pool individually. That explains why the
keyed cache's 91% API-call reduction only moved wall-clock 4-8%, and why
miniWeather's per-timestep shape costs ~6% (it pays the per-sync constant twice
as often).

Consequence for claspr: there is no host-side lever left worth pulling here. The
remaining cost is a runtime property. On a driver where CB replay is one hardware
submission the picture should differ sharply — worth re-running this on Intel
GPU/Level Zero before drawing conclusions about CB value in general.

### PoCL: sustained CB replay fails `CL_INVALID_OPERATION` — fixed LOCALLY only

Replaying one finalized command buffer in a `sync` loop returned `-59` from
`clEnqueueCommandBufferKHR` at a random iteration. PoCL race (diagnosed 2026-09-04):
`pocl_update_event_finished` publishes `CL_COMPLETE` before decrementing the buffer's
`pending` count, so a waiter arriving in that window re-enqueues while `pending != 0`.
claspr is conformant (waiting on the completion event before re-enqueueing).

The fix (branch `fix/cmdbuf-pending-race`, repro `scratch/pocl_cmdbuf_replay_race.c`)
is installed in the local pocl 8.0-pre (`~/.local`), so `cb_bench` and the suites run
clean here. **Upstream it was NOT merged**: pocl/pocl#2306 was closed 2026-09-09 — the
maintainer would rather move the implementation to the v1.0 command-buffer spec
(0.9.8 drops simultaneous-use and the `pending` bookkeeping entirely), tracked as
pocl/pocl#2314 (open). So stock pocl still has the race — one more reason the pocl
CI job stays deferred until a suitable pocl is easy to install.

### rusticl (Mesa 26.2.5) driver bugs that show up in CI — reported, do not gate

- **mesa#16714** — llvmpipe SIGSEGVs JIT-compiling any kernel that calls `fma()` on
  `double` (miniweather's f64 kernels). CI runs miniweather with fp64 hidden on that
  Mesa (`6c2b322`); locally `RUSTICL_FEATURES=fp64` reproduces it.
- **mesa#16723** — heap corruption (`llvmpipe_set_global_binding` SIGSEGV or glibc
  `malloc(): corrupted top size`) when kernels in flight on two queues are joined by
  one cross-queue sync. Surfaces as a ~15% flake of `tier1 cross_queue::
  after_all_orders_launch_after_multiple_cross_queue_events`; reproduced on x86-64 and
  aarch64.

### Other driver quirks and coverage gaps

- **Intel NEO (legacy) crashes on malformed SPIR-V** — a valid-header module with a
  scrambled instruction stream SIGSEGVs inside the driver's IL frontend, and a
  half-truncated module is silently ACCEPTED (no validation). `tier1/errors.rs`
  gates its corrupted-module case off Intel devices with a loud SKIP. Driver
  defect; legacy NEO is effectively frozen so an upstream report is low-value —
  recorded here so nobody re-diagnoses the SEGV. Same laxness family: it does
  not enforce `CL_INVALID_CONTEXT` on cross-context enqueues (the write just
  succeeds), so `eager_cross_context.rs`'s reseed-failure tests behavior-gate
  themselves (loud SKIP when the replay unexpectedly succeeds).
- **PoCL device-init race (upstream-report candidate)** — concurrent first-touch
  `clGetDeviceIDs` on PoCL 8.0-pre transiently returns ZERO devices to threads that
  race another thread's first call (recovers ~100s of ms later; rusticl unaffected;
  reproduced 100% with 8 threads pre-fix). claspr works around it by serializing
  platform+device enumeration behind `CL_ENUM_LOCK` (`device.rs`), pinned by
  `tier1/concurrent_enumeration.rs` (threads-first, so the process's first
  enumeration is the concurrent one). Worth a minimized C repro + pocl issue —
  same lazy-init path our arg-info PR (#2166) territory. Surfaced only when test
  skips became loud (`CLASPR_REQUIRE_DEVICE`): the two access_frozen victims had
  been silently "passing" as skips.

- **Image dispatch is family-level, not format-level** — the macro derives
  dim/arrayed/family/access from `Image!(...)` and bounds the launcher generic, but the
  concrete channel format isn't part of the bound (Kernel-target SPIR-V carries only the
  sampled type; `format=` is Shader-only), so channel-format mismatches surface at
  runtime. Near the ceiling of what's statically checkable. (README Limitations.)
- **Cross-device + Arc-split test coverage is thin** — `cross_device` ~2 tests,
  `arc_split` sparse. Works as documented; harden when cross-device gets more usage.
- **Library-crate transitive spirv-std dependency** — pure kernel libs cfg-gate spirv-std
  imports; helpers taking device-only types can't be host-callable in that pattern.
  (CLAUDE.md gotcha.)

---

## Recent landings (last ~6, prune as new items land)

| Commit | What |
|---|---|
| `02777ad` `b94809c` | `Pipe` is not `Clone` (a handle pipe moves into the one op that drains it; duplicating it is a compile error); dedicated `Error::PipeDrained`. |
| `94e37b3` `fbe385d` | `graph!`: sequential composition as straight-line `let` bindings; llama2 attention/ffn and gray-scott meta-kernels rewritten with it. |
| `9eac837` | `examples/llama2`: a llama2.c model built at run time from a text spec, replayed once per token, greedy output byte-identical to `run.c`. |
| `a384727` `c62a138` | `rename` (per-use-site slot identity for reusable graphs); a pipe-fed copy operand goes home to its origin. |
| `43605dc` `fdff0f7` | Keyed CB cache re-keyed by slot cell contents; library-provided reusable subgraphs (curry, then compose) documented and tested. |
| `f74e66c` `a2afd59` | `and_then` erases the chain it extends; `DeviceDynOp` boxes `dyn DeviceOp` directly (no mirror trait). |
