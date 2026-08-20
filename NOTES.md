# claspr — rolling notes

Single rolling doc for **active work, deferred items, and unresolved concerns**.
Convention (see `CLAUDE.md` → "Inter-session notes"): append here; don't spawn new
planning docs; prune as items resolve. **Resolved history lives in git commit
messages, not here** — `git log` is canonical. For the stable mental model of how
the codebase is laid out, read `ARCHITECTURE.md` (the map for agents/newcomers).

---

## Active

Nothing in flight as of 2026-07-15 (working tree clean on `main`). The
2026-07-12→14 campaign — agent cost-of-entry refactor (eager.rs module split,
`ARCHITECTURE.md`, `OutputShape`), abstraction rounds, complexity-hotspot pass, and
the full quality-review fix set — has all landed. See git log; recent landings table
below.

### Decisions — assessed & DECLINED (kept so future sessions don't re-litigate)

- **`meta_kernel!` macro** (#241) — DECLINED. A repo-wide scan found `cg`'s
  `solve_with` is the ONLY generic-subgraph fn; every other reusable-graph pattern
  uses type-inferred CLOSURES with zero signature boilerplate. The macro would add a
  DSL + opaque expansion errors (a cost-of-entry COST) to save ~15 lines at exactly
  one site that `OutputShape` already made readable. **REVISIT TRIGGER: 3+ generic-fn
  subgraph authors accumulate.**
- **Leaf memory-family unification** (`leaf_fill!`/`leaf_write!`) — DECLINED. The
  fill/write families vary on ~5 axes (input/output type, in-place-`put_home` vs
  uninit-`put`, cl_mem vs SVM handle + enqueue fn, CB-recordable vs host); a macro
  would trade skimmable duplication for a fiddly muncher — a net loss for the
  cost-of-entry goal. The truly-identical alloc-uninit family WAS collapsed
  (`impl_alloc_uninit!`).
- **Concrete-head terminal boilerplate** (the ~6 `wait`/`submit` pairs) — DECLINED.
  They carry useful per-op doc comments (each op's Tier-1 spelling); a macro would
  drop those or need doc-passthrough (negating the win).
- **`DeviceOp` trait split** (`CommandBufferOp` supertrait) — DECLINED (R1). Would
  cascade `+ CommandBufferOp` bounds through ~7 CB helpers + every combinator (~53
  impl splits, ~60 call sites) with ZERO object-safety benefit (no `dyn DeviceOp`
  anywhere). The "CB is a skippable layer" goal is met by the module split instead.
- **Type-state slots for Tier 2** (graph type carries the unbound-tag set) —
  DECLINED (2026-07-15, not enough benefit for the churn). Design: `type Slots`
  (HList set) per op; `bind`/`call` type-transition `Graph<Remaining>` →
  `Graph<Remaining∖Tg>`; terminals gate on `Slots=HNil`; `Bound`(static) vs
  `Ready`(dynamic, stays runtime `check_ready`); `mutate_*` on `Graph<Bound>`.
  The prize: forgotten-bind / double-bind / typo'd-tag become COMPILE errors, so
  the whole deferred-error apparatus (`DeferredErrors` sink + `peek_deferred`
  poison + rebuild-recovery, `eager/slots.rs`) is DELETED, not maintained.
  Size/`LengthMismatch` stays a runtime terminal `Result` (raised at execute,
  never in the sink); readiness (lent/severed/pipe) stays runtime.
  **Why the residual ergonomics don't pay off — the shared-slot wall:**
  auto-dedup of a tag reused at N sites needs type-level type-equality →
  associated-type specialization's `default type` won't normalize (min_spec
  rejects outright; full spec compiles but can't be named downstream) — HARD wall
  on the pinned nightly, empirically reconfirmed. Consequences:
    - `bind(Grid(arr))` filling ALL Grid sites at once needs `RemoveAll` = same
      wall; NOT buildable. A macro can't rescue it (the occurrence count lives in
      the resolved type, invisible to a syntactic macro).
    - `g.bind::<rank>(...)` (one specific occurrence via explicit index) IS
      buildable — escape hatch only.
    - Fill-all + clone-by-multiplicity are MUTUALLY EXCLUSIVE (fill-all needs the
      tag once in the type; multiplicity-clone needs it N times).
  Three wall-free shapes, each with a real cost, none compelling now:
    - **Multiset + clone/move** (no marker; multiplicity = "consumed N times",
      clone/move falls out of ownership for free) — but repeated-tag removal is
      order-sensitive (head-pinned; inferred-index is `E0283`-ambiguous on
      repeats) and no fill-all.
    - **Dedup-by-declaration** (graph names its distinct slot set in its type —
      reusable graphs already do via `impl Graph<Slots=Slots![..]>`) — gives
      fill-all + order-free + single-move (runtime `TypeId` map fans one bind to
      all sites, no cloning) — but weaker typo-catching and a dedup annotation is
      still needed at cross-graph `bundle!` joins.
    - **`Reuse` marker at reuse sites** — REJECTED by Brice: a per-site
      annotation fights graph reusability.
  Scope if ever revived: ~36 `DeviceOp` impls gain `type Slots` (default `HNil`);
  ~9 composing combinators `Append` children's sets; close the latent
  `bind_slots` gaps (FanOut/Arced/ArcSplit/OnDevice/Profiled/DeviceDynOp — several
  don't walk slots at runtime today); `DeviceDynOp` erasure can't carry static
  `Slots` (stored graphs fall back to runtime). Verified feasible: distinct-tags
  type-state (18/18 old spike, re-greened today) and shared-via-explicit-Reuse
  (spiked green today, no unstable features). **REVISIT TRIGGER: if the
  deferred-error machinery grows further (more sink/poison cases), the trade tips
  — full type-state then deletes it wholesale; revisit dedup-by-declaration.**

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

### Inherit generated kernel deps from host workspace

`claspr-build`'s generated kernel `Cargo.toml` (`write_generated_cargo_toml`) hardcodes
`spirv-std`/`num-complex` to floating branch refs. Correct for consumers built inside the
claspr workspace (`seed_lockfile_from_host` copies the host lock), but a fresh external
build re-resolves against the floating ref. Fix: walk up from `OUT_DIR` to the host
`Cargo.lock`, extract the pinned revs, write them into the generated TOML (fallback to
today's refs). Glam is no longer in the generated TOML (spirv-std re-exports it).

### Tier 1 scoped launcher (`ctx.scope(|s| {...})`)

Sketched but resolved differently (ops carry ctx + no-arg `.wait()`/`.submit()`, commit
`2ba935a`). A real scope object only earns its keep bundled with profile-region
semantics / scope-wide event tracking / queue-model-at-boundary defaults. **Revisit
trigger:** a user wants one of those.

---

## Concerns

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
- **tier1 `image_dispatch::dim3_fill_pattern_r32_uint` (+ dim2_array) fail on pocl only** —
  NOT a pocl regression: rust-gpu emitted vec3 image coords for 3D/2D-array where the
  OpenCL SPIR-V env spec mandates vec4. FIXED in rust-gpu codegen 2026-07-12
  (writes now pass on all ICDs); re-verify once claspr picks up the rust-gpu bump.

---

## Recent landings (last ~6, prune as new items land)

| Commit | What |
|---|---|
| `6a33585` | Quality-review fix set: lent-buffer stranding guard, CopyTo2 CB-replay external deps, image length/dim → Result, CbBuilder pointer-arith `pub(crate)`, config-surface dedup, kernel-sig rejection. |
| `a50af0f` | Abstraction round 2 (−743 lines): `impl_image_verbs!`, `capabilities!` table, `impl_{acquire,release}_view!`, `impl_kernel_image_arg_matrix!`. |
| `994f6b4` | Complexity hotspot pass: `expand_kernel` (cyc 66→23), `try_bind_slot` (cog 39→7), `wait_on` decomposed. |
| `2fb5aab` | Delete `spikes/` — design prototypes fully absorbed into the shipped suite. |
| `f8c50b7` | R3: delete the legacy `RecordContext`/`RecordExt` record→replay path (−1454 lines); `record.rs` is now the CB toolkit only. |
| eager split | `eager.rs` 11,023 → 3,674 lines into `eager/{cb,leaves,combinators,slots}.rs`. |
