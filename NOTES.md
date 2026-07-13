# claspr — rolling notes

Single rolling doc for **active work, deferred items, and unresolved concerns**.
Convention (see `CLAUDE.md` → "Inter-session notes"): append here; don't spawn new
planning docs; prune as items resolve. **Resolved history lives in git commit
messages, not here** — `git log` is canonical. For the stable mental model of how
the codebase is laid out, read `ARCHITECTURE.md` (the map for agents/newcomers).

---

## Active

### 🏗 IN PROGRESS 2026-07-12 — agent cost-of-entry refactor (branch `refactor-deviceop-decompose-20260712`)

Goal (Brice): a sub-agent burns ~200k tokens before it can make ANY change — the
architecture is too complex to hold in context. Reduce that. Rollback tag
`pre-refactor-agent-cost-20260712`. Landed so far, each behavior-preserving (tier2
green on pocl, gray-scott/cg bit-identical):

- **R3 — deleted the legacy `RecordContext`/`RecordExt` record→replay path** (−1454
  lines). CB-as-execution-mode made recording automatic; the explicit `record_graph()`
  path was a redundant second model. `record.rs` is now the CB toolkit only.
- **R2 — factored the duplicated CB reach/sync-point bookkeeping** into two helpers
  (`cb_leaf_build`, `cb_forward_reach`) used at all 11 CB sites. Single source of truth
  for the once-buggy cross-seam reach propagation.
- **R1 (trait split) REJECTED** — splitting `DeviceOp` into a `CommandBufferOp`
  supertrait would cascade `+ CommandBufferOp` bounds through ~7 CB helpers + every
  combinator calling `cb_*` on generic children (~53 impl splits, ~60 call sites), with
  ZERO object-safety benefit (no `dyn DeviceOp` anywhere). Bad ROI for shrinking a
  read-once trait. The "CB is a skippable layer" goal is met instead by the module split.
- **R5 — added `ARCHITECTURE.md`** (the layered "to change X, read Y" map) and **pruned
  `NOTES.md` 3119 → ~120 lines** (resolved history is in git). CLAUDE.md points at it.
- **Module split — extracted `eager/cb.rs` (650), `eager/leaves.rs` (3042),
  `eager/combinators.rs` (2836)**. `eager.rs` 11023 → 5173 lines. `pub use mod::*`
  keeps macro-referenced public surface at its `::claspr::eager::` path;
  `AndThen`/`OnDevice`/`AndThenHost*` fields + `run_eager_chain` are `pub(crate)` (their
  `DeviceOpExt` builders stay in eager.rs).
- **`eager/slots.rs` DEFERRED (entangled)** — `SlotBinder`'s internals are driven by
  `fold_bind`/`probe_bind`, which are `DeviceOpExt` TRAIT methods in eager.rs; a clean
  move needs those drivers relocated to slots.rs as free fns first (rewrite the trait
  methods as thin wrappers). Attempted + reverted (pub(crate)-everything works but leaks
  the binder abstraction + touches delicate bind logic). A real refactor, not a move.
- **R4 — collapsed the 18 hand-written `KernelImage*Arg` marker traits into a
  `kernel_image_arg_traits!` macro + table** (image.rs −110 lines). Names preserved
  (proc-macro builds them via `format_ident!`). Bundle2..16 + the 2..16 tuple families
  (`FromCheckout`/`CheckoutSplit`/`SeamScatter`/`CallArgs`/`KernelArgs`/`BindAll`) were
  ALREADY macros — no further collapse needed.

FACTORING ROADMAP (remaining opportunities found scanning the extracted modules — each
is a REAL parameterized refactor, not a mechanical collapse, so left for a focused pass):
- **Leaf memory-family unification (biggest):** the device/mapped/usm variants of
  Fill-uninit (`FillDeviceUninit`/`FillMappedUninit`/`FillUsmUninit`), write-host-data
  (`WriteDevice{,Uninit}`/`WriteMapped{,Uninit}`/`WriteUsm{,Uninit}`), and alloc-uninit
  are ~100-line struct+impl blocks repeated ~3× across mem families. They differ by
  enqueue path + CB-recordability (device fill = weight-1 CB command; USM fill = host op
  weight-0), so a `leaf_fill!`/`leaf_write!` macro needs per-family hooks. ~15 leaves,
  could shed ~800 lines of leaves.rs.
- **Concrete-head terminal boilerplate:** `wait(self)`/`submit(self)` inherent impls
  repeated on ~12 leaves (`let ctx = concrete_buf_ctx(&self.FIELD)?; self.sync(&ctx)…`),
  varying only by field + return type — a `concrete_head_terminals!(field, Out)` macro.
- **Slots-driver relocation** (enables `eager/slots.rs`, see above).
- **`OutputShape`** (user-ergonomics track #238-240): derive `Handle`/`Checkouts` from
  `Output`, shrinking generic-subgraph where-clauses — cg `solve_with` exhibit.

VERIFIED (whole branch, tip `85af3b6`): all 9 examples build; claspr + claspr-test-image-kernels
build clean; tier2 309/0 on pocl (3-ICD sign-off pending). fmt/clippy(-workspace)/doc clean.
Rollback tag `pre-refactor-agent-cost-20260712`. First 5 commits already FF'd to main
(`17f575e`); commits since (leaves/combinators/R4) awaiting the 3-ICD run.

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

- **Image format dispatch in the proc-macro** — the macro emits `&Image2DRgba8` for every
  `&Image!(...)` param; the runtime (`Image2D<A, F>`) is generic over format. Gap is in
  the macro's dispatch only. (README Limitations.)
- **Cross-device + Arc-split test coverage is thin** — `cross_device` ~2 tests,
  `arc_split` sparse. Works as documented; harden when cross-device gets more usage.
- **Library-crate transitive spirv-std dependency** — pure kernel libs cfg-gate spirv-std
  imports; helpers taking device-only types can't be host-callable in that pattern.
  (CLAUDE.md gotcha.)

---

## Recent landings (last ~5, prune as new items land)

| Commit | What |
|---|---|
| `c9da355` | R2: factor CB reach/sync-point bookkeeping into `cb_leaf_build`/`cb_forward_reach`. |
| `f8c50b7` | R3: delete the legacy `RecordContext`/`RecordExt` record→replay path (−1454 lines). |
| `bcaf04e` | Precise per-slot command-buffer invalidation on a pipe-reachability substrate. |
| `29c783d` | A root bundle records ONE command buffer, not one per branch. |
| `0d776e1` | `Arced`/`ArcSplit`/`FanOut` are CB-able (delegate to source/branches). |
