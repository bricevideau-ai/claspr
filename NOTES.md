# claspr — rolling notes

Single rolling doc for active work, deferred items, and unresolved
concerns. Convention is documented in `CLAUDE.md` → "Inter-session
notes". **Append here; don't spawn new planning docs.** Prune as
items resolve.

---

## Active

### ✅ Feed-a-`Checkout`-forward now LENDS (not severs) — 2026-06-30 (branch `typed-slots`, UNCOMMITTED, staged)

Feeding a `Checkout` from graph A as a plain INPUT to a second graph B used to
SEVER A's home (`Input::from(self.into_inner())` → slot `Lent→Severed` / concrete
cell emptied) — A was permanently broken. FIX (`eager.rs`): the implicit
feed-as-input path (`ToInput`/`From<Checkout>`/`Arc` variant, ~1860–1940) now
LENDS via a new `Checkout::into_value_and_home()` (moves `(value, home)` out
WITHOUT firing sever/rehome — drains both `Option`s so the Checkout's own Drop
short-circuits) + new `Input::lent(value, home)` (pre-loads a `Pipe` with
`put_home`, wraps `Input::Pipe`). The home rides B exactly like any internal edge
(`resolve_home`'s Pipe arm threads it; B's terminal Checkout / undelivered drop
rehomes it to A), so A stays `Lent`/busy while B holds the buffer, then returns
on B's drop → A re-runs by plain `sync()` (no `mutate_bind`). Composes
transitively (A→B→C→…, home threads pipe→pipe, returns at the FINAL drop).
`into_inner()` UNCHANGED = explicit take-it-out (still severs). NOT touched:
bind-Checkout-into-slot (`IntoBound`) = sever+adopt (role change), and
`CopyOperand for Checkout` still severs (out of scope; flag for follow-up if
copy-src/dst should also lend). Tests: flipped `home_invariant.rs` 7/8 to
`cross_graph_handoff_lends_and_returns` / `cross_graph_as_kernel_arg_lends_and_returns`
(LEND semantics: busy-while-held, return-on-drop, plain re-`sync`) + new
`into_inner_still_severs` + `checkout_lend_transitive` (A→B→C→download). Full
tier2 0 failures, tier1 green except the 2 baseline image_dispatch pocl fails,
gray-scott builds+runs+smoke.

### ✅ Two slot-binding bugs fixed — 2026-06-30 (branch `typed-slots`, UNCOMMITTED, staged)

- **Bug 1 (bundle branches dropped slot binds):** `Bundle*` inherited the no-op
  default `bind_slots`, so a `slot!` inside a bundle branch was never reached →
  `SlotUnbound` at sync. FIX: `impl_eager_bundle!` now overrides `bind_slots` to
  recurse into EVERY branch (mirrors `AndThen::bind_slots` fan-out discipline:
  move-only stops on consume, fan-out fills all).
- **Bug 2 (zero-match bind silently succeeded):** a `bind`/`call` of a tag that
  matches NO cell now hard-errors `Error::SlotNoSuchTag` (new variant). Rule is
  AT-LEAST-ONE. `SlotBinder.matched` counts every cell whose `id` matches (in
  both `try_bind_slot` impls, before the consumed-guard), incl. idempotent-no-op /
  conflict / sever (the tag IS present); `fold_bind` raises NoSuchTag iff outcome
  Ok && matched==0. `call`/`mutate_call` get it free (per-element `fold_bind`).
- Simplified tests 7+8 (`shared_launch_slot_fans_out`, `shared_arc_buffer_fans_out`)
  from the nested-and_then + `bundle2(forward,forward)` workaround to natural
  `bundle2(siteA, siteB)` (test 8 branches `.and_then(|(_,_,out)| forward(out))`
  to stay single-output). +3 regressions: `slot_in_bundle_branch_is_bound`,
  `bind_absent_tag_errors`, `fan_out_across_bundle_branches`. slot_generalization
  12/12; full tier2 + tier1 green except the 2 baseline image_dispatch pocl fails.

### ✅ Slot generalization: scalar + launch + shared slots — 2026-06-30 (branch `typed-slots`, UNCOMMITTED)

`slot!(Tag)` now fills THREE new positions beyond buffer/image kernel args:

- **(A) scalar args** — `slot!(Factor)` in a `factor: u32` position. NON-resource:
  rides a new TWO-state cell `ScalarSlotState{Unbound,Bound}` (`eager.rs`), value
  read (cloned) at execute (never lent/severed → no `Checkout`, no 4-state machine),
  idempotency by VALUE equality (`SlotEq` for scalars = `==` / float `to_bits`).
- **(B) launch args** — `slot!(Grid)` in the grid position, `Tag::Value =
  LaunchSpec`. Same 2-state path; re-dispatch the same graph at a different extent.
- **(C) shared slots** — one tag, many sites, ONE bind fills ALL. `SlotBinder` now
  carries a `SlotValue` clone hook: clone-able values (scalar / `LaunchSpec` /
  `Arc<DeviceSlice>`) FAN OUT (clone into every matching cell, binder never
  consumed); move-only buffers stay TAKE-ONCE (first cell moves, binder consumed) —
  so move-only single-site buffer slots are unchanged (no `Clone` forced). `SlotValue`
  is an explicit per-type surface (NOT a `Clone` blanket — would coherence-clash
  with move-only buffer impls). Walk short-circuits gated on `!is_fanout() &&
  is_consumed()`.

New types (`eager.rs`): `ScalarSlotState`/`ScalarSlotCell`, `ScalarInput<V>`
(`Concrete`/`Slot`, `read()`+`try_bind_slot`), `SlotValue`. Macro (`claspr-macros`):
scalar args → `impl Into<ScalarInput<#ty>>` (the `Into` bound preserves bare-literal
inference, e.g. `fill_u32([N], buf, 5)` infers `5: u32`); grid → `impl
Into<ScalarInput<LaunchSpec>>` stored as `ScalarInput<LaunchSpec>` on `Op.spec`.
Per-type `From` impls for scalar values + grid literals + `SlotHandle<Tg>`.
Non-resource (scalar/grid) slots are READ BEFORE buffers lend in execute, so a failed
completeness check can't strand a buffer slot in `Lent`. `launch.rs`: `From<[usize;N]>
for LaunchSpec`.

Tests: new `tests/tier2/tests/slot_generalization.rs` (9 tests, all green). Full
tier2 sweep green; `safety_compile_fail` 9/9; `image_compile_fail` 11/11. (The two
`image_dispatch` failures on this box are the PRE-EXISTING PoCL 6.0 `write_imagei`
linker bug — verified identical on baseline, unrelated.)

DEFERRED: user `#[repr(C)] Copy` scalar types need manual `SlotValue`+`SlotEq`+`From<T>
for ScalarInput<T>` impls (a `scalar_slot_arg!` sugar macro is the obvious follow-up,
mirroring `scalar_arg!`). Whole-`LaunchSpec`-as-one-tag only (no per-dimension slot).

### ✅ Double-buffering (ping-pong) integration test — 2026-06-30 (branch `typed-slots`, UNCOMMITTED, staged)

New `tests/tier2/tests/double_buffering.rs` — the canonical `mutate_bind` test.
K=4 iterations of `out = in + 1` via `add_u32(slot!(In), slot!(Ones), slot!(Out))`
with a persistent all-`1`s `ones` operand (bound once; its Checkout just `drop`ped
each step → slot re-arms `Lent→Bound`, reused). The In/Out swap is the load-bearing
part: `into_inner()` both (keep buffers, slots → `Severed`), then crossed
`mutate_bind` (a plain `bind` here is `SlotSevered` — the swap is ONLY expressible
via mutate). Tests: `double_buffer_ping_pong_computes_and_handles_stable` (asserts
final = INITIAL+K = 14, AND in/out handles ∈ {hA,hB} & distinct each step — exactly
two cl_mem recycled, no per-step alloc); `double_buffer_plain_bind_after_sever_rejected`
(plain `bind` on both severed In/Out → `Error::SlotSevered`). Runs on the existing
`sync()` reuse path; no command-buffer backend. DoD: build/clippy(`-D warnings`)/fmt
clean; 2/2 green; graph_slots 11/11, home_invariant 11/11, graph_reuse 7/7 regression.

### ✅ 4th slot state `Severed` — `bind` after `into_inner` rejected, `mutate_bind` re-arms — 2026-06-30 (branch `typed-slots`, UNCOMMITTED, staged)

`SlotState` is now 4-state: `{ Unbound (virgin), Bound(T), Lent, Severed }`. Fixes
the bug where `into_inner` did `Lent → Unbound`, letting a set-once `bind` of a
DIFFERENT buffer silently re-fill a slot whose value the user deliberately took.
- `into_inner`/`SlotHome::sever`: `Lent → Severed` (was `→ Unbound`).
- `try_bind_slot` Severed arm: `bind` (Set) → `Error::SlotSevered`; `mutate_bind`
  (Mutate) → fill → `Bound`. Virgin `Unbound` unchanged (both fill).
- `lend_slot`: Severed lends as `Error::SlotUnbound` (nothing to lend, like virgin).
- `with_concrete`: Severed reads `None` (no value to inspect).
- New `Error::SlotSevered(&'static str)` in error.rs + Display. ALSO added to
  `SlotBinder::outcome()` (an `Error`-matching site — its `unreachable!()` panicked
  until the arm was added; not a `SlotState` match, so easy to miss).
- Tests: graph_slots `into_inner_severs_slot_to_unbound` → renamed/flipped to
  `into_inner_severs_slot_then_bind_rejected_mutate_rearms` (bind→SlotSevered,
  mutate_bind re-arms); new `virgin_bind_ok_and_severed_resync_without_rebind_errors`
  (virgin-bind regression + severed-no-rebind sync → SlotUnbound).
- home_invariant: scenarios 7/8/9b flipped `bind`-after-sever → `mutate_bind`
  (they relied on the OLD wrong behavior). Scenario 11
  (`multi_output_copy_independence`) UN-IGNORED: rewritten to route copy dst
  through `slot!(Dst)` (concrete src + slot dst via the landed
  `eager_copy_to(src, slot!(Dst))` path), drop src (re-arm), into_inner dst
  (→ Severed), assert dst bind→SlotSevered + mutate_bind re-arms + per-side
  independence (src same handle). **ZERO ignored in home_invariant now.**
- DoD: build/clippy(`-D warnings`)/doc(`-D warnings`)/fmt all clean; full tier2
  serial green (graph_slots 11/11, home_invariant 11/11, graph_reuse 7/7,
  copy_reuse_flaw 3/3, record_replay 9/9); safety_compile_fail 9/9 +
  image_compile_fail 11/11 (clean rlibs first). Arc/Weak same-buffer-recovery
  relaxation intentionally NOT added (Severed unconditionally rejects bind).

### ✅ Image kernels are reusable `DeviceOp`s (un-forked) — 2026-06-29 (branch `typed-slots`, UNCOMMITTED, staged)

Un-forked the image one-shot/consuming path: image kernel args now ride the SAME
`Input`/cell/`Checkout` lend-and-return machinery as slice args. No image
special-casing left in the kernel macro.
- `KernelImage*Arg` supertraits gained `+ 'static` (image.rs) → owned images
  (`Image1D/2D/3D/1DArray/2DArray/1DBuffer`, each owns its `cl_mem`) qualify;
  borrowed `Image1DBufferView<'_,…>` does NOT (it's `'a`) and lost its
  `KernelImageBuffer*Arg` impls — the view was the SOLE reason for the fork.
- New `ToInputImage<SF>` trait (image.rs, exported) = image twin of `ToInput<E>`:
  impls for owned families + `Pipe` + `Checkout` + `SlotHandle`. `RecordableBuffer`
  now impl'd for owned images (stable `cl_mem` handle accessor, for the
  home-invariant assertion + future record).
- Macro (`claspr-macros/src/lib.rs`): image arm == slice arm — `__claspr_D{n}`
  concrete (flows to Output), `__claspr_S{n}: ToInputImage<family, Buf=__D>`,
  `resolve_home`, home threaded to output pipe, `try_bind_slot` per image arg.
  Shared `arg_gen_idx` counter (slices+images). `has_image_param` now gates ONLY
  the `RecordableOp` impl (images not recordable yet). Removed the consuming
  terminal (`__claspr_run_image`), `input_resolve_consuming`,
  `op_*_consuming`. **`Input::resolve_on` removed** (its only caller was the
  image consuming terminal — now dead).
- Call-site update: `dim_buffer_view_of_slice` → `dim_buffer_owned_read_to_slice`
  (owned `Image1DBuffer`, the reusable equivalent; the view-as-kernel-arg only
  ever worked via the one-shot path). `view_access_mismatch` compile-fail fixture
  re-pointed at owned `Image1DBuffer<ReadOnly>` (same access-mismatch intent).
- `home_invariant.rs` scenario 4 (`upload_writeonly_kernel_download_x3_stable`)
  UN-IGNORED: WRITE-ONLY `Image2D<WriteOnly,R32Uint>` + `dim2_uint::fill_pattern`,
  alloc-once / never-seed / kernel-overwrites, asserts STABLE `cl_mem` + correct
  data ×3. Now 10/11 green (only scenario 11 ignored). tier2 dev-dep on
  `claspr-test-image-kernels` added.
- DoD: build/clippy/doc(`-D warnings`)/fmt clean; full tier1+tier2 green; image
  regressions identical to baseline (2 pocl `write_image` gaps remain). ui_test
  goldens re-blessed (image_compile_fail) — MUST run with a clean `target/deps`
  (stale duplicate `claspr` rlibs → spurious "multiple versions" failures; clean
  with one rlib all pass). CB-cacheability of images: still future (out of scope).

### ✅ "Homeless is never legitimate" home invariant — 2026-06-29 (branch `typed-slots`, UNCOMMITTED, staged)

Every lent buffer (user-alloc AND upload-minted) carries a home; the graph never
releases a homed buffer — it REHOMES it. `tests/tier2/tests/home_invariant.rs`
1,2,3,5,6 green (+7,8,9,10); 4 (WriteOnly image, un-forked 2026-06-29) now green
too; only 11 stays ignored.
- `PipePayload{value: Option<T>, home}` + **`Drop`**: an undelivered payload
  (value+home present) rehomes on drop. `take_home` drains both in place (no
  destructure-by-move); home moved out = disarmed (single owner, `BoxedHome:
  !Clone`).
- `Download::execute` → `resolve_home` + `rehome_consumed(buf, home)` (buffer back
  to cell, Vec out homeless). `concrete_consumed_by_download` test flipped to the
  new rehome behaviour.
- `Upload`: alloc-ONCE into a persistent `Cell` + `seeded` flag; replay re-lends
  the SAME `cl_mem`; `UploadReseed::RESEED_ON_REPLAY` (writable=reseed,
  RO/Frozen=seed-once). Lent+seeded ⇒ busy.
- `reclaim_undelivered` (DeviceOp method; AndThen recurses, kernel-macro + CopyTo2
  drain element pipes) called post-gather in `wait_on` — returns multi-output
  intermediates an `and_then` discarded BEFORE next run's upstream re-lend.
- Slot-as-copy-operand: `CopyOperand` trait (per-family + `SlotHandle`) →
  `eager_copy_to(slot!(A), slot!(B))` type-checks; `CopyHome::copy_slot_home`
  wires the formerly-dead `slot_home` through `Input::copy_input_home`; CopyTo2
  gained `bind_slots`. `SlotHome` lost its `Drop`/`fired` fallback (general
  payload-drop rehomes; consumed slot now `Lent→Bound`, not `Lent→Unbound`).
- DEFERRED (scenario 11): re-sync after `into_inner`-severing a CONCRETE copy dst
  needs re-ALLOC of that side, which needs sever-vs-busy disambiguation (tri-state)
  on a user copy cell — out of scope. Per-side independent homes themselves work.

### ✅ STEP (a) follow-up 2026-06-26 — copy-in-reused-graph re-arm + Init→Uninit downgrade (`Rehome`)

Branch `replayable-graphs` @ `afdb1d4` (pushed). Closed a real step-(a) gap: a
concrete-buffer `eager_copy_to` in a reused graph did NOT re-arm its src/dst
cells (CopyTo2 deposited outputs with `put`, home=None) → second `sync` errored
"graph busy". Fix generalized the home channel from `Option<Cell<T>>` to a
type-erased **`Rehome<Out>` trait** (`BoxedHome<Out> = Box<dyn Rehome<Out>>`):
- `impl Rehome<T> for Cell<T>` = identity (every in-place/kernel path, unchanged).
- `DowngradeRehome<U, Init>{ cell, wrap: fn(Init)->U }` returns a copy's **Init**
  output into its weaker **Uninit** cell. KEY INSIGHT (Brice): Init is the
  STRONGER capability (read+write); forgetting it to write-only Uninit is always
  SOUND — the copy already did the write that earns Uninit→Init, so handing the
  same buffer back for the next run to overwrite is correct.
- All THREE Uninit families downgrade-rehome (none left `home=None`):
  DeviceSliceUninit/MappedSliceUninit via trivial `from_init` private-field
  re-wrap; **USMSliceUninit** via `from_init` = address-preserving
  `Vec<T>→Vec<MaybeUninit<T>>` reinterpret (ManuallyDrop to skip USMSlice's
  wait-on-drop; the copy already completed). "Needs internal `unsafe`" ≠ unsound —
  it's the SAFE direction, strictly safer than `assume_init`'s existing reverse.
- CopyTo2 `execute` threads each input cell as a typed home via a `CopyHome<Out>`
  per-family helper (src = identity, dst = identity-or-downgrade).
Test `copy_reuse_flaw.rs` 3/3 (incl USM, runs on pocl fine-grain SVM); graph_reuse
7/7; copy/svm/usm regressions green. STILL `home=None` (correct): pipe-fed copy
inputs (producer re-mints each run). Process lesson reinforced: background agents
here are fire-and-forget (no SendMessage) — correct a stray agent by killing it,
fix small gaps post-hoc yourself, never spawn a helper against a running agent
(it contends the build lock).

### ✅ STEP (a) LANDED 2026-06-26 — reusable `g.sync()` (own-the-buffers; no slots yet)

Branch `replayable-graphs` @ `b35e390` (pushed, not merged to main). The op-tree
IS the reusable graph; the design below is REALITY now for step (a). What shipped:

- **`DeviceOp::execute(&self)`** — borrowing, non-consuming (was `self`). `Input`
  is a lend-and-return **cell** (`Cell<T> = Arc<Mutex<Option<T>>>`); `resolve`
  lends the buffer, the run's `Checkout` returns it on drop → `g` re-arms. The
  kernel proc-macro's `execute` was rewritten to borrow (`&self.kernel`, copy
  `LaunchSpec`, lend args).
- **`sync(&self)/wait_on(&self) -> Result<Checkout<…>>`**. `Checkout<O>`:
  Deref/DerefMut, `into_inner` (sever), return-on-drop (re-arm), busy-error on a
  second `sync` while a buffer is still checked out, + `Debug`/`PartialEq`/
  `PartialEq<O>` passthrough, + transparent as a kernel arg/`ToInput` and
  `.read()/.copy_to()` directly. So one-shot call sites mostly DON'T need
  `into_inner` — borrow via Deref; `into_inner` only to take an owned value by
  move (`Arc::new`, store, return).
- **Per-output checkouts**: multi-output terminals return a TUPLE
  `(Checkout<A>, Checkout<B>, …)`, each with its OWN home cell, via
  **home-carried-in-pipe** provenance (the cell travels with the value through
  the pipe — typed, no `Any`, no cl_mem heuristic). Same-typed multi-buffer
  kernels (`add(a,b,out)`) re-arm every cell correctly. `AndThen` delegates
  `gather_checkouts` to its tail (mirrors `collect`).
- **Alloc rule**: read-only buffers persist; mutable re-seed each run (the
  `upload` op re-mints), so `upload→kernels→download` is idempotent across
  repeated `sync` (verified ×3 no-compounding).
- **Images**: kept on the one-shot CONSUMING path (borrowed image views aren't
  `'static`, can't be cells). NOT a recordability limit — image reuse is a
  slots-era feature; the CB image commands exist (`clCommandFillImageKHR`,
  `clCommandCopy{Image,BufferToImage,ImageToBuffer}KHR`, all in opencl-sys 0.6.1)
  to record an OWNED image later. `Input::resolve_on(&self, launcher)` added for
  the image terminal (builds a transient EC from a bare Launcher).

Tests: graph_reuse 7/7 (idempotent reseed, multi-output, into_inner, busy-guard,
add-3-buffers re-arm). Full tier1+tier2 migrated to Checkout (47 files) +
examples; review agent confirmed ZERO assertion/semantic regressions. Green on
pocl; fmt/clippy/doc `-D warnings` clean. Known non-issues: 2 pocl 3D/array
`write_image` gaps (pre-existing, fail identically at HEAD); ui_test compile-fail
must run via `cargo test --test <name>` (isolated) not batched `-p` (multiple
`claspr` rlibs confuse its rlib discovery — pre-existing).

DEFERRED to later steps (design below): **(b) slots** (`slot!(Tag)`/`Tag(value)`/
`bind` runtime bind-table + typed tags → rebind different buffers per run);
**(c) convex-segment replay** (software + cached `cl_khr_command_buffer`); **(d)
mutable-dispatch** + image reuse. NOTE: `record.rs` (commits fd68c0c…2bd92a5)
is NOT dead salvage — it's a LIVE, TESTED public surface: `g.record()?` →
`RecordedGraph` → `.replay()` is the explicit CB-backed record path (real
`cl_khr_command_buffer` layer-2 backend + software fallback), green via
`tests/tier2/tests/record_replay.rs` (9/9). It is SEPARATE from `g.sync()`
reuse: `sync()` does own-the-buffers re-walk of `execute(&self)` (no CB);
`record()/replay()` is the record-once-into-a-CB path. Step (c) is about wiring
the CB segments UNDER `sync()` so the primary surface gets CB acceleration too —
NOT about resurrecting record.rs (it already works). Do not demote/delete the
`record` public exports — they back a passing feature. Process lesson:
engine-touching agents must be
SEQUENCED, not fanned out — parallel runs against a moving base caused a
stale-base merge tangle + a shared-file collision this session.

### ⭐ REUSABLE GRAPH MODEL — agreed design (2026-06-26, brainstormed w/ Brice). THIS SUPERSEDES the record/replay framing below.

Branch `replayable-graphs` built `record()`/`replay()`/`RecordedGraph` as a
SEPARATE reusable object (commits fd68c0c…2bd92a5). **Wrong reusable object.**
Brice's model: **`g` (the op-tree) IS the reusable graph**; `sync()` is the
verb; the awkward record-borrow-drop-consume dance is the symptom of getting it
wrong. The layers-1/2 code (software IR + segment partition + CB FFI loader) is
SALVAGE — it moves UNDER `g.sync()` as an invisible cache. `RecordedGraph` /
`record()` / `replay()` as a public surface goes away.

**The model (op-tree-is-g, no wrapper):**
```
g = upload(vec).and_then(|b| ks.scal(b, 2.0));   // self-contained
let out = g.sync(&ctx)?;  /* use out */          // reusable: call sync again
let out = g.sync(&ctx)?;
```

1. **Cell = Pipe.** An `Input`/edge is an interior-mutable cell
   (`Arc<Mutex<Option<resource>>>` — literally what `Pipe` already is). One
   concept unifies four origins by initial state: **owned** buffer (full) /
   **`slot!(Tag)`** (empty) / **internal edge** (filled at run) / **output**
   (drained to checkout). "A slot IS the same pipe as an internal edge" — now
   literally true.

2. **Checkout (runtime).** `sync(&self) -> Checkout<Output>`: assert every input
   cell full (else runtime Err "Tag unbound"), replay, drain outputs into the
   Checkout; on Checkout **drop** the resources RETURN to `g`'s cells (re-arming
   it). While checked out, `g` can't run (empty cell → runtime block). This one
   mechanism gives BOTH "no parallel use of g" AND safe shared-subgraph
   composition. Must support **multiple outputs** (`Checkout<(A,B,..)>`,
   per-element return). Consuming terminals are Checkout-hosted: `co.read(&mut
   v)` does the device read AND returns the buffer to `g` (consume-by-value ==
   return-on-drop — the feature, not the blocker). `into_inner()` severs the
   return (keep the buffer; `g`'s cell stays empty / re-allocs next run).

3. **Slots: typed tuple-struct tags, runtime presence, order-free, curry.**
   `slots! { Buf: DeviceSlice<u32> }` → `pub struct Buf(pub DeviceSlice<u32>)` +
   `impl Tag`. Build a hole with `slot!(Buf)`. Bind with **`Buf(value)`** (plain
   tuple-struct construction — NO fn_traits, the thing that killed the old
   `B(&b)`; carries any type incl vectors). `g.bind(Buf(b)).bind(W(w))` folds a
   `TypeId→resource` table (order-free, curryable, partial OK); completeness
   checked at `sync` (runtime). Binding MOVES the value into the cell, recovered
   via checkout (share read-only via `Arc<DeviceSlice>`). Typed = per-tag value
   type checked at compile time; NEVER set-algebra (that HList/turbofish/dedup
   pain is why compile-time-set was abandoned). `bind` returns a composable
   `DeviceOp` node → a single-output `g.bind()` is usable as a kernel arg /
   chain node, so graphs COMPOSE:
   `g2 = ks.scal(slot!(X),3.0).and_then(|b| bundle2(b, g.bind())).and_then(|(a,b)| ks.add(a,b))`.

4. **Alloc rule (resolves compounding):** read-only (`Frozen`/`ReadOnly`)
   buffers alloc **once**; mutable buffers **re-seed each run** via a software
   reset segment (the user's `upload` op IS the reset — host-writes land in a
   software segment naturally). So `upload(vec)→kernels→download` is **idempotent
   by construction**; the marker decides. Reuse-reset (NOT realloc) keeps cl_mems
   stable → cached CB stays valid across runs.

5. **Replay = convex-segment plan (CB is the cherry).** Don't gate the IR on
   today's CB features. Partition the command list into **convex** segments
   (a resource can't leave and re-enter a CB — guaranteed FREE by eager
   move-semantics: linear single-owner dataflow, no back-edges). CB-able ops
   (ndrange/fill/copy/barrier) → cached `cl_khr_command_buffer` enqueues;
   everything else (host writes/resets, download, SVM fill, and_then_host,
   map/unmap) → software segment (enqueue, fresh events per run). Events bridge
   segments; sync-points stay internal to each CB. Mutable-dispatch
   (`clUpdateMutableCommandsKHR`, kernel args only) is a LATER segment kind for
   rebinding-different-buffers — not needed for own-the-buffers reuse.

**Validity guarantees (the deliverable):** types / markers / single-writer /
acyclic = **compile-time** (survive untouched — why op-tree-is-g matters, no
`dyn` boxing). concurrent-use / checkout / slot-completeness = **runtime**.
Per-tag value type = compile-time; tag *presence* = runtime.

**Build order:** (a) cell-ify `Input` + `sync`→`Checkout` (own-the-buffers reuse,
multi-output, no slots) → (b) `slot!`/`Tag(v)`/`bind` table + completeness →
(c) segment plan (software + immutable CB, salvage layers 1/2 under sync) →
(d) mutable-dispatch segment for different-buffer rebind. Open soft spot: exact
shape of what a slotted graph hands back post-sync (bound buffers via checkout).

#### ✅ STEP (a) LANDED 2026-06-26 (branch `replayable-graphs`, 4 commits, NOT pushed)
`DeviceOp::{execute,collect,into_output}` → **`&self`** (the op-tree IS reusable).
`Input::Concrete(Cell<T>)` (= `Arc<Mutex<Option<T>>>`); `resolve(&self)` **lends**
the buffer + records the cell in a per-run ledger on `ExecutionContext`.
`sync(&self)/wait_on(&self)` → **`Checkout<Output>`**: `Deref`/`DerefMut` to read;
on **drop** returns the output to the lending cell **iff exactly one lent cell
matches `Output`'s type** (unambiguous in-place single-buffer case) — re-arming `g`.
`into_inner()` severs the return. **Busy** = a lent cell found empty on a 2nd
`resolve` → runtime Err. **Reseed** = entry leaves keep their source & re-emit:
`upload` reads `src` by ref (re-creates buffer each run → `upload→…→download`
idempotent), `value` holds `T` by-value (clone per run), `Write*`/`ImageUpload`
read by ref (no keep-alive — `&self` outlives the whole `sync`). One-shot leaves
(`lift`/`usm_slice`/host seams/`profiled`) use `Mutex<Option<_>>` with a clear
re-run Err. **Kernel macro** `execute` rewritten to borrow `&self.kernel` /
copy `self.spec` (Copy) / deref scalars / lend slice+image args (images now route
through `Input<I>` too); `profile_cb` is `Mutex<Option<…>>` taken once.
PROOF: `tests/tier2/tests/graph_reuse.rs` 6/6 on pocl (idempotent ×3 / multi-output
/ into_inner / busy+re-arm / download-consume boundary). Lib + collatz green;
fmt+clippy+doc clean on claspr+macros. **Old tier1/tier2 suites left BROKEN
(expect `sync→Output`, now `Checkout`) — NOT migrated, by plan.**
Decisions taken (not in spec): re-arm only when a single lent cell matches
`Output` type (multi-input same-typed kernels like `add(a,b,out)` don't auto-re-arm
— safe: their cells stay empty → "busy"; read via `into_inner`); `submit_on`/
`submit_value_on`/`run` left `self`-consuming (don't return `Checkout`); host-seam/
profiled/lift/usm reuse deferred (one-shot, out of step-(a) scope).
NEXT: migrate the ~175 old tests to `Checkout` (sweep `.sync()?`→`.sync()?` + read
through deref / `into_inner`); then step (b) slots.

#### ✅ STEP (a) REFINED 2026-06-26 (home-in-pipe; replaces the heuristic + per-output Checkouts)
Both step-(a) flaws fixed. (1) **Per-output Checkouts.** `DeviceOp::Checkouts`
(assoc-type-default `Checkout<Output>`); multi-output ops OVERRIDE to the tuple
`(Checkout<A>, Checkout<B>, …)` (kernel macro, `CopyTo2`, `ImageCopy`, `bundleN`,
`arc_split → [Checkout;N]`). `sync/wait_on → Self::Checkouts` via new
`gather_checkouts(&self,…) -> (Self::Checkouts, Deps)` (default drains the output
pipe via `take_home` → one `Checkout`; multi-output overrides drain each element
pipe). `FromCheckout<O>` bridges the default; tuple/array impls are `unreachable!`
(never hit — those ops override). (2) **Home-in-pipe (no heuristic, no `Any`).**
`Pipe<T>` payload is now `PipePayload{value, deps, home: Option<Cell<T>>}`; `put`
keeps its sig (home `None`, ~52 callers untouched), `put_home`/`take_home` carry
it; `take` drops it. `Input::resolve_home(&self) -> (T, Deps, Option<Cell<T>>)`:
a `Concrete` cell IS the home; a `Pipe` propagates whatever flowed in. **In-place**
ops (Fill, WriteDevice, ReadInto, SVMfill/write, transfer, Forward, all image
write/read/fill/copy, every kernel buffer/image arg) thread the input home →
output pipe; **mint/transform/consume** ops (upload/value/alloc, download's Vec,
uninit→init, host-view, copy's retyped outputs, OnDevice routing boundary, bundle
branches collapsed via `collect`) put `None`. `Checkout<O>{value, home:
Option<Cell<O>>}`: drop re-arms via the typed home, `into_inner` severs.
**DELETED**: the type-match heuristic, `Checkout.lent_cells: Vec<Box<dyn Any>>`,
and `ExecutionContext`'s `lent_cells`/`record_lent_cell`/`take_lent_cells`/
`lent_cells_handle` ledger (home-in-pipe fully covers step (a)). **Transparency**:
`ToInput`+`From<Checkout<buf>>` for Device/Mapped/USM/Arc slices (consume = sever),
inherent `read`/`copy_to` forwarding on `Checkout<DeviceSlice>` → `checkout.read(&mut v)`
+ feeding a Checkout as a kernel arg both work with NO `into_inner`. PROOF:
`graph_reuse.rs` 7/7 on pocl incl `add(a,b,out)` re-arming ALL THREE same-typed
cells across runs (the heuristic case). Lib+collatz green; fmt/clippy(-D)/doc clean.
Deferred (consistent boundary, read via `into_inner`): home re-arm through copy
(retyped outputs), OnDevice routing, and bundle branches.

#### ✅ STEP (b) LANDED 2026-06-26 (branch `typed-slots`, NOT committed — staged for review)
`slots! { Buf: DeviceSlice<u32>, … }` → `pub struct Buf(pub DeviceSlice<u32>)` +
`impl Tag for Buf { type Value = …; fn into_value(self){ self.0 } }`. `Tag`
(`Sized+'static`, `Value: Send+'static`) is the identity key via `TypeId`. NO
set-algebra: per-tag value type = compile-time, presence = runtime (checked at
sync). `slot!(Buf)` = `SlotHandle::<Buf>::new()` (fresh empty `Cell`); plugs into
KERNEL-ARG positions via a `ToInput<E, Buf=Tag::Value>` impl (the primary site,
tested). **`Input<T>` gained a 3rd arm** `Slot{id:TypeId, name:&'static str,
cell:Cell<T>}`; once bound it lends/re-arms EXACTLY like `Concrete` (shared
`lend_from_cell` helper) — a bound graph re-runs (Checkout returns to the slot
cell on drop). Empty cell at resolve → new `Error::SlotUnbound(&'static str)`
(carries `type_name::<Tag>()`). `g.bind(Tag(v)) -> &Self`: builds a `SlotBinder`
{id, boxed value} and folds it via new `DeviceOp::bind_slots(&self, &mut
SlotBinder)` (default no-op) — overridden on `AndThen` (recurse, short-circuit on
`is_consumed`) + the kernel macro op (BOTH single- AND multi-output impls — easy
to miss the 2nd!). Each `bind` carries one tag → order-free/curryable/partial
falls out; binding MOVES into the first matching cell; a 2nd `bind(Tag(other))`
rebinds. PROOF: `tests/tier2/tests/graph_slots.rs` 5/5 on pocl (bind+data /
order-free / re-run / unbound-Err / rebind). graph_reuse 7/7 + eager_chain/
buffer_ops regress green; full tier2 builds; claspr build+clippy(-D)+doc(-D)+fmt
clean. **DEFERRED w/ TODO**: (1) `bind` returns `&self` (serves `.bind().sync()`
+ chained binds); the composable single-output `g.bind()` as a kernel arg /
`bundle2(b, g.bind())` nesting (NOTES §3) waits on step (c). (2) `slot!` in
`Into<Input<_>>` positions (download/fill/write/copy) needs explicit
`SlotHandle::into_slot_input()` — a direct `From<SlotHandle> for Input<Value>`
collides with the blanket `From<T> for Input<T>` (coherence: `Value` could ==
`SlotHandle`). (3) `bind_slots` overridden only on AndThen + kernels; other
`Input`-leaves (download/fill/copy/bundle) keep the no-op default → a slot there
stays unbound (caught loudly at sync). ALSO FIXED HERE: the `safety_compile_fail`
fixture `fill_on_frozen.stderr` golden drifted (`buffer.rs:485`→`501`) — the
replayable-graphs merge grew `buffer.rs` (byte_len/RecordableBuffer/from_init) but
that one golden wasn't re-blessed at promotion, so it fails on `main`/`origin/main`
too (CI would catch it). Re-blessed on this branch; **main needs the same one-line
bless** (track separately).

#### step (b) follow-up — slot-binding verb 2×2 (branch `typed-slots`, UNCOMMITTED)
Replaced the slot's `Cell<T>=Option` with **tri-state** `SlotState<T> {Unbound,
Bound(T), Lent}` (`Input::Slot.cell: SlotCell<T>=Arc<Mutex<SlotState>>`; Concrete
arm unchanged). Distinguishes never-bound from checked-out → enables the matrix.
Transitions: lend `Bound→Lent` (`lend_slot`), Checkout-drop `Lent→Bound`
(`SlotHome::rehome`), `into_inner` `Lent→Unbound` (`Rehome::sever`, new trait
method, no-op on `Cell`), **dropped-unfired `Lent→Unbound`** (`SlotHome: Drop` w/
`fired` flag — covers download-CONSUMED slots, else stuck Lent forever; this was
the subtle bug). VERBS now return `Result<&Self>`: `bind` set-once (idempotent on
==, `SlotConflict` on ≠), `mutate_bind` set/change (fills unbound, no
SlotConflict), both `SlotCheckedOut` on Lent. `call((A(a),B(b),..))` /
`mutate_call` = multi-fill via `BindAll` tuple trait arity 1..=8 (folds each thru
the single-slot path, all-or-nothing in tuple order). Equality = **cl_mem/SVM
handle identity** via new `SlotEq` trait (buffer families + `Arc<DeviceSlice>`),
bound on `Tg::Value`; comparator captured into `SlotBinder` as type-erased
`SlotEqFn` (try_bind_slot is generic, no SlotEq bound). Errors threaded out of the
fold via `SlotBinder::outcome()` (binder gained `mode: BindMode`, `eq`, `outcome`).
New errors `SlotConflict`/`SlotCheckedOut(&'static str)`. graph_slots.rs rewritten
10/10 (matrix incl checked-out + sever); regress graph_reuse/copy_reuse_flaw/
eager_chain/eager_buffer_ops green; build+clippy(-D)+doc(-D)+fmt clean on pocl.
**NOT routed**: slot used DIRECTLY as copy src/dst (output type ≠ input type; needs
CopyHome-style bridge — `Input::slot_home` exists for it) → stays Lent after 1 run,
loud busy on re-sync. `mutate_*` is just cell-overwrite; clUpdateMutableCommandsKHR
in-place dispatch is step c/d (segment-plan). bind still returns `&Self` (composable
node deferred, same as step b).

### ✅ PROMOTED TO MAIN 2026-06-24: eager struct-graph cutover (72 commits)

`eager-cutover` fast-forwarded onto `main` at `6d76fe2` (linear history, no merge
commit) and pushed. This lands the whole reunification: `DeviceOp`/`DeviceOpExt`
struct-graph, unified blocking/async terminals, host seam (start-gate + worker-join
+ event-based chained-cancel), device-by-index routing, prelude, the
Context/default-queue Arc-cycle fix, and the marker-in-start-gate fix.

Promoted on a FULL green gate: lint/doc clean (fmt + release clippy `-D warnings`
+ doc `-D warnings` + compile_fail-fixture rustfmt); tier1+tier2 263 passed/0
failed and all 9 examples green on **all three ICDs** (pocl, Intel NEO legacy,
rusticl); async error path stressed 40-80×/ICD, 0 crashes.

**OPEN DEPENDENCY — 3 pocl PRs (not blocking the claspr merge, but required for
the green result on stock distro pocl):** #2214 failed-dependency, #2215
inline-retain, #2216 double-finish guard (all bricevideau-ai/pocl → pocl/pocl).
The green gate ran against a local pocl carrying all three. claspr's own
marker-in-start-gate fix (`6d76fe2`) makes the async terminal correct
independently, but the general multi-dependency-error case still needs the pocl
fixes. Details + root causes: see Concerns → "RESIDUAL pocl crashes" below. Once
the PRs land + a release ships, revisit `project_claspr_ci_deferred` (CI was
waiting on PoCL 7.2 anyway).

### ✅ LANDED 2026-06-24: removed `and_then_with_context`; device-by-index routing is now structural

The execute-time `and_then_with_context` combinator (built its downstream op at
EXECUTE, so the host-seam `contains_host_seam()` gate couldn't see through it —
the one documented gap in the start-gate fix) is GONE. Its sole real use across
~18 call sites was device selection (`ec.device_at(i)` fed into
`on_device`/`transfer_to_device`), so it was re-expressed structurally: a
`pub(crate) enum DeviceTarget { Concrete(Device), Index(usize) }` on `OnDevice`
+ `TransferToDevice`, resolved at the top of each `execute`. New builders
`DeviceOpExt::on_device_at(i)` + free `transfer_to_device_at(buf, i)` (latter
re-exported at crate root + prelude); concrete `on_device`/`transfer_to_device`
unchanged. The whole graph is now build-time inspectable, so the gap is CLOSED
(no un-gated host seam can hide inside an execute-time closure anymore). Migrated
call sites in eager_{on_device_suite,cross_device,transfer_to_device,alloc_ops,
cutover}.rs; the same-device read-after-write regression now rides the pipe-fed
`.and_then` event edge (stronger than the removed barrier hack) and still asserts
==15. `and_then_host_with_context` (a DIFFERENT, fully-gated construct) untouched.
Verified: build/clippy/doc green (default + async-events); all single- and
multi-device subset tests pass on pocl (2-device sub-device-partition context, so
`on_device_at`/`transfer_to_device_at` were genuinely exercised, not skipped).

### ✅ LANDED 2026-06-24: `and_then_host` error path — START-GATE + WORKER-JOIN + EVENT-BASED CHAINED-CANCEL

**Status: the NEO lost-wakeup deadlock is FIXED (was ~1-in-5 / ~100% under
cliloader). NEO + rusticl 100% clean; pocl has 0 hangs, with a residual
pocl-internal SIGSEGV (~15%) isolated to the error→downstream-device-op shape.**

Two-event seam (unchanged, from 2026-06-23): `fire` gates the unmaps (always
`CL_COMPLETE`, one clean unmap per buffer — the prior double-unmap that corrupted
NEO is gone); `proceed` gates downstream (`CL_COMPLETE` on success, negative on
error). The negative `proceed` was driver-unsafe ON ITS OWN (NEO lost-wakeup race
in a downstream blocking transfer's wait-commit window). The fix makes it safe by
ensuring the WHOLE graph is enqueued before ANY of it runs — validated in
`scratch/start_threaded.c` (NEO 40/40, 0 hung; the threaded-start variant, NOT a
queue barrier or marker). Five pieces, all gated on a transitive flag so
no-host-seam graphs are unchanged (zero cost):

1. **`DeviceOp::contains_host_seam() -> bool`** (eager.rs trait method, default
   `false`). `AndThenHost`/`AndThenHostWithContext` → `true`; combinators OR their
   owned children — `AndThen` (source||next, so a seam built in the closure IS
   seen), `Bundle2..16`, `FanOut`, `ArcSplit`, `Arced`, `OnDevice`, `Profiled`,
   `DeviceDynOp` (via a new `contains_host_seam_erased` on `ErasedDeviceOp`).
   `CopyTo2` + `AndThenWithContext` are NOT in the OR set: CopyTo2's inputs are
   pipes (edges; the producer op is owned elsewhere and ORs it), and
   AndThenWithContext builds its downstream at execute time (not structurally
   inspectable). A host seam inside an `and_then_with_context` closure is the one
   documented un-gated shape — use the structural `and_then_host` surface.
2. **Start gate, THREADED per entry-leaf** (NOT a barrier — a barrier would block
   concurrent graphs; the validated `start_threaded.c` merges `start` into each
   entry command's wait-list). `ExecutionContext` gained `start: Option<cl_event>`;
   `Input::resolve(self, ec)` — the ONE entry-gating edit — threads `start` into a
   `Concrete` input's deps (Concrete == chain head == entry leaf), `clRetainEvent`
   per dep. The `Pipe` arm is unchanged (transitively gated). All ~28 `.resolve()`
   call sites + the kernel macro updated to `.resolve(ec)`.
3. **Worker join** (independent correctness fix — worker was DETACHED). EC holds
   `workers: Arc<Mutex<Vec<JoinHandle>>>`; `run_host_seam` pushes its handle;
   terminals join AFTER the device wait (so no worker's late CL calls — signal
   events, drop retained queue — race the caller dropping the Context).
   `with_host_error_slot` propagates `start` + shares the SAME `workers` Arc so a
   routed `on_device` sub-chain's workers are joined too.
4. **Terminals restructured** (`wait_on` blocking + `run_eager_chain` async). When
   `contains_host_seam`: create `start`, `set_start`, `collect(Pipelined)` (NOT
   Blocking — a Blocking leaf would wait inline on a start-gated command and
   deadlock), complete `start` `CL_COMPLETE` (after the whole graph is enqueued),
   wait on the deps (NEVER `clFinish` — clFinish on a terminated command is the
   pocl hang), then `join_workers`. Non-host-seam: unchanged Blocking fast path.
   The async future joins its workers when the marker resolves.
5. **Chained-cancel via the EVENT, not a host slot** (Brice 2026-06-24: the
   slot check raced the upstream worker's stash — confirmed 1/40 NEO failure with
   the slot version). `run_host_worker` now propagates cancellation through the
   cl_event dependency: an upstream seam's negative `proceed` is in this seam's
   `source_deps`, so after waiting we re-read each event's COMMITTED
   `command_execution_status` (new `event_is_cancelled` helper) — a negative
   value short-circuits (don't run the closure), driver-independently. Reading the
   committed status (not the `clWaitForEvents` return code) defeats the same NEO
   wakeup race on the host wait: 60/60 NEO with this version. Makes
   `and_then_host(Err).and_then_host(side_effect)` skip the second closure.

**VERIFY (this box, serial, every device run timeout-wrapped):**
- NEO eager_cutover (full, incl. the re-enabled error test) **40/40, 0 hangs**;
  isolated error test **40/40**; eager_error (chained-cancel counter==0) **40/40**;
  chained-cancel isolated **60/60**; non-host-seam eager_chain/bundle/buffer_ops
  clean (no regression).
- rusticl: all host-seam suites **100% clean** (eager_cutover/error/
  error_fidelity/with_context/host_view).
- pocl: host-seam-only suites **100% clean** (eager_error 30/30, fidelity 15/15);
  the error→`.and_then(download)` shape **34/40** (6 SIGSEGV/abort, **0 hangs**,
  passes when it completes). The SIGSEGV is the pocl-internal terminated-read
  cleanup bug — NOT a claspr correctness issue and NOT a hang.

TEST: `eager_and_then_host_error_propagates` (eager_cutover.rs) RE-ENABLED
(`#[ignore]` removed). Passes on NEO + rusticl (the gating platforms); may
SIGSEGV ~15% on pocl (documented in the test doc + here). The
`and_then_host_error_stops_chain_immediately` (eager_error.rs, `counter==0`) is
the chained-cancel correctness lock.

**RESIDUAL pocl crashes on the error path: ROOT-CAUSED + FIXED 2026-06-24 = THREE
distinct pocl bugs, plus one claspr-side ordering fix. All resolved; the error
path is now 0-crash on NEO + rusticl + pocl.** The original ~15% SIGSEGV turned
out to be three independent races (a single fix only halved it):

1. **Already-failed dependency runs anyway** — `pocl_create_event_sync` treated a
   dep that was ALREADY failed (status<0) at wiring time identically to a completed
   one (skip the sync edge). A negative status must FAIL the waiter, not let it run;
   with no edge the failure cascade never reaches it → `pocl_exec_command` memcpy on
   freed mem → SIGSEGV (`__memcpy <- pocl_exec_command <- pocl_pthread_driver_thread`).
   Fix: `failed_dependency` flag on `_cl_event`, set in `pocl_create_event_sync` +
   `pocl_broadcast`, honored in the pthread/basic submit+notify paths.
   → **PR #2214** (bricevideau-ai/pocl `fix-event-sync-failed-dependency`); jansol
   reviewed, all comments addressed (renamed flag, dropped the helper, folded the
   notify check into the existing failure clause). Revised commit `b55a569a5`.
2. **clSetEventCallback inline callback no-retain** — the synchronous (inline)
   callback path fired `callback_function` holding no reference, so a concurrent
   release (or the callback's own `clReleaseEvent`) could free the event mid-call.
   Spec: "all callbacks registered for an event must be called before the event is
   destroyed." Fix: retain across the inline call, release after.
   → **PR #2215** (`pr-inline-retain`, commit `23c8ce3cd`).
3. **Concurrent double-finish** — any command with ≥2 wait-list deps that reach a
   terminal state concurrently can be finished twice (two error broadcasts, or a
   fail racing a completion), tripping `assert(event->status > CL_COMPLETE)` at
   pocl_util.c:1690 (Debug) / UAF in clReleaseEvent (NDEBUG). NOT marker-specific —
   confirmed a plain `clEnqueueReadBuffer` with two concurrently-failed deps
   double-finishes (tried scoping the guard to MARKER/BARRIER, re-enabled the assert
   for other types, and it immediately caught READ_BUFFER → guard must be
   unconditional). Fix: idempotency guard — event state is monotonic, so return if
   already terminal under the locks already held.
   → **PR #2216** (`pr-double-finish`, commit `4eb82e2f7`); reproducers
   `scratch/pocl_double_error_repro.c` (pure two-errors) + `pocl_failed_dep_gate_repro.c`.
4. **claspr-side: marker not inside the start-gate** (commit `6d76fe2`). The async
   terminal (`run_eager_chain`) released `start` BEFORE enqueuing the terminal
   marker, so the marker's wait-list edges were wired against deps already free to
   resolve/fail — defeating the start-gate for the one command that aggregates the
   rest, and feeding bugs #1/#3 above. Fixed: enqueue the marker before releasing
   `start` (non-blocking, no deadlock); all release paths routed through one
   `release_start()` helper. The blocking terminal was already correct.

Heisenbug notes (kept for the next person): valgrind memcheck + helgrind + gdb all
HIDE these (serialization kills the timing); catch via NATIVE core dump (apport
stash `/var/lib/apport/coredump/`) or a Debug pocl build (the assert turns the
race into a clean, catchable abort). The two-thread backtrace at the assert is in
PR #2216's body.

**Integrated validation (pocl built with all 3 fixes, Debug/asserts-active,
installed ~/local/pocl):** full green on ALL THREE ICDs — tier1+tier2 263 passed
0 failed × {pocl, NEO legacy Iris Plus Gen11, rusticl/llvmpipe}; async error-path
stress 40-80×/ICD with 0 aborts; all 9 examples build + test + run clean ×3 ICDs.
The historical NEO host-seam deadlock and the rusticl image2d SEGV baseline both
came back CLEAN (no known-failures left to exclude). pocl ctest: no regressions.

### Blocking borrowing upload verb `write_sync` (branch `eager-cutover`, 2026-06-23)

**✅ DONE.** Additive softening of the async owned `write`'s move-out tax. The
async `write` consumes its `UploadSource<T>` because `clEnqueueWriteBuffer` is
NON-BLOCKING — the source must outlive the event, so the op owns it + holds it via
a drop-callback (forces `buf.write(data.clone())` to keep `data`). `write_sync` is
the BLOCKING counterpart that legitimately borrows `&[T]`: it waits inline, so the
copy finishes before the call returns → no keep-alive, no ownership transfer, true
zero-extra-alloc borrow.

- **Shape:** `fn write_sync(&mut self, data: &[T]) -> Result<()>` on both
  `DeviceSlice` and `MappedSlice`. `&mut self` (not `&self`): the opencl3
  `enqueue_write_buffer` calls `buffer.get_mut()`, so the device path needs `&mut`
  anyway; using `&mut self` for SVM too keeps the verb consistent and honest
  ("contents change"). Borrows BOTH buffer and data → caller keeps everything,
  which is the whole point. Returns `Result<()>` (plain side-effect, NOT a
  `DeviceOp` — blocking write isn't a graph node, can't `.and_then`/`bundle!`).
- **Scope:** only the two host-source-CONSUMING async upload verbs got it —
  `DeviceSlice::write` + `MappedSlice::write`. Skipped: images already borrow
  `&'a [Pixel]` (no move-out tax); `USMSlice::write_from` is already a synchronous
  borrowing host memcpy on the uninit type; fill/read/copy have no host slice.
- **Reuse:** delegates to the existing raw blocking enqueue helpers, no clEnqueue
  body duplicated. `write_buffer_enqueue(self, &ctx, data, true, &[])` was reusable
  AS-IS (already takes a `blocking` flag). `svm_write_enqueue` has NO blocking flag
  (SVM lacks a native one) → `write_sync` enqueues non-blocking then `event.wait()`
  inline; helper untouched.
- **Marker bound:** mirrors `write` exactly (`M: HostWritable`) — blocking write to
  a host-RO buffer still rejected. New compile-fail fixture
  `buffer_ops_write_sync_on_host_read_only` proves it (`HostReadOnly: HostWritable`
  not satisfied), blessed + rustfmt'd.
- **Tests** (in `eager_buffer_ops.rs`): borrowed-write-then-readback for both
  types, each asserting the SOURCE SLICE and the BUFFER are still usable after
  (reuse `data` for a 2nd write, reuse buffer for a 2nd write); plus a
  length-mismatch test.
- **Chosen over** `From<&[T]>` (would still go through the owning async path /
  keep-alive) and over "give-back-the-Vec" (forces an owned source the caller may
  not have, and complicates the Output type). The async `write`/`upload` and every
  Output type are UNCHANGED — purely additive.

### Eager struct-graph cutover (branch `eager-cutover`, from main, 2026-06-18)

**✅ SVM/Mapped cutover DONE (2026-06-23) — the LAST dual-idiom path is gone.**
`MappedSlice`'s `write`/`fill`/`copy_to` were the only remaining borrow-based
Tier-1 builders (`SvmWriteOp`/`SvmFillOp`/`SvmCopyOp`); everything else had moved
to the eager `DeviceOp` graph. **PARTIAL cutover** because SVM `copy_to` was
ALREADY eager (`CopyTo2`→`CopyToOp` in `copy.rs`, all 10 cross-type pairs) — only
`write`/`fill` needed new ops. What changed: (1) the three old SVM builders'
enqueue bodies were lifted into `pub(crate) fn svm_{fill,write,copy}_enqueue` raw
helpers in `mapped.rs` (each does enqueue + retain + `register_use` on the
owner(s), via a shared `register_event_on`; all NON-BLOCKING — SVM has no native
`CL_BLOCKING` flag, the terminal waits); (2) new eager leaves `FillMapped` /
`WriteMapped` in `eager.rs` (init `MappedSlice`, mirror `Fill`/`WriteDevice`:
concrete-head no-arg `wait()`/`submit()` via new `concrete_svm_ctx`, pipe-fed
`wait_on`/`sync` from the blanket); (3) `MappedSlice::{fill,write}` retargeted to
consume `self` + return the eager op, `copy_to` retargeted to the eager `CopyTo`
trait (`CopyTo2`, consuming, yields `(src,dst)`); (4) `Pipe<MappedSlice>` got
`write`/`fill`/`copy_to` inherent verbs matching `Pipe<DeviceSlice>`; (5) the old
builders DELETED; their internal callers re-wired to the raw helpers
(`copy.rs` Mapped↔Mapped pair, `eager.rs` `WriteMappedUninit`/`FillMappedUninit`,
`mapped.rs` `alloc_zero`). **`map`/`map_mut` deliberately KEPT as `MapOp`/
`MapMutOp` host-access RAII** — they return guards for host reads/writes, are NOT
graph nodes (exactly like `DeviceSlice::map`/`map_mut` → `DeviceMapOp` which the
DeviceSlice fold also kept). SVM-specific semantics preserved: fill/write stay
non-blocking; non-blocking host-source writes `register_drop_callback` to keep the
source alive across the async window (mirrors `WriteDevice`); the copy/fill/write
events still auto-register on the buffer(s)' `last_use` so Drop's
`clEnqueueSVMFree` queue-orders after them. USM: no init-`USMSlice` write/fill
verbs existed (USM is host memory — `USMSliceUninit::{fill_into,write_from}` are
the synchronous host paths; `copy_to` is the eager `CopyTo` trait) — nothing to
cut. Test sites respelled to move-out form (`let buf = buf.fill(v).wait()?`):
`eager_svm_fill_copy.rs` (6 tier-1 fns; the `.after(&fill_evt)` write-after-fill
became sequential fill→write) + `eager_buffer_ops.rs` (1 fn). Re-exports: dropped
`Svm*Op` from `mapped`, added `FillMapped`/`WriteMapped`/`fill_mapped`/
`write_mapped` to the eager crate-root set. Full gate green: workspace build
(default + async-events, all-targets), fmt, clippy (default + async, `-D warnings`),
`RUSTDOCFLAGS=-D warnings doc`, compile-fail rustfmt; SVM/USM suites serial on pocl
(eager_svm_fill_copy 11, eager_buffer_ops 13, tier1 svm 9, stress_svm 1, svm_drop
3, eager_usm 8, eager_svm_chain 2, eager_cutover 20, safety_compile_fail 8, +
chain/terminals/host_view/piped/alloc/marker collateral). NOTE: `safety_compile_fail`
first-run-after-edit hit the documented ui_test-against-stale-rlibs artifact
([[compiletests_no_release]]) — green deterministically on re-run.

**✅ and_then_host async regression FIXED (cc5f3bc, 2026-06-22).** The eager host
seam had been (mis)ported to run the closure SYNCHRONOUSLY on the submit thread,
discarding the whole point of the map/user-event machinery. Restored the
in-queue worker-thread model from `and_then_host.rs`: `run_host_seam` enqueues
maps over the source's events, creates a user event, enqueues unmaps gated on it,
SPAWNS a worker (new `run_host_worker`), and returns the unmap events as deps —
chain continues at submit time, host stage overlaps device work. Worker waits map
events, runs closure under `catch_unwind`, stashes errors in the
`ExecutionContext` host-error slot, defensive-unmaps on error, signals the user
event. Applies to both `AndThenHost` + `AndThenHostWithContext`; closures now
need `+ 'static`. THREE latent issues the sync seam masked, all fixed in the same
commit: (1) terminals (`sync` + `EagerChainFuture`) must drain the host-error
slot even on `Ok` — pocl's `clEnqueueMarkerWithWaitList` does NOT cascade
negative user-event status, so a failed worker can leave the marker reporting
CL_COMPLETE; a non-empty slot is itself the failure signal; (2) `EagerChainFuture
::Running` gained a `host_error` Arc; (3) ORPHANED DEPS — `.and_then(|_buf|
value(0))` discards the source handle, so a host worker's user event never
reached the terminal (`sync` returned before the worker ran); `AndThen` now
threads the source pipe's un-taken deps into the result
(`thread_orphaned_source_deps`), as the old layer did via `next.execute(deps)`.

**host_view `View<'a>` RISK RETIRED (probed).** The flagged-medium-risk
`View<'a>` borrow is NOT in the host_view DeviceOperation leaves — `Acquire/
ReleaseDeviceSliceOp::Output` is an OWNED `DeviceSliceHostView` (owns buf +
host_ptr + RetainedQueue), so those leaves port to EagerOp by move like any
other. The `for<'a> FnOnce(View<'a>)` borrow lives ONLY in `and_then_host`'s
closure — the genuine host seam, which the design ALWAYS kept as an explicit
closure-at-execute boundary node (the host reads real mapped data mid-graph; it
is not an eager builder by nature). So: the eager model has exactly ONE
closure-bearing node — the host seam — by design, not as a limitation. No
blocker. host_view acquire/release leaves are mechanical ports; and_then_host
stays a closure boundary (its closure runs at execute, segmenting the graph).

**RESOLUTION for multi-output shapes (spiked green) — the parity recipe.**
The suite survey shows the hard shapes are: multi-output kernels (`add_u32` →
`(a,b,out)`), element-selection (`|(_a,_b,out)| download(out)`), bundle tuple
destructure (`|(a,b,out)|`), Arc fan-out (`arc_split::<N>`, `.arc()`+clone).
All reduce to ONE mechanism: **a multi-output op's `Handle` is a TUPLE OF PIPES
(one per element), and `execute` SCATTERS its runtime tuple into them.** Then a
downstream `|(_a, _b, out)| …` closure receives `(Pipe<A>, Pipe<B>, Pipe<Out>)`
— selection is just dropping the unused pipes; no runtime-tuple-destructure
needed. Spiked: `Kernel3{Output=(A,B,C), Handle=(Pipe<A>,Pipe<B>,Pipe<C>)}`,
`handle()` returns the three, `execute` puts each — `|(_a,_b,out)| sink(out)`
works. TODO to reach parity:
  - kernel macro: when Output is a tuple, emit `Handle = (Pipe<..>, …)` + per-
    element output pipes + scatter in execute (currently one `Pipe<Output>`).
  - bundle: override `Handle = (A::Handle, …)` (branch pipes already held).
  - Arc fan-out: a `split::<N>`/clone-at-execute combinator — `Arc<T>` is `Clone`
    so the consumer pipes each get a clone (N readers); the producer scatters
    N clones. arc_split is this with a fixed N.
  - stateful `(buf, step)`: falls out of tuple-of-pipes (step is just a
    `Pipe<u32>` element).
  - host seams (`and_then_host`/`_with_context`): stay closure-at-execute nodes.
LESSON (Brice): should've ported the suite directly (all-fail-then-fix) to see
this shape set at once instead of piecemeal.

**⚠ GAPS FOUND porting the full suite (systematic, sub-agent clusters) — the
parity backlog. ALL 8 GAPS CLOSED 2026-06-22 (commits 4811c5b small gaps,
c130145 transfer+async, d756e0d bundle gather + arity 2..=16 + eager_bundle!,
2f681d2 EagerDynOp, 81e5d7e heterogeneous carry) + the and_then_host async
regression FIXED (cc5f3bc, above).**

**⭐ NEXT: Tier-1/Tier-2 REUNIFICATION (plan approved 2026-06-22, subsumes the
destructive cleanup).** Collapse `EagerOp`+`KernelOp`+`DeviceOperation`+the
standalone Tier-1 buffer builders into ONE trait `DeviceOp` (abbrev of
cuda-oxide's `DeviceOperation`; the `sync`/`wait`/`submit` terminal vocabulary is
cuda-oxide / Rust-CUDA heritage — README must credit both). Unified terminals
(`wait`/`wait_on`/`submit`/`submit_on`/`sync`/`run`); buffer verbs become methods
on concrete AND piped buffers; concrete-head ops keep context-free `wait()`;
marker-turbofish ergonomics (`upload(src, ReadWrite)` witness-arg; `from_slice`
stays the ONLY immutable-init path); delete old closure layer + flatten the
`eager` namespace to crate root. 7 staged green sub-commits. Plan file:
`.claude/plans/declarative-hopping-parrot.md`.

**PROGRESS:** stages 1+2 (`38109f1`, `9177443`); stages 4+5 LANDED together
(kernel macro is `DeviceOp`-only + old closure layer deleted + namespace
flattened). The old `DeviceOperation` trait is gone; its only residue is a tiny
`pub trait DeviceEnqueue { type Output; fn run(ec, deps) -> (Output, Deps) }` in
`eager.rs` that the host-view acquire/release leaves (`host_view.rs`) and the
`copy_to` family (`copy.rs`) delegate their raw enqueue body to — the eager wrappers
can't reconstruct those private-field types, so they hold the buffer/view and call
`.run()`. `Dep`/`Deps`/`deps_as_events`/`wrap_event` moved from the deleted
`device_op.rs` into `eager.rs`. `eager_bundle!`→`bundle!`. Entry macros
(`upload!`/`download!`/`device_slice!`/`mapped_slice!`/`usm_slice!`/…) re-pointed
to the eager free fns. `eager_*` crate-root aliases dropped; eager items
re-exported with plain names (`claspr::eager::X` paths still work — tests unchanged
there). **CAUGHT + FIXED a stage-4 regression:** the deleted `KernelOp::enqueue_into`
carried an `assert_same_context` loop over caller-added `.after()` deps that the new
`DeviceOp::execute` had dropped — cross_queue's cross-context-panic test went red;
restored the loop in both execute bodies (single + multi-output). Stage 3
(`e805d4a`, `daf5cc4`: fold Tier-1 buffer + image builders into DeviceOps), stage 6
(`0038e9f`: marker ergonomics + piped-buffer methods + G1). **STAGE 7 (FINAL) LANDED
— reunification COMPLETE.** The two examples (async-pipeline, batch-inference) were
migrated off the deleted closure surface (`upload!`→`upload`, `download!`→`download`,
`DeviceOperation`→`DeviceOpExt`, `claspr_async::`→`claspr::eager::`; the batch
fan_out keeps its `Arc<DeviceSlice>` shared-weights pattern) and re-added to the
workspace members; both build + run + pass their inline tests on pocl. Their
`claspr-async` deps were dropped (they depend only on `claspr` + `claspr-build`).
README: cuda-oxide / Rust-CUDA credit added to "Prior art and inspiration" plus the
device-graph + workspace-layout + three-layers sections rewritten off the
two-tier/`claspr-async` framing onto the unified `DeviceOp` surface. Compile-fail
suite RE-CREATED (not deferred): `tests/tier2/tests/safety_compile_fail.rs` (ui_test
direct-rustc harness cloned from tier1's `image_compile_fail`) with two fixtures —
`fill_on_frozen` (`Frozen` isn't `Fillable`) and `arc_to_writable_arg`
(`Arc<DeviceSlice>` impls only `KernelSliceReadArg`); goldens blessed, CI's
`tests/*/compile_fail` rustfmt glob picks them up automatically. The
`use-after-move` / `host-view-escape` invariants from the old suite were NOT
re-created (they're move-semantics / lifetime checks that the eager move-out idiom
already enforces structurally; the two re-created fixtures are the marker/trait-bound
ones worth a golden). `claspr-async` (the thin `pub use claspr::*` re-export shim)
has since been DELETED (post-reunification cleanup): the crate directory, its
workspace-member entry, and the last `claspr-async` dep (spike) are gone, and the
remaining `claspr_async::` mentions in test/doc comments were re-pointed to
`claspr::` or rewritten as historical migration notes. Full gate green: workspace
build (default + `--features async-events`), `cargo fmt --check`, clippy (default +
async-events), `RUSTDOCFLAGS=-D warnings cargo doc`, compile-fail rustfmt, and the
entire tier1+tier2 suite serial on pocl. Pre-existing env failure unrelated to this
work: `image_dispatch` dim2_array/dim3 (pocl lacks 3D/2D-array `write_image`
builtins on this CPU — fails identically at HEAD).

**📋 POST-REUNIFICATION (Brice): re-read cuda-oxide's async docs and decide
match-vs-diverge per primitive.**
https://nvlabs.github.io/cuda-oxide/async-programming/the-device-operation-model.html
and .../combinators-and-composition.html — compare claspr's final DeviceOp model
against cuda-oxide's; where we're close, match their naming/shape; where we
genuinely diverge (eager inspectable struct-graph vs theirs), keep ours and
document why.
- ✅ **transfer_to_device** — DONE (c130145). Eager leaf `transfer_to_device(buf,
  device)` wrapping clEnqueueMigrateMemObjects on the target OOO queue;
  re-export `eager_transfer_to_device`; composes with `.on_device`.
- ✅ **DynOp → EagerDynOp** — DONE (2f681d2). Object-safe `ErasedEagerOp<T>` shim
  (`collect_erased(self: Box<Self>)`, blanket over every `O: EagerOp`, delegates
  to `O::collect`) boxed into single-output `EagerDynOp<'op, T>`. Multi-output
  inner ops erase cleanly (tuple becomes `T` via `collect`; per-element handle
  dropped — fine for conditional arms agreeing on one Output). All of
  conditional.rs ported (eager_conditional 10/10, was 1 + 8 blocked).
- ✅ **Host-value passthrough / host reduction / scalar carry — DONE (81e5d7e),
  the LAST gap.** Fixed by a type-system change, NOT a host-value seam (an
  `and_then_host_value` was explicitly rejected: sending host data TO the gpu is
  `and_then_host`'s job [map→write→unmap], and host scalars are computable
  eagerly — the real question was just whether they can flow as graph edges, and
  they can). Three composable pieces: (1) `Pipe<T>: EagerOp` (identity node) so a
  bare pipe is a bundle/and_then source with no `forward()`; (2) `Value<T: Clone>`
  exposes a BY-VALUE handle (`Handle = T`) so a downstream closure gets the value
  not a pipe → build-time host compute works (`value(n).and_then(|n| value(n+1))`;
  carried `step + 1` in-chain); non-Clone owned resources use the new `lift()`
  leaf (default Pipe handle); (3) `bundle` composes per-branch handles
  (`Handle = (<$ty>::Handle,)`) so `bundle!(kernel, value(7))` hands down
  `(Pipe<DeviceSlice>, u32)` — buffer-pipe + scalar-by-value. Un-blocked all 3
  arc_split host-reductions (no arc_split op needed — by-value `value` covers
  host fan-out) + ml_pass repack (faithful). ALSO fixed a latent recurrence of
  the d756e0d multi-output gather bug: every single-source wrapper
  (and_then_host{,_with_context}/on_device/arced/arc_split/and_then_with_context/
  profiled) drained `source.output_pipe()` → broken for bundle sources; all now
  `source.collect()`. NEW requirement-lock suite `eager_heterogeneous_carry`
  (4 tests) makes pipe+scalar carry + in-chain scalar compute a COMPILE/RUN
  requirement so a redesign can't silently drop it.
- ✅ **FanOutExt method form** (`vec.fan_out(op)`) — DONE (4811c5b). `EagerFanOutExt`.
- ✅ **async terminal `.run().await`** — DONE (c130145, extended d756e0d).
  `EagerChainFuture` + `EagerOpExt::run` (async-events feature); arity-agnostic —
  multi-output works via the `collect` seam (single-output limitation lifted).
- ✅ **eager `.profiled(cb)`** — DONE (4811c5b). `Profiled` + `EagerProfileExt`.
- ✅ **`catch_unwind` in the host seam** — DONE (4811c5b). `run_host_seam` wraps
  the closure; panic → `Error::HostPanic`.

**✅ ROOT-CAUSE BUG FIXED (d756e0d) — nested multi-output gather.** Composites
(bundle*, fan_out) drained each branch's single `output_pipe().take()`; a branch
that is itself multi-output (nested bundle, arc_split, copy pair, multi-output
kernel) never fills that pipe → `NotSupported("a branch produced no output")`.
Failed at HEAD: eager_diamond (nested bundle-of-bundles), eager_cutover arc_split
fan-out. (Believed-green earlier — nested shapes weren't run serially on the
correct ICD; see [[pocl-icd-path-per-machine]].) FIX: non-blocking gather seam
`EagerOp::collect(ec,mode)->(Output,Deps)` (default single-pipe drain; multi-output
ops override it instead of `into_output`). `into_output` = `collect` + wait once;
composites call `branch.collect(Pipelined)`; `run` uses `collect` too. Net: N
`into_output` overrides → N `collect` overrides + ONE wait. Also restored
`Bundle2..=16` + variadic `eager_bundle!` (the suite port had only 2/3/4, nesting
bundle2 for wider — which is what surfaced the bug). The two `chain.rs` gaps below
(bundle multi-arg Handle, host-scalar transform) are subsumed: bundle Handle is
already per-branch pipes, and the multi-output gather now composes through nesting.

**⚠ TWO GAPS FOUND porting chain.rs (eager_chain.rs proof, 5/5 green):**
1. **`bundle(...).and_then(|(a,b,out)| kernel(a,b,out))` — bundle Handle is one
   `Pipe<(A,B,C)>`, not per-branch pipes.** So a bundle can't feed a multi-arg
   kernel directly (the workhorse shape; diamond_arc uses it heavily). FIX: apply
   the SAME multi-output treatment bundle's siblings already have (CopyTo2 / the
   multi-output kernel macro): bundle stores per-branch pipes (it already does),
   override `type Handle = (A::Handle, B::Handle, …)` + `handle()` returns them +
   `into_output` reconstructs the tuple for the terminal (move-once: branch pipes
   are the storage, NOT drained into a single `out`). REAL, fixable, contained.
2. **`value(x).and_then(|n| value(n+1))` — host-scalar transform mid-graph.**
   `and_then` hands a `Pipe<u32>`, not the scalar; `and_then_host` is for device
   `&mut [T]` views, not host scalars. Arguably a non-shape (`value(42)
   .and_then(|n| value(n+1))` IS `value(43)` — no device work), but a host-value
   `map` seam is trivial if wanted. LOW priority; the test rewrote to up-front
   compute.

**⚠ KNOWN GAP — `and_then_with_context` dep edge (fix during suite port).**
The eager `and_then_with_context(|ec, value| …)` closure receives the upstream
VALUE, so the downstream op takes it as `Input::Concrete` (EMPTY deps) → no
event edge to the source's command. The impl merges source deps into the
downstream's OUTPUT deps (terminal completion correct), but on a strict
out-of-order queue the downstream command has no enqueue wait on the source →
potential data race (pocl happens to order it, so the test passes). Contrast:
regular eager `and_then(|pipe| …)` passes a PIPE → downstream resolves it →
deps reach the enqueue → correct. FIX: make `and_then_with_context`'s closure
receive `Self::Handle` (the pipe), matching `and_then`, so the downstream
threads deps. The real call sites (device routing: `|ec, buf| kernel(buf)
.on_device(...)`) feed `buf` into a kernel which takes `impl ToInput` (accepts a
pipe), so pipe-passing should typecheck — VERIFY when porting those tests
(don't guess the signature without the call sites — the lesson). on_device +
and_then_host do NOT have this gap (on_device re-points ec; host seam drains
deps before the host read).

**EXECUTE-TIME CLOSURE NODES (spiked green) — and_then_with_context / on_device
/ and_then_host.** These 3 combinators are NOT eager builders (their closure
needs the live `ec` / runtime mapped data, absent at build). They're
closure-at-EXECUTE nodes: the struct holds `f: Option<F>` + source pipe + out
pipe; `execute(self, ec, mode)` runs source, takes the upstream runtime value,
runs `f(ec, value)` (or `f(view)` for host seam) NOW to get the downstream op,
grabs its out-pipe BEFORE `run`/execute (move-once), runs it, moves result to
out. Spiked: capture `downstream.output_pipe()` before `downstream.execute()`.
host seam (`and_then_host`) additionally drains the upstream `Deps`
(blocking-wait) before the closure reads the `Mappable` View<'a> (host touches
real data). This is the ONE place closures legitimately survive in the eager
model — by design (host/scheduling concern, not graph description).

**MOVE-ONCE RESOLUTION (spiked green /tmp/inferspike) — implementation shape.**
The tension: a multi-output kernel's buffers can't be moved BOTH into a single
`Pipe<(A,B,C)>` (terminal) AND into per-element pipes (downstream) — DeviceSlice
is not Clone. Resolution: **the per-element pipes ARE the storage** (no single
output-tuple pipe for multi-output ops). `execute` scatters each buffer into its
element pipe (move-once). Two consumers, mutually exclusive by build-time wiring:
  - downstream `and_then`: `Handle = (Pipe<A>,Pipe<B>,Pipe<C>)`; closure
    `|(_a,_b,out)|` takes the pipes it wants, drops the rest (move-once OK — the
    dropped element pipes are simply never `take`n).
  - terminal `sync`/Tier-1 `wait`: RECONSTRUCTS the `Output` tuple by draining
    all element pipes (`(pa.take, pb.take, pc.take)`).
⇒ This is a TRAIT-CONTRACT change, not just a macro addition: `sync`/the terminal
must drain element-pipes-and-reconstruct for multi-output ops, while single-output
ops keep the `output_pipe().take()` path. Cleanest uniform shape to design next
session: either (a) `output_pipe()` for multi-output returns a pipe that
`execute` fills by reconstructing-after-scatter (defeats move-once — NO), or
(b) make the terminal call a new `EagerOp::into_output(self, ec, mode) ->
Result<Output>` that each op implements (single: take its pipe; multi: scatter
then reconstruct), and `and_then` keeps using `handle()`. (b) is the clean one —
unifies single+multi, no double-move. INVASIVE (trait + macro + bundle + sync
together) → do as one focused green-at-end change with the direct-suite-port
driving it. Event note: the single enqueue event is one `Dep`; put a clone
(Event is Arc) on each element pipe, or carry it on element-0 and have
reconstruct gather — decide at impl.

**~~LIMIT~~ MISDIAGNOSIS, CORRECTED (Brice caught it).** I claimed a bundle's
tuple output couldn't be split into per-branch pipes downstream. WRONG — that
was self-inflicted: I hardcoded `and_then`'s closure to receive
`Pipe<Self::Output>` (always ONE pipe). A bundle actually HOLDS `pa: Pipe<A>` +
`pb: Pipe<B>` separately, so it can hand the closure `(Pipe<A>, Pipe<B>)`.
**Fix (spiked green, incl. nesting):** give `EagerOp` an associated
`type Handle: Clone` = "the build-time downstream-facing shape", default
`Pipe<Output>`; `and_then`'s closure receives `Self::Handle`. A bundle overrides
`Handle = (A::Handle, B::Handle)` → `bundle(a,b).and_then(|(pa,pb)| …)` works,
and nests (`(Pipe<u8>, (Pipe<i8>, Pipe<i16>))`). This makes bundle MORE
expressive than the closure model (branches exposed individually at BUILD time,
not just as a destructured runtime tuple). TODO: implement the `Handle` assoc
type (currently `and_then` hardcodes `Pipe<Output>`; leaves/kernels keep the
default, bundles override). No expressiveness loss after all.

Converting the closure-based `DeviceOperation` layer to the proven closure-free
eager model (see `closure-free-graph` branch for the probe + design + 3-step
validation). Branched from **main** (clean two-crate baseline; the cb-graphs
accumulation is NOT carried — no Slots/Pick/SlotKernelCall/record cruft).

**The model:** a graph is a closure-free nested struct of `EagerOp`s; `.and_then(f)`
runs `f` at construction with a `Pipe<T>` handle (carrying `(value, Deps)`),
storing the returned op. Non-blocking enqueue threads events through pipes; one
terminal wait in `sync`. `Input<T> = Concrete|Pipe` is the unified edge.

**Progress (each step green + committed):**
- **1a DONE** (`4079a6b`): `claspr/src/eager.rs` (was claspr-async) — `EagerOp`/
  `Pipe`/`Input`/`AndThen`/`sync` + real `alloc_zero`/`fill` leaves. 3/3 hw green.
- **FOLD DONE** (`fce9bfd`): claspr-async folded into claspr. WHY: the macro emits
  `::claspr::` paths and claspr can't depend on claspr-async (circular), so for
  kernel ops to take `Input<T>` the eager core must live in claspr. (This
  reversed my initial "keep two crates" call — flagged to Brice, he said fold.)
  Cleaner than the cb-graphs merge: only opencl3 extra dep, no record/cl3. claspr
  -async = re-export shim. Whole workspace builds; existing tier2 suites green
  through the shim (no regression).
- **Transfer leaves DONE** (`8ea081e`): `upload` (alloc+COPY_HOST_PTR) +
  `download` (non-blocking read→Vec, event-threaded) eager leaves. upload→fill→
  download round-trip green (5/5). Old closure layer still live in parallel
  (kernels can't enter eager until 1b) → zero regression.
- **1b — kernel macro — DESIGN VALIDATED, REWRITE PENDING (the capstone).**
  Two coherence/inference snags solved via spikes (/tmp/inferspike, both green):
  - per-buffer-family `IntoKernelInput<E>` impls (DeviceSlice/Mapped/USM + a
    `Pipe<D>` impl) — NOT a blanket over `D` (that conflicts: a `Pipe` could be
    a `KernelSliceArg`). `kernels.foo(buf)` and `kernels.foo(pipe)` both infer,
    no turbofish.
  - **associated `Op` type** preserves Tier-1 compile-time safety: `IntoKernelInput`
    has `type Op: EagerOp`; concrete buffer → `ConcreteKernelOp` (has inherent
    `.wait()`/`.submit()`/`KernelOp`), pipe → `PipedKernelOp` (EagerOp only, no
    `.wait()`). One method serves both tiers; `.wait()` exists ONLY on the
    concrete variant. SPIKED working.
  - **Multi-arg `.wait()` finding + resolution (Brice):** with N buffer args
    each independently concrete-or-pipe, `.wait()` can't be compile-gated
    per-arg (concrete-ness is per-`Input` runtime). BUT **users cannot
    construct a `Pipe`** — pipes only exist as `and_then`'s closure parameter.
    So "holding a pipe and calling `.wait()`" is unreachable (if you have a
    pipe you're mid-graph-build, not calling a terminal). ⇒ a **unified single
    method** taking `Input` args, returning an eager Op that also carries
    `.wait()` (resolves Inputs; the all-concrete case is the only reachable
    one), is safe. No two-method split needed. Spiked: uniform `Input<D>`
    multi-arg infers for all-concrete AND mixed, no turbofish.
  - **Scope/risk:** this is the deepest single change — rewrites the macro's
    Op emission (arg classification ~497, Op struct ~683, KernelOp impl ~798)
    while keeping the existing Tier-1 surface working for ~17 kernel-chaining
    tier2 tests + all Tier-1 use. Element type `E` is fixed per kernel (from the
    sig), so the macro hardcodes it; only the buffer generic varies. The eager
    kernel leaf reuses `LaunchOp` (the same enqueue path `KernelOp` uses).
    Pending — the capstone.
  - **Exact emission shape VALIDATED** (/tmp/inferspike, green): per buffer arg
    emit TWO generics — `__D{n}: KernelSliceArg<elem>` (the buffer) +
    `__I{n}: ToInput<elem, Buf=__D{n}>` (the arg, concrete-or-pipe). Method takes
    `__I{n}`, stores `Input<__D{n}>` in the Op. `ToInput<E>` is a new claspr
    trait (per-family impls for DeviceSlice/MappedSlice/USMSlice + `Pipe<D>`).
    Op is generic over `__D{n}` only; Output flows the `__D{n}` buffers. Tier-1
    methods + `KernelOp` stay (resolve Inputs — all-concrete is the only
    reachable terminal case); add `EagerOp` impl (resolve Inputs from pipes,
    enqueue via `LaunchOp`, deposit in output pipe). Scalars unchanged.
    `ToInput` DONE + committed (`332418d`), green.
  - **UNIFIED-TERMINAL DESIGN (Brice's original intent, corrected course):**
    ONE Op structure with the **output `Pipe` as the single source of truth**.
    `EagerOp::execute` is the ONLY enqueue body (resolve `Input`s → set args →
    `LaunchOp` enqueue → deposit buffers+event in the output pipe). `wait()` /
    `submit()` (Tier-1) are thin terminals OVER that: run `execute`, take from
    the pipe, block on its deps, return the buffer(s) (the move-out contract —
    `kernels.foo(buf).wait()? -> buf` — is just "take the Output from the
    terminal pipe", verified faithful).
    **Terminal opt-in optimizations (Brice) — grounded in existing code.**
    Tier-1 ALREADY does this: `WriteOp/ReadOp::wait_on` enqueue with `CL_BLOCKING`
    (native blocking, NO event allocated), while `submit_on` uses `CL_FALSE` +
    event (buffer.rs ~571/607, ~ReadOp same). My eager `download` REGRESSED this
    — it uses `submit_on`+event even at a `.sync()` terminal (eager.rs Download),
    doing the event round-trip Tier-1's blocking read avoids. So `EagerOp::execute`
    takes an `ExecMode` param with propagation rule: the TERMINAL op (outermost,
    called by `sync`/`wait`) gets `ExecMode::Blocking`; everything upstream gets
    `Pipelined`. `AndThen::execute` passes `Pipelined` to `source`, forwards the
    caller's `mode` to `next` — so blocking is used ONLY at the tail. A
    blocking-capable op (read/write/fill/copy) given `Blocking` calls its
    `wait_on` (CL_BLOCKING, no event); given `Pipelined` it uses `submit_on`+event.
    Ops with no native blocking mode (kernels) ignore the hint. This is a real
    perf win (one fewer event+wait per chain) AND restores Tier-1 parity for the
    `…download().sync()` shape. So NO separate `KernelOp::enqueue_into`
    path and NO Tier-1/eager fork — kernels drop the `KernelOp`→old-blanket
    entirely and impl `EagerOp` only; Tier-1 terminals become inherent methods
    that drive `execute`. Simpler than dual-impl. (This supersedes the "Op impls
    both traits" idea from `332418d`'s message — that was a transition crutch;
    the unified-terminal shape is the real target.)
  - **CORRECTION: the cutover IS incremental** (my "atomic" claim was wrong —
    re-spiked). The E0034 `.and_then` ambiguity only fires when a single file
    imports BOTH `DeviceOperation` and `EagerOpExt` AND calls `.and_then` on a
    bare kernel op. A kernel Op can impl BOTH traits fine; consumers that import
    only one have no ambiguity (spiked: both-traits-on-Op + one-import = OK). And
    kernel ops used as `and_then` closure RETURNS are consumed as the *upstream*
    op's trait — the kernel op's own trait set is irrelevant there. ⇒ the macro
    op impls `EagerOp` NATIVELY *in addition to* the existing `KernelOp`→old-
    `DeviceOperation` blanket; old chains/tests keep working (import
    `DeviceOperation`), new eager tests import `EagerOpExt`. The one direct-
    `.and_then`-on-bare-kernel site (examples/batch-inference:137) just needs its
    file to import one trait. Incremental, green-at-each-step after all.

**Then (per CONVERSION PLAN, carried mentally):** port remaining leaves
(transfer/copy/uninit/usm/image_transfer/host_view), host seams (and_then_host /
and_then_with_context as execute-time boundary nodes), fan_out/bundle marker-join,
slots/bind/call, delete old closure trait. **Observe what (if anything) the new
paradigm CANNOT express** — Brice's explicit interest; surface it when hit, don't
presume.

### 🧭 Replayable-eager-graph PROPOSAL (2026-06-25, from deep study of cb-graphs-impl/closure-free-graph + spikes)

Two agents studied the prior replay impl (`origin/closure-free-graph`:
record.rs 1471 / call.rs / slots.rs / slot_ops.rs / arg_bind.rs) + the 4 spikes
(graph_cb, graph_devop_record, graph_slots, combinator), mapped onto the LANDED
eager engine. Verdict: **the eager struct-graph is strictly MORE amenable to
replay than the old closure layer** — the closure layer's trace-once machinery
existed only because `and_then`'s `FnOnce` made the graph non-inspectable
(description==execution). The eager `AndThen{source,next}` stores BUILT ops, so
the graph is already the IR.

**Linchpin VERIFIED in code:** recordable leaves can expose handle+params by
`&self` — `Fill{ buf: Input<DeviceSlice>, value: T }`, `DeviceSlice::buffer() ->
&ClBuffer` (buffer.rs:415). So **recording can be a non-consuming `&self` walk**
(`record(&self, &mut RecordContext)`), NOT the old consuming `record(self)` that
forced a factory. This is the key simplification the eager engine unlocks.

**Proposed architecture (4 layers, each shippable on its own):**

1. **`RecordableOp: DeviceOp` sub-trait** — one `record(&self, &mut RecordContext)
   -> Result<(Output-handle, SyncPoints)>` mirroring `execute` but threading
   `cl_sync_point_khr` where execute threads `Deps`. Recordability propagates as
   a conditional bound: `impl RecordableOp for AndThen<S,U> where S: RecordableOp,
   U: RecordableOp` — zero runtime walk, compile-time. Recordable leaves: kernel/
   fill/copy/barrier. Non-recordable (`Upload`/`Download`/`AndThenHost`/`OnDevice`)
   just don't impl it → a chain containing one fails to compile at `.record()`
   with an E0277 naming the offending leaf through the generic wrappers. **Drops
   in near-unchanged from spike `graph_devop_record`** (the eager AndThen is
   closer to recordable than the spike's — children already built).

2. **Dual-backend `RecordContext`** (KEEP from old record.rs near-verbatim):
   `Backend::{Cb, Software}` behind typed `fill_buffer`/`copy_buffer`/`barrier`/
   `ndrange_kernel`. Cb → real `cl_khr_command_buffer` (replay = 1
   `clEnqueueCommandBufferKHR`); Software → `Vec<SoftCommand>` replayed with
   FRESH per-replay events along the static sync-point topology (no stale events).
   FFI loader (KEEP verbatim): provisional ext → resolve via
   `clGetExtensionFunctionAddressForPlatform` on the cl3-already-dlopened loader +
   transmute to opencl-sys PFNs (opencl3 safe wrapper is unusable — returns
   -2001). RAII `RecordedCb`.

3. **Replay-twice surface.** Smallest first step (works TODAY, zero engine change)
   = the **factory `Fn() -> Graph`** rebuilt per run, for the all-buffers-in-graph
   class. Then a `RecordedGraph`/`Graph<I,O>` cache wrapper (Arc-erased, nameable
   return type for library export; per-arity `.call` macro from spike graph_cb)
   that records once and replays N times.

4. **Slots / rebindable inputs (LAST, only if needed).** `type Slots` (HList) +
   `Complete` gate on terminals + `bind`/`call::<Tags>` + `clUpdateMutableCommands
   KHR` in-place arg swap (KEEP slots.rs algebra + call.rs verbs). This is the one
   genuinely-new axis (eager `Input` is resolve-once). Defer until replay-with-
   different-buffers is actually wanted.

**KEEP from old branch:** slots.rs (whole algebra), the FFI loader + RAII +
mutable-dispatch plumbing, RecordContext dual-backend + SoftCommand/Software
CB, per-leaf record bodies (twins of execute), the call/bind/cached verb surface.
**DROP (closure-era scaffolding):** trace-once-by-running-closures, the concrete/
slot fork (Pick / SlotKernelCall-vs-Op / KernelArgBind::Dispatch second axis /
Reuse-as-machinery) — collapses because eager `Input::Concrete` + Arc-cell Pipe
identity already unify "known value" and "deferred handle".

**Buffer-persistence crux (the real design question):** an owned
`Input::Concrete(DeviceSlice)` survives in the graph struct across `&self`
replays; a `Pipe`-fed (in-graph-allocated) buffer is produced at trace time and
must be re-instantiated or kept-alive per replay. Slots/mutable-dispatch are how
you replay against DIFFERENT buffers; plain `&self` replay reuses the SAME ones.

**Recommended build order:** (1) RecordableOp sub-trait + Software backend +
`&self` record walk (no CB FFI yet, fully testable) → (2) CB backend + FFI loader
→ (3) cache wrapper + `.call` → (4) slots/mutable. Each green on its own.

### 🔬 Graph-reuse investigation (2026-06-24, post-cutover, READ-ONLY — no code yet)

Question (Brice's next step): **can `g.sync(&ctx)?` run twice — no command
buffers, no slots yet?** Investigated the LANDED eager engine (agent + direct
verification). Findings:

- **Single-use is a COMPILE-TIME guarantee, not a runtime bug.** Every terminal
  (`sync`/`wait_on`/`submit_on`/`run`) and `execute`/`collect`/`into_output`
  takes `self` by value (eager.rs:391/410/427/562/654/735), so a second
  `g.sync()` is use-after-move — it won't compile. So "run twice" is an API to
  ADD, not a bug to fix.
- **The DeviceSlice-ownership crux defines the boundary.** Buffers from in-graph
  leaves (`upload`/`alloc_*`/`value`) are allocated FRESH each `execute`
  (Upload→`DeviceSlice::from_slice` ~1893; AllocZero→`alloc_zero` ~1699), so a
  rebuilt graph re-allocates. But a CAPTURED concrete `DeviceSlice`
  (`Input::Concrete`, `lift(buf)`, a buffer handed to a kernel) is moved out by
  `Input::resolve` (~195) and `DeviceSlice` is **not `Clone`** (buffer.rs:128) —
  can never be re-run without re-supply. Host seeds (`Vec` for upload, scalars
  for fill) are cheap to clone.
- **`AndThenHost` holds an `FnOnce`** (Option<F>, ~4060) → permanently rules out
  any `Clone`/`&self`-reuse of host-seam graphs.
- **Already reuse-friendly:** `AndThen{source,next}` stores BUILT ops (eager,
  closure-free, ~486); `describe()` is a non-consuming `&self` walk (but emits
  only a flat name `Vec<String>` — no identity/edges/operands, so it can't drive
  a re-interpreter); `Pipe` is `Arc`-shared; `Context` is `Clone`.

**Approaches ranked:** (a) `&self`-execute = effectively a rewrite, blocked by
`FnOnce` host seams + non-Clone DeviceSlice — rejected. (b) **factory
`Fn() -> Graph` rebuilt per run = smallest, matches the long-standing decision
(reuse is a factory, not Clone).** Works TODAY with zero engine change for the
upload→kernels→download class: `let make = || upload(v.clone()).and_then(...);
make().sync(&ctx)?; make().sync(&ctx)?;`. Optional ~15-LOC `Pipeline { factory:
Arc<dyn Fn()->Op + Send + Sync> }` wrapper = the `Arc`-shareable, CB-free "walk
DAG" tier of the eventual `.call()` design. (c) `describe`-driven re-interpreter
= infeasible without growing `describe` into a reified IR (bigger lift; overlaps
the deferred CB IR).

**Recommendation / smallest first step:** ship the factory pattern for the
"all buffers in-graph-allocated" class (works now); optionally add the thin
`Pipeline` wrapper as the explicit eager/non-CB arm of `.call()`.

**Decisions needed from Brice:** (1) accept the natural boundary (only
in-graph-allocated buffers re-runnable; captured `DeviceSlice` needs re-supply,
maybe via `Arc<DeviceSlice>` later)? (2) ship the `Pipeline` wrapper now, or keep
reuse as a documented closure idiom until `.call()` lands? (3) build this as the
explicit non-CB tier of the CB `.call()` design so it's not throwaway? (4) leave
`describe` as a name list, or grow it into an IR (serves both re-run + CB export)?

### Command-buffer-backed graphs (design + spikes, 2026-06-12..15)

**Goal.** When the platform supports `cl_khr_command_buffer`, record a
recordable Tier 2 sub-chain into a CB and replay it with a single
`clEnqueueCommandBufferKHR` instead of N per-op enqueues. Wins on
submission overhead and unlocks record-once-replay-many. Strategic
payoff: makes *reusable pipelines* an idiom — a library can ship a
pipeline (`fn gemm(...) -> impl RecordableOp<...>`) the way it ships a
kernel, and consumers compose them via `.and_then` across crates.

**Status.** Design agreed + validated by two spikes. No real claspr
code yet. Next-slice plan below.

**Core design (Option B — extend `DeviceOperation`).**

- `RecordableOp: DeviceOperation` sub-trait — one `.record()` method
  mirroring `.execute()`, threading `cl_sync_point_khr` the way
  execute threads event deps. Base trait unchanged.
- **Recordability is a static bound on the concrete chain type.**
  Combinators (`AndThen`/`Bundle`/`FanOut`) impl `RecordableOp`
  *conditionally on their children*, so it propagates by trait bound
  with no runtime walk. Recordable leaves: kernel / fill / D2D-copy /
  image-copy / barrier. NOT recordable: upload, download, map/unmap,
  host-decided `conditional`, `on_device`, `and_then_host` — they
  simply don't impl `RecordableOp`, so a chain containing one fails to
  compile when you try to record it (crisp `E0277` naming the
  offending leaf even through generic wrappers).
- **`.call()` / `.mutate_call()` live on `DeviceOperation` itself** —
  no `Graph` / `Cached` / `Pipeline` wrapper type (an earlier pass
  built those; dead end). They return a `CallOp` (itself a
  `DeviceOperation`) composable via `.and_then` / `fan_out`, runnable
  via `.sync()` / `.run().await`. Args are **check/update only**: the
  chain's captures are the source of truth; `.call` verifies args
  match (strict), `.mutate_call` accepts compatible-different args and
  swaps via `clUpdateMutableCommandsKHR` (relaxed). The chain is not
  reparameterized. (Option 2 — true slot substitution via placeholders
  or proc-macro — deferred until needed.)
- **`.call()` is composition syntax, not a CB-enqueue verb.** The spec
  forbids nested CB enqueues, so `.call()` can't dictate CB use. The
  runtime materializes contextually: enqueue a cached CB (outside any
  recording), inline the chain's commands into an outer CB recording,
  or walk eagerly (non-cached / non-recordable / no-CB platform).
- **Reuse model: factory `Fn() -> Chain`** (not `Clone` — that would
  force every closure `Fn + Clone` and all captures `Clone`,
  ruling out today's `FnOnce` `and_then`). A reusable pipeline *is* a
  factory. Rebuilding a combinator tree per run is cheap host work.
- **Erasure handoff** (validated in `erasure.rs`): a cached/reusable
  graph can't keep the concrete chain type forever (consumed per run;
  wants to be a struct field / non-generic return). At construction —
  where `Chain: RecordableOp` is still known — capture two erased
  closures from the factory: `execute_fn` (always) and
  `record_fn: Option<…>` (`Some` iff `Chain: RecordableOp`). The
  `Some`/`None` is exactly where the compile-time bound becomes the
  runtime "is recordable?" bit. The bound on the recordable
  constructor rejects `Upload` chains *at the erasure boundary*; an
  `eager_only` constructor is the explicit no-cache degradation path.
  This resolves the apparent tension between `graph_cb` (wanted IR
  erasure for export) and `graph_devop_record` (recordability in the
  concrete type) — erasure is fine because the recorder is captured
  while the type is concrete.

**Two-tier capability model (per spec).**
`cl_khr_command_buffer_mutable_dispatch` gates BOTH the `MUTABLE_KHR`
and `SIMULTANEOUS_USE_KHR` per-CB-creation flags. So:

- **Tier 0** (`cl_khr_command_buffer`): `.call()` cached, immutable,
  one in-flight per graph.
- **Tier 1** (`+ mutable_dispatch`): opt into `.mutate_call()`
  (`MUTABLE_KHR`) and/or concurrent replay (`SIMULTANEOUS_USE_KHR`).

Opt-ins are construction-time (flags set at CB creation) and
*portable* — a graph that opts in still runs correctly on Tier 0 / no-CB
platforms, just falling back to eager walk. Users never call
`device.has_extension(...)`.

| Method | Required opt-in | Tier 1 | Tier 0 | No CB |
|---|---|---|---|---|
| `.call()` (stable, single in-flight) | none | replay cached CB; error on handle mismatch | same | walk DAG |
| `.call()` (concurrent, e.g. fan_out) | `simultaneous` | concurrent replay | fall back | walk DAG |
| `.mutate_call()` | `mutable` | update + replay | fall back | walk DAG |
| `fan_out(.., \|i\| g.mutate_call(i))` | `mutable + simultaneous` | one CB, per-call updates, concurrent | fall back | walk DAG |

**One in-flight per graph — conditional.** OOO queues mean naive
concurrent replays of a cached CB race on its buffers. `SIMULTANEOUS_USE_KHR`
is the spec's opt-in that lifts the invariant (the user asserts per-call
arg updates make destinations independent); without it, a second
concurrent `.call` while one is in flight is an error. This opt-in is
what makes the cached fan_out batch-inference pattern safe.

**`and_then`-reuse is first-class.** `G.and_then(|_| G).and_then(|_| G)`
records into ONE CB with internal sync-point edges (iteration k's tail →
k+1's head). Single enqueue, OOO scheduler still overlaps within each
iteration. Only really expressible with CB-backed graphs. Implies the
factory/erased-recorder must be cheaply shareable (`Arc`).

**Per-leaf-op work (implementation map).** Each recordable leaf grows
one `impl RecordableOp` (~15-20 LOC, mirrors its `execute` but calls
`clCommand*KHR`):

| Existing op | File | record body |
|---|---|---|
| `LaunchOp` (kernel, proc-macro) | `claspr-macros/src/lib.rs:611-621` | `clCommandNDRangeKernelKHR` |
| `FillOp` (`DeviceSlice::fill`) | `claspr/src/buffer.rs:929-1019` | `clCommandFillBufferKHR` |
| `CopyOp` (`DeviceSlice::copy_to`) | `claspr/src/buffer.rs:733-817` | `clCommandCopyBufferKHR` |
| `SvmFillOp` | `claspr/src/mapped.rs` | `clCommandSVMMemFillKHR` |
| `SvmWriteOp` (D2D in SVM) | `claspr/src/mapped.rs` | `clCommandSVMMemcpyKHR` |
| `MigrateOp` | `claspr-async/src/transfer_to_device.rs` | no direct variant — barrier or fall back |
| Image copies | (variants) | `clCommandCopyImage*KHR` |

~6-8 leaves + ~4 combinator conditional impls. Non-recordable ops
(`Upload`/`Download`/`ImageUpload`/`ImageDownload`/`AndThenHost`/`OnDevice`)
get nothing — they just don't impl the sub-trait. **Existing test
impact: zero expected** — base trait unchanged, existing impls
byte-identical, RecordableOp strictly additive.

**Next-slice plan** (each commit green on its own):

1. `claspr`: `Context::has_cl_khr_command_buffer{,_mutable_dispatch}`
   (mirror `svm_capability` at `claspr/src/context.rs:381`). ~30 LOC.
2. `claspr-async`: `RecordableOp` sub-trait + impls on leaf ops +
   conditional impls on `AndThen`/`Bundle*`/`FanOut`. ~200 LOC, no
   public-API change, existing tests stay green.
3. `claspr-async`: `.call()` on DeviceOperation + factory/erased-recorder
   cache + per-arity macro for the call surface + integration tests.
   Default Tier-0 immutable. ~700 LOC.
4. `claspr-async`: `mutable`/`simultaneous` opt-ins + `.mutate_call()` +
   cached-fan_out integration test on pocl 7.2-pre. ~400 LOC.

Requires enabling `opencl3 = { features = ["cl_khr_command_buffer"] }`
on the workspace dep (currently no features). `cl3 0.13.1` already
exposes the full FFI in `cl3::ext::*`; `opencl3 0.12.3` has a safe
`CommandBuffer` wrapper gated behind that feature.

**Open questions.**
- Final verb names (`.call` / `.mutate_call` working draft).
- Per-op profiling inside a CB — the extension only exposes whole-CB
  timestamps, not per-command.
- CI: pick up the cmdbufemu layer (`OPENCL_LAYERS`) over rusticl/NEO so
  cached paths get exercised without native CB; pair with the deferred
  pocl-7.2 ICD work (see `claspr CI deferred` in auto-memory).
- Heuristic auto-CB: the runtime has the inputs (`recordable` bit +
  call count + chain length) to decide when to materialize a CB without
  user opt-in. Could be the default with explicit opt-ins as the escape
  hatch (guaranteed-from-first-call / mutable / simultaneous /
  benchmarking). Deferred.

**Spikes (reference).**
- `spikes/graph_cb/` — `Graph<I, O>` type-system shape: per-arity
  `.call(a,b,c)` macro, `and_then` type composition, library-boundary
  export. NOTE: explored a standalone wrapper type the final design
  dropped; kept for the per-arity-macro + type-erasure techniques,
  which carry over to the `.call`-on-DeviceOperation surface.
- `spikes/graph_devop_record/` — matches the final design:
  `RecordableOp` sub-trait, conditional combinator propagation (5-deep
  AndThen, 3-level Bundle), structural opt-out (`Upload`/`OnDevice`/
  `AndThenHost`), and the erasure handoff (`erasure.rs`). 17 tests,
  `compile_fail_cases.txt` captures the 4 negative-case rustc
  diagnostics. Reviewed + extended 2026-06-15.

**Test/runtime targets.** pocl 7.2-pre (`~/local/pocl`, Tier 1 native:
`cl_khr_command_buffer` 0.9.6 + mutable_dispatch). Distro pocl 6.0
(Tier 0). [bashbaug cmdbufemu layer](https://github.com/bashbaug/SimpleOpenCLSamples/tree/main/layers/10_cmdbufemu)
(Apache-2.0, `OPENCL_LAYERS`): CB 0.9.8 + mutable_dispatch 0.9.5,
stacks over rusticl + NEO legacy (both OpenCL 2.1+, needed for
`clCloneKernel`). Proof-of-concept quality — semantic coverage, not perf.

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

### ✅ RESOLVED 2026-06-23 — Context/default-queue Arc reference CYCLE (cl_context + default queues never released)

Was: a strong Arc cycle `Context(Arc<ContextInner>)` → `ContextInner.queues`
(default `Queue<InOrder>` built at construction) → `QueueInner.ctx: Context`
(strong) closed the moment any Context was built, so strong count never hit 0 and
`ContextInner::drop` / `clReleaseContext` / default-queue release NEVER ran.
cliloader --leak-checking BEFORE: **cl_context Alloc:16 Release:0**;
**cl_command_queue Alloc:31 Release:5** (only user `Queue::new` drops). Pre-existing,
identical on `main` — not the eager cutover.

FIX (landed on `eager-cutover`, "trimmed Option B" applied to BOTH default queue
orderings): `ContextInner` now stores its DEFAULT queues as RAW
`ManuallyDrop<CommandQueue>` handles — NO `Queue` wrapper, NO `ctx` back-edge, so
the cycle is structurally impossible. New `impl Drop for ContextInner` releases
each populated raw default-queue handle BEFORE `cl_context` drops (field order:
`cl_context` declared LAST + explicit release in the Drop body), bumping
`error_state` on release Err (resurrects the previously-dead record-err path).
`default_inorder_queue` now returns an OWNED `Queue<InOrder>` and
`default_outoforder_queue` an `Arc<Queue<OutOfOrder>>`, each an on-demand wrapper
over the cached raw handle with a strong `ctx` (like a user queue) balanced by
`clRetainCommandQueue` on wrap / `clReleaseCommandQueue` on the wrapper's drop —
no double-release, no leak. OOO de-cycle was CLEAN: caching only the raw handle
(not a strong-ctx `Arc<Queue>`) satisfies the stability contract (same
`cl_command_queue` across calls) without reintroducing the cycle; the
context_builder stability/identity tests were updated from Arc/ptr identity to
`.raw().get()` cl_command_queue-handle equality. USER queues
(`Queue::new`/`on_device`) KEEP their strong `ctx` (they must outlive the caller's
Context handle). `Launcher::cl_queue(&Context) -> &CommandQueue` stays infallible
(derefs the raw `ManuallyDrop` slot).

cliloader --leak-checking AFTER (`eager_buffer_ops`, 16 tests, legacy NEO):
**No cl_context leaks detected. No cl_command_queue leaks detected.** (cl_mem /
program / kernel / event / SVM also clean). Two pure-Rust leak-regression tests
pin both directions: `tests/tier1/tests/context_drop.rs` —
`default_queues_do_not_pin_context` (touch both default paths, drop, Context's
`__test_weak` upgrades to dead) and `user_queue_outlives_its_context` (user queue
keeps ctx alive, `finish()` works, ctx released only after the queue drops).

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
| `6d76fe2` (now `main`) | **Cutover PROMOTED to main** (72-commit FF). Also the final fix: gate the terminal marker inside the start-gate in the async terminal (was released-then-enqueued, wiring marker deps in a live window). Full green ×3 ICDs. |
| `c7bad34` | Remove `and_then_with_context`; device-by-index routing structural (`DeviceTarget::{Concrete,Index}`, `on_device_at`/`transfer_to_device_at`). Closes the last un-gated-host-seam gap. |
| `02c6165` | `and_then_host` error path: start-gate + worker-join + event-based chained-cancel. NEO lost-wakeup deadlock fixed. |
| `dee8aa0`/`b29fb03` | Two-event host seam (`fire`/`proceed`); remove defensive double-unmap (was corrupting NEO). |
| `dee8aa0` | Break Context/default-queue Arc reference cycle (ManuallyDrop defaults released before cl_context). +2 leak-regression tests. |
| `2ba935a` | Ops carry ctx → no-arg `.wait()` / `.submit()` + `.wait_on(&L)`/`.submit_on(&L)` for cross-queue. 192 sites migrated. |
