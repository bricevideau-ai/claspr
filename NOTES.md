# claspr — rolling notes

Single rolling doc for active work, deferred items, and unresolved
concerns. Convention is documented in `CLAUDE.md` → "Inter-session
notes". **Append here; don't spawn new planning docs.** Prune as
items resolve.

---

## Active

### Eager struct-graph cutover (branch `eager-cutover`, from main, 2026-06-18)

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
parity backlog. None block the model; each needs a primitive or is a non-shape.**
- **transfer_to_device — NO eager cross-device migrate op.** `on_device` only
  ROUTES an op's enqueue to a device queue; it doesn't MOVE a resident buffer.
  Needs an eager `transfer_to_device(buf/pipe, device) -> EagerOp` primitive.
  (blocks transfer_to_device.rs 2 tests)
- **DynOp — NO eager type-erased op.** `EagerOp` isn't object-safe (assoc
  `Handle` + `into_output(self)`), and there's no boxing ctor. Blocks the
  if/else-arms-of-different-op-types shape (a REAL public-API pattern — `DynOp`
  is in dyn_op.rs, not test-only). Needs a `dyn`-safe erasure boundary (e.g.
  erase to `Box<dyn ErasedEagerOp>` with a monomorphic execute+output_pipe).
  Biggest gap; needs DESIGN. (blocks 7/8 conditional.rs)
- **Host-value passthrough / host reduction mid-graph.** eager `and_then` hands
  a Pipe, not the host VALUE; `and_then_host` is for device `&mut [T]` views,
  not host scalars/Arc<Vec>. So: host reductions over an `arc_split` of a lifted
  host `value(vec)` (sum/len), and `value((buf, step))` scalar-state repack, have
  no seam. Needs an `and_then_host_value`-style seam OR make host wrappers
  Mappable. (blocks all 3 arc_split.rs + 1 ml_pass.rs). Same class as chain.rs
  gap 2. NOTE: arc_split of a DEVICE buffer (the diamond) works fine — this is
  only the HOST-side reduction variant.
- **FanOutExt method form** (`vec.fan_out(op)`) — trivial alias; eager has only
  the free `fan_out(vec, op)`. (blocks 1 fan_out.rs equivalence test)

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
