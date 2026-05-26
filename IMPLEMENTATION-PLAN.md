# claspr Tier 1 + Tier 2 — implementation plan

Companion to `EXECUTION-MODEL.md`. Phased plan for landing the V2
design — simplified Tier 1, new `claspr-async` Tier 2 crate, validated
by a comprehensive runtime test suite that avoids known pocl / rusticl
driver bugs.

## Phase 0 — Correctness pre-work (1–2 days)

Bugs in current code that should be fixed regardless of the rewrite.

| Item | What | Where |
|---|---|---|
| SVM Drop UB | `SharedBuffer::Drop` calls direct `clSVMFree` (UB if commands in flight) — switch to `clEnqueueSVMFree` on source queue | `claspr/src/svm.rs` |
| Audit cl_mem Drop | Confirm `DeviceSlice` / `HostBuffer` Drop go through `clReleaseMemObject` (lazy / refcount — already safe) | `claspr/src/buffer.rs` |

**Test:** add a test that drops a `SharedBuffer` while a kernel is
"using" it (enqueue a kernel, drop without explicit sync, run
`q.finish()`, verify no UB via `cargo test`).

---

## Phase 1 — Simplify Tier 1 (3–4 days)

Drop V1 exploration cruft; settle Tier 1 on `.wait()` / `.await` only.

**Remove from the `claspr` crate:**

- `.track()`, `.detach()` terminals
- `.tracked()` modifier
- `Handle`, `Pending<T>`, `BorrowHandle<'a, T>` types
- Typestate machinery (`Untracked`, `Tracked` ZSTs)
- `LauncherAsync` trait

**Simplify `Launcher` trait:**

```rust
pub trait Launcher {
    fn cl_queue(&self) -> &CommandQueue;
    fn context(&self) -> &Context;
}
```

Drop the `launch()` default method. Op builders consume `&impl Launcher` directly.

**Each Tier 1 op gets these terminals:**

- `.wait() -> Result<()>` (sync, blocks)
- Future impl for `.await` (async)
- `.profiled(|info: ProfilingInfo| { ... }) -> Self` — modifier that
  registers a `clSetEventCallback(CL_COMPLETE, ...)` so the user
  closure receives timestamps when the kernel completes. Output of
  the underlying op is unchanged (profiling is a side-effect).
  Requires the queue to have been built with `.profiling(true)`.

**Cross-queue sync escape hatch** (the one place explicit events live
in Tier 1):

```rust
buf.copy_from(&q_a, data).wait()?;
let h = kernels.produce(&q_a, [N], &buf).submit()?;   // returns Event
kernels.consume(&q_b, [N], &buf).after(&h).wait()?;
```

`.submit()` returns a `claspr::Event` (re-export of `opencl3::Event`)
— used only for `.after(...)` cross-queue waits.

**Why `.profiled()` lands in Phase 1, not Phase 3:** the FFI shim for
`clSetEventCallback` (extern "C" thunk with `catch_unwind`, panic
safety, user-data boxing) is needed for the `.await` Future impl
anyway. Once that wrapper exists, exposing it as `.profiled(|info|
...)` is ~30 lines extra. Phase 3's Tier 2 combinator-level
`.profiled()` then reuses the same wrapper.

Tier 1 users who don't use `.profiled()` can also get timing
manually via `.submit()` + `event.profiling_command_end()? -
event.profiling_command_start()?`. Both paths work; `.profiled()`
is the ergonomic one.

**Update examples** to use simplified API:

- `examples/collatz` — drop any `.track()` / `.tracked()` usage
- `examples/raymarch` — same
- `examples/mandelbrot-kernel`, `sobel-kernel`, `image-pipeline` — same
- `examples/two-device` — uses `.after()` for cross-queue, stays

**Test:** existing example tests (`cargo test -p collatz-example` etc.)
pass on the simplified API.

---

## Phase 2 — Context with per-device default queues (2 days)

Foundation for both tiers.

**Add to `Context`:**

```rust
pub struct ContextBuilder { /* ... */ }

impl Context {
    pub fn builder() -> ContextBuilder { ... }
}

impl ContextBuilder {
    pub fn device(self, dev: &Device) -> Self;
    pub fn devices(self, devs: &[Device]) -> Self;
    pub fn profiling(self, enabled: bool) -> Self;
    pub fn build(self) -> Result<Context>;
}

impl Context {
    /// Per-device default in-order queue (Tier 1 default). Lazy.
    pub fn default_inorder_queue(&self, dev: &Device) -> Result<&Queue<InOrder>>;
    /// Per-device default out-of-order queue (Tier 2 default). Lazy.
    pub fn default_outoforder_queue(&self, dev: &Device) -> Result<&Queue<OutOfOrder>>;
}
```

Both queues honor the `profiling` setting at build time. Lazily created
on first lookup; held in `OnceCell<Queue<O>>` per device.

**Test:** Context with one device, then with two devices; verify queue
identity is consistent across calls; verify profiling enable
round-trips.

---

## Phase 3 — `claspr-async` crate (1–2 weeks)

New workspace crate. Depends on `claspr`. The Tier 2 combinator API.

**Crate structure:**

```
claspr-async/
├── Cargo.toml
├── src/
│   ├── lib.rs              # public exports
│   ├── op.rs               # DeviceOperation trait, AndThen, Value, WithContext
│   ├── bundle.rs           # Bundle2..16 + bundle! macro
│   ├── fan_out.rs          # FanOut combinator
│   ├── arc.rs              # Arced combinator + ArcSplit ext trait
│   ├── and_then_host.rs    # AndThenHost combinator
│   ├── host_view.rs        # HostAccessible trait + per-buffer impls
│   ├── future.rs           # IntoFuture + DeviceFuture (clSetEventCallback)
│   ├── profile.rs          # .profiled(|info| ...) callback registration
│   └── exec_ctx.rs         # ExecutionContext: device + queue lookup
```

**Build sub-steps:**

1. `op.rs` + `exec_ctx.rs`: core trait + sync `.sync()` terminal
2. `bundle.rs` + `fan_out.rs`: variadic structure combinators
3. `arc.rs`: `Arc<T>` wrapping + `ArcSplit::split::<N>()`
4. `future.rs`: `clSetEventCallback`-driven async — **reuses the Tier 1 callback wrapper from Phase 1** (FFI thunk + `catch_unwind`); ties it to the chain's Future poll machinery (atomic flag + `AtomicWaker`)
5. `and_then_host.rs`: trivial (just `and_then` returning a value)
6. `host_view.rs`: `HostAccessible<T>` trait + impls for `DeviceSlice<T>` (d2h + h2d), `HostBuffer<T>` (no-op), `SharedBuffer<T>` (map + unmap)
7. `profile.rs`: combinator-shape wrapper over the Tier 1 `.profiled()` for use inside lazy chains (where the underlying Event isn't user-visible); shares the callback wrapper

**Critical CL integration points:**

- `clSetEventCallback(event, CL_COMPLETE, thunk, user_data)` — Future poll machinery + profiling callbacks
- `clEnqueueMarkerWithWaitList(queue, n_wait, wait_list, &marker)` — `FanOut`'s implicit marker join
- `clEnqueueSVMMap` / `clEnqueueSVMUnmap` — `HostAccessible` for `SharedBuffer`
- `clEnqueueReadBuffer` / `clEnqueueWriteBuffer` (CL_FALSE for async) — `HostAccessible` for `DeviceSlice`

---

## Phase 4 — Proc-macro Tier 2 emission (3–4 days)

Extend `claspr-macros` to emit both Tier 1 launch and Tier 2 builder
per kernel.

For each `#[claspr::kernel] pub fn foo(...)`, generate:

```rust
impl Kernels {
    /// Tier 1: sync via &Queue, returns Event for .after() chaining
    pub fn foo(&self, launcher: &impl Launcher, grid: ..., args: ...)
        -> impl OpBuilder<Output = ()>;

    /// Tier 2: returns lazy DeviceOperation
    pub fn foo_op(&self, launcher: &impl Launcher, grid: ..., args: ...)
        -> impl DeviceOperation<Output = ()>;
}
```

`foo_op` is the same shape but emits a `DeviceOperation` instead of
an `OpBuilder`. Internal sharing via a private helper.

---

## Phase 5 — Testing (interleaved throughout)

### Test kernel library

A single fixture at `tests/kernels/src/lib.rs` providing simple,
portable test kernels.

**Design constraints to avoid known driver bugs:**

- **Avoid `Complex32` or struct-of-floats in image kernels** — pocl 7.2-pre on aarch64 hangs in `clBuildProgram` when SPV combines `OpCapability ImageBasic` with `OpTypeStruct %float %float` (see `reference_pocl_image_complex_hang` in memory).
- **Avoid combining `Image` capability with custom struct types** at all (same bug class).
- **Target opencl1.2** for maximum compatibility across pocl, rusticl, Intel runtime.
- **Use only `u32` / `f32` slices** for data; no vector types in test kernels.
- **One kernel per pipeline stage** — no megakernels.

**Kernels to provide:**

```rust
#[claspr::device]
pub mod test_kernels {
    #[claspr::kernel]
    pub fn fill_u32(
        #[spirv(global_invocation_id)] id: glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        value: u32,
    ) { data[id.x] = value; }

    #[claspr::kernel]
    pub fn add_u32(
        #[spirv(global_invocation_id)] id: glam::USizeVec3,
        #[spirv(cross_workgroup)] a: &[u32],
        #[spirv(cross_workgroup)] b: &[u32],
        #[spirv(cross_workgroup)] out: &mut [u32],
    ) { out[id.x] = a[id.x] + b[id.x]; }

    #[claspr::kernel]
    pub fn scale_u32(
        #[spirv(global_invocation_id)] id: glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
        factor: u32,
    ) { data[id.x] *= factor; }

    #[claspr::kernel]
    pub fn copy_u32(
        #[spirv(global_invocation_id)] id: glam::USizeVec3,
        #[spirv(cross_workgroup)] src: &[u32],
        #[spirv(cross_workgroup)] dst: &mut [u32],
    ) { dst[id.x] = src[id.x]; }
}
```

These are the only kernels needed; runtime tests compose them into
pipelines.

### Test organization

```
tests/
├── kernels/                # rust-gpu test kernel crate (build.rs uses claspr-build)
│   ├── Cargo.toml
│   ├── build.rs
│   └── src/lib.rs
├── tier1/
│   ├── basic.rs            # h2d → kernel → d2h
│   ├── cross_queue.rs      # .after() between queues
│   ├── multi_device.rs     # gated on >= 2 devices
│   ├── svm.rs              # SharedBuffer + map/unmap correctness
│   ├── drop_safety.rs      # "drop while in flight" tests
│   └── profile.rs          # .profiled(|info| ...) — callback fires, timestamps valid
└── tier2/
    ├── chain.rs            # AndThen + Value
    ├── bundle.rs           # Bundle2/3/4 + bundle! macro
    ├── fan_out.rs          # N-ary parallel
    ├── arc_split.rs        # Arc-shared fan-out
    ├── and_then_host.rs    # in-queue host work
    ├── host_view.rs        # HostAccessible per buffer type
    ├── profile.rs          # callback fires; timestamps non-zero
    ├── error.rs            # error short-circuits chain
    └── round_trip.rs       # value() lift + buffer reuse
```

Each file has 3–8 focused tests. Total ~50 tests.

### Test environment

- All tests skip silently (return `Ok(())` with `eprintln!("SKIP: no OpenCL device")`) if `Context::any()` fails. Matches existing claspr pattern.
- Run with `OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors cargo test` (per `CLAUDE.md`).
- Multi-device tests skip if only one device. Profiling tests skip if context not built with `profiling(true)`.
- No tests depend on Image + Complex combinations or other known-broken patterns.

### What the tests actually validate

- **Buffer correctness** — data values match expected after each op
- **Drop semantics** — no UB when dropping in-flight buffers (run under `compute-sanitizer` or pocl's debug build if available)
- **Cross-queue ordering** — `.after()` correctly waits
- **OOO scheduling** — independent ops in a bundle actually overlap (where measurable via profiling)
- **Marker fan-in** — `fan_out` produces correct results across all branches
- **Future poll behavior** — `.await` resolves correctly; callbacks fire exactly once
- **Profiling info validity** — timestamps are non-zero and monotonic after kernel completes
- **HostAccessible round-trip** — read-modify-write through host view preserves device state
- **Error propagation** — chain short-circuits cleanly on first error

---

## Phase 6 — Examples + migration (2–3 days)

- Port `examples/collatz` to simplified Tier 1 API
- Port `examples/raymarch`, image examples to simplified Tier 1
- New example: `examples/async-pipeline` — a small ML-style forward pass using `claspr-async`
- New example: `examples/batch-inference` — `fan_out` over N batches with shared `Arc<weights>`
- Update `examples/two-device` to use new `Context::builder().devices(&[a, b])`

---

## Sequencing + critical path

```
Phase 0 (SVM fix) ─┐
                   ├─► Phase 2 (Context builder) ─► Phase 3 (claspr-async) ─► Phase 4 (proc-macro Tier 2)
Phase 1 (Tier 1) ──┘                                       │
                                                            ▼
                                            Phase 5 (testing — runs alongside Phases 3, 4)
                                                            │
                                                            ▼
                                                  Phase 6 (examples + migration)
```

**Total estimate: 4–6 weeks** depending on how cleanly the CL callback
machinery lands.

**Branch strategy:** rewrite on a new branch `tier1-tier2-rewrite` cut
from `runtime-redesign`. Land in one big PR (per the design doc's
"atomic rewrite" decision). Pre-rewrite branch stays as fallback.

---

## What to validate empirically before starting Phase 3

Two small experiments to de-risk:

1. **`clSetEventCallback` overhead on pocl.** Submit 1000 kernels with callbacks, measure dispatch latency. If high, design Future poll machinery to amortize (e.g., per-chain single callback instead of per-op).
2. **`clEnqueueMarkerWithWaitList` semantics on pocl + rusticl.** Submit N independent kernels + marker, measure whether the marker actually waits for all (some drivers have flaky implementations).

Both are <100 LOC each. Worth doing before committing to Phase 3's design.

---

## Open questions to resolve during implementation

These can be punted to where they actually bite:

1. **Tokio dependency for async.** Stay runtime-agnostic via our own waker dispatch, or take Tokio? Lean: runtime-agnostic.
2. **`bundle!` upper arity bound.** 16 (matches `tokio::join!`) seems right; ship more if a user needs them.
3. **`.split::<N>()` naming on Arc.** Deferred per earlier note; pick a name when first user touches it.
4. **Per-stage error context.** `.context("stage_name")` combinator vs leave as a future addition. Lean: future addition; not blocking.
5. **`HostAccessible` read-only variant.** `acquire_host_view_ro` that skips writeback for `DeviceSlice`. Useful optimization; ship after measuring.

---

## References

- `EXECUTION-MODEL.md` — the design this implements
- `spikes/combinator/` — type-structure validation
- `CLAUDE.md` — repo conventions
