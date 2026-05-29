# claspr — review of current state

**Update 2026-05-29:** audit pass against the 2026-05-28 review below. Status of every "Real concerns" item:

| # | Item | Status |
|---|---|---|
| 1 | Tier 2 error fidelity across async chain boundaries | **Done** — `tests/tier2/tests/error_fidelity.rs` added; error-erasure comment removed from prior tests. Rust error variants now survive the user-event boundary via a per-chain `Mutex<Option<Error>>` slot; see `and_then_host.rs` module docs § "Error model". |
| 2 | `and_then_host` value-returning variant | **Won't Fix** — the async submit-vs-completion gap forbids it; see `and_then_host.rs` module docs § "Why no value-returning closure?". `Arc<Mutex<Option<T>>>` is the idiomatic shape, not a workaround. Test comments updated to reflect this. |
| 3 | `MappedSlice` falls out of typed kernel wrappers | **Done** — `tier1/svm.rs` uses typed `kernels.fill_u32(&ctx, [N], buf, ...)` directly; the proc-macro's `KernelSliceArg<T>` widening accepts `DeviceSlice` / `MappedSlice` / `USMSlice` interchangeably. |
| 4 | fp64 / vector test kernels absent | **Done** — `tier1/fp64.rs` + `tier2/fp64_chain.rs` exercise f64 paths through the runtime + chain. |
| 5 | No stress tests | **Done** — `tier1/stress_svm.rs` validates the Vec-accumulation pattern at scale. |
| 6 | README significantly stale | **Done** — current text reflects two-tier API, `Context::any` / `Context::builder`, `MappedSlice` / `USMSlice`, typed `Error` enum, CI on rusticl-on-llvmpipe. |
| 7a | `cross_device` light coverage | Marginal — 1 → 2 tests. Still thin for the load-bearing case. |
| 7b | `arc_split` low assertion density | Marginal — 2 → 3 tests. |
| 7c | `conditional.rs` non-taken branch | **Done** — explicit `panic!("non-taken branch fired …")` in the skipped arm at line 172. |
| 7d | `MappedSlice` not in `drop_safety.rs` | Acknowledged — coverage exists in `tier1/svm.rs` cross-queue tests; cross-reference added to `drop_safety.rs`. |

**Beyond the REVIEW** (work landed since 2026-05-28):

- **USMSlice tier** — new fine-grain-system SVM primitive (`usm_slice` / `usm_slice_alloc` / `usm_slice!` macro + 7 tests).
- **HostBuffer removed** — was UB per spec on rusticl (`CL_MEM_ALLOC_HOST_PTR` + persistent map). USMSlice is the spec-correct replacement.
- **SharedBuffer → MappedSlice rename** — naming family now lines up as `DeviceSlice` / `MappedSlice` / `USMSlice` with the suffix describing host-access mechanism.
- **Zero-init by default** for every `alloc`; `pub unsafe fn alloc_uninit` as opt-in escape hatch for internal write-everything-first paths. Closes the "kernel reads uninit memory at the SPIR-V level" hole.
- **`Buffer<T>` polymorphic test** restored to include USMSlice arm.

Items 7a / 7b remain as incremental hardening opportunities.

---

Written from the linux box on 2026-05-28 after a deep read of the public surface (`claspr/src/{buffer,context,device,queue,op,svm,future,image,launch}.rs`, `claspr-async/src/*.rs`), every `tests/tier1/tests/*.rs` and `tests/tier2/tests/*.rs`, and the new examples (`async-pipeline`, `batch-inference`, `two-device`, plus the unchanged `collatz`/`raymarch`/`mandelbrot-kernel`/`sobel-kernel`/`image-pipeline`).

The previous version of this doc was a pre-merge review of `runtime-redesign`. Most of its items have since landed:

| Prior item | Status |
|---|---|
| (1) Sticky-error counter wiring | Done — covered across `DeviceSlice` / `Queue` / `MappedSlice` Drop; `tier1/drop_safety.rs` + `tier1/svm.rs` assert `ctx.error_count() == 0` after forced drops |
| (2) Kernel-launching multi-device test | Done — `tier1/multi_device.rs::launch_runs_on_each_device_via_proc_macro_launcher`, with the sub-device fallback so it actually fires on common single-device boxes |
| (3) Verify `&Queue<OutOfOrder>` through the proc-macro | Exercised — `tier1/svm.rs` uses `Queue::<OutOfOrder>::on_device` against macro-typed `kernels.fill_u32` / `scale_u32` flows |
| (4) `launch_with_deps` losing macro typing | Resolved via the two-tier design — Tier 1 has `.submit()` returning `Event`, Tier 2 has macro-emitted `_op` variants composable in chains |
| (6) `Buffer<T>` rustdoc clarity | Done — `tier1/buffer_trait.rs` opens with "Per the trait's rustdoc this is **explicitly not** a tier-polymorphism point" |
| (7) `Error::Other` migration | Done on `tier1-tier2-rewrite`: commit "drop unused variants (Other, KernelArg, InvalidWorkSize) + their From shims" |
| (5) Image format dispatch in the proc-macro | Open — proc-macro still emits `&Image2DRgba8`; runtime side has rich `Image2D<A, F>` |

That's a substantial set of paper-cuts closed since the last review. What follows is what I see now.

## What's genuinely good

**Test-suite discipline is the standout.** Every file leads with a `//!` block explaining what edge it covers, what specific risk it guards against, and what it does *not* try to prove. `tier1/svm.rs` and `tier1/drop_safety.rs` are textbook examples — they articulate the failure mode, then assert it explicitly through the sticky-error counter rather than waiting for a hang. That kind of "the test knows what it's testing" hygiene is rare and pays back forever.

**The test-kernel library** (`tests/kernels/`) is small (4 kernels, all `u32`), documented with the reasoning behind each portability constraint (read-then-write to dodge a rust-gpu codegen quirk, no Image+struct combos to dodge pocl, OpenCL 1.2 baseline). Exactly what a runtime test suite needs — portability over expressiveness. The runtime tests get to be about runtime semantics rather than chasing kernel-codegen edge cases.

**Tier 1 / Tier 2 separation is clean and both are exercised end-to-end.** Tier 1 (`.wait() / .submit() / .await`): 7 test files. Tier 2 (lazy combinators in `claspr-async`): 13 test files. Both surfaces validated on a real device, not against mocks.

**Multi-device coverage uses the right kind of clever** — `tier1/multi_device.rs` has a three-stage fallback (real ≥2 devices → sub-device partition → skip). The sub-device path means the test actually fires on the common single-CPU-device dev box, instead of always green-because-skipped.

**Drop safety + cross-queue ordering for `MappedSlice`** is the load-bearing safety story, and `tier1/svm.rs` covers 7 distinct facets: basic round-trip, cross-queue `last_use`, explicit `register_use`, OOO auto-registration, empty wait-list arm, read-only map, multi-kernel pipeline. The auto-registration test validates that `KernelArg::register_completion` actually does its job under concurrent OOO launches — the bit that prevents UB in real usage.

**Examples function as integration tests.** `async-pipeline` and `batch-inference` validate the device computation against an identical host implementation; both have `#[test]` blocks at the bottom so `cargo test` exercises them. `raymarch` does pixel-level host validation too. Examples that are also tests — the right pattern.

**`tier2/run_await.rs`** wires `block_on(chain.run(&ctx))` through `clEnqueueMarkerWithWaitList` + `clSetEventCallback`-driven `ChainFuture`. The end-to-end async terminal works, including error propagation (`LengthMismatch` survives the await). That's the harder of the two terminal paths and it lands cleanly.

## Real concerns

### 1. Error type erasure across async chain boundaries

Documented in the *test comments themselves* as a known limitation, not in any design doc. From `tier2/error.rs`:

> Note: under the new async `and_then_host`, the original closure error type doesn't survive the user-event signal — what propagates is a negative `cl_event` status that surfaces as `Error::OpenCl(ClError(negative))`. These tests now assert "chain errored" rather than matching the specific Rust error variant.

Same comment shows up in `tier2/host_and_profile.rs`. This is a real ergonomic regression in the Tier 2 path vs Tier 1: a user who fails with `Error::SvmNotAvailable` inside a host closure sees `Error::OpenCl(-N)` at the terminal. Pattern-matching error variants for recovery is broken across the boundary.

The tests accommodate it; users will work around it; but it's a sharp edge that should be either fixed (carry the original `Error` through alongside the user-event signal — a `Mutex<Option<Error>>` on the chain that the host-closure worker writes and the terminal reads after the user event signals) or explicitly called out in the public docs with a recommended idiom for variant-sensitive recovery.

### 2. `and_then_host` can only mutate in place, not return values

Visible across `tier2/{ml_pass,host_and_profile,conditional}.rs` — every reduction-style test uses `Arc<Mutex<u32>>` capture to get a value back out. The `forward_pass_carries_scalar_state_via_value_tuple_repack` test even has the comment "Tuple-repack at every stage is the documented cost."

That's a fair design tradeoff (closure runs on a worker thread; returning involves lifetime work), but the `Arc<Mutex<_>>` workaround is ugly enough that it'll show up in user code constantly. Worth either:
- a builder method like `.and_then_host_returning::<T>(f)` for the value-carrying variant, or
- a typed reduction primitive (`reduce_host::<T>(...)`) that does the Arc/Mutex dance once internally.

The current shape pushes ergonomic cost onto every user who wants a reduction (which is most of them).

### 3. `MappedSlice` falls out of the typed kernel wrappers

Visible in both `tier1/svm.rs` and `tier2/svm_chain.rs`: every SVM kernel launch drops to `LaunchOp::new(&ec, &kernel, [N].into_launch_spec(), (&buf, …))` because the proc-macro-emitted `kernels.foo(...)` only accepts `&DeviceSlice<T>`. From `tier1/svm.rs`:

> We can't pass `&MappedSlice` to the typed `kernels.fill_u32` (which is typed against `DeviceSlice<u32>`), so we drop to the lower-level path: build the LaunchOp manually via a kernel handle and the `KernelArgs` tuple.

The fix is broadening the trait bound on the macro signature (`impl KernelArg<T>` instead of `&DeviceSlice<T>`), or emitting a separate `kernels.foo_svm(...)`. But it's not done. Users who want SVM lose all the type safety the rest of claspr provides — which is the wrong incentive direction, since SVM is the easiest path on capable devices and making it the least ergonomic punishes the use case.

### 4. fp64 / vector kernels are intentionally absent from the test suite

The kernel library is u32-only by design (driver bug avoidance), which is correct for portability. But that means **no test ever exercises the fp64 / `cl::*` vector / `opencl_std::*` math paths through claspr's runtime.** rust-gpu's difftest suite covers codegen, but claspr-side runtime-and-arg-marshalling for fp64 / vectors is untested.

Either:
- Add a separate optional kernel set gated on `Float64` capability discovery, exercising f32/f64/vector ops (one or two kernels is enough)
- Document explicitly that fp64/vec is *unvalidated through claspr's runtime tests* and tested only via the upstream rust-gpu difftest suite + the raymarch sample

I'd lean toward the first — one f64 + one `cl::Float3` kernel in a `tests/kernels/extras.rs` (compile-gated; capability-gated at run) would close a real coverage gap without bloating the portable suite.

### 5. No stress tests

The biggest test runs 8 in-flight launches. The Vec-accumulation pattern in `MappedSlice` (events drained at Drop) could grow unboundedly if a user holds a `MappedSlice` while submitting thousands of ops. A test that submits 1000+ ops onto an OOO queue holding a single SVM and asserts memory stays bounded (and `error_count == 0`) would be cheap insurance against a regression that's otherwise invisible.

### 6. README is significantly stale

Describes the pre-runtime-redesign API: `Context::new`, `ctx.upload`, `ctx.download`, `Result = Box<dyn Error>` listed in Limitations. Actual code has `Context::any() / Context::builder().build()`, `DeviceSlice::upload(&launcher, ...)`, typed `Error` enum, two tiers, `LaunchOp` builder.

A reader landing on github gets a wrong first impression of where the project is. This is a 1-hour fix and disproportionately important — the README is the public face.

### 7. A few thin spots in coverage

- **`tier2/cross_device.rs` is a single test** for the cross-device DeviceSlice flow. Given how load-bearing multi-device + memory transfer is for HPC, this feels light compared to multi-device kernel-launch coverage (3 tests in `tier1/multi_device.rs`). Cross-context buffer flow (download + re-upload) stays user-managed per the design — but it's where most users will hit problems first.
- **`tier2/arc_split.rs`** (2 tests, 0 `assert!` macros) covers shape but not error or N=1 edge cases.
- **`tier2/conditional.rs`** has 8 tests for `DynOp` but mostly verifies the wrapper compiles. Doesn't test that the non-taken branch is actually skipped at runtime — you could feed a `cond=false` branch that panics in its closure and prove it never fires.
- **No `drop_safety.rs` coverage for `MappedSlice`** directly — SVM has its own file but the more general "drop a MappedSlice while a *non-launch* operation is in flight" pattern isn't covered there. (The cross-queue last-use case in `tier1/svm.rs` covers part of this.)

None of these block anything; they're incremental hardening.

## Suggested priority

If I had to rank these, this is the order I'd attack them:

1. **README rewrite** — public-facing, cheap, high signal.
2. **`MappedSlice` through the typed launchers** (#3) — paper-cut visible in every SVM-touching test; the workaround code is exactly what claspr is supposed to abstract away.
3. **Tier 2 error fidelity** (#1) — the most likely to bite a real user. A `Mutex<Option<Error>>` on the chain that the host-closure worker populates is a few-dozen-line change that restores variant-matching.
4. **fp64 / vector test kernels** (#4) — closes the only meaningful coverage hole.
5. **`and_then_host` value-returning variant** (#2) — quality-of-life; could land alongside the error-fidelity fix since both touch the host-closure plumbing.
6. **Stress test** (#5) and **thin-spot fills** (#7) — incremental.

## Overall

The project is in a **healthier shape than I expected** given how recently the runtime-redesign + tier1/tier2 rewrite landed. The test discipline carries the cost of the rewrite gracefully — there's a coherent shape across 20 test files, the hardest correctness story (SVM Drop + cross-queue) is exercised carefully, and the two-tier architecture works in practice (async-pipeline + batch-inference are real-feeling examples, not stubs).

The remaining concerns are **rough edges around the seams**, not architectural problems. The biggest single thing a new contributor would notice is the stale README; the biggest single thing an existing user would notice is `MappedSlice` falling out of the typed launcher system. Neither is hard to fix.

— Reviewed 2026-05-28 on the linux box. Public surface, all 20 tier test files, both new examples, CLAUDE.md, IMPLEMENTATION-PLAN.md.
