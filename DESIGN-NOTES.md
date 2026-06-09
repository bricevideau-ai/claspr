# claspr — design notes & deferred work

Planning doc for upcoming work that has been scoped but not yet
implemented. Each section is self-contained enough that a future
session can pick a single item and act. Items are ordered roughly
by priority — earlier items are "act now" with small blast radius;
later items are deferred until concrete pressure surfaces.

This doc was originally the scope-launcher design note alone;
expanded 2026-06-08 with three items surfaced by the
`/home/claudecode/projects/CLASPR-REVIEW.md` follow-up review.

---

## 1. Seal `KernelOp` (priority: act now)

**Status:** ready, ~5-line change, no migration cost since the only
external impl today is the `claspr-async` blanket impl that lives
in the same workspace.

### Why

`KernelOp` is currently `pub trait KernelOp: Send + Sized` at
`claspr/src/kernel_op.rs:46`. It's re-exported from the crate root,
making it look like a public extension point. But:

- The trait is the **integration boundary** between Tier 1 and
  Tier 2: `claspr-async/src/op.rs` has
  `impl<O: KernelOp + 'static> DeviceOperation for O`. Every
  `KernelOp` impl automatically becomes a Tier 2 `DeviceOperation`.
- A wrong `KernelOp` impl breaks subtle invariants:
  - Forgetting to register the completion event on every kernel-
    arg buffer's `last_use` list silently corrupts Drop ordering
    on `Arc<DeviceSlice>` / `MappedSlice` (the `clEnqueueSVMFree`
    wait-list would miss in-flight uses).
  - Misthreading `deps` skips event-graph dependencies, producing
    races that pass tests on in-order queues and corrupt on
    out-of-order.
- The only legitimate producer of `KernelOp` impls is the
  `#[claspr::kernel]` proc-macro (`claspr-macros/src/lib.rs:758`),
  which gets these details right by construction.

Sealing now while there's one impl is forward-compatible: we can
always unseal later if a real third-party extension case surfaces.
Unsealing later is breaking.

### What

Apply the standard sealed-trait pattern:

```rust
// claspr/src/kernel_op.rs

mod sealed {
    pub trait Sealed {}
}

pub trait KernelOp: sealed::Sealed + Send + Sized {
    // ... unchanged contract ...
}

// In claspr-macros/src/lib.rs, alongside the existing
// `impl ::claspr::KernelOp for Op` emission, also emit:
//
//   impl ::claspr::kernel_op::sealed::Sealed for Op { }
//
// Or — cleaner — make Sealed pub(crate) and add the impl on the
// generated Op type via a blanket-friendly pattern. The cleanest
// shape is to just have the macro emit both impls; it's two extra
// lines per kernel.
```

### Open question

Do we want a documented `unsafe trait KernelOp` for "advanced
users" instead of sealing? My take: no — until a concrete external
use case appears, sealed is the conservative call. If a real
extension case appears later, document the invariants then and
unseal in the same release.

### How to apply

1. Add `sealed::Sealed` private module in `claspr/src/kernel_op.rs`.
2. Change trait bound to `sealed::Sealed + Send + Sized`.
3. Update `claspr-macros/src/lib.rs:758`-region to emit the matching
   `Sealed` impl alongside the `KernelOp` impl.
4. Verify all 235 tests still pass. No user-facing API change.

---

## 2. README + docs for the three modes (priority: soon)

**Status:** the recent `claspr::kernels!` decoupling work created
three meaningfully different ways to use claspr; the public docs
only describe single-source. Newcomers using clang-produced SPIR-V
or runtime-loaded blobs have nothing to read.

### Why

After commits `cc2437d` and the earlier explicit-compile refactors,
claspr supports three first-class modes:

| Mode | SPIR-V production | Host API declaration | Runtime binding |
|---|---|---|---|
| **Single-source** | `claspr-build` extracts `#[claspr::device]` mod, runs spirv-builder | `#[claspr::kernel]` proc-macro implicit | `gpu::kernels(&ctx)?` |
| **Explicit compile** | `claspr-build` from a separate kernel crate, emits `SPV_BYTES` + `ENTRY_POINTS` | `claspr::kernels!` declaration | `Kernels::load_from(&ctx, SPV_BYTES)?` |
| **Runtime-loaded SPIR-V** | External (clang, downloaded blob, code-generator) | `claspr::kernels!` declaration | `Kernels::bind(program)?` |

Mode 1 is what the current `README.md` shows; Modes 2 and 3 are
new and not pitched anywhere user-facing. A user with clang-compiled
SPIR-V has to grep `tests/explicit-compile/` to figure out the
shape.

### What

Two-part doc work:

**Part A — README restructure** (small, ~30-line addition):

Keep single-source as the headline (it's the primary value
proposition — kernel + host in one file is what's novel). Add a
new "Other modes" section after "What it looks like" with one
short subsection per mode:

- **Pre-compiled SPIR-V from a separate kernel crate.** Shows the
  `claspr-build` explicit-compile flow + `claspr::kernels!` block
  + `Kernels::load_from`. ~10 lines of code, one paragraph.
- **External SPIR-V (clang, downloaded blobs).** Shows
  `claspr::kernels!` + `Kernels::bind(program)`. ~10 lines. Mention
  this is how to use kernels produced by toolchains outside the
  Rust ecosystem.

Each subsection links to the canonical reference test
(`tests/explicit-compile/`).

**Part B — Optional GUIDE.md** (medium effort, can be deferred):

If the README additions feel cramped, spin out a `GUIDE.md` with:
- When to pick each mode
- Migration paths between modes (e.g., starting with single-source
  and later splitting into a kernel library)
- Limits of each mode (single-source can't accept external SPIR-V;
  external SPIR-V loses build-time kernel arg-info validation; etc.)

Part A is the priority; Part B is bonus.

### Open question

Should there be a fourth "mode" entry for **library-crate
composition** (the mandelbrot-kernel / sobel-kernel pattern)? My
take: no — that's an *organizational* pattern that works equally
well with any of the three modes. Mention it in the README's
existing "Workspace layout" or "Composing kernels" section, not
as a separate mode.

### How to apply

1. Add an "Other modes" section to `README.md` after the existing
   "What it looks like".
2. Each subsection: 1 paragraph rationale + 1 code block (~10
   lines) + link to canonical reference test.
3. Optionally write `GUIDE.md` and link from README.
4. No code changes.

---

## 3. Inherit generated kernel deps from host workspace (priority: medium)

**Status:** known issue, already in `CLAUDE.md`'s deferred backlog.
The follow-up review re-flagged it. Worth promoting.

### Why

`claspr-build/src/lib.rs:897` (`write_generated_cargo_toml`)
hardcodes the floating-branch dep block:

```toml
spirv-std = { git = "https://github.com/bricevideau-ai/rust-gpu.git", branch = "opencl-kernel-support" }
glam = { version = ">=0.30.8, <0.33", default-features = false, features = ["libm"] }
num-complex = { version = "0.4", default-features = false }
```

The host workspace pins these to specific commits via `Cargo.lock`,
but the generated kernel crate references the *branch tip*. The
existing `seed_lockfile_from_host` (`claspr-build/src/lib.rs:883`)
copies the host's lockfile into the generated crate as a mitigation
— but it's a workaround. Fresh consumers without a host workspace
lockfile, or branch movement upstream, can drift unpredictably.

### What

Two plausible approaches:

**Approach A — inherit from host workspace** (correct but more work):

At build-script time, walk up from `OUT_DIR` to find the host's
`Cargo.toml` / `Cargo.lock`. Extract the `spirv-std` and `glam`
entries from the lockfile. Write them into the generated TOML as
exact `rev = "..."` pins (not branch refs).

Pros:
- Reproducibility derives from the same source of truth as the
  rest of the workspace.
- New consumers Just Work — they pull the same pin the host pulled.
- No new builder API surface.

Cons:
- `claspr-build` needs to parse `Cargo.lock` (small dep, but
  surface area).
- Walking up to find the workspace root is heuristic; needs
  fallback when it can't be found.

**Approach B — builder configuration** (explicit, simpler):

Add `CompileBuilder::spirv_std_dep(spec: &str)`,
`CompileBuilder::glam_dep(spec: &str)`. Users opt into a non-default
spec; the default stays as today.

Pros:
- Explicit. No magic dep resolution.
- Easy to implement (string substitution).

Cons:
- Pushes the work to every consumer.
- Most consumers won't bother → drift problem persists.

**Recommendation: do A with B as escape hatch.** Walk for the host
lockfile; if found, use the pinned `rev`. If not found (standalone
test crates, weird workspace shapes), fall back to today's
hardcoded branch shape. Builder methods from B can land on top of
this for the rare case where a user wants to override.

### Open questions

1. How aggressive should the walk be? `OUT_DIR` → parent dirs until
   a `Cargo.lock` is found? With a depth cap?
2. If lockfile-extraction fails, do we warn (via `cargo:warning=`)
   or silently fall back?

My answers: walk up to 8 levels; on parse failure warn-and-fall-
back rather than hard-error (we don't want to break builds over a
reproducibility improvement).

### How to apply

1. In `claspr-build/src/lib.rs`, add `find_host_lockfile(out_dir)`
   helper.
2. Add `extract_kernel_deps(lockfile_path)` returning a small struct
   with the spirv-std + glam + num-complex pins.
3. Rewrite `write_generated_cargo_toml` to use the extracted pins
   if available, hardcoded fallback otherwise.
4. Update the inline comment at line 898 to reflect the new policy.
5. `seed_lockfile_from_host` becomes redundant for the common path
   but keep as belt-and-suspenders.

### Test plan

- Existing examples build green (they go through the workspace
  lockfile path).
- `tests/explicit-compile` builds green (separate kernel crate;
  exercises the lockfile-walk happy path).
- A synthetic "no-lockfile" test crate (probably `spikes/`) builds
  green via the fallback path.

---

## 4. Tier 1 no-arg `.wait()` / `.submit()` (priority: SHIPPED)

**Status:** done — different mechanism than the original sketch, no
`scope` introduced.

### What landed

Every Tier 1 op carries a reference to its owning [`Context`]
(buffer ops gained an `&'a Context` field; the kernel `Op` gained
an owned `Context` field, cloned in cheaply since the type is
`Arc`-internal). The op's two existing terminals were renamed
`.wait` → `.wait_on(&L)` / `.submit` → `.submit_on(&L)`, and two
new no-arg variants `.wait()` / `.submit()` were added that submit
on the carried context's default in-order queue.

```rust
// Before:
let mut buf = DeviceSlice::<u32>::alloc_zero(&ctx, N)?;
buf.write(&data).wait(&ctx)?;
let buf = kernels.fill_u32([N], buf, 42).wait(&ctx)?;
buf.read(&mut out).wait(&ctx)?;

// After:
let mut buf = DeviceSlice::<u32>::alloc_zero(&ctx, N)?;
buf.write(&data).wait()?;                    // common case — uses carried ctx
let buf = kernels.fill_u32([N], buf, 42).wait()?;
buf.read(&mut out).wait()?;

// Cross-queue case (rare) uses the renamed explicit form:
buf.write(&data).wait_on(&other_queue)?;
```

### Why this shape instead of `ctx.scope(|s| { ... })`

The original DESIGN-NOTES sketch proposed a SYCL-style scoped
launcher resolving the launcher via TLS, closure-captured handles,
or a renamed binding. All three had real costs (TLS silently
miscompiles across `thread::spawn`; closure-captured-handle needed
either pervasive lifetime params on buffer types or a per-op
wrapping trait that didn't actually save characters vs. `.wait(s)`;
the renamed-binding shape was cosmetic only).

Carrying the context inside the op is structurally simpler and
gives the user the same one-character call shape (`.wait()`) with
no implicit globals and no new wrapper type. The cost is one extra
field per op struct and a one-time mechanical rename of 192 call
sites from `.wait(&ctx)` → `.wait()` (most got SHORTER) and the
rare 7 cross-queue sites from `.wait(&queue)` → `.wait_on(&queue)`.

### Adjacent capability gaps — also SHIPPED

Two real Tier 1 surface items the late-bind work surfaced —
**resolved together** rather than separately, with a unified shape:

1. **`DeviceSlice::map` Tier 1** ✅. Mirrors the existing
   `MappedSlice::map` / `map_mut` shape: lazy builder, `.wait()`
   blocking terminal returning a `DeviceMappedRead/WriteGuard`
   (Deref / DerefMut, unmap on Drop), `.submit()` non-blocking
   terminal returning a `DeviceMapRead/WritePending` (carries the
   map event for chain ordering; `.wait()` consumes into the guard).

2. **Non-blocking `MappedSlice::map`** ✅. Same `.submit()`
   terminal added to the existing builder; returns
   `MappedRead/WritePending` with the same pending-to-guard shape.

**The "two shapes" framing was wrong.** The original DESIGN-NOTES
text claimed the SVM pending could just Deref immediately with
caller-responsible-for-wait. That's an unsafe-by-default API hiding
behind a comment — `Deref<Target=[T]>` implies "safe to read",
while the OpenCL spec on `clEnqueueSVMMap(blocking=CL_FALSE)`
says bytes are only valid after the map event completes.
Structurally, SVM and cl_mem are identical: in both, the pointer
is known at submit time but bytes are only spec-valid after the
event. The honest sound shape is the same for both — a non-Deref
pending that you `.wait()` into a Deref guard.

**Bug fix landed alongside.** The blocking-only path for
`MappedReadGuard::drop` / `MappedWriteGuard::drop` was discarding
the unmap event. When map/unmap happened on a non-default queue,
`MappedSlice::drop`'s `clEnqueueSVMFree` (on the context default
queue, with `last_use` as its wait-list) didn't include the unmap
event, opening a latent cross-queue race. The guard's Drop now
registers the unmap event via `register_use` — fixes both the new
non-blocking case and the pre-existing blocking case.

**For cl_mem (DeviceSlice) guards.** No `last_use` analog — OpenCL
refcounts `cl_mem` internally during enqueue, so
`clReleaseMemObject` in `DeviceSlice::drop` waits without explicit
help. The guards expose `release(self) -> Result<Event>` for users
who want the unmap event for explicit cross-queue chain ordering
(consumes the guard, returns the event, suppresses the Drop unmap).

---

## Lower-priority deferred maintenance

These came out of the same review but I'd push back on or
skip rather than act on.

- **Audit codegen workarounds in `tests/kernels/src/lib.rs`.** The
  review suggested grepping for workaround comments and linking each
  to an upstream rust-gpu issue. Commit `195d4f0` already removed
  the largest set (defensive read-then-write kernel shapes) because
  the underlying issue was fixed upstream. A quick `grep -rn
  "workaround\|defensive" tests/kernels/ examples/` confirms what's
  left, but I expect a short list.

- **rust-gpu compatibility matrix as a hand-maintained doc.** The
  review suggested a matrix doc covering nightly, rust-gpu commit,
  OpenCL ICDs, known failures. A hand-maintained matrix rots. Better
  shape: derive it from `.github/workflows/ci.yaml` at CI time and
  publish it as a CI artifact (or just point at the workflow file
  itself as the source of truth).

- **`cargo check --workspace` timeout finding.** Mostly
  environmental — first-build including `rustc_codegen_spirv` (the
  compiler dylib) takes 4+ minutes on any cold target/. Not really
  a project finding; `CLAUDE.md`'s Build/Test section already covers
  the targeted-command alternatives.

---

## References

- Late-bind refactor (prerequisite for scope): claspr commit `f19457d`
- `KernelOp` split: `claspr/src/kernel_op.rs:46`,
  `claspr-async/src/op.rs`, `claspr-macros/src/lib.rs:758`
- Generated kernel TOML hardcoding: `claspr-build/src/lib.rs:897`
- `seed_lockfile_from_host` workaround: `claspr-build/src/lib.rs:883`
- Three-modes reference test: `tests/explicit-compile/`
- Follow-up review document:
  `/home/claudecode/projects/CLASPR-REVIEW.md` (outside repo)
- SYCL `queue.submit(|handler| { ... })` — inspiration for the
  scope shape; the handler captures the queue + accessor lifetime
  for the closure body.
- Repo planning docs that share this style: `IMPLEMENTATION-PLAN.md`,
  `EXECUTION-MODEL.md`, `REVIEW.md`.
