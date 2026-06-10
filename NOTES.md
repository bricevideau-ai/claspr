# claspr — rolling notes

Single rolling doc for active work, deferred items, and unresolved
concerns. Convention is documented in `CLAUDE.md` → "Inter-session
notes". **Append here; don't spawn new planning docs.** Prune as
items resolve.

---

## Active

*(nothing scoped — `git log` for what just shipped, or check Recent
landings below.)*

---

## Deferred

### Inherit generated kernel deps from host workspace

`claspr-build`'s generated kernel `Cargo.toml`
(`claspr-build/src/lib.rs:897`, `write_generated_cargo_toml`)
hardcodes the spirv-std / glam / num-complex deps to floating branch
refs. The host workspace pins them via `Cargo.lock`, but new
consumers without a workspace lockfile drift unpredictably. The
existing `seed_lockfile_from_host` (`claspr-build/src/lib.rs:883`)
is a mitigation, not a fix.

Approach (sketched): walk up from `OUT_DIR` to find the host
`Cargo.lock`, extract the pinned `rev` for spirv-std/glam/num-complex,
write those into the generated TOML. Fallback to today's hardcoded
branch refs if no lockfile found.

**Why deferred:** rust-gpu upstream landed a workaround for the same
class of glam/dep issue and it broke their CI. Anything we ship
here is likely incompatible with whatever shape they converge on.
Check rust-gpu's CI / recent merges / glam-workaround thread before
re-picking. Until then, the README "Limitations" bullet about the
hardcoded branch ref stays accurate.

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
