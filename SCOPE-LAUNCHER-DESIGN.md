# Tier 1 ergonomics — scoped launcher (Suggestion A)

Design note for a deferred Tier 1 ergonomics improvement. Flagged by
Brice as "Suggestion A" right after the late-bind refactor (commit
`f19457d`, 2026-06-04). Mirrored to the repo so the Mac session can
pick this up — the same content lives in this dev environment's
`project_claspr_scope_launcher` memo.

## The problem

After the late-bind refactor, every Tier 1 op terminal takes `&ctx`:

```rust
let mut buf = DeviceSlice::<u32>::alloc_zero(&ctx, N)?;
buf.write(&data).wait(&ctx)?;
let buf = kernels.fill_u32([N], buf, 42).wait(&ctx)?;
buf.read(&mut out).wait(&ctx)?;
```

The shape is **consistent** now (good — it's the same `&ctx` at every
terminal, the prerequisite for the next move), but the repetition is
visible noise on any sequence longer than three steps. For Tier 1
tutorials and short utility scripts the `&ctx` becomes line-by-line
boilerplate that obscures the actual work.

This is purely an ergonomics improvement. Nothing is broken; nothing
new is enabled. The goal is **drop one binding noise without
sacrificing the late-bind surface**.

## The proposed shape

A SYCL-inspired scoped launcher: borrow the context once, run a
closure where `.wait()` finds the launcher implicitly.

```rust
ctx.scope(|s| {
    let mut buf = DeviceSlice::alloc_zero(s, N)?;
    buf.write(&data).wait()?;             // s implicit via scope
    let buf = kernels.fill_u32([N], buf, 42).wait()?;
    buf.read(&mut out).wait()?;
    Ok(())
})?;
```

`s` is a `&Launcher`-shaped handle borrowed by the closure for its
lifetime. `.wait()` (no arg) inside the scope binds to `s`
automatically.

## Open design questions

These need to be resolved BEFORE implementation. Picking the wrong
mechanism here would be expensive to undo.

### Q1 — How does `.wait()` find `s` without an explicit arg?

Three plausible mechanisms, in order of magic-vs-mechanism trade-off:

1. **Thread-local stash.** `s` registers itself for the closure's
   lifetime via `tls`. Inside the scope, every `.wait()` reads the
   TLS slot. Clean syntax, invisible-magic, broken across spawned
   threads (the scope's TLS doesn't propagate, so a `std::thread::
   spawn` inside the closure would silently lose the launcher).
   Multi-context tests need care.

2. **Closure-captured handle.** `s` is a `Launcher` value the closure
   captures, and `.wait()` is a method on a wrapped `Op` type tied to
   the scope's lifetime. No TLS, no spawn hazard. Cost: every Op
   variant needs a `.wait()` overload that knows about the scope —
   meaningful proc-macro / type-level work.

3. **Don't go implicit.** `.wait()` still takes a launcher, but `s`
   is just a shorter binding: `scope(|s| { … buf.write(&data).wait(s)?; … })`.
   Less ergonomic win, but it's a trivial change — no plumbing.

My current lean: **Option 2** for correctness, with the trade-off
that we'd be committing to a moderate proc-macro / trait-impl
expansion. Option 1 silently miscompiles under spawned threads,
which is the wrong default for a "make it cleaner" feature. Option 3
is honest but doesn't actually buy enough to justify the API
addition.

If Option 2 is too much work, Option 3 is the safe fallback —
saves the `&` but keeps the explicit launcher visible.

### Q2 — Does `scope` give a single fixed launcher, or can users pick queues inside?

SYCL's `handler` is one queue per scope. For claspr we have
[`InOrder`/`OutOfOrder`] queues + multi-device support — so even
inside `scope`, an explicit `.wait(L)` for non-default queues
probably stays available. The scope is just a default. Decide
explicitly: does `scope` accept a queue argument, or always grab
the context's default in-order queue?

Suggested: `ctx.scope(|s| ...)` defaults to the in-order queue
(matches Tier 1's existing default); `ctx.scope_on(&queue, |s| ...)`
for explicit picks. Mirror the existing `ctx.launch_on(&queue, ...)`
naming if that already exists; otherwise just `scope_on`.

### Q3 — Interaction with Tier 2

Inside `scope`, can users mix in `.sync(s)?` for a chain? Probably
yes — `s` is a Launcher, and Tier 2's `sync(&ctx)` already accepts
any `Launcher`-shaped thing. Worth a test fixture to lock that in,
not a design question per se.

### Q4 — Naming bikeshed

`ctx.scope(|s| ...)` matches SYCL. Alternatives:
- `ctx.with_launcher(|l| ...)` — more explicit, more typing
- `ctx.run(|s| ...)` — too generic, conflicts with possible future verb
- `ctx.batch(|s| ...)` — implies batching that isn't happening

`scope` reads as the right verb. Keep it.

## Why save this rather than do it now

The late-bind refactor was the prerequisite, but the next move is a
**separate, larger design exercise**:

- Touches every Tier 1 op's terminal signature again (or only the
  convenience wrappers — design choice in itself).
- Needs decisions on Q1 (implicit vs explicit) that benefit from at
  least one real heavy Tier 1 use case to design against.
- Doing it cold without that signal risks over-designing for the
  wrong shape.

## When to revisit

- A user (or doc reviewer) hits a Tier 1 multi-step sequence and
  complains about the noise.
- We want to write a tutorial that needs less ceremony than today's
  shape — `&ctx` repetition would clutter the worked example.
- We have a concrete multi-step example exercising 4+ buffer ops in
  sequence where the `&ctx` repetition is visible enough to motivate
  the cost of implementation.

## Adjacent capability gaps (deferred separately)

Two real new Tier 1 surface items the late-bind work surfaced but
didn't ship — separate from the scope idea, just listed here so we
remember:

1. **`DeviceSlice::map` Tier 1.** Today `DeviceSlice` has no Tier 1
   zero-copy host access — only `.read()` (which copies) and the
   Tier 2 `host_view` chain (which forces you into combinator land).
   A builder + RAII guard for `clEnqueueMapBuffer` would parallel
   `MappedSlice::map`. Real new capability.

2. **Non-blocking `MappedSlice::map`.** `MappedSlice::map(&ctx)`
   today is blocking-only (`CL_TRUE`). A `.submit(&ctx)?` non-
   blocking variant returning `(Guard, Event)` would let Tier 1
   callers thread the map event through cross-queue ordering.

For both, the SVM and `cl_mem` cases diverge on what the non-blocking
shape returns:
- `clEnqueueSVMMap` returns only an event (pointer is owned by the
  allocation). Guard can deref immediately, with caller responsible
  for waiting on the event before reading bytes (mirrors OpenCL
  spec).
- `clEnqueueMapBuffer` returns pointer + event in the same call; the
  pointer is set synchronously but bytes are only valid after the
  event fires. Needs a `MapHandle` (no Deref) → `handle.into_view()`
  after event wait → Deref-able guard.

Forcing the two primitives into one type either gives SVM a
pointless `into_view()` step or makes `cl_mem` Deref unsafe-by-
default. The honest API has two shapes.

## References

- Late-bind refactor: claspr commit `f19457d` (2026-06-04) — every
  Tier 1 op terminal takes `&L: Launcher`. Prerequisite for scope.
- SYCL `queue.submit(|handler| { ... })` is the canonical
  inspiration; the handler captures the queue + accessor lifetime
  for the closure body.
- Repo planning docs that share this style: `IMPLEMENTATION-PLAN.md`,
  `EXECUTION-MODEL.md`, `REVIEW.md`.
