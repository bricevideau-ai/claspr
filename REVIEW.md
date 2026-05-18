# `runtime-redesign` branch — review notes

Review of the 12 commits on `runtime-redesign` (vs `main`), written from the linux box after reading every commit message + the new module surface. Direction: strong; this is the right shape for an OpenCL-with-rust-gpu framework. The Launcher trait + proc-macro retarget is the clever bit — preserves the trivial `kernels.foo(&ctx, ...)` while opening `&queue` for users who want explicit control. Cleanly aligns with SYCL 2020 (selectors, type-stated Queue, opt-in profiling, three explicit memory tiers, event-based async).

The `Image2D` redo is exemplary: the `Unorm`-vs-`Uint` diagnosis via Intel OpenCL Intercept Layer, the deliberate rename to mirror `cl_channel_type` instead of Vulkan/D3D, and the willingness to revert the first attempt to get the second one right — all of that pays back forever.

What follows is what I'd want resolved or considered before this lands on `main`. Roughly ordered by what I'd block on vs flag for follow-up.

## Things to resolve before merging

### 1. Finish or scope down the sticky-error counter

`Context::error_state: AtomicU32` is wired into `HostBuffer::Drop` only. `Queue::Drop`, `DeviceSlice::Drop`, and `SharedBuffer::Drop` don't bump it yet. The 78427d0 commit message explicitly flagged this as deferred ("Drop impls for buffers and queues will populate it in a later commit"); 98d95d4 wired the first one.

The cuda-oxide pattern's value is *coverage* — a half-wired counter is worse than no counter, because users who see `error_count() == 0` will believe nothing went wrong when in fact two of three buffer tiers and the queue itself don't report. Either:

- finish wiring it across every `Drop` impl that calls a fallible release function, or
- downgrade `Context::record_err` to `pub(crate)` and `Context::error_count` to a TODO marker until coverage is real.

`SharedBuffer::Drop` already records errors per its commit message — that's good. Need the same for `DeviceSlice::Drop` (release_mem_object) and `QueueInner::Drop` (release_command_queue).

### 2. Add a kernel-launching multi-device test

`examples/two-device` deliberately omits the kernel side because of an unrelated pocl 7.2-pre aarch64 build hang. That makes the multi-device *runtime API* proven but the end-to-end multi-device path unproven — no test exercises `Queue::on_device` + a kernel + cross-device buffer flow + the proc-macro's `&impl Launcher` on a `Queue<_>`.

On linux + Intel Iris Plus Gen11 + distro PoCL (or Intel + sub-device fallback), a kernel-launching variant should work. Worth adding before merge even if the pocl/aarch64 path stays `#[cfg(not(...))]`-gated or `#[ignore]`. The runtime-only proof is good for the API but misses the proc-macro integration question: does `kernels.foo(&q1, ...)` compile and run correctly when `q1: Queue<InOrder>` lives on a non-default device of a multi-device Context?

### 3. Verify the `&Queue<OutOfOrder>` path through the proc-macro

The blanket `impl<L: Launcher + ?Sized> Launcher for &L` should make `kernels.foo(&out_of_order_queue, [N], &buf)` Just Work, calling through to the sync `Launcher::launch` default. But it's the kind of inference + coherence interaction that can surprise you when combined with the macro's exact emitted signature.

A one-line test like:

```rust
let q = Queue::<OutOfOrder>::new(&ctx)?;
kernels.collatz_kernel(&q, [N], &buf)?;  // should block, like &ctx does
```

would close the loop. If it works, document it as the supported pattern; if it doesn't compile, either fix the trait setup or document that out-of-order requires `launch_with_deps` explicitly.

## Things worth thinking about, not necessarily blocking

### 4. `launch_with_deps` loses the macro typing

It's an inherent method on `Queue<OutOfOrder>` taking a raw `Kernel` + arg tuple. Users dropping into DAG composition lose `kernels.foo(...)` typed wrappers — they're back to manual `clSetKernelArg`-style arg construction with whatever the macro would have done for them. No `kernels.foo_async(&q, deps, [N], &buf) -> Event` generated.

Two reasonable directions:

- **`LauncherAsync` trait** with a default `launch_with_deps` method, and the proc-macro keys off it to emit `_async` variants alongside the sync ones. Cost: doubles the macro-generated surface; users who never go async pay nothing.
- **Leave it explicit** — the moment you want DAG composition, you commit to dropping into the lower-level API. Documented as a deliberate handoff, not a cliff.

Either works. Current state is the latter implicitly; if that's the intent, say so in the rustdoc on `launch_with_deps` (and on the proc-macro overview).

### 5. `Image2D<A, F>` type-state has out-run the proc-macro

Host now has rich `Image2D<ReadOnly, R32Float>` etc. type-state, but the proc-macro still emits `&Image2DRgba8` for every `&Image!(...)` kernel parameter. So a user who allocates `Image2D<WriteOnly, R32Float>` can't actually pass it to a generated wrapper — they'd be forced back to the `Image2DRgba8` alias by the macro signature.

This is already a backlog item ("Image format dispatch in the proc-macro" in claspr's CLAUDE.md). Just noting that the runtime side has out-paced the macro side, and the gap is visible to users *today* who try to use the new format/access markers. The cleanest sequencing is probably: land this branch, then a follow-up branch that teaches the proc-macro to read `format=`/`sampled=`/access from the `Image!` macro tokens and emit the matching `&Image2D<A, F>` signature.

### 6. `Buffer<T>` trait is "mostly informational" by its own admission

That's honest, but it means user code can't really write `fn upload_and_run<B: Buffer<u32>>(...)` because upload semantics differ per tier (`DeviceSlice::upload(&launcher, &data)` vs `HostBuffer::map_mut + index` vs `SharedBuffer::map_mut + index`).

If the goal was "common `len`/`is_empty`/`ctx` accessor for plumbing," fine — name and dock it as such. If the goal is real tier polymorphism, the trait needs to grow (a `BufferUpload<T>` super-trait with a uniform upload method that's a no-op for HostBuffer/SharedBuffer and a real `clEnqueueWriteBuffer` for DeviceSlice). My read of the current shape is the former; if so, the rustdoc could be a touch more explicit about "this is plumbing, not polymorphism."

### 7. The `Error::Other(String)` carryover and migration plan

The new typed enum is the right direction. `Other` is a deliberate catchall per the commit message, and `From<String>/&str` shims keep existing call sites compiling. Worth tracking: which call sites *currently* hit `Other`, and what typed variants should subsume each? Otherwise `Other` becomes permanent furniture. A grep-and-categorize task that turns into a few small follow-up commits seems right.

## Strategic note — connection to capability auto-declare

The redundancy I flagged on a prior call — host-side `Image2D<A, F>` vs kernel-side `Image!(access=…, format=…)` — gets *deeper* with this branch, not shallower. That's not a blocker; the runtime-side type-state is the right thing to have for ergonomics and safety. But it reinforces the case for the eventual capability auto-declare work (the `auto()` build.rs item in CLAUDE.md backlog) flowing *type information* from kernel source into host wrappers, not just `OpCapability` declarations.

The runtime redesign and the cap auto-declare direction are pulling toward the same destination: **one source of truth between kernel and host**. Worth keeping in mind when the macro-side Image format dispatch (item 5) gets designed — that work is the natural bridge between these two threads.

## Net recommendation

Merge once **(1)** and **(2)** are addressed. **(3)** is a quick test and should be in the same merge. **(4)–(7)** are follow-up tickets.

— Reviewed on the linux box, 2026-05-18, after reading commit messages + new module surface (`error/device/context/queue/buffer/svm/image/future/launch`) + the two-device example + the `Image2D` rewrite.
