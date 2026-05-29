# claspr execution model — WIP design

**Status: work in progress.** Captures the design conversation around
the unified sync/async/event/deps execution model that supersedes
today's `Launcher` / `LauncherAsync` split. Sections marked
**DECIDED** are settled; **OPEN** is still being argued; **DEFERRED**
is "agreed to do later." Update as the conversation moves.

> ## ⚠️ Major reframing (2026-05-20)
>
> After studying [cuda-oxide](https://nvlabs.github.io/cuda-oxide/) and
> spiking a combinator API, **the design has been reframed as two tiers**:
>
> - **Tier 1 (explicit-queue layer)** — what most of this doc describes,
>   minus the `Handle` / `.track()` / `Pending<T>` / `BorrowHandle`
>   complexity. Pure `.wait()` / `.await` over explicit `&Queue<O>`.
>   For scripts, tests, and users who want full control.
> - **Tier 2 (combinator async layer)** — new. Lazy `DeviceOperation`
>   composed via `and_then` / `bundle!` / `fan_out` / `arc`. Hides queues
>   and events. Inspired by cuda-oxide but takes advantage of OpenCL's
>   native event model, multi-device contexts, and (eventually) command
>   buffers. For pipelines that need overlap, DAGs, async runtimes.
>
> See the new **"Two-tier architecture (Tier 1 + Tier 2)"** section at
> the bottom of this doc for the V2 design. The detailed material in
> the middle of this doc captures V1 exploration — preserved as design
> history. Where V1 and V2 conflict, V2 wins.

## Motivation

Today's claspr API has three different shapes for "enqueue something":

- `DeviceSlice::upload` / `download` block via `CL_BLOCKING` (sync, no event)
- `DeviceSlice::copy_to` returns an `Event` (async, event-back)
- `Image2D::download_bytes`, `MappedSlice::map_mut` block
- Kernel launches: `Launcher::launch` is sync, `LauncherAsync::launch_with_deps` is async-with-event

Three+ verb shapes for users to learn. Plus inconsistencies:

- `LauncherAsync` is gated to `Queue<OutOfOrder>` only — **incorrect**, since CL allows wait-list events on InOrder queues for cross-queue sync.
- No API path for the CUDA-stream pattern (async, no event allocated — opencl3 always materialises an Event).
- No `Queue::barrier()` for the OOO fork-join pattern.

Design constraint: **events are an opt-in cost.** SYCL was bitten by
mandatory events (every accessor / every queue.wait materialised one,
adding driver overhead). Untracked sync ops should use `CL_TRUE`
blocking enqueues where the CL spec allows it and skip event
allocation entirely.

Goal: one uniform verb shape per enqueue op that honestly exposes the
meaningful axes.

## Axes — DECIDED

Three orthogonal user choices per call:

1. **Block on this call?** sync vs async
2. **Get a completion handle back?** no vs yes (the `.tracked()` modifier)
3. **Wait on external events first?** no deps vs deps list

Two queue properties (independent of call-site choices):

- **Queue ordering** — InOrder (queue serialises its own commands) vs OutOfOrder (commands run when explicit deps resolve)
- **Source queue** — events/handles carry an `Arc` to their source queue so auto-flush works on cross-queue use

## API shape — DECIDED

Single builder per operation. Typestate-parameterised
(`Untracked` / `Tracked`). `#[must_use]` on the builder so a forgotten
terminal is a warning, not a silent bug.

```rust
op.SOMETHING(launcher, ...args)            // returns OpBuilder<_, Untracked, T>

    // Modifiers
    .after(deps)                            // wait-list handles (omit when no deps)
    .tracked()                              // flips to OpBuilder<_, Tracked, T>

    // Terminals
    .wait()                                 // sync, block until done
    .await                                  // Future impl, async runtime
    .detach()                               // fire-and-forget, no event allocated
    .track()                                // submit + return Handle, no wait
```

### Terminal/modifier matrix

For **pure ops** (kernels, barrier, finish, unmap):

|              | `Untracked`                | `Tracked` |
|---|---|---|
| `.wait()`    | `Result<()>`               | `Result<Handle>` |
| `.await`     | `Result<()>`               | `Result<Handle>` |
| `.detach()`  | `Result<()>`               | ✗ typestate error |
| `.track()`   | `Result<Handle>`           | ✗ typestate error |
| `.tracked()` | → Tracked                  | ✗ typestate error |

For **pure ops with a borrowed source/destination** (copy_from from a
host slice, copy_to to a host slice, etc.):

|              | `Untracked`                       | `Tracked` |
|---|---|---|
| `.wait()`    | `Result<()>`                      | `Result<Handle>` |
| `.await`     | `Result<()>`                      | `Result<Handle>` |
| `.detach()`  | ✗ method absent (unsafe — borrow ends, runtime still uses pointer) | ✗ typestate error |
| `.track()`   | `Result<BorrowHandle<'a, T>>`     | ✗ typestate error |
| `.tracked()` | → Tracked                         | ✗ typestate error |

For **resource ops** (`MappedSlice::map_mut`, etc.):

|              | `Untracked`                   | `Tracked` |
|---|---|---|
| `.wait()`    | `Result<T>`                   | `Result<(T, Handle)>` |
| `.await`     | `Result<T>`                   | `Result<(T, Handle)>` |
| `.detach()`  | ✗ method absent (loses T)     | ✗ typestate error |
| `.track()`   | `Result<Pending<T>>`          | ✗ typestate error |
| `.tracked()` | → Tracked                     | ✗ typestate error |

`.tracked().wait()` and `.track().wait()` give the same final
`(T, Handle)` shape for resource ops — they're two ways to express
"I want both the resource and the profiling event," differing in
whether the wait happens at the call site or is deferred. Both kept;
slight redundancy serves different use cases (immediate-block vs
deferred-completion).

`.detach()` is **method-absent** (not typestate-forbidden) in two
cases, for different reasons:

- **Resource ops** (map_mut): detach loses the resource entirely. Never correct.
- **Copy ops with borrowed source/dest**: detach is *unsafe in Rust terms*. The OpBuilder takes `&mut [T]` (or `&[T]`); `.detach()` would consume the builder and end the borrow, but the CL runtime keeps using the pointer after the call returns — the user could read/write the host slice while the runtime is mid-transfer (UB). The borrow MUST extend until completion is signalled. Use `.track() -> BorrowHandle<'a, T>` instead — the handle holds the borrow until `.wait()`/`.await` releases it.

The fix is honest: ops that fundamentally can't satisfy detach's contract simply don't expose it. Spike validates the rejection produces a clean `E0599: no method named 'detach' found` error.

`.track()` on resource ops is safe via `Pending<T>` because the resource is held inside the wrapper and only released on wait.

### Code-smell prevention via typestate

```rust
op.tracked().track()?;        // error: no method `.track()` on Tracked builder
op.tracked().detach()?;       // error: no method `.detach()` on Tracked builder
op.tracked().tracked();       // error: no method `.tracked()` on Tracked builder
```

Each invalid combination produces a clear `no method named X found
for type OpBuilder<_, Tracked, _>` error. No runtime check needed.

### Event-allocation cost model

| Path | Event allocated? | Notes |
|---|---|---|
| `.wait()` untracked | **No** (uses CL_TRUE blocking flag where CL supports it) | Genuine zero-cost-event path |
| `.await` untracked | **Yes** (needed for `clSetEventCallback`; not exposed) | No way around it — async needs the callback |
| `.detach()` untracked | **No** (NULL event-out param) | Required for CUDA-stream batching |
| `.track()` untracked | Yes (returned as Handle) | The DAG-composition path |
| `.tracked().wait()` | Yes (returned as Handle) | Adds cost vs untracked `.wait()` |
| `.tracked().await` | Yes (returned as Handle) | Same cost as untracked `.await` — event existed anyway |

The asymmetry is intentional: `.tracked().wait()` is the one
combination that actually adds an event allocation over its untracked
sibling. `.tracked().await` is "free" — the event was needed for the
callback anyway, we just expose it instead of dropping.

### Why this shape

- **`.wait()` + `.await`** as siblings: same builder, same args; the user picks blocking vs async at the call site.
- **`.detach()`** opts out of event allocation entirely. Required for CUDA-stream batching to actually be cheap.
- **`.track()`** for DAG composition — submit, get handle, defer waiting.
- **`.tracked()` modifier** is the opt-in for "I want the event back from the same terminal that's waiting" — common for profiling.
- **`#[must_use]`** prevents "I called the method but forgot to submit" silent bugs.
- **Typestate** makes redundant/contradictory combinations compile errors instead of runtime surprises.

## Construction vs enqueue — DECIDED

Discipline: **any op that enqueues returns a builder; any op that pure-constructs returns the value directly.**

- Allocators (`DeviceSlice::alloc`, `MappedSlice::alloc`, `Image2D::alloc`) are pure context ops — they `clCreateBuffer`/`clCreateImage` only, no enqueue. Return `Result<Self>` directly.
- Transfer ops (`copy_from` / `copy_to` / kernel launch / image read-write / SVM map-unmap / queue barrier-finish) all return `OpBuilder<...>` and require a terminal.

For the prototype path, sync sugar constructors that combine alloc + transfer + wait:

- `DeviceSlice::from_slice(&ctx, &host_data) -> Result<DeviceSlice<T>>` — alloc + `copy_from(&host_data).wait()` in one call. Mirrors `Vec::from`. Sync only — async users go through `alloc` + `copy_from` explicitly.

## Verb naming — DECIDED

- `copy_from(src)` replaces `upload` (host → device) and absorbs the device→device half of today's `copy_to`.
- `copy_to(dst)` replaces `download` (device → host) and absorbs the other half of today's `copy_to`.
- Direction encoded in the parameter type via trait dispatch (`CopyFrom<&[T]>`, `CopyFrom<&DeviceSlice<T>>`, `CopyTo<&mut [T]>`, `CopyTo<&mut DeviceSlice<T>>`). Same verb pair generalises to all source/destination kinds.
- Matches `cust` (the CUDA-Rust crate) terminology — alignment with existing Rust GPU ecosystem.

## Handle types — DECIDED

What `.track()` returns depends on the op's shape. Three flavors, all
implementing `Future` and exposing the underlying `Event` for profiling:

### `Handle` — pure op, no borrow

```rust
pub struct Handle {
    event: opencl3::Event,
    queue: Arc<QueueInner>,    // source queue, for auto-flush
}

impl Clone for Handle { /* cheap: cl_event refcount + Arc refcount */ }
impl Future for Handle { type Output = Result<()>; ... }
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Handle {
    pub fn wait(&self) -> Result<()> { ... }                    // by ref — usable after
    pub fn event(&self) -> &opencl3::Event { ... }              // profiling, raw access
}
```

Returned by `.track()` on pure ops (kernel launches, barrier, finish,
unmap, copy ops where the destination is a non-borrowed value).
**Cloneable** so the same handle can be a dep for many later ops
without lifetime gymnastics.

### `Pending<T>` — resource op with deferred result

```rust
#[must_use = "pending operations require .wait() or .await to extract the resource"]
pub struct Pending<T> {
    // Real impl: handle + ManuallyDrop<T>; spike uses Option<T>.
    handle: Handle,
    resource: T,
}

impl<T> Pending<T> {
    pub fn wait(self) -> Result<(T, Handle)> { ... }            // consumes, yields tuple
    pub fn event(&self) -> &opencl3::Event { ... }              // pre-completion profiling setup
}

impl<T: Unpin> Future for Pending<T> {
    type Output = Result<(T, Handle)>;
    ...
}
```

Returned by `.track()` on resource ops (`MappedSlice::map_mut`, etc.).
The resource `T` is held *inside* the Pending — the only way to access
it is `.wait()` or `.await`, both of which yield `(T, Handle)`. The
compiler enforces "don't touch the resource until the op completes"
because there's no method on `Pending<T>` that yields `T` without
waiting. **Not Cloneable** (single ownership of T).

Drop without wait: real impl needs a strategy (likely fire-and-forget
callback that drops T after the event fires). Spike leaks.

### `BorrowHandle<'a, T>` — pure op with a destination borrow

```rust
#[must_use = "in-flight copy must be waited on to release the destination borrow"]
pub struct BorrowHandle<'a, T> {
    handle: Handle,
    _dst: &'a mut [T],         // borrows the destination for handle's lifetime
}

impl<'a, T> BorrowHandle<'a, T> {
    pub fn wait(self) -> Result<()> { ... }
    pub fn event(&self) -> &opencl3::Event { ... }
}

impl<'a, T: Unpin> Future for BorrowHandle<'a, T> {
    type Output = Result<()>;
    ...
}
```

Returned by `.track()` on copy ops whose destination is a borrowed
`&mut [T]` (`DeviceSlice::copy_to`, `Image2D::copy_to`, `HostBuffer`,
`MappedSlice` host transfers). The lifetime `'a` ties the handle to
the destination — the borrow checker rejects any user attempt to read
the destination while the handle is alive. `.wait()` / `.await`
consumes the handle, releasing the borrow.

The compiler error a user sees if they try is clean:
```
error[E0499]: cannot borrow `host` as mutable more than once at a time
```

### `IntoDeps` trait (powers `.after(...)`)

Accepts the following shapes for `Handle`:

- `&Handle` — single dep
- `[&Handle; N]` — fixed-size array of borrows
- `&[Handle]` — borrowed slice
- `Handle` — single owned (uses Clone internally to extract cl_event)
- `Vec<Handle>` — owned vec (Clone internally per element)

`Pending<T>` and `BorrowHandle<'a, T>` are NOT directly usable as deps
— they're single-ownership / borrow-bound. To use their event as a
dep, call `.wait()` / `.await` first to get back the underlying
`Handle` (which is Cloneable).

**No `()` case** for `.after()`. If you want no deps, you don't call
`.after()` at all. Falls out of the builder pattern.

**`.after()` does NOT accept in-progress builders.** A builder hasn't
been submitted yet; there's no cl_event to wait on. The user must call
`.track()` first to get a Handle. This is a hard rule.

### `IntoDeps` trait (powers `.after(...)`)

Accepts the following shapes:

- `&Handle` — single dep
- `[&Handle; N]` — fixed-size array of borrows
- `&[Handle]` — borrowed slice
- `Handle` — single owned (uses Clone internally to extract cl_event)
- `Vec<Handle>` — owned vec (Clone internally per element)

**No `()` case.** If you want no deps, you don't call `.after()` at all.
That simplification falls out of the builder pattern.

**`.after()` does NOT accept in-progress builders.** A builder hasn't
been submitted yet; there's no cl_event to wait on. The user must call
`.track()` first to get a Handle. This is a hard rule — keeps lifetimes
sane and the model honest about what events actually exist.

## MappedSlice guards — DECIDED

`MappedSlice::map_mut` returns a builder; terminals can yield the
guard alone (untracked), `(guard, Handle)` (tracked), or a
`Pending<MapGuard>` (track for deferred completion). Unmap is
explicit, no RAII fallback.

```rust
// Untracked sync — zero event allocation (CL_TRUE blocking_map)
let g = shared.map_mut(&q).wait()?;
g[i] = value;
shared.unmap(g).wait()?;

// Untracked async — event allocated internally for callback, not exposed
let g = shared.map_mut(&q).await?;
g[i] = value;
shared.unmap(g).await?;

// Tracked sync/async — event also exposed for profiling
let (g, h) = shared.map_mut(&q).tracked().await?;
let map_dur = h.event().profiling_command_end()? - h.event().profiling_command_start()?;
g[i] = value;
shared.unmap(g).await?;

// Deferred (.track()) — submit now, get guard later, optionally
// do other work in between
let p = shared.map_mut(&q).track()?;
do_other_async_work().await?;
let (g, h) = p.wait()?;     // or p.await?
g[i] = value;
shared.unmap(g).await?;
```

Properties:

- **Composable everywhere.** Unmap is a normal builder with all terminals.
- **Async-safe.** No syscall hidden in Drop.
- **`#[must_use]` on the guard** catches the most common bug at compile time.
- **Panics in Drop** if the guard wasn't unmapped — gated on `!std::thread::panicking()` to avoid double-panic abort. In panic-in-panic case, leak the map (resource leak > abort).

The scoped-closure alternative was considered and rejected — interacts
badly with `.await` (async closures unstable, `?` doesn't propagate
cleanly across closures).

## Launcher trait — DECIDED

Today's `Launcher::launch` default method goes away. Trait shrinks to
two pure accessors:

```rust
pub trait Launcher {
    fn cl_queue(&self) -> &CommandQueue;
    fn context(&self) -> &Context;
}

impl<O: QueueOrder> Launcher for Queue<O> { ... }
impl Launcher for Context { ... }                       // uses bundled default queue
impl<L: Launcher + ?Sized> Launcher for &L { ... }      // blanket for references
```

Name kept as `Launcher` — even with `.launch` gone, it's still "the
thing you launch ops onto," and existing users won't be surprised. The
trait is a minimum surface; queue-specific methods like
`.barrier()`/`.finish()`/`.on_device()` stay as inherent methods on
`Queue<O>`.

OpBuilders are generic over `L: Launcher` and typestate `S`:

```rust
pub struct KernelOp<'a, L: Launcher, S = Untracked> {
    launcher: &'a L,
    kernel: &'a Kernel,
    args: ...,
    deps: Vec<cl_event>,
    _state: PhantomData<S>,
}

pub struct Untracked;
pub struct Tracked;

impl<'a, L: Launcher> KernelOp<'a, L, Untracked> {
    pub fn after(mut self, deps: impl IntoDeps) -> Self { ... }
    pub fn tracked(self) -> KernelOp<'a, L, Tracked> { ... }
    pub fn wait(self) -> Result<()> { ... }
    pub fn detach(self) -> Result<()> { ... }
    pub fn track(self) -> Result<Handle> { ... }
}
impl<'a, L: Launcher> KernelOp<'a, L, Tracked> {
    pub fn wait(self) -> Result<Handle> { ... }
}
impl<'a, L: Launcher, S> Future for KernelOp<'a, L, S> {
    type Output = ...;     // varies by S
    ...
}
```

Each op gets its own builder type (`UploadOp`, `KernelOp`, `BarrierOp`,
etc.), keeping claspr's typed-launch-wrapper pitch intact. The
proc-macro emits a named builder type per kernel.

## Auto-flush mechanism — DECIDED v0; v1 DEFERRED

Every Handle carries an `Arc<QueueInner>` of its source queue.
Auto-flush on use:

- `Handle::wait()` — flush source queue first, then wait.
- `Handle::poll()` (Future impl, first poll) — flush source queue first, then register callback.
- `IntoDeps::into_deps()` (for `.after(...)`) — flush each handle's source queue first.

### v0 (do first)

Always call `clFlush` on each touch. Driver typically makes redundant
`clFlush` cheap. Ship this, measure if it matters.

### v1 (deferred, if profiling shows v0 is too noisy)

Logical clock per queue. Counter incremented per enqueue. Handle carries
its timestamp. Queue tracks `last_flushed` timestamp. Skip `clFlush` if
`handle.ts <= queue.last_flushed`.

Subtleties for v1:

1. **Snapshot counter BEFORE clFlush, not after.** Race: while in the syscall another thread can enqueue. Snapshot-then-flush-then-store-snapshot. Worst case: one extra flush, never a missed one.
2. **Funnel every implicit-flush path through one helper.** `clFinish`, queue `Drop`, anywhere implicit-flush happens — all must update `last_flushed` or it drifts behind reality and you pay extra flushes.

Cross-queue deps flush the **dep's source queue**, not the destination
queue. (Source must run to completion; destination doesn't need flushing
until someone waits on its output.) That's baseline for both v0 and v1.

## Per-operation table

| Type | Pure constructors | Enqueue ops (return OpBuilder) | T |
|---|---|---|---|
| `DeviceSlice<T>` | `alloc(&ctx, len)`, `from_slice(&ctx, &data)` (sync sugar) | `copy_from(launcher, src)`, `copy_to(launcher, dst)` | `()` |
| `HostBuffer<T>` | `alloc(&ctx, len)`, `from_slice(&ctx, &data)` (sync sugar) | `copy_from(launcher, src)`, `copy_to(launcher, dst)` * | `()` |
| `MappedSlice<T>` | `alloc(&ctx, len)`, `from_slice(&ctx, &data)` (sync sugar) | `map_mut(launcher)` | `MapGuard` |
| | | `unmap(guard)`, `copy_from`, `copy_to` * | `()` |
| `Image2D<A, F>` | `alloc(&ctx, width, height)` | `copy_from(launcher, host_pixels)`, `copy_to(launcher, host_pixels)` | `()` |
| `Queue<O>` | `new(&ctx)`, `on_device(&ctx, &dev)` | `marker()`, `barrier()`, `finish()` | `()` |
| Kernels (proc-macro emits) | (none) | per-kernel: `foo(launcher, grid, args...)` returns named OpBuilder | `()` |

\* `HostBuffer` and `MappedSlice` transfers may be pure host memcpy
under the hood (no enqueue) but still return OpBuilders for API
uniformity. See open Q.

## Worked scenarios

### Scenario 1 — sync prototype

```rust
let ctx = Context::any()?;
let kernels = gpu::kernels(&ctx)?;
let mut data: Vec<u32> = (1..=1024).collect();
let buf = DeviceSlice::from_slice(&ctx, &data)?;        // sync sugar
kernels.collatz_kernel(&ctx, [N], &buf).wait()?;
buf.copy_to(&ctx, &mut data).wait()?;
```

### Scenario 2 — CUDA-stream batching (Async + InOrder, no events)

```rust
let q = Queue::<InOrder>::new(&ctx)?;
let buf = DeviceSlice::<u32>::alloc(&ctx, N)?;
for batch in batches {
    buf.copy_from(&q, batch).detach()?;
    kernels.filter(&q, [N], &buf).detach()?;
    kernels.transform(&q, [N], &buf).detach()?;
    kernels.reduce(&q, [N], &buf, &out).detach()?;
}
q.finish().wait()?;
```

Zero event allocations in the loop.

### Scenario 3 — fork-join with barriers (OOO, no events)

```rust
let q = Queue::<OutOfOrder>::new(&ctx)?;
kernels.kern_a(&q, [N], &buf_a).detach()?;
kernels.kern_b(&q, [N], &buf_b).detach()?;
kernels.kern_c(&q, [N], &buf_c).detach()?;
kernels.kern_d(&q, [N], &buf_d).detach()?;
q.barrier().detach()?;                                  // structuring sync point
kernels.combine(&q, [N], &buf_a, &buf_b, &buf_c, &buf_d, &buf_out).detach()?;
q.finish().wait()?;
```

Zero events.

### Scenario 4 — cross-queue / multi-device

```rust
let ctx = Context::for_devices(&[dev_a, dev_b])?;
let q_a = Queue::<InOrder>::on_device(&ctx, &dev_a)?;
let q_b = Queue::<InOrder>::on_device(&ctx, &dev_b)?;
let kernels = gpu::kernels(&ctx)?;

buf.copy_from(&q_a, data).detach()?;
let h_produce = kernels.produce(&q_a, [N], &buf).track()?;
let h_consume = kernels.consume(&q_b, [N], &buf)
    .after(&h_produce)
    .track()?;
h_consume.wait()?;
```

Auto-flush: `h_consume.wait()` flushes `q_b`; `.after(&h_produce)`
flushes `q_a` (the dep's source queue).

### Scenario 5 — async runtime

```rust
async fn pipeline(q: &Queue<InOrder>, kernels: &Kernels) -> Result<()> {
    let h = kernels.foo(&q, [N], &buf).track()?;
    do_other_async_work().await?;
    h.await?;
    Ok(())
}
```

### Scenario 6 — kernel profiling (sync)

```rust
let h = kernels.foo(&q, [N], &buf).tracked().wait()?;
let dur = h.event().profiling_command_end()?
        - h.event().profiling_command_start()?;
```

### Scenario 7 — map profiling (async)

```rust
async fn map_profiled(shared: &MappedSlice<u8>, q: &Queue<InOrder>) -> Result<()> {
    let (g, h) = shared.map_mut(q).tracked().await?;    // (guard, handle)
    let map_dur = h.event().profiling_command_end()?
                - h.event().profiling_command_start()?;
    g[i] = value;
    shared.unmap(g).await?;
    Ok(())
}
```

### Scenario 8 — deferred map via `Pending<MapGuard>`

```rust
async fn deferred_map(shared: &MappedSlice<u8>, q: &Queue<InOrder>) -> Result<()> {
    let p = shared.map_mut(q).track()?;      // Pending<MapGuard>, not yet usable
    do_other_async_work().await?;            // map happens in background
    let (g, _h) = p.await?;                  // wait for map, then access guard
    g[i] = value;
    shared.unmap(g).await?;
    Ok(())
}
```

### Scenario 9 — copy with borrow-extending track

```rust
async fn copy_then_other(buf: &DeviceSlice<u32>, q: &Queue<InOrder>) -> Result<()> {
    let mut host = vec![0u32; N];

    let h = buf.copy_to(q, &mut host).track()?;     // BorrowHandle<'_, u32>
    // `host` is borrowed by `h` — compile error if you read host here
    do_other_async_work().await?;
    h.await?;
    // borrow released; host accessible now
    println!("{:?}", &host[..4]);
    Ok(())
}
```

### Scenario 10 — copy + buffer reuse: three patterns

Same outcome ("copy from device, then reuse the host buffer"), three
expressions depending on whether the user wants overlap:

```rust
// 10a. Simple sync — no overlap, simplest code, zero event allocation
let mut host = vec![0u32; N];
buf.copy_to(&q, &mut host).wait()?;     // blocks; CL_TRUE blocking, no event
host[0] = 99;                            // safe — borrow released at wait
println!("{:?}", host);

// 10b. Sync with CPU overlap — do CPU work during the copy
let mut host = vec![0u32; N];
let h = buf.copy_to(&q, &mut host).track()?;   // BorrowHandle<'_, u32>
let cpu_work = do_other_cpu_work();             // overlaps with GPU copy
h.wait()?;                                       // consumes h, releases borrow
host[0] = 99;                                    // safe — borrow released
println!("{:?}", host);

// 10c. Async with overlap — do other async work during the copy
async fn flow(buf: &DeviceSlice<u32>, q: &Queue<InOrder>) -> Result<()> {
    let mut host = vec![0u32; N];
    let h = buf.copy_to(q, &mut host).track()?;
    do_other_async_work().await?;                 // executor multiplexes
    h.await?;
    host[0] = 99;
    println!("{:?}", host);
    Ok(())
}
```

The borrow checker is what makes the "host[0] = 99" line safe in all
three. If you tried to touch `host` between `.track()?` and the
wait/await in 10b/10c:

```rust
let h = buf.copy_to(&q, &mut host).track()?;
host[0] = 99;       // error[E0499]: cannot borrow `host` as mutable
                    //               more than once at a time
h.wait()?;
```

You get the error at compile time, not a heisenbug at runtime.

## Implementation touch points (rough)

- `claspr/src/queue.rs`:
  - `Launcher` trait — shrunk to two accessors; `launch()` default impl removed.
  - `LauncherAsync` removed (the OOO gate was wrong; the new model has no need for it).
  - `Queue::marker()` added (`clEnqueueMarkerWithWaitList` — event-producing fan-in, no block on later commands).
  - `Queue::barrier()` added (`clEnqueueBarrierWithWaitList` — event-producing fan-in, blocks later commands).
  - `Queue::finish()` becomes a builder (`Queue::finish() -> FinishOp`).
- `claspr/src/handle.rs` (NEW): `Handle` (pure-op completion, Clone+Send+Sync, Future), `Pending<T>` (deferred-resource wrapper for resource ops, single-ownership, Future yielding `(T, Handle)`), `BorrowHandle<'a, T>` (borrow-extending wrapper for copy ops, Future yielding `()`), source-queue Arc, Future impls, `IntoDeps` trait, `Untracked`/`Tracked` typestate markers.
- `claspr/src/buffer.rs`:
  - `DeviceSlice::upload` → split into `alloc` (pure) + `copy_from` (builder) + `from_slice` (sync sugar).
  - `DeviceSlice::download` → `copy_to` (builder).
  - `DeviceSlice::copy_to` (current cross-buffer) → unified into the new `copy_from`/`copy_to` dispatch.
  - `HostBuffer` gets the same `alloc` + `copy_from` + `copy_to` + `from_slice` surface.
- `claspr/src/svm.rs`:
  - `MappedSlice::map_mut` returns a builder; terminal returns guard alone (untracked) or `(guard, Handle)` (tracked).
  - New `MappedSlice::unmap(guard)` builder.
  - Guard's Drop panics if not explicit-unmapped (guarded on `!thread::panicking()`).
- `claspr/src/image.rs`: image read/write rewritten as `copy_from`/`copy_to` builders.
- `claspr/src/future.rs`: `EventFuture` deleted (Handle's Future impl replaces it). The `async-events` feature flag retires.
- `claspr-macros/src/lib.rs`: kernel proc-macro emits one method per kernel returning a named OpBuilder type, typestate-parameterised. No more `_async` sibling.
- `claspr-build/src/lib.rs`: explicit-mode codegen emits same shape.

Plus every example updated.

## Open questions

1. **`Queue::finish()` `.after(...)` semantics.** Keep open. `clFinish` alone doesn't take deps, but OpenCL has `clEnqueueMarkerWithWaitList` / `clEnqueueBarrierWithWaitList` which fan-in events into the queue. So `.finish().after(deps)` could reasonably desugar to "enqueue a barrier-with-wait-list for deps, then clFinish" — one-line shorthand for the explicit two-line version. Decide based on whether the use case shows up.
2. **`DeviceSlice` sub-region copies.** Symmetric with the image `.region(...)` modifier — `clEnqueueCopyBuffer` and `clEnqueueWrite/ReadBuffer` all support offset + size for partial transfers. Probably want the same modifier shape on `DeviceSlice::copy_from` / `copy_to`. Not blocking the rewrite — can add post.

## Implementation notes

### `#[must_use]` wording

```rust
#[must_use = "operations don't execute until terminated; call .wait(), .await, .detach(), or .track()"]
pub struct OpBuilder<L, S, T> { ... }

#[must_use = "tracked operations don't execute until terminated; call .wait() or .await"]
pub struct TrackedOpBuilder<L, T> { ... }  // or however the typestate distinction lands at the type level
```

Match std's passive-voice-plus-actionable-hint convention.

### Image sub-region transfers

Modifier on the existing `copy_from`/`copy_to` builder, matching `.after()` / `.tracked()`:

```rust
img.copy_to(&q, &mut host_pixels).wait()?;                           // full image (default)
img.copy_to(&q, &mut host_pixels).region([x, y], [w, h]).wait()?;    // sub-region
img.copy_from(&q, &host_pixels)
    .region([x, y], [w, h])
    .row_pitch(stride_bytes)                                          // optional, default 0
    .wait()?;
```

Builder internally stores `Option<(origin, region, row_pitch, slice_pitch)>`; when `None`, computes the full image at terminal time. For 3D images, region becomes `([x, y, z], [w, h, d])` — type-specific to `Image2D` vs `Image3D`.

### Migration plan

Atomic rewrite on a fresh branch (e.g. `execution-model-rewrite` cut from `runtime-redesign`). Commit structure inside the PR:

1. Add `Handle`, `IntoDeps`, `Untracked`/`Tracked` typestate ZSTs (new `claspr/src/handle.rs`).
2. Refactor `Launcher` trait (strip `launch()` default, remove `LauncherAsync`).
3. Rewrite `DeviceSlice` (`alloc` / `copy_from` / `copy_to` / `from_slice` + builder).
4. Rewrite `HostBuffer` (same surface).
5. Rewrite `MappedSlice` (explicit unmap, guard with panic-on-Drop).
6. Rewrite `Image2D` (`copy_from` / `copy_to` builders with `.region(...)` / `.row_pitch(...)` modifiers).
7. Add `Queue::marker()` / `barrier()` / `finish()` builders.
8. Update `claspr-macros` (emit named per-kernel typestate-parameterised builders).
9. Update `claspr-build` (same shape for explicit-mode codegen).
10. Delete `EventFuture`, retire `async-events` feature flag.
11. Update every example.

Lint-clean and tests-pass enforced at the end of the PR. Intermediate commits at obvious-boundary steps (1, 2, 7, 11) should compile cleanly; the rest may be transient.

Single PR with this commit structure keeps git history reviewable even if intermediate states don't compile. The pre-rewrite branch stays intact as a fallback.

## Decisions log (so we don't re-litigate)

- **Three axes are orthogonal** — sync/async, event/no-event, deps/no-deps. The `LauncherAsync`-gated-to-OOO design was wrong; deps work on InOrder too (cross-queue sync).
- **`op_enqueue` and `op_async` can't be built from each other** — `_enqueue` opts out of event allocation at the CL level (NULL out-param); you can't retroactively make an event from a no-event submit, nor avoid event cost by discarding after the fact. Hence the typestate / multi-terminal design.
- **Builder per op, not method-per-axis.** Compresses the 3+ verbs into one entry point per op with terminals on the builder.
- **Drop the sync-default bare verb.** The async-default with `.wait()` for the prototype path is cleaner and more honest.
- **`track` over `keep`** for the event-returning terminal. `keep` isn't a Rust idiom; `track` matches our earlier `tracked()` typestate terminology.
- **Source-queue Arc on Handle** is baseline for any auto-flush design (not just the v1 counter optimization).
- **v0 auto-flush** is "always clFlush on each touch"; v1 logical-clock is deferred until profiling shows the redundant flushes matter.
- **Construction vs enqueue split** — pure constructors return `Result<Self>`; enqueue ops return `OpBuilder<...>`. `from_slice` sync sugar bridges the common prototype case.
- **`copy_from` / `copy_to`** replaces `upload` / `download`. Single verb pair, direction via parameter type. Matches `cust` naming.
- **MappedSlice unmap is explicit** — no RAII fallback. Guard's Drop panics (or leaks in panic-in-panic) if unmap was forgotten. Composability and async-safety win over implicit cleanup.
- **`.after()` accepts Handles only** — not in-progress builders. Hard rule. Builders must be terminated with `.track()` to be usable as deps.
- **Handle is Clone + Send + Sync** — cheap clone via cl_event refcount + Arc clone. Enables fan-out without lifetime tangles.
- **`Handle::wait(&self)`** — by reference, not by value. Handle remains usable after for profiling, further deps, etc.
- **Launcher trait kept** — shrunk to `cl_queue()` + `context()` accessors. No default methods. Queue-specific methods stay inherent on `Queue<O>`.
- **Per-kernel named OpBuilder types** emitted by proc-macro — preserves claspr's typed-launch-wrapper pitch over a generic erased `KernelOp`.
- **Typestate `.tracked()` modifier** for "I want a handle from the same terminal that's waiting" — pure ops return `Result<Handle>` from terminals when tracked, resource ops return `Result<(T, Handle)>` tuples. The typestate makes redundant combinations (`tracked().track()`, `tracked().detach()`, `tracked().tracked()`) compile errors automatically.
- **Events stay opt-in cost** — untracked sync ops use `CL_TRUE` blocking enqueues where the CL spec allows (map_mut, copy_from, copy_to), allocating zero events. Untracked async ops must allocate an event internally (callback machinery needs it) but don't expose it. Tracked variants expose the event as a Handle. This avoids SYCL's "every op materialises an event" footgun.
- **HostBuffer / SVM `copy_from`/`copy_to` always go through CL enqueue** — even though the buffers are host-visible and the impl could short-circuit to plain memcpy after sync, going through `clEnqueueSVMMemcpy` / `clEnqueueWriteBuffer` gives uniform semantics (Handle, deps, tracked variants all work the same as for DeviceSlice) and lets the queue handle ordering for free. Users who want the "I'll do the memcpy myself after sync" pattern can write it explicitly: `q.finish().await?; host.copy_from_slice(hb.as_slice())` or `let g = svm.map_mut(q).await?; host.copy_from_slice(&g); svm.unmap(g).await?`. That's a user-side choice, not our implementation's.
- **`Pending<T>` for resource-op `.track()`** — earlier conservative call ("no `.track()` on resource ops") was wrong. The deferred-resource pattern is safe via a wrapper that holds the resource internally and only releases it on `.wait()` / `.await`. The compiler enforces "don't touch the resource until the op completes" because there's no method on `Pending<T>` that yields `T` without going through the wait. `Pending<T>` is NOT Clone (single ownership of the resource); `Pending<T>::wait()` returns `(T, Handle)` so the user gets the resource AND a Handle for profiling. Validated via spike.
- **`BorrowHandle<'a, T>` for copy-op `.track()`** — when the op's destination is a borrowed `&mut [T]`, `.track()` returns a Handle with a `PhantomData<&'a mut [T]>` so the destination borrow is extended for the lifetime of the handle. Rust's borrow checker rejects any attempt to read the destination while the handle is alive (`E0499: cannot borrow as mutable more than once at a time`). `.wait()` / `.await` consumes the handle, releasing the borrow. Validated via spike.
- **`.detach()` is method-absent on borrowed-source/dest copy ops** — fundamentally unsafe. The OpBuilder takes `&mut [T]` (or `&[T]`); `.detach()` would consume the builder and end the borrow, but the CL runtime keeps using the pointer after the non-blocking enqueue returns. The user could read/write the host slice while the runtime is mid-transfer — UB. Forced users to use `.wait()` (simple sync) or `.track() -> BorrowHandle<'a, T>` (overlap pattern). For users wanting zero-event fire-and-forget into host-side storage, the answer is to use a host-managed buffer type (`HostBuffer`, `MappedSlice`) where the "host side" is CL-managed memory with no external Rust borrow. Spike validates this rejection produces a clean `E0599: no method named 'detach' found` error.

## References

- `REVIEW.md` items 4 (LauncherAsync) and 5 (Image format dispatch) — landed but item 4's OOO gate was wrong; this design supersedes it.
- This design conversation transcript (session pending compaction).
- [cuda-oxide book](https://nvlabs.github.io/cuda-oxide/) — the design we're benchmarking against for Tier 2.

---

# Two-tier architecture (Tier 1 + Tier 2) — DECIDED

The earlier sections of this doc were a single-tier attempt that kept
growing complexity (BorrowHandle, Pending\<T\>, .tracked() typestate)
to handle async patterns inside an explicit-queue API. After studying
cuda-oxide and spiking a combinator API, we landed on a cleaner split.

## Tier 1 — explicit-queue layer (simplified)

The thin wrapper around OpenCL primitives. Direct, simple, sequential.

**What stays from V1:**
- `Context`, `Queue<InOrder>` / `Queue<OutOfOrder>`, `Device`, `Launcher` trait.
- Construction-vs-enqueue split — `alloc` is pure, transfer ops return builders.
- `copy_from` / `copy_to` verbs (matches cust naming).
- `from_slice` sync sugar constructor.
- Auto-flush v0 on `Handle::wait` / Future poll / IntoDeps.
- `MappedSlice::map_mut` returns a guard, explicit `unmap`, panic-on-Drop.
- `Image2D<A, F>` format/access typestate.

**What's dropped from V1:**
- `.track()` terminal — no Handle for "submit + return event without wait"
- `.detach()` terminal — no fire-and-forget at this tier (Tier 2 handles batching)
- `.tracked()` modifier — no typestate-flipped terminal returning a Handle
- `BorrowHandle<'a, T>` — no borrow-extending handle
- `Pending<T>` — no deferred-resource wrapper
- `Handle` as a user-facing type — internal only, used for `.after()` deps in cross-queue sync
- `Queue::marker()` — moves to Tier 2 (or stays as a queue inherent for direct use)

**What Tier 1 looks like:**

```rust
let ctx = Context::any()?;
let kernels = gpu::kernels(&ctx)?;
let mut data: Vec<u32> = (1..=1024).collect();
let buf = DeviceSlice::from_slice(&ctx, &data)?;        // sync sugar
kernels.collatz_kernel(&ctx, [N], &buf).wait()?;
buf.copy_to(&ctx, &mut data).wait()?;
```

Two terminals on every op-builder: `.wait()` and `.await`. That's it.
No event juggling, no DAG composition, no fire-and-forget. For
prototype scripts and users who want "I'm in charge of when each
thing happens."

The `.after()` modifier stays for cross-queue sync (the genuine
multi-queue case where you have an event from queue A that queue B's
op needs to wait on):

```rust
buf.copy_from(&q_a, data).wait()?;
let h = kernels.produce(&q_a, [N], &buf).submit()?;    // explicit handle
kernels.consume(&q_b, [N], &buf).after(&h).wait()?;
```

`.submit()` here is "submit without blocking, give me the handle for
cross-queue sync." This is the *one* explicit-event path in Tier 1.
Most users never reach for it.

## Tier 2 — combinator async layer (new)

Lazy `DeviceOperation` composed via combinators. Hides queues and
events. Validated by the spike at `/tmp/claspr-combinator-spike/`.

### Core types (from the spike)

```rust
pub trait DeviceOperation: Send + Sized {
    type Output: Send;
    fn execute(self, ctx: &ExecutionContext) -> Result<Self::Output>;

    fn and_then<F, U>(self, f: F) -> AndThen<Self, F>
    where F: FnOnce(Self::Output) -> U + Send,
          U: DeviceOperation;

    fn arc(self) -> Arced<Self>;  // wrap output in Arc<T>
}

pub fn value<T: Send>(v: T) -> Value<T>;                       // lift host data
pub fn with_context<F, O>(f: F) -> WithContext<F>;             // defer to ctx-aware closure
pub fn bundle2<A, B>(a: A, b: B) -> Bundle2<A, B>;             // 2-tuple
pub fn bundle3<A, B, C>(a: A, b: B, c: C) -> Bundle3<A, B, C>; // 3-tuple
// ... up to BundleN via macro-generated structs
pub fn fan_out<I, F, U>(inputs: Vec<I>, f: F) -> FanOut<I, F>; // N-ary tile parallel
```

Plus `#[macro] bundle!` which dispatches by arity to the
appropriate `BundleN` struct. Variadic (up to whatever bound we ship
— probably 16 like `tokio::join!`), not capped.

### Execution surfaces

- `.sync(log)` — block, run via default scheduler. Scripts/tests.
- `.await` (via IntoFuture impl) — async, yields to runtime.
- (Future) `tokio::spawn(op.into_future())` — true cross-stream concurrency.

### Scenarios validated by the spike

All 14 scenarios compile and run:

1. Linear chain (producer/consumer) ✓
2. Independent parallel branches via `bundle!` ✓
3. Diamond (fan-out + fan-in via `Arc`) ✓
4. ML-pass-style multi-stage with state carried forward ✓
5. In-place mutation chain ✓
6. N-ary fan-out via `fan_out(tiles, f)` then combine ✓
7. Multi-producer single consumer via `bundle3` + combine ✓
8. Mixed sync/async (split await with host work in between) ✓
9. Conditional graph shape via `DynOp` (`Box<dyn ...>`) ✓
10. Error propagation through `and_then` (short-circuits via `?`) ✓
11. Buffer round-trip (pass into chain, get back out) ✓
12. Profiling via `.profiled()` combinator returning `Profiled<T>` ✓
13. Concurrent pipelines with `Arc`-shared inputs ✓
14. Cross-device pipeline with `transfer_to_device(buf, n)` ops ✓

### Complexities surfaced

1. **Rust 2024 `impl Trait` capture rules** — kernel methods on the
   `Kernels` struct must take `self` by value (Clone + 'static) or
   use the verbose `+ use<>` precise-capture syntax. The proc-macro
   should emit `Kernels` as Clone + 'static (Arc-backed internally),
   and method signatures like `fn kernel_name(self, ...) -> impl Op`.
   Equivalent to cuda-oxide's `Arc<CudaModule>` pattern.

2. **`Arc<T>` requires `T: Sync` for `.arc()`** — small bound, easy
   to satisfy (`DeviceSlice`, `MappedSlice` are Send+Sync since
   `cl_mem` is thread-safe per CL spec).

3. **Conditional graphs need type erasure (`DynOp`/`Box<dyn ...>`)** —
   same situation as cuda-oxide. One Box allocation per conditional
   branch. Tolerable.

4. **State carried through stages is wordy** — `let pipeline = ... .and_then(|x| op.and_then(move |()| value((x, other, more))))` —
   tuple pack/unpack at every stage boundary. cuda-oxide has the
   same pattern in their async_mlp example.

5. **Multiple `Arc::clone()` before `bundle!`** — `let s1 = shared.clone(); let s2 = shared.clone(); bundle!(use(s1), use(s2))`.
   A `shared.split2()` / `split3()` helper that returns N clones
   would clean this up.

### Improvements over cuda-oxide that OpenCL enables

These are the genuine "Tier 2 + OpenCL > Tier 2 + CUDA" arguments:

- **N-ary fan-in** via `clEnqueueMarkerWithWaitList` (CL marker with
  arbitrary wait-list). cuda-oxide's `zip!` is fixed to 2-3 arity;
  arbitrary fan-in requires Tokio. We can support a true `join_all`
  combinator on top of CL markers.
- **Multi-device contexts** — single `cl_context` spans devices.
  Cross-device deps are just events flowing through the same context.
  Tier 2 graphs naturally span devices without per-device runtime
  partitioning (cuda-oxide needs `init_device_contexts(default, n)`
  + explicit per-device pools).
- **Batch parallelism via `fan_out` + marker, no spawn required.** N
  homogeneous pipelines (typical batch-inference / per-tile pattern)
  bundle into one `fan_out` value; submission puts all N chains on
  the OOO queue with no inter-chain deps; implicit
  `clEnqueueMarkerWithWaitList` joins their final events at the end.
  `.sync()` / `.await` blocks on the marker. cuda-oxide needs
  `tokio::spawn` for this because their chains lock to one stream;
  we get it from OOO + marker without a runtime dependency. Spawn is
  still needed for streaming (batches arrive over time), heterogeneous
  dynamic-arity work, and host-side concurrency (e.g., web servers).
- **Sub-buffers** (`clCreateSubBuffer`) for tile-parallel writes with
  static disjointness. Compiler reasons about parent/sub-buffer
  separation; no `DisjointSlice`-style runtime trust.
- **Queue-level profiling** (`CL_QUEUE_PROFILING_ENABLE`) is opt-in at
  context-creation time. When enabled, every event has timestamps for
  free — per-op `.profiled()` adds no further allocation. Off by
  default; external profilers (cliloader, vendor tools) don't need it.
- **Command-buffer extension** (`cl_khr_command_buffer`) for replay
  patterns. Lazy graph values that get executed many times can be
  recorded once and replayed with minimal overhead — CUDA Graphs
  equivalent. Ben Ashbaugh's Command Buffer Layer enables prototyping
  even on drivers without native support. Implementation detail for
  now; future optimization path.
- **`.and_then_host(|x| ...)` combinator** for in-queue host computation.
  Inspired by `clEnqueueNativeKernel` (which most GPU drivers don't
  support) but **emulated as a pure combinator** — the chain's
  execute order already provides the "after prior op, before next op"
  sequencing we need. No driver capability dependency. Implementation
  is ~20 lines: an `AndThenHost<S, F>` struct whose `execute` runs
  the source op, then calls the closure, returning the value
  directly (vs `and_then`'s "closure returns a new op"). Strictly
  better than CUDA's `cuLaunchHostFunc` (fire-and-forget, no return
  value) because we get to return values that feed downstream ops.
  Lifetime story is also cleaner than split-await — buffers flow
  through closures and never escape into long-lived Rust bindings
  mid-pipeline.

### Execution model: trust OOO, default queues per device

No pool, no scheduling policy, no round-robin. The `Context` carries,
per device:

- One default **in-order queue** (lazily created on first Tier 1 use)
- One default **out-of-order queue** (lazily created on first Tier 2 use)

Profiling is **off by default** on both queues. Users who want
programmatic timestamps opt in at context creation:

```rust
let ctx = Context::builder()
    .device(dev)
    .profiling(true)    // CL_QUEUE_PROFILING_ENABLE on default queues
    .build()?;
```

External profilers (Intel OpenCL Intercept Layer / cliloader, vendor
tools) work without `CL_QUEUE_PROFILING_ENABLE` because they
intercept CL calls and time externally. Most perf debugging reaches
for those anyway. Opt-in matches our "events are opt-in cost"
principle — users who don't profile don't pay.

Implication for Tier 2's `.profiled()` combinator: it errors at
submit time (not graph-build time, since the graph is queue-agnostic
until execution) if the queue doesn't have profiling enabled.
Friendly error message points to `Context::builder().profiling(true)`.

**Tier 1** uses the per-device default in-order queue. Power users
pass their own queue: `kernels.foo(&my_queue, ...).wait()?`.

**Tier 2** walks the graph and enqueues each node on its device's
default OOO queue with prior nodes' events as wait-list. The OOO queue
+ hardware does the actual parallelization via event dependencies — we
just submit. The Tier 2 scheduler is ~30 lines.

Cross-device coordination is natively handled by the shared CL context
— events on device 0's queue can be deps for ops on device 1's queue
without any framework glue.

Power user escape hatch: `op.on_queue(&custom_q)` per node, for the
case where the driver fake-serializes OOO and the user wants to fan
out manually across multiple queues. Lives in their code, not ours.

We deliberately do NOT hedge against drivers that fake OOO. If the
driver collapses an OOO queue to serial, the user sees serial perf.
That's the driver's problem; papering over it would mean re-inventing
cuda-oxide's stream-pool tuning question for our users.

### Buffer Drop semantics — cl_mem vs SVM

OpenCL's two memory-release APIs have **opposite semantics**, and the
buffer types' Drop impls must respect this:

**`clReleaseMemObject`** (for `cl_mem` — `DeviceSlice<T>`, `Image2D<A,F>`,
`HostBuffer<T>`): **lazy / refcount-based.**

> After the memory_object reference count becomes zero **and commands
> queued for execution on a command-queue(s) that use memory_object
> have finished**, the memory object is deleted.

Host-side Drop just decrements refcount; CL runtime defers actual
deletion until in-flight commands using the buffer complete. Drop is
non-blocking and safe to call while commands are in flight. This is
the same property cuda-oxide engineered into `DeviceBox` via
`cuMemFreeAsync`; OpenCL gives it to us by default.

**`clSVMFree`** (for SVM — `MappedSlice<T>`): **immediate, UB if used
while commands in flight.**

> Note that clSVMFree does NOT wait for previously enqueued commands
> that may be using svm_pointer to finish before freeing svm_pointer.
> It is the responsibility of the application to make sure that
> enqueued commands that use svm_pointer have finished before freeing
> svm_pointer. … The behavior of using svm_pointer after it has been
> freed is undefined.

For SVM we must use **`clEnqueueSVMFree`**, the safe queued variant
that schedules the free after specified wait-list events fire.

The Tier 1/Tier 2 design therefore:

- **`DeviceSlice::Drop`** and friends call `clReleaseMemObject` —
  non-blocking, safe always. (We already wrap opencl3's Drop in
  `ManuallyDrop` to capture release errors into the sticky-error
  counter; the semantics are correct.)
- **`MappedSlice::Drop`** calls `clEnqueueSVMFree` on the source
  queue with the last-known event as a dep. The Rust `Arc<QueueInner>`
  it holds is dropped immediately after; if that hits zero,
  `clReleaseCommandQueue` fires; CL defers queue deletion until the
  SVMFree completes. Drop ordering is correct via the CL runtime's
  lazy-deletion semantics — no callback-keep-alive gymnastics needed
  in Rust.

> ⚠️ **Audit item for current claspr Tier 1**: today's `MappedSlice`
> may be calling `clSVMFree` directly via opencl3 wrappers. That's
> UB if the buffer is dropped while a kernel is using it. Needs
> verification + fix to use `clEnqueueSVMFree`.

### Open design questions for Tier 2

1. **`Kernels` struct shape for the proc-macro.** Clone + 'static
   (Arc<Inner> internally) — vs ZST + lookup-by-name. Spike used
   Clone + Copy (ZST); real impl needs Arc for the inner program/kernel
   handles.

2. **`Arc::split_n()` ergonomics.** Worth a helper combinator? Or
   leave to the user?

3. **Profiling integration — callback-based, not wrapped-Output.** The
   spike's `Profiled<T>` wrapped-Output shape only works synchronously
   (it reads timestamps inside the chain's `execute`, where the event
   is already complete). In real async claspr the cl_event exists but
   hasn't fired by the time the next stage's closure runs — reading
   `clGetEventProfilingInfo` returns `CL_PROFILING_INFO_NOT_AVAILABLE`.

   Correct shape: `.profiled(|info| ...)` takes a callback registered
   via `clSetEventCallback(event, CL_COMPLETE, thunk)`. The callback
   fires when the GPU finishes; the user closure runs on a CL driver
   thread and receives the timestamps. Chain doesn't block; Output type
   stays unchanged (profiling is side-effect, not data flow).

   FFI safety requirements for the thunk:
   - User closure boxed as `FnOnce(ProfilingInfo) + Send + 'static`;
     pointer passed via clSetEventCallback's `user_data`
   - Thunk uses `catch_unwind` to prevent panics across FFI (UB otherwise)
   - Event retained at registration time, released after callback fires
   - Errors from `clGetEventProfilingInfo` logged, not propagated (no
     way to propagate from a callback)

   For whole-pipeline profiling, layer a `ProfileCollector`
   (`Arc<Mutex<Vec<Entry>>>`) on top — each `.profile_into(&collector,
   "name")` registers a callback that pushes an entry on completion.
   After `.sync()`/`.await` the collector has all stages' timings.

4. **Cross-device routing.** Spike used explicit `transfer_to_device(buf, n)`.
   Could be auto-inferred from the queue's device — `kernel.run(...)
   .on_device(&dev_b)` would imply a transfer if buf is on dev_a.

5. **Concurrent pipelines and the Tokio dependency.** cuda-oxide
   takes Tokio. We want to stay runtime-agnostic (work with smol,
   async-std, tokio, custom). Need our own waker dispatch driven
   by `clSetEventCallback`. ~few hundred LOC; doable.

6. **`bundle!` upper bound and naming.** Variadic via macro-generated
   `Bundle2..BundleN` structs (~30 LOC of macro to emit each N). Cap
   at 16 or so by convention — same approach `tokio::join!` uses.
   `bundle!` chosen over `zip!` (iterator-naming mismatch) and
   `join!` (would collide with future `spawn`/`join` runtime API).
   Spike's hand-coded `Bundle2`/`Bundle3` is just the bootstrap; the
   real impl generates them.

7. **Borrow extension when ownership-transfer-into-chain isn't a fit.**
   Some patterns want "borrow a buffer for the chain's lifetime,
   release at end." Combinator equivalent would be `with_borrow(&buf,
   |bref| pipeline_using_bref)`. Worth designing.

8. **Command-buffer integration.** When does a lazy graph value get
   lowered to a `cl_khr_command_buffer` for replay vs enqueued
   per-execution? User-facing hint vs auto-detection of
   execute-many-times patterns?

9. **Error context.** Errors from mid-chain stages don't carry context
   ("which stage failed?"). Could add a `.named("stage_name")` combinator.

10. **Tier 1 / Tier 2 interop.** Should a Tier 2 graph be able to
    consume Tier 1 buffers / handles? Likely yes — the buffer types
    should be shared. The graph wraps them with appropriate lifetime
    extension.

11. **`.and_then_host(|x| ...)` combinator.** Pure-combinator
    emulation of in-queue host work — no `clEnqueueNativeKernel`
    needed. Open question is mostly about the async-runtime story:
    - For sync `.sync()`: closure runs on the calling thread between
      ops. Easy.
    - For `.await`: closure runs inside Future::poll on the executor's
      thread. If expensive, blocks the executor. Document as caveat;
      user can wrap with `tokio::task::spawn_blocking` (or runtime
      equivalent) inside the closure if needed.

12. **`HostAccessible<T>` trait with three-stage acquire/work/release
    pattern.** The "host accesses device data" pattern needs different
    CL machinery per buffer type. The cleanest API splits it into three
    chain stages so the acquire and release are real, queue-ordered ops
    that pipeline with other in-flight work:

    ```rust
    .and_then(|buf|  buf.acquire_host_view())     // d2h or map, queue-ordered
    .and_then_host(|mut view| {
        view[0] += 1.0;
        Ok(view)
    })
    .and_then(|view| view.release_to_device())    // h2d or unmap
    ```

    Per buffer type:
    - `DeviceSlice<T>`: acquire = d2h to scratch Vec; release = h2d back
    - `MappedSlice<T>` (coarse SVM): acquire = clEnqueueSVMMap; release = clEnqueueSVMUnmap
    - `HostBuffer<T>` (persistent-mapped): acquire/release = no-op (value-wrap)
    - Fine-grain SVM / Shared USM: acquire/release = no-op

    Why the split form beats putting acquire/release inside the host
    closure with CL_TRUE blocking:
    - Each stage is a Future-shaped enqueue; in async contexts, the
      executor isn't blocked by the GPU transfers, only by the actual
      host work.
    - Other in-flight chain branches (e.g., from `bundle!`) can
      overlap with the host work, since acquire/release are async.
    - Lifetime of the HostView is bounded by the chain (it's a value
      flowing through), not by user-managed scope.

    Open subquestions:
    - HostView's exposed type — uniform `HostView<T>` with internals
      varying per buffer source, or per-type concrete (`DeviceSliceHostView`,
      `SvmMapGuard`, etc.)? The user writes the same three-stage form
      either way.
    - Read-only acquire that skips the writeback for DeviceSlice
      (`acquire_host_view_ro` for the common case of "inspect, don't
      modify"). Saves a transfer when no modifications happen.
    - For fine-grain SVM where the host pointer is coherent, even the
      acquire/release no-op stages could be elided entirely — but
      having them present keeps the API uniform.

### Implementation plan (sketch)

1. **Simplify Tier 1.** Remove `.track()`/`Handle`/`Pending<T>`/
   `BorrowHandle`/`.tracked()` from current claspr code. Keep
   `Handle` as an internal type for the rare cross-queue `.after()`
   case in Tier 1. Result: Tier 1 is small, sharp, sequential.

2. **Build `claspr-async` crate.** New crate alongside claspr that
   provides the Tier 2 combinator API. Depends on claspr for the
   underlying buffer/queue/kernel types. No async-events feature
   gate — the crate IS the async story.

3. **Reuse the spike's structure.** Core trait, AndThen, Bundle2..N,
   FanOut, Value, WithContext, Arced, .arc(), .profiled() all
   ported directly. Add Future + IntoFuture impls driven by
   `clSetEventCallback`.

4. **Build the trivial scheduler.** Per-device default OOO queue
   carried by Context (lazy init). Tier 2's `execute` walks the
   graph and enqueues each node on its device's queue with prior
   events as wait-list. No pool, no policy. ~30 LOC.

5. **Proc-macro emits both surfaces.** Same kernel definition emits
   `kernels.foo(launcher, ...).wait()` (Tier 1) AND
   `kernels.foo_op(...)` (Tier 2 builder returning DeviceOperation).
   Like cuda-oxide's `vecadd` / `vecadd_async`.

6. **Reuse existing examples.** Port collatz/raymarch/etc to use
   Tier 1's simplified API. Add a new MLP-pipeline-style example that
   uses Tier 2 for the composition.

### Spike location

`/tmp/claspr-combinator-spike/` — single-file (`src/main.rs`, ~750
lines), standalone Cargo project, all 14 scenarios run cleanly.
Validates the type structure end-to-end; everything is faked at the
runtime level (no real OpenCL calls).
