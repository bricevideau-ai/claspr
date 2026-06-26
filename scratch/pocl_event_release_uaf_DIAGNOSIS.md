# pocl event use-after-free on the async completion path — diagnosis

Status: ROOT-CAUSED. The originally-hypothesized fix (a lock inside
`clReleaseEvent`) does NOT close it; the real defect is a refcount
**over-release of one** on the marker/async-completion path, which turns the
application's own `clReleaseEvent` into a use-after-free. This needs surgery in
the event-lifecycle code, so it is reported rather than patched.

## Symptom

claspr's `await_propagates_chain_error` test SIGSEGVs ~25-30% of runs on pocl
(release build, `USE_POCL_MEMMANAGER=OFF`, the Ubuntu/distro default).

Crash backtrace (the application/executor thread):

```
Thread 2 received signal SIGSEGV
0x... in clReleaseEvent () from /usr/lib/x86_64-linux-gnu/libOpenCL.so.1   <- ICD loader
#1 cl3::runtime::...::clReleaseEvent
#4 cl3::event::release_event
#5 opencl3::event::Event::drop                     (event.rs:44)
#6 drop_glue<claspr::future::EventFuture>
#7 drop_glue<claspr::eager::DeviceChainFuture<...>>
#8 futures_executor::block_on
```

The faulting instruction is in the **ICD loader's** `clReleaseEvent`, not pocl:

```
mov (%rdi),%rax      ; rax = event->dispatch   (first member of cl_event)
cmp %rdx,(%rax)      ; *** FAULT: dereference event->dispatch ***
```

i.e. the loader reads `event->dispatch` from a `cl_event` whose memory pocl has
already `free()`d — a use-after-free across the ICD boundary, before pocl's
`POclReleaseEvent` is even entered. No lock inside `POclReleaseEvent` can help:
the unsafe read happens in the loader.

## Why it is pocl's bug, not claspr's

claspr's `EventFuture` (src/future.rs) holds exactly one `opencl3::Event` (one
reference, released once in Drop) and registers one `CL_COMPLETE` callback. That
is spec-legal and the reference accounting on the claspr side is balanced.

## Mechanism (proven by instrumenting pocl's retain/release/free)

The crashing event is the marker created by `clEnqueueMarkerWithWaitList`
(claspr enqueues a marker, awaits it). Tagging every retain/release/free with
thread id + refcount, the marker's full lifecycle in a CRASHING run is:

```
CREATE      id=6 rc=1   (app/main thread)         <- reference the user's handle owns
[retain]    id=6 ->2    (queue-internal, e.g. last_event / cq->events)
RET-SYNC    id=6 ->3    (in-order wait-list edge)
RET-SYNC    id=6 ->4    (in-order wait-list edge)
RET-CBPUSH  id=6 ->5    (async callback, pocl_event_cb_push)
REL         id=6 ->4    (worker)
REL         id=6 ->3    (worker)
REL         id=6 ->2    (worker)
REL         id=6 ->1    (worker)
REL         id=6 ->0 + FREE   (worker)            <- struct returned to heap
... then the app/main thread calls clReleaseEvent on the SAME handle  -> UAF
```

5 references exist; pocl issues **5 internal releases that drive the count to 0
and free the struct**, then the application releases its own (create) reference
— a **6th release of a 5-reference object**. pocl released the reference the
application owns. In runs that happen NOT to crash, the timing lets the app's
release be the one that reaches 0 (and a worker's release is the harmless
non-zero one), so it survives by luck — but it is still an over-release.

## Decisive experiments

1. Instrumented retain/release/free trace (above): 5 refs, 6 releases on the
   crash path; all 5 counted releases come from pocl worker threads, the app's
   release is uncounted (it crashes in the loader before pocl's decrement).
2. Making `clReleaseEvent` LEAK the event (skip `POCL_DESTROY_OBJECT` +
   `pocl_mem_manager_free_event`): **0 / 60 crashes** (vs ~25% baseline). The
   over-release becomes harmless because the memory is never reclaimed. This
   confirms the crash is a UAF on freed event memory, and that the extra
   release exists.
3. A global lock around the whole decision-and-free inside `POclReleaseEvent`
   (the originally-proposed option (a)): **18 / 100 still crash** — ineffective,
   because the faulting `event->dispatch` read is in the ICD loader, outside
   pocl, before the lock.

## Relevant code

- `lib/CL/clEnqueueMarkerWithWaitList.c` -> `pocl_create_command` (returns the
  marker at rc=1, the user's reference) -> `pocl_command_enqueue`
  (`lib/CL/pocl_util.c` ~779-820): in-order queue wires `pocl_create_event_sync`
  edges (each `POCL_RETAIN_OBJECT_UNLOCKED(waiting_event)`, pocl_util.c:422) and
  sets `command_queue->last_event.event` (pocl_util.c:820) — a queue-internal
  reference.
- Completion: `pocl_update_event_finished` (pocl_util.c:1678) updates status,
  calls `pocl_event_updated` -> `pocl_event_cb_push` (retains for async cb,
  pocl_util.c:2335), then `ops->broadcast` (releases the notify_list/sync
  references via `clReleaseEvent`, devices/common.c:914), and finally
  `clReleaseEvent(event)` at pocl_util.c:1786 (the command's own reference).
- The async worker releases the callback reference in `process_event_cb`
  (pocl_util.c:2400).

The over-release is in the interaction of these completion releases with the
queue-internal (`last_event`) reference vs the user-returned reference for a
MARKER. Pinning the exact extra release requires tracing which completion
release is meant for `last_event` vs the user handle; that is the surgery to do
upstream.

## Fix options (for upstream)

(a) **Correct the reference ownership** so pocl issues exactly one release per
    internal retain and never releases the user-returned reference. This is the
    right fix but needs careful auditing of the marker / `last_event` /
    sync-edge / completion releases to find the off-by-one. Highest correctness,
    requires the most care (regression risk across the event system).

(b) **Defer event reclamation** so the struct (and its embedded mutex) is not
    returned to the heap while any concurrent releaser may still deref it. This
    is essentially the `USE_POCL_MEMMANAGER` design, whose comment in
    `pocl_threads_c.h:139` states: "We recycle OpenCL objects by not actually
    freeing them until the very end. Thus, the lock should not be destroyed at
    the refcount 0." With the mem-manager OFF (default), `clReleaseEvent`
    violates that invariant: it both `POCL_DESTROY_OBJECT`s (destroys the mutex)
    and `free()`s at rc=0. Option (b) would make the default build honor the
    same no-early-free invariant (epoch/quiescent reclamation, or an event free
    list always-on). Masks an over-release rather than fixing it, but closes the
    UAF window for ALL such races, not just this one.

(c) A lock inside `POclReleaseEvent` (the original hypothesis): REJECTED — the
    faulting read is in the ICD loader, before pocl runs. Verified ineffective.

Recommended: (a) for correctness; (b) as defense-in-depth (the early
free+mutex-destroy at rc=0 is unsafe against any concurrent cross-thread
release regardless of this particular off-by-one).
