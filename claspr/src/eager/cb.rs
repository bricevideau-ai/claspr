//! Command-buffer machinery for the eager graph (CB-as-EXECUTION-MODE).
//!
//! The `CbCache` home + the record-time boundary/reach helpers that turn a
//! seam-free device subtree into one `cl_khr_command_buffer` and replay it. This
//! is the SKIPPABLE layer: nothing here is needed to understand a leaf's normal
//! `execute`; read it only when touching command-buffer recording. See
//! `ARCHITECTURE.md` -> "CB-as-EXECUTION-MODE".

use super::*;
// `ExecutionContext` comes via `use super::*` (eager's import). `CbWalk` is
// re-exported below (`pub use crate::exec_ctx::CbWalk`) and flows back to eager.rs
// through its `pub(crate) use cb::*`.

// ── CbCache: the graph-owned home for a node's finalized command buffer ──────

/// A CB-capable node's **owned, interior-mutable home** for the
/// [`FinalizedCb`](crate::record::FinalizedCb) it built (design v2, CB-as-
/// execution-mode). Each node that can *create/home* a command buffer — the
/// combinators (`AndThen`, `Bundle*`, `FanOut`, structural passthroughs) and the
/// device leaves (`Fill`, `CopyTo2`, the macro-generated kernel `Op`) — carries
/// one, created empty at build. It is NOT routed through
/// [`output_pipe`](DeviceOp::output_pipe): the CB-creating node is frequently a
/// composite whose `output_pipe` is `None` (e.g. CG's root `AndThen` over a
/// `bundle2`), so the cache must ride the node itself.
///
/// The per-node algorithm reads/writes its OWN field: "home it in yourself" =
/// store here; the replay fast-path "if a CB is homed, and one is given that
/// matches → do nothing" and "no CB given, homed CB valid → replay" both check
/// THIS field. It drops with the graph (the `FinalizedCb`'s RAII release), so the
/// cache needs no global table and cannot ABA. [`mutate_bind`](DeviceOpExt::mutate_bind)
/// clears the caches of the CB-homing nodes a mutated slot reaches (via the
/// recursive [`invalidate_cbs`](DeviceOp::invalidate_cbs) walk).
pub type CbCache = Arc<Mutex<Option<Arc<crate::record::FinalizedCb>>>>;

/// A fresh, empty [`CbCache`] — every CB-capable node initializes its field with
/// this at build time.
pub fn new_cb_cache() -> CbCache {
    Arc::new(Mutex::new(None))
}

/// Clear this node's OWN homed CB iff it baked a buffer/scalar traceable to a mutated
/// slot (`captured_slots ∩ mutated ≠ ∅`). The own-cache half of
/// [`invalidate_cbs`](DeviceOp::invalidate_cbs); each CB-homing combinator calls this
/// on its `cb_cache` field then recurses into its own children (the recursion shape —
/// `source`/`next`, a field list, or `self.ops` — is the part that legitimately varies
/// per combinator, so it stays explicit at the call site).
pub fn cb_cache_invalidate(cache: &CbCache, mutated: &std::collections::BTreeSet<usize>) {
    let mut g = lock_unpoisoned(cache);
    if g.as_ref().is_some_and(|cb| cb.depends_on_any(mutated)) {
        *g = None;
    }
}

/// Push this node's OWN homed [`FinalizedCb`](crate::record::FinalizedCb) identity
/// (`Arc::as_ptr`) into `out`, if it currently holds one. The own-cache half of
/// [`collect_cb_ids`](DeviceOp::collect_cb_ids); the caller recurses into its children.
pub fn cb_cache_collect_id(cache: &CbCache, out: &mut Vec<usize>) {
    if let Some(arc) = lock_unpoisoned(cache).as_ref() {
        out.push(Arc::as_ptr(arc) as usize);
    }
}

// ── CB-mode fork helpers (design v2, CB-as-EXECUTION-MODE) ───────────────────
//
// These are the shared primitives the per-node fork uses so every CB-capable op's
// `execute` (leaf) or terminal gather (boundary node) stays small. The fork lives
// in each op (per the spec), but the mechanical parts — "collect an entry leaf's
// external cl_events into this CB's ext accumulator", "run the boundary protocol
// (build-or-replay a CB, home it, enqueue with external deps)" — are here.

pub use crate::exec_ctx::CbWalk;

/// A leaf in CB-mode collects its resolved input's `cl_event` deps into the CB's
/// EXTERNAL accumulator (the event↔sync-point boundary): a NON-EMPTY wait-list on
/// a resolved input means a producer OUTSIDE this CB (a host step, or the start
/// gate) — those events gate `clEnqueueCommandBufferKHR`, not any CB-internal
/// command. A producer INSIDE the CB deposited EMPTY deps (its ordering is the
/// sync points), so this is a no-op for internal edges. Idempotent + cheap.
pub fn cb_collect_external(ext: &Mutex<Deps>, deps: &Deps) {
    if !deps.is_empty() {
        lock_unpoisoned(ext).extend(deps.iter().cloned());
    }
}

/// The precise-invalidation ORIGIN SET of a leaf's buffer/image arg: which slot
/// cells a `mutate_bind` of would make a CB that baked this buffer stale. Two
/// contributions, unioned:
///
/// - `slot_id` — the arg IS a `slot!(Tag)` position (in-place fill/copy/kernel of a
///   bound or fed slot). The slot cell is itself an origin (mutating the tag rebinds
///   the cell → the baked `cl_mem` changes).
/// - `pipe_id`'s ambient reach — the arg is fed by an upstream pipe (a direct
///   `Input::Pipe` OR a `FedByPipe` slot). The upstream producer's reach (the slots
///   THAT buffer threads back to) propagated onto the pipe's cell at record time.
///
/// A `FedByPipe` slot has BOTH: its own cell (the re-bind target) AND the upstream
/// reach — so both are collected. A plain concrete arg has neither → empty set
/// (no slot can invalidate a CB over a concrete buffer). This is the per-arg-local
/// forward map the pipe-reachability substrate composes along the walk.
pub(crate) fn cb_origins_of(
    ec: &ExecutionContext<'_>,
    slot_id: Option<usize>,
    pipe_id: Option<usize>,
) -> std::collections::BTreeSet<usize> {
    let mut origins = ec.cb_reach_of(pipe_id);
    if let Some(s) = slot_id {
        origins.insert(s);
    }
    origins
}

/// **CB Build-mode prologue for a single-input → single-output COMMAND leaf** — the
/// shared bookkeeping every recording leaf (`fill`, SVM fill, a single-output
/// kernel arg) does before emitting its one `clCommand*KHR`. Factored into ONE
/// place because it is the precise-invalidation substrate (the subtle, once-buggy
/// cross-seam reach propagation) and must stay identical across leaves:
///
/// 1. thread this span's ENTRY deps (producers outside the CB) into `ext`;
/// 2. compute the input's sync-point wait-list (returned, for the command);
/// 3. resolve the input's slot origins ([`cb_origins_of`]) and `note_slot` each
///    into the live CB (so a `mutate_bind` of that slot invalidates this CB);
/// 4. propagate those origins onto the OUTPUT cell's reach, so a downstream leaf —
///    even across a host seam — traces this buffer back to the mutable slot.
///
/// The caller then records its command with the returned waits and deposits the
/// (lent) buffer with empty deps. See [`cb_forward_reach`] for the passthrough
/// (no-command) twin.
pub(crate) fn cb_leaf_build(
    ec: &ExecutionContext<'_>,
    builder: &crate::record::CbBuilder,
    ext: &Mutex<Deps>,
    deps: &Deps,
    in_slot: Option<usize>,
    in_pipe: Option<usize>,
    out_cell: usize,
) -> crate::exec_ctx::SyncPoints {
    cb_collect_external(ext, deps);
    let waits = ec.sp_lookup(in_pipe);
    let origins = cb_origins_of(ec, in_slot, in_pipe);
    for &s in &origins {
        builder.note_slot(s);
    }
    ec.cb_reach_extend(out_cell, origins);
    waits
}

/// **CB reach propagation for a PASSTHROUGH node** (no recorded command) — the
/// structural twin of [`cb_leaf_build`] for `forward` / `lift` / `arced` /
/// `arc_split` / bundles / host seams. A passthrough emits nothing into the CB, so
/// it neither `note_slot`s nor gathers waits; it only forwards the reach of its
/// source cell onto `out_cell` so the precise-invalidation origin trail survives
/// the alias. `src_cell` is the producer whose reach to inherit (a slot origin, if
/// any, is unioned in).
pub(crate) fn cb_forward_reach(
    ec: &ExecutionContext<'_>,
    in_slot: Option<usize>,
    src_cell: Option<usize>,
    out_cell: usize,
) {
    ec.cb_reach_extend(out_cell, cb_origins_of(ec, in_slot, src_cell));
}

/// **The full passthrough Build-arm forward**: alias the source cell's sync points
/// onto `out_cell` AND forward its precise-invalidation reach. A structural
/// passthrough (`forward` / `arced` / `arc_split` / bundle branch) that aliases one
/// producer cell onto its output MUST do BOTH, always together:
///
/// - **sync-point alias** (`sp_lookup` → `sp_register`): a downstream CB consumer of
///   this passthrough's output cell reads its wait-list from that cell's sync points;
///   miss it and the consumer loses its in-CB ordering.
/// - **reach forward** ([`cb_forward_reach`]): carry the origin trail so a
///   `mutate_bind` of an upstream slot still invalidates a CB that baked this buffer;
///   miss it and slot invalidation silently no-ops.
///
/// Forgetting either half is a distinct latent bug, so they live in one call. (The
/// reach-only cases — a chain-ENTRY `lift` with no upstream sync points, or a
/// mid-graph host seam — call [`cb_forward_reach`] directly instead.)
pub(crate) fn cb_forward_passthrough(
    ec: &ExecutionContext<'_>,
    in_slot: Option<usize>,
    src_cell: Option<usize>,
    out_cell: usize,
) {
    ec.sp_register(out_cell, ec.sp_lookup(src_cell));
    cb_forward_reach(ec, in_slot, src_cell, out_cell);
}

/// Whether a graph is CB-eligible RIGHT NOW: the platform advertises
/// `cl_khr_command_buffer` and the graph has no host seam. (The all-`cl_mem`
/// requirement is enforced dynamically — an SVM command marks the live builder
/// ineligible via [`CbBuilder`], and the boundary discards it — so SVM graphs
/// transparently fall back to per-op execute without a static type gate here.)
pub(crate) fn cb_graph_eligible<O: DeviceOp + ?Sized>(op: &O, ec: &ExecutionContext<'_>) -> bool {
    ec.context().has_cl_khr_command_buffer() && op.cb_addable()
}

/// The **CB BOUNDARY protocol** (design v2): gather a maximal CB-eligible subtree
/// `op` as ONE command buffer, homing it in `op`'s [`cb_cache`](DeviceOp::cb_cache),
/// and return its terminal [`Checkouts`](DeviceOp::Checkouts) + the ONE completion
/// event (as [`Deps`]) the caller waits on. Used by the terminal for a whole
/// all-device graph AND by a host seam for each device sub-subtree — both are "run
/// this seam-free subtree as a CB".
///
/// - **replay** (a valid homed CB for this queue): re-walk in
///   [`LendOnly`](CbWalk::LendOnly) (lend every buffer, build Checkouts, add/enqueue
///   NOTHING), then enqueue the cached CB once with the run's EXTERNAL events.
/// - **build** (no/stale CB): re-walk in [`Build`](CbWalk::Build) (lend + add each
///   command → sync points), [`finalize`](crate::record::CbBuilder::finalize) into
///   the cache, enqueue with external events.
/// - **ineligible fallback** (an SVM command marked the live build ineligible):
///   the Build pass ENQUEUED NOTHING real (leaves only added to the now-discarded
///   CB), so drop the lent Checkouts (rehome), reclaim, and re-run `op` in the
///   normal per-op [`Off`](CbWalk::Off) path — correct results, no double-execute.
///
/// The event↔sync-point boundary: CB-internal ordering is the sync points added in
/// Build; the ONLY `cl_event`s are `ext` (producers OUTSIDE this CB) applied at
/// `clEnqueueCommandBufferKHR`, and the returned completion event handed UP.
pub(crate) fn cb_boundary_gather<O>(
    op: &O,
    ec: &ExecutionContext<'_>,
    mode: ExecMode,
) -> Result<(O::Checkouts, Deps)>
where
    O: DeviceOp,
    O::Output: Send + 'static,
    O::Checkouts: FromCheckout<O::Output>,
{
    use crate::Launcher;
    let queue = ec.cl_queue().get();
    let cache = op
        .cb_cache()
        .expect("cb_boundary_gather: boundary node must carry a cb_cache");

    // The EXTERNAL cl_event accumulator for THIS command buffer, on this frame.
    let ext: Mutex<Deps> = Mutex::new(Deps::new());
    // The span-closed latch (see `CbWalk::Build::closed`), owned by this frame.
    let closed = std::sync::atomic::AtomicBool::new(false);

    // Is there a valid cached CB for this queue? (Replay fast-path.)
    let cached = cb_lookup_cached(cache, queue);

    if let Some(cb) = cached {
        // REPLAY: re-walk lend-only (materialize buffers + build Checkouts), then
        // one clEnqueueCommandBufferKHR with the run's external events.
        let build_ec = ec.with_cb(CbWalk::LendOnly {
            ext: &ext,
            cache,
            closed: &closed,
        });
        let (checkouts, internal) = op.gather_checkouts(&build_ec, mode)?;
        if closed.load(std::sync::atomic::Ordering::SeqCst) {
            // The span CLOSED mid-walk (an interior seam / transfer): the close point
            // enqueued the cached span CB before the boundary, and the tail ran in
            // `Off`. `internal` carries the tail's completion events. Do NOT enqueue
            // again (that would be `CL_INVALID_OPERATION`).
            return Ok((checkouts, internal));
        }
        let waits = drain_ext(&ext);
        let event = cb.enqueue(&waits)?;
        return Ok((checkouts, single_dep(event)));
    }

    // BUILD: create a live CB, re-walk adding each command, finalize, home, enqueue.
    let platform = ec.context().device().platform().raw_id();
    let Some(builder) = crate::record::CbBuilder::new(platform, queue) else {
        // Extension unreachable after all — fall back to the normal path.
        return op.gather_checkouts(ec, mode);
    };
    let build_ec = ec.with_cb(CbWalk::Build {
        builder: &builder,
        ext: &ext,
        cache,
        closed: &closed,
    });
    let (checkouts, internal) = op.gather_checkouts(&build_ec, mode)?;

    if builder.is_finalized() {
        // The span CLOSED mid-walk (an interior seam / transfer): the close point
        // already sealed + enqueued the CB and restamped `source`'s pipes; the tail
        // ran in `Off`, so `internal` is the correct downstream wait-list.
        return Ok((checkouts, internal));
    }

    if !builder.is_eligible() {
        // An SVM (or otherwise un-addable) command: the build pass enqueued NOTHING
        // real, so drop the lent Checkouts (rehome every buffer to its cell),
        // reclaim mid-graph intermediates, and re-run the normal per-op path.
        drop(checkouts);
        op.reclaim_undelivered();
        return op.gather_checkouts(ec, mode);
    }

    if builder.recorded() == 0 {
        // EMPTY-CB guard: a span of pure structural passthroughs (a bare `Pipe`
        // aliasing an upstream, `lift`ed device cells) added zero commands.
        // Finalizing + enqueuing such a CB is pure event-sync overhead. Discard it
        // (its `Drop` releases the handle) and hand the Checkouts back gated on the
        // deps the Build-pass gather already produced — a bare `Pipe` branch carries
        // its UPSTREAM producer's completion event (the earlier span's CB event,
        // stamped onto the pipe by `cb_restamp`) in `internal`, NOT in `ext` (which
        // only entry-leaf resolves populate). Returning `internal` keeps the seam /
        // downstream waiting on the real upstream work.
        return Ok((checkouts, internal));
    }

    let Some(finalized) = builder.finalize() else {
        // Finalize failed: same clean fallback as ineligible.
        drop(checkouts);
        op.reclaim_undelivered();
        return op.gather_checkouts(ec, mode);
    };
    let finalized = Arc::new(finalized);
    // HOME the CB in the boundary node's OWN cache (drops with the graph).
    *lock_unpoisoned(cache) = Some(Arc::clone(&finalized));

    let waits = drain_ext(&ext);
    let event = finalized.enqueue(&waits)?;
    Ok((checkouts, single_dep(event)))
}

/// Run a combinator's CHILD, opening a mid-graph command buffer for it when
/// appropriate (design v2). The single decision point every combinator
/// (`AndThen`, bundles, the host seam) routes a child `execute` through:
///
/// - in [`Off`](CbWalk::Off) mode, if the platform supports CBs and the child's
///   WHOLE subtree is [`cb_addable`](DeviceOp::cb_addable), the child is a maximal
///   seam-free span → run it as ONE command buffer via [`cb_boundary_execute`]
///   (which fills the child's pipes + stamps the CB completion event);
/// - otherwise (already inside a CB → `Build`/`LendOnly`; or the child contains a
///   seam / transfer → not addable; or no extension) → a plain `execute`, which
///   forwards the mode and lets any addable sub-span deeper down open its own CB.
///
/// This is what makes host-seam graphs segment into sub-tree CBs: the seam feeds
/// its device source through here in `Off`, so the source span becomes its own CB.
/// Whether `child` should be run as ONE command buffer at THIS boundary — the shared
/// eligibility gate of [`cb_exec_child`] / [`cb_gather_child`]. True iff we are in
/// [`Off`](CbWalk::Off) (not already inside a CB), the child homes a CB, its whole
/// subtree is CB-eligible ([`cb_graph_eligible`]), and it holds `>= 2` commands.
///
/// The `>= 2` cutoff: a single-command span runs per-op — a CB holding one command is
/// pure create/finalize/enqueue overhead with no batching benefit. Exact here (the
/// boundaried span IS the whole child subtree).
fn cb_should_open_boundary<O: DeviceOp>(child: &O, ec: &ExecutionContext<'_>) -> bool {
    matches!(ec.cb(), CbWalk::Off)
        && child.cb_cache().is_some()
        && cb_graph_eligible(child, ec)
        && child.cbable_weight() >= 2
}

/// The replay fast-path lookup shared by both boundary functions
/// ([`cb_boundary_execute`] / [`cb_boundary_gather`]): a still-valid cached
/// [`FinalizedCb`] for THIS queue, or `None` (build a fresh one). A CB is queue-bound
/// — a cache homed against a different queue is stale and must be rebuilt.
fn cb_lookup_cached(
    cache: &CbCache,
    queue: opencl3::types::cl_command_queue,
) -> Option<Arc<crate::record::FinalizedCb>> {
    let guard = lock_unpoisoned(cache);
    match guard.as_ref() {
        Some(cb) if cb.queue() == queue => Some(Arc::clone(cb)),
        _ => None,
    }
}

pub(crate) fn cb_exec_child<O: DeviceOp>(
    child: &O,
    ec: &ExecutionContext<'_>,
    mode: ExecMode,
) -> Result<()> {
    if cb_span_closed(ec) {
        // A deeper close already sealed this frame's span CB → run the child in `Off`
        // (it opens its own fresh boundary if eligible). Never add to a sealed CB.
        return cb_exec_child(child, &ec.with_cb(CbWalk::Off), mode);
    }
    if cb_should_open_boundary(child, ec) {
        cb_boundary_execute(child, ec, mode)
    } else {
        child.execute(ec, mode)
    }
}

/// The gather-position analog of [`cb_exec_child`] — used when a combinator
/// gathers its TAIL (`AndThen::next`, the terminal-shaped child). If the tail's
/// whole subtree is a maximal seam-free span (and we're in `Off`), run it as ONE
/// command buffer via [`cb_boundary_gather`] (build Checkouts + return the CB
/// completion event as the deps); otherwise a plain `gather_checkouts` (which
/// recurses, letting deeper addable spans open their own CBs).
pub(crate) fn cb_gather_child<O>(
    child: &O,
    ec: &ExecutionContext<'_>,
    mode: ExecMode,
) -> Result<(O::Checkouts, Deps)>
where
    O: DeviceOp,
    O::Output: Send + 'static,
    O::Checkouts: FromCheckout<O::Output>,
{
    if cb_span_closed(ec) {
        // A deeper close already sealed this frame's span CB → gather the child in
        // `Off` (opens its own fresh boundary if eligible).
        return cb_gather_child(child, &ec.with_cb(CbWalk::Off), mode);
    }
    if cb_should_open_boundary(child, ec) {
        cb_boundary_gather(child, ec, mode)
    } else {
        child.gather_checkouts(ec, mode)
    }
}

/// **Finalize-at-close** (design v2, maximal-span batching). Called at the exact
/// point a maximal seam-free span CLOSES — an [`AndThen`] executing INSIDE a command
/// buffer (`Build`/`LendOnly`) whose `next` cannot continue the span (a host seam /
/// transfer). This SEALS + ENQUEUES the span's command buffer and returns its single
/// completion event, so the caller can [`cb_restamp`](DeviceOp::cb_restamp) it onto
/// the span's output pipes BEFORE the seam (run next, in `Off`) reads them — the
/// whole reason singletons were the prior floor: at boundary-return the seam had
/// already mapped the outputs with no wait (the race).
///
/// - **Build**: [`finalize`](crate::record::CbBuilder::finalize) the live builder
///   (idempotent — the boundary-return frame sees it already sealed and skips), home
///   the [`FinalizedCb`](crate::record::FinalizedCb) in the span head's `cache`
///   (threaded through [`CbWalk`]), enqueue with the span's external deps.
/// - **LendOnly** (replay): read the cached CB from `cache` and enqueue it.
///
/// Returns `Ok(None)` when there is no CB to close (extension absent, or the span
/// recorded zero commands — the empty-CB case, where the caller keeps the pipes as
/// the Build pass left them). `Off` is unreachable (only called from a Build/LendOnly
/// AndThen).
pub(crate) fn cb_close_span(ec: &ExecutionContext<'_>) -> Result<Option<Dep>> {
    use crate::Launcher;
    use std::sync::atomic::Ordering;
    let (ext, cache, closed, is_build): (
        &Mutex<Deps>,
        &CbCache,
        &std::sync::atomic::AtomicBool,
        bool,
    ) = match ec.cb() {
        CbWalk::Build {
            ext, cache, closed, ..
        } => (ext, cache, closed, true),
        CbWalk::LendOnly {
            ext, cache, closed, ..
        } => (ext, cache, closed, false),
        CbWalk::Off => return Ok(None),
    };
    let queue = ec.cl_queue().get();

    // IDEMPOTENCY: a span closes EXACTLY once. The close point is deep inside a source
    // subtree, so several ancestor `AndThen`s may reach `cb_close_before_seam` after
    // it; only the first actually seals + enqueues. In Build, `finalize` is itself
    // idempotent, but the LendOnly (replay) path re-reads the cache and would
    // re-enqueue the SAME CB (`CL_INVALID_OPERATION`) — this latch guards BOTH.
    if closed.swap(true, Ordering::SeqCst) {
        return Ok(None);
    }

    // REPLAY: enqueue the cached span CB (read from the head's cache).
    if !is_build {
        let cached: Option<Arc<crate::record::FinalizedCb>> = {
            let g = lock_unpoisoned(cache);
            g.as_ref().filter(|cb| cb.queue() == queue).map(Arc::clone)
        };
        let Some(cb) = cached else {
            // Shouldn't happen (replay implies a homed CB), but stay safe: no CB to
            // close → caller keeps the Build-pass pipes.
            return Ok(None);
        };
        let waits = drain_ext(ext);
        let event = cb.enqueue(&waits)?;
        return Ok(Some(wrap_event(event)));
    }

    // BUILD: seal the live builder (idempotent), home it, enqueue.
    let CbWalk::Build { builder, .. } = ec.cb() else {
        unreachable!("is_build implies Build");
    };
    if builder.recorded() == 0 {
        // Empty span (pure passthroughs): nothing to seal. The caller keeps the
        // Build-pass pipes (their payloads already carry the upstream events).
        return Ok(None);
    }
    let Some(finalized) = builder.finalize() else {
        // Ineligible / finalize failure: no CB. The span's leaves added to a builder
        // that will be discarded → the caller must fall back. Signalled as `None`;
        // the AndThen close handles the reclaim + Off re-run.
        return Ok(None);
    };
    let finalized = Arc::new(finalized);
    *lock_unpoisoned(cache) = Some(Arc::clone(&finalized));
    let waits = drain_ext(ext);
    let event = finalized.enqueue(&waits)?;
    Ok(Some(wrap_event(event)))
}

/// Drain a CB's external dep accumulator into a raw `cl_event` wait-list for
/// `clEnqueueCommandBufferKHR`.
fn drain_ext(ext: &Mutex<Deps>) -> Vec<crate::cl_event> {
    deps_to_wait_list(&lock_unpoisoned(ext))
}

/// Whether `op` (an [`AndThen`]) should OPEN a maximal seam-free span here (design
/// v2, finalize-at-close). True iff we are in [`Off`](CbWalk::Off), the platform has
/// the extension, `op` is CB-capable, its leading source is addable
/// ([`cb_spine_head_addable`](DeviceOp::cb_spine_head_addable)), yet `op` is NOT
/// wholly addable — i.e. it CONTAINS a host seam further down `next`. The wholly-
/// addable case (`cb_addable()`) is the existing whole-subtree boundary that the
/// PARENT's `cb_exec_child`/`cb_gather_child` handles; only the interior-seam
/// spine-head opens its span from inside its own `execute`/`gather`.
pub(crate) fn cb_should_open_span<O: DeviceOp>(op: &O, ec: &ExecutionContext<'_>) -> bool {
    matches!(ec.cb(), CbWalk::Off)
        && op.cb_cache().is_some()
        && ec.context().has_cl_khr_command_buffer()
        && op.cb_spine_head_addable()
        && !op.cb_addable()
        // >= 2: don't open a span CB that would record a single command (per-op is
        // cheaper). NOTE: unlike the other boundary sites, here the recorded span is
        // the maximal seam-free PREFIX of `op`, not its whole subtree — so
        // `cbable_weight()` (the whole-subtree count) is an OVER-approximation: a
        // subtree with a 1-command prefix + a post-seam command reports 2 but would
        // record a 1-command CB. That is the same coarseness the old
        // `cb_records_command` (>= 1) had here; the runtime empty-CB guard still
        // discards a 0-command CB, and a rare 1-command interior span is a minor
        // missed optimization, not a correctness issue. Precise prefix-weight is a
        // follow-up if it ever matters.
        && op.cbable_weight() >= 2
}

/// Whether this walk position's span has already CLOSED (its latch is set — see
/// [`CbWalk::Build`]`::closed`). Once closed, all remaining work under the same
/// boundary frame must run in [`Off`](CbWalk::Off): a deeper close already sealed +
/// enqueued the span CB, so no ancestor may keep adding to it. `Off` positions have
/// no span → never closed.
pub(crate) fn cb_span_closed(ec: &ExecutionContext<'_>) -> bool {
    use std::sync::atomic::Ordering;
    match ec.cb() {
        CbWalk::Build { closed, .. } | CbWalk::LendOnly { closed, .. } => {
            closed.load(Ordering::SeqCst)
        }
        CbWalk::Off => false,
    }
}

/// The **span-close decision** for an [`AndThen`] running INSIDE a command buffer.
/// If `next` cannot continue the maximal seam-free span (a host seam / transfer —
/// [`cb_spine_head_addable`](DeviceOp::cb_spine_head_addable) is false) and we are in
/// `Build`/`LendOnly`, this SEALS + ENQUEUES the span CB via [`cb_close_span`] and
/// [`cb_restamp`](DeviceOp::cb_restamp)s its completion event onto `source`'s output
/// pipes — `source` is the last span node before the seam, and its pipes are exactly
/// what `next` (the seam) is about to read. Returns `true` iff the span closed here
/// (the caller must then run `next` in [`Off`](CbWalk::Off)); `false` to continue the
/// span (run `next` in the same Build/LendOnly `ec`).
///
/// Idempotent across ancestor `AndThen`s: once a deeper close sealed the builder, a
/// higher `AndThen` whose `next` also can't continue re-enters here, gets `None` from
/// [`cb_close_span`] (already finalized), skips the restamp, and still runs its `next`
/// in `Off` — so no post-seam op joins the dead builder.
pub(crate) fn cb_close_before_seam<S, U>(
    source: &S,
    next: &U,
    ec: &ExecutionContext<'_>,
) -> Result<bool>
where
    S: DeviceOp,
    U: DeviceOp,
{
    match ec.cb() {
        CbWalk::Build { .. } | CbWalk::LendOnly { .. } if !next.cb_spine_head_addable() => {
            if let Some(ev) = cb_close_span(ec)? {
                source.cb_restamp(&Deps::from([ev]));
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// The **execute-position CB boundary** (design v2) — the mid-graph analog of
/// [`cb_boundary_gather`]. When a combinator descending in [`Off`](CbWalk::Off)
/// mode reaches a CB-eligible child subtree (e.g. a device span under a host seam,
/// or the source of an `and_then` whose sibling is a seam), that child is a
/// boundary: run it as ONE command buffer, home it in the child's `cb_cache`, then
/// STAMP the CB's single completion event onto the child's output pipe(s) so a
/// downstream (non-CB) consumer waits on the whole CB (the event↔sync-point
/// boundary in reverse — markers INSIDE, one cl_event OUT).
///
/// Fills pipes only (returns `()`), like [`DeviceOp::execute`]; the caller
/// (`AndThen`/bundle) then reads the stamped pipes normally. Falls back to a plain
/// `execute` if the extension is unreachable or the build is ineligible (SVM).
pub(crate) fn cb_boundary_execute<O>(
    op: &O,
    ec: &ExecutionContext<'_>,
    mode: ExecMode,
) -> Result<()>
where
    O: DeviceOp,
{
    use crate::Launcher;
    let queue = ec.cl_queue().get();
    let cache = op
        .cb_cache()
        .expect("cb_boundary_execute: boundary node must carry a cb_cache");
    let ext: Mutex<Deps> = Mutex::new(Deps::new());
    let closed = std::sync::atomic::AtomicBool::new(false);

    // Replay fast-path: a valid cached CB for this queue.
    let cached = cb_lookup_cached(cache, queue);

    if let Some(cb) = cached {
        let build_ec = ec.with_cb(CbWalk::LendOnly {
            ext: &ext,
            cache,
            closed: &closed,
        });
        op.execute(&build_ec, mode)?;
        if closed.load(std::sync::atomic::Ordering::SeqCst) {
            // The span CLOSED mid-walk (an interior seam / transfer): the close point
            // enqueued the cached span CB before the boundary and restamped
            // `source`'s pipes; the tail ran in `Off`. Nothing to do at return.
            return Ok(());
        }
        let waits = drain_ext(&ext);
        let event = cb.enqueue(&waits)?;
        op.cb_restamp(&single_dep(event));
        return Ok(());
    }

    let platform = ec.context().device().platform().raw_id();
    let Some(builder) = crate::record::CbBuilder::new(platform, queue) else {
        return op.execute(ec, mode);
    };
    let build_ec = ec.with_cb(CbWalk::Build {
        builder: &builder,
        ext: &ext,
        cache,
        closed: &closed,
    });
    op.execute(&build_ec, mode)?;

    if builder.is_finalized() {
        // The span CLOSED mid-walk (an interior seam / transfer): the close point
        // already sealed + enqueued the CB and restamped `source`'s pipes; the tail
        // ran in `Off`. DONE.
        return Ok(());
    }

    if !builder.is_eligible() {
        // The build enqueued nothing; the pipes hold the lent buffers with empty
        // deps. Reclaim (rehome) them and re-run the normal per-op path so the
        // buffers are re-lent + real device work runs.
        op.reclaim_undelivered();
        return op.execute(ec, mode);
    }

    if builder.recorded() == 0 {
        // EMPTY-CB guard (mid-graph): a pure-passthrough span recorded no commands.
        // Discard the CB with NO finalize / enqueue. Leave the output pipes exactly
        // as the Build pass left them — a bare `Pipe` passthrough already carries its
        // upstream producer's completion event in its payload; only the ext deps (an
        // entry leaf that crossed into this would-be CB) need stamping on top, so a
        // downstream consumer waits on the real upstream work. `ext` is usually empty
        // for a pure-passthrough span (nothing resolved a cross-boundary input).
        let evs: Deps = std::mem::take(&mut lock_unpoisoned(&ext));
        if !evs.is_empty() {
            op.cb_restamp(&evs);
        }
        return Ok(());
    }

    let Some(finalized) = builder.finalize() else {
        op.reclaim_undelivered();
        return op.execute(ec, mode);
    };
    let finalized = Arc::new(finalized);
    *lock_unpoisoned(cache) = Some(Arc::clone(&finalized));
    let waits = drain_ext(&ext);
    let event = finalized.enqueue(&waits)?;
    op.cb_restamp(&single_dep(event));
    Ok(())
}
