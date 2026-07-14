//! Structural combinators for the eager graph — the nodes that COMPOSE other ops
//! rather than do device work: `AndThen` (source->next), `Value`/`Lift` (lift a
//! host value / owned resource), `Forward` (identity/select), `DeviceDynOp`
//! (type-erased single-output), `Arced`/`ArcSplit` (share one output read-only),
//! `Bundle*`/`FanOut` (N independent branches). Each delegates execution/CB/slots
//! to its children. See `ARCHITECTURE.md`.

use super::*;

// ── AndThen: source then next; next eagerly built over source's pipe ───

/// If `src_pipe` still holds a value (the `and_then` closure discarded the
/// source's handle), merge its stranded events into `out_pipe`'s deps so the
/// whole chain's events still gate the terminal. See `AndThen::collect`. No-op
/// when the source pipe was consumed by `next` (the normal case). The discarded
/// source value drops.
///
/// Both pipes are `Option` (the [`output_pipe`](DeviceOp::output_pipe) shape): a
/// **multi-output** source or `next` has NO single storage pipe (`None`), so its
/// storage is per-branch element pipes and there is nothing to thread through
/// here — the multi-output source's completion events ride its own per-branch
/// gather (`collect`/`gather_checkouts`), and a multi-output `next`'s deps are
/// threaded by its own override. A `None` on either side is therefore a no-op —
/// byte-identical to the pre-Option behaviour, where a multi-output op returned a
/// fresh never-filled `Pipe::new()` that `take()` always drained as empty.
fn thread_orphaned_source_deps<A, B>(src_pipe: &Option<Pipe<A>>, out_pipe: &Option<Pipe<B>>) {
    // Multi-output source (`None`) → no single pipe to thread; or the source pipe
    // was consumed by `next` (the normal case) → nothing to thread.
    let Some((_discarded, src_deps)) = src_pipe.as_ref().and_then(|p| p.take()) else {
        return;
    };
    // Merge the stranded source events into the out pipe's deps. If `out_pipe` is
    // absent/empty (a multi-output `next` whose storage is its element pipes, not
    // `output_pipe`), `execute` isn't the gather path — `collect` handles
    // orphaned deps for that case directly, so this is a no-op here.
    if let Some(out_pipe) = out_pipe.as_ref()
        && let Some((v, mut deps)) = out_pipe.take()
    {
        deps.extend(src_deps);
        out_pipe.put(v, deps);
    }
}

/// Sequential composition node. Holds the source op and the **already-built**
/// downstream op (which reads the source's output via a [`Pipe`]). No `FnOnce`.
pub struct AndThen<S, U> {
    // pub(crate): the `and_then` builder (a `DeviceOpExt` method in eager.rs)
    // constructs this by struct literal.
    pub(crate) source: S,
    pub(crate) next: U,
    /// Design-v2 CB home. When this `AndThen` is the outermost CB-eligible node of
    /// a seam-free subtree (e.g. CG's root chain), the whole subtree is ONE command
    /// buffer homed here and replayed across syncs. See [`CbCache`].
    pub(crate) cb_cache: CbCache,
    /// Precomputed `source.cbable_weight() + next.cbable_weight()` (static under
    /// mutate — see [`DeviceOp::cbable_weight`]). Set once by [`and_then`].
    pub(crate) cbable_weight: usize,
}

impl<S, U> DeviceOp for AndThen<S, U>
where
    S: DeviceOp,
    U: DeviceOp,
{
    type Output = U::Output;
    // The chain's downstream handle is the tail op's handle.
    type Handle = U::Handle;
    // The chain's terminal checkout shape is the tail op's: a multi-output tail
    // (bundle*, arc_split, CopyTo pair) yields its tuple/array of `Checkout`s.
    type Checkouts = U::Checkouts;

    fn output_pipe(&self) -> Option<Pipe<U::Output>> {
        self.next.output_pipe()
    }

    fn handle(&self) -> U::Handle {
        self.next.handle()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // OPEN a maximal span at SELF (finalize-at-close): a `!cb_addable` chain whose
        // leading source is addable (`cb_spine_head_addable`) but which CONTAINS a
        // host seam is the HEAD of a maximal seam-free span. Open ONE CB here and
        // re-enter in `Build` (or `LendOnly` on replay), where the source/next
        // dispatch below extends the CB across the addable prefix and CLOSES it at
        // the seam. A FULLY-addable chain (no interior seam) is caught by the parent's
        // `cb_exec_child` as a whole-subtree boundary and never reaches here in Off.
        if cb_should_open_span(self, ec) {
            return cb_boundary_execute(self, ec, mode);
        }
        // Source is always upstream → must pipeline (its output feeds `next`).
        let src_pipe = self.source.output_pipe();
        let out_pipe = self.next.output_pipe();
        cb_exec_child(&self.source, ec, ExecMode::Pipelined)?;
        // CLOSE the span at the seam (Build/LendOnly + next can't continue): seal +
        // enqueue the CB and restamp `source`'s pipes with its completion event, then
        // run `next` in `Off`. Otherwise forward normally (continue the span in Build,
        // or open a fresh boundary in Off).
        if cb_close_before_seam(&self.source, &self.next, ec)? {
            self.next.execute(&ec.with_cb(CbWalk::Off), mode)?;
        } else {
            cb_exec_child(&self.next, ec, mode)?;
        }
        thread_orphaned_source_deps(&src_pipe, &out_pipe);
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(U::Output, Deps)>
    where
        Self: Sized,
    {
        // Delegate the gather to the tail op so a multi-output `next` (bundle*,
        // arc_split, CopyTo pair) runs its *overridden* `collect`
        // (scatter-then-reconstruct over its per-element pipes) rather than the
        // default single-pipe drain — whose `output_pipe` it never fills. The
        // source pipelines; only the tail observes the terminal `mode`.
        let src_pipe = self.source.output_pipe();
        cb_exec_child(&self.source, ec, ExecMode::Pipelined)?;
        // CLOSE the span at the seam: seal + enqueue + restamp `source`, then collect
        // `next` in `Off`. Otherwise collect `next` normally (continue in Build).
        let (value, mut deps) = if cb_close_before_seam(&self.source, &self.next, ec)? {
            self.next.collect(&ec.with_cb(CbWalk::Off), mode)?
        } else {
            self.next.collect(ec, mode)?
        };
        // ORPHANED DEPS: if the `and_then` closure discarded the source's handle
        // (e.g. `.and_then(|_buf| value(0))`), the source's value + events are
        // still sitting un-taken in its output pipe. Those events MUST still gate
        // the terminal — most critically when the source is an `and_then_host`
        // whose worker thread signals completion via a user event; without this,
        // `sync` would return before the worker ran. Thread them in. (The old
        // closure layer got this for free by passing `prior_evts` into
        // `next.execute`.) The discarded value drops here — the closure didn't
        // want it; its `cl_mem` is retained by any in-flight unmap until done.
        if let Some((_discarded, src_deps)) = src_pipe.and_then(|p| p.take()) {
            deps.extend(src_deps);
        }
        Ok((value, deps))
    }

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(U::Output, Deps, Option<BoxedHome<U::Output>>)>
    where
        Self: Sized,
        U::Output: Send + 'static,
    {
        // Mirror `collect`, but preserve the tail's return home so an
        // `and_then`-terminated chain nested as a bundle branch (e.g.
        // `upload(buf).and_then(|b| fill(b, v))`) re-arms its caller buffer. The
        // home is the tail's — the source pipelines and its orphaned deps thread
        // in exactly as in `collect`.
        let src_pipe = self.source.output_pipe();
        cb_exec_child(&self.source, ec, ExecMode::Pipelined)?;
        let (value, mut deps, home) = if cb_close_before_seam(&self.source, &self.next, ec)? {
            self.next.collect_home(&ec.with_cb(CbWalk::Off), mode)?
        } else {
            self.next.collect_home(ec, mode)?
        };
        if let Some((_discarded, src_deps)) = src_pipe.and_then(|p| p.take()) {
            deps.extend(src_deps);
        }
        Ok((value, deps, home))
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)>
    where
        Self: Sized,
        Self::Output: Send + 'static,
        Self::Checkouts: FromCheckout<Self::Output>,
    {
        // OPEN a maximal span at SELF (see `execute`): an interior-seam spine-head
        // reached in gather position (the terminal drives the root through here for a
        // host-seam graph). Re-enters in `Build`/`LendOnly`; the close below batches
        // the addable prefix and seals at the seam.
        if cb_should_open_span(self, ec) {
            return cb_boundary_gather(self, ec, mode);
        }
        // Mirror `collect`: delegate to the tail so a multi-output `next` builds
        // its per-element `Checkout` tuple via its OWN `gather_checkouts` override
        // (the default single-pipe drain reads `output_pipe`, which a multi-output
        // op never fills → "op produced no output"). Source pipelines; tail takes
        // the terminal `mode`. Same orphaned-source-deps threading as `collect`.
        let src_pipe = self.source.output_pipe();
        cb_exec_child(&self.source, ec, ExecMode::Pipelined)?;
        // CLOSE the span at the seam: seal + enqueue + restamp `source`, then gather
        // `next` in `Off`. Otherwise gather `next` normally (continue in Build, or
        // open a fresh boundary in Off).
        let (checkouts, mut deps) = if cb_close_before_seam(&self.source, &self.next, ec)? {
            self.next.gather_checkouts(&ec.with_cb(CbWalk::Off), mode)?
        } else {
            cb_gather_child(&self.next, ec, mode)?
        };
        if let Some((_discarded, src_deps)) = src_pipe.and_then(|p| p.take()) {
            deps.extend(src_deps);
        }
        Ok((checkouts, deps))
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        self.next.describe(out);
    }

    fn node_label(&self) -> String {
        "and_then(source→next)".to_string()
    }

    fn dump_graph(&self, depth: usize, out: &mut Vec<GraphNode>) {
        // Push a node for the chain itself: its OUTPUT is the tail's output pipe
        // (its `output_pipe` delegates to `next`), its flags are the whole
        // subtree's. Then recurse SOURCE FIRST, then NEXT — the same execution
        // order `execute`/`describe` use, so the flattened list reads top-down in
        // dependency order and the depth column shows the source/next nesting.
        let out_cells = self
            .output_pipe()
            .map(|p| p.cell_id())
            .into_iter()
            .collect();
        out.push(GraphNode {
            depth,
            name: self.node_label(),
            out_cells,
            in_cells: Vec::new(),
            cb_addable: self.cb_addable(),
            seam: self.contains_host_seam(),
        });
        self.source.dump_graph(depth + 1, out);
        self.next.dump_graph(depth + 1, out);
    }

    fn bind_slots(&self, binder: &mut SlotBinder) {
        // Walk source then next (execution order). A move-only binder stops once its
        // single value has landed (one `bind` binds one cell); a fan-out binder
        // (clone-able value — scalar / launch / `Arc`) NEVER stops, so one `bind`
        // fills EVERY matching cell across both subtrees (the shared-slot path).
        self.source.bind_slots(binder);
        if !binder.is_fanout() && binder.is_consumed() {
            return;
        }
        self.next.bind_slots(binder);
    }

    fn check_ready(&self) -> Result<()> {
        // Recurse source then next — the SAME traversal `execute`/`describe`/
        // `bind_slots` use, so the pre-pass covers exactly the inputs `execute`
        // resolves. Fail fast on the first unsatisfiable input.
        self.source.check_ready()?;
        self.next.check_ready()
    }

    fn reclaim_undelivered(&self) {
        // Mop up undelivered intermediates across the WHOLE subtree: the source's
        // outputs that `next` discarded (its element pipes), plus any intermediates
        // nested in source or next. Draining an already-consumed pipe (the normal
        // case) is a no-op (`take_home` yields `None`). The chain's own delivered
        // output lives in `next`'s output pipe, already drained by the terminal's
        // `gather_checkouts` before this runs → its `reclaim` is also a no-op.
        self.source.reclaim_undelivered();
        self.next.reclaim_undelivered();
    }

    fn contains_host_seam(&self) -> bool {
        // `next` is the downstream op built eagerly inside the `and_then` closure
        // at construction (a real owned field, not a deferred closure), so a host
        // seam built in the closure IS visible here.
        self.source.contains_host_seam() || self.next.contains_host_seam()
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn cb_addable(&self) -> bool {
        // A chain is CB-addable iff BOTH halves are (a transfer / host seam in
        // either disqualifies the whole subtree — coarse whole-graph gating).
        self.source.cb_addable() && self.next.cb_addable()
    }

    fn cbable_weight(&self) -> usize {
        // Precomputed at construction = source.weight + next.weight. Static under
        // mutate (see the trait doc), so a stored field keeps this O(1).
        self.cbable_weight
    }

    fn cb_spine_head_addable(&self) -> bool {
        // The chain CONTINUES / HEADS a maximal seam-free span as long as its
        // LEADING source can (recursively) — EVEN IF the whole chain is `!cb_addable`
        // (a seam lives further down `next`). Recursing through `source` (not just
        // `source.cb_addable()`) lets the span extend across nested spine `AndThen`s
        // whose own `next` is still addable, closing only at the `AndThen` whose
        // `next` is the seam.
        self.source.cb_spine_head_addable()
    }

    fn cb_restamp(&self, evs: &[Dep]) {
        // A chain's OUTPUT is its tail's output (its own `output_pipe` delegates to
        // `next`, and for a multi-output tail that is `None` so the default no-ops).
        // Delegate to the tail so a boundaried chain stamps the CB completion event
        // onto the pipes a downstream consumer actually reads.
        self.next.cb_restamp(evs);
    }

    fn invalidate_cbs(&self, mutated: &std::collections::BTreeSet<usize>) {
        // Own CB, then recurse into both children (a CB homed in either sub-position
        // is checked against `mutated` at its own node).
        cb_cache_invalidate(&self.cb_cache, mutated);
        self.source.invalidate_cbs(mutated);
        self.next.invalidate_cbs(mutated);
    }

    fn collect_cb_ids(&self, out: &mut Vec<usize>) {
        cb_cache_collect_id(&self.cb_cache, out);
        self.source.collect_cb_ids(out);
        self.next.collect_cb_ids(out);
    }
}

// ── Value: lift a host value into the graph ────────────────────────────

/// A host value lifted into the graph — produces it with no device work and no
/// events. Useful as a chain head, or to thread a host value alongside buffers.
///
/// ## By-VALUE handle (the whole point)
///
/// Unlike most ops (whose `Handle` defaults to `Pipe<Output>`), `Value` exposes
/// its **value itself** as the build-time handle (`Handle = T`, requires
/// `T: Clone`). So a downstream `and_then` / bundle closure receives the actual
/// `T`, NOT a `Pipe<T>` — letting host-side computation happen at build:
///
/// ```ignore
/// value(1u32).and_then(|n| value(n + 1))            // n is u32, n+1 works
/// bundle!(kernel, value(1u32)).and_then(|(buf, step)| {  // step is u32...
///     bundle!(kernel2(buf), value(step + 1))        // ...so step+1 computes here
/// })
/// ```
///
/// This is why `value` requires `T: Clone`: `handle(&self)` clones the value out
/// while the op keeps its own copy for `execute` (which still deposits into the
/// pipe for the terminal/bundle reconstruction path). For a non-`Clone` owned
/// resource (a `DeviceSlice` etc.), use [`lift`] instead — it keeps the default
/// `Pipe` handle (a buffer can't and shouldn't be computed on at build).
pub struct Value<T: Send> {
    // Held by value (not `Option`): `T: Clone`, so each run re-emits a fresh
    // clone and the source is never drained — the graph is reusable + idempotent.
    v: T,
    out: Pipe<T>,
}

/// Lift a `Clone` host value into the graph with a **by-value** handle (so
/// downstream closures get the value, not a pipe — see [`Value`]). For a
/// non-`Clone` owned resource use [`lift`].
///
/// `value` + [`lift`] together ≈ cuda-oxide's `value` (one host value into the
/// graph), split here by whether the handle is by-value (`value`) or by-pipe
/// (`lift`).
pub fn value<T: Send + Clone + 'static>(v: T) -> Value<T> {
    Value {
        v,
        out: Pipe::new(),
    }
}

impl<T: Send + Clone + 'static> DeviceOp for Value<T> {
    type Output = T;
    // By-value handle: downstream gets `T`, enabling build-time host compute.
    type Handle = T;

    fn output_pipe(&self) -> Option<Pipe<T>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        // Clone the value out for the downstream closure; `self.v` keeps its own
        // copy for `execute` (which clones again each run — idempotent).
        self.v.clone()
    }

    fn execute(&self, _ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Re-emit a fresh clone every run — never drains the source.
        self.out.put(self.v.clone(), Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("value".into());
    }
}

// ── Lift: an owned (non-Clone) resource as a SELF-REHOMING leaf ──────────

/// An owned resource lifted into the graph as a leaf — like [`Value`] but with
/// the **default `Pipe` handle** instead of by-value, so it works for non-`Clone`
/// types (a [`DeviceSlice`] etc. owns a `cl_mem` and cannot be cloned). Use it
/// to make a caller-owned buffer a chain head or a bundle branch:
///
/// ```ignore
/// lift(buf).and_then(acquire_mapped_view)                  // buffer chain head
/// bundle!(lift(buf), upload(v), alloc_zero(N))             // buffer bundle branch
/// ```
///
/// A buffer can't be computed on at build time anyway, so its downstream handle
/// is a `Pipe` (the value flows; you don't read it until execute). For a `Clone`
/// host value you want to compute on downstream, use [`value`] (by-value handle).
///
/// ## Self-rehoming: reusable across replays
///
/// `Lift` holds its value in a [`Cell`] (`Arc<Mutex<Option<T>>>`) — the SAME cell
/// an [`Input::Concrete`] uses — and **lends-and-returns** it: on each run the
/// value is taken out, threaded downstream with a home pointing back at this very
/// cell, and rehomed on the run's [`Checkout`] drop (the home invariant). So a
/// graph containing a `lift` **replays** across `sync`s over the SAME `cl_mem`
/// handle — a `lift`ed scalar / buffer is a re-arming bundle branch, exactly like
/// a concrete cell fed to an in-place verb. This is the "just present this owned
/// resource as a re-homing branch" primitive (no device work, unlike fill/scale).
///
/// A second `sync` while a previous [`Checkout`] is still alive (the value not yet
/// rehomed) errors "already lent — the graph is busy", like any concrete input.
/// [`Checkout::into_inner`] severs (takes the value for good); the cell then stays
/// empty and a subsequent `sync` reports busy — the concrete-cell sever semantics.
pub struct Lift<T: Send> {
    // The lifted value lives in a `Cell` (`Arc<Mutex<Option<T>>>`): lent on each
    // run and returned on `Checkout` drop, so the lift node is its OWN home and
    // the graph replays. Wrapping it as an `Input::Concrete` reuses the exact
    // lend/rehome/check_ready machinery a concrete kernel input uses.
    input: Input<T>,
    out: Pipe<T>,
}

/// Lift an owned resource into the graph (default `Pipe` handle — see [`Lift`]).
/// With [`value`], together ≈ cuda-oxide's `value` (the by-pipe half, for
/// non-`Clone` owned resources). Self-rehoming: the graph replays across `sync`s.
///
/// `T: RecordableBuffer` — a `lift` presents an owned **device** resource (a
/// buffer / scalar) as a re-homing graph edge, so it carries a recording handle
/// like every other device leaf; this lets a `lift`ed cell live inside a
/// command-buffer subtree. (A `Clone` host value you compute on downstream is
/// [`value`], not `lift`.)
pub fn lift<T: crate::record::RecordableBuffer + Send + 'static>(v: T) -> Lift<T> {
    Lift {
        input: Input::from(v),
        out: Pipe::new(),
    }
}

impl<T: crate::record::RecordableBuffer + Send + 'static> DeviceOp for Lift<T> {
    type Output = T;
    // Default `Handle = Pipe<T>` — a resource flows, it isn't read at build.

    fn output_pipe(&self) -> Option<Pipe<T>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Lend the value out of the concrete cell WITH its home (this very cell),
        // threading the host-seam start gate — exactly the concrete-input lend
        // path. The home flows downstream via `put_home`, so the value returns to
        // this cell on `Checkout` drop and the graph re-arms for the next run.
        let (v, deps, home) = self.input.resolve_home(ec)?;
        match ec.cb() {
            CbWalk::Off => {
                self.out.put_home(v, deps, home);
            }
            CbWalk::Build { ext, .. } | CbWalk::LendOnly { ext, .. } => {
                // A lifted concrete resource is an ENTRY into the CB — no upstream
                // producer, so no sync points to register (a consumer's sp_lookup on
                // our cell yields empty; the CB's external deps carry any real
                // wait). Its own deps (e.g. the start gate) become external deps.
                cb_collect_external(ext, &deps);
                // Passthrough (no command): forward this lift's slot/pipe origins onto
                // the output pipe so a downstream leaf still traces to the mutable slot.
                cb_forward_reach(
                    ec,
                    self.input.slot_cell_id(),
                    self.input.pipe_cell_id(),
                    self.out.cell_id(),
                );
                self.out.put_home(v, Deps::new(), home);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        // Pre-run atomicity: the cell is empty iff a previous run's Checkout still
        // holds the value (busy) or it was severed — the concrete-input check.
        self.input.check_ready()
    }

    fn cb_addable(&self) -> bool {
        // A lifted owned device resource is a valid CB entry leaf (no command).
        true
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("lift".into());
    }
}

// ── Forward: select/identity — make one upstream Pipe a single-output op ──

/// Forward a single upstream value (a `Pipe<T>`) onward as a single-output
/// [`DeviceOp`]. The identity op: it resolves its input and re-deposits it
/// (threading the deps), changing nothing. Its purpose is **shape**, not work —
/// it lets you pick ONE element out of a multi-output op's handle (e.g. a
/// kernel's `(Pipe<a>, Pipe<b>, Pipe<out>)`, or a bundle's per-branch pipes) and
/// continue on-device with that single value, instead of dropping to the host
/// or inserting a no-op kernel. The selected pipe becomes a normal
/// `DeviceOp<Output = T>` that composes via `and_then` / `bundle` like any leaf.
///
/// ```ignore
/// // pick `out` from add_u32's 3-tuple handle and keep going on-device:
/// ks.add_u32([N], a, b, out).and_then(|(_a, _b, out)| forward(out))
/// ```
pub struct Forward<T: Send> {
    input: Input<T>,
    out: Pipe<T>,
}

/// Forward an upstream value onward unchanged (identity op; see [`Forward`]).
/// Takes a [`Pipe<T>`] directly (the selected element of a multi-output handle)
/// — `Pipe` rather than `impl Into<Input>` so `T` infers cleanly from the
/// selected element without an annotation.
pub fn forward<T: Send + 'static>(pipe: Pipe<T>) -> Forward<T> {
    Forward {
        input: Input::Pipe(pipe),
        out: Pipe::new(),
    }
}

impl<T: Send + 'static> DeviceOp for Forward<T> {
    type Output = T;

    fn output_pipe(&self) -> Option<Pipe<T>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // CB-mode: identity re-aliases the upstream's sync points under OUR output
        // cell so a downstream CB consumer finds them, and deposits empty deps (the
        // ordering is the CB-internal markers). No device work either way.
        let upstream_cell = self.input.pipe_cell_id();
        let (v, deps, home) = self.input.resolve_home(ec)?;
        match ec.cb() {
            CbWalk::Off => {
                self.out.put_home(v, deps, home);
            }
            CbWalk::Build { ext, .. } => {
                cb_collect_external(ext, &deps);
                let sps = ec.sp_lookup(upstream_cell);
                ec.sp_register(self.out.cell_id(), sps);
                // Passthrough (identity alias): forward the upstream origins onto our
                // output cell, parallel to the sync-point re-register above.
                cb_forward_reach(
                    ec,
                    self.input.slot_cell_id(),
                    upstream_cell,
                    self.out.cell_id(),
                );
                self.out.put_home(v, Deps::new(), home);
            }
            CbWalk::LendOnly { ext, .. } => {
                cb_collect_external(ext, &deps);
                self.out.put_home(v, Deps::new(), home);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.input.check_ready()
    }

    fn cb_addable(&self) -> bool {
        // A structural passthrough over a pipe — always CB-addable (it adds no
        // command; it aliases the upstream inside the CB).
        true
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("forward".into());
    }
}

// ── DeviceDynOp: type-erased single-output op for conditional graphs ─────

/// Object-safe erasure of [`DeviceOp`], specialised to output `T`. Crate-internal
/// — users go through [`DeviceDynOp`]. `DeviceOp` itself is NOT object-safe (it has
/// an associated `Handle` type and `self`-consuming `collect`/`into_output`), so
/// this mirror trait restates the one operation a terminal/branch needs —
/// gather `(value, deps)` — as a `self: Box<Self>` method that *is*
/// dyn-dispatchable. It delegates to the concrete op's [`collect`](DeviceOp::collect),
/// which already reconstructs any arity down to a single `Output`, so even a
/// multi-output inner op erases cleanly to a single-output `DeviceDynOp`.
trait ErasedDeviceOp<T>: Send {
    fn collect_erased(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(T, Deps)>;

    fn describe_erased(&self, out: &mut Vec<String>);

    /// Forwards [`DeviceOp::check_ready`] through the erasure so a [`DeviceDynOp`]'s
    /// atomicity pre-pass covers whichever concrete arm it wraps (the arm's inputs
    /// are resolved by `collect_erased` at run, so they must be pre-checked too).
    fn check_ready_erased(&self) -> Result<()>;

    /// Forwards [`DeviceOp::contains_host_seam`] through the erasure so a
    /// [`DeviceDynOp`] reports the right gate bit for whichever concrete arm it
    /// wraps.
    fn contains_host_seam_erased(&self) -> bool;
}

impl<O> ErasedDeviceOp<O::Output> for O
where
    O: DeviceOp,
{
    fn collect_erased(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(O::Output, Deps)> {
        self.collect(ec, mode)
    }

    fn describe_erased(&self, out: &mut Vec<String>) {
        self.describe(out);
    }

    fn check_ready_erased(&self) -> Result<()> {
        self.check_ready()
    }

    fn contains_host_seam_erased(&self) -> bool {
        self.contains_host_seam()
    }
}

/// Type-erased single-output [`DeviceOp`] yielding `T`. Lets `if` / `match` arms
/// produce DIFFERENT concrete op types as long as they agree on `Output` — the
/// eager analog of the legacy closure-layer `DynOp`.
///
/// Each combinator chain has its own deeply-nested concrete type
/// (`AndThen<Upload, AndThen<…>>` vs `Value<T>`), so an `if`/`else` that builds a
/// chain in each arm is a type-mismatch error. Wrapping each arm in
/// `DeviceDynOp::new(...)` erases the concrete type to one nominal
/// `DeviceDynOp<'op, T>`, which is itself an [`DeviceOp`] and composes with
/// `and_then` / `bundle` / `fan_out` like any single-output leaf.
///
/// ```ignore
/// let chain: DeviceDynOp<u32> = if use_kernel {
///     DeviceDynOp::new(upload(v).and_then(|b| ks.fill_u32([N], b, 9)).and_then(|_| value(0u32)))
/// } else {
///     DeviceDynOp::new(value(0u32))            // different concrete type, same Output
/// };
/// let r = chain.sync(&ctx)?;
/// ```
///
/// One heap allocation per erased op. The `'op` lifetime lets the boxed op borrow
/// from the surrounding scope (typically `&Kernels` for kernel launches); it
/// infers to `'static` for chains built from owned data only.
///
/// **Single-output.** `Handle = Pipe<T>` (the default). A multi-output op CAN be
/// erased — its tuple `Output` becomes the `T` of the `DeviceDynOp` (reconstructed
/// via the inner op's `collect`), but the per-element build-time handle is gone;
/// downstream sees one `Pipe<tuple>`. For the conditional-graph use case (arms
/// agreeing on one `Output`) that is exactly right.
pub struct DeviceDynOp<'op, T> {
    inner: Box<dyn ErasedDeviceOp<T> + 'op>,
    out: Pipe<T>,
}

impl<'op, T: Send + 'static> DeviceDynOp<'op, T> {
    /// Erase a concrete op into a single-output `DeviceDynOp`. Both arms of an
    /// `if`/`match` can produce `DeviceDynOp::new(...)` of the same `T` without
    /// their concrete types matching.
    pub fn new<O>(op: O) -> Self
    where
        O: DeviceOp<Output = T> + 'op,
    {
        DeviceDynOp {
            inner: Box::new(op),
            out: Pipe::new(),
        }
    }
}

impl<T: Send + 'static> DeviceOp for DeviceDynOp<'_, T> {
    type Output = T;

    fn output_pipe(&self) -> Option<Pipe<T>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // Gather the erased inner op (any arity → one value + deps) and deposit
        // into our own pipe, so the default collect/into_output/handle path treats
        // this as an ordinary single-output leaf. The inner op observes `mode`
        // (it is the real terminal work when this DeviceDynOp is the chain tail).
        // `&self`: the inner op runs by reference, so an erased arm is reusable.
        let (v, deps) = self.inner.collect_erased(ec, mode)?;
        self.out.put(v, deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.inner.check_ready_erased()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("dyn_op{".into());
        self.inner.describe_erased(out);
        out.push("}".into());
    }

    fn contains_host_seam(&self) -> bool {
        self.inner.contains_host_seam_erased()
    }
}

// ── Arced: wrap the output in Arc<T> ───────────────────────────────────

/// Wrap an upstream op's output in [`Arc`] for shared fan-out. Passes events
/// through unchanged.
pub struct Arced<S: DeviceOp> {
    source: S,
    out: Pipe<Arc<S::Output>>,
    /// Design-v2 CB home: `Arced` adds no command (an `Arc` wrap), but it DELEGATES
    /// to its source — a CB-addable source subtree records through here.
    cb_cache: CbCache,
}

/// Wrap `source`'s output in `Arc`. ≈ cuda-oxide's `.arc()` (also available here
/// as the [`arc`](DeviceOpExt::arc) method).
pub fn arced<S: DeviceOp>(source: S) -> Arced<S>
where
    S::Output: Sync,
{
    Arced {
        source,
        out: Pipe::new(),
        cb_cache: new_cb_cache(),
    }
}

impl<S> DeviceOp for Arced<S>
where
    S: DeviceOp,
    S::Output: Sync,
{
    type Output = Arc<S::Output>;

    fn output_pipe(&self) -> Option<Pipe<Arc<S::Output>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Gather the source via `collect` (reconstructs any arity — a bundle
        // source fills element pipes, not output_pipe), then wrap in Arc. `collect`
        // threads `ec`'s CbWalk, so a Build/LendOnly pass records the source's
        // commands into the CB and deposits its output with EMPTY deps.
        let (v, deps) = self.source.collect(ec, ExecMode::Pipelined)?;
        match ec.cb() {
            CbWalk::Off => {
                self.out.put(Arc::new(v), deps);
            }
            CbWalk::Build { .. } => {
                // Alias the source's sync points under OUR output cell so a downstream
                // CB consumer of the `Arc<buffer>` finds them (mirrors Forward). The
                // source's single output pipe carries them (arced wraps single-output).
                if let Some(src_cell) = self.source.output_pipe().map(|p| p.cell_id()) {
                    let sps = ec.sp_lookup(Some(src_cell));
                    ec.sp_register(self.out.cell_id(), sps);
                    // Passthrough: forward the source's origins onto our cell too
                    // (parallel to the sync-point alias).
                    cb_forward_reach(ec, None, Some(src_cell), self.out.cell_id());
                }
                self.out.put(Arc::new(v), Deps::new());
            }
            CbWalk::LendOnly { .. } => {
                self.out.put(Arc::new(v), Deps::new());
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn invalidate_cbs(&self, mutated: &std::collections::BTreeSet<usize>) {
        cb_cache_invalidate(&self.cb_cache, mutated);
        self.source.invalidate_cbs(mutated);
    }

    fn collect_cb_ids(&self, out: &mut Vec<usize>) {
        cb_cache_collect_id(&self.cb_cache, out);
        self.source.collect_cb_ids(out);
    }

    fn cb_addable(&self) -> bool {
        // No command of its own — CB-addable iff the source subtree is.
        self.source.cb_addable()
    }

    fn cbable_weight(&self) -> usize {
        // Arc-wrap records nothing; the weight is the source's.
        self.source.cbable_weight()
    }

    fn cb_spine_head_addable(&self) -> bool {
        self.source.cb_spine_head_addable()
    }

    fn cb_restamp(&self, evs: &[Dep]) {
        // Stamp the CB completion event onto our output pipe for a cross-CB consumer.
        if let Some((v, _d)) = self.out.take() {
            self.out.put(v, Vec::from(evs));
        }
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("arced".into());
    }

    fn contains_host_seam(&self) -> bool {
        self.source.contains_host_seam()
    }
}

// ── ArcSplit: fan one Arc output to N read-only branches ───────────────

/// Fan a single `Arc<T>` upstream output out to `N` independent downstream
/// branches, each receiving its own cheap `Arc::clone`. Mirrors the closure
/// layer's `.arc().and_then(|arc| { let [a, b, c] = arc.split::<3>(); … })`:
/// one producer, `N` read-only consumers.
///
/// `Handle = [Pipe<S::Output>; N]` — the downstream `and_then` closure
/// destructures the array (`let [a, b, c] = handle`) and each element is a
/// `Pipe<Arc<T>>` that flows into a branch op (e.g. a read-only kernel, or a
/// download). `execute` runs the source once, then scatters `Arc::clone(&arc)`
/// (plus a clone of the producer's wait-list) into each of the `N` element
/// pipes — every branch sees the same value and waits on the same producer
/// event. `Output = [S::Output; N]` (the `N` clones) for the terminal case.
///
/// Use [`arc_split`] to build one — it follows an [`arced`] source.
pub struct ArcSplit<S: DeviceOp, const N: usize>
where
    S::Output: Clone,
{
    source: S,
    // One element pipe per branch (move-once storage); each gets an
    // `Arc::clone` of the source value in `execute`.
    outs: [Pipe<S::Output>; N],
    /// Design-v2 CB home: adds no command (Arc-clone scatter), delegates to source.
    cb_cache: CbCache,
}

/// Build an [`ArcSplit`]: fan `source`'s `Arc<T>` output to `N` read-only
/// branches. `source` is typically an [`arced`] op (`Output = Arc<T>`), so the
/// per-branch clone is a cheap refcount bump. Pick `N` via turbofish to match
/// the destructure arity: `arc_split::<3, _>(arced(upload(…)))`.
///
/// ≈ a homogeneous N-ary `unzip!` over a shared [`arc`](DeviceOpExt::arc)ed input
/// (one producer, `N` read-only consumers).
pub fn arc_split<const N: usize, S: DeviceOp>(source: S) -> ArcSplit<S, N>
where
    S::Output: Clone,
{
    ArcSplit {
        source,
        outs: std::array::from_fn(|_| Pipe::new()),
        cb_cache: new_cb_cache(),
    }
}

impl<S, const N: usize> DeviceOp for ArcSplit<S, N>
where
    S: DeviceOp,
    S::Output: Clone,
{
    type Output = [S::Output; N];
    // An array of N element pipes; the downstream closure does
    // `let [a, b, c] = handle` and routes each pipe into its own branch.
    type Handle = [Pipe<S::Output>; N];
    // Per-branch Checkouts (homes are all `None` — these are read-only Arc clones).
    type Checkouts = [Checkout<S::Output>; N];

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        // Multi-output storage is the per-element pipes; there is no single
        // storage pipe (the default `into_output` is overridden, and `and_then`
        // uses `handle()`), so return `None`.
        None
    }

    fn handle(&self) -> Self::Handle {
        self.outs.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Gather the source via `collect` (any arity), then scatter a clone of
        // its value + events into every branch pipe (Arc::clone is a cheap
        // refcount bump; Deps clone shares the same producer events). `collect`
        // threads `ec`'s CbWalk, so a Build/LendOnly pass records the source's
        // commands into the CB.
        let (v, deps) = self.source.collect(ec, ExecMode::Pipelined)?;
        match ec.cb() {
            CbWalk::Off => {
                for out in &self.outs {
                    out.put(v.clone(), deps.clone());
                }
            }
            CbWalk::Build { .. } => {
                // Alias the source's sync points under EVERY branch cell so each
                // downstream CB consumer of a clone finds them; deposit empty deps.
                let src_cell = self.source.output_pipe().map(|p| p.cell_id());
                let sps = src_cell.map(|c| ec.sp_lookup(Some(c))).unwrap_or_default();
                // Passthrough: every branch inherits the source's origins (parallel
                // to the sync-point alias).
                for out in &self.outs {
                    ec.sp_register(out.cell_id(), sps.clone());
                    cb_forward_reach(ec, None, src_cell, out.cell_id());
                    out.put(v.clone(), Deps::new());
                }
            }
            CbWalk::LendOnly { .. } => {
                for out in &self.outs {
                    out.put(v.clone(), Deps::new());
                }
            }
        }
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
    where
        Self: Sized,
    {
        // Grab the element pipes before consuming `self`, scatter via `execute`,
        // then drain all N to reconstruct the `[clone; N]` array, gathering
        // every branch's deps (the terminal `into_output` waits on them once).
        let outs = self.outs.clone();
        self.execute(ec, mode)?;
        let mut all_deps: Deps = Deps::new();
        let mut vals: Vec<S::Output> = Vec::with_capacity(N);
        for p in &outs {
            let (v, d) = p.take().ok_or(Error::NotSupported(
                "eager arc_split: a branch produced no output",
            ))?;
            vals.push(v);
            all_deps.extend(d);
        }
        // `vals` has exactly N elements (one per element pipe) — the conversion
        // cannot fail.
        let arr = vals
            .try_into()
            .unwrap_or_else(|_| unreachable!("arc_split drained exactly N branch pipes"));
        Ok((arr, all_deps))
    }

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Output, Deps, Option<BoxedHome<Self::Output>>)> {
        // Multi-output (`[T; N]` fan-out of read-only Arc clones): no single
        // collapsed home. Delegate to `collect`; `home == None`.
        let (value, deps) = self.collect(ec, mode)?;
        Ok((value, deps, None))
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // Drain each branch pipe with its home → a `[Checkout; N]`. Arc-clone
        // branches carry no home (read-only fan-out), so each Checkout's home is
        // `None`; the per-branch shape is still correct.
        let outs = self.outs.clone();
        self.execute(ec, mode)?;
        let mut all_deps: Deps = Deps::new();
        let mut cos: Vec<Checkout<S::Output>> = Vec::with_capacity(N);
        for p in &outs {
            let (v, d, home) = p.take_home().ok_or(Error::NotSupported(
                "eager arc_split: a branch produced no output",
            ))?;
            cos.push(Checkout::new(v, home));
            all_deps.extend(d);
        }
        let arr = cos
            .try_into()
            .unwrap_or_else(|_| unreachable!("arc_split drained exactly N branch pipes"));
        Ok((arr, all_deps))
    }

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn invalidate_cbs(&self, mutated: &std::collections::BTreeSet<usize>) {
        cb_cache_invalidate(&self.cb_cache, mutated);
        self.source.invalidate_cbs(mutated);
    }

    fn collect_cb_ids(&self, out: &mut Vec<usize>) {
        cb_cache_collect_id(&self.cb_cache, out);
        self.source.collect_cb_ids(out);
    }

    fn cb_addable(&self) -> bool {
        // No command of its own (Arc-clone scatter) — addable iff the source is.
        self.source.cb_addable()
    }

    fn cbable_weight(&self) -> usize {
        self.source.cbable_weight()
    }

    fn cb_spine_head_addable(&self) -> bool {
        self.source.cb_spine_head_addable()
    }

    fn cb_restamp(&self, evs: &[Dep]) {
        // Multi-output: stamp the CB completion event onto every branch pipe.
        for out in &self.outs {
            if let Some((v, _d)) = out.take() {
                out.put(v, Vec::from(evs));
            }
        }
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push(format!("arc_split[{N}]"));
    }

    fn contains_host_seam(&self) -> bool {
        self.source.contains_host_seam()
    }
}

// ── Bundle: N independent branches, joined by a marker ─────────────────

/// Join `n` branch wait-lists into one marker event on `ec`'s queue.
fn join_marker(ec: &ExecutionContext<'_>, branch_deps: &[Deps]) -> Result<Deps> {
    use crate::Launcher;
    let all: Vec<crate::cl_event> = branch_deps
        .iter()
        .flat_map(|d| d.iter().map(|e| e.as_ref().get()))
        .collect();
    // SAFETY: cl_event handles are valid — held by the branch `Deps` Arcs
    // until this call returns.
    let marker =
        unsafe { ec.cl_queue().enqueue_marker_with_wait_list(&all) }.map_err(Error::OpenCl)?;
    Ok(vec![wrap_event(marker)])
}

/// Reduce a gathered op's [`Deps`] to a single owned [`Event`](crate::Event) — used by the
/// Tier-1 `submit`/`submit_on` terminals (which hand the caller a chainable
/// event). Always enqueues a marker over the deps, so the result is one owned
/// `Event` that completes exactly when the op's work does, regardless of how
/// many events the op produced (one for a single-output kernel, several for a
/// multi-output launch). Empty deps → a bare marker (fires immediately).
pub fn deps_into_single_event(ec: &ExecutionContext<'_>, deps: Deps) -> Result<crate::Event> {
    use crate::Launcher;
    let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
    // SAFETY: the cl_event handles are held alive by `deps` across this call.
    let marker =
        unsafe { ec.cl_queue().enqueue_marker_with_wait_list(&raw) }.map_err(Error::OpenCl)?;
    drop(deps);
    Ok(marker)
}

macro_rules! impl_eager_bundle {
    ($name:ident, $ctor:ident, $($field:ident : $ty:ident : $pf:ident),+) => {
        #[doc = concat!("Eager bundle of independent branches (arity ",
            stringify!($name), "). Built by [`", stringify!($ctor),
            "`]; branches run with no inter-ordering, joined by a marker.")]
        pub struct $name<$($ty: DeviceOp),+> {
            $($field: $ty,)+
            /// Design-v2 CB home: when this bundle is the outermost CB-eligible node
            /// of a seam-free subtree, the whole subtree is ONE command buffer homed
            /// here. See [`CbCache`].
            cb_cache: CbCache,
            /// Precomputed sum of the branches' [`cbable_weight`](DeviceOp::cbable_weight)
            /// (static under mutate). Set once by the constructor.
            cbable_weight: usize,
        }

        #[doc = concat!("Construct an eager [`", stringify!($name),
            "`]. \u{2248} cuda-oxide's `zip!` at this fixed arity.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $ctor<$($ty: DeviceOp),+>($($field: $ty),+) -> $name<$($ty),+> {
            let cbable_weight = 0 $(+ $field.cbable_weight())+;
            $name { $($field,)+ cb_cache: new_cb_cache(), cbable_weight }
        }

        impl<$($ty: DeviceOp),+> DeviceOp for $name<$($ty),+>
        where
            // Each branch's output must be `Send + 'static` so its return home (a
            // `BoxedHome`, i.e. `Box<dyn Rehome + 'static>`) can ride the branch's
            // own pipe(s) in `execute`/`gather_checkouts` — the seam that re-arms a
            // bundle over caller-owned buffers. Buffer outputs are always `'static`.
            $(<$ty as DeviceOp>::Output: Send + 'static,)+
            // Each branch's terminal `Checkouts` must reconstruct from its own
            // `Output` — the branch's OWN `gather_checkouts` bound. Holds for every
            // real op: a single-output branch's `Checkout<O>` via the identity impl,
            // a multi-output branch's tuple via the recursive `FromCheckout` family.
            $(<$ty as DeviceOp>::Checkouts: FromCheckout<<$ty as DeviceOp>::Output>,)+
        {
            type Output = ( $(<$ty as DeviceOp>::Output,)+ );
            // A tuple of each branch's OWN build-time handle (NOT forced to a
            // pipe). For a buffer-producing branch that handle defaults to
            // `Pipe<buffer>` (so a multi-arg op consumes it via `ToInput`, as
            // before); for a `value(scalar)` branch it is the scalar BY VALUE, so
            // the downstream closure can compute on it at build (e.g. `step + 1`);
            // for a nested bundle / multi-output branch it is that branch's own
            // composite handle. Composing per-branch handles (rather than
            // flattening to pipes) is what carries computable host values down.
            type Handle = ( $(<$ty as DeviceOp>::Handle,)+ );
            // STRUCTURE-PRESERVING per-branch Checkouts: each branch contributes
            // its OWN `Checkouts` — NOT a single `Checkout` over the whole branch
            // output. A single-output branch contributes `Checkout<buf>`; a
            // **multi-output** branch contributes ITS tuple `(Checkout, Checkout,
            // …)`; a nested bundle contributes its own nested-by-branch shape. So
            // the bundle's `Checkouts` is grouped-by-branch, per-buffer WITHIN each
            // branch (e.g. `((Checkout<a>, Checkout<b>), Checkout<c>)`), matching
            // `Output` (also nested-by-branch). Delegation, not flatten: each
            // branch buffer is individually droppable / lendable / mappable, and
            // every branch — at any output multiplicity or nesting depth — threads
            // its OWN per-buffer return homes via its OWN `gather_checkouts`, so a
            // bundle of multi-output branches RE-ARMS (the former limitation, now
            // fixed).
            type Checkouts = ( $(<$ty as DeviceOp>::Checkouts,)+ );

            fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
                // Multi-output storage is the per-branch pipes (owned by each
                // branch); there is no single storage pipe (the default
                // `into_output` is overridden, and `and_then` uses `handle()`), so
                // return `None`.
                None
            }

            fn handle(&self) -> Self::Handle {
                // Delegate to each branch's own `handle()` — preserves by-value
                // for `value`, pipe for buffers, composite for nested bundles.
                ( $(self.$field.handle(),)+ )
            }

            fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
                // Each branch runs independently (pipelined) and fills its OWN
                // handle storage — a single-output branch its output pipe, a
                // multi-output branch its element pipes. This is the MID-GRAPH
                // scatter: a downstream `and_then` reads each branch through
                // `handle()` (which delegates per-branch to the SAME pipes filled
                // here), so there is no reconstruction / round-trip. The TERMINAL
                // gather (`gather_checkouts` / `collect`) instead delegates to each
                // branch's own gather so per-buffer homes are threaded — see below.

                // CB boundary: when the WHOLE bundle is a fully-addable device region
                // and we're in `Off` (nothing above opened a CB), open ONE command
                // buffer for the bundle. `cb_boundary_execute` re-enters this `execute`
                // in `Build`, where the guard below is false (not `Off`) so each branch
                // runs via `cb_exec_child`'s "already inside a CB → forward" arm and
                // records into the SAME CB — its parallel branches are joined by their
                // independent sync points, no cross-branch dep. `cb_restamp` (this
                // impl, below) stamps the one CB event onto every branch pipe. Without
                // this, each branch would open its OWN CB (per-branch spans) — correct
                // but N create/finalize/enqueue instead of one.
                if matches!(ec.cb(), CbWalk::Off)
                    && ec.context().has_cl_khr_command_buffer()
                    && self.cbable_weight() >= 2
                    && self.cb_addable()
                {
                    return cb_boundary_execute(self, ec, ExecMode::Pipelined);
                }
                $( cb_exec_child(&self.$field, ec, ExecMode::Pipelined)?; )+
                Ok(())
            }

            fn collect(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<(Self::Output, Deps)>
            where
                Self: Sized,
            {
                // Terminal by-value gather (async `run` / `into_output`): delegate
                // to each branch's OWN `collect` so a multi-output branch runs its
                // override (scatter-then-reconstruct over its element pipes) rather
                // than a drain of a pipe it never fills. Branches pipeline; the
                // terminal waits on the joined marker.
                let mut branch_deps: Vec<Deps> = Vec::new();
                let outputs = ( $({
                    let (v, d) = self.$field.collect(ec, ExecMode::Pipelined)?;
                    branch_deps.push(d);
                    v
                },)+ );
                let joined = join_marker(ec, &branch_deps)?;
                Ok((outputs, joined))
            }

            #[allow(clippy::type_complexity)]
            fn collect_home(
                &self,
                ec: &ExecutionContext<'_>,
                mode: ExecMode,
            ) -> Result<(Self::Output, Deps, Option<BoxedHome<Self::Output>>)> {
                // A bundle's per-branch homes ride each branch's own `Checkout`
                // (built in `gather_checkouts` by delegating to the branch), NOT one
                // collapsed tuple home — so a single-slot `BoxedHome` over the whole
                // tuple would be wrong. Return `home == None` here; this path is now
                // only a fallback (the terminal gather delegates to branch
                // `gather_checkouts`, which threads every branch's real per-buffer
                // homes directly). Delegate the value to `collect`.
                let (value, deps) = self.collect(ec, mode)?;
                Ok((value, deps, None))
            }

            fn gather_checkouts(
                &self,
                ec: &ExecutionContext<'_>,
                _mode: ExecMode,
            ) -> Result<(Self::Checkouts, Deps)> {
                // TERMINAL gather — the structure-preserving delegate. Each branch
                // runs its OWN `gather_checkouts`, producing its OWN `Checkouts`
                // (a `Checkout` for a single-output branch, a tuple for a
                // multi-output branch, a nested shape for a nested bundle) with its
                // OWN per-buffer return homes. So EVERY branch buffer — at any
                // output multiplicity or nesting depth — re-arms its origin cell on
                // drop, and the bundle re-runs (the fix: no more "multi-output
                // branch collapses to home == None"). Branches pipeline; join their
                // wait-lists into one marker.
                // CB mid-graph boundaries: a fully-addable branch gathers as its own
                // command buffer (via `cb_gather_child`); mixed branches recurse.
                let mut branch_deps: Vec<Deps> = Vec::new();
                let checkouts = ( $({
                    let (co, d) = cb_gather_child(&self.$field, ec, ExecMode::Pipelined)?;
                    branch_deps.push(d);
                    co
                },)+ );
                let joined = join_marker(ec, &branch_deps)?;
                Ok((checkouts, joined))
            }

            fn reclaim_undelivered(&self) {
                // Mid-graph mop-up: a downstream `and_then` closure may discard some
                // branch handles (e.g. keep only one branch's output). Each branch
                // still filled its own pipe(s) at `execute`; delegate to each
                // branch's `reclaim_undelivered` so those undelivered homed buffers
                // return to their origin cells for the next run. Pipes already
                // drained by the terminal / a downstream consumer are no-ops.
                $( self.$field.reclaim_undelivered(); )+
            }

            fn describe(&self, out: &mut Vec<String>) {
                out.push(concat!(stringify!($name), "{").into());
                $(self.$field.describe(out);)+
                out.push("}".into());
            }

            fn bind_slots(&self, binder: &mut SlotBinder) {
                // Recurse into EVERY branch — a `slot!(Tag)` placed inside any
                // bundle branch must be reachable by `g.bind(Tag(v))`. Without
                // this override the bundle would inherit the no-op default and
                // silently skip its branches, leaving the slot `Unbound` until
                // `sync` errors `SlotUnbound`.
                //
                // Fan-out discipline mirrors `AndThen::bind_slots`: a move-only
                // binder stops once its single value has landed (one `bind` →
                // one cell); a fan-out binder (clone-able value — scalar / launch
                // / `Arc`) NEVER consumes, so one `bind` fills EVERY matching cell
                // across all branches. We call each branch unconditionally; a
                // consumed move-only binder no-ops at the next branch via its own
                // `value.is_none()` guard (so the early-return below is purely an
                // optimization, not a correctness requirement).
                $(
                    self.$field.bind_slots(binder);
                    if !binder.is_fanout() && binder.is_consumed() {
                        return;
                    }
                )+
            }

            fn check_ready(&self) -> Result<()> {
                // Recurse into EVERY branch — `execute` `collect`s each, so each
                // branch's inputs are resolved and must be pre-checked. Fail-fast
                // on the first unsatisfiable branch.
                $(self.$field.check_ready()?;)+
                Ok(())
            }

            fn contains_host_seam(&self) -> bool {
                // OR every branch: a host seam in any one of them must gate.
                false $(|| self.$field.contains_host_seam())+
            }

            fn cb_addable(&self) -> bool {
                // AND every branch: the whole bundle is CB-addable iff every branch
                // is (a transfer / host seam in any branch disqualifies it).
                true $(&& self.$field.cb_addable())+
            }

            fn cbable_weight(&self) -> usize {
                // Precomputed sum of the branches' weights (static under mutate). A
                // `bundle` of pure passthroughs (CG's `bundle6(p, ap, …)` /
                // `bundle2(x, rsnew)`) is 0 → never opens a CB.
                self.cbable_weight
            }

            fn cb_restamp(&self, evs: &[Dep]) {
                // Stamp the CB completion event onto every branch's output pipe(s) —
                // delegate to each branch's own `cb_restamp` (a multi-output branch
                // stamps each element pipe). So a downstream consumer of ANY branch
                // waits on the one CB enqueue event.
                $(self.$field.cb_restamp(evs);)+
            }

            fn cb_cache(&self) -> Option<&CbCache> {
                Some(&self.cb_cache)
            }

            fn invalidate_cbs(&self, mutated: &::std::collections::BTreeSet<usize>) {
                $crate::eager::cb_cache_invalidate(&self.cb_cache, mutated);
                $(self.$field.invalidate_cbs(mutated);)+
            }

            fn collect_cb_ids(&self, out: &mut ::std::vec::Vec<usize>) {
                $crate::eager::cb_cache_collect_id(&self.cb_cache, out);
                $(self.$field.collect_cb_ids(out);)+
            }
        }
    };
}

impl_eager_bundle!(Bundle2, bundle2, a: A: pa, b: B: pb);
impl_eager_bundle!(Bundle3, bundle3, a: A: pa, b: B: pb, c: C: pc);
impl_eager_bundle!(Bundle4, bundle4, a: A: pa, b: B: pb, c: C: pc, d: D: pd);
impl_eager_bundle!(Bundle5, bundle5, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe);
impl_eager_bundle!(Bundle6, bundle6, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf);
impl_eager_bundle!(Bundle7, bundle7, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg);
impl_eager_bundle!(Bundle8, bundle8, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph);
impl_eager_bundle!(Bundle9, bundle9, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi);
impl_eager_bundle!(Bundle10, bundle10, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj);
impl_eager_bundle!(Bundle11, bundle11, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk);
impl_eager_bundle!(Bundle12, bundle12, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl);
impl_eager_bundle!(Bundle13, bundle13, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl, m: M: pm);
impl_eager_bundle!(Bundle14, bundle14, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl, m: M: pm, n: N: pn);
impl_eager_bundle!(Bundle15, bundle15, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl, m: M: pm, n: N: pn, o: O: po);
impl_eager_bundle!(Bundle16, bundle16, a: A: pa, b: B: pb, c: C: pc, d: D: pd, e: E: pe, f: F: pf, g: G: pg, h: H: ph, i: I: pi, j: J: pj, k: K: pk, l: L: pl, m: M: pm, n: N: pn, o: O: po, p: P: pp);

/// Variadic constructor for [`Bundle2`] through [`Bundle16`] — picks the right
/// `bundleN` based on the number of arguments. Each arm runs its branches
/// independently and joins them with a single marker event.
///
/// ≈ cuda-oxide's `zip!` (heterogeneous parallel join into a tuple).
///
/// ```ignore
/// let (a, b) = bundle!(op_a, op_b).sync(&ctx)?;
/// let (a, b, c) = bundle!(op_a, op_b, op_c).sync(&ctx)?;
/// // ... up to 16 children
/// ```
#[macro_export]
macro_rules! bundle {
    ($a:expr, $b:expr $(,)?) => {
        $crate::eager::bundle2($a, $b)
    };
    ($a:expr, $b:expr, $c:expr $(,)?) => {
        $crate::eager::bundle3($a, $b, $c)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr $(,)?) => {
        $crate::eager::bundle4($a, $b, $c, $d)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr $(,)?) => {
        $crate::eager::bundle5($a, $b, $c, $d, $e)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr $(,)?) => {
        $crate::eager::bundle6($a, $b, $c, $d, $e, $f)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr $(,)?) => {
        $crate::eager::bundle7($a, $b, $c, $d, $e, $f, $g)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr $(,)?) => {
        $crate::eager::bundle8($a, $b, $c, $d, $e, $f, $g, $h)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr $(,)?) => {
        $crate::eager::bundle9($a, $b, $c, $d, $e, $f, $g, $h, $i)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr $(,)?) => {
        $crate::eager::bundle10($a, $b, $c, $d, $e, $f, $g, $h, $i, $j)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr $(,)?) => {
        $crate::eager::bundle11($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr $(,)?) => {
        $crate::eager::bundle12($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr, $m:expr $(,)?) => {
        $crate::eager::bundle13($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr, $m:expr, $n:expr $(,)?) => {
        $crate::eager::bundle14($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr, $m:expr, $n:expr, $o:expr $(,)?) => {
        $crate::eager::bundle15($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $l:expr, $m:expr, $n:expr, $o:expr, $p:expr $(,)?) => {
        $crate::eager::bundle16(
            $a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p,
        )
    };
}

// ── FanOut: a homogeneous Vec of branches, joined by a marker ──────────

/// Eager fan-out: build one op per input (the builder `f` runs at construction
/// — eager — over the known input list), run them independently, join via a
/// marker. Output is `Vec<U::Output>`.
///
/// ## Per-branch re-arm (the dynamic-`Vec` analog of `bundle!`)
///
/// Like [`bundle!`](crate::bundle) (whose per-branch [`Checkout`] tuple carries
/// each branch's return [`home`](BoxedHome) so a bundle over caller-owned buffers
/// replays), `FanOut`'s terminal [`Checkouts`](DeviceOp::Checkouts) is a
/// `Vec<U::Checkouts>` — ONE `Checkout` (or nested `Checkouts` tuple, for a
/// multi-output branch) per branch, each threading its OWN return home. So a
/// fan-out whose branches are IN-PLACE ops over caller-owned buffers (e.g.
/// `fan_out(bufs, |b| fill(b, v))`) returns every buffer to its cell on drop, and
/// the SAME `FanOut` graph **replays** with stable `cl_mem` handles — exactly like
/// `bundle!`, just at dynamic arity. This is the bundle-arity `gather_checkouts`
/// per-branch delegation ([#207] / [#212]) generalised from a fixed tuple to a
/// runtime `Vec`. Fan-out over **minted** buffers (`upload`/`alloc`) or read-only
/// inputs also replays: those branches carry `home == None` (nothing to return),
/// which is fine.
///
/// The by-value paths ([`collect`](DeviceOp::collect) / async `run`) collapse to a
/// `Vec<U::Output>` with no per-element home (a `Vec` value has one slot, `N` homes
/// can't ride it) — the same by-value boundary `bundle!` documents; the re-arm
/// rides the [`Checkout`] terminal ([`gather_checkouts`](DeviceOp::gather_checkouts)),
/// which every waiting terminal uses.
pub struct FanOut<U: DeviceOp> {
    ops: Vec<U>,
    out: Pipe<Vec<U::Output>>,
    /// Design-v2 CB home: no command of its own, but its N branches record through
    /// here when they're all CB-addable.
    cb_cache: CbCache,
}

/// Build a fan-out: `f` is called now for each input, producing the branch ops.
pub fn fan_out<I, F, U>(inputs: Vec<I>, mut f: F) -> FanOut<U>
where
    F: FnMut(I) -> U,
    U: DeviceOp,
{
    let ops: Vec<U> = inputs.into_iter().map(&mut f).collect();
    FanOut {
        ops,
        out: Pipe::new(),
        cb_cache: new_cb_cache(),
    }
}

/// Method-call form of [`fan_out`]: `vec![a, b].fan_out(|i| value(i))`.
///
/// Reads as data → operation and composes cleanly with downstream `.and_then`;
/// the free-fn form stays available — use whichever fits the call site.
pub trait DeviceFanOutExt<I>: Sized {
    /// See [`fan_out`] — this delegates to it.
    fn fan_out<F, U>(self, f: F) -> FanOut<U>
    where
        F: FnMut(I) -> U,
        U: DeviceOp;
}

impl<I> DeviceFanOutExt<I> for Vec<I> {
    fn fan_out<F, U>(self, f: F) -> FanOut<U>
    where
        F: FnMut(I) -> U,
        U: DeviceOp,
    {
        fan_out(self, f)
    }
}

impl<U: DeviceOp> DeviceOp for FanOut<U>
where
    // Each branch's output must be `Send + 'static` so its return home (a
    // `BoxedHome`) can ride the branch's own `Checkout` — the seam that re-arms a
    // fan-out over caller-owned buffers. Buffer outputs are always `'static`.
    U::Output: Send + 'static,
    // Each branch's terminal `Checkouts` must reconstruct from its own `Output` —
    // the branch's OWN `gather_checkouts` bound (single-output via the identity
    // impl, multi-output via the recursive tuple family).
    U::Checkouts: FromCheckout<U::Output>,
{
    type Output = Vec<U::Output>;
    // STRUCTURE-PRESERVING per-branch Checkouts: a `Vec` of each branch's OWN
    // `Checkouts` (a `Checkout<O>` for a single-output branch; its tuple for a
    // multi-output branch), each threading its own per-buffer return home — so
    // every branch re-arms its origin cell on drop and the fan-out replays. The
    // dynamic-`Vec` analog of `bundle!`'s per-branch tuple `Checkouts`.
    type Checkouts = Vec<U::Checkouts>;

    fn output_pipe(&self) -> Option<Pipe<Vec<U::Output>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // MID-GRAPH / by-value scatter: `collect` each branch (not `execute`) so a
        // multi-output branch runs its own gather → one reconstructed value + deps.
        // Deposits the collapsed `Vec<Output>` into the single output pipe (home
        // `None` — the by-value boundary; per-branch re-arm rides `gather_checkouts`).
        let n = self.ops.len();
        let mut branch_deps: Vec<Deps> = Vec::with_capacity(n);
        let mut outputs: Vec<U::Output> = Vec::with_capacity(n);
        for op in &self.ops {
            let (v, d) = op.collect(ec, ExecMode::Pipelined)?;
            outputs.push(v);
            branch_deps.push(d);
        }

        // CB-mode: each branch's `collect(ec)` recorded into the CB (ordering =
        // sync points). Union every branch's sync points under OUR output cell so a
        // downstream CB consumer of the Vec waits on all branches; deposit empty deps
        // (no marker — join_marker would ENQUEUE, which the CB build must not do).
        match ec.cb() {
            CbWalk::Build { .. } => {
                // The collapsed `Vec` output depends on EVERY branch's sync points +
                // origins — union them onto our one output cell (cb_forward_reach
                // accumulates across branches).
                let mut all_sps = std::collections::BTreeSet::new();
                for op in &self.ops {
                    if let Some(p) = op.output_pipe() {
                        all_sps.extend(ec.sp_lookup(Some(p.cell_id())));
                        cb_forward_reach(ec, None, Some(p.cell_id()), self.out.cell_id());
                    }
                }
                ec.sp_register(self.out.cell_id(), all_sps);
                self.out.put(outputs, Deps::new());
                return Ok(());
            }
            CbWalk::LendOnly { .. } => {
                self.out.put(outputs, Deps::new());
                return Ok(());
            }
            CbWalk::Off => {}
        }

        let joined = join_marker(ec, &branch_deps)?;
        self.out.put(outputs, joined);
        Ok(())
    }

    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Output, Deps, Option<BoxedHome<Self::Output>>)> {
        // A `Vec<Output>` value has ONE home slot but `N` per-branch homes that
        // can't ride it — return `home == None` (the same by-value boundary
        // `bundle!`/`arc_split` document). The Checkout terminal re-arms per branch.
        let (value, deps) = self.collect(ec, mode)?;
        Ok((value, deps, None))
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        _mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // TERMINAL gather — the per-branch re-arm. Each branch runs its OWN
        // `gather_checkouts` (a single-output branch builds one `Checkout` with its
        // return home; a multi-output branch its tuple, every buffer homed), so
        // EVERY branch buffer re-arms its origin cell on drop and the fan-out
        // replays. Branches pipeline; join their wait-lists into one marker. The
        // dynamic-`Vec` analog of `bundle!`'s per-branch `gather_checkouts` delegation.
        let n = self.ops.len();
        let mut branch_deps: Vec<Deps> = Vec::with_capacity(n);
        let mut cos: Vec<U::Checkouts> = Vec::with_capacity(n);
        for op in &self.ops {
            let (co, d) = op.gather_checkouts(ec, ExecMode::Pipelined)?;
            cos.push(co);
            branch_deps.push(d);
        }
        let joined = join_marker(ec, &branch_deps)?;
        Ok((cos, joined))
    }

    fn reclaim_undelivered(&self) {
        // Mid-graph mop-up: a downstream consumer of the collapsed `Vec` handle may
        // discard branch outputs. Delegate to each branch's `reclaim_undelivered`
        // so any undelivered homed buffer returns to its cell for the next run
        // (already-drained pipes are no-ops).
        for op in &self.ops {
            op.reclaim_undelivered();
        }
    }

    fn check_ready(&self) -> Result<()> {
        // `execute` `collect`s every branch op — pre-check each, fail-fast.
        for op in &self.ops {
            op.check_ready()?;
        }
        Ok(())
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn invalidate_cbs(&self, mutated: &std::collections::BTreeSet<usize>) {
        cb_cache_invalidate(&self.cb_cache, mutated);
        for op in &self.ops {
            op.invalidate_cbs(mutated);
        }
    }

    fn collect_cb_ids(&self, out: &mut Vec<usize>) {
        cb_cache_collect_id(&self.cb_cache, out);
        for op in &self.ops {
            op.collect_cb_ids(out);
        }
    }

    fn cb_addable(&self) -> bool {
        // No command of its own — CB-addable iff EVERY branch is (mirrors bundle).
        // Empty fan-out records nothing → not addable (cbable_weight 0 gates it too).
        !self.ops.is_empty() && self.ops.iter().all(|op| op.cb_addable())
    }

    fn cbable_weight(&self) -> usize {
        // Sum of the branches' weights.
        self.ops.iter().map(|op| op.cbable_weight()).sum()
    }

    fn cb_restamp(&self, evs: &[Dep]) {
        // Single output pipe (the Vec) — stamp the CB completion event onto it.
        if let Some((v, _d)) = self.out.take() {
            self.out.put(v, Vec::from(evs));
        }
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("fan_out[{}]", self.ops.len()));
    }

    fn contains_host_seam(&self) -> bool {
        self.ops.iter().any(|op| op.contains_host_seam())
    }
}

/// Allocate a zero-initialised `DeviceSlice<T, M>` of `len` elements. Eager
/// leaf: produces a usable buffer, no upstream input. (`alloc_zero` is
/// synchronous internally, so it carries no in-flight events.)
pub struct AllocZero<T, M: MemMode = ReadWrite> {
    len: usize,
    out: Pipe<DeviceSlice<T, M>>,
    _t: PhantomData<fn() -> T>,
}

/// Build a zero-init alloc leaf with the **default [`ReadWrite`] marker** — no
/// turbofish: `alloc_zero(N)`. For a non-default marker use [`alloc_zero_as`]
/// with a marker witness (`alloc_zero_as(N, HostReadOnly)`).
pub fn alloc_zero<T>(len: usize) -> AllocZero<T, ReadWrite>
where
    T: Copy + Default + Send + Sync + 'static,
{
    alloc_zero_as(len, ReadWrite)
}

/// Build a zero-init alloc leaf with an **explicit access marker**, inferred
/// from the `marker` witness — no turbofish: `alloc_zero_as(N, HostReadOnly)`.
/// The default-marker shorthand is [`alloc_zero`].
pub fn alloc_zero_as<T, M>(len: usize, marker: M) -> AllocZero<T, M>
where
    T: Copy + Default + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    let _ = marker;
    AllocZero {
        len,
        out: Pipe::new(),
        _t: PhantomData,
    }
}

impl<T, M> DeviceOp for AllocZero<T, M>
where
    T: Copy + Default + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // alloc_zero is synchronous internally; no in-flight event, mode N/A.
        let buf = DeviceSlice::<T, M>::alloc_zero(ec.context(), self.len)?;
        self.out.put(buf, Deps::new());
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push(format!("alloc_zero(len={})", self.len));
    }
}

// ═══ Tail combinators: device routing, host seam, profiling, async terminal ═══

// ── DeviceTarget: a device picked either concretely or by context index ──

/// How a routing op (`OnDevice` / `TransferToDevice`) names its target device:
/// either a concrete [`Device`](crate::Device) resolved at build, or an index
/// into the running [`ExecutionContext`]'s `context().devices()`, resolved at
/// execute. The index form is what lets device-by-index routing be expressed
/// structurally (no execute-time closure), so the host-seam gate sees through it.
pub(crate) enum DeviceTarget {
    Concrete(crate::Device),
    Index(usize),
}

// ── OnDevice: re-point the op at a different device's queue at execute ──

/// Route `source`'s `execute` to a **different** device's default
/// out-of-order queue — built by [`on_device`](DeviceOpExt::on_device) (concrete
/// device) or [`on_device_at`](DeviceOpExt::on_device_at) (device-by-index).
///
/// No user closure: at execute it resolves the target device (concrete, or by
/// index into the running context's device list), then its default queue, builds
/// a sibling [`ExecutionContext`] (same context + same host-error slot, different
/// device + queue), and runs `source` against it. The source's events are valid
/// across queues of the same context, so downstream stages on the parent's queue
/// can wait on them cross-device.
pub struct OnDevice<S: DeviceOp> {
    pub(crate) source: S,
    pub(crate) target: DeviceTarget,
    pub(crate) out: Pipe<S::Output>,
}

impl<S, S2> DeviceOp for OnDevice<S>
where
    S: DeviceOp<Output = S2>,
    S2: Send,
{
    type Output = S::Output;

    fn output_pipe(&self) -> Option<Pipe<S::Output>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, parent: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // Resolve the target device (concrete, or by index into the running
        // context's device list) before anything else; the rest is unchanged.
        let device = match &self.target {
            DeviceTarget::Concrete(d) => d.clone(),
            DeviceTarget::Index(i) => parent.context().devices()[*i].clone(),
        };
        // Resolve the target queue from the running context (cached, so the
        // terminal's flush_all_outoforder_queues picks it up).
        let target_q = parent.context().default_outoforder_queue(&device)?;
        // Sibling EC: same context + same host-error slot, different device +
        // queue. `target_q` lives on this frame; its `.raw()` borrows for the
        // inner execute().
        let child = ExecutionContext::with_host_error_slot(
            parent.context(),
            device.clone(),
            target_q.raw(),
            parent.host_error_slot(),
            parent.start_dep(),
            parent.workers_handle(),
            parent.sp_edges_handle(),
            parent.cb_reach_handle(),
        );
        // Gather the source against the child EC via `collect` (any arity). The
        // routed sub-chain collapses to OnDevice's single output pipe, so any
        // home a concrete head carried is not threaded across the routing boundary
        // in step (a) (a routed concrete buffer is read via `into_inner`, not
        // auto-re-armed) — the same boundary as the copy/transform ops.
        let (value, deps) = self.source.collect(&child, mode)?;
        self.out.put(value, deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("on_device".into());
    }

    fn contains_host_seam(&self) -> bool {
        self.source.contains_host_seam()
    }
}

// ── AndThenHost / AndThenHostWithContext: the host seam ──

/// Run a host closure on a borrowed [`Mappable::View`](crate::mappable::Mappable::View) of the upstream output,
/// in chain order — built by [`and_then_host`](DeviceOpExt::and_then_host).
///
/// ## In-queue, worker-thread (NOT submit-thread)
///
/// At execute (on the submitting thread, all non-blocking): gather the source,
/// enqueue maps for its value (wait-list = its upstream events), create a user
/// event downstream gates on, enqueue the unmaps gated on that user event, then
/// **spawn a worker thread** and return immediately — the chain continues at
/// submit time, the host stage overlaps pipelined device work.
///
/// The worker waits the map events (host-side, on its own thread), runs the
/// closure on the view inside `catch_unwind` (mutations persist via the unmap),
/// and signals the user event: `CL_COMPLETE` on success, negative on closure
/// `Err` / panic (→ `Error::HostPanic`) / map failure. `Output = S::Output`
/// (the value passes through unchanged).
///
/// **Errors** are stashed in the chain-wide host-error slot on the
/// `ExecutionContext` and the user event is signalled
/// negative; the status cascades through the in-queue dependency graph and the
/// terminal (`sync` / `run`) prefers the stashed rich variant over the
/// `Error::OpenCl(-1)` cascade. On error the worker forces a defensive
/// synchronous unmap so the buffer is left clean. This mirrors the old
/// closure-layer `and_then_host.rs` exactly. (Distinct from any future
/// host-VALUE seam, which would be pure host compute with no map.)
pub struct AndThenHost<S: DeviceOp, F>
where
    S::Output: crate::mappable::Mappable,
    S::Checkouts: SeamScatter<Value = S::Output>,
{
    pub(crate) source: S,
    pub(crate) f: Arc<F>,
    // The per-branch, pipe-shaped downstream handle — `Pipe<O>` for a
    // single-output source (the pre-#212 default), a tuple of pipes for a bundle /
    // multi-output source. `execute` scatters the seam-mutated value+homes into
    // these, so downstream can route each written branch to its own kernel AND
    // every branch re-homes across replays. Owned (not `Pipe::new()` per run) so
    // `handle()` hands out stable pipe identities.
    pub(crate) handle: <S::Checkouts as SeamScatter>::Handle,
}

/// Like [`AndThenHost`] but the closure also receives `&Context` — built by
/// [`and_then_host_with_context`](DeviceOpExt::and_then_host_with_context).
pub struct AndThenHostWithContext<S: DeviceOp, F>
where
    S::Output: crate::mappable::Mappable,
    S::Checkouts: SeamScatter<Value = S::Output>,
{
    pub(crate) source: S,
    pub(crate) f: Arc<F>,
    // See `AndThenHost::handle`.
    pub(crate) handle: <S::Checkouts as SeamScatter>::Handle,
}

/// Shared body for the host seam: enqueue maps for the source value (wait-list =
/// its upstream events), create a user event downstream gates on, enqueue the
/// unmaps (gated on the user event), then **spawn a worker thread** that waits
/// the map events, runs `host_call` on the view, and signals the user event.
/// Returns the (unchanged) value + the unmap events as deps **without blocking
/// the submitting thread** — so the host stage sits *in-queue* and overlaps
/// pipelined device work. This is the async machinery ported from the old
/// closure-layer `and_then_host.rs`; running it synchronously was a regression.
///
/// ## Error / abort path — TWO user events
///
/// Errors (closure `Err`, panic → `HostPanic`, map-wait failure) are stashed in
/// the chain-wide host-error slot (first-writer-wins). The slot — **not** any
/// cl_event status nor the terminal's blocking return code — is the
/// authoritative caller-facing error channel: the terminal (`sync`/`run`/async
/// poll) checks it even on a "successful" wait and surfaces the rich error.
///
/// The seam is gated by **two** user events, because one event cannot do both
/// jobs at once:
/// - **`fire`** gates the unmaps and is **always** completed `CL_COMPLETE`
///   (success *and* error). Each buffer is unmapped exactly once, cleanly. We do
///   NOT additionally issue a host-side defensive unmap — a *second* unmap on an
///   already-unmapped buffer (legacy Intel NEO decrements map-count at enqueue,
///   not execution) returns `CL_INVALID_VALUE` and corrupts queue state. Folding
///   the defensive unmap away is the concrete bug this two-event split fixes vs.
///   the previous single-event design.
/// - **`proceed`** gates downstream and, on success, is `CL_COMPLETE`. On error
///   it is completed with a negative status to abort downstream device work.
///
/// ### Error path: the start-gate makes the negative `proceed` driver-safe
///
/// Completing a *wait-list* user event with a negative status to abort the
/// downstream was historically unreliable: on legacy Intel NEO a blocking
/// transfer parked on the event could lose its wakeup if the negative status
/// landed in the wait-commit window → deadlock. The fix (landed, see the
/// [`contains_host_seam`](DeviceOp::contains_host_seam) and terminal docs) is the
/// **start-gate**: the waiting terminals (`wait_on`/`sync`, `run`) gate the WHOLE
/// graph on a `start` user event and release it only after everything — including
/// the terminal marker — is enqueued, so a negative `proceed` can never race a
/// downstream command's wait-commit. NEO + rusticl are clean; the error-path
/// tests run normally (no `#[ignore]`).
///
/// Driver note: on pocl the concurrent error cascade exercised three distinct
/// pocl-internal event-handling races (an already-failed dependency running
/// anyway, a `clSetEventCallback` inline-callback use-after-free, and a
/// concurrent double-finish of a multi-dependency event). Those are pocl bugs,
/// fixed in `bricevideau-ai/pocl` (PRs #2214/#2215/#2216) — not claspr
/// correctness issues. See NOTES → Concerns for the root causes.
fn run_host_seam<O, F>(
    source_value: O,
    source_deps: Deps,
    ec: &ExecutionContext<'_>,
    host_call: F,
) -> Result<(O, Deps)>
where
    O: crate::mappable::Mappable,
    F: for<'a> FnOnce(<O as crate::mappable::Mappable>::View<'a>) -> Result<()> + Send + 'static,
{
    use crate::mappable::Mappable;
    use crate::{Launcher, complete_user_event, create_user_event};
    use std::sync::Arc;

    let q = ec.cl_queue();
    // Non-blocking maps with the upstream events as wait-list — the worker waits
    // these (host-side, on its own thread) before reading the mapped memory.
    let source_cl: Vec<crate::cl_event> = source_deps.iter().map(|d| d.as_ref().get()).collect();
    let (mut handle, map_events) = source_value.map(q, &source_cl)?;

    // `fire` gates the unmaps (always completed CL_COMPLETE — one clean unmap per
    // buffer). `proceed` gates downstream and carries the abort. See the doc
    // comment above for why these must be two distinct events.
    let fire = Arc::new(create_user_event(ec.context())?);
    let proceed = Arc::new(create_user_event(ec.context())?);

    // Enqueue the unmaps gated on `fire`. After this point we MUST complete BOTH
    // user events before any early return, or the queue would wait forever.
    let unmap_events = match <O as Mappable>::enqueue_unmap(&mut handle, q, &[fire.get()]) {
        Ok(evs) => evs,
        Err(e) => {
            let _ = complete_user_event(&fire, -1);
            let _ = complete_user_event(&proceed, -1);
            return Err(e);
        }
    };

    // Spawn the worker. It owns the handle, the map events, the source events
    // (for upstream-error short-circuit), both user-event Arc clones, the chain's
    // host-error slot, and the closure.
    let worker_fire = Arc::clone(&fire);
    let worker_proceed = Arc::clone(&proceed);
    let worker_host_error = ec.host_error_slot();
    let handle = std::thread::spawn(move || {
        let (status, handle, rust_err) =
            run_host_worker::<O, F>(handle, map_events, source_deps, host_call);
        // Stash the rich Rust error first (first-writer-wins — a concurrent
        // failing worker in the same bundle/fan-out may already have written).
        if let Some(err) = rust_err {
            let mut slot = worker_host_error.lock().unwrap();
            if slot.is_none() {
                *slot = Some(err);
            }
        }
        // ALWAYS fire the unmaps cleanly (single unmap per buffer). No defensive
        // unmap — that double-unmap is what corrupts legacy NEO.
        let _ = complete_user_event(&worker_fire, opencl3::event::CL_COMPLETE);
        // Allow / abort downstream device work via `proceed` (negative on error).
        // With the start gate (the whole graph is enqueued before `start` is
        // released), a negative `proceed` can no longer race a downstream blocking
        // transfer's wait-commit on legacy NEO.
        let _ = complete_user_event(&worker_proceed, status);
        // `handle` drops here: `enqueue_unmap` succeeded so its `unmap_enqueued`
        // flag is set and Drop is a no-op — the gated unmap fired via `fire`.
        drop(handle);
    });
    // Hand the worker to the EC so the terminal joins it AFTER the device wait —
    // no detached worker whose CL calls (signal events, drop retained queue) race
    // the caller dropping the Context.
    ec.push_worker(handle);

    // Downstream gates on ALL unmap events PLUS `proceed`. When the output has no
    // buffers (scalar / unit), there are no unmaps and `fire` gates nothing — but
    // `proceed` still gives downstream its proceed/abort gate.
    let mut deps_out: Deps = unmap_events.into_iter().map(wrap_event).collect();
    deps_out.push(proceed);
    Ok((source_value, deps_out))
}

/// Has `ev` been COMMITTED to a terminal error/cancellation state? Reads the
/// event's `command_execution_status` (the authoritative committed value, unlike
/// the `clWaitForEvents` return code which can lose the abort to a legacy-NEO
/// wakeup race) and reports `true` when it is negative — i.e. the command (a
/// device command, or an upstream host seam's `proceed` user event) terminated
/// abnormally. A query failure is treated as "not cancelled" (conservative: we
/// then fall through to the normal closure path rather than spuriously aborting).
fn event_is_cancelled(ev: &crate::Event) -> bool {
    ev.command_execution_status().map(|s| s.0).unwrap_or(0) < 0
}

/// Worker body for [`run_host_seam`]. Waits the source + map events, runs the
/// closure under `catch_unwind`, and returns `(status, handle, optional rich
/// error)`. The caller stashes the error in the chain host-error slot, always
/// completes the `fire` event (so the unmaps run cleanly), and forwards `status`
/// to the `proceed` event (negative aborts downstream). `status` is
/// `CL_COMPLETE` on success, negative otherwise. The returned `handle` is
/// dropped by the caller as a no-op (its unmap already fired via `fire`).
fn run_host_worker<O, F>(
    mut handle: O::MapHandle,
    map_events: Vec<crate::Event>,
    source_deps: Deps,
    host_call: F,
) -> (i32, O::MapHandle, Option<Error>)
where
    O: crate::mappable::Mappable,
    F: for<'a> FnOnce(<O as crate::mappable::Mappable>::View<'a>) -> Result<()> + Send + 'static,
{
    use crate::mappable::Mappable;
    use opencl3::event::CL_COMPLETE;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    // CHAINED-CANCEL via the EVENT DEPENDENCY (not a host-side slot — that races
    // the upstream worker's stash). An upstream host seam that fails completes its
    // `proceed` user event with a NEGATIVE status; that `proceed` is in THIS
    // seam's `source_deps` (it is the upstream op's output dep). So once the
    // upstream cancels, the cancellation is carried by the EVENT, and this seam
    // short-circuits — DO NOT run its closure — then completes its own `proceed`
    // negative so the rest of the chain cancels too.
    //
    // Robustness on legacy NEO: we do NOT trust `clWaitForEvents`' RETURN code to
    // report the abort (a lost-wakeup-style race can let the wait return
    // CL_SUCCESS even when the event was set negative — the same NEO race the
    // start-gate fixes for device commands, but here on a host-side wait). After
    // waiting, we re-read the event's COMMITTED `command_execution_status`: a
    // negative value is the authoritative "this command terminated in error"
    // signal and is not subject to the wakeup race. That negative status is what
    // we treat as cancellation.
    //
    // Source events first: a negative committed status here is an upstream chain
    // abort, not a fault of this seam — short-circuit WITHOUT stashing (the
    // upstream already stashed the first/authoritative error).
    for ev in &source_deps {
        let _ = ev.as_ref().wait();
        if event_is_cancelled(ev.as_ref()) {
            return (-1, handle, None);
        }
    }
    // Map events: enqueued over the source events, so an upstream cancel also
    // errors these. The source check above already caught the upstream-cancel
    // case, so a negative status here means a real map failure — stash the cause.
    for ev in &map_events {
        let _ = ev.wait();
        if event_is_cancelled(ev) {
            // Best-effort fetch of the concrete code; fall back to a generic
            // exec-status error if the query itself fails.
            let code = ev
                .command_execution_status()
                .map(|s| s.0)
                .unwrap_or(opencl3::error_codes::CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
            return (
                -1,
                handle,
                Some(Error::OpenCl(opencl3::error_codes::ClError(code))),
            );
        }
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let view = <O as Mappable>::view(&mut handle);
        host_call(view)
    }));
    match result {
        Ok(Ok(())) => (CL_COMPLETE, handle, None),
        Ok(Err(rust_err)) => (-1, handle, Some(rust_err)),
        Err(panic) => {
            // `catch_unwind` yields `Box<dyn Any + Send>`; the payload is
            // typically `&'static str` (`panic!("lit")`) or `String`
            // (`panic!("{}", x)`). Anything else gets a placeholder.
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            (-1, handle, Some(Error::HostPanic(msg)))
        }
    }
}

impl<S, F> DeviceOp for AndThenHost<S, F>
where
    S: DeviceOp,
    S::Output: crate::mappable::Mappable,
    // The source's terminal `Checkouts` must SPLIT into the assembled tuple value
    // (`S::Output`, what the seam maps) plus its per-branch homes, and REASSEMBLE
    // from a (seam-mutated) value + those homes. For a single-output source this
    // is `Checkout<S::Output>` (identity split — the #211 path stays byte-for-byte
    // the same); for a bundle / multi-output source it is that source's per-branch
    // `Checkouts` tuple, split/reassembled recursively (any arity, any nesting).
    S::Checkouts: CheckoutSplit<Value = S::Output>,
    // MID-GRAPH re-scatter (#212 completion): the same per-branch structure, but
    // exposed as element PIPES downstream + re-homed via `execute`. Single-output
    // → `Handle = Pipe<O>` (pre-#212 default, byte-identical); multi-output → a
    // tuple of pipes, so written branches route to separate downstream kernels AND
    // re-home across replays.
    S::Checkouts: SeamScatter<Value = S::Output>,
    // The source's own `gather_checkouts` needs this bound too — a bundle/multi-
    // output source's `Checkouts` satisfies it via the recursive `FromCheckout`
    // family, a single-output source via the identity impl.
    S::Checkouts: FromCheckout<S::Output>,
    F: for<'a> Fn(<S::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
        + Send
        + Sync
        + 'static,
{
    type Output = S::Output;
    // The downstream-facing handle is the source's per-branch PIPE shape (via
    // `SeamScatter::Handle`): `Pipe<O>` for a single-output source (unchanged), a
    // tuple of pipes for a bundle/multi-output source — so `and_then(|(a, b)| …)`
    // can route each written branch to its own kernel.
    type Handle = <S::Checkouts as SeamScatter>::Handle;
    // The seam's terminal result IS the source's — one `Checkout` for a
    // single-output source, a per-branch tuple for a bundle/multi-output source.
    // The seam re-threads EACH branch's home (via `CheckoutSplit`) so every branch
    // re-arms its origin cell across `sync`s — the multi-home replay a single
    // collapsed `collect_home` slot cannot carry.
    type Checkouts = S::Checkouts;

    fn output_pipe(&self) -> Option<Pipe<S::Output>> {
        // Single-output: `Some` of the storage pipe; multi-output: `None`
        // (storage is the element pipes). See `SeamScatter::output_pipe_view`.
        <S::Checkouts as SeamScatter>::output_pipe_view(&self.handle)
    }

    fn handle(&self) -> Self::Handle {
        self.handle.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // MID-GRAPH path (the seam nested as an `and_then` source, or async `run`).
        // Gather the source PER-BRANCH via its own `gather_checkouts` (a
        // single-output source builds one `Checkout`; a bundle/multi-output source
        // delegates to each branch, threading every branch's per-buffer return
        // home) — the SAME per-branch gather the terminal uses. Then SPLIT into the
        // assembled value (mapped + handed to the closure) + per-branch homes,
        // run the seam, and RE-SCATTER each written-back branch (value+home) into
        // its OWN element pipe. So downstream reads each branch as its own pipe AND
        // every branch re-homes on drop — the mid-graph multi-home replay (#212
        // completion). A single-output source scatters into one pipe with its home
        // preserved (byte-identical to the pre-#212 `collect_home` + `put_home`).
        let (src_cos, deps) = cb_gather_child(&self.source, ec, ExecMode::Pipelined)?;
        // Precise-invalidation reach across the seam (single-output): the seam maps
        // its source's buffer to host in place and hands the SAME `cl_mem` onward, so
        // a downstream CB (region 2) bakes a buffer that still traces to the source's
        // slot origins. `cb_gather_child` above just ran region 1's Build pass, which
        // populated the source output cell's reach; forward it onto the seam's output
        // cell so region 2's later Build pass finds it via `cb_reach_of` (the reach map
        // is SHARED across CB boundaries, unlike sync points). Multi-output seams
        // re-scatter per branch (their per-branch reach is not threaded here — deferred
        // until a multi-output-across-seam mutable-CB case needs it; immutable
        // invalidation stays correct because region 1's CB already `note_slot`'d it).
        let src_cell = self.source.output_pipe().map(|p| p.cell_id());
        if let Some(dc) = self.output_pipe().map(|p| p.cell_id()) {
            cb_forward_reach(ec, None, src_cell, dc);
        }
        let (value, homes) = src_cos.split();
        // Reusable: `Arc::clone` the closure so the per-run worker thread gets
        // its OWN owned handle to move in (it runs off the submitting thread).
        // `run_host_seam` keeps its `FnOnce` param — the clone is a fresh
        // one-shot callable per replay; the closure itself (`Fn`) re-runs.
        let f = Arc::clone(&self.f);
        let (out_value, out_deps) =
            run_host_seam::<S::Output, _>(value, deps, ec, move |view| (*f)(view))?;
        <S::Checkouts as SeamScatter>::scatter(&self.handle, out_value, homes, &out_deps);
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(S::Output, Deps)>
    where
        Self: Sized,
    {
        // BY-VALUE gather (async `run` / `into_output`): scatter via `execute`,
        // then reconstruct the assembled value by draining the element pipe(s).
        // Single-output drains one pipe (unchanged); multi-output reconstructs the
        // tuple, joining deps.
        self.execute(ec, mode)?;
        <S::Checkouts as SeamScatter>::reconstruct(&self.handle)
    }

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(S::Output, Deps, Option<BoxedHome<S::Output>>)>
    where
        Self: Sized,
        S::Output: Send + 'static,
    {
        // Home-preserving by-value gather: single-output preserves its one home
        // (the #211 nested-in-`and_then` re-arm); multi-output returns `home ==
        // None` (per-branch homes ride the Checkout / element-pipe path).
        self.execute(ec, mode)?;
        <S::Checkouts as SeamScatter>::reconstruct_home(&self.handle)
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        _mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // TERMINAL / CHECKOUT gather — the #212 pass-1 path, UNCHANGED (byte-for-
        // byte). Gather the source per-branch, split into value + homes, run the
        // seam, REASSEMBLE the checkouts re-threading each ORIGINAL home so every
        // branch re-arms on drop. (Distinct from `execute`, which re-scatters into
        // the seam's own element pipes for a DOWNSTREAM consumer; here the terminal
        // takes the checkouts directly.)
        let (src_cos, deps) = cb_gather_child(&self.source, ec, ExecMode::Pipelined)?;
        let (value, homes) = src_cos.split();
        let f = Arc::clone(&self.f);
        let (out_value, out_deps) =
            run_host_seam::<S::Output, _>(value, deps, ec, move |view| (*f)(view))?;
        let checkouts = <S::Checkouts as CheckoutSplit>::reassemble(out_value, homes);
        Ok((checkouts, out_deps))
    }

    fn reclaim_undelivered(&self) {
        // Mid-graph mop-up: a downstream `and_then` closure may discard some of the
        // seam's element pipes (e.g. keep only the written α, drop −α). Each was
        // filled by `execute`; drain + rehome any the consumer left, so those
        // branches re-arm their origin cells for the next run. Already-drained pipes
        // are no-ops. The source's own reclaim runs too (its outputs were consumed
        // into the seam's checkouts at `execute`, so this is a no-op there, but keep
        // the traversal complete).
        <S::Checkouts as SeamScatter>::reclaim(&self.handle);
        self.source.reclaim_undelivered();
    }

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
    }

    fn bind_slots(&self, binder: &mut SlotBinder) {
        // A host seam is a structural pass-through: recurse into the source so
        // `bind`/`call` reach any `slot!` cells the source op carries.
        self.source.bind_slots(binder);
    }

    fn invalidate_cbs(&self, mutated: &std::collections::BTreeSet<usize>) {
        // A seam has no cb_cache of its OWN, but its source (a pre-seam device
        // region) homes CBs — recurse so a mutated slot upstream of the seam clears
        // the region-1 CB. Without this, the default (own-cache-only) would miss it.
        self.source.invalidate_cbs(mutated);
    }

    fn collect_cb_ids(&self, out: &mut Vec<usize>) {
        self.source.collect_cb_ids(out);
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("and_then_host".into());
    }

    fn contains_host_seam(&self) -> bool {
        true
    }
}

impl<S, F> DeviceOp for AndThenHostWithContext<S, F>
where
    S: DeviceOp,
    S::Output: crate::mappable::Mappable,
    // Same `CheckoutSplit` + `SeamScatter` bounds as `AndThenHost` — see its impl
    // for the rationale (terminal split/reassemble + mid-graph re-scatter).
    // Single-output source keeps the #211 / pre-#212 paths identical.
    S::Checkouts: CheckoutSplit<Value = S::Output>,
    S::Checkouts: SeamScatter<Value = S::Output>,
    // The source's own `gather_checkouts` needs this bound too — a bundle/multi-
    // output source's `Checkouts` satisfies it via the recursive `FromCheckout`
    // family, a single-output source via the identity impl.
    S::Checkouts: FromCheckout<S::Output>,
    F: for<'a> Fn(&Context, <S::Output as crate::mappable::Mappable>::View<'a>) -> Result<()>
        + Send
        + Sync
        + 'static,
{
    type Output = S::Output;
    // See `AndThenHost` — per-branch pipe handle downstream, per-branch checkouts
    // at the terminal.
    type Handle = <S::Checkouts as SeamScatter>::Handle;
    type Checkouts = S::Checkouts;

    fn output_pipe(&self) -> Option<Pipe<S::Output>> {
        <S::Checkouts as SeamScatter>::output_pipe_view(&self.handle)
    }

    fn handle(&self) -> Self::Handle {
        self.handle.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // MID-GRAPH re-scatter — twin of `AndThenHost::execute` (see it for the full
        // rationale); the closure additionally gets `&Context`.
        let (src_cos, deps) = cb_gather_child(&self.source, ec, ExecMode::Pipelined)?;
        // Forward the pre-seam reach across the seam — see `AndThenHost::execute`.
        let src_cell = self.source.output_pipe().map(|p| p.cell_id());
        if let Some(dc) = self.output_pipe().map(|p| p.cell_id()) {
            cb_forward_reach(ec, None, src_cell, dc);
        }
        let (value, homes) = src_cos.split();
        // Reusable: `Arc::clone` the closure and clone a fresh `Context` per run,
        // then move both into a fresh one-shot callable for the worker thread.
        // The closure (`Fn`) re-runs on every replay; captures are borrowed via
        // the Arc rather than move-consumed.
        let f = Arc::clone(&self.f);
        let context = ec.context().clone();
        let (out_value, out_deps) =
            run_host_seam::<S::Output, _>(value, deps, ec, move |view| (*f)(&context, view))?;
        <S::Checkouts as SeamScatter>::scatter(&self.handle, out_value, homes, &out_deps);
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(S::Output, Deps)>
    where
        Self: Sized,
    {
        self.execute(ec, mode)?;
        <S::Checkouts as SeamScatter>::reconstruct(&self.handle)
    }

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(S::Output, Deps, Option<BoxedHome<S::Output>>)>
    where
        Self: Sized,
        S::Output: Send + 'static,
    {
        self.execute(ec, mode)?;
        <S::Checkouts as SeamScatter>::reconstruct_home(&self.handle)
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        _mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // TERMINAL / CHECKOUT gather — the #212 pass-1 path, UNCHANGED (twin of
        // `AndThenHost::gather_checkouts`; closure also gets `&Context`).
        let (src_cos, deps) = cb_gather_child(&self.source, ec, ExecMode::Pipelined)?;
        let (value, homes) = src_cos.split();
        let f = Arc::clone(&self.f);
        let context = ec.context().clone();
        let (out_value, out_deps) =
            run_host_seam::<S::Output, _>(value, deps, ec, move |view| (*f)(&context, view))?;
        let checkouts = <S::Checkouts as CheckoutSplit>::reassemble(out_value, homes);
        Ok((checkouts, out_deps))
    }

    fn reclaim_undelivered(&self) {
        // Mid-graph mop-up — twin of `AndThenHost::reclaim_undelivered`.
        <S::Checkouts as SeamScatter>::reclaim(&self.handle);
        self.source.reclaim_undelivered();
    }

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
    }

    fn bind_slots(&self, binder: &mut SlotBinder) {
        // Pass-through: recurse into the source so `bind`/`call` reach its slots.
        self.source.bind_slots(binder);
    }

    fn invalidate_cbs(&self, mutated: &std::collections::BTreeSet<usize>) {
        // See `AndThenHost::invalidate_cbs` — recurse into the pre-seam source's CBs.
        self.source.invalidate_cbs(mutated);
    }

    fn collect_cb_ids(&self, out: &mut Vec<usize>) {
        self.source.collect_cb_ids(out);
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("and_then_host_with_context".into());
    }

    fn contains_host_seam(&self) -> bool {
        true
    }
}

// ── Profiled: wall-clock timing for a sub-chain ────────────────────────

/// Times whatever the source op enqueued, registering a completion callback —
/// built by [`profiled`](DeviceProfileExt::profiled). Mirrors the old
/// closure-layer `Profiled`: at execute it runs the source, enqueues an
/// `clEnqueueMarkerWithWaitList` over the source's events, registers the user
/// callback on the marker via [`register_profiling_callback`](crate::register_profiling_callback), and forwards the
/// **same** value downstream with the marker as its deps (so anything after the
/// `.profiled()` waits on the marker, which subsumes the source's events).
///
/// Requires the chain's OOO queue to have `CL_QUEUE_PROFILING_ENABLE` (build
/// the [`Context`] with [`.profiling(true)`](crate::context::ContextBuilder::profiling));
/// otherwise `execute` returns [`Error::ProfilingDisabled`] up front (the
/// source op still ran — profiling is a host side-effect, not data flow).
///
/// **Reusable / replayable.** The callback is `Fn` (not `FnOnce`) behind an
/// `Arc`, so a `.profiled()` graph can be `sync`'d repeatedly — each replay
/// `Arc::clone`s the callback and re-registers a fresh one-shot shim on that run's
/// marker, so the callback fires once per run with that run's timestamps. Profiling
/// is a pure host side-effect (no rehoming value), so unlike the host seam there is
/// no home/checkout threading — this is strictly a subset of the `and_then_host`
/// reusability change.
pub struct Profiled<S: DeviceOp, F> {
    source: S,
    // Reusable: the callback is kept in an `Arc` and re-invoked on every replay
    // (each run boxes a fresh `FnOnce` shim that calls the `Fn`). Was a
    // `Mutex<Option<F>>` drained once — a one-shot that broke a second `sync`.
    cb: Arc<F>,
    out: Pipe<S::Output>,
}

/// Extension trait adding [`profiled`](Self::profiled) to every [`DeviceOp`].
/// Separate from [`DeviceOpExt`] to mirror the old layer's
/// `DeviceOperationProfileExt`. Blanket-implemented.
pub trait DeviceProfileExt: DeviceOp + Sized {
    /// Register `cb` to receive the wall-clock [`ProfilingInfo`](crate::ProfilingInfo) for everything
    /// `self` enqueued onto the chain's queue. The closure fires on an OpenCL
    /// callback thread when the marker event completes. See [`Profiled`].
    ///
    /// **Reusable / replayable.** `cb` is `Fn` (not `FnOnce`), so a `.profiled()`
    /// graph can be `sync`'d repeatedly — the callback re-fires each run with that
    /// run's timestamps (borrow / `Arc` / clone captures, don't move-consume them).
    fn profiled<F>(self, cb: F) -> Profiled<Self, F>
    where
        F: Fn(Result<crate::ProfilingInfo>) + Send + Sync + 'static,
    {
        Profiled {
            source: self,
            cb: Arc::new(cb),
            out: Pipe::new(),
        }
    }
}
impl<T: DeviceOp> DeviceProfileExt for T {}

impl<S, F> DeviceOp for Profiled<S, F>
where
    S: DeviceOp,
    // `collect_home` (used in `execute` to thread the source's return home so a
    // profiled graph replays) requires the output be `Send + 'static` — every real
    // buffer/scalar output is.
    S::Output: Send + 'static,
    F: Fn(Result<crate::ProfilingInfo>) + Send + Sync + 'static,
{
    // Profiling is a host side-effect; the chain's data flow is unchanged.
    type Output = S::Output;

    fn output_pipe(&self) -> Option<Pipe<S::Output>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        use crate::Launcher;
        // Gather the source WITH its return home (any arity) — profiling is a
        // transparent passthrough, so the source's home must ride through to the
        // terminal `Checkout` for the graph to REPLAY (else a caller-owned / minted
        // source buffer never rehomes and the 2nd `sync` reports "already lent").
        // A multi-output source collapses to `home == None` (the documented
        // by-value boundary) — profiling wraps a single logical output in practice.
        let (value, source_deps, home) = self.source.collect_home(ec, ExecMode::Pipelined)?;
        // Same up-front check as the old layer / Tier 1: the queue needs
        // profiling enabled before we waste a marker + callback registration.
        if (ec.cl_queue().properties()? & crate::CL_QUEUE_PROFILING_ENABLE) == 0 {
            return Err(Error::ProfilingDisabled);
        }
        // The marker waits for the source op's events, so the timestamps
        // reflect the source's wall-clock duration (first command queued to
        // last command finished).
        let wait_list: Vec<crate::cl_event> =
            source_deps.iter().map(|d| d.as_ref().get()).collect();
        // SAFETY: cl_event handles are valid — held by `source_deps` Arcs until
        // this call returns.
        let marker = unsafe { ec.cl_queue().enqueue_marker_with_wait_list(&wait_list) }
            .map_err(Error::OpenCl)?;
        // `source_deps` keeps the underlying cl_events alive across the
        // enqueue; safe to drop after.
        drop(source_deps);
        // Reusable: `Arc::clone` the callback and wrap it in a fresh one-shot
        // `FnOnce` shim per run. `register_profiling_callback` takes a boxed
        // `FnOnce` (fired exactly once when THIS run's marker completes); the
        // clone lets the underlying `Fn` re-fire on the next replay's marker.
        let cb = Arc::clone(&self.cb);
        crate::register_profiling_callback(&marker, Box::new(move |info| (*cb)(info)))?;
        // The marker becomes this op's completion event for downstream
        // chaining (it subsumes the source's events); thread the source's home so
        // the terminal `Checkout` rehomes it and the graph re-arms.
        self.out.put_home(value, vec![wrap_event(marker)], home);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.source.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        self.source.describe(out);
        out.push("profiled".into());
    }

    fn contains_host_seam(&self) -> bool {
        self.source.contains_host_seam()
    }
}

// ── DeviceChainFuture: the async `.run().await` terminal ────────────────

/// Future returned by [`DeviceOpExt::run`]. Resolves to `Result<T>` once the
/// chain's commands have all completed on the device (or immediately, with an
/// error, if the chain failed to submit or any host seam returned `Err`).
///
/// The future returned by the async terminal [`run`](DeviceOpExt::run). The host
/// seam (`run_host_seam`) runs its closure on a worker thread and stashes any
/// failure into the chain's host-error slot before signalling its user event with a
/// negative status. That status cascades into the trailing marker, so the
/// future's marker poll resolves with `Err`; [`Running`](Self::Running) then
/// prefers the stashed rich variant (closure `Err`, `HostPanic`) over the
/// `Error::OpenCl(-1)` cascade — mirroring the `sync` terminal.
#[cfg(feature = "async-events")]
pub enum DeviceChainFuture<T> {
    /// Chain failed during setup, `execute`, or marker enqueue. The error
    /// surfaces on the first `poll`.
    Errored(Option<Error>),
    /// Chain submitted successfully; waiting for the trailing marker event to
    /// complete. The host-side `Output` is already materialised (drained from
    /// the output pipe at `run` time); the future just gates *when* the caller
    /// sees it on whether the queue work is done. `host_error` is the chain's
    /// stash, preferred over the cl_event cascade if the marker resolves `Err`.
    Running {
        output: Option<T>,
        event_future: crate::EventFuture,
        host_error: std::sync::Arc<std::sync::Mutex<Option<Error>>>,
        /// Host-seam worker handles, joined once the marker resolves (so a
        /// worker's late CL calls finish before the caller can drop the
        /// `Context`). Empty for pure device graphs. Shared `Arc` with the EC
        /// that spawned them (including routed `on_device` sub-chains).
        workers: std::sync::Arc<std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>>,
    },
}

/// Drain and join every host-seam worker handle (no-op when empty). Mirrors
/// [`ExecutionContext::join_workers`] for the async terminal, which only has the
/// shared `Arc` (the EC dropped at `run` time).
#[cfg(feature = "async-events")]
fn join_chain_workers(
    workers: &std::sync::Arc<std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>>,
) {
    let handles: Vec<std::thread::JoinHandle<()>> = std::mem::take(&mut *workers.lock().unwrap());
    for h in handles {
        let _ = h.join();
    }
}

// `T: Unpin` covers every realistic chain output (`Vec<u8>`, `DeviceSlice<T>`,
// `Arc<T>`, tuples of those, ...) and lets us pin-project via the cheap
// `Pin::get_mut`. Mirrors `ChainFuture`'s bound.
#[cfg(feature = "async-events")]
impl<T: Unpin> std::future::Future for DeviceChainFuture<T> {
    type Output = Result<T>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        use std::task::Poll;
        let this = self.as_mut().get_mut();
        match this {
            DeviceChainFuture::Errored(slot) => Poll::Ready(Err(slot
                .take()
                .expect("DeviceChainFuture polled after Ready (Errored)"))),
            DeviceChainFuture::Running {
                output,
                event_future,
                host_error,
                workers,
            } => match std::pin::Pin::new(event_future).poll(cx) {
                Poll::Pending => Poll::Pending,
                // Marker resolved Err: a host worker (or a CL command) failed.
                // Prefer the rich Rust variant the worker stashed over the
                // cl_event cascade, mirroring `sync`.
                Poll::Ready(Err(e)) => {
                    // Marker done → all device work (and the seam's signals) are
                    // settled; join workers before returning.
                    join_chain_workers(workers);
                    Poll::Ready(Err(host_error.lock().unwrap().take().unwrap_or(e)))
                }
                // Even on a "successful" marker, a host worker may have stashed
                // an error the marker did NOT propagate: pocl's
                // `clEnqueueMarkerWithWaitList` does not cascade negative status
                // from a user event in its wait-list (it reports CL_COMPLETE while
                // the chain genuinely failed). A non-empty slot is itself the
                // failure signal. (Same handling as the old `ChainFuture`.)
                Poll::Ready(Ok(())) => {
                    join_chain_workers(workers);
                    if let Some(rust_err) = host_error.lock().unwrap().take() {
                        return Poll::Ready(Err(rust_err));
                    }
                    Poll::Ready(Ok(output
                        .take()
                        .expect("DeviceChainFuture polled after Ready (Running)")))
                }
            },
        }
    }
}

/// Crate-internal worker behind [`DeviceOpExt::run`]: build the
/// [`ExecutionContext`] (default OOO queue, like `sync`), gather via `collect` in
/// [`ExecMode::Pipelined`], enqueue a marker over the chain's deps (before
/// releasing the `start` gate, when present), and wrap it in an
/// [`EventFuture`](crate::EventFuture).
///
/// Synchronous-error paths invalidate the context's cached OOO queue.
#[cfg(feature = "async-events")]
pub(crate) fn run_eager_chain<Op>(chain: Op, context: &Context) -> DeviceChainFuture<Op::Output>
where
    Op: DeviceOp,
    Op::Output: Unpin,
{
    use crate::EventFutureExt;

    // Atomicity pre-pass: validate all inputs before any enqueue, mirroring wait_on.
    // Surfaced SYNCHRONOUSLY here (before the queue is acquired and any command is
    // enqueued), returned as an eager `Errored` future so the caller sees an
    // unsatisfiable-input error at `run()` time, not at poll/await.
    if let Err(e) = chain.check_ready() {
        return DeviceChainFuture::Errored(Some(e));
    }

    // 1. Pick the per-device default OOO queue (same as `sync`).
    let device = context.device().clone();
    let queue = match context.default_outoforder_queue(&device) {
        Ok(q) => q,
        Err(e) => return DeviceChainFuture::Errored(Some(e)),
    };
    let mut ec = ExecutionContext::new(context, device.clone(), queue.raw());
    // Clone the chain's host-error slot + worker-join list out before `ec` drops
    // — host-seam workers stash failures in the slot, and the future joins the
    // workers + reconciles errors at poll time.
    let host_error = ec.host_error_slot();
    let workers = ec.workers_handle();

    // START-GATE (only when the chain contains a host seam): create a `start`
    // user event, gate the whole graph on it, enqueue everything, then release
    // `start`. Mirrors the blocking terminal — closes the legacy NEO lost-wakeup
    // window. Pure device graphs skip this entirely (start = None, zero cost).
    let start = if chain.contains_host_seam() {
        match crate::create_user_event(context) {
            Ok(ev) => {
                ec.set_start(ev.get());
                Some(ev)
            }
            Err(e) => {
                drop(queue);
                context.invalidate_default_outoforder_queue(&device);
                return DeviceChainFuture::Errored(Some(e));
            }
        }
    } else {
        None
    };

    // 2-3. Run the chain non-blocking and gather its result via `collect` —
    //    the uniform gather seam. `collect` dispatches to the right per-op
    //    reconstruction (single OR multi-output: bundle*, arc_split, the copy
    //    pair all yield their reconstructed value + joined deps), so the async
    //    terminal supports every arity the blocking `sync` does. A host-seam
    //    setup error (map/unmap enqueue) still surfaces synchronously here; a
    //    host-CLOSURE failure surfaces at poll time via the host-error slot.
    let collected = chain.collect(&ec, ExecMode::Pipelined);

    // Helper: release the start-gate (if any) so gated commands can drain/abort
    // instead of waiting on `start` forever. Used on every exit path below.
    let release_start = |start: &Option<crate::Event>| {
        if let Some(start) = start {
            let _ = crate::complete_user_event(start, opencl3::event::CL_COMPLETE);
        }
    };

    let (output, deps) = match collected {
        Ok(pair) => pair,
        Err(e) => {
            // Setup failed before/while enqueueing — release the gate so any
            // already-enqueued commands drain, then join workers and surface the
            // error (prefer the seam's rich stash).
            release_start(&start);
            join_chain_workers(&workers);
            drop(queue);
            context.invalidate_default_outoforder_queue(&device);
            return DeviceChainFuture::Errored(Some(
                host_error.lock().unwrap().take().unwrap_or(e),
            ));
        }
    };

    // 4. Enqueue a marker over every event the chain produced. Precise
    //    wait-list — we don't penalise other work sharing this OOO queue.
    //    SAFETY: each `cl_event` is held alive by the `deps` Arc wrappers for
    //    the duration of this call; the marker enqueue retains them internally.
    //
    //    ORDER: enqueue the marker BEFORE releasing `start`. The marker is part
    //    of the chain's terminal join, so it must be inside the start-gate like
    //    every other command — otherwise its wait-list edges get wired against
    //    deps that may already be resolving/failing concurrently (the gate's
    //    whole point is that the *entire* graph is committed before anything
    //    runs). The marker enqueue is non-blocking, so wiring it while `start`
    //    is still held does not deadlock.
    let wait_list: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
    let marker = match unsafe { queue.raw().enqueue_marker_with_wait_list(&wait_list) } {
        Ok(ev) => ev,
        Err(code) => {
            release_start(&start);
            drop(deps);
            join_chain_workers(&workers);
            drop(queue);
            context.invalidate_default_outoforder_queue(&device);
            return DeviceChainFuture::Errored(Some(Error::OpenCl(code)));
        }
    };
    drop(deps);

    // The whole graph — including the terminal marker — is now enqueued and
    // gated on `start`. Release it so everything drains (or aborts via its own
    // `proceed`).
    release_start(&start);

    // 4a. clFlush — push every queue the chain touched without blocking.
    //     rusticl is spec-strict and keeps commands `CL_QUEUED` until an
    //     explicit flush, so the marker's `CL_COMPLETE` callback would never
    //     fire and the future would deadlock. flush_all also covers
    //     `.on_device(&dev_b)` chains whose commands land on non-primary queues.
    if let Err(e) = context.flush_all_outoforder_queues() {
        join_chain_workers(&workers);
        drop(queue);
        context.invalidate_default_outoforder_queue(&device);
        return DeviceChainFuture::Errored(Some(e));
    }
    // `start` (if any) has been completed; its retained dep refs keep the cl_event
    // alive in the wait-lists until those commands run. Safe to drop here.
    drop(start);

    // 5. Wrap the marker in the EventFuture machinery (clSetEventCallback).
    //    The future joins `workers` when the marker resolves.
    DeviceChainFuture::Running {
        output: Some(output),
        event_future: marker.into_future(),
        host_error,
        workers,
    }
}
