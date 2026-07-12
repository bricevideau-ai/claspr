//! [`ExecutionContext`] — what every [`DeviceOp`]'s `execute`
//! method receives.
//!
//! Carries the [`Context`], the current [`Device`], and a borrowed
//! [`CommandQueue`] to enqueue on. Implements [`Launcher`] so any
//! existing Tier 1 op (e.g. proc-macro-generated `kernels.foo(...)`)
//! composes directly inside a chain. Device-by-index routing is
//! expressed structurally via `on_device_at` / `transfer_to_device_at`:
//!
//! ```ignore
//! .and_then(move |buf| {
//!     // route the kernel onto the device at context index 1
//!     kernels.foo([N], buf, scalar).on_device_at(1)
//! })
//! ```
//!
//! [`DeviceOp`]: crate::DeviceOp
//! [`Context`]: crate::Context
//! [`Device`]: crate::Device
//! [`CommandQueue`]: opencl3::command_queue::CommandQueue
//! [`Launcher`]: crate::Launcher

use crate::{CommandQueue, Context, Device, Error, Launcher};
use opencl_sys::cl_sync_point_khr;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// The CB-internal marker edge-map (design v2): a producer inside a command
/// buffer registers its output command's [`cl_sync_point_khr`]s under its output
/// pipe's [`cell_id`](crate::Pipe::cell_id); a consumer inside the SAME CB looks
/// them up by its input's upstream cell id to build its `sync_point_wait_list`.
///
/// This is the "return markers UP" channel that the spec pairs with the
/// forwarded-CB "down" channel. It is legitimately AMBIENT (shared behind
/// `Arc<Mutex>`, keyed by globally-unique cell ids) rather than positional —
/// unlike the CB-visibility, which must be per-subtree. `cl_event` deps still
/// flow through the pipes; only these CB-internal markers ride here. Live for one
/// `sync` (a fresh `ExecutionContext` is built per terminal call).
pub type SyncPointEdges = Arc<Mutex<HashMap<usize, Vec<cl_sync_point_khr>>>>;

/// The command-buffer walk mode for a walk position (design v2, CB-as-execution-
/// mode). Threaded IMMUTABLY in each [`ExecutionContext`] value; positional
/// visibility comes from each recursion arm building its own child value (see
/// [`ExecutionContext::with_cb`]).
///
/// Three states because a REPLAY still has to re-walk the graph to LEND buffers
/// and build the terminal `Checkout`s (the buffers flow every run for stable
/// handles + rehoming) — it just must not re-ENQUEUE / re-ADD the device work the
/// cached CB already carries:
/// - [`Off`](CbWalk::Off) — no command buffer here; a device leaf ENQUEUES
///   normally (the current per-op path). A CB-capable node that sees `Off` and is
///   the outermost CB-eligible node of its subtree is the CB BOUNDARY: it
///   builds-or-replays a CB (in the terminal / at a host seam), forwarding `Build`
///   or `LendOnly` to its children.
/// - [`Build`](CbWalk::Build) — inside a CB being BUILT: a device leaf resolves
///   (lends) its buffer, ADDS its command to the builder (recording a
///   [`cl_sync_point_khr`]), and fills its output pipe with EMPTY `cl_event` deps
///   (ordering is the sync points, not events).
/// - [`LendOnly`](CbWalk::LendOnly) — inside a CB being REPLAYED: a device leaf
///   resolves (lends) its buffer and fills its output pipe with empty deps, but
///   ADDS NOTHING and ENQUEUES NOTHING (the cached CB does the work). This is what
///   lets a replay materialize buffers + build `Checkout`s without re-executing
///   device work — the double-execution hazard the superseded design hit,
///   dissolved by making replay a lend-only pass of the SAME walk.
#[derive(Clone, Copy)]
pub(crate) enum CbWalk<'a> {
    Off,
    Build {
        /// The live command buffer this subtree adds its commands to.
        builder: &'a crate::record::CbBuilder,
        /// The EXTERNAL `cl_event` dep accumulator for THIS command buffer (the
        /// event↔sync-point boundary). A leaf whose resolved input carries a
        /// NON-EMPTY `cl_event` wait-list — a producer OUTSIDE this CB (a host
        /// step, or the start gate) — pushes those events here; the homing node
        /// waits on them at `clEnqueueCommandBufferKHR`. Producers INSIDE the CB
        /// deposit EMPTY `cl_event` deps (their ordering is the CB-internal sync
        /// points), so nothing internal lands here. Owned on the boundary node's
        /// stack; fresh per CB, so nested CBs never mix external deps.
        ext: &'a Mutex<Vec<crate::eager::Dep>>,
    },
    LendOnly {
        /// See [`Build::ext`](CbWalk::Build) — the same external-dep accumulator on
        /// the replay pass (the cached CB still needs its external wait-list each
        /// replay, since a host step re-produces fresh events every run).
        ext: &'a Mutex<Vec<crate::eager::Dep>>,
    },
}

/// Execution-time environment for a [`DeviceOp`].
///
/// Built by [`DeviceOp::sync`] (and later, by the async terminal
/// in Phase 3.4); op authors don't construct this directly. The
/// `'ctx` lifetime is the lifetime of the borrow into the parent
/// [`Context`] and its per-device default OOO queue.
///
/// [`DeviceOp`]: crate::DeviceOp
/// [`DeviceOp::sync`]: crate::DeviceOpExt::sync
pub struct ExecutionContext<'ctx> {
    context: &'ctx Context,
    device: Device,
    cl_queue: &'ctx CommandQueue,
    /// Slot that `and_then_host` workers stash their failing
    /// [`Error`] into before signalling `clSetUserEventStatus(_, -1)`.
    /// Terminals read it after the marker event resolves with Err
    /// and prefer the rich variant over the CL cascade. `Arc<Mutex<_>>`
    /// because workers run on per-call threads. First-writer-wins
    /// when multiple `and_then_host`s in a bundle/fan-out fail
    /// concurrently — subsequent writers leave the slot alone.
    host_error: Arc<Mutex<Option<Error>>>,
    /// The **start gate** — a host-side user event that every *entry* leaf
    /// (a [`Concrete`](crate::Input::Concrete) input, i.e. a chain head with no
    /// upstream) threads into its `clEnqueue*` wait-list. `None` for pure device
    /// graphs (the zero-overhead fast path); `Some` only when the chain
    /// [`contains_host_seam`](crate::DeviceOp::contains_host_seam).
    ///
    /// The terminal creates it, sets it here, enqueues the WHOLE graph (now
    /// gated, so nothing runs yet), then completes it `CL_COMPLETE` to release
    /// the graph — only after every command is enqueued. This closes the legacy
    /// NEO lost-wakeup window where a host seam's negative `proceed` could race a
    /// downstream blocking transfer's wait-commit. Validated in
    /// `scratch/start_threaded.c`. Raw `cl_event` (not [`Event`](crate::Event)):
    /// it is owned by the terminal for the whole enqueue, and each entry leaf
    /// `clRetainEvent`s it independently when wrapping it as a dep.
    start: Option<crate::cl_event>,
    /// Host-seam worker [`JoinHandle`]s, joined at the terminal AFTER the device
    /// wait. `run_host_seam` pushes its worker here instead of detaching it, so
    /// no worker's CL calls (signalling `fire`/`proceed`, then
    /// `release_command_queue` on the worker's retained-queue drop) can race the
    /// caller dropping the [`Context`]. Shared `Arc` so a routed sub-chain's
    /// workers (via [`with_host_error_slot`](Self::with_host_error_slot)) are
    /// joined by the same terminal.
    workers: Arc<Mutex<Vec<JoinHandle<()>>>>,
    /// The **forwarded command buffer** for this walk position (design v2,
    /// CB-as-execution-mode). `Some` iff this op is currently INSIDE a command
    /// buffer being built — set by a CB-creating node for its children's
    /// sub-`ExecutionContext` (via [`with_cb`](Self::with_cb)), `None` otherwise.
    ///
    /// This is IMMUTABLE per `ExecutionContext` value — positional visibility comes
    /// from each recursion arm getting its OWN child `ExecutionContext`
    /// ([`Build`](CbWalk::Build)`(&builder)` for a creator building, unchanged `ec`
    /// for a forwarder / bundle sibling, [`Off`](CbWalk::Off) for a seam-boundary
    /// source). No ambient mutable "current CB" slot: two siblings with different CB
    /// visibility are two distinct `ExecutionContext` values. The borrow lifetime
    /// `'ctx` is the creating node's stack frame, which outlives its children's
    /// `execute` calls.
    cb: CbWalk<'ctx>,
    /// The CB-internal [`SyncPointEdges`] marker map — the "markers UP" channel.
    /// Shared across the whole walk (cloned into every child `ExecutionContext`),
    /// keyed by unique cell ids, so it is ambient by design (see
    /// [`SyncPointEdges`]).
    sp_edges: SyncPointEdges,
}

impl<'ctx> ExecutionContext<'ctx> {
    /// Construct an `ExecutionContext` bound to `device`'s default
    /// out-of-order queue from `context`. Crate-internal — terminals
    /// in [`crate::device_op`] call this.
    pub(crate) fn new(
        context: &'ctx Context,
        device: Device,
        cl_queue: &'ctx CommandQueue,
    ) -> Self {
        ExecutionContext {
            context,
            device,
            cl_queue,
            host_error: Arc::new(Mutex::new(None)),
            start: None,
            workers: Arc::new(Mutex::new(Vec::new())),
            cb: CbWalk::Off,
            sp_edges: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Sibling-constructor: build a fresh `ExecutionContext` that
    /// shares the host-error slot with an existing one but targets a
    /// different device + queue. Used by the [`OnDevice`](crate::OnDevice)
    /// combinator to route a sub-chain to a non-default device's queue
    /// without losing the chain-wide error stash.
    ///
    /// The new EC's lifetime `'a` is whatever lifetime the caller
    /// can provide for `context` and `cl_queue` — typically a
    /// stack-local `Arc<Queue>` whose `.raw()` is borrowed for the
    /// duration of the child op's `execute()`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_host_error_slot<'a>(
        context: &'a Context,
        device: Device,
        cl_queue: &'a CommandQueue,
        host_error: Arc<Mutex<Option<Error>>>,
        start: Option<crate::cl_event>,
        workers: Arc<Mutex<Vec<JoinHandle<()>>>>,
        sp_edges: SyncPointEdges,
    ) -> ExecutionContext<'a> {
        ExecutionContext {
            context,
            device,
            cl_queue,
            host_error,
            start,
            workers,
            // A routed sub-chain runs on a DIFFERENT queue, so it cannot share the
            // parent's command buffer (a CB is single-queue). It opens its own CB
            // if eligible; here it starts outside any CB.
            cb: CbWalk::Off,
            sp_edges,
        }
    }

    /// Build a child `ExecutionContext` identical to `self` but with the CB walk
    /// mode set to `cb` (design v2). This is the POSITIONAL CB-visibility
    /// mechanism: a CB-creating node calls `ec.with_cb(CbWalk::Build(&builder))`
    /// (or `LendOnly` on replay) for its children; a seam-boundary node calls
    /// `ec.with_cb(CbWalk::Off)` for its device source; a plain forwarder passes
    /// `ec` unchanged. Each returned value has an IMMUTABLE `cb` fixed for its whole
    /// subtree — no save/restore, so two siblings with different CB visibility are
    /// simply two distinct values.
    ///
    /// Shares the host-error slot, start gate, worker list, and sync-point edge map
    /// (all `Arc` clones / `Copy`), re-borrowing `context`/`cl_queue` for `'a`.
    pub(crate) fn with_cb<'a>(&'a self, cb: CbWalk<'a>) -> ExecutionContext<'a> {
        ExecutionContext {
            context: self.context,
            device: self.device.clone(),
            cl_queue: self.cl_queue,
            host_error: Arc::clone(&self.host_error),
            start: self.start,
            workers: Arc::clone(&self.workers),
            cb,
            sp_edges: Arc::clone(&self.sp_edges),
        }
    }

    /// The CB walk mode at this walk position. The per-node fork reads this to
    /// decide build / replay-lend / normal-enqueue. See [`with_cb`](Self::with_cb).
    pub(crate) fn cb(&self) -> CbWalk<'_> {
        self.cb
    }

    /// Look up the sync-point markers a producer registered under `cell_id` (its
    /// output pipe's [`cell_id`](crate::Pipe::cell_id)). Empty if the producer is
    /// outside this CB (an entry into the CB from outside — no CB-internal
    /// predecessor) or has not run yet. The CB-mode fork uses this as a leaf's
    /// `sync_point_wait_list`. `None` upstream cell (a concrete/slot input, no
    /// pipe) yields no markers.
    pub(crate) fn sp_lookup(&self, cell_id: Option<usize>) -> Vec<cl_sync_point_khr> {
        match cell_id {
            Some(id) => self
                .sp_edges
                .lock()
                .unwrap()
                .get(&id)
                .cloned()
                .unwrap_or_default(),
            None => Vec::new(),
        }
    }

    /// Register a producer's output command sync point(s) under its output pipe's
    /// `cell_id`, so a CB-internal consumer resolves them via [`sp_lookup`](Self::sp_lookup).
    pub(crate) fn sp_register(&self, cell_id: usize, sps: Vec<cl_sync_point_khr>) {
        self.sp_edges.lock().unwrap().insert(cell_id, sps);
    }

    /// `Arc` clone of the sync-point edge map, for the [`OnDevice`](crate::OnDevice)
    /// sibling EC constructor.
    pub(crate) fn sp_edges_handle(&self) -> SyncPointEdges {
        Arc::clone(&self.sp_edges)
    }

    /// `Arc` clone of the worker-join list, for the [`OnDevice`](crate::OnDevice)
    /// sibling EC so a routed sub-chain's host-seam workers are joined by the
    /// same terminal that owns the parent EC.
    pub(crate) fn workers_handle(&self) -> Arc<Mutex<Vec<JoinHandle<()>>>> {
        Arc::clone(&self.workers)
    }

    /// Cheap `Arc` clone of the host-error slot for an `and_then_host`
    /// worker to carry across `thread::spawn`. Workers populate it
    /// (first-writer-wins) before signalling negative user-event
    /// status; terminals drain it via [`take_host_error`](Self::take_host_error).
    pub(crate) fn host_error_slot(&self) -> Arc<Mutex<Option<Error>>> {
        Arc::clone(&self.host_error)
    }

    /// Take the stashed host error (if any). Crate-internal — called
    /// by terminals (`sync` / `run`'s poll) after the chain's events
    /// resolve with Err, so the original Rust variant surfaces instead
    /// of the `Error::OpenCl(-1)` cascade from the user-event signal.
    pub(crate) fn take_host_error(&self) -> Option<Error> {
        self.host_error.lock().unwrap().take()
    }

    /// Install the start gate (raw `cl_event`). Called by a terminal BEFORE it
    /// runs the graph, only when the chain
    /// [`contains_host_seam`](crate::DeviceOp::contains_host_seam). After this,
    /// every entry leaf's enqueue waits on `ev` (see [`start_dep`](Self::start_dep)),
    /// so nothing executes until the terminal completes `ev`.
    pub(crate) fn set_start(&mut self, ev: crate::cl_event) {
        self.start = Some(ev);
    }

    /// The start gate, if set. An entry leaf ([`Concrete`](crate::Input::Concrete)
    /// input) threads this into its `clEnqueue*` wait-list so the whole graph is
    /// committed before any of it runs.
    pub(crate) fn start_dep(&self) -> Option<crate::cl_event> {
        self.start
    }

    /// Push a host-seam worker handle to be joined at the terminal (after the
    /// device wait). [`run_host_seam`](crate::run_host_seam) calls this instead
    /// of detaching its worker.
    pub(crate) fn push_worker(&self, h: JoinHandle<()>) {
        self.workers.lock().unwrap().push(h);
    }

    /// Drain and join every host-seam worker. Called by the terminal AFTER the
    /// chain's device events have been waited on, so a worker's late CL calls
    /// (signalling its user events, then dropping its retained queue) complete
    /// before the caller can drop the [`Context`].
    pub(crate) fn join_workers(&self) {
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut *self.workers.lock().unwrap());
        for h in handles {
            // A worker panic is already surfaced as a `HostPanic` in the
            // host-error slot (the seam catches it); ignore the join error here.
            let _ = h.join();
        }
    }

    /// The [`Context`] this op-chain is running against.
    pub fn context(&self) -> &Context {
        self.context
    }

    /// The [`Device`] this op currently targets. Use
    /// [`DeviceOpExt::on_device`](crate::DeviceOpExt::on_device)
    /// to route a sub-chain to a different device's queue, or
    /// [`transfer_to_device`](crate::transfer_to_device()) to migrate
    /// the buffer between devices in the same context.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The chain's running context's full device list — same as
    /// `self.context().devices()`, surfaced here for ergonomics.
    pub fn devices(&self) -> &[Device] {
        self.context.devices()
    }

    /// Shortcut for `&self.context().devices()[i]`. Panics if `i` is
    /// out of range (mirrors slice indexing). For routing/transfer by
    /// device index, prefer the structural builders that resolve the
    /// index at execute:
    ///
    /// ```ignore
    /// .and_then(move |buf| kernels.foo([N], buf).on_device_at(1))
    /// ```
    pub fn device_at(&self, i: usize) -> &Device {
        &self.context.devices()[i]
    }
}

// Letting `ExecutionContext` itself act as a Launcher means a Tier 1
// op inside a Tier 2 closure (e.g. `kernels.foo(ctx, ...)`) routes
// transparently through the chain's queue — no need for the user to
// dig out the queue handle.
impl<'ctx> Launcher for ExecutionContext<'ctx> {
    fn cl_queue(&self) -> &CommandQueue {
        self.cl_queue
    }

    fn context(&self) -> &Context {
        self.context
    }
}
