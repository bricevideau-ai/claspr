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
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

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
    pub(crate) fn with_host_error_slot<'a>(
        context: &'a Context,
        device: Device,
        cl_queue: &'a CommandQueue,
        host_error: Arc<Mutex<Option<Error>>>,
        start: Option<crate::cl_event>,
        workers: Arc<Mutex<Vec<JoinHandle<()>>>>,
    ) -> ExecutionContext<'a> {
        ExecutionContext {
            context,
            device,
            cl_queue,
            host_error,
            start,
            workers,
        }
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
