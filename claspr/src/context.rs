//! [`Context`] — the OpenCL context wrapper, Arc-shared internally
//! so it's cheap to clone and `Send + Sync`.
//!
//! Each `Context` carries a per-device pair of lazy default queues
//! (in-order + out-of-order). Most user code never names a
//! [`Queue`] explicitly — passing `&ctx` to a
//! generated launch wrapper routes through the in-order default
//! for [`ctx.device()`](Context::device). Advanced callers reach
//! for [`Context::default_inorder_queue`] /
//! [`Context::default_outoforder_queue`] (per device, lazy) or
//! create explicit [`Queue<O>`] handles when
//! they want their own command stream.
//!
//! Profiling is opt-in on the [`ContextBuilder`]; the per-device
//! default queues — and any [`Queue::new`](crate::queue::Queue::new) /
//! [`Queue::on_device`](crate::queue::Queue::on_device) built later —
//! inherit it. Matches SYCL 2020's
//! `property::queue::enable_profiling` semantics.

use crate::device::Device;
use crate::error::{Error, Result};
use crate::queue::{InOrder, Launcher, OutOfOrder, Queue};
use opencl3::command_queue::CommandQueue;
use opencl3::device::{
    CL_DEVICE_SVM_ATOMICS, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER, CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
    CL_DEVICE_SVM_FINE_GRAIN_SYSTEM,
};
use opencl3::kernel::Kernel;
use opencl3::program::Program;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// OpenCL context + the devices it spans + lazy per-device default
/// queue pairs.
///
/// Cheap to clone (one `Arc` increment). `Send + Sync`. The OpenCL
/// ICD does its own internal refcount under the hood; this Rust
/// `Arc` is the only per-process refcount we add.
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

struct ContextInner {
    cl_context: opencl3::context::Context,
    /// All devices the context spans. `devices[0]` is the default
    /// (returned by `device()`); multi-device contexts add more.
    devices: Vec<Device>,
    /// Whether queues built from this context — both the per-device
    /// defaults below and any `Queue::new` / `Queue::on_device` the
    /// user constructs later — enable `CL_QUEUE_PROFILING_ENABLE`.
    profiling: bool,
    /// Per-device default queue pair. `queues[i]` corresponds to
    /// `devices[i]`. Lazily populated on first lookup except
    /// `queues[0].in_order` which is created at build time so the
    /// `Launcher::cl_queue` implementation for `&Context` can return
    /// a `&CommandQueue` infallibly.
    queues: Vec<DeviceQueues>,
    /// Sticky-error counter. `Drop` impls that discover an OpenCL
    /// release failure can't propagate it; they bump this instead so
    /// callers who care can audit via [`Context::error_count`].
    error_state: AtomicU32,
    /// Lazily-built program holding the built-in fill kernels (see
    /// [`crate::fill_kernel`]). Built on first device-fill use via
    /// [`Context::fill_program`]; cached for the context's lifetime.
    /// `None` until first device-fill — most contexts that only ever
    /// fill HostWritable buffers (runtime path) never build this.
    fill_program: OnceLock<Program>,
}

/// Lazy queue pair for one device in a [`Context`].
///
/// `in_order` is `OnceLock` — stable reference for the lifetime of
/// the context, never rebuilt. `out_of_order` is `Mutex<Option<Arc>>`
/// so it can be invalidated and rebuilt after a terminated command
/// renders the queue unusable. Per the OpenCL spec on command
/// execution status: "If the execution of a command is terminated,
/// the command-queue associated with this terminated command, and
/// the associated context (and all other command-queues in this
/// context) may no longer be available. The behavior of OpenCL API
/// calls that use this context (and command-queues associated with
/// this context) are now considered to be implementation-defined."
/// We hit this path whenever a host closure errors and
/// `claspr_async::AndThenHost` signals the user event with a
/// negative status. Some drivers (e.g. rusticl) make the queue
/// permanently propagate the negative status to subsequent
/// commands; others (e.g. pocl) keep it usable. Rebuilding the
/// queue on the next sync covers either case.
///
/// The mutex is touched at most once per `DeviceOperation::sync` /
/// `DeviceOperation::run` — both grab an `Arc<Queue>` once and pass
/// the raw `cl_command_queue` through to every enqueue call.
struct DeviceQueues {
    in_order: OnceLock<Queue<InOrder>>,
    out_of_order: Mutex<Option<Arc<Queue<OutOfOrder>>>>,
}

impl DeviceQueues {
    fn empty() -> Self {
        DeviceQueues {
            in_order: OnceLock::new(),
            out_of_order: Mutex::new(None),
        }
    }
}

/// Stand-in for the unstable [`OnceLock::get_or_try_init`]. Cheap
/// fast path on hit; race-safe on miss (if two threads call this
/// simultaneously, both may run `f`, but only one value survives —
/// callers see whichever one wins the `set` race).
fn once_lock_get_or_try_init<T, E, F>(cell: &OnceLock<T>, f: F) -> std::result::Result<&T, E>
where
    F: FnOnce() -> std::result::Result<T, E>,
{
    if let Some(v) = cell.get() {
        return Ok(v);
    }
    let v = f()?;
    // Discard our value if another thread won the race; either way
    // `cell.get()` then returns the stored value.
    let _ = cell.set(v);
    Ok(cell.get().expect("OnceLock just set"))
}

// SAFETY: cl_context, cl_command_queue, and cl_device_id are opaque
// handles. OpenCL API calls on them are thread-safe per the spec
// (CL §3.4.1).
unsafe impl Send for ContextInner {}
unsafe impl Sync for ContextInner {}

impl Context {
    // ── Builder + canned constructors ──────────────────────────────

    /// Start a [`ContextBuilder`]. The general entry point for
    /// constructing a context — accumulate devices, enable profiling
    /// if desired, and `.build()`.
    pub fn builder() -> ContextBuilder {
        ContextBuilder::new()
    }

    /// Build a context pinned to `device`, with profiling off.
    /// Shortcut for `Context::builder().device(device).build()`.
    pub fn for_device(device: &Device) -> Result<Self> {
        Self::builder().device(device).build()
    }

    /// Build a multi-device context with profiling off. All `devices`
    /// must come from the same platform (OpenCL spec requirement).
    /// Shortcut for `Context::builder().devices(devices).build()`.
    pub fn for_devices(devices: &[Device]) -> Result<Self> {
        Self::builder().devices(devices).build()
    }

    /// Pick the first available device of any type and build a
    /// context on it. Convenience for trivial single-device setups.
    pub fn any() -> Result<Self> {
        Self::for_device(&Device::any()?)
    }

    /// Pick a device with a SYCL-style scoring closure (highest
    /// score wins, negative excludes) and build a context on it.
    pub fn select<F>(score: F) -> Result<Self>
    where
        F: FnMut(&Device) -> i32,
    {
        Self::for_device(&Device::find(score)?)
    }

    // ── Devices ────────────────────────────────────────────────────

    /// The default device for this context — `devices()[0]`.
    /// For single-device contexts this is the only device.
    pub fn device(&self) -> &Device {
        &self.inner.devices[0]
    }

    /// Every device the context spans. Length 1 for contexts built
    /// via [`Context::for_device`], length N for multi-device
    /// contexts via [`Context::for_devices`].
    pub fn devices(&self) -> &[Device] {
        &self.inner.devices
    }

    // ── Default queues ─────────────────────────────────────────────

    /// Per-device default in-order queue (the Tier 1 default).
    /// Lazily created on first lookup; stable reference thereafter.
    ///
    /// `device` must be one of the devices the context was built
    /// with — otherwise returns [`Error::InvalidArgument`]. Honors
    /// the [`profiling`](Self::profiling) setting from the builder.
    pub fn default_inorder_queue(&self, device: &Device) -> Result<&Queue<InOrder>> {
        let idx = self.device_index(device)?;
        once_lock_get_or_try_init(&self.inner.queues[idx].in_order, || {
            Queue::<InOrder>::on_device(self, device)
        })
    }

    /// Per-device default out-of-order queue (the Tier 2 default).
    /// Lazily created on first lookup; subsequent calls return the
    /// same `Arc` clone — until
    /// [`invalidate_default_outoforder_queue`](Self::invalidate_default_outoforder_queue)
    /// is called (e.g. by a failed chain's `sync()`), after which the
    /// next call rebuilds the queue.
    ///
    /// Returns an owned `Arc<Queue>` rather than a borrow so the
    /// caller can hold it across the chain's execution without
    /// pinning the context's internal mutex. The mutex is acquired
    /// once per call.
    ///
    /// `device` must be one of the devices the context was built
    /// with — otherwise returns [`Error::InvalidArgument`]. Honors
    /// the [`profiling`](Self::profiling) setting from the builder.
    pub fn default_outoforder_queue(&self, device: &Device) -> Result<Arc<Queue<OutOfOrder>>> {
        let idx = self.device_index(device)?;
        let mut slot = self.inner.queues[idx]
            .out_of_order
            .lock()
            .expect("DeviceQueues out_of_order mutex poisoned");
        if let Some(q) = slot.as_ref() {
            return Ok(Arc::clone(q));
        }
        let q = Arc::new(Queue::<OutOfOrder>::on_device(self, device)?);
        *slot = Some(Arc::clone(&q));
        Ok(q)
    }

    /// Drop the cached default out-of-order queue for `device`, if
    /// any. The next call to
    /// [`default_outoforder_queue`](Self::default_outoforder_queue)
    /// will build a fresh one.
    ///
    /// Used by Tier 2 terminals (`DeviceOperation::sync` /
    /// `DeviceOperation::run`) on the error path. Per the OpenCL
    /// spec on command execution status, once a command is
    /// terminated (e.g. via `clSetUserEventStatus(_, -1)` from our
    /// `and_then_host` error path), the queue and context "may no
    /// longer be available" and the behavior of subsequent API
    /// calls on them is implementation-defined. Observed in
    /// practice: rusticl propagates the negative status to
    /// subsequent unrelated commands on the same queue; pocl keeps
    /// the queue usable. Rebuilding the queue cache on the next
    /// terminal is the defensive choice that works across both.
    ///
    /// `device` not being part of this context is silently ignored —
    /// the public guarantee is "next default_outoforder_queue rebuilds
    /// if applicable," not strict validation.
    pub fn invalidate_default_outoforder_queue(&self, device: &Device) {
        let Ok(idx) = self.device_index(device) else {
            return;
        };
        let mut slot = self.inner.queues[idx]
            .out_of_order
            .lock()
            .expect("DeviceQueues out_of_order mutex poisoned");
        *slot = None;
    }

    /// Flush every per-device out-of-order queue this context has
    /// lazily instantiated. No-op for devices whose queue has never
    /// been accessed (lazy construction in
    /// [`default_outoforder_queue`](Self::default_outoforder_queue)
    /// means the slot stays empty until first use).
    ///
    /// Used by `claspr-async`'s terminals (sync / run) to push
    /// multi-device chains on non-eager implementations: rusticl
    /// (spec-strict) keeps enqueued commands in `CL_QUEUED` until an
    /// explicit `clFlush`, so a chain that touches `dev_b`'s queue
    /// via `.on_device(&dev_b)` would deadlock at the trailing
    /// marker without this. pocl flushes eagerly so the call is a
    /// no-op there.
    ///
    /// Per-queue `clFlush` is non-blocking; the call returns once
    /// every touched queue has been pushed (it does NOT wait for
    /// commands to complete — that's the terminal's separate
    /// event-wait responsibility).
    pub fn flush_all_outoforder_queues(&self) -> Result<()> {
        for slot in &self.inner.queues {
            let guard = slot
                .out_of_order
                .lock()
                .expect("DeviceQueues out_of_order mutex poisoned");
            if let Some(q) = guard.as_ref() {
                q.raw().flush()?;
            }
        }
        Ok(())
    }

    /// Finish (`clFinish`: flush + wait-for-completion) every
    /// per-device out-of-order queue this context has lazily
    /// instantiated. No-op for slots that are still empty.
    ///
    /// **Not called by `claspr-async`'s terminals** — those would
    /// over-block on other users' work since the OOO queues are
    /// shared (cached per-device on the Context). Provided here as
    /// an explicit "drain everything on this context" primitive
    /// for shutdown / synchronisation points where the caller
    /// genuinely wants all in-flight commands done before
    /// proceeding.
    ///
    /// Synchronous: blocks until every cached OOO queue has drained.
    pub fn finish_all_outoforder_queues(&self) -> Result<()> {
        for slot in &self.inner.queues {
            let guard = slot
                .out_of_order
                .lock()
                .expect("DeviceQueues out_of_order mutex poisoned");
            if let Some(q) = guard.as_ref() {
                q.raw().finish()?;
            }
        }
        Ok(())
    }

    /// `true` if the context was built with `.profiling(true)`.
    /// Every default queue and every [`Queue::new`](crate::queue::Queue::new) /
    /// [`Queue::on_device`](crate::queue::Queue::on_device) built off
    /// this context inherits the flag.
    pub fn profiling(&self) -> bool {
        self.inner.profiling
    }

    /// Position of `device` in [`devices`](Self::devices) by handle
    /// identity (raw `cl_device_id`). Used by the default-queue
    /// accessors to index into the per-device queue pair.
    fn device_index(&self, device: &Device) -> Result<usize> {
        self.inner
            .devices
            .iter()
            .position(|d| d.raw_id() == device.raw_id())
            .ok_or(Error::InvalidArgument("device is not part of this Context"))
    }

    // ── Raw escape hatches ─────────────────────────────────────────

    /// Borrow the underlying [`opencl3::context::Context`] for
    /// operations claspr doesn't surface yet.
    pub fn raw_context(&self) -> &opencl3::context::Context {
        &self.inner.cl_context
    }

    /// Borrow the default in-order command queue for
    /// [`device()`](Self::device). This is the queue the
    /// [`Launcher`] impl for `&Context` routes through.
    ///
    /// Always succeeds — created eagerly at build time.
    pub fn raw_default_queue(&self) -> &CommandQueue {
        // `queues[0].in_order` is populated at build time; unwrap is
        // safe. See `ContextBuilder::build`.
        self.inner.queues[0]
            .in_order
            .get()
            .expect("devices[0] default in-order queue is populated at build time")
            .raw()
    }

    /// Borrow the lazily-built fill `Program` for the device-kernel
    /// fill path (see [`crate::fill_kernel`]). Builds on first call;
    /// subsequent calls return the cached program.
    ///
    /// Internal — used by [`crate::buffer::FillOp`] and
    /// [`crate::mapped::SvmFillOp`] when the marker's
    /// [`FillStrategy`](crate::FillStrategy) is `DeviceKernel`. Users
    /// should not need this directly.
    pub(crate) fn fill_program(&self) -> Result<&Program> {
        once_lock_get_or_try_init(&self.inner.fill_program, || {
            Program::create_and_build_from_source(
                &self.inner.cl_context,
                crate::fill_kernel::FILL_PROGRAM_SOURCE,
                "",
            )
            .map_err(|log| Error::Build { log })
        })
    }

    // ── SVM capability query ───────────────────────────────────────

    /// What level of Shared Virtual Memory this context's device
    /// supports — useful for gating [`crate::mapped::MappedSlice`]
    /// construction or picking between SVM and host-mapped tiers.
    ///
    /// Returns the highest applicable level. A device may report
    /// multiple capability bits; ordering is
    /// `FineSystem > FineBuffer > CoarseBuffer > None`. The atomics
    /// extension is independent — query via [`SvmLevel::has_atomics`]
    /// on the returned value.
    pub fn svm_capability(&self) -> SvmLevel {
        let caps = self.inner.devices[0].cl3().svm_mem_capability();
        SvmLevel::from_caps(caps)
    }

    // ── Sticky-error counter ───────────────────────────────────────

    /// How many sticky errors have been recorded by Drop impls.
    /// Always zero unless an OpenCL release call failed during
    /// teardown — fault accumulator pattern from cuda-oxide.
    pub fn error_count(&self) -> u32 {
        self.inner.error_state.load(Ordering::Relaxed)
    }

    /// Bump the sticky-error counter. Called from `Drop` impls in
    /// dependent types when a release fails and the error can't be
    /// propagated.
    ///
    /// Public so external claspr-async impls (e.g. the SVM view's
    /// Drop in `host_view`) can record into the same counter.
    pub fn record_err(&self) {
        self.inner.error_state.fetch_add(1, Ordering::Relaxed);
    }

    // ── Program / kernel ─────────────────────────────────────────────

    /// Create + build an OpenCL program from raw SPIR-V bytes.
    /// Returns the build log on failure.
    pub fn build_program(&self, spv_bytes: &[u8]) -> Result<Program> {
        let mut program = Program::create_from_il(&self.inner.cl_context, spv_bytes)?;
        if let Err(e) = program.build(self.inner.cl_context.devices(), "") {
            let log = program
                .get_build_log(self.inner.devices[0].raw_id())
                .unwrap_or_else(|_| "no build log".into());
            return Err(crate::Error::Build {
                log: format!("{e}\n{log}"),
            });
        }
        Ok(program)
    }

    /// Look up a kernel by entry-point name in a built program.
    pub fn kernel(&self, program: &Program, name: &str) -> Result<Kernel> {
        Ok(Kernel::create(program, name)?)
    }

    /// Convenience: [`build_program`](Self::build_program) +
    /// [`kernel`](Self::kernel) in one call. The intermediate
    /// `Program` is dropped — OpenCL refcounts it internally and
    /// the kernel keeps it alive.
    pub fn kernel_from_spv(&self, spv_bytes: &[u8], name: &str) -> Result<Kernel> {
        let program = self.build_program(spv_bytes)?;
        self.kernel(&program, name)
    }
}

// ── ContextBuilder ──────────────────────────────────────────────────

/// Builder for [`Context`]. Accumulate devices via [`device`](Self::device)
/// / [`devices`](Self::devices), opt into profiling via
/// [`profiling`](Self::profiling), then [`build`](Self::build).
///
/// Use the canned [`Context::for_device`] / [`Context::for_devices`]
/// shortcuts for single-call construction with profiling off.
pub struct ContextBuilder {
    devices: Vec<Device>,
    profiling: bool,
}

impl ContextBuilder {
    fn new() -> Self {
        ContextBuilder {
            devices: Vec::new(),
            profiling: false,
        }
    }

    /// Add a device to the context. Chainable — call once per device,
    /// or use [`devices`](Self::devices) to add a slice in one go.
    pub fn device(mut self, dev: &Device) -> Self {
        self.devices.push(dev.clone());
        self
    }

    /// Add multiple devices. All devices must come from the same
    /// platform (OpenCL spec requirement; the underlying
    /// `clCreateContext` rejects mixed platforms).
    pub fn devices(mut self, devs: &[Device]) -> Self {
        self.devices.extend_from_slice(devs);
        self
    }

    /// Enable `CL_QUEUE_PROFILING_ENABLE` on every default queue and
    /// every [`Queue::new`](crate::queue::Queue::new) /
    /// [`Queue::on_device`](crate::queue::Queue::on_device) built off
    /// this context. Off by default.
    ///
    /// Required for [`LaunchOp::profiled`](crate::op::LaunchOp::profiled)
    /// to return real timestamps.
    pub fn profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    /// Materialise the context. The default in-order queue for the
    /// first device is created eagerly so `&Context` can be used as
    /// a `Launcher` without an extra fallible step; every other
    /// per-device queue is created on first lookup.
    pub fn build(self) -> Result<Context> {
        if self.devices.is_empty() {
            return Err(Error::InvalidArgument(
                "ContextBuilder::build: no devices selected",
            ));
        }
        let ids: Vec<_> = self.devices.iter().map(|d| d.raw_id()).collect();
        let cl_context =
            opencl3::context::Context::from_devices(&ids, &[], None, std::ptr::null_mut())?;
        let queues: Vec<DeviceQueues> = (0..self.devices.len())
            .map(|_| DeviceQueues::empty())
            .collect();
        let ctx = Context {
            inner: Arc::new(ContextInner {
                cl_context,
                devices: self.devices,
                profiling: self.profiling,
                queues,
                error_state: AtomicU32::new(0),
                fill_program: OnceLock::new(),
            }),
        };
        // Eagerly populate the in-order queue for devices[0] so
        // `Launcher::cl_queue(&ctx)` never has to fail. The lazy
        // initialiser inside `default_inorder_queue` is the same
        // path; calling it once now stores the queue into the
        // OnceLock.
        let dev0 = ctx.inner.devices[0].clone();
        ctx.default_inorder_queue(&dev0)?;
        Ok(ctx)
    }
}

// ── SvmLevel ─────────────────────────────────────────────────────────

/// The highest level of Shared Virtual Memory the device supports.
///
/// SVM tiers in OpenCL 2.0+ form a strict hierarchy: a device that
/// supports fine-grain system also supports fine-grain buffer and
/// coarse-grain buffer. claspr surfaces only the top level reached;
/// query [`has_atomics`](SvmLevel::has_atomics) separately for the
/// orthogonal atomics extension.
///
/// Used to gate [`crate::mapped::MappedSlice`] construction —
/// `MappedSlice::alloc` returns [`crate::Error::SvmNotAvailable`]
/// when the device reports [`SvmLevel::None`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum SvmLevel {
    /// `CL_DEVICE_SVM_CAPABILITIES` is zero — SVM not supported.
    None,
    /// `CL_DEVICE_SVM_COARSE_GRAIN_BUFFER`: SVM allocations behave
    /// like cl_mem buffers; host access requires map/unmap.
    CoarseBuffer,
    /// `CL_DEVICE_SVM_FINE_GRAIN_BUFFER`: host and device can access
    /// the buffer concurrently at byte granularity, no map/unmap
    /// needed.
    FineBuffer,
    /// `CL_DEVICE_SVM_FINE_GRAIN_SYSTEM`: any host pointer (`Vec`,
    /// `Box`, arbitrary malloc) is shareable with the device — no
    /// claspr-side allocator needed; just pass `&Vec<T>` directly.
    FineSystem,
}

impl SvmLevel {
    fn from_caps(caps: opencl3::types::cl_device_svm_capabilities) -> SvmLevel {
        // Per the OpenCL spec the higher levels imply the lower
        // ones, but devices may report any subset (e.g. fine-grain
        // buffer without coarse-grain). Pick the highest set bit.
        if caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM != 0 {
            SvmLevel::FineSystem
        } else if caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER != 0 {
            SvmLevel::FineBuffer
        } else if caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER != 0 {
            SvmLevel::CoarseBuffer
        } else {
            SvmLevel::None
        }
    }

    /// `true` if the SVM atomics extension is also available.
    /// Orthogonal to the level — both queried from the same flags
    /// word but reported separately because the level alone doesn't
    /// imply atomics.
    pub fn has_atomics(&self, ctx: &Context) -> bool {
        let caps = ctx.inner.devices[0].cl3().svm_mem_capability();
        caps & CL_DEVICE_SVM_ATOMICS != 0
    }
}

// ── Launcher impl ────────────────────────────────────────────────────

impl Launcher for Context {
    fn cl_queue(&self) -> &CommandQueue {
        self.raw_default_queue()
    }

    fn context(&self) -> &Context {
        self
    }
}
