//! Command-buffer recording toolkit for the eager graph (CB-as-EXECUTION-MODE).
//!
//! This module provides the low-level pieces the graph's `execute` walk uses to
//! record a seam-free device subtree into a real `cl_khr_command_buffer` and
//! replay it across `sync`s — the recording is a MODE of `execute`, not a
//! separate record/replay API (see [`crate::eager`] and the `CbWalk` fork).
//!
//! # Pieces
//!
//! - [`CbBuilder`] — a live command-buffer recording target. The graph opens one
//!   at a CB boundary, each leaf adds its device command (fill / copy / ndrange,
//!   buffer + SVM + image variants), and [`CbBuilder::finalize`] seals it into an
//!   immutable [`FinalizedCb`] cached on the boundary node.
//! - [`FinalizedCb`] — a finalized, replayable command buffer (RAII-released on
//!   drop). Enqueued once per `sync` with the run's external events; ordering
//!   INSIDE the CB is sync points, not `cl_event`s.
//! - [`BufHandle`] / [`MemRef`] / [`RecordableBuffer`] — the recording twins of a
//!   `(value, Deps)`: a buffer's raw `cl_mem`/SVM reference + byte length, so a
//!   leaf can bake its arg into a recorded command by handle.
//! - `CommandBufferExt` (crate-internal) — the provisional extension's entry
//!   points, resolved via `clGetExtensionFunctionAddressForPlatform` (opencl3's
//!   safe wrapper can't reach them).
//!
//! The provisional CB extension is opt-in per platform: a driver without it (or
//! lacking an SVM/image command PFN) resolves the entry point null, and the graph
//! falls back to the per-op `execute` path — same results, no CB acceleration.
use crate::error::{Error, Result};
use crate::exec_ctx::SyncPoints;
use crate::util::lock_unpoisoned;
use opencl3::types::{cl_command_queue, cl_event, cl_kernel, cl_mem, cl_uint};
use std::ffi::{CStr, c_void};
use std::ptr;
use std::sync::Mutex;

/// One command the recording walk issued, kept in software so a build that can't
/// become a real command buffer (a rejected command, a missing optional entry
/// point, a failed finalize) still runs EXACTLY what was recorded — enqueued
/// directly by [`CbBuilder::replay`] — instead of re-running the graph, which
/// can't re-supply inputs that already flowed into the build through pipes.
enum SwCmd {
    Fill {
        mem: MemRef,
        pattern: Vec<u8>,
        offset: usize,
        size: usize,
    },
    Copy {
        src: MemRef,
        dst: MemRef,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    },
    FillImage {
        image: cl_mem,
        color: Vec<u8>,
        origin: [usize; 3],
        region: [usize; 3],
    },
    CopyImage {
        src: cl_mem,
        dst: cl_mem,
        src_origin: [usize; 3],
        dst_origin: [usize; 3],
        region: [usize; 3],
    },
    NdRange {
        kernel: cl_kernel,
        global: Vec<usize>,
        local: Vec<usize>,
    },
    /// A command claspr declines to record (a copy whose sizes differ): replaying
    /// it is this error — what the per-op path reports.
    Fail(Option<Error>),
}

/// A logged command and the commands (indices into the log) it waits on.
struct SwEntry {
    cmd: SwCmd,
    waits: Vec<usize>,
}

thread_local! {
    static INJECT_CB_FAILURE: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// Test hook: make a `cl_khr_command_buffer` recording or finalize call on THIS
/// thread fail as if the driver rejected it — `Some(0)` the next call, `Some(n)`
/// the one after `n` more, `None` to cancel. Forces the replay fallback on any
/// device (a graph records on the thread that runs its terminal). Not API.
#[doc(hidden)]
pub fn __inject_cb_failure(after: Option<u32>) {
    INJECT_CB_FAILURE.with(|c| c.set(after));
}

/// Whether the injection hook fails this call (consuming it).
fn cb_injected_failure() -> bool {
    INJECT_CB_FAILURE.with(|c| match c.get() {
        Some(0) => {
            c.set(None);
            true
        }
        Some(n) => {
            c.set(Some(n - 1));
            false
        }
        None => false,
    })
}

/// A command-buffer call's status, or a driver-style failure when the injection
/// hook fires — so an injected failure takes exactly the real failure path.
fn cb_call(call: impl FnOnce() -> opencl_sys::cl_int) -> opencl_sys::cl_int {
    if cb_injected_failure() {
        opencl_sys::CL_OUT_OF_RESOURCES
    } else {
        call()
    }
}

/// `CLASPR_CB_TRACE=1`: log every failed `cl_khr_command_buffer` call and every
/// build marked ineligible to stderr — to diagnose an implementation (or an
/// emulation layer) that rejects what claspr records. Read once.
fn cb_trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CLASPR_CB_TRACE").is_some_and(|v| v != "0"))
}

/// Trace a `cl_khr_command_buffer` call's status when it failed.
fn cb_trace(call: &str, status: opencl_sys::cl_int) {
    if status != opencl_sys::CL_SUCCESS && cb_trace_enabled() {
        eprintln!("[claspr cb] {call} -> {status}");
    }
}

fn cb_trace_ineligible(at: &std::panic::Location<'_>) {
    if cb_trace_enabled() {
        eprintln!("[claspr cb] build marked ineligible at {at}");
    }
}

// ── cl_khr_command_buffer FFI (layer 2) ─────────────────────────────────────
//
// The extension is PROVISIONAL, so the ICD loader's dispatch table does NOT
// export its entry points. We resolve them per-platform via
// `clGetExtensionFunctionAddressForPlatform` (a core CL 1.2 call the loader DOES
// export) and call through opencl-sys's PFN typedefs. opencl3's safe
// `CommandBuffer` wrapper is unusable for this (its dlsym path returns -2001).

use opencl_sys::{
    cl_command_buffer_khr, cl_platform_id, cl_sync_point_khr, clCommandCopyBufferKHR_t,
    clCommandCopyImageKHR_t, clCommandFillBufferKHR_t, clCommandFillImageKHR_t,
    clCommandNDRangeKernelKHR_t, clCommandSVMMemFillKHR_t, clCommandSVMMemcpyKHR_t,
    clCreateCommandBufferKHR_t, clEnqueueCommandBufferKHR_t, clFinalizeCommandBufferKHR_t,
    clReleaseCommandBufferKHR_t,
};

/// The `cl_khr_command_buffer` entry points, resolved for one platform. Each
/// field is the opencl-sys PFN typedef (`Option<unsafe extern "C" fn …>`);
/// `None` means the loader returned a null address.
#[derive(Clone, Copy)]
struct CommandBufferExt {
    create: clCreateCommandBufferKHR_t,
    finalize: clFinalizeCommandBufferKHR_t,
    enqueue: clEnqueueCommandBufferKHR_t,
    release: clReleaseCommandBufferKHR_t,
    fill_buffer: clCommandFillBufferKHR_t,
    copy_buffer: clCommandCopyBufferKHR_t,
    ndrange_kernel: clCommandNDRangeKernelKHR_t,
    // Optional commands (extension >= 0.9.4 / OpenCL 2.0 for SVM). These are NOT in
    // the mandatory load gate below — a driver lacking them keeps buffer/kernel CBs
    // working, and the per-command `?` on these `Option`s falls back to software.
    fill_image: clCommandFillImageKHR_t,
    copy_image: clCommandCopyImageKHR_t,
    svm_memfill: clCommandSVMMemFillKHR_t,
    svm_memcpy: clCommandSVMMemcpyKHR_t,
}

fn ext_addr(rt: &cl3::OpenClRuntime, platform: cl_platform_id, name: &CStr) -> *mut c_void {
    rt.clGetExtensionFunctionAddressForPlatform(platform, name.as_ptr())
        .unwrap_or(ptr::null_mut())
}

impl CommandBufferExt {
    /// Resolve the entry points for `platform`, or `None` if the core
    /// command-buffer lifecycle isn't reachable (extension absent/provisional).
    fn load(platform: cl_platform_id) -> Option<Self> {
        let rt = cl3::load_library().as_ref().ok()?;
        // SAFETY: each address came from clGetExtensionFunctionAddressForPlatform
        // for this platform + the matching name, so the ABI matches the PFN
        // typedef; a null address transmutes to `None` via the fn-pointer niche.
        let ext =
            unsafe {
                CommandBufferExt {
                    create: std::mem::transmute::<*mut c_void, clCreateCommandBufferKHR_t>(
                        ext_addr(rt, platform, c"clCreateCommandBufferKHR"),
                    ),
                    finalize: std::mem::transmute::<*mut c_void, clFinalizeCommandBufferKHR_t>(
                        ext_addr(rt, platform, c"clFinalizeCommandBufferKHR"),
                    ),
                    enqueue: std::mem::transmute::<*mut c_void, clEnqueueCommandBufferKHR_t>(
                        ext_addr(rt, platform, c"clEnqueueCommandBufferKHR"),
                    ),
                    release: std::mem::transmute::<*mut c_void, clReleaseCommandBufferKHR_t>(
                        ext_addr(rt, platform, c"clReleaseCommandBufferKHR"),
                    ),
                    fill_buffer: std::mem::transmute::<*mut c_void, clCommandFillBufferKHR_t>(
                        ext_addr(rt, platform, c"clCommandFillBufferKHR"),
                    ),
                    copy_buffer: std::mem::transmute::<*mut c_void, clCommandCopyBufferKHR_t>(
                        ext_addr(rt, platform, c"clCommandCopyBufferKHR"),
                    ),
                    ndrange_kernel: std::mem::transmute::<*mut c_void, clCommandNDRangeKernelKHR_t>(
                        ext_addr(rt, platform, c"clCommandNDRangeKernelKHR"),
                    ),
                    fill_image: std::mem::transmute::<*mut c_void, clCommandFillImageKHR_t>(
                        ext_addr(rt, platform, c"clCommandFillImageKHR"),
                    ),
                    copy_image: std::mem::transmute::<*mut c_void, clCommandCopyImageKHR_t>(
                        ext_addr(rt, platform, c"clCommandCopyImageKHR"),
                    ),
                    svm_memfill: std::mem::transmute::<*mut c_void, clCommandSVMMemFillKHR_t>(
                        ext_addr(rt, platform, c"clCommandSVMMemFillKHR"),
                    ),
                    svm_memcpy: std::mem::transmute::<*mut c_void, clCommandSVMMemcpyKHR_t>(
                        ext_addr(rt, platform, c"clCommandSVMMemcpyKHR"),
                    ),
                }
            };
        // The full set we use must be present (lifecycle + the three commands).
        if ext.create.is_some()
            && ext.finalize.is_some()
            && ext.enqueue.is_some()
            && ext.release.is_some()
            && ext.fill_buffer.is_some()
            && ext.copy_buffer.is_some()
            && ext.ndrange_kernel.is_some()
        {
            Some(ext)
        } else {
            None
        }
    }
}

// ── CbBuilder / FinalizedCb: the CB-as-EXECUTION-MODE toolkit (design v2) ────
//
// `CbBuilder` is a LIVE command buffer that the SAME execution walk adds commands
// to as it descends — "a node adds itself to the CB INSTEAD of enqueuing"
// (CB-as-execution-mode). Commands are `clCommand*KHR` calls that return
// `cl_sync_point_khr` MARKERS (the CB-internal ordering primitive). When the walk
// finishes, the homing node `finalize()`s the builder into a `FinalizedCb` and
// caches the `Arc<FinalizedCb>` in its OWN cb-cache field; subsequent syncs replay
// via one `clEnqueueCommandBufferKHR`.
//
// A `CbBuilder` is INTERIOR-MUTABLE (the live `cl_command_buffer_khr` handle plus
// a `Mutex` guarding eligibility state) so it can be threaded DOWN the walk as a
// shared `&CbBuilder`: bundle siblings all add to the SAME builder, while a
// seam-boundary child is simply handed `None` and opens its own. This is what
// makes the per-node CB-visibility POSITIONAL without any save/restore.

/// A live `cl_khr_command_buffer` being built by the execution walk. Each device
/// leaf that is "in CB mode" calls the crate-internal `fill_buffer` /
/// `copy_buffer` / [`ndrange_kernel`](Self::ndrange_kernel), which add a
/// `clCommand*KHR` to the buffer and return the command's [`cl_sync_point_khr`]
/// MARKER; a consumer inside the same CB waits on its producer's markers. When the
/// walk completes, the homing node calls [`finalize`](Self::finalize) to seal it
/// into a [`FinalizedCb`].
pub struct CbBuilder {
    cb: cl_command_buffer_khr,
    queue: cl_command_queue,
    ext: CommandBufferExt,
    /// Kernels retained at build (`clRetainKernel`), released when the
    /// [`FinalizedCb`] drops — the CB references them for its whole lifetime.
    kernels: Mutex<Vec<cl_kernel>>,
    /// Set false if any command could not be added (e.g. a driver missing the SVM
    /// or image command PFN). A non-eligible builder is discarded and the caller
    /// falls back to per-op execute.
    eligible: Mutex<bool>,
    /// Set true once [`finalize`](Self::finalize) has sealed this builder into a
    /// [`FinalizedCb`] and handed off the CB handle + retained kernels. Makes
    /// finalize idempotent (finalize-at-close: the close point seals the CB, and the
    /// boundary-return frame must NOT re-seal / re-enqueue) and tells [`Drop`] to
    /// release NOTHING (the `FinalizedCb` now owns the handle + kernels).
    finalized: Mutex<bool>,
    /// The set of SLOT cell ids (`Arc::as_ptr` of an `Input::Slot`'s cell) whose
    /// buffer/scalar this CB baked into a recorded command — INCLUDING slots reached
    /// transitively through pipes (a `FedByPipe` arg, or a buffer threaded through an
    /// upstream kernel), noted via [`note_slot`](Self::note_slot). Moved into
    /// [`FinalizedCb::captured_slots`](FinalizedCb) at finalize; precise per-slot
    /// invalidation clears this CB iff a mutated slot is in the set.
    slots: Mutex<std::collections::BTreeSet<usize>>,
    /// What each slot cell under the recording boundary held when recording began
    /// (`cell id → digest`), collected from the cells themselves by the boundary
    /// ([`set_slot_digests`](Self::set_slot_digests)). Moved into the
    /// [`FinalizedCb`]; `CbLru::install` keys the CB from it.
    slot_digests: Mutex<std::collections::BTreeMap<usize, Option<u64>>>,
    /// Every command the walk issued, in order (see [`SwCmd`]). The sync point a
    /// recording method returns is a VIRTUAL one — its 1-based index here — so a
    /// consumer's waits stay meaningful even past a command the driver rejected.
    sw: Mutex<Vec<SwEntry>>,
    /// The REAL `cl_sync_point_khr` of each logged command that made it into the
    /// command buffer (`None` once the build stopped recording).
    real_sp: Mutex<Vec<Option<cl_sync_point_khr>>>,
    /// Set when a command-buffer call failed in the DRIVER (not a command claspr
    /// declined): the boundary then disables command buffers on the context.
    driver_failed: std::sync::atomic::AtomicBool,
    /// Set once [`replay`](Self::replay) ran the log: the build is settled (like a
    /// finalize) — a later boundary-return frame must not run it again.
    replayed: std::sync::atomic::AtomicBool,
}

// SAFETY: the CB / queue / kernel handles are opaque handles into the
// internally-synchronized runtime; the ext PFNs are plain fn pointers. The
// interior state is `Mutex`-guarded.
unsafe impl Send for CbBuilder {}
unsafe impl Sync for CbBuilder {}

/// Process-wide count of command buffers actually CREATED
/// (`clCreateCommandBufferKHR`) — the record-churn metric. A replayed graph
/// should create ONE per distinct binding state and then leave this flat; a
/// graph that re-records every `sync` makes it climb linearly with the loop.
///
/// Test/bench hook (`#[doc(hidden)]`), in the same spirit as
/// [`collect_cb_ids`](crate::DeviceOp::collect_cb_ids): it lets a test ASSERT the
/// keyed-cache hit rate instead of inferring it from wall-clock. Relaxed ordering
/// — this is a diagnostic counter, not a synchronization point, and it rides a
/// driver call that dwarfs it.
static CB_CREATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Read [`CB_CREATES`] — total command buffers created since process start.
/// Take a delta around the region you care about. PROCESS-WIDE: a delta also
/// counts every other graph's records, including concurrently running tests', so
/// to assert one graph's count read its root cache instead
/// (`g.cb_cache()` → [`CbLru::records`](crate::eager::CbLru::records), or
/// `cb_records_of` in the test-support crate).
#[doc(hidden)]
pub fn cb_records_total() -> u64 {
    CB_CREATES.load(std::sync::atomic::Ordering::Relaxed)
}

impl CbBuilder {
    /// Create a fresh live command buffer over `queue` (single-queue CB). Returns
    /// `None` if the extension's lifecycle isn't reachable for `platform`.
    pub fn new(platform: cl_platform_id, queue: cl_command_queue) -> Option<Self> {
        let ext = CommandBufferExt::load(platform)?;
        let create = ext.create?;
        let mut q = queue;
        let mut err: opencl_sys::cl_int = 0;
        // SAFETY: `create` is the resolved clCreateCommandBufferKHR for `platform`;
        // one queue, default properties.
        let cb = unsafe { create(1, &mut q as *mut _, ptr::null(), &mut err) };
        cb_trace("clCreateCommandBufferKHR", err);
        if err != opencl_sys::CL_SUCCESS || cb.is_null() {
            return None;
        }
        CB_CREATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(CbBuilder {
            cb,
            queue,
            ext,
            kernels: Mutex::new(Vec::new()),
            eligible: Mutex::new(true),
            finalized: Mutex::new(false),
            slots: Mutex::new(std::collections::BTreeSet::new()),
            slot_digests: Mutex::new(std::collections::BTreeMap::new()),
            sw: Mutex::new(Vec::new()),
            real_sp: Mutex::new(Vec::new()),
            driver_failed: std::sync::atomic::AtomicBool::new(false),
            replayed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Log `cmd`; return its virtual sync point and — while the build can still
    /// record — the real sync points of the commands it waits on (`None` once a
    /// command was rejected: nothing more goes into the real command buffer).
    fn log(
        &self,
        cmd: SwCmd,
        waits: &SyncPoints,
    ) -> (cl_sync_point_khr, Option<Vec<cl_sync_point_khr>>) {
        let idx: Vec<usize> = waits.iter().map(|&w| w as usize - 1).collect();
        let mut real_sp = lock_unpoisoned(&self.real_sp);
        let real = if self.is_eligible() {
            idx.iter()
                .map(|&i| real_sp.get(i).copied().flatten())
                .collect()
        } else {
            None
        };
        let mut sw = lock_unpoisoned(&self.sw);
        sw.push(SwEntry { cmd, waits: idx });
        real_sp.push(None);
        (sw.len() as cl_sync_point_khr, real)
    }

    /// Note that virtual command `v` made it into the command buffer as `sp`.
    fn recorded_as(&self, v: cl_sync_point_khr, sp: cl_sync_point_khr) {
        lock_unpoisoned(&self.real_sp)[v as usize - 1] = Some(sp);
    }

    /// A command-buffer call failed in the driver: stop recording (the build will
    /// replay) and remember it was the driver.
    #[track_caller]
    fn driver_failure(&self, call: &str, status: opencl_sys::cl_int) {
        cb_trace(call, status);
        self.driver_failed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.mark_ineligible();
    }

    /// Whether a command-buffer call failed in the driver during this build.
    pub(crate) fn driver_failed(&self) -> bool {
        self.driver_failed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Log a command claspr declines to record: replaying it returns `err` (the
    /// error its per-op path would report). Returns its virtual sync point.
    #[track_caller]
    pub(crate) fn fail(&self, err: Error, waits: &SyncPoints) -> cl_sync_point_khr {
        self.mark_ineligible();
        self.log(SwCmd::Fail(Some(err)), waits).0
    }

    /// Mark the build as ineligible (an SVM command or a failed add) so the homing
    /// node discards it and falls back to per-op execute. `pub(crate)` so the copy
    /// ops (`copy.rs`) can force the per-op fallback on a size mismatch — routing
    /// the error through the per-op path that already returns `LengthMismatch`,
    /// instead of the CB path silently truncating.
    #[track_caller]
    pub(crate) fn mark_ineligible(&self) {
        cb_trace_ineligible(std::panic::Location::caller());
        *lock_unpoisoned(&self.eligible) = false;
    }

    /// Note that this CB baked in a buffer/scalar traceable to slot cell `id`
    /// (`Arc::as_ptr` of an `Input::Slot`'s cell) — directly or transitively through
    /// pipes. Drives precise per-slot invalidation: mutating a slot clears exactly
    /// the CBs whose `captured_slots` contains it. Cheap idempotent set insert.
    pub fn note_slot(&self, id: usize) {
        lock_unpoisoned(&self.slots).insert(id);
    }

    /// Record what the slot cells under the recording boundary hold — collected
    /// from the cells by the boundary before it records. See `slot_digests`.
    pub(crate) fn set_slot_digests(&self, digests: std::collections::BTreeMap<usize, Option<u64>>) {
        *lock_unpoisoned(&self.slot_digests) = digests;
    }

    /// How many commands the walk issued (logged, recorded or not). Zero → the
    /// boundary discards the CB (empty-CB guard).
    pub fn recorded(&self) -> usize {
        lock_unpoisoned(&self.sw).len()
    }

    /// Whether [`finalize`](Self::finalize) has already sealed this builder. The
    /// finalize-at-close span opener reads this at return: `true` → the close point
    /// sealed + enqueued the CB (nothing to do); `false` with `recorded() > 0` and
    /// eligible → the close never fired (a span with no interior seam) → the opener
    /// finalizes + enqueues at return itself.
    pub fn is_finalized(&self) -> bool {
        *lock_unpoisoned(&self.finalized) || self.replayed.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Add a buffer fill. A `cl_mem` buffer records via `clCommandFillBufferKHR`;
    /// an SVM buffer via `clCommandSVMMemFillKHR` (extension >= 0.9.4). Always
    /// logged (see [`SwCmd`]): if the command can't be recorded — absent entry
    /// point, a driver rejection — the build stops recording and the boundary
    /// replays the log instead. Returns the command's (virtual) sync point.
    pub(crate) fn fill_buffer(
        &self,
        mem: MemRef,
        pattern: &[u8],
        offset: usize,
        size: usize,
        waits: &SyncPoints,
    ) -> Option<cl_sync_point_khr> {
        let cmd = SwCmd::Fill {
            mem,
            pattern: pattern.to_vec(),
            offset,
            size,
        };
        let (v, real) = self.log(cmd, waits);
        let Some(real) = real else { return Some(v) };
        let (wptr, n) = wait_ptr(&real);
        let mut sp: cl_sync_point_khr = 0;
        let st = match mem {
            MemRef::Buffer(m) => {
                let Some(fill) = self.ext.fill_buffer else {
                    self.mark_ineligible();
                    return Some(v);
                };
                // SAFETY: live CB; `m` a valid cl_mem kept alive by the graph;
                // `pattern` outlives the call; `real` are this CB's sync points.
                cb_call(|| unsafe {
                    fill(
                        self.cb,
                        // The recording queue must be NULL: claspr creates every command
                        // buffer with ONE queue, and without `cl_khr_command_buffer_multi_device`
                        // the spec requires NULL (with it, NULL means that one queue).
                        ptr::null_mut(),
                        ptr::null(),
                        m as opencl_sys::cl_mem,
                        pattern.as_ptr() as *const c_void,
                        pattern.len(),
                        offset,
                        size,
                        n,
                        wptr,
                        &mut sp,
                        ptr::null_mut(),
                    )
                })
            }
            MemRef::Svm(p) => {
                let Some(svm_fill) = self.ext.svm_memfill else {
                    self.mark_ineligible();
                    return Some(v);
                };
                // SVM fill has no offset param — offset the pointer directly.
                let base = unsafe { (p as *mut u8).add(offset) } as *mut c_void;
                // SAFETY: `p` a valid SVM pointer kept alive by the graph; as above.
                cb_call(|| unsafe {
                    svm_fill(
                        self.cb,
                        ptr::null_mut(),
                        ptr::null(),
                        base,
                        pattern.as_ptr() as *const c_void,
                        pattern.len(),
                        size,
                        n,
                        wptr,
                        &mut sp,
                        ptr::null_mut(),
                    )
                })
            }
        };
        if st != opencl_sys::CL_SUCCESS {
            self.driver_failure("clCommandFillBufferKHR", st);
        } else {
            self.recorded_as(v, sp);
        }
        Some(v)
    }

    /// Add a device-to-device copy. Buffer↔buffer records via
    /// `clCommandCopyBufferKHR`; SVM↔SVM via `clCommandSVMMemcpyKHR` (extension
    /// 0.9.4 or later). A mixed cl_mem/SVM pair has no command-buffer command (and no
    /// per-op copy either): it replays as an error. Always logged; returns the
    /// command's (virtual) sync point.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_buffer(
        &self,
        src: MemRef,
        dst: MemRef,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        waits: &SyncPoints,
    ) -> Option<cl_sync_point_khr> {
        let cmd = SwCmd::Copy {
            src,
            dst,
            src_offset,
            dst_offset,
            size,
        };
        let (v, real) = self.log(cmd, waits);
        let Some(real) = real else { return Some(v) };
        let (wptr, n) = wait_ptr(&real);
        let mut sp: cl_sync_point_khr = 0;
        let st = match (src, dst) {
            (MemRef::Buffer(s), MemRef::Buffer(d)) => {
                let Some(copy) = self.ext.copy_buffer else {
                    self.mark_ineligible();
                    return Some(v);
                };
                // SAFETY: live CB; `s`/`d` valid cl_mem kept alive by the graph.
                cb_call(|| unsafe {
                    copy(
                        self.cb,
                        ptr::null_mut(),
                        ptr::null(),
                        s as opencl_sys::cl_mem,
                        d as opencl_sys::cl_mem,
                        src_offset,
                        dst_offset,
                        size,
                        n,
                        wptr,
                        &mut sp,
                        ptr::null_mut(),
                    )
                })
            }
            (MemRef::Svm(s), MemRef::Svm(d)) => {
                let Some(svm_copy) = self.ext.svm_memcpy else {
                    self.mark_ineligible();
                    return Some(v);
                };
                // SVM copy has no offset params — offset the pointers directly.
                let src_ptr = unsafe { (s as *const u8).add(src_offset) } as *const c_void;
                let dst_ptr = unsafe { (d as *mut u8).add(dst_offset) } as *mut c_void;
                // SAFETY: `s`/`d` valid SVM pointers kept alive by the graph.
                cb_call(|| unsafe {
                    svm_copy(
                        self.cb,
                        ptr::null_mut(),
                        ptr::null(),
                        dst_ptr,
                        src_ptr,
                        size,
                        n,
                        wptr,
                        &mut sp,
                        ptr::null_mut(),
                    )
                })
            }
            _ => {
                self.mark_ineligible();
                return Some(v);
            }
        };
        if st != opencl_sys::CL_SUCCESS {
            self.driver_failure("clCommandCopyBufferKHR", st);
        } else {
            self.recorded_as(v, sp);
        }
        Some(v)
    }

    /// Add an image fill via `clCommandFillImageKHR`. `fill_color` is the
    /// format-appropriate fill value (4×component); `origin` and `region` are
    /// 3-element arrays (`clEnqueueFillImage` shape). Always logged; returns the
    /// command's (virtual) sync point.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fill_image(
        &self,
        image: MemRef,
        fill_color: &[u8],
        origin: [usize; 3],
        region: [usize; 3],
        waits: &SyncPoints,
    ) -> Option<cl_sync_point_khr> {
        // An image is always a `cl_mem` (never SVM).
        let MemRef::Buffer(m) = image else {
            return Some(self.fail(
                Error::NotSupported("an image fill on an SVM pointer"),
                waits,
            ));
        };
        let cmd = SwCmd::FillImage {
            image: m,
            color: fill_color.to_vec(),
            origin,
            region,
        };
        let (v, real) = self.log(cmd, waits);
        let Some(real) = real else { return Some(v) };
        let Some(fill) = self.ext.fill_image else {
            self.mark_ineligible();
            return Some(v);
        };
        let (wptr, n) = wait_ptr(&real);
        let mut sp: cl_sync_point_khr = 0;
        // SAFETY: live CB; `m` a valid image cl_mem kept alive by the graph;
        // `fill_color`/`origin`/`region` outlive the call.
        let st = cb_call(|| unsafe {
            fill(
                self.cb,
                ptr::null_mut(),
                ptr::null(),
                m as opencl_sys::cl_mem,
                fill_color.as_ptr() as *const c_void,
                origin.as_ptr(),
                region.as_ptr(),
                n,
                wptr,
                &mut sp,
                ptr::null_mut(),
            )
        });
        if st != opencl_sys::CL_SUCCESS {
            self.driver_failure("clCommandFillImageKHR", st);
        } else {
            self.recorded_as(v, sp);
        }
        Some(v)
    }

    /// Add an image→image copy via `clCommandCopyImageKHR`. `src`/`dst` are image
    /// `cl_mem`s; `src_origin`/`dst_origin`/`region` are 3-element arrays. Always
    /// logged; returns the command's (virtual) sync point.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_image(
        &self,
        src: MemRef,
        dst: MemRef,
        src_origin: [usize; 3],
        dst_origin: [usize; 3],
        region: [usize; 3],
        waits: &SyncPoints,
    ) -> Option<cl_sync_point_khr> {
        let (MemRef::Buffer(s), MemRef::Buffer(d)) = (src, dst) else {
            return Some(self.fail(
                Error::NotSupported("an image copy on an SVM pointer"),
                waits,
            ));
        };
        let cmd = SwCmd::CopyImage {
            src: s,
            dst: d,
            src_origin,
            dst_origin,
            region,
        };
        let (v, real) = self.log(cmd, waits);
        let Some(real) = real else { return Some(v) };
        let Some(copy) = self.ext.copy_image else {
            self.mark_ineligible();
            return Some(v);
        };
        let (wptr, n) = wait_ptr(&real);
        let mut sp: cl_sync_point_khr = 0;
        // SAFETY: as `fill_image`; both images valid cl_mem kept alive by the graph.
        let st = cb_call(|| unsafe {
            copy(
                self.cb,
                ptr::null_mut(),
                ptr::null(),
                s as opencl_sys::cl_mem,
                d as opencl_sys::cl_mem,
                src_origin.as_ptr(),
                dst_origin.as_ptr(),
                region.as_ptr(),
                n,
                wptr,
                &mut sp,
                ptr::null_mut(),
            )
        });
        if st != opencl_sys::CL_SUCCESS {
            self.driver_failure("clCommandCopyImageKHR", st);
        } else {
            self.recorded_as(v, sp);
        }
        Some(v)
    }

    /// Set one buffer `(pointer, len)` argument pair on `kernel` for CB-mode
    /// build. Advances `arg_index` by 2.
    ///
    /// # Safety
    /// `kernel` must be valid and the two `arg_index` slots must be this buffer's
    /// `(cl_mem, len)` pair; `mem` must outlive the recorded command.
    pub unsafe fn set_buffer_arg(
        &self,
        kernel: cl_kernel,
        arg_index: &mut cl_uint,
        mem: MemRef,
        elem_count: usize,
    ) -> Result<()> {
        use std::ffi::c_void;
        match mem {
            MemRef::Buffer(m) => unsafe {
                cl3::kernel::set_kernel_arg(
                    kernel,
                    *arg_index,
                    std::mem::size_of::<cl_mem>(),
                    (&m as *const cl_mem) as *const c_void,
                )
            },
            MemRef::Svm(p) => unsafe {
                cl3::kernel::set_kernel_arg_svm_pointer(kernel, *arg_index, p as *const c_void)
            },
        }
        .map_err(|s| Error::OpenCl(opencl3::error_codes::ClError(s)))?;
        *arg_index += 1;
        unsafe {
            cl3::kernel::set_kernel_arg(
                kernel,
                *arg_index,
                std::mem::size_of::<usize>(),
                (&elem_count as *const usize) as *const c_void,
            )
        }
        .map_err(|s| Error::OpenCl(opencl3::error_codes::ClError(s)))?;
        *arg_index += 1;
        Ok(())
    }

    /// Set one buffer/image/scalar-ref POINTER argument (a single `cl_mem` or SVM
    /// pointer) on `kernel` for CB-mode build. Advances `arg_index` by 1.
    ///
    /// # Safety
    /// `kernel` must be valid and `arg_index` must be this arg's slot; `mem` must
    /// outlive the recorded command.
    pub unsafe fn set_mem_arg(
        &self,
        kernel: cl_kernel,
        arg_index: &mut cl_uint,
        mem: MemRef,
    ) -> Result<()> {
        use std::ffi::c_void;
        match mem {
            MemRef::Buffer(m) => unsafe {
                cl3::kernel::set_kernel_arg(
                    kernel,
                    *arg_index,
                    std::mem::size_of::<cl_mem>(),
                    (&m as *const cl_mem) as *const c_void,
                )
            },
            MemRef::Svm(p) => unsafe {
                cl3::kernel::set_kernel_arg_svm_pointer(kernel, *arg_index, p as *const c_void)
            },
        }
        .map_err(|s| Error::OpenCl(opencl3::error_codes::ClError(s)))?;
        *arg_index += 1;
        Ok(())
    }

    /// Set one by-value scalar argument on `kernel` from its raw bytes for CB-mode
    /// build. Advances `arg_index` by 1.
    ///
    /// # Safety
    /// `kernel` must be valid, `arg_index` must be this scalar's slot, and `bytes`
    /// must be the correct size for the kernel's declared arg type.
    pub unsafe fn set_scalar_arg(
        &self,
        kernel: cl_kernel,
        arg_index: &mut cl_uint,
        bytes: &[u8],
    ) -> Result<()> {
        use std::ffi::c_void;
        unsafe {
            cl3::kernel::set_kernel_arg(
                kernel,
                *arg_index,
                bytes.len(),
                bytes.as_ptr() as *const c_void,
            )
        }
        .map_err(|s| Error::OpenCl(opencl3::error_codes::ClError(s)))?;
        *arg_index += 1;
        Ok(())
    }

    /// Add an ND-range kernel launch. The kernel's args must already be set (the
    /// caller sets them at build time); the kernel is retained for the build's
    /// lifetime (and the command buffer's, once finalized). Always logged; returns
    /// the command's (virtual) sync point.
    ///
    /// # Safety
    /// `kernel` must be a valid `cl_kernel` whose args are set to the buffers this
    /// launch uses; `waits` must be sync points from this same builder.
    pub unsafe fn ndrange_kernel(
        &self,
        kernel: cl_kernel,
        global: &[usize],
        local: &[usize],
        waits: &SyncPoints,
    ) -> Option<cl_sync_point_khr> {
        // Own a refcount for as long as the build (a replay) or the CB needs it.
        if let Err(code) = unsafe { cl3::kernel::retain_kernel(kernel) } {
            return Some(self.fail(Error::OpenCl(opencl3::error_codes::ClError(code)), waits));
        }
        lock_unpoisoned(&self.kernels).push(kernel);
        let cmd = SwCmd::NdRange {
            kernel,
            global: global.to_vec(),
            local: local.to_vec(),
        };
        let (v, real) = self.log(cmd, waits);
        let Some(real) = real else { return Some(v) };
        let Some(ndr) = self.ext.ndrange_kernel else {
            self.mark_ineligible();
            return Some(v);
        };
        let (wptr, n) = wait_ptr(&real);
        let mut sp: cl_sync_point_khr = 0;
        // SAFETY: live CB; kernel valid + args set + retained; `real` are this
        // CB's sync points.
        let st = cb_call(|| unsafe {
            ndr(
                self.cb,
                // The recording queue must be NULL: claspr creates every command
                // buffer with ONE queue, and without `cl_khr_command_buffer_multi_device`
                // the spec requires NULL (with it, NULL means that one queue).
                ptr::null_mut(),
                ptr::null(),
                kernel as opencl_sys::cl_kernel,
                global.len() as cl_uint,
                ptr::null(),
                global.as_ptr(),
                if local.is_empty() {
                    ptr::null()
                } else {
                    local.as_ptr()
                },
                n,
                wptr,
                &mut sp,
                ptr::null_mut(),
            )
        });
        if st != opencl_sys::CL_SUCCESS {
            self.driver_failure("clCommandNDRangeKernelKHR", st);
        } else {
            self.recorded_as(v, sp);
        }
        Some(v)
    }

    /// Whether every command added so far is CB-eligible (all `cl_mem`, no failed
    /// add). A `false` here means the homing node must discard the builder and fall
    /// back to the per-op execute path.
    pub fn is_eligible(&self) -> bool {
        *lock_unpoisoned(&self.eligible)
    }

    /// Finalize the live CB into a replayable [`FinalizedCb`] — **interior-mutable +
    /// IDEMPOTENT** (`&self`, not `self`). This is what the *finalize-at-close* path
    /// needs: the span CLOSE point (a `Build`→`Off` transition at a host seam)
    /// seals+enqueues the CB through a SHARED borrow — the builder is behind
    /// `&CbBuilder` in [`CbWalk::Build`](crate::exec_ctx::CbWalk), never owned there,
    /// while the boundary-return frame still holds the same borrow.
    ///
    /// Returns `Some(cb)` exactly ONCE (the first call that seals it); every
    /// subsequent call returns `None` (already sealed → the boundary-return frame
    /// must reuse the homed [`FinalizedCb`], not re-seal). `None` also on ineligible
    /// / missing-PFN — the caller then falls back and `Drop` releases the live CB.
    ///
    /// After a successful seal the CB handle + retained kernels are OWNED by the
    /// returned [`FinalizedCb`]; `self`'s [`Drop`] releases nothing (the `finalized`
    /// flag).
    pub fn finalize(&self) -> Option<FinalizedCb> {
        // Idempotency + eligibility gate. Hold the `finalized` lock across the whole
        // seal so two racing closers can't both hand off the same handle.
        let mut done = lock_unpoisoned(&self.finalized);
        if *done || !self.is_eligible() {
            return None;
        }
        let finalize = self.ext.finalize?;
        // SAFETY: sealing the live CB built above.
        let st = cb_call(|| unsafe { finalize(self.cb) });
        if st != opencl_sys::CL_SUCCESS {
            self.driver_failure("clFinalizeCommandBufferKHR", st);
            return None;
        }
        let enqueue = self.ext.enqueue?;
        let release = self.ext.release?;
        // Hand off the CB handle (Copy) + retained kernels to the FinalizedCb. Mark
        // sealed BEFORE releasing the lock so `Drop` (and any later `finalize`) skips
        // the handle we just gave away.
        let kernels = std::mem::take(&mut *lock_unpoisoned(&self.kernels));
        let captured_slots = std::mem::take(&mut *lock_unpoisoned(&self.slots));
        let slot_digests = std::mem::take(&mut *lock_unpoisoned(&self.slot_digests));
        *done = true;
        Some(FinalizedCb {
            cb: self.cb,
            queue: self.queue,
            enqueue,
            release,
            kernels,
            captured_slots,
            slot_digests,
            // Stamped by `CbLru::install` from `slot_digests` + its own bindings.
            key: Mutex::new(None),
        })
    }
}

impl CbBuilder {
    /// Run every logged command directly on the build's queue — the fallback
    /// when the build couldn't become a command buffer (a rejected command, an
    /// absent entry point, a failed finalize). Commands run in recorded order,
    /// each waiting on `wait` (the build's external deps) and on the commands
    /// it depended on; returns one completion event over all of them, standing
    /// in for the command buffer's. A command claspr declined to record returns
    /// its error here.
    pub fn replay(&self, wait: &[cl_event]) -> Result<crate::Event> {
        let mut sw = lock_unpoisoned(&self.sw);
        self.replayed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let mut events: Vec<crate::Event> = Vec::with_capacity(sw.len());
        let to_err = |code| Error::OpenCl(opencl3::error_codes::ClError(code));
        for entry in sw.iter_mut() {
            let mut deps: Vec<cl_event> = wait.to_vec();
            deps.extend(entry.waits.iter().map(|&i| events[i].get()));
            let (dptr, n) = wait_ptr_ev(&deps);
            let q = self.queue;
            // SAFETY: every handle was valid when logged and is kept alive by the
            // graph (buffers, images) or by this builder (kernels, retained); `deps`
            // are live events; pointer arguments outlive each call.
            let raw = unsafe {
                match &mut entry.cmd {
                    SwCmd::Fill {
                        mem: MemRef::Buffer(m),
                        pattern,
                        offset,
                        size,
                    } => cl3::command_queue::enqueue_fill_buffer(
                        q,
                        *m,
                        pattern.as_ptr() as *const c_void,
                        pattern.len(),
                        *offset,
                        *size,
                        n,
                        dptr,
                    ),
                    SwCmd::Fill {
                        mem: MemRef::Svm(p),
                        pattern,
                        offset,
                        size,
                    } => cl3::command_queue::enqueue_svm_mem_fill(
                        q,
                        (*p as *mut u8).add(*offset) as *mut c_void,
                        pattern.as_ptr() as *const c_void,
                        pattern.len(),
                        *size,
                        n,
                        dptr,
                    ),
                    SwCmd::Copy {
                        src: MemRef::Buffer(s),
                        dst: MemRef::Buffer(d),
                        src_offset,
                        dst_offset,
                        size,
                    } => cl3::command_queue::enqueue_copy_buffer(
                        q,
                        *s,
                        *d,
                        *src_offset,
                        *dst_offset,
                        *size,
                        n,
                        dptr,
                    ),
                    SwCmd::Copy {
                        src: MemRef::Svm(s),
                        dst: MemRef::Svm(d),
                        src_offset,
                        dst_offset,
                        size,
                    } => cl3::command_queue::enqueue_svm_mem_cpy(
                        q,
                        opencl3::types::CL_NON_BLOCKING,
                        (*d as *mut u8).add(*dst_offset) as *mut c_void,
                        (*s as *const u8).add(*src_offset) as *const c_void,
                        *size,
                        n,
                        dptr,
                    ),
                    SwCmd::Copy { .. } => {
                        return Err(Error::NotSupported(
                            "a copy between a cl_mem buffer and an SVM pointer",
                        ));
                    }
                    SwCmd::FillImage {
                        image,
                        color,
                        origin,
                        region,
                    } => cl3::command_queue::enqueue_fill_image(
                        q,
                        *image,
                        color.as_ptr() as *const c_void,
                        origin.as_ptr(),
                        region.as_ptr(),
                        n,
                        dptr,
                    ),
                    SwCmd::CopyImage {
                        src,
                        dst,
                        src_origin,
                        dst_origin,
                        region,
                    } => cl3::command_queue::enqueue_copy_image(
                        q,
                        *src,
                        *dst,
                        src_origin.as_ptr(),
                        dst_origin.as_ptr(),
                        region.as_ptr(),
                        n,
                        dptr,
                    ),
                    SwCmd::NdRange {
                        kernel,
                        global,
                        local,
                    } => cl3::command_queue::enqueue_nd_range_kernel(
                        q,
                        *kernel,
                        global.len() as cl_uint,
                        ptr::null(),
                        global.as_ptr(),
                        if local.is_empty() {
                            ptr::null()
                        } else {
                            local.as_ptr()
                        },
                        n,
                        dptr,
                    ),
                    SwCmd::Fail(err) => {
                        return Err(err
                            .take()
                            .unwrap_or(Error::NotSupported("a declined command replayed twice")));
                    }
                }
            };
            events.push(crate::Event::new(raw.map_err(to_err)?));
        }
        let all: Vec<cl_event> = events.iter().map(|e| e.get()).collect();
        let all = if all.is_empty() { wait.to_vec() } else { all };
        let (aptr, n) = wait_ptr_ev(&all);
        // SAFETY: live queue; `all` are live events (held by `events` / the caller).
        let done =
            unsafe { cl3::command_queue::enqueue_marker_with_wait_list(self.queue, n, aptr) }
                .map_err(to_err)?;
        Ok(crate::Event::new(done))
    }
}

impl Drop for CbBuilder {
    fn drop(&mut self) {
        // A successful `finalize` set `finalized` and handed the CB + kernels to the
        // `FinalizedCb`, which now owns them → release NOTHING here. Only the DISCARD
        // path (never finalized: ineligible / empty-CB guard / finalize failure)
        // reaches the releases below.
        if *lock_unpoisoned(&self.finalized) {
            return;
        }
        for k in lock_unpoisoned(&self.kernels).drain(..) {
            let _ = unsafe { cl3::kernel::release_kernel(k) };
        }
        if let Some(release) = self.ext.release {
            unsafe { release(self.cb) };
        }
    }
}

/// A finalized, replayable command buffer homed in a graph node's cb-cache field.
/// Replay is one [`enqueue`](Self::enqueue) (`clEnqueueCommandBufferKHR`) returning
/// a completion event. RAII-releases the CB + its retained kernels on drop, so the
/// cache "drops with the graph" (no global table, no ABA).
pub struct FinalizedCb {
    cb: cl_command_buffer_khr,
    queue: cl_command_queue,
    enqueue: unsafe extern "C" fn(
        cl_uint,
        *mut cl_command_queue,
        cl_command_buffer_khr,
        cl_uint,
        *const cl_event,
        *mut cl_event,
    ) -> opencl_sys::cl_int,
    release: unsafe extern "C" fn(cl_command_buffer_khr) -> opencl_sys::cl_int,
    kernels: Vec<cl_kernel>,
    /// Slot cell ids this CB baked a buffer/scalar from (directly or transitively
    /// through pipes). Precise invalidation clears this CB iff a mutated slot is here.
    captured_slots: std::collections::BTreeSet<usize>,
    /// What each slot cell under the boundary held when this CB was recorded
    /// (`cell id → digest`, from the cells themselves). See `CbBuilder::slot_digests`.
    slot_digests: std::collections::BTreeMap<usize, Option<u64>>,
    /// **The binding state this CB was recorded in** — the digest of every slot in
    /// [`captured_slots`](Self::captured_slots) at record time, filed by
    /// [`CbLru::install`](crate::eager::CbLru). `None` means unkeyable (some captured
    /// slot had no digest — a pipe-fed cell), so the CB can be replayed while it is
    /// current but never parked and recognised later.
    ///
    /// Interior-mutable because the key is known to the CACHE, not to the recording
    /// walk: the digests come from the binder that set the values, so `finalize`
    /// produces the CB and `install` stamps its key a moment later.
    key: Mutex<Option<crate::eager::CbKey>>,
}

// SAFETY: as `RecordedCb` — opaque handles + plain fn pointers.
unsafe impl Send for FinalizedCb {}
unsafe impl Sync for FinalizedCb {}

impl FinalizedCb {
    /// The queue this CB was finalized for — the homing node checks a cached CB is
    /// valid for the current sync's queue before replaying.
    pub fn queue(&self) -> cl_command_queue {
        self.queue
    }

    /// Whether this CB baked a buffer/scalar from ANY slot in `mutated` (directly or
    /// transitively through pipes) — i.e. a mutate of one of those slots stales it.
    pub fn depends_on_any(&self, mutated: &std::collections::BTreeSet<usize>) -> bool {
        !self.captured_slots.is_disjoint(mutated)
    }

    /// The slot cells this CB baked — see [`captured_slots`](Self::captured_slots).
    pub fn captured_slots(&self) -> &std::collections::BTreeSet<usize> {
        &self.captured_slots
    }

    /// What slot cell `cell` held when this CB was recorded: `Some(digest)` if the
    /// recording boundary's collection reached it (`digest` itself `None` when the
    /// cell had nothing keyable), `None` if it did not.
    pub(crate) fn slot_digest(&self, cell: usize) -> Option<Option<u64>> {
        self.slot_digests.get(&cell).copied()
    }

    /// The binding state this CB was recorded in, or `None` if unkeyable. See
    /// [`key`](Self::key).
    pub fn key(&self) -> Option<crate::eager::CbKey> {
        lock_unpoisoned(&self.key).clone()
    }

    /// Stamp this CB's cache key. Called once by
    /// [`CbLru::install`](crate::eager::CbLru) right after `finalize`.
    pub fn set_key(&self, key: Option<crate::eager::CbKey>) {
        *lock_unpoisoned(&self.key) = key;
    }

    /// Enqueue the whole CB on its queue with `wait` as the EXTERNAL `cl_event`
    /// wait-list (the event↔sync-point boundary: external deps apply ONLY here,
    /// never on the CB-internal `clCommand*KHR` commands). Returns the completion
    /// event wrapped for the pipe/Deps path.
    pub fn enqueue(&self, wait: &[cl_event]) -> Result<crate::Event> {
        let mut queue = self.queue;
        let (wptr, n) = wait_ptr_ev(wait);
        let mut event: cl_event = ptr::null_mut();
        // SAFETY: finalized CB for `queue`; `wait` are live events.
        let status = unsafe {
            (self.enqueue)(
                1,
                &mut queue as *mut cl_command_queue,
                self.cb,
                n,
                wptr,
                &mut event,
            )
        };
        cb_trace("clEnqueueCommandBufferKHR", status);
        if status != opencl_sys::CL_SUCCESS {
            return Err(Error::OpenCl(opencl3::error_codes::ClError(status)));
        }
        Ok(crate::Event::new(event))
    }
}

impl Drop for FinalizedCb {
    fn drop(&mut self) {
        for k in self.kernels.drain(..) {
            let _ = unsafe { cl3::kernel::release_kernel(k) };
        }
        unsafe { (self.release)(self.cb) };
    }
}

/// `(ptr, count)` for a sync-point wait-list (null when empty).
fn wait_ptr(waits: &[cl_sync_point_khr]) -> (*const cl_sync_point_khr, cl_uint) {
    if waits.is_empty() {
        (ptr::null(), 0)
    } else {
        (waits.as_ptr(), waits.len() as cl_uint)
    }
}

/// `(ptr, count)` for a `cl_event` wait-list (null when empty).
fn wait_ptr_ev(waits: &[cl_event]) -> (*const cl_event, cl_uint) {
    if waits.is_empty() {
        (ptr::null(), 0)
    } else {
        (waits.as_ptr(), waits.len() as cl_uint)
    }
}

// ── Handles + sync points: the recording twins of (value, Deps) ─────────────

/// Where a recorded buffer lives: a `cl_mem` (a `DeviceSlice`) or a coarse/fine
/// SVM pointer (a `MappedSlice`/`USMSlice`). The record/replay command path
/// dispatches the right CL entry point on this (`clEnqueueFillBuffer` vs
/// `clEnqueueSVMMemFill`, `clSetKernelArg` vs `clSetKernelArgSVMPointer`).
#[derive(Clone, Copy)]
pub enum MemRef {
    /// `cl_mem`-backed buffer (`DeviceSlice`). OpenCL 1.0+.
    Buffer(cl_mem),
    /// SVM-pointer-backed buffer (`MappedSlice`/`USMSlice`). OpenCL 2.0+.
    Svm(*mut std::ffi::c_void),
}

/// The recording twin of a buffer-typed output: a memory reference + byte
/// length, threaded producer→consumer through the record walk where `execute`
/// threads the owned `DeviceSlice`/`MappedSlice`/`USMSlice`. `Copy` — the buffer
/// it refers to is owned by the source graph and must outlive the recording.
#[derive(Clone, Copy)]
pub struct BufHandle {
    /// The backing device memory (cl_mem or SVM pointer).
    pub mem: MemRef,
    /// Size of the buffer in bytes.
    pub byte_len: usize,
}

// SAFETY: a raw `cl_mem`/SVM pointer + length, opaque handles into the
// internally-synchronized OpenCL runtime; only used to bake args into a command
// buffer. The owning slice carries the real Send/Sync story; this is a borrowed
// view whose buffer the graph keeps live across replays (the home invariant).
unsafe impl Send for BufHandle {}
unsafe impl Sync for BufHandle {}

/// A concrete device buffer that can hand out its recording [`BufHandle`].
/// Implemented by `DeviceSlice` (cl_mem) and `MappedSlice`/`USMSlice` (SVM).
/// Lets polymorphic leaves (the copy verb) record over any buffer family.
pub trait RecordableBuffer {
    /// This buffer's recording handle (memory reference + byte length).
    fn record_handle(&self) -> BufHandle;
}
