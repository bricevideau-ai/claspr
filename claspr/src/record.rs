//! Record-and-replay for eager device graphs.
//!
//! A recordable [`DeviceOp`] chain can be recorded once into a reusable command
//! list and **replayed** many times — the foundation for reusable pipelines.
//! This is *staging* infrastructure: the surface (`record` / `replay`) is
//! provisional and may change as the design settles.
//!
//! # Two backends, one IR
//!
//! [`record_graph`](RecordExt::record_graph) walks the graph by `&self` and
//! builds a software command list (the portable IR). On the first [`replay`]:
//! - if the platform supports `cl_khr_command_buffer` and the recording is
//!   all-`cl_mem` (the SVM command variants — `clCommandSVMMemcpyKHR` /
//!   `clCommandSVMMemFillKHR`, extension >= 0.9.4 — are not yet wired here;
//!   see the `cb_eligible` TODO), the list is compiled **once** into a real
//!   command buffer and cached — subsequent replays are a single
//!   `clEnqueueCommandBufferKHR`;
//! - otherwise (no extension, or an SVM command) replay re-issues the software
//!   list with fresh events each call.
//!
//! Either way the result is identical; [`RecordedGraph::using_command_buffer`]
//! reports which path engaged. The provisional CB extension's entry points are
//! resolved via `clGetExtensionFunctionAddressForPlatform` (opencl3's safe
//! wrapper can't reach them).
//!
//! [`replay`]: RecordedGraph::replay
//!
//! # Why the eager engine makes this clean
//!
//! The eager `AndThen{source, next}` stores its **built** children (not a
//! closure), so the graph is structurally inspectable without running it.
//! Recording therefore walks the graph by **`&self`** and threads buffer
//! handles producer→consumer **through the same pipe topology** that `execute`
//! threads values through: each op resolves its inputs (a concrete entry-leaf
//! buffer, or a producer's output looked up by the pipe's
//! [`cell_id`](crate::eager::Pipe)), records its device commands, and registers
//! its output handle(s) under its output pipe(s). Because recording reads
//! handles by reference and never moves the buffers, it needs no keep-alive: the
//! recorded [`RecordedGraph`] borrows the source graph for `'g`, so the buffers
//! its commands reference stay live across every replay.
//!
//! # Recordability is a run-time property
//!
//! [`DeviceOp::record`] is the recording twin of `execute`, with a DEFAULT that
//! errors `NotSupported`. Device-side leaves (fill, copy, kernels) and the
//! structural combinators OVERRIDE it; host-touching leaves (`Upload`,
//! `Download`, `AndThenHost`, `OnDevice`, …) inherit the erroring default. So a
//! chain containing a non-recordable node does not fail to compile — it errors at
//! run time when [`record_graph`](RecordExt::record_graph) reaches that node.
//! (This is deliberate: the automatic segmenter records only the seam-free
//! subtrees it has already proven recordable via
//! [`contains_host_seam`](DeviceOp::contains_host_seam), and walks mixed graphs by
//! `&dyn DeviceOp`, which a compile-time bound could not express.)
//!
//! # Current scope
//!
//! - Replay re-runs against the **same** buffers. Rebinding inputs (replay
//!   against different buffers, via `cl_khr_command_buffer_mutable_dispatch`) is
//!   a later layer (slots + mutable dispatch).
//! - Recordable leaves: fill, copy (same-family), kernels (buffer + scalar
//!   args, no image). Host-touching leaves inherit the erroring
//!   [`DeviceOp::record`] default.

use crate::eager::DeviceOp;
use crate::error::{Error, Result};
use crate::queue::Launcher;
use opencl3::types::{cl_command_queue, cl_event, cl_kernel, cl_mem, cl_uint};
use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Mutex;

// ── cl_khr_command_buffer FFI (layer 2) ─────────────────────────────────────
//
// The extension is PROVISIONAL, so the ICD loader's dispatch table does NOT
// export its entry points. We resolve them per-platform via
// `clGetExtensionFunctionAddressForPlatform` (a core CL 1.2 call the loader DOES
// export) and call through opencl-sys's PFN typedefs. opencl3's safe
// `CommandBuffer` wrapper is unusable for this (its dlsym path returns -2001).

use opencl_sys::{
    cl_command_buffer_khr, cl_platform_id, cl_sync_point_khr, clCommandCopyBufferKHR_t,
    clCommandFillBufferKHR_t, clCommandNDRangeKernelKHR_t, clCreateCommandBufferKHR_t,
    clEnqueueCommandBufferKHR_t, clFinalizeCommandBufferKHR_t, clReleaseCommandBufferKHR_t,
};

/// The `cl_khr_command_buffer` entry points, resolved for one platform. Each
/// field is the opencl-sys PFN typedef (`Option<unsafe extern "C" fn …>`);
/// `None` means the loader returned a null address.
struct CommandBufferExt {
    create: clCreateCommandBufferKHR_t,
    finalize: clFinalizeCommandBufferKHR_t,
    enqueue: clEnqueueCommandBufferKHR_t,
    release: clReleaseCommandBufferKHR_t,
    fill_buffer: clCommandFillBufferKHR_t,
    copy_buffer: clCommandCopyBufferKHR_t,
    ndrange_kernel: clCommandNDRangeKernelKHR_t,
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

/// A finalized `cl_khr_command_buffer` + the queue it was built for, RAII-
/// releasing on drop. Replay is one `clEnqueueCommandBufferKHR`.
struct RecordedCb {
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
}

// SAFETY: cl_command_buffer_khr / cl_command_queue are opaque handles into the
// internally-synchronized runtime; the PFNs are plain fn pointers.
unsafe impl Send for RecordedCb {}
unsafe impl Sync for RecordedCb {}

impl Drop for RecordedCb {
    fn drop(&mut self) {
        unsafe { (self.release)(self.cb) };
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
// internally-synchronized OpenCL runtime; only used to issue commands on a
// single queue during replay. The owning slice carries the real Send/Sync
// story; this is a borrowed view kept live by the `RecordedGraph`.
unsafe impl Send for BufHandle {}
unsafe impl Sync for BufHandle {}

/// A concrete device buffer that can hand out its recording [`BufHandle`].
/// Implemented by `DeviceSlice` (cl_mem) and `MappedSlice`/`USMSlice` (SVM).
/// Lets polymorphic leaves (the copy verb) record over any buffer family.
pub trait RecordableBuffer {
    /// This buffer's recording handle (memory reference + byte length).
    fn record_handle(&self) -> BufHandle;
}

/// A structural reference to an earlier command — its index in the command list.
/// The recording twin of a `cl_event` dependency.
pub type SyncPoint = usize;

/// The wait-list a recorded op threads to its consumers — the recording twin of
/// [`Deps`](crate::eager::Deps).
pub type SyncPoints = Vec<SyncPoint>;

/// What a producer registered for one of its output pipes: the buffer handle
/// plus the sync points a consumer must wait on.
#[derive(Clone)]
struct EdgeValue {
    handle: BufHandle,
    waits: SyncPoints,
}

// ── Software command list (the layer-1 recording target) ────────────────────

/// One recorded command — the software twin of a `clCommand*KHR` entry. Baked
/// args by raw handle + **structural** dependency edges (`waits`: indices of
/// earlier commands), NOT `cl_event`s.
enum SoftCommand {
    Fill {
        buffer: MemRef,
        pattern: Vec<u8>,
        offset: usize,
        size: usize,
        waits: Vec<SyncPoint>,
    },
    Copy {
        src: MemRef,
        dst: MemRef,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        waits: Vec<SyncPoint>,
    },
    NdRange {
        /// Retained at record (`clRetainKernel`), released on `Drop`.
        kernel: cl_kernel,
        global: Vec<usize>,
        local: Vec<usize>,
        waits: Vec<SyncPoint>,
    },
}

impl SoftCommand {
    fn waits(&self) -> &[SyncPoint] {
        match self {
            SoftCommand::Fill { waits, .. }
            | SoftCommand::Copy { waits, .. }
            | SoftCommand::NdRange { waits, .. } => waits,
        }
    }

    /// Enqueue this command on `queue` with `wait` as the event wait-list.
    ///
    /// # Safety
    /// Baked handles must be valid on `queue`'s context (kept live by the
    /// `RecordedGraph` borrow) and `wait` must be live events.
    unsafe fn enqueue(&self, queue: cl_command_queue, wait: &[cl_event]) -> Result<cl_event> {
        use cl3::command_queue as q;
        let wait_ptr = if wait.is_empty() {
            ptr::null()
        } else {
            wait.as_ptr()
        };
        let n = wait.len() as cl_uint;
        let ev = match self {
            SoftCommand::Fill {
                buffer,
                pattern,
                offset,
                size,
                ..
            } => match buffer {
                MemRef::Buffer(mem) => unsafe {
                    q::enqueue_fill_buffer(
                        queue,
                        *mem,
                        pattern.as_ptr() as *const c_void,
                        pattern.len(),
                        *offset,
                        *size,
                        n,
                        wait_ptr,
                    )
                },
                MemRef::Svm(svm) => unsafe {
                    // SVM fill: byte offset into the SVM pointer.
                    let dst = (*svm as *mut u8).add(*offset) as *mut c_void;
                    q::enqueue_svm_mem_fill(
                        queue,
                        dst,
                        pattern.as_ptr() as *const c_void,
                        pattern.len(),
                        *size,
                        n,
                        wait_ptr,
                    )
                },
            },
            SoftCommand::Copy {
                src,
                dst,
                src_offset,
                dst_offset,
                size,
                ..
            } => match (src, dst) {
                (MemRef::Buffer(s), MemRef::Buffer(d)) => unsafe {
                    q::enqueue_copy_buffer(
                        queue,
                        *s,
                        *d,
                        *src_offset,
                        *dst_offset,
                        *size,
                        n,
                        wait_ptr,
                    )
                },
                (MemRef::Svm(s), MemRef::Svm(d)) => unsafe {
                    let sp = (*s as *const u8).add(*src_offset) as *const c_void;
                    let dp = (*d as *mut u8).add(*dst_offset) as *mut c_void;
                    q::enqueue_svm_mem_cpy(
                        queue,
                        opencl3::types::CL_FALSE,
                        dp,
                        sp,
                        *size,
                        n,
                        wait_ptr,
                    )
                },
                // Mixed cl_mem<->SVM copies aren't a single CL primitive; the
                // record path only produces same-family copies.
                _ => {
                    return Err(Error::NotSupported(
                        "record: mixed cl_mem/SVM copy is not a single CL command",
                    ));
                }
            },
            SoftCommand::NdRange {
                kernel,
                global,
                local,
                ..
            } => unsafe {
                q::enqueue_nd_range_kernel(
                    queue,
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
                    wait_ptr,
                )
            },
        };
        ev.map_err(|status| Error::OpenCl(opencl3::error_codes::ClError(status)))
    }
}

// ── RecordContext: the seam leaf `record` bodies emit into ──────────────────

/// The seam a [`DeviceOp::record`] body emits commands into: a software
/// command list (the portable IR) + a map from graph-edge identity
/// ([`cell_id`](crate::eager::Pipe)) to the producer's recorded output. The
/// software list is what [`RecordedGraph`] replays, or compiles once into a real
/// `cl_khr_command_buffer` — leaf bodies are backend-agnostic either way.
pub struct RecordContext {
    commands: Vec<SoftCommand>,
    /// Producer outputs keyed by their output pipe's `cell_id`.
    edges: HashMap<usize, EdgeValue>,
}

impl RecordContext {
    fn new() -> Self {
        RecordContext {
            commands: Vec::new(),
            edges: HashMap::new(),
        }
    }

    /// Resolve a leaf input: either its own concrete buffer (`concrete`), or the
    /// producer output registered under the upstream pipe `cell_id`. Returns the
    /// handle + the sync points to wait on.
    pub fn resolve_input(
        &self,
        concrete: Option<BufHandle>,
        upstream_cell: Option<usize>,
    ) -> Result<(BufHandle, SyncPoints)> {
        if let Some(h) = concrete {
            return Ok((h, SyncPoints::new()));
        }
        let cell = upstream_cell.ok_or(Error::NotSupported(
            "record: op input is neither concrete nor pipe-fed",
        ))?;
        let e = self.edges.get(&cell).ok_or(Error::NotSupported(
            "record: upstream producer was not recorded before its consumer \
             — internal ordering bug",
        ))?;
        Ok((e.handle, e.waits.clone()))
    }

    /// Register a producer output under its output pipe's `cell_id`.
    pub fn register_output(&mut self, cell: usize, handle: BufHandle, waits: SyncPoints) {
        self.edges.insert(cell, EdgeValue { handle, waits });
    }

    /// Record a buffer fill (cl_mem or SVM). Returns its [`SyncPoint`].
    pub fn fill_buffer(
        &mut self,
        buffer: MemRef,
        pattern: Vec<u8>,
        offset: usize,
        size: usize,
        waits: SyncPoints,
    ) -> SyncPoint {
        let idx = self.commands.len();
        self.commands.push(SoftCommand::Fill {
            buffer,
            pattern,
            offset,
            size,
            waits,
        });
        idx
    }

    /// Record a device-to-device copy (both cl_mem or both SVM). Returns its
    /// [`SyncPoint`].
    pub fn copy_buffer(
        &mut self,
        src: MemRef,
        dst: MemRef,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        waits: SyncPoints,
    ) -> SyncPoint {
        let idx = self.commands.len();
        self.commands.push(SoftCommand::Copy {
            src,
            dst,
            src_offset,
            dst_offset,
            size,
            waits,
        });
        idx
    }

    /// Set one buffer argument on `kernel` at `arg_index`, following claspr's
    /// `(pointer, len)` convention (slice params are a pointer + a `usize`
    /// length). Dispatches `clSetKernelArg` for a `cl_mem` or
    /// `clSetKernelArgSVMPointer` for an SVM buffer. Advances `arg_index` by 2.
    ///
    /// # Safety
    /// `kernel` must be valid and `mem` a live buffer; `arg_index`/`arg_index+1`
    /// must be the pointer/length slots of a slice parameter.
    pub unsafe fn set_buffer_arg(
        &self,
        kernel: cl_kernel,
        arg_index: &mut cl_uint,
        mem: MemRef,
        elem_count: usize,
    ) -> Result<()> {
        use std::ffi::c_void;
        // arg N: the buffer pointer (cl_mem object or raw SVM pointer).
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
        // arg N+1: the element-count length (matches the slice `set` convention).
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

    /// Set one buffer **pointer** argument on `kernel` at `arg_index` — the
    /// scalar-by-reference shape (`#[spirv(cross_workgroup)] &T` / `&mut T`),
    /// which rust-gpu lowers to a bare pointer-to-scalar with **no** length
    /// operand. Dispatches `clSetKernelArg` for a `cl_mem` or
    /// `clSetKernelArgSVMPointer` for an SVM buffer, then advances
    /// `arg_index` by exactly 1 (unlike [`set_buffer_arg`](Self::set_buffer_arg), which sets a
    /// `(pointer, len)` pair and advances by 2).
    ///
    /// # Safety
    /// `kernel` must be valid and `mem` a live buffer; `arg_index` must be the
    /// single pointer slot of a scalar-ref parameter.
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

    /// Set one scalar (by-value) argument on `kernel` at `arg_index` from its
    /// raw bytes, then advance `arg_index` by 1.
    ///
    /// # Safety
    /// `kernel` must be valid and `bytes` must be the correct size/layout for
    /// the scalar parameter at `arg_index`.
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

    /// Record an ND-range kernel launch. The kernel is retained for the
    /// recording's lifetime and released on drop. Returns its [`SyncPoint`].
    ///
    /// # Safety
    /// `kernel` must be a valid `cl_kernel` whose arguments are already set to
    /// the buffers/scalars this launch uses (the caller sets them at record
    /// time); it must remain valid for the recording (the retain ensures this).
    pub unsafe fn ndrange_kernel(
        &mut self,
        kernel: cl_kernel,
        global: Vec<usize>,
        local: Vec<usize>,
        waits: SyncPoints,
    ) -> Result<SyncPoint> {
        // Own a refcount for the recording's lifetime.
        unsafe { cl3::kernel::retain_kernel(kernel) }
            .map_err(|s| Error::OpenCl(opencl3::error_codes::ClError(s)))?;
        let idx = self.commands.len();
        self.commands.push(SoftCommand::NdRange {
            kernel,
            global,
            local,
            waits,
        });
        Ok(idx)
    }
}

// ── RecordedGraph: the materialized, replayable form ────────────────────────

/// A recorded eager graph, ready to [`replay`](RecordedGraph::replay) as many
/// times as wanted. Borrows the source graph for `'g` so the buffers (and
/// kernels) its commands reference stay live across every replay.
///
/// The software command list is the portable IR. On the first replay, if the
/// platform supports `cl_khr_command_buffer` AND the recording is all-`cl_mem`
/// (SVM CB commands exist — `clCommandSVMMem{cpy,Fill}KHR` — but are not yet
/// wired, so SVM recordings currently take the software path), the list is
/// compiled once into a real command buffer and cached; subsequent replays are a single
/// `clEnqueueCommandBufferKHR`. Otherwise (no extension / SVM commands), replay
/// re-issues the software list with fresh events each call. Either way the
/// observable result is identical.
pub struct RecordedGraph<'g> {
    commands: Vec<SoftCommand>,
    /// Lazily-built command buffer, keyed by the queue it was finalized for.
    /// `None` until the first replay decides whether a CB is usable.
    cb: Mutex<Option<RecordedCb>>,
    _borrow: PhantomData<&'g ()>,
}

impl Drop for RecordedGraph<'_> {
    fn drop(&mut self) {
        // Release the kernel refcounts retained at record time.
        for cmd in &self.commands {
            if let SoftCommand::NdRange { kernel, .. } = cmd {
                // SAFETY: each was `retain_kernel`'d in `ndrange_kernel`; balance
                // it. Best-effort on drop.
                let _ = unsafe { cl3::kernel::release_kernel(*kernel) };
            }
        }
    }
}

impl RecordedGraph<'_> {
    /// Whether a real `cl_khr_command_buffer` has been built and cached for this
    /// recording (set on the first replay that successfully compiles one). Until
    /// the first replay, or when replay falls back to the software path (no
    /// extension / SVM commands), this is `false`. Introspection for tests +
    /// callers that want to confirm the fast path engaged.
    pub fn using_command_buffer(&self) -> bool {
        self.cb.lock().unwrap().is_some()
    }

    /// True if every command is `cl_mem`-backed. SVM commands currently force
    /// software replay — NOT because the extension lacks SVM variants (it has
    /// `clCommandSVMMemcpyKHR` / `clCommandSVMMemFillKHR`, on OpenCL 2.0+ with
    /// extension version 0.9.4 or newer), but because those entry points aren't
    /// wired into [`CommandBufferExt`] yet. TODO: resolve the SVM command PFNs
    /// and emit them for `MemRef::Svm`, then allow SVM recordings into the CB.
    fn cb_eligible(&self) -> bool {
        self.commands.iter().all(|c| match c {
            SoftCommand::Fill { buffer, .. } => matches!(buffer, MemRef::Buffer(_)),
            SoftCommand::Copy { src, dst, .. } => {
                matches!(src, MemRef::Buffer(_)) && matches!(dst, MemRef::Buffer(_))
            }
            SoftCommand::NdRange { .. } => true,
        })
    }

    /// Replay the recording on `launcher`'s queue, blocking until every recorded
    /// command completes. Reusable — call repeatedly (serially; the buffers are
    /// shared across replays). Uses a cached `cl_khr_command_buffer` when one was
    /// built, else the software path.
    pub fn replay<L: Launcher + ?Sized>(&self, launcher: &L) -> Result<()> {
        let queue = launcher.cl_queue().get();

        // Fast path: a previously-built CB for this queue → one enqueue + wait.
        {
            let guard = self.cb.lock().unwrap();
            if let Some(rec) = guard.as_ref()
                && rec.queue == queue
            {
                return unsafe { Self::enqueue_cb(rec, queue) };
            }
        }

        // Try to build a CB once (all-cl_mem + extension present). On any
        // shortfall, fall through to software replay.
        if self.cb_eligible()
            && let Some(rec) = self.try_build_cb(launcher, queue)
        {
            let r = unsafe { Self::enqueue_cb(&rec, queue) };
            *self.cb.lock().unwrap() = Some(rec);
            return r;
        }

        self.replay_software(queue)
    }

    /// One `clEnqueueCommandBufferKHR` + wait on its completion event.
    ///
    /// # Safety
    /// `rec` must be a finalized CB built for `queue`.
    unsafe fn enqueue_cb(rec: &RecordedCb, mut queue: cl_command_queue) -> Result<()> {
        let mut event: cl_event = ptr::null_mut();
        let status = unsafe {
            (rec.enqueue)(
                1,
                &mut queue as *mut cl_command_queue,
                rec.cb,
                0,
                ptr::null(),
                &mut event,
            )
        };
        if status != opencl_sys::CL_SUCCESS {
            return Err(Error::OpenCl(opencl3::error_codes::ClError(status)));
        }
        crate::Event::new(event).wait().map_err(Error::OpenCl)
    }

    /// Build + finalize a command buffer from the software command list, or
    /// `None` if the extension isn't reachable / a command fails to record.
    fn try_build_cb<L: Launcher + ?Sized>(
        &self,
        launcher: &L,
        queue: cl_command_queue,
    ) -> Option<RecordedCb> {
        let platform = launcher.context().device().platform().raw_id();
        let ext = CommandBufferExt::load(platform)?;
        let (create, finalize, enqueue, release) =
            (ext.create?, ext.finalize?, ext.enqueue?, ext.release?);
        let fill = ext.fill_buffer?;
        let copy = ext.copy_buffer?;
        let ndr = ext.ndrange_kernel?;

        // Create the command buffer over the single queue.
        let mut q = queue;
        let mut err: opencl_sys::cl_int = 0;
        let cb = unsafe { create(1, &mut q as *mut _, ptr::null(), &mut err) };
        if err != opencl_sys::CL_SUCCESS || cb.is_null() {
            return None;
        }
        // RAII from here so any early return releases the CB.
        let rec = RecordedCb {
            cb,
            queue,
            enqueue,
            release,
        };

        // Record each command, threading sync points by list index.
        let mut sps: Vec<cl_sync_point_khr> = Vec::with_capacity(self.commands.len());
        for cmd in &self.commands {
            let waits: Vec<cl_sync_point_khr> = cmd.waits().iter().map(|&i| sps[i]).collect();
            let wptr = if waits.is_empty() {
                ptr::null()
            } else {
                waits.as_ptr()
            };
            let n = waits.len() as cl_uint;
            let mut sp: cl_sync_point_khr = 0;
            let st = match cmd {
                SoftCommand::Fill {
                    buffer: MemRef::Buffer(mem),
                    pattern,
                    offset,
                    size,
                    ..
                } => unsafe {
                    fill(
                        cb,
                        queue,
                        ptr::null(),
                        *mem as opencl_sys::cl_mem,
                        pattern.as_ptr() as *const c_void,
                        pattern.len(),
                        *offset,
                        *size,
                        n,
                        wptr,
                        &mut sp,
                        ptr::null_mut(),
                    )
                },
                SoftCommand::Copy {
                    src: MemRef::Buffer(s),
                    dst: MemRef::Buffer(d),
                    src_offset,
                    dst_offset,
                    size,
                    ..
                } => unsafe {
                    copy(
                        cb,
                        queue,
                        ptr::null(),
                        *s as opencl_sys::cl_mem,
                        *d as opencl_sys::cl_mem,
                        *src_offset,
                        *dst_offset,
                        *size,
                        n,
                        wptr,
                        &mut sp,
                        ptr::null_mut(),
                    )
                },
                SoftCommand::NdRange {
                    kernel,
                    global,
                    local,
                    ..
                } => unsafe {
                    ndr(
                        cb,
                        queue,
                        ptr::null(),
                        *kernel as opencl_sys::cl_kernel,
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
                },
                // SVM commands aren't wired into the CB yet (see `cb_eligible`),
                // so it already excluded these — unreachable until the SVM CB
                // command PFNs are resolved.
                _ => return None,
            };
            if st != opencl_sys::CL_SUCCESS {
                return None;
            }
            sps.push(sp);
        }

        if unsafe { finalize(cb) } != opencl_sys::CL_SUCCESS {
            return None;
        }
        Some(rec)
    }

    /// Software replay: re-issue each command on `queue`, threading fresh events
    /// along the recorded structural edges, then wait on every command's event.
    fn replay_software(&self, queue: cl_command_queue) -> Result<()> {
        let mut events: Vec<cl_event> = Vec::with_capacity(self.commands.len());
        for cmd in &self.commands {
            let wait_events: Vec<cl_event> = cmd.waits().iter().map(|&i| events[i]).collect();
            // SAFETY: handles captured from the borrowed graph (live for `'g`);
            // `wait_events` are this replay's live events.
            let ev = unsafe { cmd.enqueue(queue, &wait_events)? };
            events.push(ev);
        }
        let mut first_err: Option<Error> = None;
        for ev in events {
            let e = crate::Event::new(ev);
            if let Err(err) = e.wait()
                && first_err.is_none()
            {
                first_err = Some(Error::OpenCl(err));
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

// ── RecordExt: the `.record_graph()` entry ──────────────────────────────────

/// `.record_graph()` on any [`DeviceOp`] chain — an internal helper that records
/// the WHOLE chain into one [`RecordedGraph`] via [`DeviceOp::record`]. Blanket-
/// implemented for every `DeviceOp`; a chain containing a non-recordable node (a
/// host seam / transfer) surfaces the `DeviceOp::record` default error at RUN time
/// rather than being rejected at compile time (the former `RecordableOp` bound).
///
/// This is no longer the user-facing path — the command-buffer layer is invisible
/// (built + cached automatically by `sync`); `record_graph` records a single
/// seam-free subtree and is used by the segmenter and the record/replay tests.
pub trait RecordExt: DeviceOp {
    /// Record this graph into a reusable [`RecordedGraph`] without running it.
    /// The returned graph borrows `self`, so the buffers/kernels its leaves
    /// reference stay live for every replay. Errors if any node is not
    /// device-recordable (the [`DeviceOp::record`] default).
    fn record_graph(&self) -> Result<RecordedGraph<'_>> {
        let mut ctx = RecordContext::new();
        DeviceOp::record(self, &mut ctx)?;
        Ok(RecordedGraph {
            commands: ctx.commands,
            cb: Mutex::new(None),
            _borrow: PhantomData,
        })
    }
}

impl<O: DeviceOp + ?Sized> RecordExt for O {}

// Re-export the leaf/combinator record impls (kept in eager.rs / the macro,
// next to the `execute` bodies they mirror).
