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
use opencl3::types::{cl_command_queue, cl_event, cl_kernel, cl_mem, cl_uint};
use std::collections::BTreeSet;
use std::ffi::{CStr, c_void};
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
/// leaf that is "in CB mode" calls [`fill_buffer`](Self::fill_buffer) /
/// [`copy_buffer`](Self::copy_buffer) / [`ndrange_kernel`](Self::ndrange_kernel),
/// which add a `clCommand*KHR` to the buffer and return the command's
/// [`cl_sync_point_khr`] MARKER; a consumer inside the same CB waits on its
/// producer's markers. When the walk completes, the homing node calls
/// [`finalize`](Self::finalize) to seal it into a [`FinalizedCb`].
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
    /// Count of `clCommand*KHR` commands successfully recorded. Used by the boundary
    /// to DISCARD an EMPTY command buffer — a span of pure structural passthroughs
    /// (a bare `Pipe` aliasing an upstream, a `lift` of a device-resident cell) adds
    /// zero commands, and finalizing + enqueuing such a CB is pure event-sync
    /// overhead (an empty CB masquerading as work). `recorded() == 0` → the boundary
    /// skips the CB and just forwards the events the pipes already carry.
    recorded: Mutex<usize>,
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
}

// SAFETY: the CB / queue / kernel handles are opaque handles into the
// internally-synchronized runtime; the ext PFNs are plain fn pointers. The
// interior state is `Mutex`-guarded.
unsafe impl Send for CbBuilder {}
unsafe impl Sync for CbBuilder {}

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
        if err != opencl_sys::CL_SUCCESS || cb.is_null() {
            return None;
        }
        Some(CbBuilder {
            cb,
            queue,
            ext,
            kernels: Mutex::new(Vec::new()),
            eligible: Mutex::new(true),
            recorded: Mutex::new(0),
            finalized: Mutex::new(false),
            slots: Mutex::new(std::collections::BTreeSet::new()),
        })
    }

    /// Mark the build as ineligible (an SVM command or a failed add) so the homing
    /// node discards it and falls back to per-op execute.
    fn mark_ineligible(&self) {
        *self.eligible.lock().unwrap() = false;
    }

    /// Bump the recorded-command counter (called on each successful `clCommand*KHR`).
    fn count(&self) {
        *self.recorded.lock().unwrap() += 1;
    }

    /// Note that this CB baked in a buffer/scalar traceable to slot cell `id`
    /// (`Arc::as_ptr` of an `Input::Slot`'s cell) — directly or transitively through
    /// pipes. Drives precise per-slot invalidation: mutating a slot clears exactly
    /// the CBs whose `captured_slots` contains it. Cheap idempotent set insert.
    pub fn note_slot(&self, id: usize) {
        self.slots.lock().unwrap().insert(id);
    }

    /// How many `clCommand*KHR` commands have been recorded. Zero → the boundary
    /// discards the CB (empty-CB guard).
    pub fn recorded(&self) -> usize {
        *self.recorded.lock().unwrap()
    }

    /// Whether [`finalize`](Self::finalize) has already sealed this builder. The
    /// finalize-at-close span opener reads this at return: `true` → the close point
    /// sealed + enqueued the CB (nothing to do); `false` with `recorded() > 0` and
    /// eligible → the close never fired (a span with no interior seam) → the opener
    /// finalizes + enqueues at return itself.
    pub fn is_finalized(&self) -> bool {
        *self.finalized.lock().unwrap()
    }

    /// Add a buffer fill to the CB. A `cl_mem` buffer records via
    /// `clCommandFillBufferKHR`; an SVM buffer records via `clCommandSVMMemFillKHR`
    /// (extension >= 0.9.4 — if that PFN is absent, the build is marked ineligible so
    /// the boundary falls back to software). Returns the command's sync point, or
    /// `None` on any shortfall.
    pub fn fill_buffer(
        &self,
        mem: MemRef,
        pattern: &[u8],
        offset: usize,
        size: usize,
        waits: &BTreeSet<cl_sync_point_khr>,
    ) -> Option<cl_sync_point_khr> {
        let waits: Vec<cl_sync_point_khr> = waits.iter().copied().collect();
        let (wptr, n) = wait_ptr(&waits);
        let mut sp: cl_sync_point_khr = 0;
        let st = match mem {
            MemRef::Buffer(m) => {
                let fill = self.ext.fill_buffer?;
                // SAFETY: live CB + queue; `m` a valid cl_mem kept alive by the
                // graph; `pattern` outlives the call; `waits` are markers from this CB.
                unsafe {
                    fill(
                        self.cb,
                        self.queue,
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
                }
            }
            MemRef::Svm(p) => {
                // `?`: an SVM fill on a driver without the extension-0.9.4 command
                // falls back (None → boundary discards → software), never a null call.
                let svm_fill = self.ext.svm_memfill?;
                // SVM fill has no offset param — offset the pointer directly.
                let base = unsafe { (p as *mut u8).add(offset) } as *mut c_void;
                // SAFETY: `p` a valid SVM pointer kept alive by the graph; `pattern`
                // outlives the call; `waits` are markers from this CB.
                unsafe {
                    svm_fill(
                        self.cb,
                        self.queue,
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
                }
            }
        };
        if st != opencl_sys::CL_SUCCESS {
            self.mark_ineligible();
            return None;
        }
        self.count();
        Some(sp)
    }

    /// Add a device-to-device copy to the CB. Buffer↔buffer records via
    /// `clCommandCopyBufferKHR`; SVM↔SVM via `clCommandSVMMemcpyKHR` (extension
    /// version 0.9.4+ — absent PFN falls back: ineligible → software). A MIXED
    /// cl_mem/SVM pair has no single CB command and marks the build ineligible.
    /// Returns its sync point, or `None` on any shortfall.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_buffer(
        &self,
        src: MemRef,
        dst: MemRef,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        waits: &BTreeSet<cl_sync_point_khr>,
    ) -> Option<cl_sync_point_khr> {
        let waits: Vec<cl_sync_point_khr> = waits.iter().copied().collect();
        let (wptr, n) = wait_ptr(&waits);
        let mut sp: cl_sync_point_khr = 0;
        let st = match (src, dst) {
            (MemRef::Buffer(s), MemRef::Buffer(d)) => {
                let copy = self.ext.copy_buffer?;
                // SAFETY: live CB + queue; `s`/`d` valid cl_mem kept alive by the
                // graph; `waits` are markers from this CB.
                unsafe {
                    copy(
                        self.cb,
                        self.queue,
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
                }
            }
            (MemRef::Svm(s), MemRef::Svm(d)) => {
                // `?`: SVM copy on a driver without the 0.9.4 command falls back.
                let svm_copy = self.ext.svm_memcpy?;
                // SVM copy has no offset params — offset the pointers directly.
                let src_ptr = unsafe { (s as *const u8).add(src_offset) } as *const c_void;
                let dst_ptr = unsafe { (d as *mut u8).add(dst_offset) } as *mut c_void;
                // SAFETY: `s`/`d` valid SVM pointers kept alive by the graph; `waits`
                // are markers from this CB.
                unsafe {
                    svm_copy(
                        self.cb,
                        self.queue,
                        ptr::null(),
                        dst_ptr,
                        src_ptr,
                        size,
                        n,
                        wptr,
                        &mut sp,
                        ptr::null_mut(),
                    )
                }
            }
            // A mixed cl_mem/SVM copy has no single CB command → software fallback.
            _ => {
                self.mark_ineligible();
                return None;
            }
        };
        if st != opencl_sys::CL_SUCCESS {
            self.mark_ineligible();
            return None;
        }
        self.count();
        Some(sp)
    }

    /// Add an image fill to the CB via `clCommandFillImageKHR` (extension present →
    /// records; absent → ineligible → software). `image` is the image's `cl_mem`;
    /// `fill_color` is the format-appropriate fill value (4×component); `origin` and
    /// `region` are 3-element arrays (`clEnqueueFillImage` shape). Returns its sync
    /// point, or `None` on any shortfall.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_image(
        &self,
        image: MemRef,
        fill_color: &[u8],
        origin: [usize; 3],
        region: [usize; 3],
        waits: &BTreeSet<cl_sync_point_khr>,
    ) -> Option<cl_sync_point_khr> {
        // An image is always a `cl_mem` (never SVM).
        let m = match image {
            MemRef::Buffer(m) => m,
            MemRef::Svm(_) => {
                self.mark_ineligible();
                return None;
            }
        };
        let fill = self.ext.fill_image?;
        let waits: Vec<cl_sync_point_khr> = waits.iter().copied().collect();
        let (wptr, n) = wait_ptr(&waits);
        let mut sp: cl_sync_point_khr = 0;
        // SAFETY: live CB + queue; `m` a valid image cl_mem kept alive by the graph;
        // `fill_color`/`origin`/`region` outlive the call; `waits` are this CB's markers.
        let st = unsafe {
            fill(
                self.cb,
                self.queue,
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
        };
        if st != opencl_sys::CL_SUCCESS {
            self.mark_ineligible();
            return None;
        }
        self.count();
        Some(sp)
    }

    /// Add an image→image copy to the CB via `clCommandCopyImageKHR`. `src`/`dst` are
    /// image `cl_mem`s; `src_origin`/`dst_origin`/`region` are 3-element arrays.
    /// Returns its sync point, or `None` on any shortfall.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_image(
        &self,
        src: MemRef,
        dst: MemRef,
        src_origin: [usize; 3],
        dst_origin: [usize; 3],
        region: [usize; 3],
        waits: &BTreeSet<cl_sync_point_khr>,
    ) -> Option<cl_sync_point_khr> {
        let (s, d) = match (src, dst) {
            (MemRef::Buffer(s), MemRef::Buffer(d)) => (s, d),
            _ => {
                self.mark_ineligible();
                return None;
            }
        };
        let copy = self.ext.copy_image?;
        let waits: Vec<cl_sync_point_khr> = waits.iter().copied().collect();
        let (wptr, n) = wait_ptr(&waits);
        let mut sp: cl_sync_point_khr = 0;
        // SAFETY: as `fill_image`; both images valid cl_mem kept alive by the graph.
        let st = unsafe {
            copy(
                self.cb,
                self.queue,
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
        };
        if st != opencl_sys::CL_SUCCESS {
            self.mark_ineligible();
            return None;
        }
        self.count();
        Some(sp)
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

    /// Add an ND-range kernel launch to the CB. The kernel's args must already be
    /// set (the caller sets them at build time, exactly like the software record
    /// path). The kernel is retained for the CB's lifetime. Returns its sync point.
    ///
    /// # Safety
    /// `kernel` must be a valid `cl_kernel` whose args are set to the buffers this
    /// launch uses; `waits` must be sync points from this same CB.
    pub unsafe fn ndrange_kernel(
        &self,
        kernel: cl_kernel,
        global: &[usize],
        local: &[usize],
        waits: &BTreeSet<cl_sync_point_khr>,
    ) -> Option<cl_sync_point_khr> {
        let ndr = self.ext.ndrange_kernel?;
        // Own a refcount for the CB's lifetime.
        if unsafe { cl3::kernel::retain_kernel(kernel) }.is_err() {
            self.mark_ineligible();
            return None;
        }
        self.kernels.lock().unwrap().push(kernel);
        // Materialize the set into the contiguous slice the C call needs. The
        // wait-list is a SET (wait for all markers; order irrelevant) — carrying it
        // as a `BTreeSet` up to here means a consumer reading two pipes of one
        // producer can't submit that marker twice.
        let waits: Vec<cl_sync_point_khr> = waits.iter().copied().collect();
        let (wptr, n) = wait_ptr(&waits);
        let mut sp: cl_sync_point_khr = 0;
        // SAFETY: live CB + queue; kernel valid + args set + retained; `waits` are
        // markers from this same CB.
        let st = unsafe {
            ndr(
                self.cb,
                self.queue,
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
        };
        if st != opencl_sys::CL_SUCCESS {
            self.mark_ineligible();
            return None;
        }
        self.count();
        Some(sp)
    }

    /// Whether every command added so far is CB-eligible (all `cl_mem`, no failed
    /// add). A `false` here means the homing node must discard the builder and fall
    /// back to the per-op execute path.
    pub fn is_eligible(&self) -> bool {
        *self.eligible.lock().unwrap()
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
        let mut done = self.finalized.lock().unwrap();
        if *done || !self.is_eligible() {
            return None;
        }
        let finalize = self.ext.finalize?;
        // SAFETY: sealing the live CB built above.
        if unsafe { finalize(self.cb) } != opencl_sys::CL_SUCCESS {
            return None;
        }
        let enqueue = self.ext.enqueue?;
        let release = self.ext.release?;
        // Hand off the CB handle (Copy) + retained kernels to the FinalizedCb. Mark
        // sealed BEFORE releasing the lock so `Drop` (and any later `finalize`) skips
        // the handle we just gave away.
        let kernels = std::mem::take(&mut *self.kernels.lock().unwrap());
        let captured_slots = std::mem::take(&mut *self.slots.lock().unwrap());
        *done = true;
        Some(FinalizedCb {
            cb: self.cb,
            queue: self.queue,
            enqueue,
            release,
            kernels,
            captured_slots,
        })
    }
}

impl Drop for CbBuilder {
    fn drop(&mut self) {
        // A successful `finalize` set `finalized` and handed the CB + kernels to the
        // `FinalizedCb`, which now owns them → release NOTHING here. Only the DISCARD
        // path (never finalized: ineligible / empty-CB guard / finalize failure)
        // reaches the releases below.
        if *self.finalized.lock().unwrap() {
            return;
        }
        for k in self.kernels.lock().unwrap().drain(..) {
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
