//! Shared Virtual Memory ([`MappedSlice`]) — OpenCL 2.0+ coarse-grain
//! SVM buffers.
//!
//! SVM gives kernel and host the *same pointer* into a single
//! allocation. claspr exposes coarse-grain SVM today: host access
//! requires [`map`](MappedSlice::map) / [`map_mut`](MappedSlice::map_mut)
//! around the bytes you want to read or write, with the runtime
//! ensuring the device-side view is coherent at the boundaries.
//!
//! Construction is gated on [`crate::SvmLevel`]:
//! `MappedSlice::alloc` returns [`crate::Error::SvmNotAvailable`]
//! when the device reports [`crate::SvmLevel::None`]. Check
//! `ctx.svm_capability()` if you want to fall back to a
//! [`crate::DeviceSlice`] gracefully.
//!
//! # Example
//!
//! ```ignore
//! use claspr::{Context, MappedSlice, SvmLevel};
//!
//! let ctx = Context::any()?;
//! if ctx.svm_capability() == SvmLevel::None {
//!     return skip("device has no SVM");
//! }
//!
//! let mut buf = MappedSlice::<u32>::alloc(&ctx, 1024)?;
//! {
//!     let mut view = buf.map_mut(&ctx)?;
//!     for (i, slot) in view.iter_mut().enumerate() {
//!         *slot = i as u32;
//!     }
//! } // view drops -> implicit clEnqueueSVMUnmap
//!
//! kernels.process(&ctx, [1024], &buf)?;
//!
//! let view = buf.map(&ctx)?;
//! assert_eq!(view[0], 0);
//! ```

use crate::buffer::Buffer;
use crate::context::{Context, SvmLevel};
use crate::error::{Error, Result};
use crate::launch::KernelArg;
use crate::op::{ProfileCb, ProfilingInfo, register_profiling_callback};
use crate::queue::Launcher;
use opencl3::command_queue::{CommandQueue, enqueue_svm_map, enqueue_svm_unmap};
use opencl3::event::{Event, release_event, retain_event};
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE, CL_MEM_READ_WRITE, svm_alloc};
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_event, cl_int, cl_uint};
use std::ffi::c_void;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::sync::{Arc, Mutex};

fn cl_to_err(code: cl_int) -> Error {
    Error::OpenCl(opencl3::error_codes::ClError(code))
}

// ── MappedSlice ────────────────────────────────────────────────────

/// A typed Shared Virtual Memory allocation.
///
/// Construction allocates via `clSVMAlloc`; Drop releases via
/// `clEnqueueSVMFree` on the context's default queue (queue-ordered,
/// so it can't race in-flight commands using the pointer — the
/// immediate `clSVMFree` would be UB in that case per the CL spec).
/// Host access is RAII-guarded via [`map`](Self::map) and
/// [`map_mut`](Self::map_mut): each returns a guard that issues
/// `clEnqueueSVMMap` on construction and `clEnqueueSVMUnmap` on Drop.
///
/// ## Cross-queue ordering on Drop
///
/// Drop's `clEnqueueSVMFree` runs on the context's default in-order
/// queue, with every recorded use as its wait-list. Uses are
/// recorded automatically:
///
/// - **Kernel launches** that take `MappedSlice<T>` as a `KernelArg`:
///   [`LaunchOp::into_event`][lo] calls [`KernelArg::register_completion`][ka]
///   after enqueue, which retains the completion event and pushes it
///   onto this buffer's in-flight-use list.
/// - **Host-view release** path: `MappedSliceHostView::Drop` /
///   `ReleaseMappedSliceOp` push the unmap event via [`register_use`](Self::register_use).
///
/// The accumulation is correct under out-of-order scheduling: every
/// in-flight use is in the wait-list, not just the most recently
/// enqueued one. The cross-queue case (chain's OOO queue vs the
/// context's default in-order queue) is handled by `clEnqueueSVMFree`'s
/// wait-list semantics — events from any queue in the context can
/// gate it.
///
/// Hand-rolled SVM use (`ctx.launch(&shared_buf, ...)`, manual
/// `clSetKernelArgSVMPointer`) that doesn't go through `LaunchOp`
/// must call [`register_use`](Self::register_use) explicitly to
/// stay Drop-safe.
///
/// [lo]: crate::LaunchOp
/// [ka]: crate::KernelArg::register_completion
pub struct MappedSlice<T> {
    ptr: *mut T,
    len: usize,
    ctx: Context,
    /// Every event that touched this SVM pointer and is still in
    /// flight. Drop passes all of them as the `clEnqueueSVMFree`
    /// wait-list, so the free queue-orders after every prior use no
    /// matter which queue produced it.
    ///
    /// Vec, not Option, because on an out-of-order queue "most
    /// recent enqueue" ≠ "last to finish" — every in-flight use must
    /// be in the wait-list, not just the most recent one. Auto-fed
    /// by [`KernelArg::register_completion`] for kernel launches
    /// that take `MappedSlice<T>` as an arg, and by the host-view
    /// release path's unmap event.
    ///
    /// Mutex-protected because the buffer is commonly shared via
    /// `Arc<MappedSlice<T>>` (e.g. through `.arc()` in claspr-async
    /// chains) and multiple threads may register concurrently.
    last_use: Mutex<Vec<Arc<Event>>>,
}

// SAFETY: the SVM pointer is a runtime-owned allocation in
// host-accessible memory; OpenCL guarantees thread-safety for API
// calls on it (CL §3.4.1). Aliasing is governed by the map guards,
// which use the borrow checker to enforce exclusivity for `map_mut`.
unsafe impl<T: Send> Send for MappedSlice<T> {}
unsafe impl<T: Sync> Sync for MappedSlice<T> {}

impl<T: Default + Copy> MappedSlice<T> {
    /// Allocate `len` elements of T in SVM memory, zero-initialised
    /// via `clEnqueueSVMMemFill(T::default())` on the context's
    /// default queue. Blocks until the fill completes.
    ///
    /// Returns [`Error::SvmNotAvailable`] if the context's device
    /// reports [`SvmLevel::None`] for `CL_DEVICE_SVM_CAPABILITIES`.
    ///
    /// The `T: Default + Copy` bound makes the buffer's contents a
    /// valid `T` value before any read; matches
    /// [`DeviceSlice::alloc`](crate::DeviceSlice::alloc) and
    /// [`USMSlice::alloc`](crate::USMSlice::alloc).
    ///
    /// Code paths that write the entire buffer immediately afterward
    /// (e.g. `mapped_slice_filled`, `mapped_slice_upload`) should use
    /// [`alloc_uninit`](Self::alloc_uninit) instead to skip the
    /// redundant zero-fill.
    pub fn alloc(ctx: &Context, len: usize) -> Result<Self> {
        let slice = Self::alloc_uninit(ctx, len)?;
        slice.fill(ctx, T::default()).wait()?;
        Ok(slice)
    }
}

impl<T> MappedSlice<T> {
    /// Allocate `len` elements of T in SVM memory, leaving the bytes
    /// uninitialised. Cheaper than [`alloc`](Self::alloc) when the
    /// caller writes the whole buffer before any read.
    ///
    /// Reading uninitialised bytes is unspecified at the OpenCL level
    /// and UB at the Rust level if interpreted as a non-trivial `T`.
    /// Prefer [`alloc`](Self::alloc) unless you know the buffer is
    /// fully written before any read.
    pub fn alloc_uninit(ctx: &Context, len: usize) -> Result<Self> {
        if ctx.svm_capability() == SvmLevel::None {
            return Err(Error::SvmNotAvailable);
        }
        let size = len.saturating_mul(std::mem::size_of::<T>());
        // SAFETY: CL_MEM_READ_WRITE is a valid flag combination;
        // alignment is the natural alignment of T (must fit in u32 —
        // any T with alignment > u32::MAX is degenerate). svm_alloc
        // returns null on failure, which cl3 maps to CL_INVALID_VALUE.
        let raw = unsafe {
            svm_alloc(
                ctx.raw_context().get(),
                CL_MEM_READ_WRITE,
                size,
                std::mem::align_of::<T>() as cl_uint,
            )
            .map_err(cl_to_err)?
        };
        Ok(MappedSlice {
            ptr: raw.cast::<T>(),
            len,
            ctx: ctx.clone(),
            last_use: Mutex::new(Vec::new()),
        })
    }

    /// Append `event` to this buffer's in-flight-use list. Drop passes
    /// every accumulated event to `clEnqueueSVMFree`'s wait-list, so
    /// the free is queue-ordered after every recorded use — including
    /// concurrent ones on an out-of-order queue, where "most recent
    /// enqueue" is not the same as "last to finish".
    ///
    /// Most users never call this directly: [`KernelArg::register_completion`]
    /// invokes it automatically for every kernel launch whose args
    /// include a `MappedSlice<T>`. The host-view release path also
    /// records its unmap event. The public entry-point is exposed so
    /// hand-rolled SVM use (raw `ctx.launch`, manual `clSetKernelArgSVMPointer`)
    /// can keep Drop safe.
    pub fn register_use(&self, event: Arc<Event>) {
        self.last_use
            .lock()
            .expect("last_use mutex poisoned")
            .push(event);
    }

    /// Map this buffer for host read access. Returns a RAII guard
    /// that derefs to `&[T]` and unmaps on Drop.
    ///
    /// The map is blocking — `clEnqueueSVMMap` with `CL_TRUE` for
    /// the blocking flag and `CL_MAP_READ`.
    pub fn map<'a, L: Launcher>(&'a self, launcher: &L) -> Result<MappedReadGuard<'a, T>> {
        MappedReadGuard::new(self, launcher)
    }

    /// Map this buffer for host read+write access. Returns a RAII
    /// guard that derefs to `&mut [T]` and unmaps on Drop. The
    /// `&mut self` receiver gives the borrow checker the
    /// exclusivity guarantee needed for `DerefMut`.
    pub fn map_mut<'a, L: Launcher>(&'a mut self, launcher: &L) -> Result<MappedWriteGuard<'a, T>> {
        MappedWriteGuard::new(self, launcher)
    }

    /// Raw SVM pointer for direct use (e.g. passing to a kernel arg
    /// out-of-band, or interoperating with hand-written OpenCL).
    pub fn ptr(&self) -> *mut T {
        self.ptr
    }

    /// Begin filling this SVM buffer's contents with `value` repeated
    /// for every element — wraps `clEnqueueSVMMemFill` (CL 2.0+).
    /// SVM analog of [`crate::DeviceSlice::fill`].
    ///
    /// Takes `&self` because the underlying SVM pointer is intentionally
    /// aliased (host map guards govern exclusivity separately). The
    /// resulting event is auto-registered on this buffer's
    /// last-use list, so Drop's `clEnqueueSVMFree` waits for the
    /// fill to finish.
    pub fn fill<'a, L: Launcher + ?Sized>(&'a self, launcher: &'a L, value: T) -> SvmFillOp<'a, T>
    where
        T: Copy,
    {
        SvmFillOp {
            queue: launcher.cl_queue(),
            owner: self,
            pattern: value,
            deps: Vec::new(),
            profile_cb: None,
        }
    }

    /// Begin a SVM→SVM copy from `self` into `dst` — wraps
    /// `clEnqueueSVMMemcpy` (CL 2.0+). SVM analog of
    /// [`crate::DeviceSlice::copy_to`].
    ///
    /// Both buffers must be on the same `Context` (the runtime
    /// enforces this; mismatch surfaces as a CL error at terminal
    /// time). The resulting event is registered on **both** buffers'
    /// last-use lists so Drop on either side is ordered after the
    /// copy.
    pub fn copy_to<'a, L: Launcher + ?Sized>(
        &'a self,
        dst: &'a MappedSlice<T>,
        launcher: &'a L,
    ) -> SvmCopyOp<'a, T> {
        SvmCopyOp {
            queue: launcher.cl_queue(),
            src: self,
            dst,
            deps: Vec::new(),
            profile_cb: None,
        }
    }
}

/// Metadata-only `Debug` — does not read through the SVM pointer
/// (would race with in-flight kernel work and require holding a map
/// guard) and doesn't require `T: Debug`.
impl<T> fmt::Debug for MappedSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedSlice")
            .field("len", &self.len)
            .field("element_size", &std::mem::size_of::<T>())
            .finish_non_exhaustive()
    }
}

impl<T> Buffer<T> for MappedSlice<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn ctx(&self) -> &Context {
        &self.ctx
    }
}

// ── SvmFillOp / SvmCopyOp builders ─────────────────────────────────
//
// Same terminal / modifier shape as `DeviceSlice`'s FillOp / CopyOp.
// Take `&MappedSlice<T>` rather than a raw `cl_mem` reference because
// the terminal needs to call `register_use` on the buffer(s) so Drop's
// `clEnqueueSVMFree` waits for the fill/copy event.

/// Lazy builder for `clEnqueueSVMMemFill`. Returned by
/// [`MappedSlice::fill`].
pub struct SvmFillOp<'a, T: Copy> {
    queue: &'a CommandQueue,
    owner: &'a MappedSlice<T>,
    pattern: T,
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a, T: Copy> SvmFillOp<'a, T> {
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    pub fn wait(self) -> Result<()> {
        let event = self.into_event()?;
        event.wait()?;
        Ok(())
    }

    pub fn submit(self) -> Result<Event> {
        self.into_event()
    }

    pub(crate) fn into_event(self) -> Result<Event> {
        let size = self.owner.len * std::mem::size_of::<T>();
        // SAFETY: svm_ptr is a valid SVM allocation in the queue's
        // context (caller's responsibility — same as
        // `DeviceSlice::fill`). Pattern is a single T byte-copied
        // across the buffer.
        let event = unsafe {
            self.queue.enqueue_svm_mem_fill(
                self.owner.ptr as *mut c_void,
                std::slice::from_ref(&self.pattern),
                size,
                &self.deps,
            )?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        // Auto-register on the source buffer's last-use list so
        // Drop's free waits for this fill. clRetainEvent bumps the
        // cl_event refcount so the returned `event` and the
        // registered Arc<Event> each hold an independent reference;
        // both `Event::drop`s call `clReleaseEvent` to balance.
        // SAFETY: event.get() is live; retain is paired with the
        // Event::drop inside the Arc.
        unsafe {
            retain_event(event.get())
                .map_err(|code| Error::OpenCl(opencl3::error_codes::ClError(code)))?;
        }
        self.owner
            .register_use(std::sync::Arc::new(Event::new(event.get())));
        Ok(event)
    }
}

/// Lazy builder for `clEnqueueSVMMemcpy`. Returned by
/// [`MappedSlice::copy_to`].
pub struct SvmCopyOp<'a, T> {
    queue: &'a CommandQueue,
    src: &'a MappedSlice<T>,
    dst: &'a MappedSlice<T>,
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a, T> SvmCopyOp<'a, T> {
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    pub fn wait(self) -> Result<()> {
        let event = self.into_event()?;
        event.wait()?;
        Ok(())
    }

    pub fn submit(self) -> Result<Event> {
        self.into_event()
    }

    pub(crate) fn into_event(self) -> Result<Event> {
        if self.src.len != self.dst.len {
            return Err(Error::LengthMismatch {
                src: self.src.len,
                dst: self.dst.len,
            });
        }
        let size = self.src.len * std::mem::size_of::<T>();
        // SAFETY: both SVM pointers are valid allocations in the
        // queue's context (caller's responsibility — runtime gives
        // CL_INVALID_CONTEXT on mismatch). CL_NON_BLOCKING so the
        // event encodes completion; .wait()/.submit() pick how to
        // observe it.
        let event = unsafe {
            self.queue.enqueue_svm_mem_cpy(
                CL_NON_BLOCKING,
                self.dst.ptr as *mut c_void,
                self.src.ptr as *const c_void,
                size,
                &self.deps,
            )?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        // Two extra refcounts (one per buffer's last_use). The
        // returned `event` keeps its original refcount. Three
        // independent Event::drop → clReleaseEvent, balanced.
        // SAFETY: event.get() is live; each retain is paired with
        // a matching Event::drop in the registered Arc.
        unsafe {
            retain_event(event.get())
                .map_err(|code| Error::OpenCl(opencl3::error_codes::ClError(code)))?;
            retain_event(event.get())
                .map_err(|code| Error::OpenCl(opencl3::error_codes::ClError(code)))?;
        }
        let src_arc = std::sync::Arc::new(Event::new(event.get()));
        let dst_arc = std::sync::Arc::new(Event::new(event.get()));
        self.src.register_use(src_arc);
        self.dst.register_use(dst_arc);
        Ok(event)
    }
}

impl<T> Drop for MappedSlice<T> {
    fn drop(&mut self) {
        // Use `clEnqueueSVMFree` on the context's default queue, NOT
        // the immediate `clSVMFree`. Per the CL spec, `clSVMFree` does
        // NOT wait for in-flight commands using the pointer to finish
        // before deallocation — using the pointer after `clSVMFree`
        // is UB. `clEnqueueSVMFree` queues the free so it runs only
        // after the queue's prior commands complete and after any
        // explicit wait-list events.
        //
        // Cross-queue ordering: every recorded use is in the wait-list,
        // so the free is queue-side-ordered after every in-flight
        // touch of the SVM pointer, including OOO-concurrent uses on
        // queues other than the default in-order one. Registrations
        // come from `KernelArg::register_completion` (automatic for
        // every kernel launch) and the host-view release path's unmap.
        let queue = self.ctx.raw_default_queue();
        let svm_ptr = self.ptr as *const std::ffi::c_void;
        let events: Vec<Arc<Event>> =
            std::mem::take(&mut *self.last_use.lock().expect("last_use mutex poisoned"));
        let wait_list: Vec<cl_event> = events.iter().map(|e| e.get()).collect();
        // SAFETY: ptr was returned by svm_alloc on this context; we
        // queue exactly one free for it. Every wait-list event is
        // held alive via its Arc until after the enqueue returns —
        // OpenCL retains them internally for the wait-list once
        // enqueued.
        let res =
            unsafe { queue.enqueue_svm_free(&[svm_ptr], None, std::ptr::null_mut(), &wait_list) };
        // Hold the Arcs until after the enqueue call.
        drop(events);
        if res.is_err() {
            self.ctx.record_err();
        }
    }
}

impl<T> KernelArg for MappedSlice<T> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        // Slice decomposition matches rust-gpu's
        // `#[spirv(cross_workgroup)] &mut [T]` lowering: SVM
        // pointer first (via clSetKernelArgSVMPointer), then length
        // as a regular scalar arg.
        let len: usize = self.len;
        // SAFETY: set_arg_svm is unsafe because the pointer must be
        // a valid SVM allocation on the kernel's context — which
        // ours is, by construction.
        unsafe {
            exec.set_arg_svm(self.ptr).set_arg(&len);
        }
    }

    /// Retain the kernel's completion event and push it onto our
    /// `last_use` list, so Drop's `clEnqueueSVMFree` queue-orders
    /// after this launch. Without this, dropping a MappedSlice
    /// while a kernel using its SVM pointer is still in flight
    /// would be UB.
    fn register_completion(&self, event: &Event) {
        let raw = event.get();
        // SAFETY: retain bumps cl_event's refcount; we construct a
        // second owning `Event` from the raw handle that will release
        // on Drop. Pair: retain here, release when the Arc<Event>
        // (held in `last_use`) drops after the Drop's enqueue.
        if unsafe { opencl3::event::retain_event(raw) }.is_err() {
            self.ctx.record_err();
            return;
        }
        let owned = Event::from(raw);
        self.register_use(Arc::new(owned));
    }
}

// ── Map guards ──────────────────────────────────────────────────────

/// RAII guard for a SVM read map. Drop issues `clEnqueueSVMUnmap`
/// and the inner `RetainedQueue` releases the queue handle.
pub struct MappedReadGuard<'a, T> {
    buf: &'a MappedSlice<T>,
    queue: crate::util::RetainedQueue,
}

impl<'a, T> MappedReadGuard<'a, T> {
    fn new<L: Launcher>(buf: &'a MappedSlice<T>, launcher: &L) -> Result<Self> {
        let queue = crate::util::RetainedQueue::from_queue(launcher.cl_queue())?;
        // SAFETY: blocking map for read; queue is alive (RetainedQueue).
        let evt = unsafe {
            enqueue_svm_map(
                queue.raw(),
                CL_BLOCKING,
                CL_MAP_READ,
                buf.ptr.cast(),
                buf.len * std::mem::size_of::<T>(),
                0,
                ptr::null(),
            )
            .map_err(cl_to_err)?
        };
        unsafe { release_event(evt).map_err(cl_to_err)? };
        Ok(MappedReadGuard { buf, queue })
    }
}

impl<T> Deref for MappedReadGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: the SVM pointer is valid + mapped for read for
        // this guard's lifetime.
        unsafe { crate::util::mapped_slice(self.buf.ptr, self.buf.len) }
    }
}

impl<T> Drop for MappedReadGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: ptr was mapped in `new`; unmap exactly once now.
        // The `queue: RetainedQueue` field drops after this body
        // returns, releasing the queue handle.
        let unmap =
            unsafe { enqueue_svm_unmap(self.queue.raw(), self.buf.ptr.cast(), 0, ptr::null()) };
        if let Ok(evt) = unmap {
            let _ = unsafe { release_event(evt) };
        } else {
            self.buf.ctx.record_err();
        }
    }
}

/// RAII guard for a SVM write map. Drop issues `clEnqueueSVMUnmap`
/// and the inner `RetainedQueue` releases the queue handle.
pub struct MappedWriteGuard<'a, T> {
    buf: &'a mut MappedSlice<T>,
    queue: crate::util::RetainedQueue,
}

impl<'a, T> MappedWriteGuard<'a, T> {
    fn new<L: Launcher>(buf: &'a mut MappedSlice<T>, launcher: &L) -> Result<Self> {
        let queue = crate::util::RetainedQueue::from_queue(launcher.cl_queue())?;
        let ptr = buf.ptr;
        let len = buf.len;
        // SAFETY: blocking map for read+write.
        let evt = unsafe {
            enqueue_svm_map(
                queue.raw(),
                CL_BLOCKING,
                CL_MAP_READ | CL_MAP_WRITE,
                ptr.cast(),
                len * std::mem::size_of::<T>(),
                0,
                ptr::null(),
            )
            .map_err(cl_to_err)?
        };
        unsafe { release_event(evt).map_err(cl_to_err)? };
        Ok(MappedWriteGuard { buf, queue })
    }
}

impl<T> Deref for MappedWriteGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: see MappedReadGuard.
        unsafe { crate::util::mapped_slice(self.buf.ptr, self.buf.len) }
    }
}

impl<T> DerefMut for MappedWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: `&mut self` upgrades to a unique mutable slice;
        // mapped read+write for the guard's lifetime.
        unsafe { crate::util::mapped_slice_mut(self.buf.ptr, self.buf.len) }
    }
}

impl<T> Drop for MappedWriteGuard<'_, T> {
    fn drop(&mut self) {
        let unmap =
            unsafe { enqueue_svm_unmap(self.queue.raw(), self.buf.ptr.cast(), 0, ptr::null()) };
        if let Ok(evt) = unmap {
            let _ = unsafe { release_event(evt) };
        } else {
            self.buf.ctx.record_err();
        }
    }
}
