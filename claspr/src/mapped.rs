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
//! let mut buf = MappedSlice::<u32>::alloc_zero(&ctx, 1024)?;
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

use crate::access::{FillStrategy, Fillable, HostReadable, HostWritable, MemMode, ReadWrite};
use crate::buffer::Buffer;
use crate::context::{Context, SvmLevel};
use crate::error::{Error, Result};
use crate::launch::KernelArg;
use crate::map_primitive;
use crate::queue::Launcher;
use opencl3::event::{Event, retain_event};
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE, svm_alloc};
use opencl3::types::{CL_NON_BLOCKING, cl_event, cl_int, cl_uint};
use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
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
///   [`LaunchOp::into_event`][lo] calls [`KernelArg::register_completion`]
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
pub struct MappedSlice<T, M: MemMode = ReadWrite> {
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
    /// `Arc<MappedSlice<T>>` (e.g. through `.arc()` in device-op
    /// chains) and multiple threads may register concurrently.
    last_use: Mutex<Vec<Arc<Event>>>,
    /// Type-level access-mode tag. SVM at the OpenCL level only
    /// accepts kernel-side flags (`CL_MEM_READ_WRITE` /
    /// `CL_MEM_READ_ONLY` / `CL_MEM_WRITE_ONLY`); the host-side
    /// portion of [`MemMode`] is type-level only — SVM allocations
    /// are always host-RW at the runtime. The marker still gates
    /// which methods are callable from Rust, which gives users the
    /// same compile-time enforcement as `DeviceSlice<T, M>` even
    /// though the OpenCL runtime won't independently check.
    _mode: PhantomData<fn() -> M>,
}

// SAFETY: the SVM pointer is a runtime-owned allocation in
// host-accessible memory; OpenCL guarantees thread-safety for API
// calls on it (CL §3.4.1). Aliasing is governed by the map guards,
// which use the borrow checker to enforce exclusivity for `map_mut`.
unsafe impl<T: Send, M: MemMode> Send for MappedSlice<T, M> {}
unsafe impl<T: Sync, M: MemMode> Sync for MappedSlice<T, M> {}

impl<T: Default + Copy + Send + Sync + 'static, M: MemMode + Fillable + Send + 'static>
    MappedSlice<T, M>
{
    /// Allocate `len` elements of T in SVM memory, zero-initialised
    /// via `clEnqueueSVMMemFill(T::default())` on the context's
    /// default queue. Blocks until the fill completes.
    ///
    /// Returns [`Error::SvmNotAvailable`] if the context's device
    /// reports [`SvmLevel::None`] for `CL_DEVICE_SVM_CAPABILITIES`.
    ///
    /// The `M: Fillable` bound excludes [`crate::Frozen`]; markers
    /// that need initial data without runtime writability use
    /// [`from_slice`](Self::from_slice). Honest name: this is
    /// `alloc + zero-init via fill`. Matches
    /// [`DeviceSlice::alloc_zero`](crate::DeviceSlice::alloc_zero).
    pub fn alloc_zero(ctx: &Context, len: usize) -> Result<Self> {
        // SAFETY: synchronous fill below overwrites every byte
        // before returning, so no path can observe uninit data.
        let slice = unsafe { Self::alloc_uninit(ctx, len)?.assume_init() };
        // `fill` is now a graph node that consumes the buffer and rebinds it out.
        let slice = slice.fill(T::default()).wait()?;
        Ok(slice)
    }
}

impl<T, M: MemMode> MappedSlice<T, M> {
    /// Allocate `len` elements of T in SVM memory, leaving the bytes
    /// uninitialised. Returns a [`MappedSliceUninit<T, M>`] wrapper
    /// — host reads are type-blocked, transition to an initialised
    /// [`MappedSlice<T, M>`] via the wrapper's methods or
    /// [`unsafe fn assume_init`](MappedSliceUninit::assume_init).
    ///
    /// SVM analog of
    /// [`DeviceSlice::alloc_uninit`](crate::DeviceSlice::alloc_uninit).
    /// No marker bound — the type-state wrapper is the safety gate.
    /// Surfaces [`Error::SvmNotAvailable`] on devices without SVM.
    pub fn alloc_uninit(ctx: &Context, len: usize) -> Result<MappedSliceUninit<T, M>> {
        if ctx.svm_capability() == SvmLevel::None {
            return Err(Error::SvmNotAvailable);
        }
        let size = len.saturating_mul(std::mem::size_of::<T>());
        // SAFETY: M::KERNEL_FLAGS is one of the valid SVM-side flag
        // bits (READ_WRITE / READ_ONLY / WRITE_ONLY — never the
        // CL_MEM_HOST_* bits, which SVM doesn't accept; only the
        // kernel-side classification is forwarded to the runtime).
        // Alignment is the natural alignment of T.
        let raw = unsafe {
            svm_alloc(
                ctx.raw_context().get(),
                M::KERNEL_FLAGS,
                size,
                std::mem::align_of::<T>() as cl_uint,
            )
            .map_err(cl_to_err)?
        };
        Ok(MappedSliceUninit {
            inner: MappedSlice {
                ptr: raw.cast::<T>(),
                len,
                ctx: ctx.clone(),
                last_use: Mutex::new(Vec::new()),
                _mode: PhantomData,
            },
        })
    }
}

/// Type-state wrapper returned by [`MappedSlice::alloc_uninit`].
/// SVM analog of [`crate::DeviceSliceUninit`] — host reads are
/// blocked by the type system; transition via [`assume_init`](Self::assume_init).
pub struct MappedSliceUninit<T, M: MemMode = ReadWrite> {
    inner: MappedSlice<T, M>,
}

impl<T, M: MemMode> MappedSliceUninit<T, M> {
    /// Skip safe initialization. See
    /// [`crate::DeviceSliceUninit::assume_init`] for the safety
    /// contract.
    ///
    /// # Safety
    ///
    /// Every byte must be written by SOME path before any read.
    pub unsafe fn assume_init(self) -> MappedSlice<T, M> {
        self.inner
    }

    /// Re-wrap an already-initialised [`MappedSlice`] back into the uninit
    /// type-state — the sound downgrade used by the reusable-graph home channel
    /// (see [`crate::DeviceSliceUninit::from_init`]). Safe private-field re-wrap,
    /// the inverse of [`assume_init`](Self::assume_init).
    pub(crate) fn from_init(inner: MappedSlice<T, M>) -> Self {
        MappedSliceUninit { inner }
    }

    pub fn len(&self) -> usize {
        self.inner.len
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    pub fn ctx(&self) -> &Context {
        &self.inner.ctx
    }
}

impl<T, M: MemMode> crate::record::RecordableBuffer for MappedSliceUninit<T, M> {
    fn record_handle(&self) -> crate::record::BufHandle {
        self.inner.record_handle()
    }
}

impl<T, M: MemMode> fmt::Debug for MappedSliceUninit<T, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedSliceUninit")
            .field("len", &self.inner.len)
            .field("element_size", &std::mem::size_of::<T>())
            .finish_non_exhaustive()
    }
}

// ── from_slice / from_vec — bake in initial data via SVM map ───────
//
// Same role as DeviceSlice::from_slice but the underlying mechanism
// differs: clSVMAlloc doesn't accept CL_MEM_COPY_HOST_PTR, so we
// alloc + map + memcpy + unmap. The CopyOp flavoured fill path would
// also work but pays an extra round trip; the map path is one CL call.

impl<T: Copy, M: MemMode> MappedSlice<T, M> {
    /// Create an SVM buffer whose contents are copied from `data` at
    /// construction time. Errors with
    /// [`Error::SvmNotAvailable`] if
    /// the device doesn't support SVM.
    ///
    /// Works for any marker — for kernel-RO markers like
    /// [`crate::ReadOnly`] / [`crate::Frozen`] this is the ONLY
    /// constructor (no alloc+fill path because fill needs kernel-write
    /// access). For other markers it's a convenience over `alloc +
    /// map + memcpy`.
    pub fn from_slice(ctx: &Context, data: &[T]) -> Result<Self> {
        if ctx.svm_capability() == SvmLevel::None {
            return Err(Error::SvmNotAvailable);
        }
        let size = data.len().saturating_mul(std::mem::size_of::<T>());
        // SAFETY: M::KERNEL_FLAGS is one of the valid SVM-side flag
        // bits. Alignment is the natural alignment of T.
        let raw = unsafe {
            svm_alloc(
                ctx.raw_context().get(),
                M::KERNEL_FLAGS,
                size,
                std::mem::align_of::<T>() as cl_uint,
            )
            .map_err(cl_to_err)?
        };
        // Map for write, memcpy, unmap. Blocking is fine for
        // construction.
        // SAFETY: raw is a fresh, live SVM allocation in ctx; the
        // queue we use is ctx's default queue; the size matches what
        // we just allocated.
        let queue = ctx.cl_queue();
        unsafe {
            let _map_evt = map_primitive::svm_map(queue.get(), true, CL_MAP_WRITE, raw, size, &[])?;
            // memcpy from host to the now-host-accessible SVM region.
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, raw as *mut u8, size);
            let _unmap_evt = map_primitive::svm_unmap(queue.get(), raw, &[])?;
        }
        Ok(MappedSlice {
            ptr: raw.cast::<T>(),
            len: data.len(),
            ctx: ctx.clone(),
            last_use: Mutex::new(Vec::new()),
            _mode: PhantomData,
        })
    }

    /// Take a Vec by value — wrapper over [`from_slice`](Self::from_slice).
    pub fn from_vec(ctx: &Context, data: Vec<T>) -> Result<Self> {
        Self::from_slice(ctx, &data)
    }
}

// ── MappedScalar — the SVM-backed device scalar ─────────────────────

/// A [`Scalar`](crate::Scalar) backed by a length-1 [`MappedSlice<T, M>`]
/// (coarse-grain SVM). The SVM tier's device-resident scalar, symmetric
/// with [`DeviceScalar`](crate::DeviceScalar) — binds ONLY to scalar-ref
/// kernel params and delegates the SVM allocation + in-flight-free
/// bookkeeping to the backing `MappedSlice`.
pub type MappedScalar<T, M = ReadWrite> = crate::Scalar<MappedSlice<T, M>>;

/// The uninit type-state for a [`MappedScalar<T, M>`].
pub type MappedScalarUninit<T, M = ReadWrite> = crate::ScalarUninit<MappedSliceUninit<T, M>>;

impl<T: Copy, M: MemMode> crate::Scalar<MappedSlice<T, M>> {
    /// Create an SVM device scalar seeded with `value` (length-1
    /// [`MappedSlice::from_slice`]). Errors with
    /// [`Error::SvmNotAvailable`] on a no-SVM device.
    pub fn new_mapped(ctx: &Context, value: T) -> Result<Self> {
        Ok(crate::Scalar {
            inner: MappedSlice::from_slice(ctx, std::slice::from_ref(&value))?,
        })
    }
}

impl<T, M: MemMode> crate::Scalar<MappedSlice<T, M>> {
    /// Allocate an uninitialised SVM device scalar, returning the
    /// type-state-gated [`MappedScalarUninit`]. Mirrors
    /// [`MappedSlice::alloc_uninit`].
    pub fn uninit_mapped(ctx: &Context) -> Result<MappedScalarUninit<T, M>> {
        Ok(crate::ScalarUninit {
            inner: MappedSlice::alloc_uninit(ctx, 1)?,
        })
    }

    /// Borrow the backing length-1 [`MappedSlice<T, M>`].
    pub fn as_mapped_slice(&self) -> &MappedSlice<T, M> {
        &self.inner
    }
}

impl<T, M: MemMode> crate::ScalarUninit<MappedSliceUninit<T, M>> {
    /// Assert the scalar has been (or will be) written before any read.
    ///
    /// # Safety
    ///
    /// Same contract as [`MappedSliceUninit::assume_init`].
    pub unsafe fn assume_init_mapped(self) -> MappedScalar<T, M> {
        // SAFETY: forwarded to the caller (see this fn's Safety section).
        crate::Scalar {
            inner: unsafe { self.inner.assume_init() },
        }
    }
}

/// Free-fn ctor for a seeded [`MappedScalar<T>`] (default marker).
pub fn mapped_scalar<T: Copy>(ctx: &Context, value: T) -> Result<MappedScalar<T, ReadWrite>> {
    MappedScalar::new_mapped(ctx, value)
}

/// Free-fn ctor for an uninitialised [`MappedScalar<T>`] (default marker).
pub fn mapped_scalar_uninit<T>(ctx: &Context) -> Result<MappedScalarUninit<T, ReadWrite>> {
    MappedScalar::<T, ReadWrite>::uninit_mapped(ctx)
}

impl<T, M: MemMode> MappedSlice<T, M> {
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

    /// Begin a host read map of this buffer. Returns a lazy [`MapOp`]
    /// builder — call [`wait`](MapOp::wait) on it with a launcher to
    /// actually issue the `clEnqueueSVMMap(CL_BLOCKING, CL_MAP_READ)`
    /// and receive a RAII [`MappedReadGuard`] that derefs to `&[T]`
    /// and unmaps on Drop.
    ///
    /// Mirrors the late-bind pattern of every other claspr op
    /// (`buf.write(&data).wait(&ctx)?`, `kernels.foo([N], buf).wait(&ctx)?`)
    /// — the launcher arrives at the terminal, not at construction.
    /// Non-blocking variant (`.submit(&launcher)`) is a deferred
    /// follow-up (see `claspr-scope-launcher-followup` memory).
    pub fn map(&self) -> MapOp<'_, T, M>
    where
        M: HostReadable,
    {
        MapOp { owner: self }
    }

    /// Begin a host read+write map of this buffer. Returns a lazy
    /// [`MapMutOp`] builder — call [`wait`](MapMutOp::wait) on it
    /// with a launcher to issue the
    /// `clEnqueueSVMMap(CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE)` and
    /// receive a RAII [`MappedWriteGuard`] that derefs to `&mut [T]`
    /// and unmaps on Drop. The `&mut self` receiver gives the borrow
    /// checker the exclusivity guarantee needed for `DerefMut`.
    pub fn map_mut(&mut self) -> MapMutOp<'_, T, M>
    where
        M: HostWritable + HostReadable,
    {
        MapMutOp { owner: self }
    }

    /// Raw SVM pointer for direct use (e.g. passing to a kernel arg
    /// out-of-band, or interoperating with hand-written OpenCL).
    pub fn ptr(&self) -> *mut T {
        self.ptr
    }

    /// Fill this SVM buffer's contents with `value` repeated for every
    /// element (wraps `clEnqueueSVMMemFill`, CL 2.0+, or a built-in
    /// fill kernel for kernel-RO markers). SVM analog of
    /// [`crate::DeviceSlice::fill`].
    ///
    /// Returns the [`FillMapped`](crate::eager::FillMapped) graph node
    /// (a [`DeviceOp`](crate::DeviceOp)). Usable standalone — `let buf =
    /// buf.fill(v).wait()?;` (the buffer moves in and rebinds out so it
    /// can be reused) — or composed in a graph via `.and_then(...)`.
    ///
    /// The fill event is auto-registered on the buffer's last-use list
    /// at execute, so Drop's `clEnqueueSVMFree` waits for the fill.
    ///
    /// **Marker constraint:** `M: Fillable`. Runtime-side fill counts
    /// as a write at the OpenCL level, so kernel-RO markers
    /// (`ReadOnly`, `Frozen`) can't be filled.
    pub fn fill(self, value: T) -> crate::eager::FillMapped<T, M>
    where
        T: Copy + Send + Sync + 'static,
        M: Fillable + Send + 'static,
    {
        crate::eager::fill_mapped(self, value)
    }

    /// SVM→SVM copy from `self` into `dst` — wraps `clEnqueueSVMMemcpy`
    /// (CL 2.0+). SVM analog of [`crate::DeviceSlice::copy_to`].
    ///
    /// Returns the [`CopyTo2`](crate::eager::CopyTo2) graph node (a
    /// [`DeviceOp`](crate::DeviceOp)) whose output is `(src, dst)`.
    /// Usable standalone via `.sync(&ctx)` / `.wait_on(&queue)` or
    /// composed in a graph. Polymorphic over the SVM `(src, dst)`
    /// families (`MappedSlice`/`USMSlice`, init / uninit dst) via the
    /// [`CopyTo`](crate::CopyTo) trait.
    ///
    /// Both buffers must be on the same `Context` (the runtime
    /// enforces this; mismatch surfaces as a CL error at terminal
    /// time). The copy event is registered on **both** buffers'
    /// last-use lists so Drop on either side is ordered after it.
    pub fn copy_to<Dst>(self, dst: Dst) -> crate::eager::CopyTo2<Self, Dst>
    where
        Self: crate::CopyTo<Dst>,
        <<Self as crate::CopyTo<Dst>>::Op as crate::eager::DeviceEnqueue>::Output:
            crate::eager::CopyOutputs,
        Dst: crate::eager::CopyOperand<Dst>,
    {
        crate::eager::eager_copy_to(self, dst)
    }

    /// Write host `data` into this buffer — wraps `clEnqueueSVMMemcpy`
    /// (CL 2.0+) with a host pointer as source. SVM analog of
    /// [`crate::DeviceSlice::write`].
    ///
    /// Returns the [`WriteMapped`](crate::eager::WriteMapped) graph
    /// node (a [`DeviceOp`](crate::DeviceOp)). Usable standalone — `let
    /// buf = buf.write(data).wait()?;` (buffer rebinds out for reuse) —
    /// or composed in a graph.
    ///
    /// The write stays NON-BLOCKING (`CL_NON_BLOCKING` enqueue); the
    /// terminal waits. For the non-blocking terminal the host `src` is
    /// kept alive until the write event fires via
    /// `register_drop_callback`. The write event is auto-registered on
    /// the buffer's last-use list so Drop's `clEnqueueSVMFree` waits.
    ///
    /// **Marker constraint:** `M: HostWritable`. Excludes
    /// [`crate::HostReadOnly`] and [`crate::Frozen`] — post-creation
    /// host writes break the contract those markers advertise.
    pub fn write<S>(self, src: S) -> crate::eager::WriteMapped<T, M>
    where
        T: Send + Sync + 'static,
        M: HostWritable + Send + 'static,
        S: Into<crate::transfer::UploadSource<T>>,
    {
        crate::eager::write_mapped(self, src)
    }

    /// Blocking, **borrowing** host→SVM upload — the synchronous
    /// counterpart to the async owned [`write`](Self::write). SVM analog of
    /// [`crate::DeviceSlice::write_sync`].
    ///
    /// Borrows `data` as `&[T]`, enqueues a `clEnqueueSVMMemcpy` from the
    /// host source pointer, and waits on the copy event **inline** before
    /// returning. (SVM has no native `CL_BLOCKING` flag, so this is a
    /// non-blocking enqueue followed by an immediate `event.wait()` — the
    /// observable effect is identical: the copy is done when the call
    /// returns.) Because the wait happens here, the borrowed source only
    /// needs to live across the call — **no ownership transfer and no
    /// keep-alive allocation**. Both this buffer and `data` stay usable
    /// afterwards (`buf.write_sync(&data)?;` instead of
    /// `buf.write(data.clone())`).
    ///
    /// Tradeoffs vs [`write`](Self::write):
    /// - **Blocks the calling thread** until the SVM copy finishes — no
    ///   overlap with other device work. For pipelined uploads (the
    ///   non-blocking enqueue whose owned source is kept alive via a
    ///   drop-callback), use [`write`](Self::write).
    /// - **Not a graph node** — returns a plain `Result<()>`, not a
    ///   [`DeviceOp`](crate::DeviceOp), so it can't be `.and_then(...)`-ed
    ///   or `bundle!`-d.
    ///
    /// `data.len()` must equal `self.len()` (returns
    /// [`Error::LengthMismatch`] otherwise). The copy event is
    /// auto-registered on the buffer's last-use list (inside the raw
    /// helper) so Drop's `clEnqueueSVMFree` queue-orders after it.
    ///
    /// **Marker constraint:** `M: HostWritable` — identical to
    /// [`write`](Self::write). Excludes [`crate::HostReadOnly`] and
    /// [`crate::Frozen`].
    pub fn write_sync(&mut self, data: &[T]) -> Result<()>
    where
        M: HostWritable,
    {
        let ctx = self.ctx.clone();
        // Non-blocking SVM enqueue + inline wait: `svm_write_enqueue` has no
        // blocking flag (SVM lacks a native one), so we wait on the returned
        // event here. The borrowed `data` lives across the wait, so — like the
        // DeviceSlice blocking write — no keep-alive / ownership transfer is
        // needed. Reuses the same raw helper the eager `WriteMapped` op uses.
        let event = svm_write_enqueue(self, &ctx, data, &[])?;
        event.wait().map_err(Error::OpenCl)?;
        Ok(())
    }
}

/// Metadata-only `Debug` — does not read through the SVM pointer
/// (would race with in-flight kernel work and require holding a map
/// guard) and doesn't require `T: Debug`.
impl<T, M: MemMode> fmt::Debug for MappedSlice<T, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedSlice")
            .field("len", &self.len)
            .field("element_size", &std::mem::size_of::<T>())
            .finish_non_exhaustive()
    }
}

impl<T, M: MemMode> Buffer<T> for MappedSlice<T, M> {
    fn len(&self) -> usize {
        self.len
    }

    fn ctx(&self) -> &Context {
        &self.ctx
    }
}

// ── Raw SVM enqueue helpers — the fold seam for the eager SVM ops ────
//
// Each helper is the `clEnqueueSVM*` body the matching Tier-1 builder
// used to own, lifted out so the eager graph nodes (`FillMapped` /
// `WriteMapped` in `eager.rs`, plus the `CopyTo` family in `copy.rs`)
// can enqueue directly against a `MappedSlice` without round-tripping
// through a borrow-based builder. Each does the enqueue, retains the
// event, and registers it on the buffer(s)' last-use list so Drop's
// `clEnqueueSVMFree` queue-orders after every recorded use. `deps` is
// the already-collected `cl_event` wait-list (the eager op flattens
// its `Deps` to raw handles, held alive across the call). All are
// NON-BLOCKING enqueues — the eager op's `Blocking` terminal waits on
// the returned event, exactly as the builder's `wait_on` did
// (`submit + event.wait()`). SVM has no native `CL_BLOCKING` flag for
// fill/memcpy the way `clEnqueueWriteBuffer` does.

/// Retain `event` once and register it on `owner`'s last-use list, so
/// Drop's `clEnqueueSVMFree` waits for it. `clRetainEvent` bumps the
/// `cl_event` refcount: the returned `event` and the registered
/// `Arc<Event>` each hold an independent reference, balanced by their
/// respective `Event::drop` → `clReleaseEvent`.
fn register_event_on<T, M: MemMode>(owner: &MappedSlice<T, M>, event: &Event) -> Result<()> {
    // SAFETY: event.get() is live; retain pairs with the Event::drop
    // inside the Arc held by last_use.
    unsafe {
        retain_event(event.get())
            .map_err(|code| Error::OpenCl(opencl3::error_codes::ClError(code)))?;
    }
    owner.register_use(std::sync::Arc::new(Event::new(event.get())));
    Ok(())
}

/// Raw `clEnqueueSVMMemFill` (or kernel fill) over `owner` — body of
/// the former `SvmFillOp::into_event`. Non-blocking; auto-registers
/// the fill event on `owner`'s last-use list.
pub(crate) fn svm_fill_enqueue<T, M, L>(
    owner: &MappedSlice<T, M>,
    launcher: &L,
    pattern: T,
    deps: &[cl_event],
) -> Result<Event>
where
    T: Copy,
    M: MemMode + Fillable,
    L: Launcher + ?Sized,
{
    let event = match M::FILL_STRATEGY {
        FillStrategy::Runtime => {
            let size = owner.len * std::mem::size_of::<T>();
            // SAFETY: owner.ptr is a valid SVM allocation in the queue's
            // context (caller's responsibility — same as
            // `DeviceSlice::fill`). Pattern is a single T byte-copied
            // across the buffer.
            unsafe {
                launcher.cl_queue().enqueue_svm_mem_fill(
                    owner.ptr as *mut c_void,
                    std::slice::from_ref(&pattern),
                    size,
                    deps,
                )?
            }
        }
        FillStrategy::DeviceKernel => fill_via_kernel_svm(
            &owner.ctx,
            launcher,
            owner.ptr as *mut c_void,
            &pattern,
            owner.len,
            deps,
        )?,
    };
    register_event_on(owner, &event)?;
    Ok(event)
}

/// Launch the built-in fill kernel for an SVM allocation
/// (MappedSlice backing memory). Mirror of
/// [`crate::buffer::fill_via_kernel_buffer`] but uses
/// `set_arg_svm_pointer` for the buffer arg since SVM pointers
/// can't go through the regular `set_arg` path. Returns the
/// launch event.
pub(crate) fn fill_via_kernel_svm<T: Copy, L: Launcher + ?Sized>(
    ctx: &Context,
    launcher: &L,
    svm_ptr: *mut c_void,
    pattern: &T,
    count: usize,
    deps: &[cl_event],
) -> Result<Event> {
    use opencl3::kernel::{ExecuteKernel, Kernel};
    let pattern_size = std::mem::size_of::<T>();
    let count_u32 =
        u32::try_from(count).map_err(|_| Error::InvalidArgument("fill count exceeds u32::MAX"))?;
    let program = ctx.fill_program()?;

    if let Some(name) = crate::fill_kernel::fast_path_kernel_name(pattern_size) {
        let kernel = Kernel::create(program, name)?;
        let mut exec = ExecuteKernel::new(&kernel);
        // SAFETY: arg 0 = SVM pointer (matches the kernel's
        // `__global X*` arg). arg 1 = pattern by value. arg 2 =
        // element count.
        unsafe {
            exec.set_arg_svm(svm_ptr);
            exec.set_arg(pattern);
            exec.set_arg(&count_u32);
            exec.set_global_work_size(count);
            exec.set_event_wait_list(deps);
            Ok(exec.enqueue_nd_range(launcher.cl_queue())?)
        }
    } else {
        // Byte-generic path: pattern as a small read-only buffer,
        // memcpy bytes in via blocking write, then launch
        // claspr_fill_bytes with the SVM data pointer + pattern
        // buffer.
        let pattern_size_u32 = u32::try_from(pattern_size)
            .map_err(|_| Error::InvalidArgument("fill pattern size exceeds u32::MAX"))?;
        // SAFETY: pattern is a live &T; read `pattern_size` bytes.
        let pattern_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pattern as *const T as *const u8, pattern_size) };
        use opencl3::memory::{Buffer as ClBuffer, CL_MEM_READ_ONLY};
        use opencl3::types::CL_BLOCKING;
        let mut pattern_buf = unsafe {
            ClBuffer::<u8>::create(
                ctx.raw_context(),
                CL_MEM_READ_ONLY,
                pattern_size,
                std::ptr::null_mut(),
            )?
        };
        let _write_evt = unsafe {
            launcher.cl_queue().enqueue_write_buffer(
                &mut pattern_buf,
                CL_BLOCKING,
                0,
                pattern_bytes,
                &[],
            )?
        };
        let kernel = Kernel::create(program, crate::fill_kernel::KERNEL_BYTES)?;
        let mut exec = ExecuteKernel::new(&kernel);
        // SAFETY: arg 0 = SVM pointer (data), arg 1 = pattern
        // buffer, arg 2 = pattern byte count, arg 3 = slot count.
        let event = unsafe {
            exec.set_arg_svm(svm_ptr);
            exec.set_arg(&pattern_buf);
            exec.set_arg(&pattern_size_u32);
            exec.set_arg(&count_u32);
            exec.set_global_work_size(count);
            exec.set_event_wait_list(deps);
            exec.enqueue_nd_range(launcher.cl_queue())?
        };
        Ok(event)
    }
}

/// Raw `clEnqueueSVMMemcpy` with a host source pointer over `owner` —
/// body of the former `SvmWriteOp::into_event`. Non-blocking
/// (`CL_NON_BLOCKING`); the caller (eager `WriteMapped`) keeps `data`
/// alive until the event fires. Auto-registers the write event on
/// `owner`'s last-use list.
pub(crate) fn svm_write_enqueue<T, M, L>(
    owner: &MappedSlice<T, M>,
    launcher: &L,
    data: &[T],
    deps: &[cl_event],
) -> Result<Event>
where
    M: MemMode,
    L: Launcher + ?Sized,
{
    if data.len() != owner.len {
        return Err(Error::LengthMismatch {
            src: data.len(),
            dst: owner.len,
        });
    }
    let size = owner.len * std::mem::size_of::<T>();
    // SAFETY: SVM ptr is a valid allocation in the queue's context
    // (caller's responsibility, same as DeviceSlice::write).
    // data.as_ptr() points to host memory the eager op keeps alive past
    // the event via register_drop_callback (CL_NON_BLOCKING contract).
    let event = unsafe {
        launcher.cl_queue().enqueue_svm_mem_cpy(
            CL_NON_BLOCKING,
            owner.ptr as *mut c_void,
            data.as_ptr() as *const c_void,
            size,
            deps,
        )?
    };
    register_event_on(owner, &event)?;
    Ok(event)
}

/// Raw `clEnqueueSVMMemcpy` SVM→SVM from `src` into `dst` — body of the
/// former `SvmCopyOp::into_event`. Non-blocking; auto-registers the
/// copy event on **both** buffers' last-use lists so Drop on either
/// side queue-orders after it. Used by the `CopyTo` family (`copy.rs`)
/// for the same-type `MappedSlice → MappedSlice` pair.
pub(crate) fn svm_copy_enqueue<T, M1, M2, L>(
    src: &MappedSlice<T, M1>,
    dst: &MappedSlice<T, M2>,
    launcher: &L,
    deps: &[cl_event],
) -> Result<Event>
where
    M1: MemMode,
    M2: MemMode,
    L: Launcher + ?Sized,
{
    if src.len != dst.len {
        return Err(Error::LengthMismatch {
            src: src.len,
            dst: dst.len,
        });
    }
    let size = src.len * std::mem::size_of::<T>();
    // SAFETY: both SVM pointers are valid allocations in the queue's
    // context (caller's responsibility — runtime gives
    // CL_INVALID_CONTEXT on mismatch). CL_NON_BLOCKING so the event
    // encodes completion; the eager terminal picks how to observe it.
    let event = unsafe {
        launcher.cl_queue().enqueue_svm_mem_cpy(
            CL_NON_BLOCKING,
            dst.ptr as *mut c_void,
            src.ptr as *const c_void,
            size,
            deps,
        )?
    };
    register_event_on(src, &event)?;
    register_event_on(dst, &event)?;
    Ok(event)
}

impl<T, M: MemMode> Drop for MappedSlice<T, M> {
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

impl<T, M: MemMode> crate::record::RecordableBuffer for MappedSlice<T, M> {
    fn record_handle(&self) -> crate::record::BufHandle {
        crate::record::BufHandle {
            mem: crate::record::MemRef::Svm(self.ptr as *mut std::ffi::c_void),
            byte_len: self.len * std::mem::size_of::<T>(),
        }
    }
}

impl<T, M: MemMode> KernelArg for MappedSlice<T, M> {
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

// ── Map builders (Op shape, late-bind launcher) ────────────────────
//
// Construction is `buf.map()` / `buf.map_mut()` — no launcher yet,
// just borrows `buf`. Two terminals (mirroring every other Tier 1 op):
//
//   .wait(&launcher)?    blocking — enqueue with CL_TRUE, return guard
//   .submit(&launcher)?  non-blocking — enqueue with CL_FALSE, return
//                        a `MappedReadPending` / `MappedWritePending`
//                        carrying the map event; consume via `.wait()`
//                        on the pending to get the guard once the map
//                        completes (or `.event()` for chain ordering
//                        before then).
//
// Drop on the guards enqueues `clEnqueueSVMUnmap` AND registers the
// unmap event on the buffer's `last_use` list — so `clEnqueueSVMFree`
// in `MappedSlice::Drop` waits on it even when the map/unmap queue
// is not the context's default in-order queue (closes a latent
// cross-queue race that the old blocking-only path had too).

/// Lazy builder for [`MappedSlice::map`]. Borrows the source buffer;
/// pick a terminal — [`wait`](MapOp::wait) (blocking) or
/// [`submit`](MapOp::submit) (non-blocking).
pub struct MapOp<'a, T, M: MemMode> {
    owner: &'a MappedSlice<T, M>,
}

impl<'a, T, M: MemMode + HostReadable> MapOp<'a, T, M> {
    /// Blocking terminal on the owning buffer's context default queue.
    pub fn wait(self) -> Result<MappedReadGuard<'a, T, M>> {
        let ctx = self.owner.ctx.clone();
        self.wait_on(&ctx)
    }

    /// Blocking terminal with an explicit launcher. Enqueues
    /// `clEnqueueSVMMap(CL_TRUE, CL_MAP_READ)`; returns a RAII guard
    /// that derefs to `&[T]` and unmaps on Drop.
    pub fn wait_on<L: Launcher>(self, launcher: &L) -> Result<MappedReadGuard<'a, T, M>> {
        let (guard, event) = MappedReadGuard::enqueue_map(self.owner, launcher, true)?;
        // Blocking map already complete; event has nothing to wait on,
        // drop it (the cl_event refcount releases here).
        drop(event);
        Ok(guard)
    }

    /// Non-blocking terminal on the owning buffer's context default queue.
    pub fn submit(self) -> Result<MappedReadPending<'a, T, M>> {
        let ctx = self.owner.ctx.clone();
        self.submit_on(&ctx)
    }

    /// Non-blocking terminal with an explicit launcher. Enqueues
    /// `clEnqueueSVMMap(CL_FALSE, CL_MAP_READ)` and returns a
    /// [`MappedReadPending`] carrying the map event. Consume via
    /// [`MappedReadPending::wait`] to get the guard; use
    /// [`MappedReadPending::event`] to thread the map event into
    /// cross-queue chain ordering before then.
    pub fn submit_on<L: Launcher>(self, launcher: &L) -> Result<MappedReadPending<'a, T, M>> {
        let (guard, event) = MappedReadGuard::enqueue_map(self.owner, launcher, false)?;
        Ok(crate::map_primitive::MapPending::new(guard, event))
    }
}

/// Lazy builder for [`MappedSlice::map_mut`]. Borrows the source
/// buffer mutably; pick a terminal — [`wait`](MapMutOp::wait)
/// (blocking) or [`submit`](MapMutOp::submit) (non-blocking).
pub struct MapMutOp<'a, T, M: MemMode> {
    owner: &'a mut MappedSlice<T, M>,
}

impl<'a, T, M: MemMode + HostWritable + HostReadable> MapMutOp<'a, T, M> {
    /// Blocking terminal on the owning buffer's context default queue.
    pub fn wait(self) -> Result<MappedWriteGuard<'a, T, M>> {
        let ctx = self.owner.ctx.clone();
        self.wait_on(&ctx)
    }

    /// Blocking terminal with an explicit launcher. Enqueues
    /// `clEnqueueSVMMap(CL_TRUE, CL_MAP_READ | CL_MAP_WRITE)` and
    /// returns a RAII guard that derefs to `&mut [T]` and unmaps on Drop.
    pub fn wait_on<L: Launcher>(self, launcher: &L) -> Result<MappedWriteGuard<'a, T, M>> {
        let (guard, event) = MappedWriteGuard::enqueue_map(self.owner, launcher, true)?;
        drop(event);
        Ok(guard)
    }

    /// Non-blocking terminal on the owning buffer's context default queue.
    pub fn submit(self) -> Result<MappedWritePending<'a, T, M>> {
        let ctx = self.owner.ctx.clone();
        self.submit_on(&ctx)
    }

    /// Non-blocking terminal with an explicit launcher. Enqueues
    /// `clEnqueueSVMMap(CL_FALSE, CL_MAP_READ | CL_MAP_WRITE)` and
    /// returns a [`MappedWritePending`] carrying the map event.
    pub fn submit_on<L: Launcher>(self, launcher: &L) -> Result<MappedWritePending<'a, T, M>> {
        let (guard, event) = MappedWriteGuard::enqueue_map(self.owner, launcher, false)?;
        Ok(crate::map_primitive::MapPending::new(guard, event))
    }
}

// ── Map pendings (non-blocking submit results) ─────────────────────

/// Result of [`MapOp::submit`] — a non-blocking SVM read map in flight. The map
/// enqueue has returned; the bytes are NOT spec-valid for host reads until the map
/// event completes. A [`MapPending`](crate::map_primitive::MapPending) over a
/// [`MappedReadGuard`]: [`wait`](crate::map_primitive::MapPending::wait) blocks on the
/// event and yields the guard; [`event`](crate::map_primitive::MapPending::event)
/// borrows the map event for cross-queue chaining first.
///
/// (No explicit `Drop`: if the pending is dropped without `wait`, the guard inside
/// `MapPending`'s `Option` drops — enqueuing the unmap on the same in-order queue, its
/// event registered on `last_use` so SVMFree waits on it.)
pub type MappedReadPending<'a, T, M> = crate::map_primitive::MapPending<MappedReadGuard<'a, T, M>>;

/// Result of [`MapMutOp::submit`] — the read/write twin of [`MappedReadPending`],
/// yielding a [`MappedWriteGuard`] (`DerefMut` to `&mut [T]`).
pub type MappedWritePending<'a, T, M> =
    crate::map_primitive::MapPending<MappedWriteGuard<'a, T, M>>;

// ── Map guards ──────────────────────────────────────────────────────

/// RAII guard for a SVM read map. Drop issues `clEnqueueSVMUnmap`
/// and registers the unmap event on the buffer's `last_use` so
/// [`MappedSlice`]'s `Drop` impl's `clEnqueueSVMFree` waits on it.
pub struct MappedReadGuard<'a, T, M: MemMode> {
    buf: &'a MappedSlice<T, M>,
    queue: crate::util::RetainedQueue,
}

impl<'a, T, M: MemMode> MappedReadGuard<'a, T, M> {
    /// Internal: enqueue the SVM map (blocking or not), return the
    /// guard + the map event. The map event's host-meaning differs
    /// per blocking flag — blocking callers drop it; non-blocking
    /// callers thread it through a `MappedReadPending`.
    fn enqueue_map<L: Launcher>(
        buf: &'a MappedSlice<T, M>,
        launcher: &L,
        blocking: bool,
    ) -> Result<(Self, Event)> {
        let queue = crate::util::RetainedQueue::from_queue(launcher.cl_queue())?;
        // SAFETY: queue is alive (RetainedQueue); buf.ptr is the live
        // SVM allocation; map_size matches the allocation's byte length.
        let event = unsafe {
            map_primitive::svm_map(
                queue.raw(),
                blocking,
                CL_MAP_READ,
                buf.ptr.cast(),
                buf.len * std::mem::size_of::<T>(),
                &[],
            )?
        };
        Ok((MappedReadGuard { buf, queue }, event))
    }
}

impl<T, M: MemMode> Deref for MappedReadGuard<'_, T, M> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: the SVM pointer is valid + mapped for read for
        // this guard's lifetime.
        unsafe { crate::util::mapped_slice(self.buf.ptr, self.buf.len) }
    }
}

impl<T, M: MemMode> Drop for MappedReadGuard<'_, T, M> {
    fn drop(&mut self) {
        // SAFETY: ptr was mapped in `enqueue_map`; unmap exactly once.
        // The unmap event is registered on the buffer's last_use so
        // MappedSlice::Drop's clEnqueueSVMFree waits on it — closes
        // the cross-queue Drop race (map/unmap on launcher queue,
        // SVMFree on ctx default queue).
        match unsafe { map_primitive::svm_unmap(self.queue.raw(), self.buf.ptr.cast(), &[]) } {
            Ok(evt) => self.buf.register_use(Arc::new(evt)),
            Err(_) => self.buf.ctx.record_err(),
        }
        // `queue: RetainedQueue` drops after this body returns,
        // releasing the queue handle.
    }
}

/// RAII guard for a SVM write map. Drop issues `clEnqueueSVMUnmap`
/// and registers the unmap event on the buffer's `last_use` — same
/// shape as [`MappedReadGuard`].
pub struct MappedWriteGuard<'a, T, M: MemMode> {
    buf: &'a mut MappedSlice<T, M>,
    queue: crate::util::RetainedQueue,
}

impl<'a, T, M: MemMode> MappedWriteGuard<'a, T, M> {
    fn enqueue_map<L: Launcher>(
        buf: &'a mut MappedSlice<T, M>,
        launcher: &L,
        blocking: bool,
    ) -> Result<(Self, Event)> {
        let queue = crate::util::RetainedQueue::from_queue(launcher.cl_queue())?;
        let ptr = buf.ptr;
        let len = buf.len;
        // SAFETY: see MappedReadGuard::enqueue_map.
        let event = unsafe {
            map_primitive::svm_map(
                queue.raw(),
                blocking,
                CL_MAP_READ | CL_MAP_WRITE,
                ptr.cast(),
                len * std::mem::size_of::<T>(),
                &[],
            )?
        };
        Ok((MappedWriteGuard { buf, queue }, event))
    }
}

impl<T, M: MemMode> Deref for MappedWriteGuard<'_, T, M> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: see MappedReadGuard.
        unsafe { crate::util::mapped_slice(self.buf.ptr, self.buf.len) }
    }
}

impl<T, M: MemMode> DerefMut for MappedWriteGuard<'_, T, M> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: `&mut self` upgrades to a unique mutable slice;
        // mapped read+write for the guard's lifetime.
        unsafe { crate::util::mapped_slice_mut(self.buf.ptr, self.buf.len) }
    }
}

impl<T, M: MemMode> Drop for MappedWriteGuard<'_, T, M> {
    fn drop(&mut self) {
        // See MappedReadGuard::drop for the last_use registration
        // rationale.
        match unsafe { map_primitive::svm_unmap(self.queue.raw(), self.buf.ptr.cast(), &[]) } {
            Ok(evt) => self.buf.register_use(Arc::new(evt)),
            Err(_) => self.buf.ctx.record_err(),
        }
    }
}
