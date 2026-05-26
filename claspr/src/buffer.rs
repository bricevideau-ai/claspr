//! Typed device-side buffers and the [`Buffer`] trait that abstracts
//! over them.
//!
//! Two tiers live in this module:
//!
//! | Type | Backing | Host access | Use case |
//! |------|---------|-------------|----------|
//! | [`DeviceSlice<T>`] | `CL_MEM_READ_WRITE` | via [`upload`](DeviceSlice::upload) / [`download`](DeviceSlice::download) | classic device-side buffer, opaque host pointer |
//! | [`HostBuffer<T>`] | `CL_MEM_ALLOC_HOST_PTR` + persistent map | direct via `Deref<Target=[T]>` + `DerefMut` | pinned, runtime-chosen-host-accessible memory; zero-copy where the device supports it |
//!
//! The third tier (SVM / [`SharedBuffer`](crate::svm::SharedBuffer))
//! lives in [`crate::svm`].
//!
//! See the [`Buffer`] trait's own docs for what it does and does
//! not abstract over.

use crate::context::Context;
use crate::error::{Error, Result};
use crate::queue::Launcher;
use opencl3::command_queue::{
    enqueue_map_buffer, enqueue_unmap_mem_object, release_command_queue, retain_command_queue,
};
use opencl3::memory::{
    Buffer as ClBuffer, CL_MAP_READ, CL_MAP_WRITE, CL_MEM_ALLOC_HOST_PTR, CL_MEM_READ_WRITE, ClMem,
    release_mem_object,
};
use opencl3::types::{CL_BLOCKING, cl_command_queue, cl_int, cl_mem};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::slice;

/// Raw cl3 functions return `Result<_, cl_int>`. Wrap into our
/// typed `Error` via opencl3's `ClError` newtype.
fn cl_to_err(code: cl_int) -> Error {
    Error::OpenCl(opencl3::error_codes::ClError(code))
}

// ── Buffer trait ────────────────────────────────────────────────────

/// Common accessors shared by the buffer tiers — [`DeviceSlice`],
/// [`HostBuffer`], and [`crate::svm::SharedBuffer`].
///
/// **Scope: plumbing, not tier polymorphism.** This trait exposes
/// only the inspect-the-buffer accessors that mean the same thing
/// across every tier: element count and the owning [`Context`]. It
/// is *not* an upload/download polymorphism point — those operations
/// stay on each concrete type because their signatures and lifetimes
/// genuinely differ:
///
/// - [`DeviceSlice::upload`] / [`DeviceSlice::download`] enqueue a
///   `clEnqueueRead`/`WriteBuffer` against a [`Launcher`].
/// - [`HostBuffer`] is permanently mapped — host writes/reads go
///   through `Deref<Target=[T]>` + `DerefMut`. No "upload" step.
/// - [`SharedBuffer`](crate::svm::SharedBuffer) maps lazily on demand
///   via [`SharedBuffer::map_mut`](crate::svm::SharedBuffer::map_mut)
///   and unmaps when the guard drops.
///
/// So code like `fn upload_and_run<B: Buffer<T>>(b: &mut B, data: &[T])`
/// is intentionally not possible — there is no single "upload" verb
/// that does the right thing on all three tiers, and pretending one
/// exists would force the polymorphic body to pick a worst-case
/// strategy (e.g. unconditional `clEnqueueWriteBuffer`) that pessimises
/// the zero-copy tiers.
///
/// Use this trait when you want a [`len`]/[`is_empty`]/[`ctx`]
/// accessor without committing to a tier:
///
/// ```ignore
/// fn print_size<T, B: claspr::Buffer<T>>(b: &B) {
///     println!("{} elements on {}", b.len(), b.ctx().device().name().unwrap());
/// }
/// ```
///
/// ## Future direction
///
/// If a real tier-polymorphism need surfaces (e.g. a benchmark
/// harness that wants `upload_then_run` over all three tiers), the
/// likely shape is a separate `BufferUpload<T>: Buffer<T>` super-trait
/// with a single `upload(&mut self, launcher, data)` method whose
/// impls call `clEnqueueWriteBuffer` for `DeviceSlice` and become a
/// memcpy through `DerefMut` / `map_mut` for the host-mapped tiers.
/// That can be added later without breaking the present trait's
/// callers.
///
/// [`len`]: Self::len
/// [`is_empty`]: Self::is_empty
/// [`ctx`]: Self::ctx
pub trait Buffer<T> {
    /// Number of `T` elements in the buffer.
    fn len(&self) -> usize;

    /// `true` when the buffer has zero elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The context this buffer was allocated on.
    fn ctx(&self) -> &Context;
}

// ── DeviceSlice ─────────────────────────────────────────────────────

/// A typed device-side buffer paired with its element count.
///
/// Mirrors rust-gpu's slice decomposition for kernel parameters: a
/// `&mut [T]` kernel arg becomes two `clSetKernelArg` calls (data
/// pointer + `usize` length). When passed as a launch argument,
/// claspr sets both — see the [`KernelArg`] impl in [`crate::launch`].
///
/// Construct via [`DeviceSlice::alloc`] (uninitialised) or
/// [`DeviceSlice::upload`] (with initial host data). Read back via
/// [`DeviceSlice::download`].
///
/// Host code never sees the bytes directly — for that, use
/// [`HostBuffer<T>`] (pinned host memory) or
/// [`crate::svm::SharedBuffer<T>`] (SVM).
///
/// [`KernelArg`]: crate::launch::KernelArg
pub struct DeviceSlice<T> {
    /// `ManuallyDrop` so opencl3's `Buffer::drop` (which panics on
    /// release failure) doesn't fire — our own [`Drop`] impl below
    /// calls `release_mem_object` and records into the context's
    /// sticky-error counter on failure instead.
    pub(crate) buffer: ManuallyDrop<ClBuffer<T>>,
    pub(crate) len: usize,
    pub(crate) ctx: Context,
}

impl<T> Drop for DeviceSlice<T> {
    fn drop(&mut self) {
        let mem = self.buffer.get();
        // SAFETY: mem was returned by clCreateBuffer in `alloc` and
        // we hold the only owner (ManuallyDrop suppressed opencl3's
        // own release path). Release exactly once now.
        let res = unsafe { release_mem_object(mem) };
        if res.is_err() {
            self.ctx.record_err();
        }
    }
}

impl<T> DeviceSlice<T> {
    /// Allocate a device buffer of `len` elements, uninitialised.
    ///
    /// Pure context op — no command queue needed (`clCreateBuffer`
    /// doesn't enqueue anything). Pass any `Context` (e.g. from
    /// `Context::any()` or as borrowed from a `Launcher`).
    pub fn alloc(ctx: &Context, len: usize) -> Result<Self> {
        // SAFETY: passing a null host pointer means OpenCL allocates
        // fresh device memory and ignores the host-pointer contract
        // that makes `Buffer::create` generally unsafe.
        let buffer = unsafe {
            ClBuffer::<T>::create(ctx.raw_context(), CL_MEM_READ_WRITE, len, ptr::null_mut())?
        };
        Ok(DeviceSlice {
            buffer: ManuallyDrop::new(buffer),
            len,
            ctx: ctx.clone(),
        })
    }

    /// Allocate a device buffer and write `data` into it (blocking).
    ///
    /// Needs a [`Launcher`] for the queue side. Pass `&ctx` for the
    /// default queue or `&queue` for an explicit one.
    pub fn upload<L: Launcher>(launcher: &L, data: &[T]) -> Result<Self> {
        let mut slice = Self::alloc(launcher.context(), data.len())?;
        // SAFETY: blocking write into the buffer we just allocated;
        // no aliasing, no concurrent device access.
        unsafe {
            launcher
                .cl_queue()
                .enqueue_write_buffer(&mut *slice.buffer, CL_BLOCKING, 0, data, &[])?
                .wait()?;
        }
        Ok(slice)
    }

    /// Read the buffer back into a host slice (blocking).
    ///
    /// `dst` must have the same length as `self`. Returns
    /// [`Error::LengthMismatch`] otherwise.
    pub fn download<L: Launcher>(&self, launcher: &L, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::LengthMismatch {
                src: self.len,
                dst: dst.len(),
            });
        }
        // SAFETY: blocking read; no aliasing of `dst`.
        unsafe {
            launcher
                .cl_queue()
                .enqueue_read_buffer(&*self.buffer, CL_BLOCKING, 0, dst, &[])?
                .wait()?;
        }
        Ok(())
    }

    /// Borrow the underlying opencl3 [`ClBuffer`](opencl3::memory::Buffer)
    /// for cases that need direct OpenCL access.
    pub fn buffer(&self) -> &ClBuffer<T> {
        &self.buffer
    }

    /// Copy this buffer into `dst` on the given launcher's queue.
    ///
    /// Both buffers must be on the same `Context` — OpenCL's
    /// `clEnqueueCopyBuffer` only works within one context. For
    /// cross-context transfers, download to host then re-upload.
    /// Returns the completion [`Event`](opencl3::event::Event) from
    /// the queued copy (non-blocking — call `.wait()?` on it, or feed
    /// it into a downstream [`LaunchOp::after`](crate::op::LaunchOp::after)
    /// for cross-queue chaining).
    pub fn copy_to<L: Launcher>(
        &self,
        dst: &mut DeviceSlice<T>,
        launcher: &L,
    ) -> Result<opencl3::event::Event> {
        if self.len != dst.len {
            return Err(Error::LengthMismatch {
                src: self.len,
                dst: dst.len,
            });
        }
        let bytes = self.len * std::mem::size_of::<T>();
        // SAFETY: enqueue_copy_buffer is unsafe only because the
        // caller must ensure src/dst belong to the queue's context.
        // We check len equality above; context cross-checking is on
        // the caller (pocl panics if mismatched — preferable to a
        // silent miscopy).
        let event = unsafe {
            launcher.cl_queue().enqueue_copy_buffer(
                &*self.buffer,
                &mut *dst.buffer,
                0,
                0,
                bytes,
                &[],
            )?
        };
        Ok(event)
    }
}

impl<T> Buffer<T> for DeviceSlice<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn ctx(&self) -> &Context {
        &self.ctx
    }
}

// ── HostBuffer ──────────────────────────────────────────────────────

/// A typed buffer allocated with `CL_MEM_ALLOC_HOST_PTR`: the OpenCL
/// runtime keeps the storage in host-accessible memory (typically
/// pinned), and on devices that share an address space with the host
/// (CPU OpenCL, integrated GPUs) the same allocation is the device's
/// memory too — kernel reads and writes are zero-copy.
///
/// Host access is direct via [`Deref<Target=[T]>`](std::ops::Deref)
/// and [`DerefMut`]. No `upload` / `download` needed — write the
/// slice in place, then launch the kernel; the runtime synchronises
/// the view at launch / completion boundaries.
///
/// Use [`HostBuffer::alloc`] for an uninitialised buffer or
/// [`HostBuffer::from_slice`] to copy from a host slice at construction.
pub struct HostBuffer<T> {
    buffer: ClBuffer<T>,
    /// Mapped host pointer — valid for the entire lifetime of the
    /// HostBuffer (mapped once at construction, unmapped on Drop).
    host_ptr: *mut T,
    len: usize,
    ctx: Context,
    /// Retained cl_command_queue handle used for the matching
    /// unmap call in Drop. Retained in `alloc`, released in Drop
    /// after the unmap is enqueued.
    map_queue: cl_command_queue,
}

// SAFETY: cl_mem and cl_command_queue are opaque handles; OpenCL API
// calls on them are thread-safe per the spec (CL §3.4.1). `host_ptr`
// is a stable mapped pointer for the lifetime of the buffer;
// aliasing guarantees come from the borrow checker via
// Deref/DerefMut on `&self` / `&mut self`.
unsafe impl<T: Send> Send for HostBuffer<T> {}
unsafe impl<T: Sync> Sync for HostBuffer<T> {}

impl<T> HostBuffer<T> {
    /// Allocate a `len`-element host-accessible buffer. Contents
    /// are uninitialised — write before reading.
    pub fn alloc<L: Launcher>(launcher: &L, len: usize) -> Result<Self> {
        let ctx = launcher.context();
        let q_raw: cl_command_queue = launcher.cl_queue().get();
        // SAFETY: null host pointer + CL_MEM_ALLOC_HOST_PTR tells
        // OpenCL to allocate fresh host-accessible memory.
        let buffer = unsafe {
            ClBuffer::<T>::create(
                ctx.raw_context(),
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                len,
                ptr::null_mut(),
            )?
        };
        // Map once for the lifetime of this HostBuffer. cl3's raw
        // `enqueue_map_buffer` writes the host-accessible pointer
        // into an out-param (opencl3's wrapper has the same shape).
        // We use the raw call so we don't have to thread an opencl3
        // CommandQueue wrapper into the struct.
        let mut mapped_ptr: cl_mem = ptr::null_mut();
        let map_event = unsafe {
            enqueue_map_buffer(
                q_raw,
                buffer.get(),
                CL_BLOCKING,
                CL_MAP_READ | CL_MAP_WRITE,
                0,
                len * std::mem::size_of::<T>(),
                &mut mapped_ptr,
                0,
                ptr::null(),
            )
            .map_err(cl_to_err)?
        };
        // CL_BLOCKING ensures the map completes; release the event.
        // SAFETY: map_event was returned by the call above; we own it.
        unsafe { opencl3::event::release_event(map_event).map_err(cl_to_err)? };
        let host_ptr = mapped_ptr.cast::<T>();
        // Retain the queue so its raw handle is valid for the unmap
        // we issue in Drop, even if the user's Launcher gets dropped
        // earlier.
        // SAFETY: q_raw was just obtained from a live CommandQueue.
        unsafe { retain_command_queue(q_raw).map_err(cl_to_err)? };
        Ok(HostBuffer {
            buffer,
            host_ptr,
            len,
            ctx: ctx.clone(),
            map_queue: q_raw,
        })
    }

    /// Allocate and initialise from a host slice. Same as
    /// [`alloc`](Self::alloc) followed by `copy_from_slice` on the
    /// `DerefMut` view.
    pub fn from_slice<L: Launcher>(launcher: &L, data: &[T]) -> Result<Self>
    where
        T: Copy,
    {
        let mut buf = Self::alloc(launcher, data.len())?;
        buf.copy_from_slice(data);
        Ok(buf)
    }

    /// Borrow the underlying opencl3 [`ClBuffer`](opencl3::memory::Buffer).
    pub fn buffer(&self) -> &ClBuffer<T> {
        &self.buffer
    }
}

impl<T> Buffer<T> for HostBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn ctx(&self) -> &Context {
        &self.ctx
    }
}

impl<T> Deref for HostBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: host_ptr is a stable mapped pointer for `len`
        // elements of T; the OpenCL runtime guarantees it remains
        // valid until we unmap (Drop only).
        unsafe { slice::from_raw_parts(self.host_ptr, self.len) }
    }
}

impl<T> DerefMut for HostBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: see Deref. `&mut self` upgrades the unique-access
        // guarantee from the borrow checker into a mutable slice.
        unsafe { slice::from_raw_parts_mut(self.host_ptr, self.len) }
    }
}

impl<T> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: we mapped this pointer in `alloc` and held it for
        // the whole lifetime; unmap once now. We don't wait on the
        // returned event — clReleaseMemObject (in our Buffer's Drop)
        // refcounts the cl_mem until queued commands complete, so
        // the buffer stays alive long enough for the unmap to land.
        // Errors here can't be propagated; bump the sticky counter.
        let unmap_res = unsafe {
            enqueue_unmap_mem_object(
                self.map_queue,
                self.buffer.get(),
                self.host_ptr.cast(),
                0,
                ptr::null(),
            )
        };
        if unmap_res.is_err() {
            self.ctx.record_err();
        }
        // SAFETY: queue was retained in `alloc`; release exactly once.
        let rel_res = unsafe { release_command_queue(self.map_queue) };
        if rel_res.is_err() {
            self.ctx.record_err();
        }
    }
}
