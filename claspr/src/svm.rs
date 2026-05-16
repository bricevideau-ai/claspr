//! Shared Virtual Memory ([`SharedBuffer`]) — OpenCL 2.0+ coarse-grain
//! SVM buffers.
//!
//! SVM gives kernel and host the *same pointer* into a single
//! allocation. claspr exposes coarse-grain SVM today: host access
//! requires [`map`](SharedBuffer::map) / [`map_mut`](SharedBuffer::map_mut)
//! around the bytes you want to read or write, with the runtime
//! ensuring the device-side view is coherent at the boundaries.
//!
//! Construction is gated on [`crate::SvmLevel`]:
//! `SharedBuffer::alloc` returns [`crate::Error::SvmNotAvailable`]
//! when the device reports [`crate::SvmLevel::None`]. Check
//! `ctx.svm_capability()` if you want to fall back to a
//! [`crate::DeviceSlice`] or [`crate::HostBuffer`] gracefully.
//!
//! # Example
//!
//! ```ignore
//! use claspr::{Context, SharedBuffer, SvmLevel};
//!
//! let ctx = Context::any()?;
//! if ctx.svm_capability() == SvmLevel::None {
//!     return skip("device has no SVM");
//! }
//!
//! let mut buf = SharedBuffer::<u32>::alloc(&ctx, 1024)?;
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
use crate::queue::Launcher;
use opencl3::command_queue::{
    enqueue_svm_map, enqueue_svm_unmap, release_command_queue, retain_command_queue,
};
use opencl3::event::release_event;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE, CL_MEM_READ_WRITE, svm_alloc, svm_free};
use opencl3::types::{CL_BLOCKING, cl_command_queue, cl_int, cl_uint};
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::slice;

fn cl_to_err(code: cl_int) -> Error {
    Error::OpenCl(opencl3::error_codes::ClError(code))
}

// ── SharedBuffer ────────────────────────────────────────────────────

/// A typed Shared Virtual Memory allocation.
///
/// Construction allocates via `clSVMAlloc`; Drop releases via
/// `clSVMFree`. Host access is RAII-guarded via [`map`](Self::map) and
/// [`map_mut`](Self::map_mut): each returns a guard that issues
/// `clEnqueueSVMMap` on construction and `clEnqueueSVMUnmap` on Drop.
pub struct SharedBuffer<T> {
    ptr: *mut T,
    len: usize,
    ctx: Context,
}

// SAFETY: the SVM pointer is a runtime-owned allocation in
// host-accessible memory; OpenCL guarantees thread-safety for API
// calls on it (CL §3.4.1). Aliasing is governed by the map guards,
// which use the borrow checker to enforce exclusivity for `map_mut`.
unsafe impl<T: Send> Send for SharedBuffer<T> {}
unsafe impl<T: Sync> Sync for SharedBuffer<T> {}

impl<T> SharedBuffer<T> {
    /// Allocate `len` elements of T in SVM memory, uninitialised.
    ///
    /// Returns [`Error::SvmNotAvailable`] if the context's device
    /// reports [`SvmLevel::None`] for `CL_DEVICE_SVM_CAPABILITIES`.
    pub fn alloc(ctx: &Context, len: usize) -> Result<Self> {
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
        Ok(SharedBuffer {
            ptr: raw.cast::<T>(),
            len,
            ctx: ctx.clone(),
        })
    }

    /// Map this buffer for host read access. Returns a RAII guard
    /// that derefs to `&[T]` and unmaps on Drop.
    ///
    /// The map is blocking — `clEnqueueSVMMap` with `CL_TRUE` for
    /// the blocking flag and `CL_MAP_READ`.
    pub fn map<'a, L: Launcher>(&'a self, launcher: &L) -> Result<SharedReadGuard<'a, T>> {
        SharedReadGuard::new(self, launcher)
    }

    /// Map this buffer for host read+write access. Returns a RAII
    /// guard that derefs to `&mut [T]` and unmaps on Drop. The
    /// `&mut self` receiver gives the borrow checker the
    /// exclusivity guarantee needed for `DerefMut`.
    pub fn map_mut<'a, L: Launcher>(&'a mut self, launcher: &L) -> Result<SharedWriteGuard<'a, T>> {
        SharedWriteGuard::new(self, launcher)
    }

    /// Raw SVM pointer for direct use (e.g. passing to a kernel arg
    /// out-of-band, or interoperating with hand-written OpenCL).
    pub fn ptr(&self) -> *mut T {
        self.ptr
    }
}

impl<T> Buffer<T> for SharedBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn ctx(&self) -> &Context {
        &self.ctx
    }
}

impl<T> Drop for SharedBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: ptr was returned by svm_alloc on this context;
        // free exactly once. Errors here can't be propagated;
        // record in the sticky counter.
        let res = unsafe { svm_free(self.ctx.raw_context().get(), self.ptr.cast()) };
        if res.is_err() {
            self.ctx.record_err();
        }
    }
}

impl<T> KernelArg for SharedBuffer<T> {
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
}

// ── Map guards ──────────────────────────────────────────────────────

/// RAII guard for a SVM read map. Drop issues `clEnqueueSVMUnmap`
/// and releases the retained queue handle.
pub struct SharedReadGuard<'a, T> {
    buf: &'a SharedBuffer<T>,
    queue: cl_command_queue,
}

impl<'a, T> SharedReadGuard<'a, T> {
    fn new<L: Launcher>(buf: &'a SharedBuffer<T>, launcher: &L) -> Result<Self> {
        let q_raw: cl_command_queue = launcher.cl_queue().get();
        // SAFETY: blocking map for read; queue is alive (we hold &launcher).
        let evt = unsafe {
            enqueue_svm_map(
                q_raw,
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
        // Retain the queue so its handle stays valid for unmap on Drop.
        unsafe { retain_command_queue(q_raw).map_err(cl_to_err)? };
        Ok(SharedReadGuard { buf, queue: q_raw })
    }
}

impl<T> Deref for SharedReadGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: the SVM pointer is valid + mapped for read for
        // this guard's lifetime.
        unsafe { slice::from_raw_parts(self.buf.ptr, self.buf.len) }
    }
}

impl<T> Drop for SharedReadGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: ptr was mapped in `new`; unmap exactly once now.
        let unmap = unsafe { enqueue_svm_unmap(self.queue, self.buf.ptr.cast(), 0, ptr::null()) };
        if let Ok(evt) = unmap {
            let _ = unsafe { release_event(evt) };
        } else {
            self.buf.ctx.record_err();
        }
        let rel = unsafe { release_command_queue(self.queue) };
        if rel.is_err() {
            self.buf.ctx.record_err();
        }
    }
}

/// RAII guard for a SVM write map. Drop issues `clEnqueueSVMUnmap`
/// and releases the retained queue handle.
pub struct SharedWriteGuard<'a, T> {
    buf: &'a mut SharedBuffer<T>,
    queue: cl_command_queue,
}

impl<'a, T> SharedWriteGuard<'a, T> {
    fn new<L: Launcher>(buf: &'a mut SharedBuffer<T>, launcher: &L) -> Result<Self> {
        let q_raw: cl_command_queue = launcher.cl_queue().get();
        let ptr = buf.ptr;
        let len = buf.len;
        // SAFETY: blocking map for read+write.
        let evt = unsafe {
            enqueue_svm_map(
                q_raw,
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
        unsafe { retain_command_queue(q_raw).map_err(cl_to_err)? };
        Ok(SharedWriteGuard { buf, queue: q_raw })
    }
}

impl<T> Deref for SharedWriteGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: see SharedReadGuard.
        unsafe { slice::from_raw_parts(self.buf.ptr, self.buf.len) }
    }
}

impl<T> DerefMut for SharedWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: `&mut self` upgrades to a unique mutable slice;
        // mapped read+write for the guard's lifetime.
        unsafe { slice::from_raw_parts_mut(self.buf.ptr, self.buf.len) }
    }
}

impl<T> Drop for SharedWriteGuard<'_, T> {
    fn drop(&mut self) {
        let unmap = unsafe { enqueue_svm_unmap(self.queue, self.buf.ptr.cast(), 0, ptr::null()) };
        if let Ok(evt) = unmap {
            let _ = unsafe { release_event(evt) };
        } else {
            self.buf.ctx.record_err();
        }
        let rel = unsafe { release_command_queue(self.queue) };
        if rel.is_err() {
            self.buf.ctx.record_err();
        }
    }
}
