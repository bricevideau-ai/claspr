//! [`USMSlice<T>`] — wraps a host `Vec<T>` as a fine-grain-system
//! SVM slice, passing its pointer directly to kernels via
//! `clSetKernelArgSVMPointer`.
//!
//! Requires `CL_DEVICE_SVM_FINE_GRAIN_SYSTEM` (the OpenCL 2.0+
//! capability where any host pointer is valid in a kernel — no
//! allocator call, no map/unmap, no explicit sync). claspr's
//! [`SvmLevel::FineSystem`](crate::SvmLevel::FineSystem) reports it;
//! [`USMSlice::new`] errors with [`Error::NotSupported`] when absent.
//!
//! # Why "USM"
//!
//! The name decouples this primitive from the OpenCL-spec mechanism
//! (fine-grain system SVM today) so we can later add an Intel
//! Shared-USM backend without renaming. At the host-facing level
//! both look the same: a host pointer the kernel just reads.
//!
//! # Compared to [`crate::DeviceSlice`] and [`crate::MappedSlice`]
//!
//! | type | host access | kernel access | needs |
//! |------|-------------|---------------|-------|
//! | [`DeviceSlice<T>`](crate::DeviceSlice) | via `upload` / `download` | direct (cl_mem arg) | OpenCL 1.0+ |
//! | [`MappedSlice<T>`](crate::MappedSlice) | via `.map()` / `.map_mut()` guard | direct (SVM ptr arg) | SVM coarse-grain |
//! | [`USMSlice<T>`] | direct (`Deref<Target=[T]>` / `DerefMut`) | direct (SVM ptr arg) | SVM fine-grain system |
//!
//! `USMSlice` is the only one of the three where host code can read
//! / write the buffer while a kernel is concurrently using it — that's
//! the "fine grain" guarantee. The cost: narrower runtime support.
//!
//! [`Error::NotSupported`]: crate::Error::NotSupported

use crate::buffer::Buffer;
use crate::context::{Context, SvmLevel};
use crate::error::{Error, Result};
use crate::launch::KernelArg;
use opencl3::event::{Event, retain_event};
use opencl3::kernel::ExecuteKernel;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

/// A host `Vec<T>` wrapped as a fine-grain-system SVM slice. The
/// host's `Vec` owns the memory; the kernel reads/writes through
/// the same pointer.
///
/// Construction gates on
/// [`SvmLevel::FineSystem`](crate::SvmLevel::FineSystem) — errors
/// with [`Error::NotSupported`](crate::Error::NotSupported) on
/// devices without fine-grain system SVM. Host access goes through
/// `Deref<Target=[T]>` / `DerefMut` directly on the wrapped Vec;
/// kernel access goes through `clSetKernelArgSVMPointer` (driven by
/// the `KernelArg` impl).
///
/// ## Drop semantics
///
/// `USMSlice::Drop` **blocks the host** on every in-flight event
/// recorded via [`register_use`](Self::register_use) before letting
/// the Vec drop. The kernel may still be reading from the Vec's
/// bytes; without the wait, freeing the host allocation would race
/// the kernel. Unlike `MappedSlice`'s `clEnqueueSVMFree` (which can
/// queue-order behind in-flight events), the host `Vec` drop is
/// synchronous — so the wait must be too.
pub struct USMSlice<T> {
    data: Vec<T>,
    ctx: Context,
    /// Every kernel-launch event that received this slice's SVM
    /// pointer. Populated by [`KernelArg::register_completion`] via
    /// [`register_use`](Self::register_use). Drop waits on each
    /// before letting the host Vec drop.
    in_flight: Mutex<Vec<Arc<Event>>>,
}

// SAFETY: USMSlice contains an owned Vec<T> plus an Arc<Event> Vec;
// it's Send/Sync whenever its T is. The SVM pointer derived from the
// Vec is shared between host and kernel by design (fine-grain system
// SVM), but Rust-level aliasing rules apply to the Vec view:
// `Deref<Target=[T]>` on `&self` gives `&[T]`, `DerefMut` on `&mut self`
// gives `&mut [T]`. Concurrent kernel access to the bytes while host
// code holds either is fine at the OpenCL level (fine-grain semantics)
// but the user is responsible for not racing host-side mutation with
// in-flight kernel writes — same contract as any shared-memory model.
unsafe impl<T: Send> Send for USMSlice<T> {}
unsafe impl<T: Sync> Sync for USMSlice<T> {}

impl<T> USMSlice<T> {
    /// Wrap `data` as a USMSlice. Errors with
    /// [`Error::NotSupported`](crate::Error::NotSupported) if the
    /// context's device doesn't advertise
    /// `CL_DEVICE_SVM_FINE_GRAIN_SYSTEM`.
    ///
    /// Doesn't enqueue or allocate; just records the Vec + context.
    /// The host pointer is the Vec's existing allocation.
    pub fn new(ctx: &Context, data: Vec<T>) -> Result<Self> {
        if ctx.svm_capability() != SvmLevel::FineSystem {
            return Err(Error::NotSupported(
                "CL_DEVICE_SVM_FINE_GRAIN_SYSTEM required for USMSlice",
            ));
        }
        Ok(USMSlice {
            data,
            ctx: ctx.clone(),
            in_flight: Mutex::new(Vec::new()),
        })
    }

    /// Append `event` to the in-flight-use list. Drop blocks on
    /// every recorded event before letting the host Vec drop.
    ///
    /// Most users never call this directly: the [`KernelArg`] impl
    /// invokes it automatically for every kernel launch whose args
    /// include this USMSlice. Public so hand-rolled SVM use
    /// (manual `clSetKernelArgSVMPointer`) can stay Drop-safe.
    pub fn register_use(&self, event: Arc<Event>) {
        self.in_flight
            .lock()
            .expect("in_flight mutex poisoned")
            .push(event);
    }

    /// Raw SVM pointer (the wrapped Vec's allocation). Same escape
    /// hatch as [`MappedSlice::ptr`](crate::MappedSlice::ptr).
    pub fn ptr(&self) -> *mut T {
        self.data.as_ptr() as *mut T
    }
}

impl<T: Default + Copy + Send + 'static> USMSlice<T> {
    /// Allocate a USMSlice of `len` elements initialised to
    /// `T::default()`. Convenience wrapper over
    /// [`new(ctx, vec![T::default(); len])`](Self::new), symmetric
    /// with [`DeviceSlice::alloc`](crate::DeviceSlice::alloc) and
    /// [`MappedSlice::alloc`](crate::MappedSlice::alloc).
    ///
    /// No perf win over the explicit `new` form — the Vec still needs
    /// to be initialised before construction (`USMSlice` derefs to
    /// `&[T]` so uninit bytes would be unsound to expose). The benefit
    /// is API symmetry across tiers and a shorter call site for the
    /// common "I just want N zeroed elements the kernel will fill"
    /// pattern.
    pub fn alloc(ctx: &Context, len: usize) -> Result<Self> {
        Self::new(ctx, vec![T::default(); len])
    }
}

impl<T> Buffer<T> for USMSlice<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn ctx(&self) -> &Context {
        &self.ctx
    }
}

impl<T> Deref for USMSlice<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.data
    }
}

impl<T> DerefMut for USMSlice<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Metadata-only `Debug` — doesn't print the Vec contents (could be
/// huge / sensitive) and doesn't require `T: Debug`.
impl<T> fmt::Debug for USMSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("USMSlice")
            .field("len", &self.data.len())
            .field("element_size", &std::mem::size_of::<T>())
            .finish_non_exhaustive()
    }
}

impl<T> KernelArg for USMSlice<T> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        // Same slice-decomposition as MappedSlice: SVM pointer
        // first (via clSetKernelArgSVMPointer), then length as a
        // regular scalar arg.
        let len: usize = self.data.len();
        // SAFETY: set_arg_svm is unsafe because the pointer must be
        // a valid SVM allocation on the kernel's context. With
        // fine-grain system SVM, any host pointer qualifies — and
        // ours is the Vec's backing storage, alive for the lifetime
        // of self.
        unsafe {
            exec.set_arg_svm(self.ptr()).set_arg(&len);
        }
    }

    /// Retain the kernel's completion event and push it onto
    /// `in_flight`. Drop will block on it before the host Vec is
    /// freed. Without this, dropping a USMSlice while a kernel is
    /// still reading from the Vec would be UB.
    fn register_completion(&self, event: &Event) {
        let raw = event.get();
        // SAFETY: retain bumps cl_event's refcount; we construct a
        // second owning Event from the raw handle that will release
        // on Drop. Pair: retain here, release when the Arc<Event>
        // (held in `in_flight`) drops after Drop's wait loop.
        if unsafe { retain_event(raw) }.is_err() {
            self.ctx.record_err();
            return;
        }
        let owned = Event::from(raw);
        self.register_use(Arc::new(owned));
    }
}

impl<T> Drop for USMSlice<T> {
    fn drop(&mut self) {
        // Block host-side on every in-flight event before the Vec
        // drops. Unlike MappedSlice (where clEnqueueSVMFree can
        // queue-order behind the in-flight events) we have no
        // OpenCL command to enqueue — Rust's Vec free runs
        // immediately and unconditionally on the host. The only way
        // to avoid a race is a synchronous wait.
        let events = std::mem::take(&mut *self.in_flight.lock().expect("in_flight mutex poisoned"));
        for ev in &events {
            // Wait errors here can't be propagated — bump the
            // sticky counter so test post-conditions catch them.
            if ev.wait().is_err() {
                self.ctx.record_err();
            }
        }
        // Vec drops here, after all in-flight kernels have completed.
    }
}
