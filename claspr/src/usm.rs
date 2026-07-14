//! [`USMSlice<T>`] — wraps a host `Vec<T>` as a fine-grain-system
//! SVM slice, passing its pointer directly to kernels via
//! `clSetKernelArgSVMPointer`.
//!
//! Requires `CL_DEVICE_SVM_FINE_GRAIN_SYSTEM` (the OpenCL 2.0+
//! capability where any host pointer is valid in a kernel — no
//! allocator call, no map/unmap, no explicit sync). claspr's
//! [`SvmLevel::FineSystem`] reports it;
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

use crate::access::{HostReadable, HostWritable, MemMode, ReadWrite};
use crate::buffer::Buffer;
use crate::context::{Context, SvmLevel};
use crate::error::{Error, Result};
use crate::launch::KernelArg;
use opencl3::event::{Event, retain_event};
use opencl3::kernel::ExecuteKernel;
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

/// A host `Vec<T>` wrapped as a fine-grain-system SVM slice. The
/// host's `Vec` owns the memory; the kernel reads/writes through
/// the same pointer.
///
/// Construction gates on
/// [`SvmLevel::FineSystem`] — errors
/// with [`Error::NotSupported`] on
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
pub struct USMSlice<T, M: MemMode = ReadWrite> {
    data: Vec<T>,
    ctx: Context,
    /// Every kernel-launch event that received this slice's SVM
    /// pointer. Populated by [`KernelArg::register_completion`] via
    /// [`register_use`](Self::register_use). Drop waits on each
    /// before letting the host Vec drop.
    in_flight: Mutex<Vec<Arc<Event>>>,
    /// Type-level access-mode tag. USMSlice's backing memory is just
    /// a Rust `Vec<T>` — no `cl_mem_flags` at construction time. The
    /// marker exists purely to enable type-level method gating
    /// uniformly with `DeviceSlice<T, M>` and `MappedSlice<T, M>`:
    /// markers without `HostWritable` lose `DerefMut`, markers
    /// without `KernelWritable` lose kernel-write capability, etc.
    /// The OpenCL runtime can't independently enforce these (no
    /// flag goes through) — the safety boundary is the Rust type
    /// system.
    _mode: PhantomData<fn() -> M>,
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
unsafe impl<T: Send, M: MemMode> Send for USMSlice<T, M> {}
unsafe impl<T: Sync, M: MemMode> Sync for USMSlice<T, M> {}

impl<T, M: MemMode> USMSlice<T, M> {
    /// Wrap `data` as a USMSlice with the marker `M`. Errors with
    /// [`Error::NotSupported`] if the
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
            _mode: PhantomData,
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

impl<T: Copy, M: MemMode> USMSlice<T, M> {
    /// Re-seed the wrapped host `Vec` in place from `src`, over the SAME
    /// allocation (the SVM pointer is unchanged). USM IS host memory
    /// (fine-grain system SVM over a Rust `Vec`), so a re-seed is a plain
    /// host `copy_from_slice` — no `clEnqueueSVMMemcpy`, no map — after
    /// draining any in-flight kernel-use events so we don't overwrite bytes
    /// a previous run's kernel is still reading.
    ///
    /// The reusable-graph twin of `Upload`'s `write_buffer_enqueue`
    /// reseed-on-replay path, used by the [`usm_slice`](crate::eager::usm_slice)
    /// leaf to keep a replayed USM chain head idempotent. `src.len()` must
    /// equal `self.len()` (the buffer is allocated once at the source's
    /// length and never resized).
    pub(crate) fn reseed_sync(&mut self, src: &[T]) -> Result<()> {
        debug_assert_eq!(
            src.len(),
            self.data.len(),
            "USMSlice::reseed_sync length mismatch"
        );
        // Drain in-flight kernel-use events before touching the bytes — a
        // prior run's kernel may still be reading this SVM region. This is
        // the same host-side wait `Drop` performs, but here we KEEP the
        // buffer alive and re-arm it. Errors bump the sticky context flag.
        let events = std::mem::take(&mut *self.in_flight.lock().expect("in_flight mutex poisoned"));
        for ev in &events {
            ev.wait().map_err(Error::OpenCl)?;
        }
        self.data.copy_from_slice(src);
        Ok(())
    }
}

impl<T: Default + Copy + Send + 'static, M: MemMode> USMSlice<T, M> {
    /// Allocate a USMSlice of `len` elements initialised to
    /// `T::default()`. Convenience wrapper over
    /// [`new(ctx, vec![T::default(); len])`](Self::new), symmetric
    /// with [`DeviceSlice::alloc_zero`](crate::DeviceSlice::alloc_zero)
    /// and [`MappedSlice::alloc_zero`](crate::MappedSlice::alloc_zero).
    ///
    /// **No marker bound** — USM is host memory backed by a Rust
    /// `Vec<T>`, and `vec![T::default(); N]` is a pure host op that
    /// works regardless of any kernel-side marker. USM markers gate
    /// kernel access (via `M::KERNEL_FLAGS`), never host alloc.
    pub fn alloc_zero(ctx: &Context, len: usize) -> Result<Self> {
        Self::new(ctx, vec![T::default(); len])
    }
}

// ── USMScalar — the fine-grain-system-SVM device scalar ─────────────

/// A [`Scalar`](crate::Scalar) backed by a length-1 [`USMSlice<T, M>`]
/// (fine-grain system SVM over a host `Vec<T>`). The USM tier's
/// device-resident scalar, symmetric with
/// [`DeviceScalar`](crate::DeviceScalar) — binds ONLY to scalar-ref
/// kernel params and delegates the host `Vec` + in-flight-wait
/// bookkeeping to the backing `USMSlice`.
pub type USMScalar<T, M = ReadWrite> = crate::Scalar<USMSlice<T, M>>;

/// The uninit type-state for a [`USMScalar<T, M>`].
pub type USMScalarUninit<T, M = ReadWrite> = crate::ScalarUninit<USMSliceUninit<T, M>>;

impl<T: Copy + Send + 'static, M: MemMode> crate::Scalar<USMSlice<T, M>> {
    /// Create a USM device scalar seeded with `value` (a length-1 host
    /// `Vec<T>` wrapped as USM). Errors with [`Error::NotSupported`] on
    /// a device without fine-grain system SVM.
    pub fn new_usm(ctx: &Context, value: T) -> Result<Self> {
        Ok(crate::Scalar {
            inner: USMSlice::new(ctx, vec![value])?,
        })
    }
}

impl<T, M: MemMode> crate::Scalar<USMSlice<T, M>> {
    /// Allocate an uninitialised USM device scalar, returning the
    /// type-state-gated [`USMScalarUninit`]. Mirrors
    /// [`USMSlice::alloc_uninit`].
    pub fn uninit_usm(ctx: &Context) -> Result<USMScalarUninit<T, M>> {
        Ok(crate::ScalarUninit {
            inner: USMSlice::alloc_uninit(ctx, 1)?,
        })
    }

    /// Borrow the backing length-1 [`USMSlice<T, M>`].
    pub fn as_usm_slice(&self) -> &USMSlice<T, M> {
        &self.inner
    }
}

impl<T, M: MemMode> crate::ScalarUninit<USMSliceUninit<T, M>> {
    /// Assert the scalar has been (or will be) written before any read.
    ///
    /// # Safety
    ///
    /// Same contract as [`USMSliceUninit::assume_init`].
    pub unsafe fn assume_init_usm(self) -> USMScalar<T, M> {
        // SAFETY: forwarded to the caller (see this fn's Safety section).
        crate::Scalar {
            inner: unsafe { self.inner.assume_init() },
        }
    }
}

/// Free-fn ctor for a seeded [`USMScalar<T>`] (default marker).
pub fn usm_scalar<T: Copy + Send + 'static>(
    ctx: &Context,
    value: T,
) -> Result<USMScalar<T, ReadWrite>> {
    USMScalar::new_usm(ctx, value)
}

/// Free-fn ctor for an uninitialised [`USMScalar<T>`] (default marker).
pub fn usm_scalar_uninit<T>(ctx: &Context) -> Result<USMScalarUninit<T, ReadWrite>> {
    USMScalar::<T, ReadWrite>::uninit_usm(ctx)
}

impl<T, M: MemMode> USMSlice<T, M> {
    /// Allocate a USMSlice of `len` elements with uninitialised
    /// contents. Returns a [`USMSliceUninit<T, M>`] wrapper — host
    /// reads are type-blocked; transition to an initialised
    /// `USMSlice<T, M>` via [`unsafe fn assume_init`](USMSliceUninit::assume_init)
    /// (caller vouches every byte gets written before any read).
    ///
    /// SVM analog of [`DeviceSlice::alloc_uninit`](crate::DeviceSlice::alloc_uninit). **No marker
    /// bound** — the type-state wrapper is the safety gate, and USM
    /// markers don't affect host allocation.
    ///
    /// Today's implementation is `Vec::with_capacity(len) + set_len`
    /// — the bytes are uninit at the Rust level. When the future
    /// Intel SharedUSM backend lands (see
    /// `[[usm-slice-shared-usm-refactor]]`), `alloc_uninit` will
    /// dispatch to the SharedUSM-specific alloc primitive and skip
    /// the wasted intermediate Vec entirely.
    pub fn alloc_uninit(ctx: &Context, len: usize) -> Result<USMSliceUninit<T, M>> {
        if ctx.svm_capability() != SvmLevel::FineSystem {
            return Err(Error::NotSupported(
                "CL_DEVICE_SVM_FINE_GRAIN_SYSTEM required for USMSlice",
            ));
        }
        // Store as Vec<MaybeUninit<T>> until assume_init — MaybeUninit
        // is always layout-equivalent to T but the "always valid"
        // wrapper sidesteps the `clippy::uninit_vec` lint we'd hit
        // with Vec<T> + set_len. The transmute back to Vec<T> happens
        // in assume_init via from_raw_parts (no realloc, same heap
        // address — SVM pointer stability preserved).
        let mut data: Vec<MaybeUninit<T>> = Vec::with_capacity(len);
        // SAFETY: MaybeUninit<T> is always valid in any bit pattern,
        // so set_len to the freshly-allocated capacity is sound.
        unsafe {
            data.set_len(len);
        }
        Ok(USMSliceUninit {
            data,
            ctx: ctx.clone(),
            _mode: PhantomData,
        })
    }
}

/// Type-state wrapper returned by [`USMSlice::alloc_uninit`]. SVM /
/// fine-grain-system analog of [`crate::DeviceSliceUninit`] /
/// [`crate::MappedSliceUninit`].
///
/// Today's implementation is `Vec<MaybeUninit<T>>` so the practical
/// benefit over `USMSlice::alloc_zero` is small. The wrapper's main
/// value is forward-compatibility: when an Intel SharedUSM backend
/// is added to USMSlice, `alloc_uninit` there will skip the wasted
/// intermediate Vec, and this wrapper stays the same shape across
/// backends.
pub struct USMSliceUninit<T, M: MemMode = ReadWrite> {
    data: Vec<MaybeUninit<T>>,
    ctx: Context,
    _mode: PhantomData<fn() -> M>,
}

// SAFETY: same justification as `USMSlice` — backing storage is
// Send/Sync whenever T is, and the MaybeUninit wrapper is layout-
// equivalent to T.
unsafe impl<T: Send, M: MemMode> Send for USMSliceUninit<T, M> {}
unsafe impl<T: Sync, M: MemMode> Sync for USMSliceUninit<T, M> {}

impl<T, M: MemMode> USMSliceUninit<T, M> {
    /// Skip safe initialization. See
    /// [`crate::DeviceSliceUninit::assume_init`] for the safety
    /// contract.
    ///
    /// # Safety
    ///
    /// Every byte of the wrapped Vec must be written by SOME path
    /// (kernel, host memcpy, etc.) before any read can observe the
    /// bytes. For numeric `T` an uninit-byte read is arbitrary
    /// garbage; for `T` with invalid bit patterns it is UB.
    pub unsafe fn assume_init(self) -> USMSlice<T, M> {
        // SAFETY: MaybeUninit<T> has the same layout as T. The
        // caller has vouched that every slot is initialized.
        // Reconstruct the Vec at the right element type without
        // realloc — heap address stays stable, so any SVM pointer
        // already taken from `data.as_ptr()` is still valid.
        let mut data = self.data;
        let ptr = data.as_mut_ptr() as *mut T;
        let len = data.len();
        let cap = data.capacity();
        std::mem::forget(data);
        let data_t: Vec<T> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
        USMSlice {
            data: data_t,
            ctx: self.ctx,
            in_flight: Mutex::new(Vec::new()),
            _mode: PhantomData,
        }
    }

    /// Downgrade an initialized [`USMSlice`] back to `USMSliceUninit` — the
    /// inverse of [`assume_init`](Self::assume_init), and the SAFE direction:
    /// `Init` is the stronger capability (read+write), so forgetting it to a
    /// write-only `Uninit` is always sound (an Init buffer can be used wherever
    /// an Uninit is expected). Used by the reuse home-channel to re-arm a
    /// `Cell<USMSliceUninit>` after a copy wrote the buffer (which earned the
    /// Uninit→Init transition; this hands the same allocation back for the next
    /// run to overwrite).
    ///
    /// Mirrors `assume_init` in reverse to preserve the heap address — the SVM
    /// pointer was taken from `data.as_ptr()` and must stay valid.
    pub(crate) fn from_init(slice: USMSlice<T, M>) -> Self {
        // `USMSlice: Drop` (it waits on in-flight events, then frees the Vec), so
        // we can't move its fields out by value. Suppress its Drop with
        // `ManuallyDrop` and read the fields out: the copy that produced this
        // buffer already completed before the Checkout returned it, so there are
        // no in-flight events to wait on, and we are REUSING the allocation, not
        // freeing it.
        let mut slice = std::mem::ManuallyDrop::new(slice);
        // SAFETY: each field is read exactly once out of the ManuallyDrop'd slice
        // (which is never dropped, so no double-free / use-after-read). `data` and
        // `ctx` are moved out; `in_flight` (already drained) and `_mode` (ZST) are
        // left to leak harmlessly inside the ManuallyDrop.
        let data: Vec<T> = unsafe { std::ptr::read(&slice.data) };
        let ctx = unsafe { std::ptr::read(&slice.ctx) };
        // Drop in_flight explicitly (an empty Vec at this point) so nothing leaks.
        unsafe { std::ptr::drop_in_place(&mut slice.in_flight) };
        // SAFETY: `MaybeUninit<T>` has the same layout as `T`. Reinterpret the
        // already-initialized `Vec<T>` as `Vec<MaybeUninit<T>>` (a DOWNGRADE — no
        // init assertion needed, unlike `assume_init`'s upgrade), reusing the same
        // allocation so the heap address (and any SVM pointer derived from it) is
        // unchanged.
        let mut data = std::mem::ManuallyDrop::new(data);
        let ptr = data.as_mut_ptr() as *mut MaybeUninit<T>;
        let len = data.len();
        let cap = data.capacity();
        let data_u: Vec<MaybeUninit<T>> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
        USMSliceUninit {
            data: data_u,
            ctx,
            _mode: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn ctx(&self) -> &Context {
        &self.ctx
    }
}

impl<T, M: MemMode> crate::record::RecordableBuffer for USMSliceUninit<T, M> {
    fn record_handle(&self) -> crate::record::BufHandle {
        crate::record::BufHandle {
            mem: crate::record::MemRef::Svm(self.data.as_ptr() as *mut std::ffi::c_void),
            byte_len: self.data.len() * std::mem::size_of::<T>(),
        }
    }
}

impl<T, M: MemMode> fmt::Debug for USMSliceUninit<T, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("USMSliceUninit")
            .field("len", &self.data.len())
            .field("element_size", &std::mem::size_of::<T>())
            .finish_non_exhaustive()
    }
}

impl<T: Copy, M: MemMode> USMSliceUninit<T, M> {
    /// Initialize every slot to `value` and transition to an
    /// initialised [`USMSlice<T, M>`]. Pure host operation — USM
    /// is host memory, no OpenCL enqueue needed. Synchronous.
    ///
    /// Used by the Tier 2 [`crate::USMSlice`]-aware `.fill()` trait
    /// method to implement the compositional
    /// `alloc_uninit + fill` pattern uniformly across buffer kinds.
    pub fn fill_into(self, value: T) -> USMSlice<T, M> {
        let mut data = self.data;
        for slot in data.iter_mut() {
            *slot = MaybeUninit::new(value);
        }
        // SAFETY: every slot just got written by the loop above.
        // Reuse the same Vec storage (no realloc — SVM-stability
        // contract preserved).
        let ptr = data.as_mut_ptr() as *mut T;
        let len = data.len();
        let cap = data.capacity();
        std::mem::forget(data);
        let data_t: Vec<T> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
        USMSlice {
            data: data_t,
            ctx: self.ctx,
            in_flight: Mutex::new(Vec::new()),
            _mode: PhantomData,
        }
    }

    /// Memcpy `src` into the uninitialised slots and transition to
    /// [`USMSlice<T, M>`]. Returns [`Error::LengthMismatch`] if `src.len()`
    /// differs from the slice length. Synchronous host operation.
    pub fn write_from(self, src: &[T]) -> Result<USMSlice<T, M>> {
        let mut data = self.data;
        if src.len() != data.len() {
            return Err(Error::LengthMismatch {
                src: src.len(),
                dst: data.len(),
            });
        }
        // SAFETY: src.as_ptr() and data.as_mut_ptr() are both valid
        // for `len` Ts; MaybeUninit<T> has the same layout as T so
        // a byte-copy into the MaybeUninit slots is sound and
        // leaves every slot initialized.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), data.as_mut_ptr() as *mut T, src.len());
        }
        let ptr = data.as_mut_ptr() as *mut T;
        let len = data.len();
        let cap = data.capacity();
        std::mem::forget(data);
        let data_t: Vec<T> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
        Ok(USMSlice {
            data: data_t,
            ctx: self.ctx,
            in_flight: Mutex::new(Vec::new()),
            _mode: PhantomData,
        })
    }
}

impl<T, M: MemMode> Buffer<T> for USMSlice<T, M> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn ctx(&self) -> &Context {
        &self.ctx
    }
}

impl<T, M: MemMode + HostReadable> Deref for USMSlice<T, M> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.data
    }
}

impl<T, M: MemMode + HostWritable + HostReadable> DerefMut for USMSlice<T, M> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Metadata-only `Debug` — doesn't print the Vec contents (could be
/// huge / sensitive) and doesn't require `T: Debug`.
impl<T, M: MemMode> fmt::Debug for USMSlice<T, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("USMSlice")
            .field("len", &self.data.len())
            .field("element_size", &std::mem::size_of::<T>())
            .finish_non_exhaustive()
    }
}

impl<T, M: MemMode> crate::record::RecordableBuffer for USMSlice<T, M> {
    fn record_handle(&self) -> crate::record::BufHandle {
        crate::record::BufHandle {
            mem: crate::record::MemRef::Svm(self.ptr() as *mut std::ffi::c_void),
            byte_len: self.data.len() * std::mem::size_of::<T>(),
        }
    }
}

impl<T, M: MemMode> KernelArg for USMSlice<T, M> {
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

impl<T, M: MemMode> Drop for USMSlice<T, M> {
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
