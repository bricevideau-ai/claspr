//! Kernel launch surface — argument traits, work-size spec, and timing.
//!
//! ## Argument shape
//!
//! Every kernel parameter implements [`KernelArg`], which sets one or
//! more `clSetKernelArg` calls. claspr ships `KernelArg` impls for:
//!
//! - [`crate::buffer::DeviceSlice<T>`] — sets *two* args (data pointer +
//!   `usize` length), matching rust-gpu's slice decomposition for
//!   `&mut [T]` kernel parameters.
//! - All primitive integer and float types — passed by value via a
//!   single `clSetKernelArg`.
//! - [`LocalBuffer`] — declares an OpenCL `__local` memory allocation.
//! - [`crate::image::Image2DRgba8`] — sets the underlying `cl_mem`
//!   handle.
//!
//! User-defined `#[repr(C)] Copy` types opt in via [`crate::scalar_arg!`],
//! which emits both a [`ScalarArg`] marker and a `KernelArg` impl in
//! one line. The stage-3 proc-macro derive will replace this manual
//! step.
//!
//! ## Tuples
//!
//! [`KernelArgs`] is implemented for tuples of arity 0 through 8.
//! Pass kernel arguments as a typed tuple at the launch site:
//!
//! ```ignore
//! kernels.foo(&ctx, [n], &buf).wait()?;
//! kernels.foo(&ctx, [w, h], &buf, vp, max_iter).wait()?;
//! ```
//!
//! Wrong-arity launches are caught at `clEnqueueNDRangeKernel` time
//! by opencl3's internal arg-count check; wrong-typed launches still
//! lower to `set_arg` byte-equivalent calls (the kernel side can't see
//! Rust types). The stage-2 build-time codegen and stage-3 proc-macro
//! will tighten this to compile-time checks.

use crate::buffer::{DeviceSlice, Scalar};
use opencl3::event::Event;
use opencl3::kernel::ExecuteKernel;
use std::sync::Arc;
use std::time::Duration;

// ── KernelArg ─────────────────────────────────────────────────────────

/// A value that can be set as one or more positional `clSetKernelArg`
/// calls when launching a kernel.
///
/// The blanket impl on `&T` and `&mut T` lets users pass references at
/// the launch site without consuming buffers (`(&buf, vp)` not
/// `(buf, vp)`).
pub trait KernelArg {
    /// Set this argument on `exec`. Most impls call `exec.set_arg(...)`
    /// once; [`DeviceSlice`] sets twice (pointer + length).
    fn set(&self, exec: &mut ExecuteKernel<'_>);

    /// Called by `LaunchOp::into_event` *after* the enqueue returns,
    /// once per arg, with the kernel's completion event. Default
    /// no-op; argument types that need to track in-flight use of
    /// their underlying resource (today: [`crate::MappedSlice`],
    /// whose Drop needs the wait-list for `clEnqueueSVMFree`)
    /// override this to retain the event and store it on the arg.
    ///
    /// The reference is borrowed; impls that want to keep it past the
    /// call must call `opencl3::event::retain_event(event.get())` and
    /// wrap the raw handle in their own `Event` (or `Arc<Event>`).
    fn register_completion(&self, _event: &::opencl3::event::Event) {}
}

impl<T: KernelArg + ?Sized> KernelArg for &T {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        (**self).set(exec)
    }
    fn register_completion(&self, event: &::opencl3::event::Event) {
        (**self).register_completion(event)
    }
}

impl<T: KernelArg + ?Sized> KernelArg for &mut T {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        (**self).set(exec)
    }
    fn register_completion(&self, event: &::opencl3::event::Event) {
        (**self).register_completion(event)
    }
}

// ── KernelSliceArg ────────────────────────────────────────────────────

/// Sealing module — keeps `KernelSliceArg` impls in-crate so we can
/// change the trait or its bounds without breaking external callers.
mod kernel_slice_arg_sealed {
    pub trait Sealed {}
}

/// Marker + capability trait identifying a value usable as the
/// host-side counterpart of a `#[spirv(cross_workgroup)] &[T]`
/// kernel parameter (read-only slice).
///
/// Implemented by every buffer kind whose marker impls
/// [`crate::KernelReadable`] — currently every buffer marker
/// (`ReadWrite`, `ReadOnly`, `HostReadOnly`, `Frozen`,
/// `DeviceScratch`). Plus the wrapper types `Arc<DeviceSlice<T, M>>`
/// where `M: KernelReadable`.
///
/// Stronger sibling [`KernelSliceReadWriteArg<T>`] is the bound used
/// for `&mut [T]` kernel params — only markers that also impl
/// [`crate::KernelWritable`] satisfy it.
///
/// The trait extends [`KernelArg`] (so the underlying
/// `clSetKernelArg` plumbing is reused) and is sealed — claspr owns
/// every impl. Users wanting a custom buffer-shaped argument should
/// open an issue rather than try to add an impl out-of-tree.
pub trait KernelSliceReadArg<T>:
    KernelArg + KernelPointerArg + Send + 'static + kernel_slice_arg_sealed::Sealed
{
    /// Number of elements in the underlying buffer. Reused by some
    /// chain combinators that need to size a downstream allocation
    /// from the upstream slice without re-fetching from OpenCL.
    fn element_count(&self) -> usize;

    /// The recording handle (memory reference + byte length) for this buffer
    /// arg, used by the record/replay path to bake it into a recorded kernel
    /// launch. Every buffer family (`DeviceSlice` → cl_mem, `MappedSlice` /
    /// `USMSlice` → SVM pointer) overrides this; the default errors only for a
    /// hypothetical arg type that has no recordable memory.
    fn record_handle(&self) -> crate::Result<crate::record::BufHandle> {
        Err(crate::Error::NotSupported(
            "record: this kernel arg type has no recordable memory handle",
        ))
    }
}

/// A buffer that can set **only** its device pointer as a kernel arg
/// (advancing the arg index by exactly one, with no trailing length).
///
/// This is the eager/record counterpart of the slice's [`KernelArg::set`]
/// (which sets *two* args: pointer + `usize` length). It backs a
/// scalar-by-reference kernel parameter (`#[spirv(cross_workgroup)] &T` /
/// `&mut T`): rust-gpu lowers a scalar-ref to a bare pointer-to-scalar
/// `OpFunctionParameter` with **no** length operand, so the host must set
/// one arg slot, not two. A length-1
/// [`DeviceSlice`]/`MappedSlice`/`USMSlice` supplies the pointer.
///
/// [`KernelSliceReadArg`] requires this (every buffer that can be a slice
/// arg can also be a scalar-ref arg), so the proc-macro's `ScalarRef` arm
/// gets it for free on the same buffer generic. `KernelArg` is a
/// supertrait so [`ScalarRefArg`] can delegate `register_completion` to
/// the underlying buffer (e.g. `MappedSlice`'s SVM-free bookkeeping).
pub trait KernelPointerArg: KernelArg {
    /// Set this buffer's device pointer on `exec` as a single kernel arg.
    fn set_pointer_only(&self, exec: &mut ExecuteKernel<'_>);
}

/// Wrapper turning a borrowed [`KernelPointerArg`] buffer into a
/// pointer-only [`KernelArg`] — used by the proc-macro's `ScalarRef`
/// launch tuple so a scalar-ref param sets exactly one arg slot.
pub struct ScalarRefArg<'a, D: KernelPointerArg + ?Sized>(pub &'a D);

impl<D: KernelPointerArg + ?Sized> KernelArg for ScalarRefArg<'_, D> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        self.0.set_pointer_only(exec);
    }
    fn register_completion(&self, event: &::opencl3::event::Event) {
        // Delegate to the buffer's own post-enqueue bookkeeping (e.g.
        // MappedSlice retains the event for its Drop's SVM-free wait-list).
        KernelArg::register_completion(self.0, event);
    }
}

/// The kernel may both read and write through this slice arg — the
/// bound for `&mut [T]` kernel parameters.
///
/// Implemented for buffer kinds whose marker impls
/// [`crate::KernelWritable`] (`ReadWrite`, `HostReadOnly`,
/// `DeviceScratch`). Markers without write capability (`ReadOnly`,
/// `Frozen`) intentionally do not impl this — passing them to a
/// `&mut [T]` kernel param is a compile error.
///
/// Extends [`KernelSliceReadArg<T>`] (read access is a prerequisite
/// for read/write access) so users with one bound can rely on the
/// other.
pub trait KernelSliceReadWriteArg<T>: KernelSliceReadArg<T> {}

/// Legacy alias for [`KernelSliceReadWriteArg<T>`] — preserved so
/// proc-macro-emitted bounds keep compiling while we migrate the
/// macro to pick Read vs ReadWrite based on slice mutability.
pub trait KernelSliceArg<T>: KernelSliceReadWriteArg<T> {}
impl<T, X: KernelSliceReadWriteArg<T>> KernelSliceArg<T> for X {}

// ── KernelScalarRefArg — the scalar-by-reference kernel-arg bound ────
//
// rust-gpu lowers a `#[spirv(cross_workgroup)] &T` / `&mut T` kernel
// param to a bare pointer-to-scalar `OpFunctionParameter` (NO length
// operand), so the host must set exactly ONE arg slot (the pointer).
// This is the DEDICATED scalar-ref trait family — the type-fidelity
// half of #208. It is impl'd ONLY for `Scalar<B>` (the device-scalar
// wrapper — every memory tier via its backing `B`; plus its
// `Pipe`/`slot!`/`Checkout` via the scalar `ToInput`), NOT for the bare
// slice tiers. Conversely `Scalar<B>` does NOT impl the slice traits
// ([`KernelSliceReadArg`] etc.). The two exclusions together make the
// &T-arg / &[T]-arg mismatch a compile error in BOTH directions.
//
// Generic over the backing buffer `B`, which supplies the pointer via
// its own [`KernelPointerArg`] (`set_pointer_only`) — so ALL three
// memory tiers (`DeviceScalar`/`MappedScalar`/`USMScalar`, backed by
// `DeviceSlice`/`MappedSlice`/`USMSlice`) get scalar-ref support
// symmetric with their slice counterparts, no regression vs #205.

/// Sealing module — keeps the scalar-ref trait impls in-crate.
mod kernel_scalar_ref_sealed {
    pub trait Sealed {}
}

/// Marker + capability trait identifying a value usable as the
/// host-side counterpart of a `#[spirv(cross_workgroup)] &T` kernel
/// parameter (a **read** scalar-by-reference).
///
/// Implemented ONLY for [`Scalar<B>`] whose backing buffer `B` can set
/// a device pointer ([`KernelPointerArg`]) — i.e. the device-scalar
/// wrapper across every memory tier. A length-1 bare slice deliberately
/// does NOT satisfy it (that is the strict-binding half of #208). It
/// extends [`KernelPointerArg`] (so the pointer-only arg-set is reused)
/// and is sealed.
///
/// Stronger sibling [`KernelScalarRefMutArg<T>`] is the bound used for
/// `&mut T` kernel params — backings whose marker also impls
/// [`crate::KernelWritable`] satisfy it.
pub trait KernelScalarRefArg<T>:
    KernelArg + KernelPointerArg + Send + 'static + kernel_scalar_ref_sealed::Sealed
{
    /// The recording handle for this scalar's backing memory, used by
    /// the record/replay path (mirrors
    /// [`KernelSliceReadArg::record_handle`]).
    fn record_handle(&self) -> crate::Result<crate::record::BufHandle> {
        Err(crate::Error::NotSupported(
            "record: this scalar-ref arg type has no recordable memory handle",
        ))
    }
}

/// The kernel may both read and write through this scalar-ref arg — the
/// bound for `&mut T` kernel parameters. Implemented for [`Scalar<B>`]
/// whose backing impls [`KernelSliceReadWriteArg<T>`] (write-capable
/// marker). Extends [`KernelScalarRefArg<T>`].
pub trait KernelScalarRefMutArg<T>: KernelScalarRefArg<T> {}

// ── Scalar<B> impls (scalar-ref only, generic over the backing) ─────
//
// The read bound keys on `B: KernelSliceReadArg<T>` — every buffer that
// can be a read slice arg backs a read scalar-ref (its `record_handle`
// and pointer-only setter are reused). The mut bound keys on
// `B: KernelSliceReadWriteArg<T>`. This gives all three memory tiers
// (DeviceScalar/MappedScalar/USMScalar) scalar-ref support for free.

impl<B> kernel_scalar_ref_sealed::Sealed for Scalar<B> {}

impl<T, B> KernelScalarRefArg<T> for Scalar<B>
where
    B: KernelSliceReadArg<T>,
{
    fn record_handle(&self) -> crate::Result<crate::record::BufHandle> {
        KernelSliceReadArg::record_handle(&self.inner)
    }
}

impl<T, B> KernelScalarRefMutArg<T> for Scalar<B> where B: KernelSliceReadWriteArg<T> {}

// `Scalar<B>` sets exactly the backing's device pointer (one arg slot,
// no trailing length) — both via `KernelArg::set` (so `ScalarRefArg`'s
// `KernelPointerArg` supertrait and the Tier-1 launch tuple work) and
// via `set_pointer_only`. `register_completion` delegates to the
// backing so an SVM scalar (`MappedScalar`/`USMScalar`) records the
// completion event for its Drop's `clEnqueueSVMFree` wait-list.
impl<B: KernelPointerArg> KernelArg for Scalar<B> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        self.inner.set_pointer_only(exec);
    }
    fn register_completion(&self, event: &::opencl3::event::Event) {
        KernelArg::register_completion(&self.inner, event);
    }
}

impl<B: KernelPointerArg> KernelPointerArg for Scalar<B> {
    fn set_pointer_only(&self, exec: &mut ExecuteKernel<'_>) {
        self.inner.set_pointer_only(exec);
    }
}

// ── DeviceSlice<T, M> impls ────────────────────────────────────────

impl<T: Send + 'static, M: crate::access::MemMode> kernel_slice_arg_sealed::Sealed
    for DeviceSlice<T, M>
{
}
impl<T: Send + 'static, M: crate::access::MemMode + crate::access::KernelReadable>
    KernelSliceReadArg<T> for DeviceSlice<T, M>
{
    fn element_count(&self) -> usize {
        crate::Buffer::len(self)
    }
    fn record_handle(&self) -> crate::Result<crate::record::BufHandle> {
        use opencl3::memory::ClMem;
        Ok(crate::record::BufHandle {
            mem: crate::record::MemRef::Buffer(self.buffer().get()),
            byte_len: self.byte_len(),
        })
    }
}
impl<T: Send + 'static, M: crate::access::MemMode + crate::access::KernelReadable> KernelPointerArg
    for DeviceSlice<T, M>
{
    fn set_pointer_only(&self, exec: &mut ExecuteKernel<'_>) {
        // Pointer only — no trailing length (scalar-ref shape).
        unsafe {
            exec.set_arg(&*self.buffer);
        }
    }
}
impl<
    T: Send + 'static,
    M: crate::access::MemMode + crate::access::KernelReadable + crate::access::KernelWritable,
> KernelSliceReadWriteArg<T> for DeviceSlice<T, M>
{
}

// ── MappedSlice<T, M> impls ────────────────────────────────────────

impl<T: Send + 'static, M: crate::access::MemMode> kernel_slice_arg_sealed::Sealed
    for crate::MappedSlice<T, M>
{
}
impl<T: Send + 'static, M: crate::access::MemMode + crate::access::KernelReadable>
    KernelSliceReadArg<T> for crate::MappedSlice<T, M>
{
    fn element_count(&self) -> usize {
        crate::Buffer::len(self)
    }
    fn record_handle(&self) -> crate::Result<crate::record::BufHandle> {
        Ok(crate::record::BufHandle {
            mem: crate::record::MemRef::Svm(self.ptr() as *mut std::ffi::c_void),
            byte_len: crate::Buffer::len(self) * std::mem::size_of::<T>(),
        })
    }
}
impl<T: Send + 'static, M: crate::access::MemMode + crate::access::KernelReadable> KernelPointerArg
    for crate::MappedSlice<T, M>
{
    fn set_pointer_only(&self, exec: &mut ExecuteKernel<'_>) {
        // SVM pointer only — no trailing length (scalar-ref shape).
        unsafe {
            exec.set_arg_svm(self.ptr());
        }
    }
}
impl<
    T: Send + 'static,
    M: crate::access::MemMode + crate::access::KernelReadable + crate::access::KernelWritable,
> KernelSliceReadWriteArg<T> for crate::MappedSlice<T, M>
{
}

// ── USMSlice<T, M> impls ───────────────────────────────────────────

impl<T: Send + 'static, M: crate::access::MemMode> kernel_slice_arg_sealed::Sealed
    for crate::USMSlice<T, M>
{
}
impl<T: Send + 'static, M: crate::access::MemMode + crate::access::KernelReadable>
    KernelSliceReadArg<T> for crate::USMSlice<T, M>
{
    fn element_count(&self) -> usize {
        crate::Buffer::len(self)
    }
    fn record_handle(&self) -> crate::Result<crate::record::BufHandle> {
        Ok(crate::record::BufHandle {
            mem: crate::record::MemRef::Svm(self.ptr() as *mut std::ffi::c_void),
            byte_len: crate::Buffer::len(self) * std::mem::size_of::<T>(),
        })
    }
}
impl<T: Send + 'static, M: crate::access::MemMode + crate::access::KernelReadable> KernelPointerArg
    for crate::USMSlice<T, M>
{
    fn set_pointer_only(&self, exec: &mut ExecuteKernel<'_>) {
        // SVM pointer only — no trailing length (scalar-ref shape).
        unsafe {
            exec.set_arg_svm(self.ptr());
        }
    }
}
impl<
    T: Send + 'static,
    M: crate::access::MemMode + crate::access::KernelReadable + crate::access::KernelWritable,
> KernelSliceReadWriteArg<T> for crate::USMSlice<T, M>
{
}

// `Arc<DeviceSlice<T, M>>` — share one cl_mem across N parallel chain
// branches without re-uploading. Pair with [`crate::Arced`]
// (built by `.arc()` on a `DeviceOp` whose Output is
// `DeviceSlice<T, M>`): the chain produces `Arc<DeviceSlice<T, M>>`,
// each branch gets an `Arc::clone`, the kernel launcher accepts the
// Arc directly as a **read** kernel-arg slot, the underlying
// `cl_mem` lives until the last clone drops (refcounted by `Arc` +
// lazy by OpenCL on top).
//
// **Read-only by design.** `Arc` exists here for diamond/fan-out
// sharing of *inputs* — letting two parallel kernels write through
// clones of the same Arc would be a host-side data race the
// borrow checker can no longer catch (`Arc::clone` gives shared
// access, not exclusive). The single-writer guarantee for writable
// kernel slots stays with `DeviceSlice<T, M>` directly: each
// launcher takes ownership of the slice via the move-in/move-out
// `Op::Output` chain, and an owned value can't simultaneously be
// in two launchers. Write-then-share patterns: write through the
// owned DeviceSlice first, then `.arc()` once writes are done to
// hand the buffer out for read sharing.
//
// Historical note: pre-`f80202c` (the Read/Write trait split), the
// catch-all `KernelSliceArg<T>` for `Arc<DeviceSlice>` was added in
// `480bcba` ("true diamond sharing") explicitly for read-only fan-
// out. The split commit mechanically gave Arc both Read and
// ReadWrite variants, preserving the bug-of-omission — every
// existing use of Arc in tests + examples + the combinator spike
// is read-only. The Write impl was reachable but never exercised;
// removing it restores the original design intent and the borrow
// checker's single-writer story.
//
// `T: Sync` is needed because `Arc<DeviceSlice<T, M>>: Send` requires
// `DeviceSlice<T, M>: Send + Sync` which propagates `T: Sync`.
impl<T: Send + Sync + 'static, M: crate::access::MemMode> kernel_slice_arg_sealed::Sealed
    for Arc<DeviceSlice<T, M>>
{
}
impl<T: Send + Sync + 'static, M: crate::access::MemMode + crate::access::KernelReadable>
    KernelSliceReadArg<T> for Arc<DeviceSlice<T, M>>
{
    fn element_count(&self) -> usize {
        crate::Buffer::len(&**self)
    }
    fn record_handle(&self) -> crate::Result<crate::record::BufHandle> {
        (**self).record_handle()
    }
}
impl<T: Send + Sync + 'static, M: crate::access::MemMode + crate::access::KernelReadable>
    KernelPointerArg for Arc<DeviceSlice<T, M>>
{
    fn set_pointer_only(&self, exec: &mut ExecuteKernel<'_>) {
        (**self).set_pointer_only(exec);
    }
}
// Deliberately no `KernelSliceReadWriteArg` impl — see comment above.

// `DeviceSlice` lives in `buffer`, but its `KernelArg` impl belongs
// here with the rest of the launch surface so `set_arg` plumbing is
// in one place. Decomposes into the `(buffer, len)` pair that
// rust-gpu's slice param expects.
impl<T, M: crate::access::MemMode> KernelArg for DeviceSlice<T, M> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let len: usize = self.len;
        unsafe {
            exec.set_arg(&*self.buffer).set_arg(&len);
        }
    }
    // No `register_completion` override: `cl_mem` Drop is lazy /
    // refcount-based, so the runtime defers actual deletion until
    // in-flight commands using the buffer finish. No host-side
    // bookkeeping needed.
}

// Arc<DeviceSlice<T>> just delegates to the inner DeviceSlice. The
// blanket `impl<T: KernelArg> KernelArg for &T` then makes
// `&Arc<DeviceSlice<T>>` (which is what the proc-macro-emitted
// launcher hands to LaunchOp) a `KernelArg` too.
impl<T, M: crate::access::MemMode> KernelArg for Arc<DeviceSlice<T, M>> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        (**self).set(exec);
    }
    // No `register_completion` override — DeviceSlice doesn't have one
    // either (cl_mem lifetime is refcounted by the OpenCL runtime).
}

// ── ScalarArg + scalar_arg! macro ─────────────────────────────────────

/// Marker trait for kernel argument types passed by value via a single
/// `clSetKernelArg` call.
///
/// claspr provides impls for every primitive integer and float type.
/// User `#[repr(C)] Copy` structs opt in via the [`crate::scalar_arg!`] macro,
/// which emits this marker plus the matching [`KernelArg`] impl in one
/// statement.
///
/// The marker exists so stage 3's `#[claspr::kernel]` derive has a
/// single trait to target, and so reviewers can grep for "what
/// types are valid by-value kernel args."
pub trait ScalarArg: Copy + 'static {}

/// Re-exports used by the [`crate::scalar_arg!`] macro. Not part of the public
/// API — the macro reaches in via `$crate::__macro_support`.
#[doc(hidden)]
pub mod __macro_support {
    pub use opencl3::kernel::ExecuteKernel;
}

/// Implement [`ScalarArg`] and [`KernelArg`] for one or more
/// `#[repr(C)] Copy` types in one statement.
///
/// ```ignore
/// #[repr(C)]
/// #[derive(Copy, Clone)]
/// struct Viewport { width: u32, height: u32, /* ... */ }
///
/// claspr::scalar_arg!(Viewport);
/// ```
///
/// The expansion is exactly:
/// ```ignore
/// impl claspr::ScalarArg for Viewport {}
/// impl claspr::KernelArg for Viewport {
///     fn set(&self, exec: &mut ExecuteKernel<'_>) {
///         unsafe { exec.set_arg(self); }
///     }
/// }
/// ```
///
/// Soundness is on the caller — the bytes of the value are passed
/// verbatim to the OpenCL runtime, so the type must match what the
/// kernel expects (`#[repr(C)]`, no padding mismatch, etc.).
#[macro_export]
macro_rules! scalar_arg {
    ($($t:ty),* $(,)?) => {
        $(
            impl $crate::ScalarArg for $t {}
            impl $crate::KernelArg for $t {
                fn set(
                    &self,
                    exec: &mut $crate::launch::__macro_support::ExecuteKernel<'_>,
                ) {
                    unsafe { exec.set_arg(self); }
                }
            }
        )*
    };
}

// Built-in scalar types. Anything `clSetKernelArg` accepts as a
// fixed-size by-value argument.
scalar_arg!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);

// ── LocalBuffer ───────────────────────────────────────────────────────

/// A `__local` memory allocation declared as a kernel parameter.
///
/// OpenCL allocates `size_bytes` bytes of workgroup-local memory per
/// workgroup when the kernel is launched. Use this when the kernel
/// declares a workgroup-memory parameter explicitly (rather than
/// declaring it module-scope via `#[spirv(workgroup)]`).
pub struct LocalBuffer {
    /// Size of the local allocation in bytes.
    pub size_bytes: usize,
}

impl LocalBuffer {
    /// Allocate `size_bytes` of `__local` memory.
    pub fn bytes(size_bytes: usize) -> Self {
        Self { size_bytes }
    }

    /// Allocate space for `count` elements of type `T`.
    pub fn of<T>(count: usize) -> Self {
        Self {
            size_bytes: count * std::mem::size_of::<T>(),
        }
    }
}

impl KernelArg for LocalBuffer {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        unsafe {
            exec.set_arg_local_buffer(self.size_bytes);
        }
    }
}

// ── KernelArgs (tuple trait) ──────────────────────────────────────────

/// A tuple of [`KernelArg`]s, set in order on launch.
///
/// Implemented for tuples up to arity 8. Pass an empty tuple `()` for
/// kernels with no arguments.
pub trait KernelArgs {
    /// Set every element of the tuple in order.
    fn set_all(&self, exec: &mut ExecuteKernel<'_>);

    /// Call [`KernelArg::register_completion`] on every element with
    /// the just-enqueued completion `event`. `LaunchOp::into_event`
    /// invokes this after enqueue so args like `MappedSlice<T>` can
    /// record the event for their Drop's `clEnqueueSVMFree` wait-list.
    fn register_all(&self, event: &::opencl3::event::Event);
}

impl KernelArgs for () {
    fn set_all(&self, _: &mut ExecuteKernel<'_>) {}
    fn register_all(&self, _: &::opencl3::event::Event) {}
}

macro_rules! impl_kernel_args_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name: KernelArg),+> KernelArgs for ($($name,)+) {
            #[allow(non_snake_case)]
            fn set_all(&self, exec: &mut ExecuteKernel<'_>) {
                let ($($name,)+) = self;
                $( $name.set(exec); )+
            }
            #[allow(non_snake_case)]
            fn register_all(&self, event: &::opencl3::event::Event) {
                let ($($name,)+) = self;
                $( $name.register_completion(event); )+
            }
        }
    };
}

impl_kernel_args_tuple!(A);
impl_kernel_args_tuple!(A, B);
impl_kernel_args_tuple!(A, B, C);
impl_kernel_args_tuple!(A, B, C, D);
impl_kernel_args_tuple!(A, B, C, D, E);
impl_kernel_args_tuple!(A, B, C, D, E, F);
impl_kernel_args_tuple!(A, B, C, D, E, F, G);
impl_kernel_args_tuple!(A, B, C, D, E, F, G, H);

// ── LaunchSpec + IntoLaunchSpec ───────────────────────────────────────

/// Work-item geometry for a single kernel launch.
///
/// OpenCL kernels run over a 1D, 2D, or 3D index space. `LaunchSpec`
/// stores the global size (always present) and an optional local
/// (workgroup) size.
///
/// Most users never name this type — `Context::launch` accepts anything
/// that implements [`IntoLaunchSpec`], which covers `[N; 1]`/`[N; 2]`/
/// `[N; 3]` for global-only and the `(global, local)` tuple form for
/// the local-size case.
#[derive(Clone, Copy, Debug)]
pub struct LaunchSpec {
    global: [usize; 3],
    local: Option<[usize; 3]>,
    dims: u8,
}

impl LaunchSpec {
    /// Number of dimensions (1, 2, or 3).
    pub fn dims(&self) -> u8 {
        self.dims
    }

    /// Global work size as a slice of length `self.dims()`.
    pub fn global(&self) -> &[usize] {
        &self.global[..self.dims as usize]
    }

    /// Local work size as a slice, or `None` if the runtime should
    /// pick.
    pub fn local(&self) -> Option<&[usize]> {
        self.local.as_ref().map(|l| &l[..self.dims as usize])
    }
}

/// Anything that can be turned into a [`LaunchSpec`].
///
/// claspr provides impls for the four common shapes:
///
/// - `[usize; 1]` / `[usize; 2]` / `[usize; 3]` — global size only,
///   runtime picks the local size.
/// - `([usize; D], [usize; D])` — global and local sizes, same
///   dimensionality.
pub trait IntoLaunchSpec {
    /// Build the launch spec.
    fn into_launch_spec(self) -> LaunchSpec;
}

impl IntoLaunchSpec for LaunchSpec {
    fn into_launch_spec(self) -> LaunchSpec {
        self
    }
}

// `From` conversions for the global-only shapes, so a `LaunchSpec` value is easy
// to mint where one is needed by value (e.g. binding a `slot!(Grid)` whose
// `Tag::Value = LaunchSpec`: `Grid(LaunchSpec::from([N]))`). These mirror the
// `IntoLaunchSpec` array impls.
impl From<[usize; 1]> for LaunchSpec {
    fn from(g: [usize; 1]) -> Self {
        g.into_launch_spec()
    }
}
impl From<[usize; 2]> for LaunchSpec {
    fn from(g: [usize; 2]) -> Self {
        g.into_launch_spec()
    }
}
impl From<[usize; 3]> for LaunchSpec {
    fn from(g: [usize; 3]) -> Self {
        g.into_launch_spec()
    }
}

impl IntoLaunchSpec for [usize; 1] {
    fn into_launch_spec(self) -> LaunchSpec {
        LaunchSpec {
            global: [self[0], 0, 0],
            local: None,
            dims: 1,
        }
    }
}

impl IntoLaunchSpec for [usize; 2] {
    fn into_launch_spec(self) -> LaunchSpec {
        LaunchSpec {
            global: [self[0], self[1], 0],
            local: None,
            dims: 2,
        }
    }
}

impl IntoLaunchSpec for [usize; 3] {
    fn into_launch_spec(self) -> LaunchSpec {
        LaunchSpec {
            global: self,
            local: None,
            dims: 3,
        }
    }
}

impl IntoLaunchSpec for ([usize; 1], [usize; 1]) {
    fn into_launch_spec(self) -> LaunchSpec {
        LaunchSpec {
            global: [self.0[0], 0, 0],
            local: Some([self.1[0], 0, 0]),
            dims: 1,
        }
    }
}

impl IntoLaunchSpec for ([usize; 2], [usize; 2]) {
    fn into_launch_spec(self) -> LaunchSpec {
        LaunchSpec {
            global: [self.0[0], self.0[1], 0],
            local: Some([self.1[0], self.1[1], 0]),
            dims: 2,
        }
    }
}

impl IntoLaunchSpec for ([usize; 3], [usize; 3]) {
    fn into_launch_spec(self) -> LaunchSpec {
        LaunchSpec {
            global: self.0,
            local: Some(self.1),
            dims: 3,
        }
    }
}

// ── Profiling ─────────────────────────────────────────────────────────

/// Wall-clock kernel runtime as reported by OpenCL command profiling.
///
/// Returns `None` if either profiling counter is unavailable. The common
/// case is that profiling was not enabled: it is opt-in (default off), so the
/// queue is created without `CL_QUEUE_PROFILING_ENABLE` unless you build the
/// context with [`ContextBuilder::profiling(true)`](crate::context::ContextBuilder::profiling).
pub fn profiling_duration(event: &Event) -> Option<Duration> {
    let start = event.profiling_command_start().ok()?;
    let end = event.profiling_command_end().ok()?;
    Some(Duration::from_nanos(end - start))
}
