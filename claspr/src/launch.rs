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

use crate::buffer::DeviceSlice;
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
/// host-side counterpart of a `#[spirv(cross_workgroup)] &mut [T]`
/// kernel parameter.
///
/// Proc-macro–emitted kernel methods (e.g. `kernels.fill_u32(...)`)
/// are generic over `D: KernelSliceArg<T>` for each slice parameter,
/// so any buffer kind that ships data and a length to a SPIR-V
/// slice-decomposed kernel arg is usable interchangeably:
///
/// - [`crate::DeviceSlice<T>`] — `clCreateBuffer`-backed device
///   memory, the default.
/// - [`crate::MappedSlice<T>`] — coarse-grain SVM, when the device
///   supports it.
/// - [`crate::USMSlice<T>`] — fine-grain-system SVM wrapping a host
///   `Vec<T>`, when the device supports it.
///
/// The trait extends [`KernelArg`] (so the underlying
/// `clSetKernelArg` plumbing is reused) and is sealed — claspr owns
/// every impl. Users wanting a custom buffer-shaped argument should
/// open an issue rather than try to add an impl out-of-tree.
pub trait KernelSliceArg<T>: KernelArg + Send + 'static + kernel_slice_arg_sealed::Sealed {
    /// Number of elements in the underlying buffer. Reused by some
    /// chain combinators that need to size a downstream allocation
    /// from the upstream slice without re-fetching from OpenCL.
    fn element_count(&self) -> usize;
}

impl<T: Send + 'static, M: crate::access::MemMode> kernel_slice_arg_sealed::Sealed
    for DeviceSlice<T, M>
{
}
impl<T: Send + 'static, M: crate::access::MemMode> KernelSliceArg<T> for DeviceSlice<T, M> {
    fn element_count(&self) -> usize {
        crate::Buffer::len(self)
    }
}

impl<T: Send + 'static> kernel_slice_arg_sealed::Sealed for crate::MappedSlice<T> {}
impl<T: Send + 'static> KernelSliceArg<T> for crate::MappedSlice<T> {
    fn element_count(&self) -> usize {
        crate::Buffer::len(self)
    }
}

impl<T: Send + 'static> kernel_slice_arg_sealed::Sealed for crate::USMSlice<T> {}
impl<T: Send + 'static> KernelSliceArg<T> for crate::USMSlice<T> {
    fn element_count(&self) -> usize {
        crate::Buffer::len(self)
    }
}

// `Arc<DeviceSlice<T>>` — share one cl_mem across N parallel chain
// branches without re-uploading. Pair with [`claspr_async::Arced`]
// (built by `.arc()` on a `DeviceOperation` whose Output is
// `DeviceSlice<T>`): the chain produces `Arc<DeviceSlice<T>>`, each
// branch gets an `Arc::clone`, the kernel launcher accepts the Arc
// directly, the underlying `cl_mem` lives until the last clone drops
// (refcounted by `Arc` + lazy by OpenCL on top).
//
// `T: Sync` is needed because `Arc<DeviceSlice<T>>: Send` requires
// `DeviceSlice<T>: Send + Sync` which propagates `T: Sync`.
impl<T: Send + Sync + 'static, M: crate::access::MemMode> kernel_slice_arg_sealed::Sealed
    for Arc<DeviceSlice<T, M>>
{
}
impl<T: Send + Sync + 'static, M: crate::access::MemMode> KernelSliceArg<T>
    for Arc<DeviceSlice<T, M>>
{
    fn element_count(&self) -> usize {
        crate::Buffer::len(&**self)
    }
}

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
/// Returns `None` if either profiling counter is unavailable (e.g. the
/// command queue was created without `CL_QUEUE_PROFILING_ENABLE`, which
/// claspr always sets — so `None` here generally signals the device
/// dropped the profiling info).
pub fn profiling_duration(event: &Event) -> Option<Duration> {
    let start = event.profiling_command_start().ok()?;
    let end = event.profiling_command_end().ok()?;
    Some(Duration::from_nanos(end - start))
}
