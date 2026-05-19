//! [`Queue<Order>`] and the [`Launcher`] trait.
//!
//! `Queue` is the OpenCL command queue, type-stated on its ordering:
//! [`InOrder`] (commands serialise w.r.t. submission order) or
//! [`OutOfOrder`] (commands run as soon as their event dependencies
//! resolve, in any order).
//!
//! [`Launcher`] is what the proc-macro keys off when generating the
//! per-kernel launch wrapper — it abstracts over `&Context` (which
//! delegates to the bundled default in-order queue) and `&Queue<_>`
//! directly, so trivial code stays trivial and advanced users get
//! explicit queue control.

use crate::context::Context;
use crate::error::Result;
use crate::launch::{IntoLaunchSpec, KernelArgs};
use opencl3::command_queue::{CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CommandQueue};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::types::{cl_command_queue_properties, cl_event};
use std::marker::PhantomData;
use std::sync::Arc;

// ── Order markers ────────────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::InOrder {}
    impl Sealed for super::OutOfOrder {}
}

/// Sealed marker trait for queue ordering. Implementors are
/// [`InOrder`] and [`OutOfOrder`] only.
pub trait QueueOrder: sealed::Sealed {
    #[doc(hidden)]
    fn properties() -> cl_command_queue_properties;
}

/// In-order queue marker: commands serialise w.r.t. submission order.
/// The default — most user code wants this.
#[derive(Clone, Copy, Debug)]
pub struct InOrder;

/// Out-of-order queue marker: commands run when their event
/// dependencies resolve, in any order. Caller manages dependencies
/// explicitly via events.
#[derive(Clone, Copy, Debug)]
pub struct OutOfOrder;

impl QueueOrder for InOrder {
    fn properties() -> cl_command_queue_properties {
        // Profiling is opt-in (SYCL `property::queue::enable_profiling`
        // pattern). The eventual `Queue::builder()` will expose it.
        0
    }
}

impl QueueOrder for OutOfOrder {
    fn properties() -> cl_command_queue_properties {
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    }
}

// ── Queue ────────────────────────────────────────────────────────────

/// An OpenCL command queue with a compile-time ordering tag.
///
/// Cheap to clone — `Queue<O>` is `Arc<QueueInner>` internally.
/// The `Context` inside is itself cheap to clone (also Arc). So
/// `Queue<O>: Clone` is two atomic increments and that's it.
pub struct Queue<O: QueueOrder> {
    inner: Arc<QueueInner>,
    _order: PhantomData<O>,
}

struct QueueInner {
    /// `ManuallyDrop` so opencl3's `CommandQueue::drop` (which panics
    /// on release failure) doesn't fire — our own [`Drop`] impl
    /// below calls `release_command_queue` and records into the
    /// context's sticky-error counter on failure instead.
    cl_queue: std::mem::ManuallyDrop<CommandQueue>,
    ctx: Context,
}

// SAFETY: cl_command_queue is an opaque handle; OpenCL API calls on
// it are thread-safe per the spec (CL §3.4.1). The Context inside
// is itself Send + Sync.
unsafe impl Send for QueueInner {}
unsafe impl Sync for QueueInner {}

impl Drop for QueueInner {
    fn drop(&mut self) {
        // SAFETY: opencl3's `CommandQueue` holds the cl_command_queue
        // we created in `from_props` / `from_props_on`; release
        // exactly once now.
        let raw = self.cl_queue.get();
        let res = unsafe { opencl3::command_queue::release_command_queue(raw) };
        if res.is_err() {
            self.ctx.record_err();
        }
    }
}

impl<O: QueueOrder> Clone for Queue<O> {
    fn clone(&self) -> Self {
        Queue {
            inner: Arc::clone(&self.inner),
            _order: PhantomData,
        }
    }
}

impl Queue<InOrder> {
    /// Create an in-order queue on this context's default device
    /// (`ctx.device()`). The OpenCL spec guarantees commands run
    /// in submission order — no event juggling required.
    pub fn new(ctx: &Context) -> Result<Self> {
        Self::from_props(ctx)
    }

    /// Create an in-order queue pinned to a specific device. Use
    /// this for multi-device contexts where you need queues on
    /// devices other than the default. `device` must be one of the
    /// devices the context was built with (see
    /// [`Context::for_devices`](crate::Context::for_devices)).
    pub fn on_device(ctx: &Context, device: &crate::Device) -> Result<Self> {
        Self::from_props_on(ctx, device)
    }
}

impl Queue<OutOfOrder> {
    /// Create an out-of-order queue on this context's default
    /// device. Commands run when their explicit event dependencies
    /// are satisfied; the caller is responsible for the dependency
    /// graph.
    pub fn new(ctx: &Context) -> Result<Self> {
        Self::from_props(ctx)
    }

    /// Create an out-of-order queue pinned to a specific device.
    /// See [`Queue::<InOrder>::on_device`].
    pub fn on_device(ctx: &Context, device: &crate::Device) -> Result<Self> {
        Self::from_props_on(ctx, device)
    }
}

// ── IntoEventList ──────────────────────────────────────────────────

/// Convert a value into a `Vec<cl_event>` for OpenCL wait-list
/// arguments. Used by [`Queue<OutOfOrder>::launch_with_deps`] so
/// callers can pass dependencies in whatever shape fits the
/// situation — a single event, an array, a borrowed slice, etc.
///
/// Standard impls:
/// - `()` — no dependencies (empty list)
/// - `&Event` — one event
/// - `[&Event; N]` (any const N) — fixed-size array of borrowed events
/// - `&[Event]` — borrowed slice
/// - `Vec<Event>` — owned vector
pub trait IntoEventList {
    fn into_event_list(self) -> Vec<cl_event>;
}

impl IntoEventList for () {
    fn into_event_list(self) -> Vec<cl_event> {
        Vec::new()
    }
}

impl IntoEventList for &Event {
    fn into_event_list(self) -> Vec<cl_event> {
        vec![self.get()]
    }
}

impl<const N: usize> IntoEventList for [&Event; N] {
    fn into_event_list(self) -> Vec<cl_event> {
        self.iter().map(|e| e.get()).collect()
    }
}

impl IntoEventList for &[Event] {
    fn into_event_list(self) -> Vec<cl_event> {
        self.iter().map(|e| e.get()).collect()
    }
}

impl IntoEventList for Vec<Event> {
    fn into_event_list(self) -> Vec<cl_event> {
        self.iter().map(|e| e.get()).collect()
    }
}

impl<O: QueueOrder> Queue<O> {
    fn from_props(ctx: &Context) -> Result<Self> {
        let cl_queue =
            CommandQueue::create_default_with_properties(ctx.raw_context(), O::properties(), 0)?;
        Ok(Queue {
            inner: Arc::new(QueueInner {
                cl_queue: std::mem::ManuallyDrop::new(cl_queue),
                ctx: ctx.clone(),
            }),
            _order: PhantomData,
        })
    }

    fn from_props_on(ctx: &Context, device: &crate::Device) -> Result<Self> {
        // SAFETY: `device` must belong to `ctx` (per the docs on
        // `on_device`). opencl3 marks this unsafe because the
        // contract is per-call; we rely on the caller respecting it.
        let cl_queue = unsafe {
            CommandQueue::create_with_properties(
                ctx.raw_context(),
                device.raw_id(),
                O::properties(),
                0,
            )?
        };
        Ok(Queue {
            inner: Arc::new(QueueInner {
                cl_queue: std::mem::ManuallyDrop::new(cl_queue),
                ctx: ctx.clone(),
            }),
            _order: PhantomData,
        })
    }

    /// Block until every previously submitted command on this queue
    /// has finished. Equivalent to `clFinish`.
    pub fn finish(&self) -> Result<()> {
        self.inner.cl_queue.finish()?;
        Ok(())
    }

    /// Borrow the underlying opencl3 command queue. Escape hatch for
    /// any operation [`Launcher`] doesn't expose yet.
    pub fn raw(&self) -> &CommandQueue {
        &self.inner.cl_queue
    }

    /// Borrow the context this queue belongs to.
    pub fn context(&self) -> &Context {
        &self.inner.ctx
    }
}

// ── Launcher trait ───────────────────────────────────────────────────

/// What the proc-macro-generated launch wrappers key off.
///
/// Implemented for `Context` (uses the bundled default in-order
/// queue) and `Queue<_>` (uses the queue directly). Most users
/// only see this trait in the signature `&impl Launcher` of a
/// generated `kernels.foo(...)` method — passing either `&ctx` or
/// `&queue` works.
///
/// Default-impl methods cover the synchronous path: launch a kernel
/// and wait. Out-of-order users who want non-blocking event chaining
/// call inherent `Queue<OutOfOrder>` methods directly.
pub trait Launcher {
    /// The OpenCL command queue this launcher will enqueue on.
    fn cl_queue(&self) -> &CommandQueue;

    /// The OpenCL context behind the queue. Needed by buffer
    /// constructors that allocate on the context (e.g.
    /// `clCreateBuffer`).
    fn context(&self) -> &Context;

    /// Launch a kernel synchronously and return its profiling event.
    ///
    /// `spec` accepts `[N]`, `[W, H]`, `[X, Y, Z]`, or
    /// `(global, local)` tuples — anything implementing
    /// [`IntoLaunchSpec`]. `args` is a typed tuple of
    /// [`KernelArg`](crate::launch::KernelArg)s, set in declaration order.
    fn launch<S, A>(&self, kernel: &Kernel, spec: S, args: A) -> Result<Event>
    where
        S: IntoLaunchSpec,
        A: KernelArgs,
    {
        let mut exec = ExecuteKernel::new(kernel);
        args.set_all(&mut exec);
        let spec = spec.into_launch_spec();
        exec.set_global_work_sizes(spec.global());
        if let Some(local) = spec.local() {
            exec.set_local_work_sizes(local);
        }
        // SAFETY: opencl3's `enqueue_nd_range` is `unsafe` because it
        // doesn't validate that the argument types passed to the
        // kernel match the kernel's actual signature. claspr's typed
        // wrapper (the proc-macro emits matched `&DeviceSlice<T>`,
        // `&Image2DRgba8`, etc.) is what makes this call safe in
        // practice.
        let event = unsafe { exec.enqueue_nd_range(self.cl_queue())? };
        event.wait()?;
        Ok(event)
    }
}

impl<O: QueueOrder> Launcher for Queue<O> {
    fn cl_queue(&self) -> &CommandQueue {
        &self.inner.cl_queue
    }

    fn context(&self) -> &Context {
        &self.inner.ctx
    }
}

// `impl Launcher for &L` so the macro-generated `&impl Launcher`
// signature accepts `&Queue<_>` references directly without an
// extra level of indirection at the call site.
impl<L: Launcher + ?Sized> Launcher for &L {
    fn cl_queue(&self) -> &CommandQueue {
        (**self).cl_queue()
    }

    fn context(&self) -> &Context {
        (**self).context()
    }
}

// ── LauncherAsync trait ──────────────────────────────────────────────

/// What the proc-macro's `_async` launch wrappers key off.
///
/// Implemented only by [`Queue<OutOfOrder>`] (and references to it):
/// in-order launchers don't expose an explicit dependency-list path
/// because their `clEnqueue*` calls already serialise against the
/// previous command on the same queue.
///
/// The proc-macro emits two methods per kernel — `kernels.foo(...)`
/// (sync, takes `&impl Launcher`, blocks on the event) and
/// `kernels.foo_async(launcher, deps, ...)` (async, takes
/// `&impl LauncherAsync`, returns the event without waiting). The
/// async path is what users compose into a DAG.
pub trait LauncherAsync: Launcher {
    /// Out-of-order launch: enqueue the kernel after `deps` complete,
    /// return the resulting [`Event`] *without blocking*. The caller
    /// composes dependency graphs by feeding returned events into
    /// later `launch_with_deps` calls.
    ///
    /// `deps` accepts any [`IntoEventList`]: `()`, `&Event`,
    /// `[&Event; N]`, `&[Event]`, `Vec<Event>`, etc.
    ///
    /// For the synchronous fire-and-wait path (matching the
    /// [`Launcher::launch`] default impl), call [`Launcher::launch`]
    /// instead — same launcher, no `deps` argument, blocks.
    fn launch_with_deps<D, S, A>(&self, deps: D, kernel: &Kernel, spec: S, args: A) -> Result<Event>
    where
        D: IntoEventList,
        S: IntoLaunchSpec,
        A: KernelArgs,
    {
        let mut exec = ExecuteKernel::new(kernel);
        args.set_all(&mut exec);
        let spec = spec.into_launch_spec();
        exec.set_global_work_sizes(spec.global());
        if let Some(local) = spec.local() {
            exec.set_local_work_sizes(local);
        }
        let wait = deps.into_event_list();
        if !wait.is_empty() {
            exec.set_event_wait_list(&wait);
        }
        // SAFETY: see Launcher::launch — claspr's typed wrappers
        // are what make this call safe. Unlike Launcher::launch,
        // we do not block on the returned event; the caller chains
        // it explicitly.
        let event = unsafe { exec.enqueue_nd_range(self.cl_queue())? };
        Ok(event)
    }
}

impl LauncherAsync for Queue<OutOfOrder> {}

// Same blanket as for `Launcher` — accepting `&Queue<OutOfOrder>` in
// `&impl LauncherAsync` parameters needs the reference-through impl.
impl<L: LauncherAsync + ?Sized> LauncherAsync for &L {}
