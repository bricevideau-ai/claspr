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
use opencl3::command_queue::{
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE, CommandQueue,
};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::types::cl_command_queue_properties;
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
        CL_QUEUE_PROFILING_ENABLE
    }
}

impl QueueOrder for OutOfOrder {
    fn properties() -> cl_command_queue_properties {
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
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
    cl_queue: CommandQueue,
    ctx: Context,
}

// SAFETY: cl_command_queue is an opaque handle; OpenCL API calls on
// it are thread-safe per the spec (CL §3.4.1). The Context inside
// is itself Send + Sync.
unsafe impl Send for QueueInner {}
unsafe impl Sync for QueueInner {}

impl<O: QueueOrder> Clone for Queue<O> {
    fn clone(&self) -> Self {
        Queue {
            inner: Arc::clone(&self.inner),
            _order: PhantomData,
        }
    }
}

impl Queue<InOrder> {
    /// Create a profiling-enabled in-order queue on this context's
    /// device. The OpenCL spec guarantees commands run in submission
    /// order — no event juggling required.
    pub fn new(ctx: &Context) -> Result<Self> {
        Self::from_props(ctx)
    }
}

impl Queue<OutOfOrder> {
    /// Create a profiling-enabled out-of-order queue. Commands run
    /// when their explicit event dependencies are satisfied; the
    /// caller is responsible for the dependency graph.
    pub fn new(ctx: &Context) -> Result<Self> {
        Self::from_props(ctx)
    }
}

impl<O: QueueOrder> Queue<O> {
    fn from_props(ctx: &Context) -> Result<Self> {
        let cl_queue =
            CommandQueue::create_default_with_properties(ctx.raw_context(), O::properties(), 0)?;
        Ok(Queue {
            inner: Arc::new(QueueInner {
                cl_queue,
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
