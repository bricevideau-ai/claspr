//! [`Context`] — the OpenCL context wrapper, Arc-shared internally
//! so it's cheap to clone and `Send + Sync`.
//!
//! Each `Context` carries a bundled in-order command queue used
//! as the default launch path. Most user code
//! never names a separate [`Queue`](crate::queue::Queue) — passing
//! `&ctx` to a generated launch wrapper routes through the default
//! queue. Advanced callers create explicit `Queue<InOrder>` /
//! `Queue<OutOfOrder>` handles when they need separate command
//! streams or out-of-order semantics.

use crate::device::Device;
use crate::error::Result;
use crate::queue::Launcher;
use opencl3::command_queue::CommandQueue;
use opencl3::kernel::Kernel;
use opencl3::program::Program;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// OpenCL context, the device it's pinned to, and the default
/// in-order command queue.
///
/// Cheap to clone (one `Arc` increment). `Send + Sync`. The OpenCL
/// ICD does its own internal refcount under the hood; this Rust
/// `Arc` is the only per-process refcount we add.
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

struct ContextInner {
    cl_context: opencl3::context::Context,
    device: Device,
    /// In-order queue (no profiling — see `for_device`). Used by
    /// the [`Launcher`] impl when the user hands `&ctx` to a
    /// kernel call.
    default_cl_queue: CommandQueue,
    /// Sticky-error counter. `Drop` impls that discover an OpenCL
    /// release failure can't propagate it; they bump this instead so
    /// callers who care can audit via [`Context::error_count`].
    error_state: AtomicU32,
}

// SAFETY: cl_context, cl_command_queue, and cl_device_id are opaque
// handles. OpenCL API calls on them are thread-safe per the spec
// (CL §3.4.1).
unsafe impl Send for ContextInner {}
unsafe impl Sync for ContextInner {}

impl Context {
    /// Build a context pinned to `device`, with an in-order default
    /// queue. The queue has no extra properties enabled — profiling
    /// is opt-in via the `Queue` builder (matches SYCL 2020's
    /// `property::queue::enable_profiling` semantics).
    pub fn for_device(device: &Device) -> Result<Self> {
        let cl_context = opencl3::context::Context::from_device(&device.cl3())?;
        let default_cl_queue = CommandQueue::create_default_with_properties(&cl_context, 0, 0)?;
        Ok(Context {
            inner: Arc::new(ContextInner {
                cl_context,
                device: device.clone(),
                default_cl_queue,
                error_state: AtomicU32::new(0),
            }),
        })
    }

    /// Pick the first available device of any type and build a
    /// context on it. Convenience for trivial single-device setups.
    pub fn any() -> Result<Self> {
        Self::for_device(&Device::any()?)
    }

    /// Pick a device with a SYCL-style scoring closure (highest
    /// score wins, negative excludes) and build a context on it.
    pub fn select<F>(score: F) -> Result<Self>
    where
        F: FnMut(&Device) -> i32,
    {
        Self::for_device(&Device::find(score)?)
    }

    /// The device this context is pinned to.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Borrow the underlying [`opencl3::context::Context`] for
    /// operations claspr doesn't surface yet.
    pub fn raw_context(&self) -> &opencl3::context::Context {
        &self.inner.cl_context
    }

    /// Borrow the default in-order command queue. The [`Launcher`]
    /// impl for `&Context` routes through this.
    pub fn raw_default_queue(&self) -> &CommandQueue {
        &self.inner.default_cl_queue
    }

    /// How many sticky errors have been recorded by Drop impls.
    /// Always zero unless an OpenCL release call failed during
    /// teardown — fault accumulator pattern from cuda-oxide.
    pub fn error_count(&self) -> u32 {
        self.inner.error_state.load(Ordering::Relaxed)
    }

    /// Bump the sticky-error counter. Called from `Drop` impls in
    /// dependent types when a release fails and the error can't be
    /// propagated. (Not yet wired — neither `QueueInner` nor
    /// `DeviceSlice` have fallible drop today; this stays
    /// `pub(crate)` for the buffer/queue Drop work in stage 2.)
    #[allow(dead_code)]
    pub(crate) fn record_err(&self) {
        self.inner.error_state.fetch_add(1, Ordering::Relaxed);
    }

    // ── Program / kernel ─────────────────────────────────────────────

    /// Create + build an OpenCL program from raw SPIR-V bytes.
    /// Returns the build log on failure.
    pub fn build_program(&self, spv_bytes: &[u8]) -> Result<Program> {
        let mut program = Program::create_from_il(&self.inner.cl_context, spv_bytes)
            .map_err(|e| crate::Error::Other(format!("create_from_il: {e}")))?;
        if let Err(e) = program.build(self.inner.cl_context.devices(), "") {
            let log = program
                .get_build_log(self.inner.device.raw_id())
                .unwrap_or_else(|_| "no build log".into());
            return Err(crate::Error::Build {
                log: format!("{e}\n{log}"),
            });
        }
        Ok(program)
    }

    /// Look up a kernel by entry-point name in a built program.
    pub fn kernel(&self, program: &Program, name: &str) -> Result<Kernel> {
        Kernel::create(program, name)
            .map_err(|e| crate::Error::Other(format!("Kernel::create({name}): {e}")))
    }

    /// Convenience: [`build_program`](Self::build_program) +
    /// [`kernel`](Self::kernel) in one call. The intermediate
    /// `Program` is dropped — OpenCL refcounts it internally and
    /// the kernel keeps it alive.
    pub fn kernel_from_spv(&self, spv_bytes: &[u8], name: &str) -> Result<Kernel> {
        let program = self.build_program(spv_bytes)?;
        self.kernel(&program, name)
    }
}

// ── Launcher impl ────────────────────────────────────────────────────

impl Launcher for Context {
    fn cl_queue(&self) -> &CommandQueue {
        &self.inner.default_cl_queue
    }

    fn context(&self) -> &Context {
        self
    }
}
