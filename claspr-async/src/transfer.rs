//! [`upload`] / [`download`] — async-capable host-to-device and
//! device-to-host transfers as [`DeviceOperation`]s.
//!
//! Where the Tier 1 builders [`DeviceSlice::write`] /
//! [`DeviceSlice::read`] take a borrowed source / destination and a
//! [`Launcher`], these consume ownership and pick the chain's queue
//! from the [`ExecutionContext`] at execute time. That makes them
//! compose cleanly into combinator chains:
//!
//! ```ignore
//! upload(host_vec).and_then(|buf| kernel_op(buf)).and_then(download).sync(&ctx)?
//! ```
//!
//! Both ops use **non-blocking enqueues** under the hood:
//!
//! - [`upload`] keeps the source host buffer alive via a
//!   `clSetEventCallback(CL_COMPLETE, ...)` that drops a boxed holder
//!   when the write finishes (the same FFI shim Tier 1 uses for
//!   profiling). The OpenCL spec (§5.2.1) requires the source to
//!   stay valid until the write event fires; the drop callback is
//!   what makes that safe with a non-blocking enqueue.
//! - [`download`] doesn't need a keep-alive: the destination `Vec<T>`
//!   moves up the chain (Rust `Vec` moves don't reallocate, the heap
//!   address stays stable), the source `DeviceSlice` drops at the end
//!   of `execute` but OpenCL retains its `cl_mem` internally until
//!   the read completes (CL spec on `clReleaseMemObject`).
//!
//! ## Sharing host data: [`UploadSource`]
//!
//! `upload` accepts any `impl Into<UploadSource<T>>` — currently
//! `Vec<T>`, `Box<[T]>`, and `Arc<[T]>`. The `Arc<[T]>` variant lets
//! the caller keep a clone of the source for their own use or upload
//! the same data to multiple buffers without copying:
//!
//! ```ignore
//! use std::sync::Arc;
//! let weights: Arc<[f32]> = Arc::from(vec![0.1, 0.2, 0.3]);
//! let buf_a = upload(Arc::clone(&weights)).sync(&ctx)?;
//! let buf_b = upload(Arc::clone(&weights)).sync(&ctx)?;
//! // weights still usable here; data heap not freed until all Arcs
//! // (including the ones held by the keep-alive callbacks) drop.
//! ```

use crate::exec_ctx::ExecutionContext;
use crate::op::DeviceOperation;
use claspr::{Buffer, DeviceSlice, Result};
use std::sync::Arc;

// ── UploadSource ────────────────────────────────────────────────────

/// Polymorphic host-data source for [`upload`]. Concrete variants
/// cover the common cases — `Vec<T>` (move and forget), `Box<[T]>`
/// (heap-allocated slice), `Arc<[T]>` (shared / caller retains a
/// clone). Construct via [`From`] / [`Into`].
pub enum UploadSource<T> {
    Vec(Vec<T>),
    Box(Box<[T]>),
    Arc(Arc<[T]>),
}

impl<T> UploadSource<T> {
    /// Borrow the underlying slice. Stable address across the
    /// lifetime of the [`UploadSource`] — OpenCL is reading from it
    /// during the non-blocking write.
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Vec(v) => v,
            Self::Box(b) => b,
            Self::Arc(a) => a,
        }
    }

    /// Element count.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// `true` if the source has zero elements.
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}

impl<T> From<Vec<T>> for UploadSource<T> {
    fn from(v: Vec<T>) -> Self {
        Self::Vec(v)
    }
}

impl<T> From<Box<[T]>> for UploadSource<T> {
    fn from(b: Box<[T]>) -> Self {
        Self::Box(b)
    }
}

impl<T> From<Arc<[T]>> for UploadSource<T> {
    fn from(a: Arc<[T]>) -> Self {
        Self::Arc(a)
    }
}

// ── upload ──────────────────────────────────────────────────────────

/// Allocate a [`DeviceSlice`] of `source.len()` elements and write
/// `source` into it.
///
/// **execute() currently blocks** on the write completing (CL_TRUE
/// internally). Reason: claspr-async's chains run on a per-device
/// out-of-order queue, and we don't yet auto-thread "last writer"
/// events between dependent ops — so a non-blocking write here would
/// race a downstream read on the same buffer (per CL §5.4 OOO commands
/// may reorder without explicit event links). Until per-buffer event
/// tracking lands, the safe choice is a blocking write inside execute.
///
/// The chain *as a whole* is still async-capable via [`run`](crate::DeviceOperation::run):
/// the upload step blocks, but kernel pipelining + the async terminal
/// stay intact. Multiple parallel uploads (e.g. inside [`fan_out`])
/// pipeline at the host level — each upload's execute blocks but they
/// can run concurrently when fan_out submits them on the OOO queue.
///
/// See [`UploadSource`] for the sources accepted via `impl Into<...>`.
pub fn upload<T, S>(source: S) -> Upload<T>
where
    // `T: Sync` is required only because `UploadSource::Arc(Arc<[T]>)`
    // needs `Arc<T>: Send` for the chain's Send bound; Vec/Box paths
    // don't need it. Cheap to keep here; common types (u32, f32, ...)
    // satisfy it.
    T: Send + Sync + 'static,
    S: Into<UploadSource<T>>,
{
    Upload {
        source: Some(source.into()),
    }
}

/// Combinator built by [`upload`]. Lazy — `execute` allocates and
/// enqueues the (blocking) write when the chain reaches it.
pub struct Upload<T> {
    source: Option<UploadSource<T>>,
}

impl<T> DeviceOperation for Upload<T>
where
    T: Send + Sync + 'static,
{
    type Output = DeviceSlice<T>;

    fn execute(mut self, ctx: &ExecutionContext<'_>) -> Result<DeviceSlice<T>> {
        let source = self
            .source
            .take()
            .expect("Upload::execute called twice — internal claspr-async bug");
        let mut buf = DeviceSlice::alloc(ctx.context(), source.len())?;
        // Blocking enqueue (CL_TRUE). See the function-level docs for
        // why: without per-buffer event tracking, a non-blocking
        // write would race a downstream read on the same OOO queue.
        buf.write(ctx, source.as_slice()).wait()?;
        // `source` is dropped at end of scope. With CL_TRUE the
        // runtime is done reading from the host heap by the time
        // .wait() returns, so the drop is safe — no keep-alive
        // callback needed.
        Ok(buf)
    }
}

// ── download ────────────────────────────────────────────────────────

/// Consume `buf` and allocate a host `Vec<T>` of `buf.len()` elements,
/// blocking-read the buffer into it.
///
/// **execute() blocks** on the read (CL_TRUE), same rationale as
/// [`upload`]: without per-buffer event tracking, a non-blocking read
/// would race subsequent ops (or post-terminator user code) that
/// touch the Vec contents. The chain itself remains async-capable
/// via [`run`](crate::DeviceOperation::run) — the download step
/// blocks, but earlier pipelined work plus the async terminal stay
/// intact.
pub fn download<T>(buf: DeviceSlice<T>) -> Download<T>
where
    T: Clone + Default + Send + 'static,
{
    Download { buf: Some(buf) }
}

/// Combinator built by [`download`]. Lazy — `execute` allocs the
/// destination Vec and enqueues the (blocking) read when the chain
/// reaches it.
pub struct Download<T> {
    buf: Option<DeviceSlice<T>>,
}

impl<T> DeviceOperation for Download<T>
where
    T: Clone + Default + Send + 'static,
{
    type Output = Vec<T>;

    fn execute(mut self, ctx: &ExecutionContext<'_>) -> Result<Vec<T>> {
        let buf = self
            .buf
            .take()
            .expect("Download::execute called twice — internal claspr-async bug");
        let mut out = vec![T::default(); buf.len()];
        // Blocking read (CL_TRUE). See function-level docs for why.
        buf.read(ctx, &mut out).wait()?;
        Ok(out)
    }
}
