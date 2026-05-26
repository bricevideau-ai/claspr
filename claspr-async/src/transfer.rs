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
use claspr::{Buffer, DeviceSlice, Result, register_drop_callback};
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

/// Allocate a [`DeviceSlice`] of `source.len()` elements and
/// non-blocking-write `source` into it. The host buffer is kept
/// alive by a `clSetEventCallback` that drops the holder when the
/// write completes — execute returns the populated [`DeviceSlice`]
/// immediately; the chain's terminator gates user visibility.
///
/// See [`UploadSource`] for the sources accepted via `impl Into<...>`.
pub fn upload<T, S>(source: S) -> Upload<T>
where
    T: Send + Sync + 'static,
    S: Into<UploadSource<T>>,
{
    Upload {
        source: Some(source.into()),
    }
}

/// Combinator built by [`upload`]. Lazy — `execute` allocates,
/// enqueues, and registers the keep-alive callback when the chain
/// reaches it.
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
        // Non-blocking enqueue via the public WriteOp builder. The
        // WriteOp captures `&source` for the duration of the enqueue
        // call; `.submit()` consumes it and returns the event, at
        // which point the borrow is released and we can move
        // `source` into the keep-alive callback.
        let event = buf.write(ctx, source.as_slice()).submit()?;
        // Move `source` into a Box, hand to the OpenCL runtime via
        // user_data. The thunk drops it when CL_COMPLETE fires —
        // exactly when OpenCL is finished reading from the host heap.
        register_drop_callback(&event, Box::new(source))?;
        Ok(buf)
    }
}

// ── download ────────────────────────────────────────────────────────

/// Consume `buf` and allocate a host `Vec<T>` of `buf.len()` elements,
/// non-blocking-read the buffer into it. The `DeviceSlice` is dropped
/// at the end of `execute` — but OpenCL keeps an internal refcount
/// on the `cl_mem` so the read completes safely. The `Vec` moves up
/// the chain (its heap address stays stable across Rust moves) and
/// the chain's terminator waits before the user sees it.
pub fn download<T>(buf: DeviceSlice<T>) -> Download<T>
where
    T: Clone + Default + Send + 'static,
{
    Download { buf: Some(buf) }
}

/// Combinator built by [`download`]. Lazy — `execute` allocs the
/// destination Vec, enqueues a non-blocking read into it, and
/// returns the Vec.
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
        // .submit() gives non-blocking enqueue + Event. We don't keep
        // the Event — the chain's terminator (queue.finish in .sync,
        // marker in .run().await) is what waits. The `out` Vec moves
        // through subsequent stages (Vec move = pointer move; the
        // heap data stays put). The `buf` DeviceSlice drops at end of
        // scope; OpenCL retains the cl_mem until the read completes.
        let _event = buf.read(ctx, &mut out).submit()?;
        Ok(out)
    }
}
