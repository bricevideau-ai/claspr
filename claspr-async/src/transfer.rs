//! [`upload!`](crate::upload) / [`download!`](crate::download) —
//! async-capable host-to-device and device-to-host transfers as
//! [`DeviceOperation`]s.
//!
//! Where the Tier 1 builders [`DeviceSlice::write`] /
//! [`DeviceSlice::read`] take a borrowed source / destination and a
//! [`Launcher`](claspr::Launcher), these consume ownership and pick the chain's queue
//! from the [`ExecutionContext`] at execute time. That makes them
//! compose cleanly into combinator chains:
//!
//! ```ignore
//! upload!(host_vec).and_then(|buf| kernel_op(buf)).and_then(|buf| download!(buf)).sync(&ctx)?
//! ```
//!
//! Both ops use **non-blocking enqueues** under the hood:
//!
//! - `upload!` keeps the source host buffer alive via a
//!   `clSetEventCallback(CL_COMPLETE, ...)` that drops a boxed holder
//!   when the write finishes (the same FFI shim Tier 1 uses for
//!   profiling). The OpenCL spec (§5.2.1) requires the source to
//!   stay valid until the write event fires; the drop callback is
//!   what makes that safe with a non-blocking enqueue.
//! - `download!` doesn't need a keep-alive: the destination `Vec<T>`
//!   moves up the chain (Rust `Vec` moves don't reallocate, the heap
//!   address stays stable), the source `DeviceSlice` drops at the end
//!   of `execute` but OpenCL retains its `cl_mem` internally until
//!   the read completes (CL spec on `clReleaseMemObject`).
//!
//! ## Sharing host data: [`UploadSource`]
//!
//! `upload!` accepts any `impl Into<UploadSource<T>>` — currently
//! `Vec<T>`, `Box<[T]>`, and `Arc<[T]>`. The `Arc<[T]>` variant lets
//! the caller keep a clone of the source for their own use or upload
//! the same data to multiple buffers without copying:
//!
//! ```ignore
//! use std::sync::Arc;
//! let weights: Arc<[f32]> = Arc::from(vec![0.1, 0.2, 0.3]);
//! let buf_a = upload!(Arc::clone(&weights)).sync(&ctx)?;
//! let buf_b = upload!(Arc::clone(&weights)).sync(&ctx)?;
//! // weights still usable here; data heap not freed until all Arcs
//! // (including the ones held by the keep-alive callbacks) drop.
//! ```

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use claspr::{Buffer, DeviceSlice, MemMode, ReadWrite, Result, register_drop_callback};
use std::marker::PhantomData;
use std::sync::Arc;

// ── UploadSource ────────────────────────────────────────────────────

/// Polymorphic host-data source for [`upload!`](crate::upload). Concrete variants
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

/// Allocate a [`DeviceSlice<T, M>`] of `source.len()` elements and
/// write `source` into it. Built by the [`upload!`](crate::upload!)
/// macro or directly via [`Self::new`].
///
/// Non-blocking: `clEnqueueWriteBuffer(CL_FALSE)` runs on the chain's
/// OOO queue with `deps` as wait-list. The host buffer is kept alive
/// by a `clSetEventCallback(CL_COMPLETE)` that drops the holder when
/// the write finishes.
///
/// **Marker bound:** `M: HostUploadable` — excludes `HostReadOnly`,
/// `Frozen`, `DeviceScratch`. For those markers, use the
/// [`device_slice_from_slice!`](crate::device_slice_from_slice!)
/// path (`CL_MEM_COPY_HOST_PTR`, no post-creation write).
pub struct Upload<T, M: MemMode = ReadWrite> {
    source: Option<UploadSource<T>>,
    _phantom: PhantomData<fn() -> M>,
}

impl<T, M> Upload<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + claspr::HostUploadable + claspr::Fillable + Send + 'static,
{
    pub fn new<S>(source: S) -> Self
    where
        S: Into<UploadSource<T>>,
    {
        Self {
            source: Some(source.into()),
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for Upload<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + claspr::HostUploadable + claspr::Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn execute(
        mut self,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(DeviceSlice<T, M>, Deps)> {
        let source = self
            .source
            .take()
            .expect("Upload::execute called twice — internal claspr-async bug");
        // SAFETY: write below covers every byte of the freshly-allocated
        // buffer; downstream stages gate on the returned write event.
        let mut buf = unsafe {
            DeviceSlice::<T, M>::alloc_uninit(ctx.context(), source.len())?.assume_init()
        };
        let event = buf
            .write(source.as_slice())
            .after_all(deps_as_events(&deps))
            .submit(ctx)?;
        register_drop_callback(&event, Box::new(source))?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// ── download ────────────────────────────────────────────────────────

/// Consume `buf`, allocate a host `Vec<T>`, and non-blocking-read the
/// buffer into it. Built by [`download!`](crate::download!) or
/// directly via [`Self::new`]. `M` is whatever marker the input buffer
/// carries; bound `M: HostReadable` (excludes `DeviceScratch`).
pub struct Download<T, M: MemMode = ReadWrite> {
    buf: Option<DeviceSlice<T, M>>,
}

impl<T, M> Download<T, M>
where
    T: Clone + Default + Send + 'static,
    M: MemMode + claspr::HostReadable + Send + 'static,
{
    pub fn new(buf: DeviceSlice<T, M>) -> Self {
        Self { buf: Some(buf) }
    }
}

impl<T, M> DeviceOperation for Download<T, M>
where
    T: Clone + Default + Send + 'static,
    M: MemMode + claspr::HostReadable + Send + 'static,
{
    type Output = Vec<T>;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Vec<T>, Deps)> {
        let buf = self
            .buf
            .take()
            .expect("Download::execute called twice — internal claspr-async bug");
        let mut out = vec![T::default(); buf.len()];
        let event = buf
            .read(&mut out)
            .after_all(deps_as_events(&deps))
            .submit(ctx)?;
        Ok((out, vec![wrap_event(event)]))
    }
}
