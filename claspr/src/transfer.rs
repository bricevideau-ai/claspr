//! [`upload!`](crate::upload) / [`download!`](crate::download) —
//! async-capable host-to-device and device-to-host transfers as
//! [`DeviceOperation`]s.
//!
//! Where the Tier 1 builders [`DeviceSlice::write`] /
//! [`DeviceSlice::read`] take a borrowed source / destination and a
//! [`Launcher`](crate::Launcher), these consume ownership and pick the chain's queue
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

use crate::device_op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use crate::exec_ctx::ExecutionContext;
use crate::{Buffer, DeviceSlice, MemMode, ReadWrite, Result};
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

// Upload moved out: the `upload!` macro is now sugar over
// `device_slice_alloc_uninit!` + `WriteUninit::write`. See lib.rs.

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
    M: MemMode + crate::HostReadable + Send + 'static,
{
    pub fn new(buf: DeviceSlice<T, M>) -> Self {
        Self { buf: Some(buf) }
    }
}

impl<T, M> DeviceOperation for Download<T, M>
where
    T: Clone + Default + Send + 'static,
    M: MemMode + crate::HostReadable + Send + 'static,
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
            .submit_on(ctx)?;
        Ok((out, vec![wrap_event(event)]))
    }
}
