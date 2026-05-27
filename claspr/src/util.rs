//! Small unsafe-FFI primitives shared across the crate (and used by
//! `claspr-async`). Keeping them here lets the unsafe surface
//! contract to a few well-documented helpers instead of being
//! sprinkled across every `*BufferHostView` / `Shared*Guard` /
//! `*MapHandle` Drop impl.
//!
//! Exposed `pub` (not `pub(crate)`) so `claspr-async` can use them
//! across the crate boundary; tagged `#[doc(hidden)]` so they don't
//! show up in the public docs — they're implementation details, not
//! user-facing surface.

use crate::error::{Error, Result};
use opencl3::command_queue::{CommandQueue, release_command_queue, retain_command_queue};
use opencl3::error_codes::ClError;
use opencl3::types::cl_command_queue;
use std::slice;

/// RAII wrapper around a retained `cl_command_queue` handle.
///
/// `clEnqueueMapBuffer` returns a host pointer that's only valid
/// until the matching `clEnqueueUnmapMemObject` fires. Code that
/// holds a mapped pointer across an arbitrary lifetime (the host
/// view types, the `Mappable` handles, [`HostBuffer`](crate::HostBuffer))
/// needs to keep a `cl_command_queue` alive for the eventual unmap
/// even if the original `Launcher` that supplied the queue has
/// long since dropped.
///
/// Construct with [`RetainedQueue::from_queue`] (safe — borrows a
/// live `CommandQueue`). Drop releases the retained reference. The
/// raw handle is exposed via [`raw`](Self::raw) for FFI use.
///
/// `unsafe impl Send` is justified by OpenCL spec §3.4.1 (command
/// queues are thread-safe except for `clSetCommandQueueProperty`
/// which we never call).
#[doc(hidden)]
pub struct RetainedQueue {
    raw: cl_command_queue,
}

// SAFETY: cl_command_queue is an opaque thread-safe handle (CL
// §3.4.1). We hold a retained reference, so the queue stays alive
// regardless of which thread drops us.
unsafe impl Send for RetainedQueue {}
// SAFETY: We never mutate the handle through &self; only Drop (which
// has &mut access) calls release.
unsafe impl Sync for RetainedQueue {}

impl RetainedQueue {
    /// Retain the queue's raw handle. Drop will release it.
    pub fn from_queue(queue: &CommandQueue) -> Result<Self> {
        let raw = queue.get();
        // SAFETY: `raw` was just queried from a live `CommandQueue`;
        // we own the retain that follows and release in Drop.
        unsafe { retain_command_queue(raw) }.map_err(|c| Error::OpenCl(ClError(c)))?;
        Ok(Self { raw })
    }

    /// Raw `cl_command_queue` for passing to FFI. The returned handle
    /// is valid for the lifetime of `&self`.
    pub fn raw(&self) -> cl_command_queue {
        self.raw
    }
}

impl Drop for RetainedQueue {
    fn drop(&mut self) {
        // SAFETY: `raw` is the handle we retained in `from_queue`;
        // pair with one release exactly. Errors here can't be
        // propagated (no `Result` from `drop`); the caller's
        // `Context::record_err` machinery isn't accessible from
        // here, so we drop the error silently. The cl_command_queue
        // refcount being wrong is recoverable at process scope.
        let _ = unsafe { release_command_queue(self.raw) };
    }
}

// ── Mapped-slice helpers ────────────────────────────────────────────

/// Construct an immutable slice from a mapped host pointer + length.
///
/// SAFETY contract:
/// - `ptr` must point to `len` contiguous, properly-aligned `T`s.
/// - The memory must remain valid for the entire lifetime `'a`.
/// - No mutable references to overlapping memory may exist for `'a`.
///
/// Typical use is in a `Deref` / `Mappable::view` impl where the
/// enclosing type's invariants establish the safety contract (e.g.
/// "host pointer obtained from `clEnqueueMapBuffer` and valid until
/// `clEnqueueUnmapMemObject` fires, which can't happen while we
/// hold a borrow of the view").
#[doc(hidden)]
#[allow(clippy::missing_safety_doc)] // safety doc lives above
pub unsafe fn mapped_slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    // SAFETY: caller's contract; see fn doc.
    unsafe { slice::from_raw_parts(ptr, len) }
}

/// Construct a mutable slice from a mapped host pointer + length.
///
/// SAFETY contract:
/// - `ptr` must point to `len` contiguous, properly-aligned `T`s.
/// - The memory must remain valid for the entire lifetime `'a`.
/// - No other references (mutable or shared) to overlapping memory
///   may exist for `'a`.
/// - The underlying mapping must permit writes (`CL_MAP_WRITE` or
///   equivalent).
#[doc(hidden)]
#[allow(clippy::missing_safety_doc)] // safety doc lives above
pub unsafe fn mapped_slice_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    // SAFETY: caller's contract; see fn doc.
    unsafe { slice::from_raw_parts_mut(ptr, len) }
}
