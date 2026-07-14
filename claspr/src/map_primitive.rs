//! Thin wrappers around the four `clEnqueueMap*` / `clEnqueueUnmap*`
//! entry points the rest of the workspace touches.
//!
//! Every call site that maps host memory — buffer and SVM, Tier 1
//! guards and Tier 2 acquire/release Ops, the `Mappable`
//! `AndThenHost` worker, and every defensive `Drop`-path unmap —
//! flows through one of the four functions here. The wrappers fold
//! the wait-list marshalling (`Vec<cl_event>` from a `&[Dep]` /
//! `Vec<Event>` / `&[cl_event]`), the `cl_int` → `Error::OpenCl`
//! conversion, and the `cl_event` → [`Event`] wrap into a single
//! shape, so callers see one canonical signature instead of N
//! near-identical ones with subtle inconsistencies.
//!
//! Kept `pub` (not `pub(crate)`) for in-crate Tier 2 plumbing, and
//! `#[doc(hidden)]` so they don't appear in user-facing docs —
//! callers are expected to be in-workspace plumbing, not user code.
//!
//! ## Safety contract (shared by all four)
//!
//! The caller must ensure:
//! - `queue` is live for the duration of the call.
//! - For buffer variants: `mem` is a live `cl_mem`; for the unmap
//!   variant, `host_ptr` was produced by a prior matching map on
//!   the same `mem`.
//! - For SVM variants: `ptr` is a live SVM allocation belonging to
//!   the same context as `queue`; `size` matches the allocation's
//!   byte length (for the map; the unmap takes no size).
//! - Every wait-list `cl_event` in `deps` remains valid for the call.
//!
//! Each function is `unsafe` because the caller alone can satisfy
//! these invariants — the same status quo as the raw `opencl3`
//! functions they wrap.

use crate::error::{Error, Result};
use opencl3::command_queue::{
    enqueue_map_buffer, enqueue_svm_map, enqueue_svm_unmap, enqueue_unmap_mem_object,
};
use opencl3::error_codes::ClError;
use opencl3::event::Event;
use opencl3::types::{
    CL_BLOCKING, CL_NON_BLOCKING, cl_bool, cl_command_queue, cl_event, cl_int, cl_map_flags, cl_mem,
};
use std::ffi::c_void;
use std::ptr;

/// A non-blocking map in flight: a map guard `G` whose backing map is still
/// enqueued, plus the map `Event`. Returned by every `*::submit` map op — the four
/// families (`cl_mem` read/write, SVM read/write) differ only in the guard type `G`,
/// so they are all type aliases of this one shape (see `DeviceMapReadPending`,
/// `MappedWritePending`, …). `event()` borrows the map event for cross-queue chain
/// ordering without consuming; `wait()` blocks then yields the guard.
///
/// The guard is held in an `Option` so `wait` can move it out; if the pending is
/// dropped WITHOUT `wait`, the `Option<G>`'s own drop glue drops the guard (which
/// enqueues the unmap on the same in-order queue) — so no explicit `Drop` is needed.
pub struct MapPending<G> {
    guard: Option<G>,
    event: Event,
}

impl<G> MapPending<G> {
    /// Wrap a freshly-enqueued map guard + its event.
    pub fn new(guard: G, event: Event) -> Self {
        MapPending {
            guard: Some(guard),
            event,
        }
    }

    /// Borrow the map [`Event`] for `.after(&evt)` cross-queue chaining, without
    /// consuming the pending (`wait` still yields the guard).
    pub fn event(&self) -> &Event {
        &self.event
    }

    /// Block on the map event, then return the guard. After `Ok`, the guard's
    /// `Deref` target is safe to access.
    pub fn wait(mut self) -> Result<G> {
        self.event.wait()?;
        Ok(self.guard.take().expect("MapPending::wait called twice"))
    }
}

fn cl_to_err(code: cl_int) -> Error {
    Error::OpenCl(ClError(code))
}

fn wait_list(deps: &[cl_event]) -> (u32, *const cl_event) {
    if deps.is_empty() {
        (0, ptr::null())
    } else {
        (deps.len() as u32, deps.as_ptr())
    }
}

/// Enqueue a buffer map. Returns the host pointer and an owning
/// [`Event`] (the wrapper's `Drop` releases the underlying
/// `cl_event`, so callers can move it into a `Vec<Event>` or
/// `Arc<Event>` and forget about the release).
///
/// `blocking` is `true` for `CL_BLOCKING`, `false` for `CL_NON_BLOCKING`.
///
/// # Safety
///
/// See module docs.
#[doc(hidden)]
pub unsafe fn map_buffer(
    queue: cl_command_queue,
    mem: cl_mem,
    blocking: bool,
    flags: cl_map_flags,
    offset: usize,
    size: usize,
    deps: &[cl_event],
) -> Result<(*mut c_void, Event)> {
    let (wait_count, wait_ptr) = wait_list(deps);
    let mut host_ptr: cl_mem = ptr::null_mut();
    let blocking_bool: cl_bool = if blocking {
        CL_BLOCKING
    } else {
        CL_NON_BLOCKING
    };
    // SAFETY: caller's contract — see module docs.
    let evt = unsafe {
        enqueue_map_buffer(
            queue,
            mem,
            blocking_bool,
            flags,
            offset,
            size,
            &mut host_ptr,
            wait_count,
            wait_ptr,
        )
        .map_err(cl_to_err)?
    };
    Ok((host_ptr.cast(), Event::new(evt)))
}

/// Enqueue a `clEnqueueUnmapMemObject` on `mem` / `host_ptr` with
/// `deps` as the wait-list. Returns the unmap [`Event`].
///
/// # Safety
///
/// See module docs.
#[doc(hidden)]
pub unsafe fn unmap_mem_object(
    queue: cl_command_queue,
    mem: cl_mem,
    host_ptr: *mut c_void,
    deps: &[cl_event],
) -> Result<Event> {
    let (wait_count, wait_ptr) = wait_list(deps);
    // SAFETY: caller's contract — see module docs.
    let evt = unsafe {
        enqueue_unmap_mem_object(queue, mem, host_ptr, wait_count, wait_ptr).map_err(cl_to_err)?
    };
    Ok(Event::new(evt))
}

/// Enqueue a `clEnqueueSVMMap` on `ptr` / `size` with `deps` as
/// the wait-list. Returns the map [`Event`].
///
/// `blocking` is `true` for `CL_BLOCKING`, `false` for `CL_NON_BLOCKING`.
///
/// # Safety
///
/// See module docs.
#[doc(hidden)]
pub unsafe fn svm_map(
    queue: cl_command_queue,
    blocking: bool,
    flags: cl_map_flags,
    ptr: *mut c_void,
    size: usize,
    deps: &[cl_event],
) -> Result<Event> {
    let (wait_count, wait_ptr) = wait_list(deps);
    let blocking_bool: cl_bool = if blocking {
        CL_BLOCKING
    } else {
        CL_NON_BLOCKING
    };
    // SAFETY: caller's contract — see module docs.
    let evt = unsafe {
        enqueue_svm_map(queue, blocking_bool, flags, ptr, size, wait_count, wait_ptr)
            .map_err(cl_to_err)?
    };
    Ok(Event::new(evt))
}

/// Enqueue a `clEnqueueSVMUnmap` on `ptr` with `deps` as the
/// wait-list. Returns the unmap [`Event`].
///
/// # Safety
///
/// See module docs.
#[doc(hidden)]
pub unsafe fn svm_unmap(
    queue: cl_command_queue,
    ptr: *mut c_void,
    deps: &[cl_event],
) -> Result<Event> {
    let (wait_count, wait_ptr) = wait_list(deps);
    // SAFETY: caller's contract — see module docs.
    let evt = unsafe { enqueue_svm_unmap(queue, ptr, wait_count, wait_ptr).map_err(cl_to_err)? };
    Ok(Event::new(evt))
}
