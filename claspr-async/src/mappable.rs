//! [`Mappable`] — types whose contents can be temporarily exposed to
//! the host via `clEnqueueMapBuffer` / `clEnqueueUnmapMemObject` (or
//! a no-op pass-through for plain scalars).
//!
//! The trait powers the async [`crate::AndThenHost`] combinator: the
//! combinator borrows the upstream output via `Mappable::map` to get
//! a host pointer, hands a borrowed view to the user closure, and
//! commits writes back to the device via the matching unmap. From
//! the chain's perspective the input passes through unchanged; the
//! only signal flowing downstream is a `clCreateUserEvent` that
//! gates the unmap.
//!
//! ## Sequencing inside `AndThenHost::execute`
//!
//! The main thread is the only thread that enqueues OpenCL commands:
//!
//! 1. `(handle, map_events) = T::map(input, queue, &source_evts)` —
//!    enqueues map(s) with upstream events as wait-list.
//! 2. `user_event = create_user_event(ctx)` — created in
//!    `CL_SUBMITTED` state; signalled later from the worker thread.
//! 3. `unmap_events = T::enqueue_unmap(&mut handle, queue, &[user_event])`
//!    — enqueues unmap(s) waiting on the user event.
//! 4. Spawn worker holding (handle, map_events, closure, user_event).
//! 5. Return `(input, unmap_events)` to downstream.
//!
//! The worker thread waits on the map events, runs the closure
//! against the borrowed view, and signals the user event — which
//! releases the queued unmaps.
//!
//! ## Defensive Drop
//!
//! Every `MapHandle` impl tracks whether `enqueue_unmap` was called.
//! If it wasn't (an error path between `map` and `enqueue_unmap`),
//! the handle's `Drop` issues a blocking unmap so the buffer doesn't
//! leak its mapped state. This is the only blocking call in the
//! whole module, and only fires on the error path.

use claspr::util::{RetainedQueue, mapped_slice_mut};
use claspr::{Buffer, DeviceSlice, Event, Result};
use opencl3::command_queue::{CommandQueue, enqueue_map_buffer, enqueue_unmap_mem_object};
use opencl3::error_codes::ClError;
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE, ClMem};
use opencl3::types::{CL_NON_BLOCKING, cl_event, cl_mem};
use std::ptr;

fn cl_to_err(code: opencl3::types::cl_int) -> claspr::Error {
    claspr::Error::OpenCl(ClError(code))
}

// ── Trait ───────────────────────────────────────────────────────────

/// Inputs the [`crate::AndThenHost`] combinator can present to a host
/// closure. Implemented for [`DeviceSlice<T>`] (real map/unmap), plain
/// scalar types (no-op pass-through), and tuples up to arity 4 of
/// other [`Mappable`] types.
pub trait Mappable: Send + 'static {
    /// Borrowed view passed to the user closure. For
    /// [`DeviceSlice<T>`] this is `&'a mut [T]`; for scalars it's the
    /// scalar by value (read-only); for tuples it's a tuple of the
    /// element views.
    type View<'a>: Send
    where
        Self: 'a;

    /// State carried on the worker thread from map() through to the
    /// closure call. Per-impl — typically a host pointer + cl_mem
    /// handle + an "unmap enqueued" flag.
    type MapHandle: Send;

    /// Enqueue a non-blocking map for every device buffer in `self`,
    /// using `deps` as the wait-list. Returns the per-instance
    /// handle and the events the worker must wait on before reading
    /// the mapped memory.
    ///
    /// Borrows `&self` — the value continues to flow through the
    /// chain as the combinator's output unchanged.
    fn map(
        &self,
        queue: &CommandQueue,
        deps: &[cl_event],
    ) -> Result<(Self::MapHandle, Vec<Event>)>;

    /// Enqueue the matching unmap(s) for `handle`, with `wait_for`
    /// (the [`crate::AndThenHost`]'s user event) as the wait-list.
    /// Mutates the handle to mark unmap as enqueued so the handle's
    /// `Drop` skips the defensive blocking unmap path.
    fn enqueue_unmap(
        handle: &mut Self::MapHandle,
        queue: &CommandQueue,
        wait_for: &[cl_event],
    ) -> Result<Vec<Event>>;

    /// Build the borrowed view passed to the user closure. Called on
    /// the worker thread after `map`'s events have been waited on.
    fn view<'a>(handle: &'a mut Self::MapHandle) -> Self::View<'a>;

    /// Mark the handle as if `enqueue_unmap` was never called, so the
    /// handle's `Drop` issues the defensive synchronous unmap.
    ///
    /// Used by [`crate::AndThenHost`] in the error path: when the
    /// user event is signalled with a negative status, the queued
    /// unmap is "terminated" by the OpenCL runtime (CL spec §5.11)
    /// rather than executing, leaving the buffer mapped indefinitely.
    /// Forcing the defensive sync unmap from `Drop` cleans the
    /// buffer's state so subsequent context-level commands aren't
    /// broken by the still-mapped buffer.
    ///
    /// No-op for impls that don't actually map anything (scalars,
    /// unit, etc.).
    fn mark_unmap_not_done(handle: &mut Self::MapHandle);
}

// ── DeviceSlice<T> impl ─────────────────────────────────────────────

/// Per-element map state for [`DeviceSlice<T>`].
///
/// `unsafe impl Send` is needed because the raw `cl_mem` handle is a
/// bare pointer. It's an opaque OpenCL handle whose thread-safety is
/// documented (CL §3.4.1: command queues, except for
/// `clSetCommandQueueProperty`, are thread-safe), and we never call
/// any of the non-thread-safe APIs on it. The queue handle is owned
/// by `RetainedQueue` (already Send).
pub struct DeviceSliceMapHandle<T> {
    host_ptr: *mut T,
    cl_mem: cl_mem,
    /// Retained queue. Released on `Drop`. Held so the defensive
    /// blocking unmap path can use it even if the original
    /// `CommandQueue` was dropped by the caller.
    map_queue: RetainedQueue,
    len: usize,
    /// `true` once `enqueue_unmap` ran. If the handle drops with
    /// this still `false`, `Drop` issues a blocking unmap to keep
    /// the cl_mem in a clean state.
    unmap_enqueued: bool,
}

// SAFETY: see struct docs.
unsafe impl<T: Send> Send for DeviceSliceMapHandle<T> {}

impl<T> Drop for DeviceSliceMapHandle<T> {
    fn drop(&mut self) {
        if !self.unmap_enqueued && !self.host_ptr.is_null() {
            // Defensive blocking unmap on the error path between
            // `map` returning Ok and `enqueue_unmap` being called.
            // SAFETY: host_ptr came from a successful map call on
            // self.cl_mem via self.map_queue. We unmap exactly once.
            let res = unsafe {
                enqueue_unmap_mem_object(
                    self.map_queue.raw(),
                    self.cl_mem,
                    self.host_ptr.cast(),
                    0,
                    ptr::null(),
                )
            };
            if let Ok(ev) = res {
                let _ = opencl3::event::wait_for_events(&[ev]);
                let _ = Event::new(ev); // drops -> releases the event
            }
        }
        // The `map_queue: RetainedQueue` field drops after this body
        // returns and releases the queue handle.
    }
}

impl<T> Mappable for DeviceSlice<T>
where
    T: Copy + Send + 'static,
{
    type View<'a> = &'a mut [T];
    type MapHandle = DeviceSliceMapHandle<T>;

    fn map(
        &self,
        queue: &CommandQueue,
        deps: &[cl_event],
    ) -> Result<(Self::MapHandle, Vec<Event>)> {
        let cl_mem = self.buffer().get();
        let len = Buffer::len(self);
        let size = len * std::mem::size_of::<T>();
        // Retain the queue so its handle stays valid for the
        // defensive Drop-time unmap if anything between this map
        // and `enqueue_unmap` errors out.
        let map_queue = RetainedQueue::from_queue(queue)?;
        let (wait_count, wait_ptr) = if deps.is_empty() {
            (0, ptr::null())
        } else {
            (deps.len() as u32, deps.as_ptr())
        };
        let mut host_ptr_raw: *mut std::ffi::c_void = ptr::null_mut();
        // SAFETY: cl_mem is a live buffer (we just borrowed `&self`);
        // the size matches the allocation's element-count × element
        // size; deps points to live cl_events for the duration of
        // this call; host_ptr_raw is a stable out-param.
        let map_event = unsafe {
            enqueue_map_buffer(
                map_queue.raw(),
                cl_mem,
                CL_NON_BLOCKING,
                CL_MAP_READ | CL_MAP_WRITE,
                0,
                size,
                &mut host_ptr_raw,
                wait_count,
                wait_ptr,
            )
            .map_err(cl_to_err)?
        };
        let handle = DeviceSliceMapHandle {
            host_ptr: host_ptr_raw.cast::<T>(),
            cl_mem,
            map_queue,
            len,
            unmap_enqueued: false,
        };
        Ok((handle, vec![Event::new(map_event)]))
    }

    fn enqueue_unmap(
        handle: &mut Self::MapHandle,
        queue: &CommandQueue,
        wait_for: &[cl_event],
    ) -> Result<Vec<Event>> {
        let q_raw = queue.get();
        let (wait_count, wait_ptr) = if wait_for.is_empty() {
            (0, ptr::null())
        } else {
            (wait_for.len() as u32, wait_for.as_ptr())
        };
        // SAFETY: handle.cl_mem is the buffer we mapped, host_ptr
        // is the pointer we got back. Enqueue exactly once per
        // map (the unmap_enqueued flag guards against double-unmap).
        let unmap_event = unsafe {
            enqueue_unmap_mem_object(
                q_raw,
                handle.cl_mem,
                handle.host_ptr.cast(),
                wait_count,
                wait_ptr,
            )
            .map_err(cl_to_err)?
        };
        handle.unmap_enqueued = true;
        Ok(vec![Event::new(unmap_event)])
    }

    fn view<'a>(handle: &'a mut Self::MapHandle) -> Self::View<'a> {
        // SAFETY: host_ptr is the mapped pointer with `len` valid
        // elements of T (CL_MAP_READ | CL_MAP_WRITE). The worker
        // calls this only after waiting on the map event, so the
        // memory is coherent. `&'a mut self` on the handle is the
        // borrow-checker proof that no other view aliases this.
        unsafe { mapped_slice_mut(handle.host_ptr, handle.len) }
    }

    fn mark_unmap_not_done(handle: &mut Self::MapHandle) {
        handle.unmap_enqueued = false;
    }
}

// ── Scalar pass-through ─────────────────────────────────────────────

/// No-op `Mappable` for plain `Copy` scalars. The "map" is a value
/// copy; the closure receives the scalar by value.
macro_rules! impl_mappable_scalar {
    ($($t:ty),*) => {$(
        impl Mappable for $t {
            type View<'a> = $t;
            type MapHandle = $t;
            fn map(
                &self,
                _queue: &CommandQueue,
                _deps: &[cl_event],
            ) -> Result<(Self::MapHandle, Vec<Event>)> {
                Ok((*self, Vec::new()))
            }
            fn enqueue_unmap(
                _handle: &mut Self::MapHandle,
                _queue: &CommandQueue,
                _wait_for: &[cl_event],
            ) -> Result<Vec<Event>> {
                Ok(Vec::new())
            }
            fn view<'a>(handle: &'a mut Self::MapHandle) -> Self::View<'a> {
                *handle
            }
            fn mark_unmap_not_done(_handle: &mut Self::MapHandle) {
                // No-op: scalars never enqueued a map.
            }
        }
    )*};
}

impl_mappable_scalar!(u8, u16, u32, u64, i8, i16, i32, i64, usize, isize, f32, f64, bool);

// `()` pass-through — chains that produce no output can still go
// through and_then_host (e.g. for side-effect-only host work after
// a fire-and-forget kernel launch).
impl Mappable for () {
    type View<'a> = ();
    type MapHandle = ();
    fn map(
        &self,
        _queue: &CommandQueue,
        _deps: &[cl_event],
    ) -> Result<(Self::MapHandle, Vec<Event>)> {
        Ok(((), Vec::new()))
    }
    fn enqueue_unmap(
        _handle: &mut Self::MapHandle,
        _queue: &CommandQueue,
        _wait_for: &[cl_event],
    ) -> Result<Vec<Event>> {
        Ok(Vec::new())
    }
    fn view<'a>(_handle: &'a mut Self::MapHandle) -> Self::View<'a> {}
    fn mark_unmap_not_done(_handle: &mut Self::MapHandle) {}
}

// ── Tuple impls (arity 2..=4) ───────────────────────────────────────

macro_rules! impl_mappable_tuple {
    // $name: type params (A, B, ...). $hname: fresh ident bindings
    // for the handles (ha, hb, ...). $idx: tuple field indices.
    // All three lists are parallel and must have the same length.
    ($($name:ident),+ ; $($hname:ident),+ ; $($idx:tt),+) => {
        impl<$($name: Mappable),+> Mappable for ($($name,)+) {
            type View<'a> = ($($name::View<'a>,)+) where Self: 'a;
            type MapHandle = ($($name::MapHandle,)+);

            fn map(
                &self,
                queue: &CommandQueue,
                deps: &[cl_event],
            ) -> Result<(Self::MapHandle, Vec<Event>)> {
                let mut events: Vec<Event> = Vec::new();
                $(
                    let ($hname, evs) = self.$idx.map(queue, deps)?;
                    events.extend(evs);
                )+
                Ok((($($hname,)+), events))
            }

            fn enqueue_unmap(
                handle: &mut Self::MapHandle,
                queue: &CommandQueue,
                wait_for: &[cl_event],
            ) -> Result<Vec<Event>> {
                let mut events: Vec<Event> = Vec::new();
                $(
                    events.extend($name::enqueue_unmap(&mut handle.$idx, queue, wait_for)?);
                )+
                Ok(events)
            }

            fn view<'a>(handle: &'a mut Self::MapHandle) -> Self::View<'a> {
                let &mut ($(ref mut $hname,)+) = handle;
                ($($name::view($hname),)+)
            }

            fn mark_unmap_not_done(handle: &mut Self::MapHandle) {
                let &mut ($(ref mut $hname,)+) = handle;
                $($name::mark_unmap_not_done($hname);)+
            }
        }
    };
}

impl_mappable_tuple!(A, B; ha, hb; 0, 1);
impl_mappable_tuple!(A, B, C; ha, hb, hc; 0, 1, 2);
impl_mappable_tuple!(A, B, C, D; ha, hb, hc, hd; 0, 1, 2, 3);

#[cfg(test)]
mod tests {
    use super::*;
    use claspr::{Context, DeviceSlice};

    /// Map a DeviceSlice, write through the view, unmap, read back via
    /// Tier 1, assert. Exercises the whole `Mappable::DeviceSlice<T>`
    /// surface (map + view + enqueue_unmap + Drop).
    #[test]
    fn deviceslice_round_trip_through_map() {
        let Ok(ctx) = Context::any() else {
            eprintln!("skipping: no OpenCL device");
            return;
        };
        let device = ctx.device().clone();
        let queue = ctx
            .default_outoforder_queue(&device)
            .expect("oo queue");
        let q = queue.raw();

        // Allocate + seed via Tier 1 write.
        let mut buf = DeviceSlice::<u32>::alloc(&ctx, 4).expect("alloc");
        let seed = [1u32, 2, 3, 4];
        buf.write(&ctx, &seed).wait().expect("seed");

        // Map (no upstream deps; seed write already done).
        let (mut handle, map_events) =
            <DeviceSlice<u32> as Mappable>::map(&buf, q, &[]).expect("map");
        // Wait map.
        for ev in &map_events {
            ev.wait().expect("map wait");
        }

        // Mutate via view.
        {
            let view = <DeviceSlice<u32> as Mappable>::view(&mut handle);
            assert_eq!(view, &mut [1, 2, 3, 4][..]);
            for x in view.iter_mut() {
                *x += 100;
            }
        }

        // Unmap (no further deps).
        let unmap_events = <DeviceSlice<u32> as Mappable>::enqueue_unmap(&mut handle, q, &[])
            .expect("enqueue_unmap");
        for ev in &unmap_events {
            ev.wait().expect("unmap wait");
        }
        // Drop the handle (no-op for unmap path; releases the queue).
        drop(handle);

        // Read back via Tier 1, confirm writes committed.
        let mut out = [0u32; 4];
        buf.read(&ctx, &mut out).wait().expect("read");
        assert_eq!(out, [101, 102, 103, 104]);
    }

    /// Same as above but goes through the (DeviceSlice<u32>, u32)
    /// tuple Mappable impl to exercise the tuple wrapper.
    #[test]
    fn tuple_round_trip_through_map() {
        let Ok(ctx) = Context::any() else {
            eprintln!("skipping: no OpenCL device");
            return;
        };
        let device = ctx.device().clone();
        let queue = ctx
            .default_outoforder_queue(&device)
            .expect("oo queue");
        let q = queue.raw();

        let mut buf = DeviceSlice::<u32>::alloc(&ctx, 4).expect("alloc");
        buf.write(&ctx, &[10, 20, 30, 40]).wait().expect("seed");

        type T = (DeviceSlice<u32>, u32);
        let tup: T = (buf, 7u32);

        let (mut handle, map_events) = T::map(&tup, q, &[]).expect("map");
        for ev in &map_events {
            ev.wait().expect("map wait");
        }

        {
            let (view_slice, scalar) = T::view(&mut handle);
            assert_eq!(scalar, 7);
            for (i, x) in view_slice.iter_mut().enumerate() {
                *x += scalar + i as u32;
            }
        }

        let unmap_events = T::enqueue_unmap(&mut handle, q, &[]).expect("enqueue_unmap");
        for ev in &unmap_events {
            ev.wait().expect("unmap wait");
        }
        drop(handle);

        let (buf, _) = tup;
        let mut out = [0u32; 4];
        buf.read(&ctx, &mut out).wait().expect("read");
        assert_eq!(out, [17, 28, 39, 50]);
    }
}
