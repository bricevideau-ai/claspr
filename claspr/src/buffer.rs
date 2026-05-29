//! Typed device-side buffers and the [`Buffer`] trait that abstracts
//! over them.
//!
//! One tier lives in this module: [`DeviceSlice<T>`] —
//! `CL_MEM_READ_WRITE`, accessed via [`upload`](DeviceSlice::upload)
//! / [`download`](DeviceSlice::download). The host-mapped tier (SVM
//! / [`MappedSlice`](crate::mapped::MappedSlice)) lives in
//! [`crate::mapped`].
//!
//! See the [`Buffer`] trait's own docs for what it does and does
//! not abstract over.

use crate::context::Context;
use crate::error::{Error, Result};
use crate::op::{ProfileCb, ProfilingInfo, register_profiling_callback};
use crate::queue::Launcher;
use opencl3::command_queue::CommandQueue;
use opencl3::event::Event;
use opencl3::memory::{Buffer as ClBuffer, CL_MEM_READ_WRITE, ClMem, release_mem_object};
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_event, cl_mem};
use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr;

// ── Buffer trait ────────────────────────────────────────────────────

/// Common accessors shared by the buffer tiers — [`DeviceSlice`]
/// and [`crate::mapped::MappedSlice`].
///
/// **Scope: plumbing, not tier polymorphism.** This trait exposes
/// only the inspect-the-buffer accessors that mean the same thing
/// across every tier: element count and the owning [`Context`]. It
/// is *not* an upload/download polymorphism point — those operations
/// stay on each concrete type because their signatures and lifetimes
/// genuinely differ:
///
/// - [`DeviceSlice::upload`] / [`DeviceSlice::download`] enqueue a
///   `clEnqueueRead`/`WriteBuffer` against a [`Launcher`].
/// - [`MappedSlice`](crate::mapped::MappedSlice) maps lazily on demand
///   via [`MappedSlice::map_mut`](crate::mapped::MappedSlice::map_mut)
///   and unmaps when the guard drops.
/// - [`USMSlice`](crate::usm::USMSlice) wraps a host `Vec<T>` directly
///   — no map step at all, requires fine-grain-system SVM.
///
/// So code like `fn upload_and_run<B: Buffer<T>>(b: &mut B, data: &[T])`
/// is intentionally not possible — there is no single "upload" verb
/// that does the right thing on every tier, and pretending one exists
/// would force the polymorphic body to pick a worst-case strategy
/// (e.g. unconditional `clEnqueueWriteBuffer`) that pessimises the
/// zero-copy tiers.
///
/// Use this trait when you want a [`len`]/[`is_empty`]/[`ctx`]
/// accessor without committing to a tier:
///
/// ```ignore
/// fn print_size<T, B: claspr::Buffer<T>>(b: &B) {
///     println!("{} elements on {}", b.len(), b.ctx().device().name().unwrap());
/// }
/// ```
///
/// ## Future direction
///
/// If a real tier-polymorphism need surfaces (e.g. a benchmark
/// harness that wants `upload_then_run` over every tier), the likely
/// shape is a separate `BufferUpload<T>: Buffer<T>` super-trait with
/// a single `upload(&mut self, launcher, data)` method whose impls
/// call `clEnqueueWriteBuffer` for `DeviceSlice` and become a memcpy
/// through `map_mut` / `&mut` on the host-mapped tiers. That can be
/// added later without breaking the present trait's callers.
///
/// [`len`]: Self::len
/// [`is_empty`]: Self::is_empty
/// [`ctx`]: Self::ctx
pub trait Buffer<T> {
    /// Number of `T` elements in the buffer.
    fn len(&self) -> usize;

    /// `true` when the buffer has zero elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The context this buffer was allocated on.
    fn ctx(&self) -> &Context;
}

// ── DeviceSlice ─────────────────────────────────────────────────────

/// A typed device-side buffer paired with its element count.
///
/// Mirrors rust-gpu's slice decomposition for kernel parameters: a
/// `&mut [T]` kernel arg becomes two `clSetKernelArg` calls (data
/// pointer + `usize` length). When passed as a launch argument,
/// claspr sets both — see the [`KernelArg`] impl in [`crate::launch`].
///
/// Construct via [`DeviceSlice::alloc`] (zero-initialised) or
/// [`DeviceSlice::upload`] (with initial host data). For output-only
/// buffers where the zero-fill would be wasted work, use
/// [`alloc_uninit`](DeviceSlice::alloc_uninit). Read back via
/// [`DeviceSlice::download`].
///
/// Host code never sees the bytes directly — for that, use
/// [`crate::mapped::MappedSlice<T>`] (coarse-grain SVM) or
/// [`crate::usm::USMSlice<T>`] (fine-grain-system SVM over a host
/// `Vec<T>`).
///
/// [`KernelArg`]: crate::launch::KernelArg
pub struct DeviceSlice<T> {
    /// `ManuallyDrop` so opencl3's `Buffer::drop` (which panics on
    /// release failure) doesn't fire — our own [`Drop`] impl below
    /// calls `release_mem_object` and records into the context's
    /// sticky-error counter on failure instead.
    pub(crate) buffer: ManuallyDrop<ClBuffer<T>>,
    pub(crate) len: usize,
    pub(crate) ctx: Context,
}

impl<T> Drop for DeviceSlice<T> {
    fn drop(&mut self) {
        let mem = self.buffer.get();
        // SAFETY: mem was returned by clCreateBuffer in `alloc` and
        // we hold the only owner (ManuallyDrop suppressed opencl3's
        // own release path). Release exactly once now.
        let res = unsafe { release_mem_object(mem) };
        if res.is_err() {
            self.ctx.record_err();
        }
    }
}

impl<T: Default + Copy> DeviceSlice<T> {
    /// Allocate a device buffer of `len` elements, zero-initialised
    /// via `clEnqueueFillBuffer(T::default())` on the context's
    /// default queue. Blocks until the fill completes.
    ///
    /// The `T: Default + Copy` bound makes the buffer's contents a
    /// valid `T` value before any read, so a host download (or kernel
    /// read-before-write) sees `T::default()` rather than uninit
    /// bytes. Matches [`MappedSlice::alloc`](crate::MappedSlice::alloc)
    /// and [`USMSlice::alloc`](crate::USMSlice::alloc).
    ///
    /// Internally a `clCreateBuffer` + a synchronous fill. Code paths
    /// that write the entire buffer immediately afterward (e.g.
    /// `upload`, `device_slice_filled`) should use
    /// [`alloc_uninit`](Self::alloc_uninit) instead to skip the
    /// redundant zero-fill.
    pub fn alloc(ctx: &Context, len: usize) -> Result<Self> {
        let mut slice = Self::alloc_uninit(ctx, len)?;
        slice.fill(ctx, T::default()).wait()?;
        Ok(slice)
    }
}

impl<T> DeviceSlice<T> {
    /// Allocate a device buffer of `len` elements, leaving the bytes
    /// uninitialised. Cheaper than [`alloc`](Self::alloc) when the
    /// caller writes the whole buffer before any read (kernel output,
    /// explicit `write`/`fill`, etc.).
    ///
    /// Reading from an unwritten buffer (via `read` / `map` /
    /// `download`) is unspecified behaviour at the OpenCL level and
    /// undefined behaviour at the Rust level if the bytes are
    /// interpreted as a non-trivial `T`. Prefer
    /// [`alloc`](Self::alloc) unless you know the buffer is fully
    /// written before any read.
    pub fn alloc_uninit(ctx: &Context, len: usize) -> Result<Self> {
        // SAFETY: passing a null host pointer means OpenCL allocates
        // fresh device memory and ignores the host-pointer contract
        // that makes `Buffer::create` generally unsafe.
        let buffer = unsafe {
            ClBuffer::<T>::create(ctx.raw_context(), CL_MEM_READ_WRITE, len, ptr::null_mut())?
        };
        Ok(DeviceSlice {
            buffer: ManuallyDrop::new(buffer),
            len,
            ctx: ctx.clone(),
        })
    }

    /// Begin writing `data` into this buffer. Returns a lazy
    /// [`WriteOp`] builder — pick a terminal ([`wait`](WriteOp::wait),
    /// [`submit`](WriteOp::submit), `.await`) to actually run.
    ///
    /// `data.len()` must equal `self.len()` (checked at terminal time —
    /// the terminals return [`Error::LengthMismatch`] otherwise).
    pub fn write<'a, L: Launcher + ?Sized>(
        &'a mut self,
        launcher: &'a L,
        data: &'a [T],
    ) -> WriteOp<'a, T> {
        WriteOp {
            queue: launcher.cl_queue(),
            buffer: &mut self.buffer,
            dst_len: self.len,
            data,
            deps: Vec::new(),
            profile_cb: None,
        }
    }

    /// Begin reading the buffer into `dst`. Returns a lazy [`ReadOp`]
    /// builder — call [`wait`](ReadOp::wait), [`submit`](ReadOp::submit),
    /// or `.await` on it to actually run.
    ///
    /// `dst.len()` must equal `self.len()` (checked at terminal time —
    /// the terminals return [`Error::LengthMismatch`] otherwise).
    pub fn read<'a, L: Launcher + ?Sized>(
        &'a self,
        launcher: &'a L,
        dst: &'a mut [T],
    ) -> ReadOp<'a, T> {
        ReadOp {
            queue: launcher.cl_queue(),
            buffer: &self.buffer,
            src_len: self.len,
            dst,
            deps: Vec::new(),
            profile_cb: None,
        }
    }

    /// Borrow the underlying opencl3 [`ClBuffer`](opencl3::memory::Buffer)
    /// for cases that need direct OpenCL access.
    pub fn buffer(&self) -> &ClBuffer<T> {
        &self.buffer
    }

    /// Begin a device-to-device copy from `self` into `dst`. Returns
    /// a lazy [`CopyOp`] builder — call [`wait`](CopyOp::wait),
    /// [`submit`](CopyOp::submit), or `.await` on it to actually run.
    ///
    /// Both buffers must be on the same `Context` — OpenCL's
    /// `clEnqueueCopyBuffer` only works within one context. For
    /// cross-context transfers, download to host then re-upload.
    pub fn copy_to<'a, L: Launcher + ?Sized>(
        &'a self,
        dst: &'a mut DeviceSlice<T>,
        launcher: &'a L,
    ) -> CopyOp<'a, T> {
        CopyOp {
            queue: launcher.cl_queue(),
            src: &self.buffer,
            dst: &mut dst.buffer,
            src_len: self.len,
            dst_len: dst.len,
            deps: Vec::new(),
            profile_cb: None,
        }
    }

    /// Begin a `clEnqueueMigrateMemObjects` for this buffer on
    /// `launcher`'s queue — hints the OpenCL runtime to ensure the
    /// buffer resides on the queue's device's memory before subsequent
    /// commands access it from that device.
    ///
    /// On topologies where all devices in the context share physical
    /// memory (sub-devices of one CPU, integrated GPUs in a single
    /// context) the migration is typically a no-op. On distributed
    /// topologies (two dGPUs in one `cl_context`) it triggers a real
    /// memory transfer. Either way the call returns a builder; the
    /// terminals enqueue the migrate as a queue command (non-blocking
    /// via [`submit`](MigrateOp::submit) / `.await` — does NOT
    /// host-block the chain).
    ///
    /// Returns a lazy [`MigrateOp`] builder — call
    /// [`wait`](MigrateOp::wait), [`submit`](MigrateOp::submit), or
    /// `.await` on it. The target device is implicit in `launcher`'s
    /// queue (`clEnqueueMigrateMemObjects` migrates to the queue's
    /// device per spec).
    pub fn migrate<'a, L: Launcher + ?Sized>(&'a self, launcher: &'a L) -> MigrateOp<'a, T> {
        MigrateOp {
            queue: launcher.cl_queue(),
            buffer: &self.buffer,
            deps: Vec::new(),
            profile_cb: None,
        }
    }

    /// Begin filling this buffer's contents with `value` repeated for
    /// every element — wraps `clEnqueueFillBuffer` on `launcher`'s
    /// queue. Useful for "zero out" / "reset to constant" patterns
    /// without uploading a host vector of N copies.
    ///
    /// Returns a lazy [`FillOp`] builder; pick a terminal
    /// ([`wait`](FillOp::wait), [`submit`](FillOp::submit), `.await`)
    /// to actually run.
    ///
    /// Takes `&mut self` because opencl3's `enqueue_fill_buffer`
    /// requires `&mut Buffer<T>` even though the cl_mem handle
    /// itself is shared / opaque at the OpenCL level — same shape
    /// as [`write`](Self::write).
    pub fn fill<'a, L: Launcher + ?Sized>(&'a mut self, launcher: &'a L, value: T) -> FillOp<'a, T>
    where
        T: Copy,
    {
        FillOp {
            queue: launcher.cl_queue(),
            buffer: &mut self.buffer,
            len: self.len,
            pattern: value,
            deps: Vec::new(),
            profile_cb: None,
        }
    }
}

// ── UploadOp / ReadOp / CopyOp builders ─────────────────────────────
//
// Same terminal-menu pattern as [`crate::op::LaunchOp`]: lazy builder
// captures everything the enqueue needs; the user picks `.wait()`
// (blocking), `.submit()` (returns Event, non-blocking), `.await`
// (registers a CL_COMPLETE callback, non-blocking), plus modifiers
// `.after(&Event)` (queue-side wait dependency) and `.profiled(|info|
// ...)` (timestamp callback). The terminal-decides-blocking trick:
// `.wait()` passes `CL_TRUE` straight to the enqueue — the driver
// blocks internally, no extra `event.wait()` roundtrip — whereas
// `.submit()` / `.await` pass `CL_FALSE` and return / register on
// the resulting event.

/// Lazy builder for `clEnqueueWriteBuffer`. Returned by
/// [`DeviceSlice::write`]. Writes into an existing buffer; for the
/// "alloc + write in one shot" convenience, see [`DeviceSlice::upload`].
pub struct WriteOp<'a, T> {
    queue: &'a CommandQueue,
    buffer: &'a mut ManuallyDrop<ClBuffer<T>>,
    dst_len: usize,
    data: &'a [T],
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a, T> WriteOp<'a, T> {
    /// Add a queue-side wait dependency. Chainable.
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    /// Add multiple wait-list events at once.
    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    /// Register a completion callback that receives the write's
    /// [`ProfilingInfo`]. Same FFI shim as
    /// [`LaunchOp::profiled`](crate::op::LaunchOp::profiled);
    /// requires the queue to have `CL_QUEUE_PROFILING_ENABLE`.
    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    /// Sync terminal — enqueue the write with `CL_TRUE`; the driver
    /// blocks until the buffer has been written.
    pub fn wait(self) -> Result<()> {
        if self.data.len() != self.dst_len {
            return Err(Error::LengthMismatch {
                src: self.data.len(),
                dst: self.dst_len,
            });
        }
        // SAFETY: CL_TRUE — the driver waits for the write to complete
        // before returning.
        let event = unsafe {
            self.queue.enqueue_write_buffer(
                &mut **self.buffer,
                CL_BLOCKING,
                0,
                self.data,
                &self.deps,
            )?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(())
    }

    /// Non-blocking terminal — enqueue the write with `CL_FALSE`,
    /// return the completion event. `data` must outlive the event.
    pub fn submit(self) -> Result<Event> {
        if self.data.len() != self.dst_len {
            return Err(Error::LengthMismatch {
                src: self.data.len(),
                dst: self.dst_len,
            });
        }
        // SAFETY: CL_FALSE; the write may complete after this call
        // returns. `data` must stay alive until the event fires.
        let event = unsafe {
            self.queue.enqueue_write_buffer(
                &mut **self.buffer,
                CL_NON_BLOCKING,
                0,
                self.data,
                &self.deps,
            )?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(event)
    }
}

/// Lazy builder for `clEnqueueReadBuffer`. Returned by
/// [`DeviceSlice::download`].
pub struct ReadOp<'a, T> {
    queue: &'a CommandQueue,
    buffer: &'a ClBuffer<T>,
    src_len: usize,
    dst: &'a mut [T],
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a, T> ReadOp<'a, T> {
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    /// Add multiple wait-list events at once.
    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    /// Sync terminal — enqueue the read with `CL_TRUE`; the driver
    /// blocks until `dst` has been filled.
    pub fn wait(self) -> Result<()> {
        if self.dst.len() != self.src_len {
            return Err(Error::LengthMismatch {
                src: self.src_len,
                dst: self.dst.len(),
            });
        }
        // SAFETY: CL_TRUE — the driver waits for the read to complete
        // before returning, so `dst` is fully populated on return.
        let event = unsafe {
            self.queue
                .enqueue_read_buffer(self.buffer, CL_BLOCKING, 0, self.dst, &self.deps)?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(())
    }

    /// Non-blocking terminal — enqueue the read with `CL_FALSE`,
    /// return the completion event. `dst` is only valid after the
    /// event fires; the caller must keep `dst` alive until then.
    pub fn submit(self) -> Result<Event> {
        if self.dst.len() != self.src_len {
            return Err(Error::LengthMismatch {
                src: self.src_len,
                dst: self.dst.len(),
            });
        }
        // SAFETY: CL_FALSE; the driver enqueues the read and returns
        // immediately. `dst` must outlive the returned event.
        let event = unsafe {
            self.queue
                .enqueue_read_buffer(self.buffer, CL_NON_BLOCKING, 0, self.dst, &self.deps)?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(event)
    }
}

/// Lazy builder for `clEnqueueCopyBuffer`. Returned by
/// [`DeviceSlice::copy_to`]. There's no `CL_BLOCKING` flag on
/// `clEnqueueCopyBuffer`, so `.wait()` is non-blocking enqueue +
/// `event.wait()` (same as [`LaunchOp`](crate::op::LaunchOp)).
pub struct CopyOp<'a, T> {
    queue: &'a CommandQueue,
    src: &'a ClBuffer<T>,
    dst: &'a mut ClBuffer<T>,
    src_len: usize,
    dst_len: usize,
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a, T> CopyOp<'a, T> {
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    /// Add multiple wait-list events at once.
    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    /// Sync terminal — enqueue + wait on the resulting event.
    pub fn wait(self) -> Result<()> {
        let event = self.into_event()?;
        event.wait()?;
        Ok(())
    }

    /// Non-blocking terminal — enqueue and return the completion event.
    pub fn submit(self) -> Result<Event> {
        self.into_event()
    }

    pub(crate) fn into_event(self) -> Result<Event> {
        if self.src_len != self.dst_len {
            return Err(Error::LengthMismatch {
                src: self.src_len,
                dst: self.dst_len,
            });
        }
        let bytes = self.src_len * std::mem::size_of::<T>();
        // SAFETY: `enqueue_copy_buffer` is `unsafe` because src/dst
        // must belong to the queue's context. Length equality checked
        // above; context cross-checking is on the caller (pocl panics
        // on mismatch — preferable to a silent miscopy).
        let event = unsafe {
            self.queue
                .enqueue_copy_buffer(self.src, self.dst, 0, 0, bytes, &self.deps)?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(event)
    }
}

/// Lazy builder for `clEnqueueMigrateMemObjects`. Returned by
/// [`DeviceSlice::migrate`]. Target device is implicit — it's
/// `launcher`'s queue's device.
///
/// Always uses flags = 0 (default migrate-to-this-queue's-device
/// semantics; preserves current contents). The hint
/// `CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED` for cases where the
/// caller knows the buffer's data isn't needed could be added later
/// as an opt-in modifier; for now the conservative default is right.
pub struct MigrateOp<'a, T> {
    queue: &'a CommandQueue,
    buffer: &'a ClBuffer<T>,
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a, T> MigrateOp<'a, T> {
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    /// Add multiple wait-list events at once.
    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    /// Sync terminal — enqueue the migrate and wait on its event.
    pub fn wait(self) -> Result<()> {
        let event = self.into_event()?;
        event.wait()?;
        Ok(())
    }

    /// Non-blocking terminal — enqueue and return the completion event.
    pub fn submit(self) -> Result<Event> {
        self.into_event()
    }

    pub(crate) fn into_event(self) -> Result<Event> {
        // SAFETY: `enqueue_migrate_mem_object` is unsafe because the
        // mem_objects pointer must point to a valid `cl_mem` for the
        // queue's context. We pass exactly one `cl_mem` from a
        // `ClBuffer` we own a reference to, which is alive for the
        // call. The buffer must belong to the queue's context — the
        // caller's responsibility, same constraint as `enqueue_copy_buffer`.
        let mem_handle: cl_mem = self.buffer.get();
        let event = unsafe {
            self.queue.enqueue_migrate_mem_object(
                1,
                &mem_handle as *const cl_mem,
                0, // flags: default = migrate to queue's device, preserve content
                &self.deps,
            )?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(event)
    }
}

/// Metadata-only `Debug` — never reads device memory (would block /
/// fault) and doesn't require `T: Debug` (the element type doesn't
/// flow through). Useful for `Result<DeviceSlice<T>, _>::expect_err`
/// / `.unwrap` and generic `{:?}` chain-output debugging.
impl<T> fmt::Debug for DeviceSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceSlice")
            .field("len", &self.len)
            .field("element_size", &std::mem::size_of::<T>())
            .finish_non_exhaustive()
    }
}

/// Lazy builder for `clEnqueueFillBuffer`. Returned by
/// [`DeviceSlice::fill`]. The pattern is a single `T` value; the
/// fill spans the whole buffer (`len * size_of::<T>()` bytes).
pub struct FillOp<'a, T: Copy> {
    queue: &'a CommandQueue,
    buffer: &'a mut ManuallyDrop<ClBuffer<T>>,
    len: usize,
    pattern: T,
    deps: Vec<cl_event>,
    profile_cb: Option<ProfileCb>,
}

impl<'a, T: Copy> FillOp<'a, T> {
    pub fn after(mut self, event: &Event) -> Self {
        self.deps.push(event.get());
        self
    }

    pub fn after_all<'e, I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = &'e Event>,
    {
        self.deps.extend(events.into_iter().map(|e| e.get()));
        self
    }

    pub fn profiled<F>(mut self, cb: F) -> Self
    where
        F: FnOnce(Result<ProfilingInfo>) + Send + 'static,
    {
        self.profile_cb = Some(Box::new(cb));
        self
    }

    pub fn wait(self) -> Result<()> {
        let event = self.into_event()?;
        event.wait()?;
        Ok(())
    }

    pub fn submit(self) -> Result<Event> {
        self.into_event()
    }

    pub(crate) fn into_event(self) -> Result<Event> {
        // SAFETY: `enqueue_fill_buffer` is unsafe because the buffer
        // must belong to the queue's context. Same constraint as
        // `enqueue_copy_buffer` / `enqueue_migrate_mem_object`. The
        // pattern is byte-copied (via opencl3's slice-of-pattern
        // shape) across the whole buffer.
        let event = unsafe {
            self.queue.enqueue_fill_buffer(
                &mut **self.buffer,
                std::slice::from_ref(&self.pattern),
                0,
                self.len * std::mem::size_of::<T>(),
                &self.deps,
            )?
        };
        if let Some(cb) = self.profile_cb {
            register_profiling_callback(&event, cb)?;
        }
        Ok(event)
    }
}

impl<T> Buffer<T> for DeviceSlice<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn ctx(&self) -> &Context {
        &self.ctx
    }
}

// `HostBuffer` (`CL_MEM_ALLOC_HOST_PTR` + persistent map) was
// removed 2026-05-29: per the CL spec on persistently-mapped
// buffers, the application must unmap before any kernel reads or
// writes the buffer, so the "zero-copy host-and-kernel share
// memory" semantics it was reaching for were UB by construction.
// Use [`USMSlice`](crate::USMSlice) (fine-grain system SVM) for
// that role on supporting devices, or [`MappedSlice`](crate::MappedSlice)
// with the map-guard pattern for coarse-grain SVM.
