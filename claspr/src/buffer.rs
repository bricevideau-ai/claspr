//! Typed device-side buffers and the [`Buffer`] trait that abstracts
//! over them.
//!
//! One tier lives in this module: [`DeviceSlice<T>`] —
//! `CL_MEM_READ_WRITE`, accessed via [`upload`](DeviceSlice::write)
//! / [`download`](DeviceSlice::read). The host-mapped tier (SVM
//! / [`MappedSlice`](crate::mapped::MappedSlice)) lives in
//! [`crate::mapped`].
//!
//! See the [`Buffer`] trait's own docs for what it does and does
//! not abstract over.

use crate::access::{FillStrategy, Fillable, HostReadable, HostWritable, MemMode, ReadWrite};
use crate::context::Context;
use crate::error::{Error, Result};
use crate::fill_kernel;
use crate::queue::Launcher;

use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer as ClBuffer, CL_MEM_COPY_HOST_PTR, ClMem, release_mem_object};
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_event, cl_mem};
use std::fmt;
use std::marker::PhantomData;
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
/// - [`DeviceSlice::write`] / [`DeviceSlice::read`] enqueue a
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
/// Construct via [`DeviceSlice::alloc_zero`] (zero-initialised) or
/// [`DeviceSlice::from_slice`] (with initial host data). Read back via
/// [`DeviceSlice::read`]. The
/// [`alloc_uninit`](DeviceSlice::alloc_uninit) escape hatch returns a
/// [`DeviceSliceUninit<T, M>`] type-stated wrapper — see its docs for
/// the safe transition path and the `unsafe assume_init()` escape
/// hatch.
///
/// Host code never sees the bytes directly — for that, use
/// [`crate::mapped::MappedSlice<T>`] (coarse-grain SVM) or
/// [`crate::usm::USMSlice<T>`] (fine-grain-system SVM over a host
/// `Vec<T>`).
///
/// [`KernelArg`]: crate::launch::KernelArg
pub struct DeviceSlice<T, M: MemMode = ReadWrite> {
    /// `ManuallyDrop` so opencl3's `Buffer::drop` (which panics on
    /// release failure) doesn't fire — our own [`Drop`] impl below
    /// calls `release_mem_object` and records into the context's
    /// sticky-error counter on failure instead.
    pub(crate) buffer: ManuallyDrop<ClBuffer<T>>,
    pub(crate) len: usize,
    pub(crate) ctx: Context,
    /// Type-level access mode tag (`ReadWrite` / `ReadOnly` /
    /// `Frozen` / etc.). Zero-sized; encoded only at the type level
    /// to gate method availability and the `clCreateBuffer` flags
    /// chosen at alloc time. See [`crate::access`].
    pub(crate) _mode: PhantomData<fn() -> M>,
}

impl<T, M: MemMode> Drop for DeviceSlice<T, M> {
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

impl<T: Default + Copy + Send + Sync + 'static, M: MemMode + Fillable + Send + 'static>
    DeviceSlice<T, M>
{
    /// Allocate a device buffer of `len` elements, zero-initialised
    /// via `clEnqueueFillBuffer(T::default())` on the context's
    /// default queue. Blocks until the fill completes.
    ///
    /// The `T: Default + Copy` bound makes the buffer's contents a
    /// valid `T` value before any read; the `M: KernelWritable` bound
    /// limits this constructor to markers whose kernel-side flag
    /// permits a runtime fill (excludes [`crate::ReadOnly`] and
    /// [`crate::Frozen`] — those use [`from_slice`](Self::from_slice)
    /// to bake in the initial data at create time instead).
    ///
    /// Matches [`MappedSlice::alloc_zero`](crate::MappedSlice::alloc_zero)
    /// and [`USMSlice::alloc_zero`](crate::USMSlice::alloc_zero).
    ///
    /// Internally a `clCreateBuffer` + a synchronous fill. The
    /// [`alloc_uninit`](Self::alloc_uninit) escape hatch returns a
    /// type-stated [`DeviceSliceUninit`] wrapper that requires
    /// explicit initialization before any read — see its docs.
    ///
    /// **Honest name**: this is `alloc + zero-init via fill`, not a
    /// pure `clCreateBuffer`. The fill cost (and the [`Fillable`]
    /// bound it imposes on the marker) is visible at the call site.
    pub fn alloc_zero(ctx: &Context, len: usize) -> Result<Self> {
        // SAFETY: we immediately overwrite every byte via the
        // synchronous `fill` below before returning. No path from
        // here can observe the uninit bytes.
        let slice = unsafe { Self::alloc_uninit(ctx, len)?.assume_init() };
        // `fill` is now a graph node that consumes the buffer and rebinds it out.
        let slice = slice.fill(T::default()).wait()?;
        Ok(slice)
    }
}

impl<T, M: MemMode> DeviceSlice<T, M> {
    /// Allocate a device buffer of `len` elements, leaving the bytes
    /// uninitialised. Returns a [`DeviceSliceUninit<T, M>`] wrapper
    /// instead of a bare `DeviceSlice` — the type-state blocks host
    /// reads (`.read()` / `download!`) at compile time so unintended
    /// uninit-byte observation is a type error rather than UB.
    ///
    /// Transition to an initialised [`DeviceSlice<T, M>`] via one of
    /// the wrapper's methods (`.fill(value)`, `.write(data)`,
    /// `unsafe fn assume_init()`), or via a Tier 2 chain that
    /// applies one of those. The fill / write paths are safe; the
    /// `assume_init` escape hatch is unsafe because rust-gpu has no
    /// `MaybeUninit` story — a kernel that reads uninit bytes
    /// interprets them as a `T` value (arbitrary garbage for
    /// numeric `T`; UB for `T` with invalid bit patterns like
    /// `bool` / `NonZeroU32` / niche-optimised enums).
    ///
    /// **No marker bound** — the type-state wrapper IS the safety
    /// gate. Any `M` works at construction; subsequent
    /// initialization paths are gated by their own marker bounds
    /// (e.g. `.fill()` requires [`Fillable`], `.write()` requires
    /// [`HostWritable`]).
    ///
    /// Cheaper than [`alloc_zero`](Self::alloc_zero) when the caller
    /// will overwrite the buffer immediately anyway (skips the
    /// initial fill).
    pub fn alloc_uninit(ctx: &Context, len: usize) -> Result<DeviceSliceUninit<T, M>> {
        // SAFETY: passing a null host pointer means OpenCL allocates
        // fresh device memory and ignores the host-pointer contract
        // that makes `Buffer::create` generally unsafe.
        let buffer =
            unsafe { ClBuffer::<T>::create(ctx.raw_context(), M::FLAGS, len, ptr::null_mut())? };
        Ok(DeviceSliceUninit {
            inner: DeviceSlice {
                buffer: ManuallyDrop::new(buffer),
                len,
                ctx: ctx.clone(),
                _mode: PhantomData,
            },
        })
    }
}

/// Type-state wrapper returned by
/// [`DeviceSlice::alloc_uninit`]: the bytes are uninitialised, host
/// reads are statically blocked, and the user must transition to an
/// initialised [`DeviceSlice<T, M>`] via either
/// [`unsafe fn assume_init`](Self::assume_init) (kernel-write-only
/// pattern; caller vouches every byte gets written before any read)
/// or by chaining through Tier 2 (eager-graph) ops that consume
/// the wrapper and produce the initialised buffer.
///
/// The wrapper has no `read` / `download` / `acquire_host_view`
/// methods — attempting any of those is a compile error rather than
/// reading uninit bytes (and possibly invoking UB for `T` with
/// invalid bit patterns).
///
/// **Note**: safe consuming `.fill()` / `.write()` methods on this
/// wrapper are an open follow-up; for now the supported transitions
/// are `assume_init` + Tier 1 fill/write, or the Tier 2 chain
/// equivalents.
pub struct DeviceSliceUninit<T, M: MemMode = ReadWrite> {
    inner: DeviceSlice<T, M>,
}

impl<T, M: MemMode> DeviceSliceUninit<T, M> {
    /// Skip safe initialization and assert that this buffer has
    /// been (or will be) fully written by some other path —
    /// typically a kernel arg launched on it that writes every
    /// slot. Escape hatch for the rust-gpu MaybeUninit gap.
    ///
    /// # Safety
    ///
    /// The caller asserts that every byte of the buffer will be
    /// written by SOME path (kernel, manual SVM copy, etc.) before
    /// any read can observe the bytes. For numeric `T` an
    /// uninit-byte read is arbitrary garbage; for `T` with invalid
    /// bit patterns (`bool`, `NonZeroU32`, niche-optimised enums)
    /// it is UB.
    pub unsafe fn assume_init(self) -> DeviceSlice<T, M> {
        self.inner
    }

    /// Length in elements — same as the eventual `DeviceSlice`'s
    /// length.
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// True when length is zero.
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// The context the buffer was allocated on.
    pub fn ctx(&self) -> &Context {
        &self.inner.ctx
    }
}

impl<T, M: MemMode> fmt::Debug for DeviceSliceUninit<T, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceSliceUninit")
            .field("len", &self.inner.len)
            .field("element_size", &std::mem::size_of::<T>())
            .finish_non_exhaustive()
    }
}

impl<T, M: MemMode + HostWritable> DeviceSlice<T, M>
where
    T: Send + Sync + 'static,
    M: Send + 'static,
{
    /// Write `src` into this buffer, returning the
    /// [`WriteDevice`](crate::eager::WriteDevice) graph node (a
    /// [`DeviceOp`](crate::DeviceOp)). It is usable standalone — `let buf =
    /// buf.write(data).wait()?;` (the buffer moves in and rebinds out so it can
    /// be reused) — or composed in a graph via `.and_then(...)` / `bundle!`.
    ///
    /// `src.len()` must equal `self.len()` (checked at terminal time — the
    /// terminals return [`Error::LengthMismatch`] otherwise).
    ///
    /// **Marker constraint:** `M: HostWritable`. Compiles for `ReadWrite` /
    /// `ReadOnly`. Markers that mark the buffer host-read-only (`HostReadOnly`,
    /// `Frozen`) or host-no-access (`DeviceScratch`) intentionally don't allow
    /// `write`.
    pub fn write<S>(self, src: S) -> crate::eager::WriteDevice<T, M>
    where
        S: Into<crate::transfer::UploadSource<T>>,
    {
        crate::eager::write(self, src)
    }

    /// Blocking, **borrowing** host→device upload — the synchronous
    /// counterpart to the async owned [`write`](Self::write).
    ///
    /// Borrows `data` as `&[T]`, enqueues a `clEnqueueWriteBuffer` with
    /// `CL_BLOCKING` on the buffer's own context default queue, and waits
    /// inline: the host copy is complete before the call returns, so the
    /// source never has to outlive the call. That means **no ownership
    /// transfer and no keep-alive allocation** — both this buffer and
    /// `data` stay fully usable afterwards. Use this when you want to keep
    /// the source (`buf.write_sync(&data)?;` instead of
    /// `buf.write(data.clone())`).
    ///
    /// Tradeoffs vs [`write`](Self::write):
    /// - **Blocks the calling thread** until the DMA finishes — no overlap
    ///   with other device work. For pipelined / overlapped uploads (the
    ///   non-blocking `clEnqueueWriteBuffer` whose source is owned and held
    ///   alive via a drop-callback), use [`write`](Self::write).
    /// - **Not a graph node** — it returns a plain `Result<()>`, not a
    ///   [`DeviceOp`](crate::DeviceOp), so it can't be `.and_then(...)`-ed
    ///   into a chain or `bundle!`-d. It's a standalone side effect on the
    ///   buffer.
    ///
    /// `data.len()` must equal `self.len()` (returns
    /// [`Error::LengthMismatch`] otherwise).
    ///
    /// **Marker constraint:** `M: HostWritable` — identical to
    /// [`write`](Self::write). A blocking write to a host-read-only buffer
    /// (`HostReadOnly`, `Frozen`) or a host-no-access buffer
    /// (`DeviceScratch`) is rejected at compile time.
    pub fn write_sync(&mut self, data: &[T]) -> Result<()> {
        let ctx = self.ctx.clone();
        // Blocking enqueue (`CL_BLOCKING`): the driver waits internally, so
        // the borrowed `data` only needs to outlive this call — no keep-alive,
        // no ownership transfer. Reuses the same raw helper the eager
        // `WriteDevice::Blocking` terminal uses.
        write_buffer_enqueue(self, &ctx, data, true, &[])?;
        Ok(())
    }
}

impl<T, M: MemMode + HostWritable + HostReadable> DeviceSlice<T, M> {
    /// Begin a host read+write map of this buffer (zero-copy
    /// alternative to a write-then-read round trip). Returns a lazy
    /// [`DeviceMapMutOp`] builder — terminals match
    /// [`DeviceSlice::map`] but the resulting guard is
    /// [`DeviceMappedWriteGuard`] (`DerefMut<Target = [T]>`).
    ///
    /// `&mut self` provides the borrow-checker exclusivity that
    /// `DerefMut` needs.
    ///
    /// **Marker constraint:** `M: HostWritable + HostReadable` —
    /// the underlying `clEnqueueMapBuffer(CL_MAP_READ | CL_MAP_WRITE)`
    /// requires both. Same shape as
    /// [`MappedSlice::map_mut`](crate::MappedSlice::map_mut).
    pub fn map_mut(&mut self) -> DeviceMapMutOp<'_, T, M> {
        DeviceMapMutOp { owner: self }
    }
}

impl<T, M: MemMode + HostReadable> DeviceSlice<T, M> {
    /// Read this buffer into the caller slice `dst`, returning the
    /// [`ReadInto`](crate::eager::ReadInto) graph node (a
    /// [`DeviceOp`](crate::DeviceOp)). Usable standalone — `let buf =
    /// buf.read(&mut dst).wait()?;` (the buffer rebinds out for reuse) — or
    /// composed in a graph. For a freshly-allocated `Vec` output instead of a
    /// caller slice, use [`download`](crate::eager::download).
    ///
    /// `dst.len()` must equal `self.len()` (checked at terminal time —
    /// the terminals return [`Error::LengthMismatch`] otherwise).
    ///
    /// **Marker constraint:** `M: HostReadable`. Compiles for every
    /// marker except `DeviceScratch` (`CL_MEM_HOST_NO_ACCESS` —
    /// host can't touch the bytes).
    pub fn read<'d>(self, dst: &'d mut [T]) -> crate::eager::ReadInto<'d, T, M>
    where
        T: Send + 'static,
        M: Send + 'static,
    {
        crate::eager::read_into(self, dst)
    }

    /// Begin a host read map of this buffer (zero-copy alternative to
    /// [`read`](Self::read)). Returns a lazy [`DeviceMapOp`] builder
    /// — pick a terminal:
    ///
    /// - [`wait`](DeviceMapOp::wait): blocking
    ///   `clEnqueueMapBuffer(CL_TRUE, CL_MAP_READ)`, returns a
    ///   [`DeviceMappedReadGuard`] (`Deref<Target = [T]>`,
    ///   unmaps on Drop).
    /// - [`submit`](DeviceMapOp::submit): non-blocking
    ///   `clEnqueueMapBuffer(CL_FALSE, CL_MAP_READ)`, returns a
    ///   [`DeviceMapReadPending`] carrying the map event; consume via
    ///   [`DeviceMapReadPending::wait`] for the guard.
    ///
    /// **Marker constraint:** `M: HostReadable`. SVM analog:
    /// [`MappedSlice::map`](crate::MappedSlice::map).
    pub fn map(&self) -> DeviceMapOp<'_, T, M> {
        DeviceMapOp { owner: self }
    }
}

impl<T, M: MemMode> DeviceSlice<T, M> {
    /// Borrow the underlying opencl3 [`ClBuffer`](opencl3::memory::Buffer)
    /// for cases that need direct OpenCL access.
    pub fn buffer(&self) -> &ClBuffer<T> {
        &self.buffer
    }

    /// Size of the backing buffer in bytes (`len * size_of::<T>()`). Used by the
    /// record/replay path to bake fill/copy sizes.
    pub(crate) fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Device-to-device copy from `self` into `dst`, returning the
    /// [`CopyTo2`](crate::eager::CopyTo2) graph node (a
    /// [`DeviceOp`](crate::DeviceOp)) whose output is `(src, dst)`. Usable
    /// standalone via `.sync(&ctx)` / `.wait_on(&queue)` or composed in a graph.
    ///
    /// Both buffers must be on the same `Context` — OpenCL's
    /// `clEnqueueCopyBuffer` only works within one context. For
    /// cross-context transfers, download to host then re-upload.
    pub fn copy_to<M2>(
        self,
        dst: DeviceSlice<T, M2>,
    ) -> crate::eager::CopyTo2<Self, DeviceSlice<T, M2>>
    where
        T: Send + 'static,
        M: Send + 'static,
        M2: MemMode + Send + 'static,
    {
        crate::eager::eager_copy_to(self, dst)
    }

    /// Migrate this buffer onto `device` via `clEnqueueMigrateMemObjects`,
    /// returning the [`TransferToDevice`](crate::eager::TransferToDevice) graph
    /// node (a [`DeviceOp`](crate::DeviceOp)) that yields the migrated buffer.
    /// The migrate is enqueued on `device`'s default out-of-order queue.
    ///
    /// On topologies where all devices in the context share physical
    /// memory (sub-devices of one CPU, integrated GPUs in a single
    /// context) the migration is typically a no-op. On distributed
    /// topologies (two dGPUs in one `cl_context`) it triggers a real
    /// memory transfer. Either way the migrate is a non-blocking queue command;
    /// downstream stages wait on its event via the graph's carried deps.
    pub fn migrate(self, device: &crate::Device) -> crate::eager::TransferToDevice<T, M>
    where
        T: Send + 'static,
        M: Send + 'static,
    {
        crate::eager::transfer_to_device(self, device)
    }

    /// Fill this buffer's contents with `value` repeated for every element
    /// (wraps `clEnqueueFillBuffer`, or a built-in fill kernel for kernel-RO
    /// markers). Useful for "zero out" / "reset to constant" patterns without
    /// uploading a host vector of N copies.
    ///
    /// Returns the [`Fill`](crate::eager::Fill) graph node (a
    /// [`DeviceOp`](crate::DeviceOp)). Usable standalone — `let buf =
    /// buf.fill(v).wait()?;` (the buffer rebinds out for reuse) — or composed in
    /// a graph.
    ///
    /// **Marker constraint:** `M: Fillable`. Runtime-side fill counts as a write
    /// at the OpenCL level, so kernel-RO markers (`ReadOnly`, `Frozen`) can't be
    /// filled.
    pub fn fill(self, value: T) -> crate::eager::Fill<T, M>
    where
        T: Copy + Send + Sync + 'static,
        M: Fillable + Send + 'static,
    {
        crate::eager::fill(self, value)
    }
}

// ── from_slice / from_vec — bake in initial data at create time ────

impl<T: Copy, M: MemMode> DeviceSlice<T, M> {
    /// Create a device buffer whose contents are copied from `data`
    /// at construction time via `CL_MEM_COPY_HOST_PTR`, with the
    /// marker's access flags applied (`M::FLAGS`).
    ///
    /// Works for any marker — `CL_MEM_COPY_HOST_PTR` is a one-shot
    /// creation-time copy that doesn't interact with the host-access
    /// or kernel-access flags applied to subsequent operations. So
    /// `DeviceSlice::<u32, Frozen>::from_slice` bakes in immutable
    /// initial data; `DeviceSlice::<u32, DeviceConstant>::from_slice`
    /// bakes in initial data the host can update later via `.write()`;
    /// `DeviceSlice::<u32>::from_slice` (default `ReadWrite`) is just
    /// alloc+write in one call.
    ///
    /// For [`crate::Frozen`] this is the ONLY constructor (both axes
    /// are read-only — no alloc-and-fill path). For other markers
    /// it's a convenience over `alloc + write`.
    pub fn from_slice(ctx: &Context, data: &[T]) -> Result<Self> {
        // SAFETY: `data.as_ptr()` is valid for `data.len() * size_of::<T>()`
        // bytes; `CL_MEM_COPY_HOST_PTR` means the runtime copies the
        // bytes into the new allocation immediately and does NOT
        // retain the pointer. So `data` doesn't need to outlive the
        // returned DeviceSlice.
        let buffer = unsafe {
            ClBuffer::<T>::create(
                ctx.raw_context(),
                M::FLAGS | CL_MEM_COPY_HOST_PTR,
                data.len(),
                data.as_ptr() as *mut std::ffi::c_void,
            )?
        };
        Ok(DeviceSlice {
            buffer: ManuallyDrop::new(buffer),
            len: data.len(),
            ctx: ctx.clone(),
            _mode: PhantomData,
        })
    }

    /// Convenience for `from_slice(ctx, &data)` that takes the Vec
    /// by value — symmetric with the `from_*` constructors on the
    /// other tiers.
    pub fn from_vec(ctx: &Context, data: Vec<T>) -> Result<Self> {
        Self::from_slice(ctx, &data)
    }
}

/// Metadata-only `Debug` — never reads device memory (would block /
/// fault) and doesn't require `T: Debug` (the element type doesn't
/// flow through). Useful for `Result<DeviceSlice<T>, _>::expect_err`
/// / `.unwrap` and generic `{:?}` chain-output debugging.
impl<T, M: MemMode> fmt::Debug for DeviceSlice<T, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceSlice")
            .field("len", &self.len)
            .field("element_size", &std::mem::size_of::<T>())
            .finish_non_exhaustive()
    }
}

/// Launch the built-in fill kernel for an OpenCL buffer (DeviceSlice
/// backing memory). `pattern_size = size_of::<T>()` selects between
/// the fast-path per-size kernels (1/2/4/8/16 bytes) and the
/// byte-generic fallback. Returns the launch event.
pub(crate) fn fill_via_kernel_buffer<T: Copy, L: Launcher + ?Sized>(
    ctx: &Context,
    launcher: &L,
    buffer: &ClBuffer<T>,
    pattern: &T,
    count: usize,
    deps: &[cl_event],
) -> Result<Event> {
    let pattern_size = std::mem::size_of::<T>();
    let count_u32 =
        u32::try_from(count).map_err(|_| Error::InvalidArgument("fill count exceeds u32::MAX"))?;
    let program = ctx.fill_program()?;

    if let Some(name) = fill_kernel::fast_path_kernel_name(pattern_size) {
        let kernel = Kernel::create(program, name)?;
        let mut exec = ExecuteKernel::new(&kernel);
        // SAFETY: kernel arg 0 expects a `__global X*` of element
        // size `pattern_size`. We pass our buffer's ClMem; the
        // kernel writes `count` elements. arg 1 is the pattern by
        // value (size matches); arg 2 is the element count.
        unsafe {
            exec.set_arg(buffer);
            exec.set_arg(pattern);
            exec.set_arg(&count_u32);
            exec.set_global_work_size(count);
            exec.set_event_wait_list(deps);
            Ok(exec.enqueue_nd_range(launcher.cl_queue())?)
        }
    } else {
        // Byte-generic path: allocate a tiny pattern buffer, copy
        // pattern bytes in, launch claspr_fill_bytes.
        let pattern_size_u32 = u32::try_from(pattern_size)
            .map_err(|_| Error::InvalidArgument("fill pattern size exceeds u32::MAX"))?;
        // SAFETY: pattern is a live &T whose byte representation we
        // read for `pattern_size` bytes.
        let pattern_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pattern as *const T as *const u8, pattern_size) };
        // SAFETY: create a fresh buffer in ctx's CL context, then
        // blocking-write the pattern bytes; lifetime ends with this
        // function — the kernel launch is serialized after the
        // write because both go through `launcher.cl_queue`.
        let mut pattern_buf = unsafe {
            ClBuffer::<u8>::create(
                ctx.raw_context(),
                opencl3::memory::CL_MEM_READ_ONLY,
                pattern_size,
                ptr::null_mut(),
            )?
        };
        let _write_evt = unsafe {
            launcher.cl_queue().enqueue_write_buffer(
                &mut pattern_buf,
                CL_BLOCKING,
                0,
                pattern_bytes,
                &[],
            )?
        };
        let kernel = Kernel::create(program, fill_kernel::KERNEL_BYTES)?;
        let mut exec = ExecuteKernel::new(&kernel);
        // SAFETY: arg 0 = data buffer (__global uchar*), arg 1 =
        // pattern buffer (__global const uchar*), arg 2 = pattern
        // byte count, arg 3 = slot count.
        let event = unsafe {
            exec.set_arg(buffer);
            exec.set_arg(&pattern_buf);
            exec.set_arg(&pattern_size_u32);
            exec.set_arg(&count_u32);
            exec.set_global_work_size(count);
            exec.set_event_wait_list(deps);
            exec.enqueue_nd_range(launcher.cl_queue())?
        };
        // pattern_buf drops at end of fn; OpenCL retains the cl_mem
        // internally for the in-flight kernel until completion.
        Ok(event)
    }
}

// ── Raw enqueue helpers — the fold seam for the eager buffer ops ──────
//
// Each helper is the `clEnqueue*` body the matching Tier-1 builder used to own
// (in its `wait_on`/`submit_on`/`into_event`), lifted out so the eager graph
// nodes (`Fill`/`Download`/`ReadInto`/`WriteDevice`/`TransferToDevice` in
// `eager.rs`) can enqueue directly against a `DeviceSlice` without round-tripping
// through a borrow-based builder. `blocking` selects `CL_BLOCKING` vs
// `CL_NON_BLOCKING` so the eager op's `ExecMode::Blocking` terminal path keeps
// the native no-event blocking enqueue the builder's `wait_on` had. `deps` is the
// already-collected `cl_event` wait-list (the eager op flattens its `Deps` to
// raw handles, held alive across the call).

/// Raw `clEnqueueWriteBuffer` over `buf` — body of the former `WriteOp`.
pub(crate) fn write_buffer_enqueue<T, M, L>(
    buf: &mut DeviceSlice<T, M>,
    launcher: &L,
    data: &[T],
    blocking: bool,
    deps: &[cl_event],
) -> Result<Event>
where
    M: MemMode,
    L: Launcher + ?Sized,
{
    if data.len() != buf.len {
        return Err(Error::LengthMismatch {
            src: data.len(),
            dst: buf.len,
        });
    }
    let cl_blocking = if blocking {
        CL_BLOCKING
    } else {
        CL_NON_BLOCKING
    };
    // SAFETY: `enqueue_write_buffer` requires the buffer belong to the queue's
    // context. `blocking` selects whether the driver waits internally; for the
    // non-blocking case the caller keeps `data` alive until the event fires.
    let event = unsafe {
        launcher
            .cl_queue()
            .enqueue_write_buffer(&mut buf.buffer, cl_blocking, 0, data, deps)?
    };
    Ok(event)
}

/// Raw `clEnqueueReadBuffer` from `buf` into `dst` — body of the former `ReadOp`.
pub(crate) fn read_buffer_enqueue<T, M, L>(
    buf: &DeviceSlice<T, M>,
    launcher: &L,
    dst: &mut [T],
    blocking: bool,
    deps: &[cl_event],
) -> Result<Event>
where
    M: MemMode,
    L: Launcher + ?Sized,
{
    if dst.len() != buf.len {
        return Err(Error::LengthMismatch {
            src: buf.len,
            dst: dst.len(),
        });
    }
    let cl_blocking = if blocking {
        CL_BLOCKING
    } else {
        CL_NON_BLOCKING
    };
    // SAFETY: same context constraint as the write path; `blocking` selects the
    // internal-wait vs return-event behaviour.
    let event = unsafe {
        launcher
            .cl_queue()
            .enqueue_read_buffer(&buf.buffer, cl_blocking, 0, dst, deps)?
    };
    Ok(event)
}

/// Raw `clEnqueueFillBuffer` (or kernel fill) over `buf` — body of the former
/// `FillOp::into_event`. Always non-blocking enqueue: the eager `Fill` op's
/// `Blocking` terminal waits on the returned event (fill has no `CL_BLOCKING`
/// flag, exactly as the builder's `wait_on` did `submit + event.wait()`).
pub(crate) fn fill_buffer_enqueue<T, M, L>(
    buf: &mut DeviceSlice<T, M>,
    launcher: &L,
    pattern: T,
    deps: &[cl_event],
) -> Result<Event>
where
    T: Copy,
    M: MemMode + Fillable,
    L: Launcher + ?Sized,
{
    let event = match M::FILL_STRATEGY {
        FillStrategy::Runtime => {
            // SAFETY: the buffer must belong to the queue's context; the pattern
            // is byte-copied across the whole buffer.
            unsafe {
                launcher.cl_queue().enqueue_fill_buffer(
                    &mut buf.buffer,
                    std::slice::from_ref(&pattern),
                    0,
                    buf.len * std::mem::size_of::<T>(),
                    deps,
                )?
            }
        }
        FillStrategy::DeviceKernel => {
            fill_via_kernel_buffer(&buf.ctx, launcher, &buf.buffer, &pattern, buf.len, deps)?
        }
    };
    Ok(event)
}

/// Raw `clEnqueueMigrateMemObjects` for `buf` — body of the former `MigrateOp`.
/// Non-blocking (migrate has no `CL_BLOCKING` flag); the caller waits on the
/// returned event if it needs blocking.
pub(crate) fn migrate_buffer_enqueue<T, M, L>(
    buf: &DeviceSlice<T, M>,
    launcher: &L,
    deps: &[cl_event],
) -> Result<Event>
where
    M: MemMode,
    L: Launcher + ?Sized,
{
    let mem_handle: cl_mem = buf.buffer.get();
    // SAFETY: exactly one valid `cl_mem` from a buffer we hold a reference to,
    // alive for the call; flags = 0 (migrate to queue's device, preserve content).
    let event = unsafe {
        launcher
            .cl_queue()
            .enqueue_migrate_mem_object(1, &mem_handle as *const cl_mem, 0, deps)?
    };
    Ok(event)
}

/// Raw `clEnqueueCopyBuffer` from `src` into `dst` — body of the former
/// `CopyOp::into_event`. Non-blocking; the caller waits on the event if needed.
pub(crate) fn copy_buffer_enqueue<T, M1, M2, L>(
    src: &DeviceSlice<T, M1>,
    dst: &mut DeviceSlice<T, M2>,
    launcher: &L,
    deps: &[cl_event],
) -> Result<Event>
where
    M1: MemMode,
    M2: MemMode,
    L: Launcher + ?Sized,
{
    if src.len != dst.len {
        return Err(Error::LengthMismatch {
            src: src.len,
            dst: dst.len,
        });
    }
    let bytes = src.len * std::mem::size_of::<T>();
    // SAFETY: `enqueue_copy_buffer` requires src/dst belong to the queue's
    // context. Length equality checked above.
    let event = unsafe {
        launcher
            .cl_queue()
            .enqueue_copy_buffer(&src.buffer, &mut dst.buffer, 0, 0, bytes, deps)?
    };
    Ok(event)
}

impl<T, M: MemMode> Buffer<T> for DeviceSlice<T, M> {
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

// ── Map builders + guards (cl_mem path) ─────────────────────────────
//
// Mirrors the SVM map surface in `crate::mapped`. Two terminals on
// each builder (`wait` blocking, `submit` non-blocking returning a
// pending); guards Deref / DerefMut to `[T]` and unmap on Drop.
//
// `cl_mem` retains internally for every enqueued op, so
// `clReleaseMemObject` in `DeviceSlice::Drop` doesn't need a
// `last_use`-style wait-list — the runtime gates the release on
// in-flight uses. For users who want to thread the unmap event into
// a cross-queue chain, `release(self) -> Result<Event>` consumes the
// guard and returns the unmap event explicitly instead of dropping it.

use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE};
use std::ops::{Deref, DerefMut};

use crate::map_primitive;

/// Lazy builder for [`DeviceSlice::map`]. Borrows the source buffer;
/// pick a terminal — [`wait`](DeviceMapOp::wait) (blocking) or
/// [`submit`](DeviceMapOp::submit) (non-blocking).
pub struct DeviceMapOp<'a, T, M: MemMode> {
    owner: &'a DeviceSlice<T, M>,
}

impl<'a, T, M: MemMode + HostReadable> DeviceMapOp<'a, T, M> {
    /// Blocking terminal — enqueue
    /// `clEnqueueMapBuffer(CL_TRUE, CL_MAP_READ)` on the owning
    /// buffer's context default queue.
    pub fn wait(self) -> Result<DeviceMappedReadGuard<'a, T, M>> {
        let ctx = &self.owner.ctx;
        self.wait_on(ctx)
    }

    /// Blocking terminal with an explicit launcher.
    pub fn wait_on<L: Launcher>(self, launcher: &L) -> Result<DeviceMappedReadGuard<'a, T, M>> {
        let (guard, event) = DeviceMappedReadGuard::enqueue_map(self.owner, launcher, true)?;
        drop(event);
        Ok(guard)
    }

    /// Non-blocking terminal — enqueue
    /// `clEnqueueMapBuffer(CL_FALSE, CL_MAP_READ)` on the owning
    /// buffer's context default queue.
    pub fn submit(self) -> Result<DeviceMapReadPending<'a, T, M>> {
        let ctx = &self.owner.ctx;
        self.submit_on(ctx)
    }

    /// Non-blocking terminal with an explicit launcher. Returns a
    /// [`DeviceMapReadPending`] carrying the map event. Bytes are
    /// NOT spec-valid for host reads until the map event completes;
    /// consume via [`DeviceMapReadPending::wait`] for the guard, or
    /// use [`DeviceMapReadPending::event`] to thread the map event
    /// into a cross-queue chain first.
    pub fn submit_on<L: Launcher>(self, launcher: &L) -> Result<DeviceMapReadPending<'a, T, M>> {
        let (guard, event) = DeviceMappedReadGuard::enqueue_map(self.owner, launcher, false)?;
        Ok(DeviceMapReadPending {
            guard: Some(guard),
            event,
        })
    }
}

/// Lazy builder for [`DeviceSlice::map_mut`]. Same shape as
/// [`DeviceMapOp`] but the resulting guard is
/// [`DeviceMappedWriteGuard`] (DerefMut to `&mut [T]`).
pub struct DeviceMapMutOp<'a, T, M: MemMode> {
    owner: &'a mut DeviceSlice<T, M>,
}

impl<'a, T, M: MemMode + HostWritable + HostReadable> DeviceMapMutOp<'a, T, M> {
    /// Blocking terminal — enqueue
    /// `clEnqueueMapBuffer(CL_TRUE, CL_MAP_READ | CL_MAP_WRITE)` on
    /// the owning buffer's context default queue.
    pub fn wait(self) -> Result<DeviceMappedWriteGuard<'a, T, M>> {
        // Need to extract ctx before moving self — owner field is
        // accessed mutably below via enqueue_map. Read the ctx ref
        // directly (Context is Clone, Arc-internal).
        let ctx = self.owner.ctx.clone();
        self.wait_on(&ctx)
    }

    /// Blocking terminal with an explicit launcher.
    pub fn wait_on<L: Launcher>(self, launcher: &L) -> Result<DeviceMappedWriteGuard<'a, T, M>> {
        let (guard, event) = DeviceMappedWriteGuard::enqueue_map(self.owner, launcher, true)?;
        drop(event);
        Ok(guard)
    }

    /// Non-blocking terminal — see [`DeviceMapOp::submit`].
    pub fn submit(self) -> Result<DeviceMapWritePending<'a, T, M>> {
        let ctx = self.owner.ctx.clone();
        self.submit_on(&ctx)
    }

    /// Non-blocking terminal with an explicit launcher.
    pub fn submit_on<L: Launcher>(self, launcher: &L) -> Result<DeviceMapWritePending<'a, T, M>> {
        let (guard, event) = DeviceMappedWriteGuard::enqueue_map(self.owner, launcher, false)?;
        Ok(DeviceMapWritePending {
            guard: Some(guard),
            event,
        })
    }
}

/// Result of [`DeviceMapOp::submit`] — a non-blocking
/// `clEnqueueMapBuffer` in flight. The pointer was set synchronously
/// inside the call but the bytes are NOT spec-valid for host reads
/// until the map event completes.
pub struct DeviceMapReadPending<'a, T, M: MemMode> {
    guard: Option<DeviceMappedReadGuard<'a, T, M>>,
    event: Event,
}

impl<'a, T, M: MemMode> DeviceMapReadPending<'a, T, M> {
    /// Borrow the map [`Event`] for cross-queue chain ordering before
    /// consuming the pending.
    pub fn event(&self) -> &Event {
        &self.event
    }

    /// Block on the map event and return the
    /// [`DeviceMappedReadGuard`]. After Ok return the guard's
    /// `Deref<Target = [T]>` is safe to read.
    pub fn wait(mut self) -> Result<DeviceMappedReadGuard<'a, T, M>> {
        self.event.wait()?;
        Ok(self
            .guard
            .take()
            .expect("DeviceMapReadPending::wait called twice"))
    }
}

/// Result of [`DeviceMapMutOp::submit`] — same shape as
/// [`DeviceMapReadPending`] but yields a [`DeviceMappedWriteGuard`].
pub struct DeviceMapWritePending<'a, T, M: MemMode> {
    guard: Option<DeviceMappedWriteGuard<'a, T, M>>,
    event: Event,
}

impl<'a, T, M: MemMode> DeviceMapWritePending<'a, T, M> {
    pub fn event(&self) -> &Event {
        &self.event
    }
    pub fn wait(mut self) -> Result<DeviceMappedWriteGuard<'a, T, M>> {
        self.event.wait()?;
        Ok(self
            .guard
            .take()
            .expect("DeviceMapWritePending::wait called twice"))
    }
}

/// RAII guard for a `cl_mem` read map. Drop issues
/// `clEnqueueUnmapMemObject` and discards the unmap event (OpenCL
/// retains the cl_mem internally, so the release in [`DeviceSlice`]'s
/// `Drop` impl gates correctly without an explicit last-use list).
///
/// Users who need the unmap event for cross-queue chain ordering can
/// call [`release`](Self::release) instead of letting Drop fire — it
/// consumes the guard and returns the unmap event.
pub struct DeviceMappedReadGuard<'a, T, M: MemMode> {
    buf: &'a DeviceSlice<T, M>,
    host_ptr: *mut T,
    queue: crate::util::RetainedQueue,
    released: bool,
}

// SAFETY: host_ptr is a mapped pointer accessed serially via
// Deref/Drop on this thread. The buffer's Send-ness is inherited
// through the borrow.
unsafe impl<T: Send, M: MemMode> Send for DeviceMappedReadGuard<'_, T, M> {}

impl<'a, T, M: MemMode> DeviceMappedReadGuard<'a, T, M> {
    fn enqueue_map<L: Launcher>(
        buf: &'a DeviceSlice<T, M>,
        launcher: &L,
        blocking: bool,
    ) -> Result<(Self, Event)> {
        let queue = crate::util::RetainedQueue::from_queue(launcher.cl_queue())?;
        let size = buf.len * std::mem::size_of::<T>();
        // SAFETY: buf.buffer is live (we hold the borrow); queue is
        // live (RetainedQueue); size matches the allocation.
        let (host_ptr_raw, event) = unsafe {
            map_primitive::map_buffer(
                queue.raw(),
                buf.buffer.get(),
                blocking,
                CL_MAP_READ,
                0,
                size,
                &[],
            )?
        };
        Ok((
            DeviceMappedReadGuard {
                buf,
                host_ptr: host_ptr_raw.cast::<T>(),
                queue,
                released: false,
            },
            event,
        ))
    }

    /// Consume the guard, enqueue the unmap, and return the unmap
    /// [`Event`] for cross-queue chain ordering. Mirrors `Drop`'s
    /// unmap but lets the caller thread the resulting event into
    /// downstream enqueues on different queues. After this returns,
    /// the guard's `Drop` is suppressed.
    pub fn release(mut self) -> Result<Event> {
        // SAFETY: host_ptr was returned by our own map_buffer call;
        // unmap exactly once.
        let event = unsafe {
            map_primitive::unmap_mem_object(
                self.queue.raw(),
                self.buf.buffer.get(),
                self.host_ptr.cast(),
                &[],
            )?
        };
        self.released = true;
        Ok(event)
    }
}

impl<T, M: MemMode> Deref for DeviceMappedReadGuard<'_, T, M> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: host_ptr is mapped + readable for this guard's
        // lifetime.
        unsafe { crate::util::mapped_slice(self.host_ptr, self.buf.len) }
    }
}

impl<T, M: MemMode> Drop for DeviceMappedReadGuard<'_, T, M> {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        // SAFETY: host_ptr was mapped via `enqueue_map`; unmap once.
        match unsafe {
            map_primitive::unmap_mem_object(
                self.queue.raw(),
                self.buf.buffer.get(),
                self.host_ptr.cast(),
                &[],
            )
        } {
            Ok(_evt) => {} // _evt drops here (OpenCL holds the cl_mem internally)
            Err(_) => self.buf.ctx.record_err(),
        }
    }
}

/// RAII guard for a `cl_mem` read+write map. Same shape as
/// [`DeviceMappedReadGuard`] with `DerefMut<Target = [T]>` added.
pub struct DeviceMappedWriteGuard<'a, T, M: MemMode> {
    buf: &'a mut DeviceSlice<T, M>,
    host_ptr: *mut T,
    queue: crate::util::RetainedQueue,
    released: bool,
}

unsafe impl<T: Send, M: MemMode> Send for DeviceMappedWriteGuard<'_, T, M> {}

impl<'a, T, M: MemMode> DeviceMappedWriteGuard<'a, T, M> {
    fn enqueue_map<L: Launcher>(
        buf: &'a mut DeviceSlice<T, M>,
        launcher: &L,
        blocking: bool,
    ) -> Result<(Self, Event)> {
        let queue = crate::util::RetainedQueue::from_queue(launcher.cl_queue())?;
        let size = buf.len * std::mem::size_of::<T>();
        let cl_mem = buf.buffer.get();
        // SAFETY: see DeviceMappedReadGuard::enqueue_map.
        let (host_ptr_raw, event) = unsafe {
            map_primitive::map_buffer(
                queue.raw(),
                cl_mem,
                blocking,
                CL_MAP_READ | CL_MAP_WRITE,
                0,
                size,
                &[],
            )?
        };
        Ok((
            DeviceMappedWriteGuard {
                buf,
                host_ptr: host_ptr_raw.cast::<T>(),
                queue,
                released: false,
            },
            event,
        ))
    }

    /// See [`DeviceMappedReadGuard::release`].
    pub fn release(mut self) -> Result<Event> {
        let event = unsafe {
            map_primitive::unmap_mem_object(
                self.queue.raw(),
                self.buf.buffer.get(),
                self.host_ptr.cast(),
                &[],
            )?
        };
        self.released = true;
        Ok(event)
    }
}

impl<T, M: MemMode> Deref for DeviceMappedWriteGuard<'_, T, M> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: see DeviceMappedReadGuard::deref.
        unsafe { crate::util::mapped_slice(self.host_ptr, self.buf.len) }
    }
}

impl<T, M: MemMode> DerefMut for DeviceMappedWriteGuard<'_, T, M> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: `&mut self` upgrades to a unique mutable slice;
        // mapped read+write for the guard's lifetime.
        unsafe { crate::util::mapped_slice_mut(self.host_ptr, self.buf.len) }
    }
}

impl<T, M: MemMode> Drop for DeviceMappedWriteGuard<'_, T, M> {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        match unsafe {
            map_primitive::unmap_mem_object(
                self.queue.raw(),
                self.buf.buffer.get(),
                self.host_ptr.cast(),
                &[],
            )
        } {
            Ok(_evt) => {}
            Err(_) => self.buf.ctx.record_err(),
        }
    }
}
