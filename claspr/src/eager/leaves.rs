//! Leaf device-operation ops for the eager graph — the concrete `DeviceOp`s that
//! do one unit of device/host work (fill, upload/download, alloc, transfer, SVM
//! fills, image transfers, host-view acquire/release, buffer<->buffer copy).
//!
//! Each is a small struct + its `DeviceOp` impl (+ a `pub fn` constructor). They
//! share the graph edge (`Input`/`Pipe`/home) and, when CB-recordable, the
//! `cb_leaf_build` prologue (see `eager/cb.rs`). Read one leaf to learn the shape;
//! they do not depend on each other. See `ARCHITECTURE.md`.

use super::*;

// ── Concrete-head terminal helper ──────────────────────────────────────
//
// The buffer-verb ops (`Fill`/`Download`/`ReadInto`/`WriteDevice`/
// `TransferToDevice`) are **concrete-head**: their input is a caller-owned
// `DeviceSlice`, whose `.ctx()` supplies the queue. That lets them offer the
// no-launcher Tier-1 terminals `wait()`/`submit()` — the context is recovered
// from the owned buffer rather than passed in. A pipe-fed op (only reachable
// inside an eager `and_then` closure) has no concrete buffer, so these terminals
// error clearly, steering the caller to `wait_on(&ctx)` / `sync(&ctx)`.

/// Recover the owning [`Context`] from a concrete-head [`Input<DeviceSlice>`],
/// or a clear "pipe-fed" error for the no-launcher concrete-head terminals.
fn concrete_buf_ctx<T, M: MemMode>(buf: &Input<DeviceSlice<T, M>>) -> Result<Context> {
    use crate::Buffer;
    buf.with_concrete(|b| b.ctx().clone())
        .ok_or(Error::NotSupported(
            "concrete-head terminal (wait/submit) on a pipe-fed buffer op — use \
         wait_on(&ctx) / sync(&ctx) for piped (graph) inputs",
        ))
}

/// SVM analog of [`concrete_buf_ctx`]: recover the owning [`Context`] from a
/// concrete-head [`Input<MappedSlice>`], or a clear "pipe-fed" error.
fn concrete_svm_ctx<T, M: MemMode>(buf: &Input<MappedSlice<T, M>>) -> Result<Context> {
    use crate::Buffer;
    buf.with_concrete(|b| b.ctx().clone())
        .ok_or(Error::NotSupported(
            "concrete-head terminal (wait/submit) on a pipe-fed SVM op — use \
         wait_on(&ctx) / sync(&ctx) for piped (graph) inputs",
        ))
}

// ── Leaf: in-place fill (eager port of DeviceSliceFillOp) ──────────────

/// Fill a buffer (upstream pipe or concrete) with `value` via a non-blocking
/// `clEnqueueFillBuffer`, threading the upstream events as the wait-list.
pub struct Fill<T: Copy, M: MemMode> {
    buf: Input<DeviceSlice<T, M>>,
    value: T,
    out: Pipe<DeviceSlice<T, M>>,
    /// Design-v2 CB home — this leaf can be a CB boundary (a single-`fill` graph)
    /// or add itself to a parent's CB. See [`CbCache`].
    cb_cache: CbCache,
}

/// Build a fill leaf over an upstream buffer.
pub fn fill<T, M>(buf: impl Into<Input<DeviceSlice<T, M>>>, value: T) -> Fill<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    Fill {
        buf: buf.into(),
        value,
        out: Pipe::new(),
        cb_cache: new_cb_cache(),
    }
}

impl<T, M> DeviceOp for Fill<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        use crate::record::MemRef;
        use opencl3::memory::ClMem;
        let (mut buf, deps, home) = self.buf.resolve_home(ec)?;

        // ── CB-mode fork (design v2) ────────────────────────────────────────
        match ec.cb() {
            CbWalk::Off => {} // fall through to the normal enqueue below.
            CbWalk::Build { builder, ext, .. } => {
                // Shared prologue: entry deps + wait-list + precise-invalidation
                // reach (note_slot origins into the CB, propagate onto the output).
                let waits = cb_leaf_build(
                    ec,
                    builder,
                    ext,
                    &deps,
                    self.buf.slot_cell_id(),
                    self.buf.pipe_cell_id(),
                    self.out.cell_id(),
                );
                let mem = MemRef::Buffer(buf.buffer().get());
                // Byte pattern of the fill value (T: Copy).
                let pattern = unsafe {
                    std::slice::from_raw_parts(
                        (&self.value as *const T) as *const u8,
                        std::mem::size_of::<T>(),
                    )
                };
                let byte_len = buf.byte_len();
                if let Some(sp) = builder.fill_buffer(mem, pattern, 0, byte_len, &waits) {
                    ec.sp_register(self.out.cell_id(), std::collections::BTreeSet::from([sp]));
                }
                // Deposit the (lent) buffer with EMPTY cl_event deps — ordering is
                // the CB-internal sync points, not events.
                self.out.put_home(buf, Deps::new(), home);
                return Ok(());
            }
            CbWalk::LendOnly { ext, .. } => {
                cb_collect_external(ext, &deps);
                // Replay: lend + deposit, add/enqueue nothing (the cached CB runs).
                self.out.put_home(buf, Deps::new(), home);
                return Ok(());
            }
        }

        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // Fill has no native CL_BLOCKING flag (it's always enqueue + optional
        // wait — exactly what the old `FillOp::wait_on` did internally), so both
        // modes enqueue non-blocking; Blocking then waits on the event here.
        // In-place: the filled buffer is the lent buffer → home threads through.
        let event = crate::buffer::fill_buffer_enqueue(&mut buf, ec, self.value, &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn cb_addable(&self) -> bool {
        true
    }

    fn cbable_weight(&self) -> usize {
        // A fill records exactly one `clCommandFillBufferKHR`.
        1
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill".into());
    }
}

impl<T, M> Fill<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    /// Concrete-head blocking terminal: fill on the buffer's own context default
    /// queue and return the (filled) buffer. The no-launcher Tier-1 spelling
    /// (`buf.fill(v).wait()?`); use [`wait_on`](DeviceOpExt::wait_on) for a
    /// specific queue, or `sync`/`wait_on` for a pipe-fed op.
    pub fn wait(self) -> Result<DeviceSlice<T, M>> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal: enqueue the fill on the buffer's own
    /// context default queue and return a completion [`Event`](crate::Event).
    pub fn submit(self) -> Result<crate::Event> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_on(&*queue)
    }
}

// ── Leaf: upload (host → device, alloc-once + persistent home) ──────────

/// Allocate a `DeviceSlice<T, M>` ONCE, seed it from `src`, and hand it a
/// **persistent home** so the SAME `cl_mem` is reused across `g.sync()` replays
/// (the home invariant: "homeless is never legitimate" — even an upload-minted
/// buffer carries a home). A chain-entry leaf — no upstream input.
///
/// ## Stable handle + access-mode reseed
///
/// The buffer is allocated on the FIRST run (`from_slice`, `CL_MEM_COPY_HOST_PTR`)
/// into a persistent [`Cell`] this op owns; that cell is the buffer's home, so a
/// run's `Checkout` / `PipePayload` drop returns the SAME buffer to it. On replay
/// the buffer is re-lent from the cell (not re-minted), and whether its contents
/// are refreshed is decided by the marker via
/// [`UploadReseed::RESEED_ON_REPLAY`](crate::UploadReseed):
/// - **kernel-writable** (`ReadWrite`, …): re-seed the host source into the SAME
///   buffer each run — `upload(RW) → scale → download` stays idempotent (no
///   compounding) over a stable handle.
/// - **kernel read-only** (`ReadOnly`, `Frozen`): seed once on run 1; skip the
///   host write on replays (the kernel never mutated it).
///
/// If a previous run's `Checkout` is still alive (the buffer is lent out), the
/// cell is empty AND it has already been seeded → a second `sync` is **graph-busy**
/// (same contract as a concrete-head cell).
pub struct Upload<T: Copy, M: MemMode = ReadWrite> {
    // The host source, RETAINED for the seed-once write and any reseed-on-replay.
    src: UploadSource<T>,
    // The persistent device buffer's home cell: allocated once (first run), then
    // re-lent + re-armed across replays so the `cl_mem` handle stays stable. Empty
    // while lent (busy if already seeded); `None`-on-take is the lend.
    buf: Cell<DeviceSlice<T, M>>,
    // Whether the buffer has ever been allocated/seeded. Distinguishes "first run
    // → alloc" (cell empty, not seeded) from "lent out → busy" (cell empty, seeded).
    seeded: Arc<Mutex<bool>>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an upload leaf from any `Vec<T>` / `Box<[T]>` / `Arc<[T]>`, with the
/// **default [`ReadWrite`] marker** — the overwhelming common case, so no
/// turbofish: `upload(vec![1u32, 2, 3])`. For a non-default marker use
/// [`upload_as`] with a marker witness (`upload_as(src, Frozen)`); both paths
/// allocate once via `from_slice` (`CL_MEM_COPY_HOST_PTR`), the only constructor
/// that can build an immutable `Frozen`/`ReadOnly` buffer.
pub fn upload<T, S>(src: S) -> Upload<T, ReadWrite>
where
    T: Copy + Send + Sync + 'static,
    S: Into<UploadSource<T>>,
{
    upload_as(src, ReadWrite)
}

/// Build an upload leaf with an **explicit access marker**, inferred from the
/// `marker` witness — no turbofish: `upload_as(src, Frozen)` /
/// `upload_as(src, ReadOnly)`. `T`/`S` infer from `src`, `M` from the witness.
/// The default-marker shorthand is [`upload`]. Like `upload`, backed by
/// `from_slice` (`CL_MEM_COPY_HOST_PTR`) on the first run.
pub fn upload_as<T, M, S>(src: S, marker: M) -> Upload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
    S: Into<UploadSource<T>>,
{
    let _ = marker; // witness only — fixes M, zero-sized, no runtime use.
    Upload {
        src: src.into(),
        buf: Arc::new(Mutex::new(None)),
        seeded: Arc::new(Mutex::new(false)),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for Upload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // The buffer is allocated ONCE and lives in `self.buf` across runs; its home
        // is that very cell, so a run's Checkout / PipePayload drop returns the SAME
        // `cl_mem` here. Three cases, decided by the cell + the `seeded` flag:
        let mut seeded = self.seeded.lock().unwrap();
        let lent = self.buf.lock().unwrap().take();
        let buf = match (lent, *seeded) {
            // First run: never seeded → alloc + seed via from_slice
            // (CL_MEM_COPY_HOST_PTR, synchronous create, no in-flight event).
            (None, false) => {
                let buf = DeviceSlice::<T, M>::from_slice(ec.context(), self.src.as_slice())?;
                *seeded = true;
                buf
            }
            // Replay: the buffer is back in the cell. Re-lend it; re-seed the host
            // source IF the marker is kernel-writable (it may have been mutated in
            // place last run) — keeping `upload(RW) → … → download` idempotent. A
            // kernel read-only marker (ReadOnly/Frozen) skips the write: its bytes
            // never changed device-side, seed-once suffices.
            (Some(mut buf), _) => {
                if M::RESEED_ON_REPLAY {
                    // Synchronous host write back into the SAME buffer (stable
                    // handle). No upstream deps — upload is a chain head.
                    crate::buffer::write_buffer_enqueue(
                        &mut buf,
                        ec,
                        self.src.as_slice(),
                        true,
                        &[],
                    )?;
                }
                buf
            }
            // Cell empty but already seeded: the buffer is lent out (a prior run's
            // Checkout is still alive) → graph-busy, the concrete-cell contract.
            (None, true) => {
                return Err(Error::NotSupported(
                    "eager graph: an upload buffer was already lent and not returned \
                     — a graph is `sync`'d while a previous `Checkout` is still alive \
                     (the graph is busy)",
                ));
            }
        };
        // The home is this op's persistent cell (identity rehome): the buffer is
        // returned here on Checkout / PipePayload drop, re-arming the upload with a
        // STABLE handle. So a downstream consume (download) rehomes it here, not the
        // releasing drop.
        let home: Option<BoxedHome<DeviceSlice<T, M>>> = Some(Box::new(Arc::clone(&self.buf)));
        self.out.put_home(buf, Deps::new(), home);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("upload".into());
    }
}

// ── Leaf: seeded device-scalar alloc (host value → DeviceScalar) ────────

/// Allocate a [`DeviceScalar<T, M>`] ONCE, seed it from `value`, and hand it a
/// **persistent home** so the SAME `cl_mem` is reused across `g.sync()` replays
/// — the scalar twin of [`Upload`]. A chain-entry leaf (no upstream input).
///
/// Same stable-handle + reseed-on-replay contract as [`Upload`]: on the first
/// run the scalar is allocated + seeded via [`DeviceScalar::new`]
/// (`CL_MEM_COPY_HOST_PTR`); on replay it is re-lent from this op's home cell,
/// and re-seeded IFF the marker is kernel-writable
/// ([`UploadReseed::RESEED_ON_REPLAY`](crate::UploadReseed)).
pub struct ScalarUpload<T: Copy, M: MemMode = ReadWrite> {
    value: T,
    buf: Cell<DeviceScalar<T, M>>,
    seeded: Arc<Mutex<bool>>,
    out: Pipe<DeviceScalar<T, M>>,
}

/// Build a seeded device-scalar alloc leaf with the **default [`ReadWrite`]
/// marker** — the scalar twin of [`upload`]: `scalar_value(0.0f32)`.
pub fn scalar_value<T>(value: T) -> ScalarUpload<T, ReadWrite>
where
    T: Copy + Send + Sync + 'static,
{
    scalar_value_as(value, ReadWrite)
}

/// Build a seeded device-scalar alloc leaf with an **explicit access marker**,
/// inferred from the `marker` witness — the scalar twin of [`upload_as`].
pub fn scalar_value_as<T, M>(value: T, marker: M) -> ScalarUpload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    let _ = marker;
    ScalarUpload {
        value,
        buf: Arc::new(Mutex::new(None)),
        seeded: Arc::new(Mutex::new(false)),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for ScalarUpload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    type Output = DeviceScalar<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceScalar<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Same three-case stable-handle logic as `Upload`, over a length-1 scalar.
        let mut seeded = self.seeded.lock().unwrap();
        let lent = self.buf.lock().unwrap().take();
        let buf = match (lent, *seeded) {
            (None, false) => {
                let buf = DeviceScalar::<T, M>::new(ec.context(), self.value)?;
                *seeded = true;
                buf
            }
            (Some(mut buf), _) => {
                if M::RESEED_ON_REPLAY {
                    crate::buffer::write_buffer_enqueue(
                        &mut buf.inner,
                        ec,
                        std::slice::from_ref(&self.value),
                        true,
                        &[],
                    )?;
                }
                buf
            }
            (None, true) => {
                return Err(Error::NotSupported(
                    "eager graph: a device-scalar upload buffer was already lent and \
                     not returned — a graph is `sync`'d while a previous `Checkout` is \
                     still alive (the graph is busy)",
                ));
            }
        };
        let home: Option<BoxedHome<DeviceScalar<T, M>>> = Some(Box::new(Arc::clone(&self.buf)));
        self.out.put_home(buf, Deps::new(), home);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("scalar_value".into());
    }
}

// ── Leaf: zero-init device-scalar alloc ────────────────────────────────

/// Allocate a [`DeviceScalar<T, M>`] zero-initialised (via a length-1
/// [`DeviceScalar::new`]`(T::default())`), with a persistent home — the scalar
/// twin of [`alloc_zero`]. A chain-entry leaf.
pub struct ScalarZero<T: Copy, M: MemMode = ReadWrite> {
    inner: ScalarUpload<T, M>,
}

/// Build a zero-init device-scalar alloc leaf with the **default [`ReadWrite`]
/// marker** — `scalar_zero::<f32>()`.
pub fn scalar_zero<T>() -> ScalarZero<T, ReadWrite>
where
    T: Copy + Default + Send + Sync + 'static,
{
    ScalarZero {
        inner: scalar_value_as(T::default(), ReadWrite),
    }
}

/// Build a zero-init device-scalar alloc leaf with an **explicit access marker**.
pub fn scalar_zero_as<T, M>(marker: M) -> ScalarZero<T, M>
where
    T: Copy + Default + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    ScalarZero {
        inner: scalar_value_as(T::default(), marker),
    }
}

impl<T, M> DeviceOp for ScalarZero<T, M>
where
    T: Copy + Default + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    type Output = DeviceScalar<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceScalar<T, M>>> {
        self.inner.output_pipe()
    }

    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        self.inner.execute(ec, mode)
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("scalar_zero".into());
    }
}

// ── Leaf: download (device → host Vec, non-blocking read) ──────────────

/// Consume an upstream buffer, alloc a host `Vec<T>`, non-blocking-read into it
/// threading the upstream events. Output is the `Vec<T>`.
pub struct Download<T, M: MemMode = ReadWrite> {
    buf: Input<DeviceSlice<T, M>>,
    out: Pipe<Vec<T>>,
}

/// Build a download leaf over an upstream buffer.
pub fn download<T, M>(buf: impl Into<Input<DeviceSlice<T, M>>>) -> Download<T, M>
where
    T: Clone + Default + Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    Download {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for Download<T, M>
where
    T: Clone + Default + Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    type Output = Vec<T>;

    fn output_pipe(&self) -> Option<Pipe<Vec<T>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        // "Homeless is never legitimate": download CONSUMES the device buffer into
        // a host `Vec`, but the buffer itself still has a home (a user-allocated
        // concrete cell, a slot, or an upload-minted persistent cell). Resolve WITH
        // the home so the device buffer is RETURNED to its origin — the same
        // `cl_mem` is reused on replay — rather than released. The OUTPUT pipe
        // carries the `Vec` with NO home: the Vec is the user's result, it has no
        // origin cell. (`ReadInto` is the in-place template; here the buffer's home
        // and the output value diverge, so the rehome happens here, not via the
        // output pipe.)
        let (buf, deps, home) = self.buf.resolve_home(ec)?;
        let mut host = vec![T::default(); buf.len()];
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        match mode {
            // Terminal: native blocking read (CL_BLOCKING) — the driver waits,
            // the host Vec is valid on return, no event. Matches Tier-1
            // `ReadOp::wait_on`; restores parity for `…download().sync()`.
            ExecMode::Blocking => {
                crate::buffer::read_buffer_enqueue(&buf, ec, &mut host, true, &raw)?;
                rehome_consumed(buf, home);
                self.out.put(host, Deps::new());
            }
            // Pipelined: non-blocking; the event gates the Vec being valid. The
            // read is enqueued before we rehome, but the rehome only re-arms the
            // origin CELL (deposits the buffer handle for the NEXT run); the
            // in-flight read still holds the live `cl_mem` via the OpenCL queue, so
            // returning the handle to its cell here does not race the read.
            ExecMode::Pipelined => {
                let event = crate::buffer::read_buffer_enqueue(&buf, ec, &mut host, false, &raw)?;
                rehome_consumed(buf, home);
                self.out.put(host, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("download".into());
    }
}

impl<T, M> Download<T, M>
where
    T: Clone + Default + Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    /// Concrete-head blocking terminal: read on the buffer's own context default
    /// queue and return the host `Vec<T>`.
    pub fn wait(self) -> Result<Vec<T>> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal returning the (not-yet-valid) host
    /// `Vec<T>` plus a completion [`Event`](crate::Event) — mirrors the Tier-1
    /// `(Output, Event)` submit contract.
    pub fn submit(self) -> Result<(Vec<T>, crate::Event)> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}

// ── Leaf: read a DeviceSlice into a caller-supplied slice (read-into) ───────

/// Read a buffer into a **caller-supplied** `&mut [T]` (rather than allocating a
/// fresh `Vec` like [`Download`]), yielding the buffer back so it can be reused.
/// The eager analog of the old Tier-1 `buf.read(&mut dst)` builder: a
/// concrete-head op (it borrows the destination slice for `'d`, so it never
/// flows through a pipe — a pipe-fed read uses [`Download`]).
///
/// `Output = DeviceSlice<T, M>`: the buffer moves in and rebinds out
/// (`let buf = buf.read(&mut dst).wait()?;`), so a caller can read into the same
/// destination repeatedly.
pub struct ReadInto<'d, T, M: MemMode = ReadWrite> {
    buf: Input<DeviceSlice<T, M>>,
    // Behind a `Mutex` so `execute(&self)` can get the `&mut [T]` it needs to
    // read into. The caller slice is borrowed for `'d`; re-runs read into the
    // same destination (overwriting it).
    dst: Mutex<&'d mut [T]>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build a read-into leaf: read `buf` into the caller slice `dst`. See
/// [`ReadInto`].
pub fn read_into<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
    dst: &mut [T],
) -> ReadInto<'_, T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    ReadInto {
        buf: buf.into(),
        dst: Mutex::new(dst),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for ReadInto<'_, T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (buf, deps, home) = self.buf.resolve_home(ec)?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let mut dst = self.dst.lock().unwrap();
        // In-place: the buffer is read and handed back unchanged → home threads.
        match mode {
            // Terminal: native blocking read — `dst` is valid on return, no event.
            ExecMode::Blocking => {
                crate::buffer::read_buffer_enqueue(&buf, ec, &mut dst, true, &raw)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            // Pipelined: non-blocking; the event gates `dst` being valid.
            ExecMode::Pipelined => {
                let event = crate::buffer::read_buffer_enqueue(&buf, ec, &mut dst, false, &raw)?;
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("read_into".into());
    }
}

impl<T, M> ReadInto<'_, T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable + Send + 'static,
{
    /// Concrete-head blocking terminal: read into the caller slice on the
    /// buffer's own context default queue; return the buffer for reuse.
    pub fn wait(self) -> Result<DeviceSlice<T, M>> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal: enqueue the read on the buffer's own
    /// context default queue and return a completion [`Event`](crate::Event).
    /// (The `dst` slice must outlive the event.)
    pub fn submit(self) -> Result<crate::Event> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_on(&*queue)
    }
}

// ── Leaf: migrate a DeviceSlice to another device (eager TransferToDevice) ──

/// Eager port of the closure-layer `transfer_to_device(buf, &dev)`. Enqueues a
/// `clEnqueueMigrateMemObjects` for the buffer on `device`'s default OOO queue,
/// yielding the (now-migrated) buffer. The matching per-op routing combinator
/// kernels need after the buffer is migrated is [`on_device`](DeviceOpExt::on_device).
///
/// ## Shape: a leaf, not a wrapping method
///
/// Unlike [`on_device`](DeviceOpExt::on_device) (which *routes* an upstream op's
/// own enqueue to another queue without touching its value), `transfer_to_device`
/// is a buffer-*consuming* leaf: it resolves the upstream `DeviceSlice` value,
/// reads its `cl_mem`, and enqueues a migrate. That puts it in the same family as
/// [`download`] / [`fill`] / [`copy_to`](crate::eager::eager_copy_to) — every member
/// takes `impl Into<Input<DeviceSlice<…>>>` as its dataflow input — and mirrors
/// the old free-fn signature `transfer_to_device(buf, dev)` 1:1. A method form
/// would have to pin `S::Output = DeviceSlice<T>` (like [`OnDevice`]) yet still
/// resolve the value (unlike `OnDevice`), fighting both patterns; the leaf form
/// composes cleanly via `.and_then(|p| transfer_to_device(p, dev))`.
///
/// ## What the migrate actually does
///
/// For two devices sharing one `cl_context`, the runtime may or may not move
/// bytes (shared-memory topologies / sub-devices: typically a no-op; two dGPUs:
/// real migration). Either way the migrate is a queue command (non-blocking) so
/// the graph stays pipelined; downstream stages wait on the migrate event via
/// the carried [`Deps`]. Cross-*context* transfer is **not** this op — that goes
/// through host bounce ([`download`] → [`upload`]).
pub struct TransferToDevice<T, M: MemMode = ReadWrite> {
    buf: Input<DeviceSlice<T, M>>,
    target: DeviceTarget,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build a transfer-to-device leaf: migrate `buf` onto `device`'s default OOO
/// queue, yielding the migrated buffer. See [`TransferToDevice`] for semantics
/// and the rationale for the leaf (free-fn) shape over a wrapping method.
pub fn transfer_to_device<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
    device: &crate::Device,
) -> TransferToDevice<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    TransferToDevice {
        buf: buf.into(),
        target: DeviceTarget::Concrete(device.clone()),
        out: Pipe::new(),
    }
}

/// Build a transfer-to-device leaf targeting the device at `index` in the
/// running context's device list, resolved at execute. See [`transfer_to_device`]
/// for migrate semantics.
///
/// **Panics** at execute if `index` is out of range for `context().devices()`
/// (same timing/semantics as resolving `ec.device_at(index)` did).
pub fn transfer_to_device_at<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
    index: usize,
) -> TransferToDevice<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    TransferToDevice {
        buf: buf.into(),
        target: DeviceTarget::Index(index),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for TransferToDevice<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // In-place: the migrated buffer is the same buffer → home threads through.
        let (buf, deps, home) = self.buf.resolve_home(ec)?;
        // Resolve the target device (concrete, or by index into the running
        // context's device list) before resolving its queue.
        let device = match &self.target {
            DeviceTarget::Concrete(d) => d.clone(),
            DeviceTarget::Index(i) => ec.context().devices()[*i].clone(),
        };
        // Resolve the target device's default OOO queue (cached on the Context,
        // so the terminal's flush_all_outoforder_queues pushes it). Same path
        // OnDevice uses to reach a non-primary device's queue.
        let target_q = ec.context().default_outoforder_queue(&device)?;
        // Enqueue the migrate with the upstream events as the wait-list, on the
        // target queue (`&*target_q` is the `Queue: Launcher`). Non-blocking —
        // mode is ignored; the chain terminal's `into_output` does the final
        // wait. The migrate body mirrors the closure layer's
        // `transfer_to_device.rs` exactly.
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let event = crate::buffer::migrate_buffer_enqueue(&buf, &*target_q, &raw)?;
        self.out.put_home(buf, vec![wrap_event(event)], home);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("transfer_to_device".into());
    }
}

// ════════════════════════════════════════════════════════════════════════
// uninit_ext.rs ports — fill / write an alloc-uninit buffer → initialised
// ════════════════════════════════════════════════════════════════════════

// ── Leaf: fill a DeviceSliceUninit → DeviceSlice (eager FillFromUninitOp) ──

/// Consume an uninit `DeviceSlice` (upstream pipe or concrete) and fill it
/// with `value`, yielding the initialised buffer. Mirrors [`Fill`] (transform
/// shape, ExecMode branch on the Tier-1 `fill` builder's `wait_on`/`submit_on`).
pub struct FillDeviceUninit<T: Copy, M: MemMode> {
    uninit: Input<DeviceSliceUninit<T, M>>,
    value: T,
    out: Pipe<DeviceSlice<T, M>>,
    /// Design-v2 CB home — a device fill-from-uninit records the SAME
    /// `clCommandFillBufferKHR` an in-place [`Fill`] does (it writes a `cl_mem`),
    /// so it is a CB command like `Fill`/`FillMappedUninit`. See [`CbCache`].
    cb_cache: CbCache,
}

/// Build an eager fill-from-uninit leaf over a `DeviceSliceUninit`.
pub fn fill_device_uninit<T, M>(
    uninit: impl Into<Input<DeviceSliceUninit<T, M>>>,
    value: T,
) -> FillDeviceUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    FillDeviceUninit {
        uninit: uninit.into(),
        value,
        out: Pipe::new(),
        cb_cache: new_cb_cache(),
    }
}

impl<T, M> DeviceOp for FillDeviceUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        use crate::record::MemRef;
        use opencl3::memory::ClMem;
        let (uninit, deps) = self.uninit.resolve(ec)?;
        // SAFETY: the fill below writes every byte; downstream gates on the
        // returned fill event (Pipelined) or the driver waits (Blocking).
        let mut buf = unsafe { uninit.assume_init() };

        // ── CB-mode fork (design v2) — records `clCommandFillBufferKHR`, exactly
        // like in-place `Fill` (this writes a `cl_mem` too, just one produced from
        // an uninit alloc). Produced-from-uninit has no upstream slot/pipe reach,
        // so the prologue degenerates to (None, None). ──────────────────────────
        match ec.cb() {
            CbWalk::Off => {}
            CbWalk::Build { builder, ext, .. } => {
                let waits = cb_leaf_build(ec, builder, ext, &deps, None, None, self.out.cell_id());
                let mem = MemRef::Buffer(buf.buffer().get());
                let pattern = unsafe {
                    std::slice::from_raw_parts(
                        (&self.value as *const T) as *const u8,
                        std::mem::size_of::<T>(),
                    )
                };
                let byte_len = buf.byte_len();
                if let Some(sp) = builder.fill_buffer(mem, pattern, 0, byte_len, &waits) {
                    ec.sp_register(self.out.cell_id(), std::collections::BTreeSet::from([sp]));
                }
                self.out.put(buf, Deps::new());
                return Ok(());
            }
            CbWalk::LendOnly { ext, .. } => {
                cb_collect_external(ext, &deps);
                self.out.put(buf, Deps::new());
                return Ok(());
            }
        }

        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // Fill has no native CL_BLOCKING flag — enqueue, then wait on Blocking.
        let event = crate::buffer::fill_buffer_enqueue(&mut buf, ec, self.value, &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn cb_addable(&self) -> bool {
        true
    }

    fn cbable_weight(&self) -> usize {
        // One `clCommandFillBufferKHR`.
        1
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill_device_uninit".into());
    }
}

// ── Leaf: fill a MappedSliceUninit → MappedSlice ───────────────────────────

/// Eager analog of `FillFromUninitOp<MappedSliceUninit, _>`: fill an uninit
/// SVM slice with `value`. Mirrors [`Fill`].
pub struct FillMappedUninit<T: Copy, M: MemMode> {
    uninit: Input<MappedSliceUninit<T, M>>,
    value: T,
    out: Pipe<MappedSlice<T, M>>,
    /// Design-v2 CB home: an SVM fill records `clCommandSVMMemFillKHR` where the
    /// extension provides it, else falls back to software.
    cb_cache: CbCache,
}

/// Build an eager fill-from-uninit leaf over a `MappedSliceUninit`.
pub fn fill_mapped_uninit<T, M>(
    uninit: impl Into<Input<MappedSliceUninit<T, M>>>,
    value: T,
) -> FillMappedUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    FillMappedUninit {
        uninit: uninit.into(),
        value,
        out: Pipe::new(),
        cb_cache: new_cb_cache(),
    }
}

impl<T, M> DeviceOp for FillMappedUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        use crate::record::RecordableBuffer;
        let (uninit, deps) = self.uninit.resolve(ec)?;
        // SAFETY: the SVM fill below writes every byte.
        let buf = unsafe { uninit.assume_init() };

        // ── CB-mode fork (design v2) — SVM fill via clCommandSVMMemFillKHR. Absent
        // PFN → CbBuilder marks ineligible → boundary falls back to per-op. ────────
        match ec.cb() {
            CbWalk::Off => {}
            CbWalk::Build { builder, ext, .. } => {
                // Produced-from-uninit: no upstream pipe/slot feeds THIS value, so the
                // prologue degenerates to just collecting external deps (empty waits,
                // no origins to note/propagate).
                let waits = cb_leaf_build(ec, builder, ext, &deps, None, None, self.out.cell_id());
                let handle = buf.record_handle(); // MemRef::Svm
                let pattern = unsafe {
                    std::slice::from_raw_parts(
                        (&self.value as *const T) as *const u8,
                        std::mem::size_of::<T>(),
                    )
                };
                if let Some(sp) =
                    builder.fill_buffer(handle.mem, pattern, 0, handle.byte_len, &waits)
                {
                    ec.sp_register(self.out.cell_id(), std::collections::BTreeSet::from([sp]));
                }
                self.out.put(buf, Deps::new());
                return Ok(());
            }
            CbWalk::LendOnly { ext, .. } => {
                cb_collect_external(ext, &deps);
                self.out.put(buf, Deps::new());
                return Ok(());
            }
        }

        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SVM fill is always a non-blocking enqueue; Blocking waits here.
        let event = crate::mapped::svm_fill_enqueue(&buf, ec, self.value, &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn cb_addable(&self) -> bool {
        true
    }

    fn cbable_weight(&self) -> usize {
        1
    }

    fn cb_restamp(&self, evs: &[Dep]) {
        if let Some((v, deps)) = self.out.take() {
            let _ = deps;
            self.out.put(v, Vec::from(evs));
        }
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill_mapped_uninit".into());
    }
}

// ── Leaf: fill a USMSliceUninit → USMSlice (pure host op) ───────────────────

/// Eager analog of `FillFromUninitOp<USMSliceUninit, _>`. USM is host memory,
/// so this is a pure host op: no enqueue, no event, deps pass through (mode
/// N/A) — mirrors [`Upload`]'s synchronous-create shape.
pub struct FillUsmUninit<T: Copy, M: MemMode> {
    uninit: Input<USMSliceUninit<T, M>>,
    value: T,
    out: Pipe<USMSlice<T, M>>,
}

/// Build an eager fill-from-uninit leaf over a `USMSliceUninit`.
pub fn fill_usm_uninit<T, M>(
    uninit: impl Into<Input<USMSliceUninit<T, M>>>,
    value: T,
) -> FillUsmUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    FillUsmUninit {
        uninit: uninit.into(),
        value,
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for FillUsmUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<USMSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Pure host op — no event; forward the upstream deps unchanged.
        let (uninit, deps) = self.uninit.resolve(ec)?;
        let buf = uninit.fill_into(self.value);
        self.out.put(buf, deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill_usm_uninit".into());
    }
}

// ── Leaf: write host data into a DeviceSliceUninit → DeviceSlice ────────────

/// Consume an uninit `DeviceSlice` and write host `src` into it, yielding the
/// initialised buffer. Mirrors [`Fill`] (ExecMode branch). For the non-blocking
/// path the host `src` is kept alive until the write event fires via
/// `register_drop_callback`; for the blocking path the write completes before
/// return, so `src` drops normally at end of `execute`.
pub struct WriteDeviceUninit<T, M: MemMode> {
    uninit: Input<DeviceSliceUninit<T, M>>,
    // Retained by value; read by reference each run (see `WriteDevice::src`).
    src: UploadSource<T>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an eager write-from-uninit leaf over a `DeviceSliceUninit`.
pub fn write_device_uninit<T, M, S>(
    uninit: impl Into<Input<DeviceSliceUninit<T, M>>>,
    src: S,
) -> WriteDeviceUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostUploadable + HostWritable + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteDeviceUninit {
        uninit: uninit.into(),
        src: src.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for WriteDeviceUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostUploadable + HostWritable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve(ec)?;
        // SAFETY: the write below covers every byte; downstream gates on the
        // returned write event (Pipelined) or the driver waits (Blocking).
        let mut buf = unsafe { uninit.assume_init() };
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        match mode {
            ExecMode::Blocking => {
                crate::buffer::write_buffer_enqueue(&mut buf, ec, self.src.as_slice(), true, &raw)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                // `self.src` is valid for the whole `sync` — no keep-alive needed.
                let event = crate::buffer::write_buffer_enqueue(
                    &mut buf,
                    ec,
                    self.src.as_slice(),
                    false,
                    &raw,
                )?;
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_device_uninit".into());
    }
}

// ── Leaf: write host data into an existing (init) DeviceSlice ───────────────

/// Write host `src` into an already-initialised `DeviceSlice`, in place, via a
/// non-blocking `clEnqueueWriteBuffer` — the eager analog of the closure layer's
/// `device_slice_write(buf, src)`. The buffer passes through as the op's output.
///
/// This is a real **async host→device transfer** (a queue command), NOT a
/// map/host-memcpy/unmap host seam: `submit_on` enqueues `CL_FALSE` and returns
/// the write event as the op's deps, so the write overlaps downstream device
/// work; `register_drop_callback` keeps the host `src` alive until the DMA
/// completes (`CL_COMPLETE`). The `Blocking` terminal path uses `wait_on`
/// (`CL_BLOCKING`) instead, mirroring [`WriteDeviceUninit`].
pub struct WriteDevice<T, M: MemMode = ReadWrite> {
    buf: Input<DeviceSlice<T, M>>,
    // Retained by value (not `Option`): the host source is read BY REFERENCE each
    // run (re-seed). `&self` outlives the whole `sync` (the terminal waits before
    // returning), so the source stays valid across the async write window — the
    // former per-run `register_drop_callback` keep-alive is unnecessary now that
    // the op lives in the reusable graph rather than being moved into an executor.
    src: UploadSource<T>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an eager in-place write leaf over an existing `DeviceSlice` (concrete or
/// piped). `M: HostWritable` — same gate as the closure layer's
/// `device_slice_write`.
pub fn write<T, M, S>(buf: impl Into<Input<DeviceSlice<T, M>>>, src: S) -> WriteDevice<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteDevice {
        buf: buf.into(),
        src: src.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for WriteDevice<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (mut buf, deps, home) = self.buf.resolve_home(ec)?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // In-place: the written buffer is the lent buffer → home threads through.
        match mode {
            ExecMode::Blocking => {
                crate::buffer::write_buffer_enqueue(&mut buf, ec, self.src.as_slice(), true, &raw)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                // Non-blocking write; `self.src` stays valid for the whole `sync`
                // (the op lives in the graph; the terminal waits before returning),
                // so no per-run keep-alive callback is needed.
                let event = crate::buffer::write_buffer_enqueue(
                    &mut buf,
                    ec,
                    self.src.as_slice(),
                    false,
                    &raw,
                )?;
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write".into());
    }
}

impl<T, M> WriteDevice<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    /// Concrete-head blocking terminal: write on the buffer's own context default
    /// queue and return the buffer for reuse (`let buf = buf.write(d).wait()?;`).
    pub fn wait(self) -> Result<DeviceSlice<T, M>> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal returning the buffer plus a completion
    /// [`Event`](crate::Event) — mirrors the Tier-1 `(Output, Event)` contract so
    /// the caller can keep using the buffer and chain via `.after(event)`.
    pub fn submit(self) -> Result<(DeviceSlice<T, M>, crate::Event)> {
        let ctx = concrete_buf_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}

// ── Leaf: write host data into a MappedSliceUninit → MappedSlice ────────────

/// Eager analog of `WriteFromUninitOp<MappedSliceUninit, _>`. Mirrors
/// [`WriteDeviceUninit`].
pub struct WriteMappedUninit<T, M: MemMode> {
    uninit: Input<MappedSliceUninit<T, M>>,
    // Retained by value; read by reference each run (see `WriteDevice::src`).
    src: UploadSource<T>,
    out: Pipe<MappedSlice<T, M>>,
}

/// Build an eager write-from-uninit leaf over a `MappedSliceUninit`.
pub fn write_mapped_uninit<T, M, S>(
    uninit: impl Into<Input<MappedSliceUninit<T, M>>>,
    src: S,
) -> WriteMappedUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteMappedUninit {
        uninit: uninit.into(),
        src: src.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for WriteMappedUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (uninit, deps) = self.uninit.resolve(ec)?;
        // SAFETY: the SVM write below covers every byte.
        let buf = unsafe { uninit.assume_init() };
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SVM write is always a non-blocking enqueue (no native CL_BLOCKING flag);
        // Blocking waits on the returned event here, Pipelined threads it
        // downstream. `self.src` is valid for the whole `sync` — no keep-alive.
        let event = crate::mapped::svm_write_enqueue(&buf, ec, self.src.as_slice(), &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put(buf, Deps::new());
            }
            ExecMode::Pipelined => {
                self.out.put(buf, vec![wrap_event(event)]);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_mapped_uninit".into());
    }
}

// ── Leaf: in-place SVM fill (eager port of SvmFillOp) ──────────────────────

/// Fill an existing (init) [`MappedSlice`] with `value` via a non-blocking
/// `clEnqueueSVMMemFill` (or kernel fill for kernel-RO markers), threading the
/// upstream events as the wait-list. SVM analog of [`Fill`]. The buffer passes
/// through as the op's output (concrete-head reusable). The fill event is
/// auto-registered on the buffer's last-use list (inside the raw helper) so
/// Drop's `clEnqueueSVMFree` waits for it.
pub struct FillMapped<T: Copy, M: MemMode> {
    buf: Input<MappedSlice<T, M>>,
    value: T,
    out: Pipe<MappedSlice<T, M>>,
    /// Design-v2 CB home: an SVM fill records `clCommandSVMMemFillKHR` where the
    /// extension provides it (>= 0.9.4), else falls back to software.
    cb_cache: CbCache,
}

/// Build an SVM fill leaf over an existing `MappedSlice` (concrete or piped).
pub fn fill_mapped<T, M>(buf: impl Into<Input<MappedSlice<T, M>>>, value: T) -> FillMapped<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    FillMapped {
        buf: buf.into(),
        value,
        out: Pipe::new(),
        cb_cache: new_cb_cache(),
    }
}

impl<T, M> DeviceOp for FillMapped<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        use crate::record::RecordableBuffer;
        let (buf, deps, home) = self.buf.resolve_home(ec)?;

        // ── CB-mode fork (design v2) — SVM fill via clCommandSVMMemFillKHR. If the
        // driver lacks that command, `CbBuilder::fill_buffer` marks the build
        // ineligible and the boundary falls back to the per-op path below. ──────
        match ec.cb() {
            CbWalk::Off => {}
            CbWalk::Build { builder, ext, .. } => {
                let waits = cb_leaf_build(
                    ec,
                    builder,
                    ext,
                    &deps,
                    self.buf.slot_cell_id(),
                    self.buf.pipe_cell_id(),
                    self.out.cell_id(),
                );
                let handle = buf.record_handle(); // MemRef::Svm
                let pattern = unsafe {
                    std::slice::from_raw_parts(
                        (&self.value as *const T) as *const u8,
                        std::mem::size_of::<T>(),
                    )
                };
                if let Some(sp) =
                    builder.fill_buffer(handle.mem, pattern, 0, handle.byte_len, &waits)
                {
                    ec.sp_register(self.out.cell_id(), std::collections::BTreeSet::from([sp]));
                }
                self.out.put_home(buf, Deps::new(), home);
                return Ok(());
            }
            CbWalk::LendOnly { ext, .. } => {
                cb_collect_external(ext, &deps);
                self.out.put_home(buf, Deps::new(), home);
                return Ok(());
            }
        }

        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SVM fill is always a non-blocking enqueue (no native CL_BLOCKING flag);
        // Blocking waits on the returned event here. In-place → home threads.
        let event = crate::mapped::svm_fill_enqueue(&buf, ec, self.value, &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn cb_addable(&self) -> bool {
        true
    }

    fn cbable_weight(&self) -> usize {
        // An SVM fill records one clCommandSVMMemFillKHR (where supported).
        1
    }

    fn cb_restamp(&self, evs: &[Dep]) {
        if let Some((v, _d, h)) = self.out.take_home() {
            self.out.put_home(v, Vec::from(evs), h);
        }
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("fill_mapped".into());
    }
}

impl<T, M> FillMapped<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    /// Concrete-head blocking terminal: fill on the buffer's own context default
    /// queue and return the (filled) buffer (`let buf = buf.fill(v).wait()?;`).
    pub fn wait(self) -> Result<MappedSlice<T, M>> {
        let ctx = concrete_svm_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal: enqueue the fill on the buffer's own
    /// context default queue and return a completion [`Event`](crate::Event).
    pub fn submit(self) -> Result<crate::Event> {
        let ctx = concrete_svm_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_on(&*queue)
    }
}

// ── Leaf: write host data into an existing (init) MappedSlice ───────────────

/// Write host `src` into an already-initialised [`MappedSlice`], in place, via a
/// non-blocking `clEnqueueSVMMemcpy` (host-pointer source). SVM analog of
/// [`WriteDevice`]. The buffer passes through as the op's output.
///
/// SVM write stays **non-blocking** regardless of terminal: `submit_on` returns
/// the write event so the copy overlaps downstream work, and
/// `register_drop_callback` keeps the host `src` alive until the memcpy completes
/// (`CL_COMPLETE`). The `Blocking` terminal waits on that same event.
pub struct WriteMapped<T, M: MemMode = ReadWrite> {
    buf: Input<MappedSlice<T, M>>,
    // Retained by value; read by reference each run (see `WriteDevice::src`).
    src: UploadSource<T>,
    out: Pipe<MappedSlice<T, M>>,
}

/// Build an in-place SVM write leaf over an existing `MappedSlice` (concrete or
/// piped). `M: HostWritable` — same gate as [`MappedSlice::write`].
pub fn write_mapped<T, M, S>(buf: impl Into<Input<MappedSlice<T, M>>>, src: S) -> WriteMapped<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteMapped {
        buf: buf.into(),
        src: src.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for WriteMapped<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<()> {
        let (buf, deps, home) = self.buf.resolve_home(ec)?;
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SVM write is always a non-blocking enqueue; Blocking waits on the event
        // here, Pipelined threads it downstream. `self.src` is valid for the whole
        // `sync` — no keep-alive callback needed. In-place → home threads through.
        let event = crate::mapped::svm_write_enqueue(&buf, ec, self.src.as_slice(), &raw)?;
        match mode {
            ExecMode::Blocking => {
                event.wait().map_err(Error::OpenCl)?;
                self.out.put_home(buf, Deps::new(), home);
            }
            ExecMode::Pipelined => {
                self.out.put_home(buf, vec![wrap_event(event)], home);
            }
        }
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_mapped".into());
    }
}

impl<T, M> WriteMapped<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    /// Concrete-head blocking terminal: write on the buffer's own context default
    /// queue and return the buffer for reuse (`let buf = buf.write(d).wait()?;`).
    pub fn wait(self) -> Result<MappedSlice<T, M>> {
        let ctx = concrete_svm_ctx(&self.buf)?;
        self.sync(&ctx).map(Checkout::into_inner)
    }

    /// Concrete-head non-blocking terminal returning the buffer plus a completion
    /// [`Event`](crate::Event) — mirrors the Tier-1 `(Output, Event)` contract.
    pub fn submit(self) -> Result<(MappedSlice<T, M>, crate::Event)> {
        let ctx = concrete_svm_ctx(&self.buf)?;
        let queue = ctx.default_outoforder_queue(ctx.device())?;
        self.submit_value_on(&*queue)
    }
}

// ── Leaf: write host data into a USMSliceUninit → USMSlice (pure host op) ───

/// Eager analog of `WriteFromUninitOp<USMSliceUninit, _>`. Pure host memcpy via
/// the Tier-1 `write_from` helper — surfaces `LengthMismatch` at execute. No
/// enqueue, deps pass through (mode N/A) — mirrors [`Upload`].
pub struct WriteUsmUninit<T: Copy, M: MemMode> {
    uninit: Input<USMSliceUninit<T, M>>,
    // Retained by value; read by reference each run (see `WriteDevice::src`).
    src: UploadSource<T>,
    out: Pipe<USMSlice<T, M>>,
}

/// Build an eager write-from-uninit leaf over a `USMSliceUninit`.
pub fn write_usm_uninit<T, M, S>(
    uninit: impl Into<Input<USMSliceUninit<T, M>>>,
    src: S,
) -> WriteUsmUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
    S: Into<UploadSource<T>>,
{
    WriteUsmUninit {
        uninit: uninit.into(),
        src: src.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for WriteUsmUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<USMSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Host memcpy via Tier-1 helper; Err on length mismatch propagates.
        let (uninit, deps) = self.uninit.resolve(ec)?;
        let buf = uninit.write_from(self.src.as_slice())?;
        self.out.put(buf, deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.uninit.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("write_usm_uninit".into());
    }
}

// ════════════════════════════════════════════════════════════════════════
// usm_op.rs ports — USM alloc / wrap (pure host, synchronous)
// ════════════════════════════════════════════════════════════════════════

// ── Leaf: wrap a host Vec<T> as a USMSlice (eager UsmSliceOp) ───────────────

/// Wrap a host `Vec<T>` as a [`USMSlice<T, M>`], allocating ONCE and re-lending
/// the SAME USM allocation across `g.sync()` replays — the USM twin of
/// [`Upload`], whose reusable structure it mirrors exactly (source leaf, no
/// upstream input; construction is pure host code — `USMSlice::new` — with no
/// enqueue / event).
///
/// Same stable-handle + reseed-on-replay contract as [`Upload`]: on the first run
/// the `Vec` is moved into a `USMSlice` (USM IS that host allocation); on replay
/// the SAME slice is re-lent from this op's home cell, and re-seeded IFF the marker
/// is kernel-writable ([`UploadReseed::RESEED_ON_REPLAY`](crate::UploadReseed)) —
/// a plain host `copy_from_slice` into the same allocation (USM is host memory),
/// keeping a replayed USM chain head idempotent. A kernel read-only marker
/// (`ReadOnly`/`Frozen`) seeds once and skips the replay write.
pub struct UsmSlice<T: Copy, M: MemMode = ReadWrite> {
    // The host source, RETAINED for the seed-once move and any reseed-on-replay
    // (the reseed copies from here into the persistent USM allocation).
    src: UploadSource<T>,
    // The persistent USM slice's home cell: allocated once (first run), then
    // re-lent + re-armed across replays so the SVM pointer stays stable. Empty
    // while lent (busy if already seeded); `None`-on-take is the lend.
    buf: Cell<USMSlice<T, M>>,
    // Whether the slice has ever been allocated/seeded. Distinguishes "first run
    // → alloc" (cell empty, not seeded) from "lent out → busy" (cell empty, seeded).
    seeded: Arc<Mutex<bool>>,
    out: Pipe<USMSlice<T, M>>,
}

/// Build an eager USM-wrap leaf from any `Vec<T>` / `Box<[T]>` / `Arc<[T]>` with
/// the **default [`ReadWrite`] marker** — no turbofish: `usm_slice(data)`. For a
/// non-default marker use [`usm_slice_as`] with a marker witness. Reusable across
/// `sync`s (stable SVM pointer, reseed-on-replay) — the USM twin of [`upload`].
pub fn usm_slice<T, S>(data: S) -> UsmSlice<T, ReadWrite>
where
    T: Copy + Send + Sync + 'static,
    S: Into<UploadSource<T>>,
{
    usm_slice_as(data, ReadWrite)
}

/// Build an eager USM-wrap leaf with an **explicit access marker**, inferred
/// from the `marker` witness — no turbofish: `usm_slice_as(data, HostReadOnly)`.
/// The default-marker shorthand is [`usm_slice`].
pub fn usm_slice_as<T, M, S>(data: S, marker: M) -> UsmSlice<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
    S: Into<UploadSource<T>>,
{
    let _ = marker; // witness only — fixes M, zero-sized, no runtime use.
    UsmSlice {
        src: data.into(),
        buf: Arc::new(Mutex::new(None)),
        seeded: Arc::new(Mutex::new(false)),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for UsmSlice<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::UploadReseed + Send + 'static,
{
    type Output = USMSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<USMSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // The USM slice is allocated ONCE and lives in `self.buf` across runs; its
        // home is that very cell, so a run's Checkout / PipePayload drop returns the
        // SAME SVM allocation here. Three cases, decided by the cell + `seeded` flag
        // — the exact shape `Upload::execute` uses (USMSlice::new is the synchronous
        // host-create analog of DeviceSlice::from_slice; reseed is a host copy).
        let mut seeded = self.seeded.lock().unwrap();
        let lent = self.buf.lock().unwrap().take();
        let buf = match (lent, *seeded) {
            // First run: never seeded → move the host source into a fresh USMSlice
            // (pure host code — USM IS the host allocation, no enqueue/event).
            (None, false) => {
                let buf = USMSlice::<T, M>::new(ec.context(), self.src.as_slice().to_vec())?;
                *seeded = true;
                buf
            }
            // Replay: the slice is back in the cell. Re-lend it; re-seed the host
            // source IF the marker is kernel-writable (it may have been mutated in
            // place last run) — keeping `usm_slice(RW) → … → download` idempotent
            // over a stable SVM pointer. A kernel read-only marker (ReadOnly/Frozen)
            // skips the write: its bytes never changed device-side, seed-once suffices.
            (Some(mut buf), _) => {
                if M::RESEED_ON_REPLAY {
                    // Plain host copy back into the SAME allocation (stable pointer),
                    // after draining in-flight kernel-use events. No SVM map/memcpy —
                    // USM is host memory.
                    buf.reseed_sync(self.src.as_slice())?;
                }
                buf
            }
            // Cell empty but already seeded: the slice is lent out (a prior run's
            // Checkout is still alive) → graph-busy, the concrete-cell contract.
            (None, true) => {
                return Err(Error::NotSupported(
                    "eager graph: a `usm_slice` buffer was already lent and not \
                     returned — a graph is `sync`'d while a previous `Checkout` is \
                     still alive (the graph is busy)",
                ));
            }
        };
        // The home is this op's persistent cell (identity rehome): the slice is
        // returned here on Checkout / PipePayload drop, re-arming the leaf with a
        // STABLE SVM pointer. So a downstream consume rehomes it here, not the
        // releasing drop.
        let home: Option<BoxedHome<USMSlice<T, M>>> = Some(Box::new(Arc::clone(&self.buf)));
        self.out.put_home(buf, Deps::new(), home);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("usm_slice".into());
    }
}

// ── Leaf: alloc an uninit slice (Device / Mapped / USM) ─────────────────────
//
// Producing SOURCE leaf: allocation happens at execute (`<Buf>::alloc_uninit`),
// so the uninit is a graph-produced value a downstream `fill_*_uninit` /
// `write_*_uninit` consumes. The three memory families are identical except the
// buffer type + names, so one macro emits all of them. `mapped_alloc_uninit` on a
// no-SVM device surfaces `Error::SvmNotAvailable` at the graph TERMINAL (the
// `alloc_uninit` call), not eagerly. `_as` takes an explicit access-marker witness;
// the bare ctor defaults to `ReadWrite` (turbofish-free).

macro_rules! impl_alloc_uninit {
    ($Op:ident, $Buf:ident, $Uninit:ident, $ctor:ident, $ctor_as:ident, $label:literal) => {
        #[doc = concat!("Graph leaf that allocates a `", stringify!($Uninit), "<T, M>` at execute. Build via [`", stringify!($ctor), "`].")]
        pub struct $Op<T, M: MemMode = ReadWrite> {
            len: usize,
            out: Pipe<$Uninit<T, M>>,
            _t: PhantomData<fn() -> (T, M)>,
        }

        #[doc = concat!("Build an eager uninit-`", stringify!($Buf), "` alloc leaf with the default [`ReadWrite`] marker (no turbofish). For a non-default marker use [`", stringify!($ctor_as), "`].")]
        pub fn $ctor<T: Send + 'static>(len: usize) -> $Op<T, ReadWrite> {
            $ctor_as(len, ReadWrite)
        }

        #[doc = concat!("Build an eager uninit-`", stringify!($Buf), "` alloc leaf with an explicit access marker inferred from the `marker` witness.")]
        pub fn $ctor_as<T, M>(len: usize, marker: M) -> $Op<T, M>
        where
            T: Send + 'static,
            M: MemMode + Send + 'static,
        {
            let _ = marker;
            $Op { len, out: Pipe::new(), _t: PhantomData }
        }

        impl<T, M> DeviceOp for $Op<T, M>
        where
            T: Send + 'static,
            M: MemMode + Send + 'static,
        {
            type Output = $Uninit<T, M>;

            fn output_pipe(&self) -> Option<Pipe<$Uninit<T, M>>> {
                Some(self.out.clone())
            }

            fn handle(&self) -> Self::Handle {
                self.out.clone()
            }

            fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
                // alloc_uninit is pure host code — no in-flight event, mode N/A.
                let uninit = $Buf::<T, M>::alloc_uninit(ec.context(), self.len)?;
                self.out.put(uninit, Deps::new());
                Ok(())
            }

            fn describe(&self, out: &mut Vec<String>) {
                out.push(format!(concat!($label, "(len={})"), self.len));
            }
        }
    };
}

impl_alloc_uninit!(
    DeviceAllocUninit,
    DeviceSlice,
    DeviceSliceUninit,
    device_alloc_uninit,
    device_alloc_uninit_as,
    "device_alloc_uninit"
);
impl_alloc_uninit!(
    MappedAllocUninit,
    MappedSlice,
    MappedSliceUninit,
    mapped_alloc_uninit,
    mapped_alloc_uninit_as,
    "mapped_alloc_uninit"
);
impl_alloc_uninit!(
    UsmAllocUninit,
    USMSlice,
    USMSliceUninit,
    usm_alloc_uninit,
    usm_alloc_uninit_as,
    "usm_alloc_uninit"
);

// ── Leaf: image upload (host pixels → image I) ──────────────────────────────

/// Allocate an image of type `I` with `dims` and write `pixels` into it.
/// Source-ish leaf (no upstream image input). The underlying image `write_op`
/// has **only a non-blocking enqueue** (no native `wait_on`), so this op always
/// uses `submit_on` and ignores `mode`; the source `pixels` is kept alive until
/// the write event fires via `register_drop_callback`. Mirrors [`Upload`]
/// (chain-entry) but carries a write event because the enqueue is non-blocking.
pub struct ImageUploadEager<I: ImageHostTransfer> {
    // Retained by value; read by reference each run (re-seed). `&self` outlives
    // the whole `sync`, so the host pixels stay valid across the async write —
    // no per-run keep-alive callback needed.
    pixels: Vec<I::Pixel>,
    dims: I::Dims,
    out: Pipe<I>,
    _ty: PhantomData<fn() -> I>,
}

/// Build an eager image-upload leaf.
pub fn image_upload<I>(pixels: Vec<I::Pixel>, dims: I::Dims) -> ImageUploadEager<I>
where
    I: ImageHostTransfer + crate::image::ImageEnqueue + Send + 'static,
    I::Pixel: Send + 'static,
{
    ImageUploadEager {
        pixels,
        dims,
        out: Pipe::new(),
        _ty: PhantomData,
    }
}

impl<I> DeviceOp for ImageUploadEager<I>
where
    I: ImageHostTransfer + crate::image::ImageEnqueue + Send + 'static,
    I::Pixel: Send + 'static,
{
    type Output = I;

    fn output_pipe(&self) -> Option<Pipe<I>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Image write has no native CL_BLOCKING flag we want here — always a
        // non-blocking enqueue, mode ignored; the chain terminal waits.
        let mut img = I::alloc(ec.context(), self.dims)?;
        let region = img.enqueue_region();
        // Source leaf: no upstream Input, so no wait-list to thread. `self.pixels`
        // stays valid for the whole `sync`, so no keep-alive callback is needed.
        let event = crate::image::write_image_enqueue(
            img.image_mut(),
            ec,
            region,
            self.pixels.as_ptr() as *const std::ffi::c_void,
            false,
            &[],
        )?;
        self.out.put(img, vec![wrap_event(event)]);
        Ok(())
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("image_upload".into());
    }
}

// ── Leaf: image download (image I → host Vec<Pixel>) ────────────────────────

/// Consume an upstream image of type `I`, alloc a host `Vec<I::Pixel>`, and
/// read the image into it. The underlying image `read_op` has **only a
/// non-blocking enqueue** (no native `wait_on`), so this op always uses
/// `submit_on` and ignores `mode`. Mirrors [`Download`] (output leaf) but
/// without the blocking branch the buffer read has.
pub struct ImageDownloadEager<I: ImageHostTransfer> {
    img: Input<I>,
    out: Pipe<Vec<I::Pixel>>,
}

/// Build an eager image-download leaf over an upstream image.
pub fn image_download<I>(img: impl Into<Input<I>>) -> ImageDownloadEager<I>
where
    I: ImageHostTransfer + crate::image::ImageEnqueue + Send + 'static,
    I::Pixel: Default + Copy + Send + 'static,
{
    ImageDownloadEager {
        img: img.into(),
        out: Pipe::new(),
    }
}

impl<I> DeviceOp for ImageDownloadEager<I>
where
    I: ImageHostTransfer + crate::image::ImageEnqueue + Send + 'static,
    I::Pixel: Default + Copy + Send + 'static,
{
    type Output = Vec<I::Pixel>;

    fn output_pipe(&self) -> Option<Pipe<Vec<I::Pixel>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Image read enqueued non-blocking; the chain terminal waits, mode ignored.
        let (img, deps) = self.img.resolve(ec)?;
        let pixel_count = img.pixel_count();
        let region = img.enqueue_region();
        let mut pixels = vec![<I::Pixel as Default>::default(); pixel_count];
        let raw: Vec<crate::cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let event = crate::image::read_image_enqueue(
            img.image_ref(),
            ec,
            region,
            pixels.as_mut_ptr() as *mut std::ffi::c_void,
            false,
            &raw,
        )?;
        self.out.put(pixels, vec![wrap_event(event)]);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.img.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("image_download".into());
    }
}

// ════════════════════════════════════════════════════════════════════════
// host_view.rs ports — acquire / release host views (map / unmap)
// ════════════════════════════════════════════════════════════════════════
//
// The host-view types (`DeviceSliceHostView` / `MappedSliceHostView`) have
// private fields, so the eager ops cannot reconstruct them directly. Instead
// each eager op holds the buffer/view and delegates the exact enqueue body to
// the host-view layer's `acquire_host_view{,_read}` / `release_to_device`
// builders and their inherent `run(ec, deps) -> (Output, Deps)` method (the
// map/unmap primitive that survives in `host_view.rs`). None of these has a
// native blocking enqueue (the map/unmap is always non-blocking `false`), so
// `mode` is ignored.

// ── Leaf: acquire a read/write DeviceSlice host view ────────────────────────

/// Acquire a read/write host view of an upstream `DeviceSlice` via a
/// non-blocking `clEnqueueMapBuffer`. Output is the owned
/// [`DeviceSliceHostView`]. No native blocking enqueue — `mode` ignored.
/// Delegates to the `AcquireDeviceSliceOp` body via `acquire_host_view`.
pub struct AcquireDeviceView<T, M: MemMode> {
    buf: Input<DeviceSlice<T, M>>,
    out: Pipe<DeviceSliceHostView<T, M, MapReadWrite>>,
}

/// Build an eager acquire-read/write-view leaf over an upstream `DeviceSlice`.
pub fn acquire_device_view<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
) -> AcquireDeviceView<T, M>
where
    T: Send + 'static,
    M: MemMode + HostWritable + HostReadable,
{
    AcquireDeviceView {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for AcquireDeviceView<T, M>
where
    T: Send + 'static,
    M: MemMode + HostWritable + HostReadable,
{
    type Output = DeviceSliceHostView<T, M, MapReadWrite>;

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve(ec)?;
        // Delegate to the old op's verbatim map body (map/unmap is always
        // non-blocking — mode ignored).
        let (view, out_deps) = buf.acquire_host_view().run(ec, deps)?;
        self.out.put(view, out_deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("acquire_device_view".into());
    }
}

// ── Leaf: acquire a read-only DeviceSlice host view ─────────────────────────

/// Acquire a read-only host view of an upstream `DeviceSlice`
/// (`clEnqueueMapBuffer(CL_MAP_READ)`). Output is the owned
/// [`DeviceSliceHostView`]. No native blocking enqueue — `mode` ignored.
pub struct AcquireDeviceViewRead<T, M: MemMode> {
    buf: Input<DeviceSlice<T, M>>,
    out: Pipe<DeviceSliceHostView<T, M, MapReadOnly>>,
}

/// Build an eager acquire-read-only-view leaf over an upstream `DeviceSlice`.
pub fn acquire_device_view_read<T, M>(
    buf: impl Into<Input<DeviceSlice<T, M>>>,
) -> AcquireDeviceViewRead<T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable,
{
    AcquireDeviceViewRead {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for AcquireDeviceViewRead<T, M>
where
    T: Send + 'static,
    M: MemMode + HostReadable,
{
    type Output = DeviceSliceHostView<T, M, MapReadOnly>;

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve(ec)?;
        let (view, out_deps) = buf.acquire_host_view_read().run(ec, deps)?;
        self.out.put(view, out_deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("acquire_device_view_read".into());
    }
}

// ── Leaf: release a DeviceSlice host view back to the device ─────────────────

/// Enqueue `clEnqueueUnmapMemObject` for an upstream
/// [`DeviceSliceHostView`] and yield the [`DeviceSlice`] back. No native
/// blocking enqueue — `mode` ignored. Generic over the view's map-access mode.
pub struct ReleaseDeviceView<T, M: MemMode, A: MapAccess> {
    view: Input<DeviceSliceHostView<T, M, A>>,
    out: Pipe<DeviceSlice<T, M>>,
}

/// Build an eager release-view leaf over an upstream `DeviceSliceHostView`.
pub fn release_device_view<T, M, A>(
    view: impl Into<Input<DeviceSliceHostView<T, M, A>>>,
) -> ReleaseDeviceView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    ReleaseDeviceView {
        view: view.into(),
        out: Pipe::new(),
    }
}

impl<T, M, A> DeviceOp for ReleaseDeviceView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    type Output = DeviceSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<DeviceSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (view, deps) = self.view.resolve(ec)?;
        let (buf, out_deps) = view.release_to_device().run(ec, deps)?;
        self.out.put(buf, out_deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.view.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("release_device_view".into());
    }
}

// ── Leaf: acquire a read/write MappedSlice (SVM) host view ──────────────────

/// Acquire a read/write SVM host view of an upstream `MappedSlice` via a
/// non-blocking `clEnqueueSVMMap`. No native blocking enqueue — `mode` ignored.
pub struct AcquireMappedView<T, M: MemMode> {
    buf: Input<MappedSlice<T, M>>,
    out: Pipe<MappedSliceHostView<T, M, MapReadWrite>>,
}

/// Build an eager acquire-read/write-SVM-view leaf over a `MappedSlice`.
pub fn acquire_mapped_view<T, M>(
    buf: impl Into<Input<MappedSlice<T, M>>>,
) -> AcquireMappedView<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + HostReadable,
{
    AcquireMappedView {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for AcquireMappedView<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + HostReadable,
{
    type Output = MappedSliceHostView<T, M, MapReadWrite>;

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve(ec)?;
        let (view, out_deps) = buf.acquire_host_view().run(ec, deps)?;
        self.out.put(view, out_deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("acquire_mapped_view".into());
    }
}

// ── Leaf: acquire a read-only MappedSlice (SVM) host view ───────────────────

/// Acquire a read-only SVM host view of an upstream `MappedSlice`
/// (`clEnqueueSVMMap(CL_MAP_READ)`). No native blocking enqueue — `mode`
/// ignored.
pub struct AcquireMappedViewRead<T, M: MemMode> {
    buf: Input<MappedSlice<T, M>>,
    out: Pipe<MappedSliceHostView<T, M, MapReadOnly>>,
}

/// Build an eager acquire-read-only-SVM-view leaf over a `MappedSlice`.
pub fn acquire_mapped_view_read<T, M>(
    buf: impl Into<Input<MappedSlice<T, M>>>,
) -> AcquireMappedViewRead<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostReadable,
{
    AcquireMappedViewRead {
        buf: buf.into(),
        out: Pipe::new(),
    }
}

impl<T, M> DeviceOp for AcquireMappedViewRead<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostReadable,
{
    type Output = MappedSliceHostView<T, M, MapReadOnly>;

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (buf, deps) = self.buf.resolve(ec)?;
        let (view, out_deps) = buf.acquire_host_view_read().run(ec, deps)?;
        self.out.put(view, out_deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.buf.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("acquire_mapped_view_read".into());
    }
}

// ── Leaf: release a MappedSlice (SVM) host view back to the device ───────────

/// Enqueue `clEnqueueSVMUnmap` for an upstream [`MappedSliceHostView`] and
/// yield the [`MappedSlice`] back. No native blocking enqueue — `mode` ignored.
/// Generic over the view's map-access mode.
pub struct ReleaseMappedView<T, M: MemMode, A: MapAccess> {
    view: Input<MappedSliceHostView<T, M, A>>,
    out: Pipe<MappedSlice<T, M>>,
}

/// Build an eager release-SVM-view leaf over an upstream `MappedSliceHostView`.
pub fn release_mapped_view<T, M, A>(
    view: impl Into<Input<MappedSliceHostView<T, M, A>>>,
) -> ReleaseMappedView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    ReleaseMappedView {
        view: view.into(),
        out: Pipe::new(),
    }
}

impl<T, M, A> DeviceOp for ReleaseMappedView<T, M, A>
where
    T: Send + 'static,
    M: MemMode,
    A: MapAccess,
{
    type Output = MappedSlice<T, M>;

    fn output_pipe(&self) -> Option<Pipe<MappedSlice<T, M>>> {
        Some(self.out.clone())
    }

    fn handle(&self) -> Self::Handle {
        self.out.clone()
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        let (view, deps) = self.view.resolve(ec)?;
        let (buf, out_deps) = view.release_to_device().run(ec, deps)?;
        self.out.put(buf, out_deps);
        Ok(())
    }

    fn check_ready(&self) -> Result<()> {
        self.view.check_ready()
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("release_mapped_view".into());
    }
}

// ── Multi-output leaf: copy_to (src, dst) → (src, dst) ──────────────────────
//
// The `copy_to` graph leaf. A copy is a **two-output** op: it returns BOTH the
// source and destination buffers so the chain can thread either onward. It
// mirrors the macro-emitted multi-output kernel shape (commit 0f7083d): two
// element pipes (`Handle = (Pipe<OS>, Pipe<OD>)`), `execute` enqueues once and
// scatters each output into its element pipe (cloning the single completion
// `Dep` onto both), and `into_output` drains both pipes to reconstruct the
// `(src, dst)` tuple.
//
// Rather than re-deriving the ten (src, dst) family bodies (incl. the unsafe
// cross-type SVM-memcpy machinery in `copy.rs`), this op **reuses** the
// `CopyTo` / [`DeviceEnqueue`] `CopyToOp` impls: resolve the two inputs, build
// the op via `src.copy_to(dst)`, run its `DeviceEnqueue::run` (which owns every
// per-family primitive + Uninit→Init transition + buffer-use registration), then
// scatter its `(out_src, out_dst)` Output across the two pipes. All ten families
// come along for free — no `copy.rs` change.
//
// Copy ops have no native blocking enqueue (`submit_on` + event is the only
// path); `mode` is therefore ignored, and copy is rarely terminal anyway (it
// returns buffers onward).

/// Split a copy op's 2-tuple `Output` into its source + destination halves so
/// the eager [`CopyTo2`] op can hold one typed element pipe per side and
/// reconstruct the tuple in `into_output`. Implemented once for every `(A, B)`.
pub trait CopyOutputs {
    /// The post-copy source buffer (element 0 of the copy Output).
    type Src: Send;
    /// The post-copy destination buffer (element 1 of the copy Output).
    type Dst: Send;
    /// Decompose into `(src, dst)`.
    fn into_parts(self) -> (Self::Src, Self::Dst);
}

impl<A: Send, B: Send> CopyOutputs for (A, B) {
    type Src = A;
    type Dst = B;
    fn into_parts(self) -> (A, B) {
        self
    }
}

// ── CopyHome: build a copy output's return home from its input cell ─────
//
// A copy lends concrete `src`/`dst` cells but may RE-TYPE the dst (`Uninit →
// Init`). `CopyHome<Out>` lets the (input-typed) cell rehome the (output-typed)
// value: identity when the input type already equals `Out`, or a downgrade when
// the input is the `Uninit` wrapper of `Out`. Implemented per buffer family;
// the [`CopyTo2`] `DeviceOp` impl bounds `Src`/`Dst` by it so every supported
// `(src, dst)` pair threads homes. A family that can't express the downgrade
// returns `None` (still safe — that side just doesn't re-arm).

/// Build the typed return [`Rehome`] for a copy output of type `Out` from the
/// (possibly weaker-typed) input cell of type `Self`. `None` when this family
/// can't soundly express the return.
pub trait CopyHome<Out>: Sized {
    /// The home that returns an `Out` into a concrete `Cell<Self>` on `Checkout`
    /// drop (or `PipePayload` drop).
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<Out>>;

    /// The home that returns an `Out` into a **slot** `SlotCell<Self>` — the
    /// four-state analogue used when a copy operand is a `slot!(Tag)` directly
    /// (scenario 6). Re-arms `Lent → Bound` on rehome and severs `Lent → Severed`
    /// on `into_inner`. Default `None`: a slot's value type is always an `Init`
    /// buffer (`Tag::Value`), so only the identity `CopyHome` impls (`Self == Out`)
    /// override this; the `Uninit → Init` downgrade impls keep the default (an
    /// uninit buffer is never a slot value, so the path is unreachable).
    fn copy_slot_home(cell: SlotCell<Self>) -> Option<BoxedHome<Out>> {
        let _ = cell;
        None
    }

    /// Forward a home threaded on a **lent pipe** input (a cross-graph `Checkout`
    /// fed as a copy operand — see `Input::lent`) as the copy output's return
    /// home. A lent-pipe operand carries the ORIGIN graph's home on its payload;
    /// the copy must pass it through so the borrowed buffer RETURNS to the origin
    /// on the copy `Checkout`'s drop (LEND semantics, matching the kernel-arg
    /// path). Only reachable for the identity impls (`Self == Out`) — a lent
    /// `Checkout` is always an `Init` buffer, so the copy never retypes it — hence
    /// the default `None` (the `Uninit → Init` downgrade impls never see one).
    fn pipe_home(home: BoxedHome<Self>) -> Option<BoxedHome<Out>> {
        let _ = home;
        None
    }
}

/// Rehome that DOWNGRADES an `Init` buffer back into a `Cell<Uninit-wrapper>`.
/// Re-wraps via the family's `from_init` (a safe private-field re-wrap) before
/// storing — `Init` is the stronger capability, so forgetting it is sound.
struct DowngradeRehome<U, Init> {
    cell: Cell<U>,
    wrap: fn(Init) -> U,
}

impl<U: Send, Init: Send> Rehome<Init> for DowngradeRehome<U, Init> {
    fn rehome(self: Box<Self>, value: Init) {
        *self.cell.lock().unwrap() = Some((self.wrap)(value));
    }
}

// Identity homes: src is never retyped, and an Init→Init dst is identity too.
// `copy_slot_home` returns the four-state `SlotHome` so a `slot!()` copy operand
// re-arms its slot (`Lent → Bound`) on rehome / severs (`Lent → Severed`) on
// `into_inner` — exactly like a slot in a kernel-arg position.
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<DeviceSlice<T, M>>
    for DeviceSlice<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<DeviceSlice<T, M>>> {
        Some(Box::new(cell))
    }
    fn copy_slot_home(cell: SlotCell<Self>) -> Option<BoxedHome<DeviceSlice<T, M>>> {
        Some(Box::new(SlotHome { cell }))
    }
    fn pipe_home(home: BoxedHome<Self>) -> Option<BoxedHome<DeviceSlice<T, M>>> {
        Some(home)
    }
}
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<MappedSlice<T, M>>
    for MappedSlice<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<MappedSlice<T, M>>> {
        Some(Box::new(cell))
    }
    fn copy_slot_home(cell: SlotCell<Self>) -> Option<BoxedHome<MappedSlice<T, M>>> {
        Some(Box::new(SlotHome { cell }))
    }
    fn pipe_home(home: BoxedHome<Self>) -> Option<BoxedHome<MappedSlice<T, M>>> {
        Some(home)
    }
}
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<USMSlice<T, M>> for USMSlice<T, M> {
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<USMSlice<T, M>>> {
        Some(Box::new(cell))
    }
    fn copy_slot_home(cell: SlotCell<Self>) -> Option<BoxedHome<USMSlice<T, M>>> {
        Some(Box::new(SlotHome { cell }))
    }
    fn pipe_home(home: BoxedHome<Self>) -> Option<BoxedHome<USMSlice<T, M>>> {
        Some(home)
    }
}

// Downgrade homes: an Uninit dst comes back Init; re-wrap into the uninit cell.
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<DeviceSlice<T, M>>
    for DeviceSliceUninit<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<DeviceSlice<T, M>>> {
        Some(Box::new(DowngradeRehome {
            cell,
            wrap: DeviceSliceUninit::from_init,
        }))
    }
}
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<MappedSlice<T, M>>
    for MappedSliceUninit<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<MappedSlice<T, M>>> {
        Some(Box::new(DowngradeRehome {
            cell,
            wrap: MappedSliceUninit::from_init,
        }))
    }
}
// USM uninit's backing is a `Vec<MaybeUninit<T>>`, so its `from_init` is a
// same-layout `Vec` reinterpret (Init→Uninit, the SAFE downgrade direction —
// the inverse of `assume_init`, with no init assertion). It preserves the heap
// address so the SVM pointer stays valid. Re-arms like the other two families.
impl<T: Send + 'static, M: MemMode + Send + 'static> CopyHome<USMSlice<T, M>>
    for USMSliceUninit<T, M>
{
    fn copy_home(cell: Cell<Self>) -> Option<BoxedHome<USMSlice<T, M>>> {
        Some(Box::new(DowngradeRehome {
            cell,
            wrap: USMSliceUninit::from_init,
        }))
    }
}

/// Eager multi-output copy: `eager_copy_to(src, dst)` enqueues a copy and yields
/// `(src, dst)`. `Handle = (Pipe<OutSrc>, Pipe<OutDst>)` — two element pipes, so
/// a downstream `.and_then(|(src, dst)| …)` selects either side. Polymorphic
/// over every supported `(src, dst)` family via the `Src: CopyTo<Dst>` bound.
pub struct CopyTo2<Src, Dst>
where
    Src: CopyTo<Dst>,
    <Src::Op as crate::eager::DeviceEnqueue>::Output: CopyOutputs,
{
    src: Input<Src>,
    dst: Input<Dst>,
    // One element pipe per copy output (move-once storage), mirroring the
    // macro-emitted multi-output kernel. The output tuple is reconstructed from
    // both in `into_output`.
    src_pipe: Pipe<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src>,
    dst_pipe: Pipe<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst>,
    /// Design-v2 CB home: a copy records `clCommandCopyBufferKHR` (cl_mem↔cl_mem) or
    /// `clCommandSVMMemcpyKHR` (SVM↔SVM) where the extension provides it; a mixed
    /// cl_mem/SVM pair or absent PFN falls back to software.
    cb_cache: CbCache,
}

/// A value usable as a [`eager_copy_to`] **operand**: a concrete buffer, an
/// upstream [`Pipe`], a [`Checkout`], or a [`slot!`](crate::slot)`(Tag)` hole. It
/// resolves to an `Input<Buf>` over the concrete buffer family `Buf` the copy
/// then drives (via `Buf: CopyTo<…>`).
///
/// ## Why a dedicated trait (not `Into<Input<Buf>>`)
///
/// `SlotHandle<Tg>` cannot impl `Into<Input<Tg::Value>>` — the blanket
/// `From<T> for Input<T>` blocks it under coherence (the compiler can't rule out
/// `Tg::Value == SlotHandle<Tg>`). `CopyOperand` is a distinct nominal trait with
/// no such clash, so a slot plugs straight into a copy operand position
/// (`eager_copy_to(slot!(Src), slot!(Dst))`) exactly as it already does in a
/// kernel-arg position via [`ToInput`]. Concrete buffers / pipes / checkouts route
/// through their existing `Into<Input<_>>` conversions.
pub trait CopyOperand<Buf> {
    /// Resolve into the copy's input edge over the concrete buffer type `Buf`.
    fn into_copy_input(self) -> Input<Buf>;
}

// A slot plugs into a copy operand position, mirroring its kernel-arg `ToInput`.
impl<Tg: Tag> CopyOperand<Tg::Value> for SlotHandle<Tg> {
    fn into_copy_input(self) -> Input<Tg::Value> {
        self.into_input()
    }
}

// A `Pipe<Buf>` (upstream producer's output edge) → a deferred input. Per-type
// (not a blanket over `Into<Input<_>>`) so it stays disjoint from the `SlotHandle`
// impl — a blanket would collide because the compiler can't rule out
// `Tg::Value == SlotHandle<Tg>`.
impl<Buf> CopyOperand<Buf> for Pipe<Buf> {
    fn into_copy_input(self) -> Input<Buf> {
        Input::Pipe(self)
    }
}

/// Implement [`CopyOperand`] for a concrete buffer family + its `Checkout`
/// wrapper (each a distinct nominal type, disjoint from the slot/pipe impls).
macro_rules! impl_copy_operand_concrete {
    ($buf:ident) => {
        impl<E, M> CopyOperand<$crate::$buf<E, M>> for $crate::$buf<E, M>
        where
            M: $crate::MemMode,
        {
            fn into_copy_input(self) -> Input<$crate::$buf<E, M>> {
                Input::from(self)
            }
        }
        impl<E, M> CopyOperand<$crate::$buf<E, M>> for Checkout<$crate::$buf<E, M>>
        where
            M: $crate::MemMode,
            E: Send,
        {
            // LEND: relocate the value + its home onto a pre-loaded pipe so the
            // home rides into the copy's graph and returns to A on drop — A stays
            // BUSY while the borrow is held, then re-arms for a plain `sync()` (no
            // `mutate_bind`). Identical semantics to the `ToInput`/`From` Checkout
            // arg paths; `.into_inner()` remains the explicit sever verb.
            fn into_copy_input(self) -> Input<$crate::$buf<E, M>> {
                let (value, home) = self.into_value_and_home();
                Input::lent(value, home)
            }
        }
    };
}
impl_copy_operand_concrete!(DeviceSlice);
impl_copy_operand_concrete!(MappedSlice);
impl_copy_operand_concrete!(USMSlice);

// The Uninit dst families are valid copy *destinations* (never a slot value), so
// they need a concrete + checkout operand impl too.
macro_rules! impl_copy_operand_uninit {
    ($buf:ident) => {
        impl<E, M> CopyOperand<$crate::$buf<E, M>> for $crate::$buf<E, M>
        where
            M: $crate::MemMode,
        {
            fn into_copy_input(self) -> Input<$crate::$buf<E, M>> {
                Input::from(self)
            }
        }
    };
}
impl_copy_operand_uninit!(DeviceSliceUninit);
impl_copy_operand_uninit!(MappedSliceUninit);
impl_copy_operand_uninit!(USMSliceUninit);

/// Build an eager copy leaf. `src` / `dst` may each be a concrete buffer, an
/// upstream [`Pipe`], a [`Checkout`], or a [`slot!`](crate::slot)`(Tag)` hole (see
/// [`CopyOperand`]). Output is `(src, dst)` (an `Uninit` dst comes back `Init` —
/// the copy wrote every byte). See [`CopyTo2`].
pub fn eager_copy_to<Src, Dst, S, D>(src: S, dst: D) -> CopyTo2<Src, Dst>
where
    Src: CopyTo<Dst>,
    <Src::Op as crate::eager::DeviceEnqueue>::Output: CopyOutputs,
    S: CopyOperand<Src>,
    D: CopyOperand<Dst>,
{
    CopyTo2 {
        src: src.into_copy_input(),
        dst: dst.into_copy_input(),
        src_pipe: Pipe::new(),
        dst_pipe: Pipe::new(),
        cb_cache: new_cb_cache(),
    }
}

impl<Src, Dst> DeviceOp for CopyTo2<Src, Dst>
where
    // `RecordableBuffer` on both operands lets the folded `record` override resolve
    // each concrete buffer's handle (a copy operand is always a device buffer that
    // satisfies it — `DeviceSlice`/`MappedSlice`/`USMSlice` + their `Uninit` dst
    // forms), so the CB path records a copy leaf; no observable narrowing.
    Src: CopyTo<Dst> + Send + crate::record::RecordableBuffer + 'static,
    Dst: Send + crate::record::RecordableBuffer + 'static,
    Src::Op: Send,
    <Src::Op as crate::eager::DeviceEnqueue>::Output: CopyOutputs,
    // Each input cell knows how to rehome its (possibly retyped) output: src is
    // identity (never retyped), dst is identity or the Uninit→Init downgrade.
    Src: CopyHome<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src>,
    Dst: CopyHome<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst>,
{
    type Output = (
        <<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src,
        <<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst,
    );
    // Two element pipes, like the multi-output kernel: the downstream closure
    // gets `(pa, pb)` and selects either buffer.
    type Handle = (
        Pipe<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src>,
        Pipe<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst>,
    );
    // Per-output Checkouts: each side independently readable / into_inner'd.
    type Checkouts = (
        Checkout<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Src>,
        Checkout<<<Src::Op as crate::eager::DeviceEnqueue>::Output as CopyOutputs>::Dst>,
    );

    fn output_pipe(&self) -> Option<Pipe<Self::Output>> {
        // Multi-output storage is the per-element pipes; there is no single
        // storage pipe (the default `into_output` is overridden, and `and_then`
        // uses `handle()`), so return `None`.
        None
    }

    fn handle(&self) -> Self::Handle {
        (self.src_pipe.clone(), self.dst_pipe.clone())
    }

    fn execute(&self, ec: &ExecutionContext<'_>, _mode: ExecMode) -> Result<()> {
        // Resolve each input → (buffer, upstream Deps, output-typed return home),
        // threading the home onto the output element pipe (re-arming `g` on Checkout
        // / PipePayload drop). `resolve_copy` unifies all three arms under the home
        // invariant: a CONCRETE cell routes through `CopyHome::copy_home` (identity,
        // or the `Uninit → Init` downgrade re-wrap); a SLOT routes through
        // `CopyHome::copy_slot_home` (a four-state `SlotHome` — re-arms `Lent →
        // Bound`, severs on `into_inner`), closing the former copy-slot gap; a
        // LENT pipe (a cross-graph `Checkout` fed as a copy operand) forwards the
        // ORIGIN's home via `CopyHome::pipe_home` so the borrow RETURNS to it on
        // drop (LEND, matching the kernel-arg path); a minted-upstream pipe is
        // `None`. Either input may be a pipe or concrete — combine their wait-lists.
        let (src, src_deps, src_home) = self.src.resolve_copy(ec)?;
        let (dst, dst_deps, dst_home) = self.dst.resolve_copy(ec)?;

        // ── CB-mode fork (design v2) — record clCommandCopyBufferKHR (cl_mem) or
        // clCommandSVMMemcpyKHR (SVM) via the copy op's `record_cb` (which also does
        // the Uninit→Init type conversion). Build records; LendOnly just converts;
        // an ineligible pair (mixed cl_mem/SVM, absent PFN) falls through to per-op. ─
        match ec.cb() {
            CbWalk::Build { builder, ext, .. } => {
                // Two inputs → two outputs: run the shared prologue per side (each
                // collects its deps, notes its slot origins into the CB, and
                // propagates them onto its OWN output pipe); union the wait-lists.
                let mut waits = cb_leaf_build(
                    ec,
                    builder,
                    ext,
                    &src_deps,
                    self.src.slot_cell_id(),
                    self.src.pipe_cell_id(),
                    self.src_pipe.cell_id(),
                );
                waits.extend(cb_leaf_build(
                    ec,
                    builder,
                    ext,
                    &dst_deps,
                    self.dst.slot_cell_id(),
                    self.dst.pipe_cell_id(),
                    self.dst_pipe.cell_id(),
                ));
                // `record_cb(Some)` ALWAYS returns the type-converted output; the sync
                // point is `Some` when the command recorded, `None` when the copy was
                // ineligible (mixed cl_mem/SVM, or absent PFN) — in which case it has
                // already called `builder.mark_ineligible()`, so the boundary will
                // DISCARD this CB and re-run the whole span in Off. Either way we
                // deposit the (lent) buffers with empty deps: on the eligible path the
                // CB does the copy; on the discard path this deposit is thrown away and
                // recomputed by the Off re-run. Ordering inside the CB is sync points.
                let (out, sp) = src
                    .copy_to(dst)
                    .record_cb(Some(builder), &waits)
                    .expect("copy record_cb(Some) always returns the converted output");
                let (out_src, out_dst) = out.into_parts();
                if let Some(sp) = sp {
                    let set = std::collections::BTreeSet::from([sp]);
                    ec.sp_register(self.src_pipe.cell_id(), set.clone());
                    ec.sp_register(self.dst_pipe.cell_id(), set);
                }
                self.src_pipe.put_home(out_src, Deps::new(), src_home);
                self.dst_pipe.put_home(out_dst, Deps::new(), dst_home);
                return Ok(());
            }
            CbWalk::LendOnly { .. } => {
                // Replay: the cached CB does the copy. Convert types only (no builder),
                // deposit with empty deps.
                let (out, _none) = src
                    .copy_to(dst)
                    .record_cb(None, &std::collections::BTreeSet::new())
                    .expect("copy record_cb(None) always converts types on replay");
                let (out_src, out_dst) = out.into_parts();
                self.src_pipe.put_home(out_src, Deps::new(), src_home);
                self.dst_pipe.put_home(out_dst, Deps::new(), dst_home);
                return Ok(());
            }
            CbWalk::Off => {}
        }

        let mut deps = src_deps;
        deps.extend(dst_deps);
        // Reuse the closure-layer copy op: it owns the right per-family
        // primitive (CopyBuffer / SVMMemcpy), the Uninit→Init transition, and
        // buffer-use registration. ONE enqueue → its returned Deps hold one
        // completion event.
        let op = src.copy_to(dst);
        let (out, out_deps) = op.run(ec, deps)?;
        let (out_src, out_dst) = out.into_parts();
        // Clone the completion Dep onto BOTH element pipes so whichever side
        // flows downstream carries the wait-list (and the terminal reconstruct
        // gathers from both). Each output carries its return home: SRC is an
        // identity rehome (the copy never retypes the source); DST is identity
        // (Init→Init) or a sound DOWNGRADE (Uninit dst comes back Init, re-wrapped
        // into its `Cell<…Uninit>` by `CopyHome`). So a concrete-buffer copy in a
        // reused graph re-arms both cells on `Checkout` drop.
        self.src_pipe.put_home(out_src, out_deps.clone(), src_home);
        self.dst_pipe.put_home(out_dst, out_deps, dst_home);
        Ok(())
    }

    fn collect(&self, ec: &ExecutionContext<'_>, mode: ExecMode) -> Result<(Self::Output, Deps)>
    where
        Self: Sized,
    {
        // Grab the element pipes before consuming `self`, then scatter via
        // `execute`, then drain + reconstruct the `(src, dst)` tuple, gathering
        // both pipes' deps (the terminal `into_output` waits on them once).
        let src_pipe = self.src_pipe.clone();
        let dst_pipe = self.dst_pipe.clone();
        self.execute(ec, mode)?;
        let (out_src, mut deps) = src_pipe.take().ok_or(Error::NotSupported(
            "eager graph: terminal copy produced no output",
        ))?;
        let (out_dst, dst_deps) = dst_pipe.take().ok_or(Error::NotSupported(
            "eager graph: terminal copy produced no output",
        ))?;
        deps.extend(dst_deps);
        Ok(((out_src, out_dst), deps))
    }

    #[allow(clippy::type_complexity)]
    fn collect_home(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Output, Deps, Option<BoxedHome<Self::Output>>)>
    where
        Self: Sized,
        Self::Output: Send + 'static,
    {
        // Multi-output (`(src, dst)`): each side's home rides its own `Checkout`
        // via `gather_checkouts`, not one collapsed tuple home. Nested as a bundle
        // branch it collapses to `home == None`. Delegate to `collect`.
        let (value, deps) = self.collect(ec, mode)?;
        Ok((value, deps, None))
    }

    fn gather_checkouts(
        &self,
        ec: &ExecutionContext<'_>,
        mode: ExecMode,
    ) -> Result<(Self::Checkouts, Deps)> {
        // Drain each element pipe with its own home → a tuple of independent
        // Checkouts. Each output carries the home `execute` threaded (concrete cell,
        // slot, or `None` for a pipe), so the two sides re-arm independently:
        // dropping one side's Checkout rehomes it while `into_inner` on the other
        // severs only that side (scenario 11).
        let src_pipe = self.src_pipe.clone();
        let dst_pipe = self.dst_pipe.clone();
        self.execute(ec, mode)?;
        let (out_src, mut deps, src_home) = src_pipe.take_home().ok_or(Error::NotSupported(
            "eager graph: terminal copy produced no output",
        ))?;
        let (out_dst, dst_deps, dst_home) = dst_pipe.take_home().ok_or(Error::NotSupported(
            "eager graph: terminal copy produced no output",
        ))?;
        deps.extend(dst_deps);
        Ok((
            (
                Checkout::new(out_src, src_home),
                Checkout::new(out_dst, dst_home),
            ),
            deps,
        ))
    }

    fn describe(&self, out: &mut Vec<String>) {
        out.push("copy_to".into());
    }

    fn cb_cache(&self) -> Option<&CbCache> {
        Some(&self.cb_cache)
    }

    fn cb_addable(&self) -> bool {
        true
    }

    fn cbable_weight(&self) -> usize {
        // A copy records one command (clCommandCopyBufferKHR / clCommandSVMMemcpyKHR).
        1
    }

    fn cb_restamp(&self, evs: &[Dep]) {
        // Multi-output: stamp the CB completion event onto BOTH element pipes.
        if let Some((v, _d, h)) = self.src_pipe.take_home() {
            self.src_pipe.put_home(v, Vec::from(evs), h);
        }
        if let Some((v, _d, h)) = self.dst_pipe.take_home() {
            self.dst_pipe.put_home(v, Vec::from(evs), h);
        }
    }

    fn bind_slots(&self, binder: &mut SlotBinder) {
        // A copy's src/dst may each be a `slot!()` operand; offer the binder to
        // both (execution order: src then dst), short-circuiting once it lands.
        // Non-slot (concrete / pipe) inputs are a no-op in `try_bind_slot`.
        self.src.try_bind_slot(binder);
        if binder.is_consumed() {
            return;
        }
        self.dst.try_bind_slot(binder);
    }

    fn check_ready(&self) -> Result<()> {
        // Both operands are resolved in `execute` (src then dst) — check both,
        // read-only, fail-fast on the first unsatisfiable one.
        self.src.check_ready()?;
        self.dst.check_ready()
    }

    fn reclaim_undelivered(&self) {
        // Two element pipes (src, dst). Drain + rehome each undelivered side so a
        // copy whose output is partly discarded (e.g. `…and_then(|(src, _dst)| …)`)
        // returns the dropped side's buffer to its origin cell. Already-drained
        // pipes (delivered to a terminal Checkout / consumed downstream) are no-ops.
        if let Some((v, _d, home)) = self.src_pipe.take_home() {
            rehome_consumed(v, home);
        }
        if let Some((v, _d, home)) = self.dst_pipe.take_home() {
            rehome_consumed(v, home);
        }
    }
}
