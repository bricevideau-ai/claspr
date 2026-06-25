//! Record-and-replay for eager device graphs (layer 1: software backend).
//!
//! A recordable [`DeviceOp`] chain can be recorded once into a reusable command
//! list and **replayed** many times — the foundation for reusable pipelines.
//! This is *staging* infrastructure: the surface (`record` / `replay`) is
//! provisional and may change as the design settles.
//!
//! # Why the eager engine makes this clean
//!
//! The eager `AndThen{source, next}` stores its **built** children (not a
//! closure), so the graph is structurally inspectable without running it.
//! Recording therefore walks the graph by **`&self`** and threads buffer
//! handles producer→consumer **through the same pipe topology** that `execute`
//! threads values through: each op resolves its inputs (a concrete entry-leaf
//! buffer, or a producer's output looked up by the pipe's
//! [`cell_id`](crate::eager::Pipe)), records its device commands, and registers
//! its output handle(s) under its output pipe(s). Because recording reads
//! handles by reference and never moves the buffers, it needs no keep-alive: the
//! recorded [`RecordedGraph`] borrows the source graph for `'g`, so the buffers
//! its commands reference stay live across every replay.
//!
//! # Recordability is a compile-time property
//!
//! [`RecordableOp`] is a sub-trait of `DeviceOp`. Device-side leaves (fill,
//! copy, kernels) implement it; host-touching leaves (`Upload`, `Download`,
//! `AndThenHost`, `OnDevice`, …) deliberately do **not**, and a combinator is
//! `RecordableOp` only when its children are. So a chain containing a
//! non-recordable leaf fails to compile at [`record`](RecordExt::record), naming
//! the offending leaf through the generic wrappers.
//!
//! # Scope of layer 1
//!
//! - Software backend only: a `Vec` of commands with **structural** dependency
//!   edges (indices), replayed by re-issuing on a queue with **fresh** events
//!   each call. No `cl_khr_command_buffer` yet (layer 2, behind the same
//!   [`RecordContext`] seam — leaf bodies won't change).
//! - Replay re-runs against the **same** buffers. Rebinding inputs (replay
//!   against different buffers) is a later layer (slots + mutable dispatch).

use crate::eager::DeviceOp;
use crate::error::{Error, Result};
use crate::queue::Launcher;
use opencl3::types::{cl_command_queue, cl_event, cl_kernel, cl_mem, cl_uint};
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

// ── Handles + sync points: the recording twins of (value, Deps) ─────────────

/// The recording twin of a buffer-typed output: a raw `cl_mem` + byte length,
/// threaded producer→consumer through the record walk where `execute` threads
/// the owned `DeviceSlice`. `Copy` — the buffer it refers to is owned by the
/// source graph and must outlive the recording.
#[derive(Clone, Copy)]
pub struct BufHandle {
    /// The backing device memory.
    pub mem: cl_mem,
    /// Size of the buffer in bytes.
    pub byte_len: usize,
}

// SAFETY: a raw `cl_mem` + length. `cl_mem` is an opaque handle into the
// internally-synchronized OpenCL runtime; it is only used to issue commands on a
// single queue during replay. The owning `DeviceSlice` carries the real
// Send/Sync story; this is a borrowed view kept live by the `RecordedGraph`.
unsafe impl Send for BufHandle {}
unsafe impl Sync for BufHandle {}

/// A structural reference to an earlier command — its index in the command list.
/// The recording twin of a `cl_event` dependency.
pub type SyncPoint = usize;

/// The wait-list a recorded op threads to its consumers — the recording twin of
/// [`Deps`](crate::eager::Deps).
pub type SyncPoints = Vec<SyncPoint>;

/// What a producer registered for one of its output pipes: the buffer handle
/// plus the sync points a consumer must wait on.
#[derive(Clone)]
struct EdgeValue {
    handle: BufHandle,
    waits: SyncPoints,
}

// ── Software command list (the layer-1 recording target) ────────────────────

/// One recorded command — the software twin of a `clCommand*KHR` entry. Baked
/// args by raw handle + **structural** dependency edges (`waits`: indices of
/// earlier commands), NOT `cl_event`s.
enum SoftCommand {
    Fill {
        buffer: cl_mem,
        pattern: Vec<u8>,
        offset: usize,
        size: usize,
        waits: Vec<SyncPoint>,
    },
    Copy {
        src: cl_mem,
        dst: cl_mem,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        waits: Vec<SyncPoint>,
    },
    NdRange {
        /// Retained at record (`clRetainKernel`), released on `Drop`.
        kernel: cl_kernel,
        global: Vec<usize>,
        local: Vec<usize>,
        waits: Vec<SyncPoint>,
    },
}

impl SoftCommand {
    fn waits(&self) -> &[SyncPoint] {
        match self {
            SoftCommand::Fill { waits, .. }
            | SoftCommand::Copy { waits, .. }
            | SoftCommand::NdRange { waits, .. } => waits,
        }
    }

    /// Enqueue this command on `queue` with `wait` as the event wait-list.
    ///
    /// # Safety
    /// Baked handles must be valid on `queue`'s context (kept live by the
    /// `RecordedGraph` borrow) and `wait` must be live events.
    unsafe fn enqueue(&self, queue: cl_command_queue, wait: &[cl_event]) -> Result<cl_event> {
        use cl3::command_queue as q;
        let wait_ptr = if wait.is_empty() {
            ptr::null()
        } else {
            wait.as_ptr()
        };
        let n = wait.len() as cl_uint;
        let ev = match self {
            SoftCommand::Fill {
                buffer,
                pattern,
                offset,
                size,
                ..
            } => unsafe {
                q::enqueue_fill_buffer(
                    queue,
                    *buffer,
                    pattern.as_ptr() as *const c_void,
                    pattern.len(),
                    *offset,
                    *size,
                    n,
                    wait_ptr,
                )
            },
            SoftCommand::Copy {
                src,
                dst,
                src_offset,
                dst_offset,
                size,
                ..
            } => unsafe {
                q::enqueue_copy_buffer(
                    queue,
                    *src,
                    *dst,
                    *src_offset,
                    *dst_offset,
                    *size,
                    n,
                    wait_ptr,
                )
            },
            SoftCommand::NdRange {
                kernel,
                global,
                local,
                ..
            } => unsafe {
                q::enqueue_nd_range_kernel(
                    queue,
                    *kernel,
                    global.len() as cl_uint,
                    ptr::null(),
                    global.as_ptr(),
                    if local.is_empty() {
                        ptr::null()
                    } else {
                        local.as_ptr()
                    },
                    n,
                    wait_ptr,
                )
            },
        };
        ev.map_err(|status| Error::OpenCl(opencl3::error_codes::ClError(status)))
    }
}

// ── RecordContext: the seam leaf `record` bodies emit into ──────────────────

/// The seam a [`RecordableOp::record`] body emits commands into. Layer 1 backs
/// it with a software command list + a map from graph-edge identity
/// ([`cell_id`](crate::eager::Pipe)) to the producer's recorded output. Layer 2
/// will add a `cl_khr_command_buffer` backend behind the same typed helpers, so
/// leaf bodies stay backend-agnostic.
pub struct RecordContext {
    commands: Vec<SoftCommand>,
    /// Producer outputs keyed by their output pipe's `cell_id`.
    edges: HashMap<usize, EdgeValue>,
}

impl RecordContext {
    fn new() -> Self {
        RecordContext {
            commands: Vec::new(),
            edges: HashMap::new(),
        }
    }

    /// Resolve a leaf input: either its own concrete buffer (`concrete`), or the
    /// producer output registered under the upstream pipe `cell_id`. Returns the
    /// handle + the sync points to wait on.
    pub fn resolve_input(
        &self,
        concrete: Option<BufHandle>,
        upstream_cell: Option<usize>,
    ) -> Result<(BufHandle, SyncPoints)> {
        if let Some(h) = concrete {
            return Ok((h, SyncPoints::new()));
        }
        let cell = upstream_cell.ok_or(Error::NotSupported(
            "record: op input is neither concrete nor pipe-fed",
        ))?;
        let e = self.edges.get(&cell).ok_or(Error::NotSupported(
            "record: upstream producer was not recorded before its consumer \
             — internal ordering bug",
        ))?;
        Ok((e.handle, e.waits.clone()))
    }

    /// Register a producer output under its output pipe's `cell_id`.
    pub fn register_output(&mut self, cell: usize, handle: BufHandle, waits: SyncPoints) {
        self.edges.insert(cell, EdgeValue { handle, waits });
    }

    /// Record a buffer fill. Returns its [`SyncPoint`].
    pub fn fill_buffer(
        &mut self,
        buffer: cl_mem,
        pattern: Vec<u8>,
        offset: usize,
        size: usize,
        waits: SyncPoints,
    ) -> SyncPoint {
        let idx = self.commands.len();
        self.commands.push(SoftCommand::Fill {
            buffer,
            pattern,
            offset,
            size,
            waits,
        });
        idx
    }

    /// Record a device-to-device buffer copy. Returns its [`SyncPoint`].
    pub fn copy_buffer(
        &mut self,
        src: cl_mem,
        dst: cl_mem,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        waits: SyncPoints,
    ) -> SyncPoint {
        let idx = self.commands.len();
        self.commands.push(SoftCommand::Copy {
            src,
            dst,
            src_offset,
            dst_offset,
            size,
            waits,
        });
        idx
    }

    /// Record an ND-range kernel launch. The kernel is retained for the
    /// recording's lifetime and released on drop. Returns its [`SyncPoint`].
    ///
    /// # Safety
    /// `kernel` must be a valid `cl_kernel` whose arguments are already set to
    /// the buffers/scalars this launch uses (the caller sets them at record
    /// time); it must remain valid for the recording (the retain ensures this).
    pub unsafe fn ndrange_kernel(
        &mut self,
        kernel: cl_kernel,
        global: Vec<usize>,
        local: Vec<usize>,
        waits: SyncPoints,
    ) -> Result<SyncPoint> {
        // Own a refcount for the recording's lifetime.
        unsafe { cl3::kernel::retain_kernel(kernel) }
            .map_err(|s| Error::OpenCl(opencl3::error_codes::ClError(s)))?;
        let idx = self.commands.len();
        self.commands.push(SoftCommand::NdRange {
            kernel,
            global,
            local,
            waits,
        });
        Ok(idx)
    }
}

// ── RecordableOp: the recordable sub-trait ──────────────────────────────────

/// A [`DeviceOp`] whose work is device-side commands with no host seam, so it
/// can be recorded into a [`RecordContext`] and replayed.
///
/// `record` is the **non-consuming** twin of [`execute`](DeviceOp::execute): it
/// walks `&self`, resolves its inputs from `ctx` (concrete buffer or upstream
/// pipe edge), emits its device commands, and registers its output handle(s)
/// under its output pipe(s) — threading [`SyncPoints`] where `execute` threads
/// `Deps`.
///
/// Recordability propagates structurally: a combinator implements `RecordableOp`
/// only when its children do (`AndThen<S, U>` iff both `S` and `U` are). Leaves
/// that touch the host don't implement it, so such chains are rejected at
/// compile time.
pub trait RecordableOp: DeviceOp {
    /// Record this op's device commands into `ctx`, resolving its inputs from
    /// `ctx`'s edge map and registering its outputs there.
    fn record(&self, ctx: &mut RecordContext) -> Result<()>;
}

// ── RecordedGraph: the materialized, replayable form ────────────────────────

/// A recorded eager graph, ready to [`replay`](RecordedGraph::replay) as many
/// times as wanted. Borrows the source graph for `'g` so the buffers (and
/// kernels) its commands reference stay live across every replay.
pub struct RecordedGraph<'g> {
    commands: Vec<SoftCommand>,
    _borrow: PhantomData<&'g ()>,
}

impl Drop for RecordedGraph<'_> {
    fn drop(&mut self) {
        // Release the kernel refcounts retained at record time.
        for cmd in &self.commands {
            if let SoftCommand::NdRange { kernel, .. } = cmd {
                // SAFETY: each was `retain_kernel`'d in `ndrange_kernel`; balance
                // it. Best-effort on drop.
                let _ = unsafe { cl3::kernel::release_kernel(*kernel) };
            }
        }
    }
}

impl RecordedGraph<'_> {
    /// Replay the recording on `launcher`'s queue, blocking until every recorded
    /// command completes. Reusable — call repeatedly (serially; the buffers are
    /// shared across replays).
    pub fn replay<L: Launcher + ?Sized>(&self, launcher: &L) -> Result<()> {
        let queue = launcher.cl_queue().get();
        let mut events: Vec<cl_event> = Vec::with_capacity(self.commands.len());
        for cmd in &self.commands {
            let wait_events: Vec<cl_event> = cmd.waits().iter().map(|&i| events[i]).collect();
            // SAFETY: handles captured from the borrowed graph (live for `'g`);
            // `wait_events` are this replay's live events.
            let ev = unsafe { cmd.enqueue(queue, &wait_events)? };
            events.push(ev);
        }
        let mut first_err: Option<Error> = None;
        for ev in events {
            let e = crate::Event::new(ev);
            if let Err(err) = e.wait()
                && first_err.is_none()
            {
                first_err = Some(Error::OpenCl(err));
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

// ── RecordExt: the `.record()` terminal ─────────────────────────────────────

/// `.record()` on any recordable [`DeviceOp`] chain.
pub trait RecordExt: RecordableOp {
    /// Record this graph into a reusable [`RecordedGraph`] without running it.
    /// The returned graph borrows `self`, so the buffers/kernels its leaves
    /// reference stay live for every replay.
    fn record(&self) -> Result<RecordedGraph<'_>> {
        let mut ctx = RecordContext::new();
        RecordableOp::record(self, &mut ctx)?;
        Ok(RecordedGraph {
            commands: ctx.commands,
            _borrow: PhantomData,
        })
    }
}

impl<O: RecordableOp> RecordExt for O {}

// Re-export the leaf/combinator record impls (kept in eager.rs / the macro,
// next to the `execute` bodies they mirror).
