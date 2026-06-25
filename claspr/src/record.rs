//! Record-and-replay for eager device graphs (layer 1: software backend).
//!
//! A recordable [`DeviceOp`] chain can be recorded
//! once into a reusable command list and **replayed** many times — the
//! foundation for reusable pipelines. This is *staging* infrastructure: the
//! surface (`record` / `replay`) is provisional and may change as the design
//! settles.
//!
//! # Why the eager engine makes this clean
//!
//! The eager `AndThen{source, next}` stores its **built** children (not a
//! closure), so the graph is structurally inspectable without running it.
//! Recording therefore walks the graph by **`&self`** and threads a
//! [`BufHandle`] (a raw `cl_mem` + byte length) producer→consumer, exactly where
//! `execute` threads the owned `DeviceSlice` value through its [`Pipe`]s. Because
//! recording reads handles by reference and never moves the buffers, it needs no
//! keep-alive: the recorded [`RecordedGraph`] borrows the source graph (and thus
//! its concrete buffers) for `'g`, so the buffers are guaranteed live across
//! every replay.
//!
//! # Recordability is a compile-time property
//!
//! [`RecordableOp`] is a sub-trait of `DeviceOp`. Device-side leaves (fill, and
//! later copy/kernel) implement it; host-touching leaves (`Upload`, `Download`,
//! `AndThenHost`, `OnDevice`) deliberately do **not**, and a combinator is
//! `RecordableOp` only when its children are (`AndThen<S, U>` iff `S` and `U`
//! are). So a chain containing a non-recordable leaf fails to compile at
//! [`record`](RecordExt::record), naming the offending leaf through the wrappers.
//!
//! # Scope of layer 1
//!
//! - Software backend only: a `Vec` of commands with **structural** dependency
//!   edges (indices), replayed by re-issuing on a queue with **fresh** events
//!   each call. No `cl_khr_command_buffer` yet (that is layer 2, behind the same
//!   [`RecordContext`] seam — leaf bodies won't change).
//! - Single-buffer outputs (`BufHandle`). Multi-output ops (bundles, multi-arg
//!   kernels) come later.
//! - Replay re-runs against the **same** buffers. Rebinding inputs (replay
//!   against different buffers) is a later layer (slots + mutable dispatch).
//!
//! [`Pipe`]: crate::eager::Pipe

use crate::eager::{AndThen, DeviceOp, Fill};
use crate::error::{Error, Result};
use crate::queue::Launcher;
use opencl3::memory::ClMem;
use opencl3::types::{cl_command_queue, cl_event, cl_mem, cl_uint};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

// ── Handles + sync points: the recording twins of (value, Deps) ─────────────

/// The recording twin of a buffer-typed [`Output`](DeviceOp::Output): a raw
/// `cl_mem` plus its byte length, threaded producer→consumer through the record
/// walk where `execute` threads the owned `DeviceSlice`. `Copy` (it is just a
/// handle) — the buffer it refers to is owned elsewhere (the source graph) and
/// must outlive the recording.
#[derive(Clone, Copy)]
pub struct BufHandle {
    pub mem: cl_mem,
    pub byte_len: usize,
}

// SAFETY: `BufHandle` is a raw `cl_mem` handle + a length. `cl_mem` is an opaque
// pointer into the OpenCL runtime, which is internally synchronized; the handle
// is only ever used to issue commands on a single queue during replay. The
// owning `DeviceSlice` (which bounds the buffer's lifetime) is what carries the
// real Send/Sync story; this is a borrowed view of it.
unsafe impl Send for BufHandle {}
unsafe impl Sync for BufHandle {}

/// A structural reference to an earlier command in the recording — its index in
/// the command list. The recording twin of a `cl_event` dependency.
pub type SyncPoint = usize;

/// The wait-list a recorded op threads to its consumers — the recording twin of
/// [`Deps`](crate::eager::Deps).
pub type SyncPoints = Vec<SyncPoint>;

// ── Software command list (the layer-1 recording target) ────────────────────

/// One recorded command — the software twin of a `clCommand*KHR` entry. Holds
/// the baked args by raw handle plus the **structural** dependency edges
/// (`waits`: indices of earlier commands), NOT `cl_event`s.
enum SoftCommand {
    Fill {
        buffer: cl_mem,
        /// The fill pattern, owned so it outlives the trace and every replay.
        pattern: Vec<u8>,
        offset: usize,
        size: usize,
        waits: Vec<SyncPoint>,
    },
}

impl SoftCommand {
    fn waits(&self) -> &[SyncPoint] {
        match self {
            SoftCommand::Fill { waits, .. } => waits,
        }
    }

    /// Enqueue this command on `queue` with `wait` as the event wait-list,
    /// returning its completion event.
    ///
    /// # Safety
    /// The baked `cl_mem` handles must be valid on `queue`'s context (the source
    /// graph's buffers, kept live by the `RecordedGraph` borrow) and `wait` must
    /// be live events.
    unsafe fn enqueue(&self, queue: cl_command_queue, wait: &[cl_event]) -> Result<cl_event> {
        use cl3::command_queue as q;
        let wait_ptr = if wait.is_empty() {
            ptr::null()
        } else {
            wait.as_ptr()
        };
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
                    wait.len() as cl_uint,
                    wait_ptr,
                )
            },
        };
        ev.map_err(|status| Error::OpenCl(opencl3::error_codes::ClError(status)))
    }
}

// ── RecordContext: the seam leaf `record` bodies emit into ──────────────────

/// The seam a [`RecordableOp::record`] body emits commands into. Layer 1 backs
/// it with a software command list; layer 2 will add a `cl_khr_command_buffer`
/// backend behind the same typed helpers, so leaf bodies stay backend-agnostic.
#[derive(Default)]
pub struct RecordContext {
    commands: Vec<SoftCommand>,
}

impl RecordContext {
    /// Record a buffer fill. Returns the new command's [`SyncPoint`] so the op
    /// can thread it to its consumers.
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
}

// ── RecordableOp: the recordable sub-trait ──────────────────────────────────

/// A [`DeviceOp`] whose work is device-side commands with no host seam, so it
/// can be recorded into a [`RecordContext`] and replayed.
///
/// `record` is the **non-consuming** twin of [`execute`](DeviceOp::execute): it
/// walks `&self`, emits each leaf's commands into `ctx`, threads [`SyncPoints`]
/// where `execute` threads `Deps`, and a [`BufHandle`] where `execute` threads
/// the owned `DeviceSlice`. `upstream` is the producer's output handle (`None`
/// at a chain head, whose buffer is its own concrete input).
pub trait RecordableOp: DeviceOp {
    /// Emit this op's device commands into `ctx`, threading `upstream` (the
    /// producer's buffer, or `None` at a chain head) and `waits` (the incoming
    /// dependency edges). Return this op's output handle + the sync points its
    /// commands produced.
    fn record(
        &self,
        ctx: &mut RecordContext,
        upstream: Option<BufHandle>,
        waits: SyncPoints,
    ) -> Result<(BufHandle, SyncPoints)>;
}

// ── RecordedGraph: the materialized, replayable form ────────────────────────

/// A recorded eager graph, ready to [`replay`](RecordedGraph::replay) as many
/// times as wanted. Borrows the source graph for `'g` so the buffers its
/// commands reference stay live across every replay.
pub struct RecordedGraph<'g> {
    commands: Vec<SoftCommand>,
    _borrow: PhantomData<&'g ()>,
}

impl RecordedGraph<'_> {
    /// Replay the recording on `launcher`'s queue, blocking until every recorded
    /// command completes. Reusable — call repeatedly (serially; the buffers are
    /// shared across replays).
    pub fn replay<L: Launcher + ?Sized>(&self, launcher: &L) -> Result<()> {
        let queue = launcher.cl_queue().get();
        // events[i] = the completion event produced by command i, THIS replay.
        // Fresh each call — never the trace's stale events.
        let mut events: Vec<cl_event> = Vec::with_capacity(self.commands.len());
        for cmd in &self.commands {
            let wait_events: Vec<cl_event> = cmd.waits().iter().map(|&i| events[i]).collect();
            // SAFETY: handles were captured from the borrowed graph's buffers
            // (live for `'g`); `wait_events` are this replay's live events.
            let ev = unsafe { cmd.enqueue(queue, &wait_events)? };
            events.push(ev);
        }
        // Wait on every command's event (covers the leaves), then release each.
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
    /// The returned graph borrows `self`, so the buffers its leaves reference
    /// (the graph's concrete inputs) stay live for every replay.
    fn record(&self) -> Result<RecordedGraph<'_>> {
        let mut ctx = RecordContext::default();
        RecordableOp::record(self, &mut ctx, None, SyncPoints::new())?;
        Ok(RecordedGraph {
            commands: ctx.commands,
            _borrow: PhantomData,
        })
    }
}

impl<O: RecordableOp> RecordExt for O {}

// ── Leaf + combinator impls ─────────────────────────────────────────────────

impl<T, M> RecordableOp for Fill<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: crate::access::MemMode + crate::Fillable + Send + 'static,
{
    fn record(
        &self,
        ctx: &mut RecordContext,
        upstream: Option<BufHandle>,
        waits: SyncPoints,
    ) -> Result<(BufHandle, SyncPoints)> {
        // The buffer is either this op's own concrete input (chain head) or the
        // handle the producer threaded in (mid-chain, in-place fill).
        let handle = match self.input_buffer() {
            Some(buf) => BufHandle {
                mem: buf.buffer().get(),
                byte_len: buf.byte_len(),
            },
            None => upstream.ok_or(Error::NotSupported(
                "record: fill has neither a concrete buffer nor an upstream handle",
            ))?,
        };
        // Byte pattern of the fill value.
        let value = self.fill_value();
        let pattern: Vec<u8> = {
            // SAFETY: `T: Copy`; we read `size_of::<T>()` bytes of its value.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    (&value as *const T) as *const u8,
                    std::mem::size_of::<T>(),
                )
            };
            bytes.to_vec()
        };
        let sp = ctx.fill_buffer(handle.mem, pattern, 0, handle.byte_len, waits);
        Ok((handle, vec![sp]))
    }
}

impl<S, U> RecordableOp for AndThen<S, U>
where
    S: RecordableOp,
    U: RecordableOp,
{
    fn record(
        &self,
        ctx: &mut RecordContext,
        upstream: Option<BufHandle>,
        waits: SyncPoints,
    ) -> Result<(BufHandle, SyncPoints)> {
        let (out, sp) = self.source_ref().record(ctx, upstream, waits)?;
        self.next_ref().record(ctx, Some(out), sp)
    }
}
