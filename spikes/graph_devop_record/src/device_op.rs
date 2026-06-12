//! The extended `DeviceOperation` trait surface.
//!
//! ## Design choice: dual method + sub-trait
//!
//! Today (real claspr):
//! ```ignore
//! trait DeviceOperation: Send + Sized {
//!     type Output: Send;
//!     fn execute(self, ctx: &ExecutionContext, deps: Deps) -> Result<(Self::Output, Deps)>;
//! }
//! ```
//!
//! Proposed extension (this spike):
//! - **Base trait stays the same** — every op still has `.execute()`
//!   for the eager-enqueue path. No change to existing impls.
//! - **Add a sub-trait** `RecordableOp: DeviceOperation` with a
//!   `.record()` method. Recordable leaf ops (kernels, fills, D2D
//!   copies, SVM copies, migrate) implement it. Non-recordable ones
//!   (upload, download, map/unmap, image upload/download,
//!   and_then_host) simply don't implement it, and trying to record
//!   a chain containing them is a **compile error**.
//! - **Combinators** (`AndThen`, `Bundle*`, `FanOut`) implement
//!   `RecordableOp` *conditionally* on their children — `impl<S, F, U>
//!   RecordableOp for AndThen<S, F> where S: RecordableOp, F: ..., U:
//!   RecordableOp`. The recordability of a composite is the AND of
//!   its children, enforced at compile time by trait bounds.
//!
//! ## Why sub-trait over dual-method-on-base?
//!
//! - Non-recordable ops don't have to implement (or stub) a record
//!   method. Cleaner. (The dual-method variant would force every leaf
//!   op to either record-correctly or return `Err(NotRecordable)` —
//!   error at runtime instead of compile time.)
//! - Recordability propagates through the type system via trait
//!   bounds on combinators. The Graph wrapper only accepts chains
//!   whose root impls `RecordableOp`; a user accidentally including
//!   an `upload()` in their pipeline gets a compile error pointing
//!   at the upload call, not a runtime "this isn't recordable"
//!   surprise.
//!
//! ## What the SyncPoints / RecordContext types look like
//!
//! Mirrors the `Deps` / `ExecutionContext` pair on the eager side:
//! - `SyncPoints` = `Vec<SyncPointId>` (sync points are sequential
//!   integers internal to the CB; the real OpenCL type is
//!   `cl_sync_point_khr` which is just `cl_uint`).
//! - `RecordContext` carries the `CommandBuffer` being recorded into,
//!   plus any per-recording state (queue, properties).
//!
//! Both are spike stubs here — fake `CommandBuffer` is a struct that
//! just counts recorded commands for the test assertions.

pub type SyncPointId = u32;
pub type SyncPoints = Vec<SyncPointId>;
pub type Deps = Vec<u64>; // fake "event id" stand-in for the spike

/// Eager-mode execution context. Real claspr has the queue + ctx
/// here; spike just carries an enqueue counter for test assertions.
pub struct ExecutionContext<'a> {
    pub enqueue_counter: &'a std::sync::atomic::AtomicUsize,
}

/// Record-mode context. Real impl wraps `cl_command_buffer_khr`;
/// spike just counts recorded commands + dispenses sync point ids.
pub struct RecordContext<'a> {
    pub command_buffer: &'a mut FakeCommandBuffer,
}

#[derive(Default)]
pub struct FakeCommandBuffer {
    pub recorded_commands: Vec<String>, // for test assertion
    next_sync_point: SyncPointId,
}

impl FakeCommandBuffer {
    pub fn allocate_sync_point(&mut self) -> SyncPointId {
        let id = self.next_sync_point;
        self.next_sync_point += 1;
        id
    }

    pub fn record(&mut self, command: &str) -> SyncPointId {
        self.recorded_commands.push(command.into());
        self.allocate_sync_point()
    }
}

pub type Result<T> = std::result::Result<T, String>;

// ─── Base trait (mirrors today's DeviceOperation) ─────────────────

pub trait DeviceOperation: Sized {
    type Output;

    /// Eager-enqueue path. Exists for every op (recordable or not).
    fn execute(self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)>;
}

// ─── Sub-trait for recordable ops ─────────────────────────────────

pub trait RecordableOp: DeviceOperation {
    /// CB-recording path. Threads sync points the way `execute`
    /// threads `Deps`.
    fn record(
        self,
        ctx: &mut RecordContext<'_>,
        sync_points: SyncPoints,
    ) -> Result<(Self::Output, SyncPoints)>;
}
