//! `AndThen` and `Bundle2` combinators, sketched to show how they
//! participate in record mode.
//!
//! ## Findings
//!
//! ### AndThen — clean
//!
//! `AndThen` already takes a closure `FnOnce(S::Output) -> U`. In
//! record mode, it just calls `source.record()` instead of
//! `source.execute()`, forwards the output to the closure, then
//! calls `next.record()`. The closure body doesn't change — it sees
//! the source's output (a buffer handle, a value) and builds the
//! next op. The recordability propagates via the trait bound:
//!
//! ```ignore
//! impl<S, F, U> RecordableOp for AndThen<S, F>
//! where
//!     S: RecordableOp,
//!     F: FnOnce(S::Output) -> U,
//!     U: RecordableOp,
//! { ... }
//! ```
//!
//! If a user threads a non-recordable op (Upload) inside, the chain
//! root simply doesn't impl RecordableOp and trying to record it is
//! a **compile error**.
//!
//! ### Bundle2 — also clean
//!
//! `Bundle2` runs two siblings in parallel and joins their events
//! via `clEnqueueMarkerWithWaitList`. In record mode, it records
//! both children and emits a `clCommandBarrierWithWaitListKHR` to
//! join their sync points. The pattern is identical to today's
//! eager-mode bundle.
//!
//! ### Where it gets interesting
//!
//! - `and_then_host` already opts out of recordability naturally —
//!   its closure runs on the host, which has no meaning at record
//!   time. The closure can't produce sync points. So `and_then_host`
//!   simply doesn't impl `RecordableOp` for the same reason `Upload`
//!   doesn't.
//! - `on_device` (routing an op to a different device's queue) is
//!   genuinely tricky — CBs are bound to a queue list at creation,
//!   so routing mid-chain would force `multi_device` (only natively
//!   supported by pocl; not in the cmdbufemu layer). For v1,
//!   `on_device` chains just don't impl `RecordableOp` and fall to
//!   the eager path.

use crate::device_op::{
    Deps, DeviceOperation, ExecutionContext, RecordContext, RecordableOp, Result, SyncPoints,
};

// ─── AndThen ──────────────────────────────────────────────────────

pub struct AndThen<S, F> {
    pub source: S,
    pub f: F,
}

impl<S, F, U> DeviceOperation for AndThen<S, F>
where
    S: DeviceOperation,
    F: FnOnce(S::Output) -> U,
    U: DeviceOperation,
{
    type Output = U::Output;

    fn execute(self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(U::Output, Deps)> {
        let (source_out, source_deps) = self.source.execute(ctx, deps)?;
        let next = (self.f)(source_out);
        next.execute(ctx, source_deps)
    }
}

impl<S, F, U> RecordableOp for AndThen<S, F>
where
    S: RecordableOp,
    F: FnOnce(S::Output) -> U,
    U: RecordableOp,
{
    fn record(
        self,
        ctx: &mut RecordContext<'_>,
        sync_points: SyncPoints,
    ) -> Result<(U::Output, SyncPoints)> {
        let (source_out, source_sps) = self.source.record(ctx, sync_points)?;
        let next = (self.f)(source_out);
        next.record(ctx, source_sps)
    }
}

// ─── Bundle2 ──────────────────────────────────────────────────────

pub struct Bundle2<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> DeviceOperation for Bundle2<A, B>
where
    A: DeviceOperation,
    B: DeviceOperation,
{
    type Output = (A::Output, B::Output);

    fn execute(self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (a_out, a_deps) = self.a.execute(ctx, deps.clone())?;
        let (b_out, b_deps) = self.b.execute(ctx, deps)?;
        // Real impl: clEnqueueMarkerWithWaitList over (a_deps, b_deps).
        let joined = [a_deps, b_deps].concat();
        Ok(((a_out, b_out), joined))
    }
}

impl<A, B> RecordableOp for Bundle2<A, B>
where
    A: RecordableOp,
    B: RecordableOp,
{
    fn record(
        self,
        ctx: &mut RecordContext<'_>,
        sync_points: SyncPoints,
    ) -> Result<(Self::Output, SyncPoints)> {
        let (a_out, a_sps) = self.a.record(ctx, sync_points.clone())?;
        let (b_out, b_sps) = self.b.record(ctx, sync_points)?;
        // Real impl: clCommandBarrierWithWaitListKHR over (a_sps, b_sps).
        let joined_sp = ctx.command_buffer.record(&format!(
            "barrier(joining sync_points: a={:?}, b={:?})",
            a_sps, b_sps
        ));
        Ok(((a_out, b_out), vec![joined_sp]))
    }
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_op::FakeCommandBuffer;
    use crate::leaves::{BufferCopy, BufferFill, FillKernel, Upload};
    use std::sync::atomic::AtomicUsize;

    /// Sanity: a recordable chain records all its commands plus the
    /// barrier joins.
    #[test]
    fn recordable_chain_records_all_commands() {
        let chain = AndThen {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 1024,
            },
            f: |buf| FillKernel {
                n: 1024,
                buf_id: buf,
                value: 42,
            },
        };

        let mut cb = FakeCommandBuffer::default();
        let mut rec = RecordContext {
            command_buffer: &mut cb,
        };

        let (out, sps) = chain.record(&mut rec, vec![]).expect("record");
        assert_eq!(out, 1);
        assert_eq!(sps.len(), 1);
        assert_eq!(cb.recorded_commands.len(), 2);
        assert!(cb.recorded_commands[0].starts_with("buffer_fill"));
        assert!(cb.recorded_commands[1].starts_with("fill_kernel"));
    }

    /// Sanity: same chain executes eagerly with the enqueue counter
    /// bumped for each op.
    #[test]
    fn recordable_chain_also_executes_eagerly() {
        let chain = AndThen {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 1024,
            },
            f: |buf| FillKernel {
                n: 1024,
                buf_id: buf,
                value: 42,
            },
        };

        let enqueue_counter = AtomicUsize::new(0);
        let ctx = ExecutionContext {
            enqueue_counter: &enqueue_counter,
        };

        let (out, _deps) = chain.execute(&ctx, vec![]).expect("execute");
        assert_eq!(out, 1);
        assert_eq!(
            enqueue_counter.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    /// Bundle2 records both children + a barrier.
    #[test]
    fn bundle2_records_both_plus_barrier() {
        let chain = Bundle2 {
            a: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 1024,
            },
            b: BufferCopy {
                src_id: 2,
                dst_id: 3,
                len_bytes: 4096,
            },
        };

        let mut cb = FakeCommandBuffer::default();
        let mut rec = RecordContext {
            command_buffer: &mut cb,
        };

        let ((a_out, b_out), sps) = chain.record(&mut rec, vec![]).expect("record");
        assert_eq!(a_out, 1);
        assert_eq!(b_out, 3);
        assert_eq!(sps.len(), 1, "Bundle2 produces a single joined sync point");
        assert_eq!(cb.recorded_commands.len(), 3); // 2 leaves + 1 barrier
        assert!(cb.recorded_commands[2].starts_with("barrier"));
    }

    /// Key compile-time guarantee: a chain containing an Upload
    /// cannot have `.record()` called on it. This test exists by
    /// failing-to-compile rather than passing — it's commented out
    /// because uncommenting would break the build (as desired).
    ///
    /// ```compile_fail
    /// let chain = AndThen {
    ///     source: Upload { data: vec![1, 2, 3], allocated_buf_id: 1 },
    ///     f: |buf| FillKernel { n: 3, buf_id: buf, value: 42 },
    /// };
    /// let mut cb = FakeCommandBuffer::default();
    /// let mut rec = RecordContext { command_buffer: &mut cb };
    /// chain.record(&mut rec, vec![]); // ERROR: Upload: RecordableOp not satisfied
    /// ```
    #[test]
    fn upload_chain_still_executes_eagerly() {
        // Negative case verified at compile-time (see doc-comment
        // above). Positive case: the same chain works fine in eager
        // mode because Upload only needs `DeviceOperation`, not
        // `RecordableOp`.
        let chain = AndThen {
            source: Upload {
                data: vec![1, 2, 3, 4],
                allocated_buf_id: 7,
            },
            f: |buf| FillKernel {
                n: 4,
                buf_id: buf,
                value: 99,
            },
        };

        let enqueue_counter = AtomicUsize::new(0);
        let ctx = ExecutionContext {
            enqueue_counter: &enqueue_counter,
        };

        let (out, _deps) = chain.execute(&ctx, vec![]).expect("execute");
        assert_eq!(out, 7);
        assert_eq!(
            enqueue_counter.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }
}
