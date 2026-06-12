//! Sketches of combinators that **opt out of recordability by
//! design**: they implement `DeviceOperation` (so they still work in
//! eager mode) but **do not** implement `RecordableOp` (so any chain
//! containing them simply fails to compile at the `.record()` call
//! site — propagation does the work).
//!
//! Two motivating cases from real claspr:
//!
//! ### `OnDevice` — routes an op to a different device's queue
//!
//! Recording into a CB pins the CB to a queue list at create time.
//! Mid-chain queue switching would force `cl_khr_command_buffer_multi_device`,
//! which only pocl supports natively (not in the cmdbufemu layer).
//! For v1 we error out cleanly: chains containing `OnDevice` simply
//! aren't recordable, full stop. The eager fallback handles them
//! perfectly today.
//!
//! ### `AndThenHost` — closure runs on host between device stages
//!
//! The closure consumes the source's output as a fully-evaluated host
//! value (e.g. summing a downloaded `Vec<u32>` to decide the next
//! launch's args). At record time, there IS no host value — the
//! download hasn't happened yet. So the closure is meaningless in
//! record mode and the combinator opts out structurally.

use crate::device_op::{Deps, DeviceOperation, ExecutionContext, Result};

// ─── OnDevice — fake routing to a different queue ──────────────────

pub struct OnDevice<S> {
    pub source: S,
    pub device_id: u32,
}

impl<S: DeviceOperation> DeviceOperation for OnDevice<S> {
    type Output = S::Output;

    fn execute(self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        // Real impl: build an ExecutionContext routed to device_id's
        // queue. Spike just forwards.
        self.source.execute(ctx, deps)
    }
}

// Deliberately NOT impl RecordableOp for OnDevice — see file
// docstring for rationale.

// ─── AndThenHost — closure runs on host between device stages ──────

pub struct AndThenHost<S, F> {
    pub source: S,
    pub f: F,
}

impl<S, F, U> DeviceOperation for AndThenHost<S, F>
where
    S: DeviceOperation,
    F: FnOnce(S::Output) -> U,
    U: DeviceOperation,
{
    type Output = U::Output;

    fn execute(self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(U::Output, Deps)> {
        // Real impl: drain source events on the host side before
        // invoking the closure (per the existing `and_then_host`
        // contract). Spike just chains the executes.
        let (source_out, source_deps) = self.source.execute(ctx, deps)?;
        let next = (self.f)(source_out);
        next.execute(ctx, source_deps)
    }
}

// Deliberately NOT impl RecordableOp for AndThenHost.

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combinators::AndThen;
    use crate::device_op::{FakeCommandBuffer, RecordContext, RecordableOp};
    use crate::leaves::{BufferFill, FillKernel};
    use std::sync::atomic::AtomicUsize;

    /// OnDevice executes fine in eager mode — same as wrapping the
    /// source op directly (no recording, no record_method on the
    /// chain root).
    #[test]
    fn on_device_executes_eagerly() {
        let chain = OnDevice {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 100,
            },
            device_id: 1,
        };
        let counter = AtomicUsize::new(0);
        let ec = ExecutionContext {
            enqueue_counter: &counter,
        };
        let (_out, _deps) = chain.execute(&ec, vec![]).expect("execute");
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    /// Wrapping a chain in OnDevice means the whole chain can't be
    /// recorded. The trait propagation does the work: trying to call
    /// `.record()` on this is a compile error (verified manually;
    /// `compile_fail/` directory has the .stderr capture).
    #[test]
    fn on_device_chain_works_in_eager_mode() {
        let chain = OnDevice {
            source: AndThen {
                source: BufferFill {
                    buf_id: 1,
                    pattern: 0,
                    len: 100,
                },
                f: |b| FillKernel {
                    n: 100,
                    buf_id: b,
                    value: 99,
                },
            },
            device_id: 2,
        };
        let counter = AtomicUsize::new(0);
        let ec = ExecutionContext {
            enqueue_counter: &counter,
        };
        let _ = chain.execute(&ec, vec![]).expect("execute");
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    /// AndThenHost similarly: works fine eagerly, can't be recorded.
    #[test]
    fn and_then_host_executes_eagerly() {
        let chain = AndThenHost {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 100,
            },
            f: |buf| {
                // In real life, this closure would do host work — sum
                // a downloaded Vec, branch on a value, etc. Here we
                // just produce the next op based on a fake decision.
                println!("(host computation on buf={buf})");
                FillKernel {
                    n: 100,
                    buf_id: buf,
                    value: 42,
                }
            },
        };
        let counter = AtomicUsize::new(0);
        let ec = ExecutionContext {
            enqueue_counter: &counter,
        };
        let _ = chain.execute(&ec, vec![]).expect("execute");
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    /// Sanity check that the type system actually catches the
    /// non-recordable case: this is a fn that consumes a
    /// `RecordableOp`, and trying to pass a chain containing an
    /// OnDevice would refuse to compile.
    fn assert_recordable<R: RecordableOp>(_r: R) {}

    #[test]
    fn type_assertion_works_for_recordable_chains() {
        // Plain AndThen chain is recordable — this compiles.
        let recordable = AndThen {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 100,
            },
            f: |b| FillKernel {
                n: 100,
                buf_id: b,
                value: 99,
            },
        };
        assert_recordable(recordable);

        // ── Three negative cases verified by hand (see
        //    compile_fail_cases.txt at the spike root for the
        //    captured rustc diagnostics). Each produces a clean
        //    E0277 with the offending type and a list of the
        //    types that DO implement RecordableOp ──
        //
        // 1. Chain containing an Upload (which doesn't impl
        //    RecordableOp):
        //
        //    let with_upload = AndThen {
        //        source: crate::leaves::Upload { data: vec![1, 2, 3], allocated_buf_id: 1 },
        //        f: |b| FillKernel { n: 3, buf_id: b, value: 99 },
        //    };
        //    assert_recordable(with_upload);  // E0277
        //
        // 2. OnDevice wrapping a recordable leaf:
        //
        //    let on_dev = OnDevice {
        //        source: BufferFill { buf_id: 1, pattern: 0, len: 100 },
        //        device_id: 1,
        //    };
        //    assert_recordable(on_dev);  // E0277
        //
        // 3. AndThenHost (host closure between device stages):
        //
        //    let with_host = AndThenHost {
        //        source: BufferFill { buf_id: 1, pattern: 0, len: 100 },
        //        f: |b| FillKernel { n: 100, buf_id: b, value: 99 },
        //    };
        //    assert_recordable(with_host);  // E0277
        //
        // ── Same with AndThenHost ──
        //
        // let with_host = AndThenHost {
        //     source: BufferFill { buf_id: 1, pattern: 0, len: 100 },
        //     f: |b| FillKernel { n: 100, buf_id: b, value: 99 },
        // };
        // assert_recordable(with_host);

        // Use the local-only side-effect-free recording to silence
        // unused-helper-fn warnings:
        let mut cb = FakeCommandBuffer::default();
        let mut rec = RecordContext {
            command_buffer: &mut cb,
        };
        let chain = AndThen {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 100,
            },
            f: |b| FillKernel {
                n: 100,
                buf_id: b,
                value: 99,
            },
        };
        let _ = chain.record(&mut rec, vec![]).expect("record");
    }
}
