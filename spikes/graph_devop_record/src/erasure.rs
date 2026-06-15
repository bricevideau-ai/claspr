//! The erasure handoff: concrete `RecordableOp` chain → reusable,
//! type-erased recorder.
//!
//! ## Why this spike exists
//!
//! `graph_devop_record` (the other modules here) proved the
//! *compile-time* guarantee: a chain carries `RecordableOp` in its
//! concrete type, so a chain containing `Upload` (which is
//! `!RecordableOp`) fails to compile when you try to `.record()` it.
//!
//! But a *reusable cached graph* can't keep the concrete chain type
//! forever:
//! - `record(self)` / `execute(self)` consume the chain. Replaying,
//!   re-recording (queue change), or running the eager-fallback path
//!   more than once all need the chain *again*.
//! - Storing the graph in a struct field, returning it from a
//!   non-generic function, or holding heterogeneous graphs together
//!   wants the chain type erased.
//!
//! Erasing a chain to `Box<dyn DeviceOperation>` **loses the
//! `RecordableOp` bound** — you can't call `.record()` through that
//! trait object. So the static guarantee only protects the boundary
//! *up to* erasure. This module demonstrates the handoff that carries
//! recordability across that boundary, and the reuse model that makes
//! the erased thing callable more than once.
//!
//! ## Reuse model: factory, not Clone
//!
//! The chain is consumed per run, so reuse needs one of:
//! - **`Clone` the chain** — forces every closure to be `Fn + Clone`
//!   and all captured data `Clone`. Very restrictive; rules out the
//!   `FnOnce` closures `and_then` uses today.
//! - **Factory `Fn() -> Chain`** — re-invocable, rebuilds a fresh
//!   chain per run. No `Clone` bound on the chain. Rebuilding a
//!   combinator tree is cheap host work (no GPU calls until
//!   execute/record runs). This is what a reusable pipeline naturally
//!   *is*: a function that builds the chain.
//!
//! We pick the **factory**. It also matches the library-export shape
//! (`fn my_pipeline(args) -> impl RecordableOp` is one invocation of
//! such a factory).
//!
//! ## The handoff
//!
//! At construction — where `Chain: RecordableOp` is still statically
//! known — we capture two erased closures built from the factory:
//! - `execute_fn` (always): runs the eager path.
//! - `record_fn` (only when `Chain: RecordableOp`): records into a CB.
//!
//! The presence/absence of `record_fn` is exactly where the
//! compile-time `RecordableOp` bound becomes the runtime
//! "is this recordable?" bit. Two constructors enforce the boundary:
//! - `ErasedGraph::recordable(factory)` requires `C: RecordableOp` —
//!   a factory producing an `Upload`-containing chain won't compile
//!   here (the guarantee, preserved up to the handoff).
//! - `ErasedGraph::eager_only(factory)` requires only
//!   `C: DeviceOperation` — `record_fn = None`, execute-only.

use crate::device_op::{
    Deps, DeviceOperation, ExecutionContext, FakeCommandBuffer, RecordContext, RecordableOp,
    Result, SyncPoints,
};
use std::sync::{Arc, Mutex};

/// Erased recorder closure type: records a fresh chain into a CB.
type RecordFn<O> = Box<dyn Fn(&mut RecordContext<'_>, SyncPoints) -> Result<(O, SyncPoints)>>;
/// Erased execute closure type: runs a fresh chain eagerly.
type ExecuteFn<O> = Box<dyn Fn(&ExecutionContext<'_>, Deps) -> Result<(O, Deps)>>;

/// Spike-only counters.
#[derive(Default, Debug)]
pub struct Instrumentation {
    pub record_count: usize,
    pub replay_count: usize,
    pub eager_count: usize,
}

/// A type-erased, reusable graph. The concrete chain type is gone;
/// `O` (the output) is all that survives in the signature, so this
/// could be a struct field or a `fn(...) -> ErasedGraph<O>` return.
pub struct ErasedGraph<O> {
    execute_fn: ExecuteFn<O>,
    /// `Some` iff built from a `RecordableOp` chain. This is the
    /// compile-time bound, converted to a runtime value at the
    /// erasure boundary.
    record_fn: Option<RecordFn<O>>,
    /// Cached recorded CB (light — just demonstrates reuse-with-cache;
    /// the full call/mutate_call/simultaneous protocol lives in the
    /// design notes, not re-litigated here).
    cache: Mutex<Option<FakeCommandBuffer>>,
    pub instr: Arc<Mutex<Instrumentation>>,
}

impl<O: 'static> ErasedGraph<O> {
    /// Recordable constructor. `C: RecordableOp` — the boundary that
    /// rejects non-recordable chains (e.g. anything containing
    /// `Upload`) at compile time. Captures both closures from the
    /// factory.
    pub fn recordable<C, F>(factory: F) -> Self
    where
        C: RecordableOp<Output = O> + 'static,
        F: Fn() -> C + 'static,
    {
        let factory = Arc::new(factory);
        let fe = Arc::clone(&factory);
        let execute_fn: ExecuteFn<O> = Box::new(move |ec, deps| fe().execute(ec, deps));
        let fr = Arc::clone(&factory);
        let record_fn: RecordFn<O> = Box::new(move |rc, sps| fr().record(rc, sps));
        Self {
            execute_fn,
            record_fn: Some(record_fn),
            cache: Mutex::new(None),
            instr: Arc::new(Mutex::new(Instrumentation::default())),
        }
    }

    /// Eager-only constructor. Accepts any `DeviceOperation` chain,
    /// including non-recordable ones. `record_fn = None`. This is the
    /// degradation path: you *can* erase a non-recordable chain, you
    /// just don't get the recording capability.
    pub fn eager_only<C, F>(factory: F) -> Self
    where
        C: DeviceOperation<Output = O> + 'static,
        F: Fn() -> C + 'static,
    {
        let factory = Arc::new(factory);
        let fe = Arc::clone(&factory);
        let execute_fn: ExecuteFn<O> = Box::new(move |ec, deps| fe().execute(ec, deps));
        Self {
            execute_fn,
            record_fn: None,
            cache: Mutex::new(None),
            instr: Arc::new(Mutex::new(Instrumentation::default())),
        }
    }

    /// Did recordability survive erasure?
    pub fn is_recordable(&self) -> bool {
        self.record_fn.is_some()
    }

    /// Eager run — rebuilds + walks the chain. Always available.
    pub fn execute_eager(&self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(O, Deps)> {
        self.instr.lock().unwrap().eager_count += 1;
        (self.execute_fn)(ec, deps)
    }

    /// Record into a fresh CB — rebuilds + records the chain. Only
    /// available when recordable; errors otherwise (in real claspr
    /// this branch is unreachable because the caller gates on
    /// `is_recordable()` / the type system).
    pub fn record_fresh(&self) -> Result<FakeCommandBuffer> {
        let record_fn = self
            .record_fn
            .as_ref()
            .ok_or("graph is not recordable (built via eager_only)")?;
        let mut cb = FakeCommandBuffer::default();
        {
            let mut rc = RecordContext {
                command_buffer: &mut cb,
            };
            let (_out, _sps) = record_fn(&mut rc, vec![])?;
        }
        self.instr.lock().unwrap().record_count += 1;
        Ok(cb)
    }

    /// Light cache demo: first call records into the cache, later
    /// calls "replay" (here: observe the cached CB is reused, bump a
    /// counter). Demonstrates the erased recorder being reusable
    /// across calls — the whole point of the handoff.
    pub fn call_cached(&self) -> Result<()> {
        let mut cache = self.cache.lock().unwrap();
        if cache.is_none() {
            let cb = {
                // Drop the cache lock around record_fresh's own
                // instr lock to avoid ordering surprises — record
                // into a local first.
                drop(cache);
                let cb = self.record_fresh()?;
                cache = self.cache.lock().unwrap();
                cb
            };
            *cache = Some(cb);
        }
        // Cache hit (or just-populated): "replay".
        self.instr.lock().unwrap().replay_count += 1;
        Ok(())
    }

    /// Inspect the cached CB's recorded commands (test helper).
    pub fn cached_commands(&self) -> Option<Vec<String>> {
        self.cache
            .lock()
            .unwrap()
            .as_ref()
            .map(|cb| cb.recorded_commands.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combinators::AndThen;
    use crate::leaves::{BufferFill, FillKernel};
    use std::sync::atomic::AtomicUsize;

    /// A recordable chain erases into an `ErasedGraph` that has lost
    /// the concrete type but KEPT the ability to record — recordability
    /// survived the boundary via `record_fn = Some`.
    #[test]
    fn recordable_chain_keeps_record_after_erasure() {
        let g: ErasedGraph<u64> = ErasedGraph::recordable(|| AndThen {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 100,
            },
            f: |b| FillKernel {
                n: 100,
                buf_id: b,
                value: 7,
            },
        });
        assert!(g.is_recordable(), "record capability survived erasure");

        let cb = g.record_fresh().expect("record");
        assert_eq!(cb.recorded_commands.len(), 2); // fill + kernel
        assert!(cb.recorded_commands[0].starts_with("buffer_fill"));
        assert!(cb.recorded_commands[1].starts_with("fill_kernel"));
    }

    /// The erased graph is REUSABLE: the factory is re-invoked per
    /// run, so we can record multiple fresh CBs (and mix in eager
    /// runs) from the same erased value. This is what `Clone`-less
    /// reuse via a factory buys us.
    #[test]
    fn erased_graph_is_reusable_via_factory() {
        let g: ErasedGraph<u64> = ErasedGraph::recordable(|| AndThen {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 100,
            },
            f: |b| FillKernel {
                n: 100,
                buf_id: b,
                value: 7,
            },
        });

        // Record twice — two independent fresh CBs, factory rebuilt
        // each time.
        let cb1 = g.record_fresh().expect("record 1");
        let cb2 = g.record_fresh().expect("record 2");
        assert_eq!(cb1.recorded_commands.len(), 2);
        assert_eq!(cb2.recorded_commands.len(), 2);

        // And an eager run on the same erased value.
        let counter = AtomicUsize::new(0);
        let ec = ExecutionContext {
            enqueue_counter: &counter,
        };
        let _ = g.execute_eager(&ec, vec![]).expect("eager");
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);

        let instr = g.instr.lock().unwrap();
        assert_eq!(instr.record_count, 2);
        assert_eq!(instr.eager_count, 1);
    }

    /// A non-recordable chain CAN still be erased via `eager_only`,
    /// but the erased graph reports `is_recordable() == false` and
    /// has no record capability. This is the runtime-bit degradation
    /// path: erasure is allowed, recording is not.
    #[test]
    fn eager_only_erasure_drops_record_capability() {
        use crate::leaves::Upload;
        let g: ErasedGraph<u64> = ErasedGraph::eager_only(|| AndThen {
            source: Upload {
                data: vec![1, 2, 3],
                allocated_buf_id: 9,
            },
            f: |b| FillKernel {
                n: 3,
                buf_id: b,
                value: 1,
            },
        });
        assert!(!g.is_recordable(), "eager_only graph is not recordable");
        assert!(
            g.record_fresh().is_err(),
            "record_fresh errors on a non-recordable erased graph"
        );

        // But eager execution works fine.
        let counter = AtomicUsize::new(0);
        let ec = ExecutionContext {
            enqueue_counter: &counter,
        };
        let _ = g.execute_eager(&ec, vec![]).expect("eager");
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2); // upload + kernel
    }

    /// The cache reuses the erased recorder: first `call_cached`
    /// records, subsequent calls replay the cached CB. record_count
    /// stays 1 across N calls; replay_count tracks calls.
    #[test]
    fn cached_call_records_once_then_replays() {
        let g: ErasedGraph<u64> = ErasedGraph::recordable(|| AndThen {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 100,
            },
            f: |b| FillKernel {
                n: 100,
                buf_id: b,
                value: 7,
            },
        });

        g.call_cached().expect("call 1");
        g.call_cached().expect("call 2");
        g.call_cached().expect("call 3");

        let instr = g.instr.lock().unwrap();
        assert_eq!(instr.record_count, 1, "recorded once");
        assert_eq!(instr.replay_count, 3, "replayed on every call");
        drop(instr);

        let cmds = g.cached_commands().expect("cache populated");
        assert_eq!(cmds.len(), 2);
    }

    // ── Compile-time boundary (see compile_fail_cases.txt) ──
    //
    // `ErasedGraph::recordable(factory)` requires the factory's chain
    // to be `RecordableOp`. A factory producing an `Upload`-containing
    // chain fails to compile HERE — the static guarantee is preserved
    // up to the erasure boundary:
    //
    //     ErasedGraph::recordable(|| AndThen {
    //         source: Upload { data: vec![1, 2, 3], allocated_buf_id: 1 },
    //         f: |b| FillKernel { n: 3, buf_id: b, value: 1 },
    //     });
    //     // error[E0277]: AndThen<Upload, ...>: RecordableOp not satisfied
    //
    // (Such a chain must use `eager_only` instead — see the test
    // above — which is the explicit "no caching for this one" path.)
}
