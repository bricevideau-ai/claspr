//! The `Graph<I, O>` type and its combinators.
//!
//! Shape choice: **single struct, generic over (I, O)** (not a trait
//! with per-combinator impl types). Rationale:
//! - Library authors can name the type in `pub fn` signatures —
//!   `pub fn gemm(...) -> Graph<(SliceA, SliceB, SliceC), (SliceC,)>`.
//! - Composition (`and_then`, future `bundle`) doesn't explode the
//!   type name — every combinator returns `Graph<I, O>` with the
//!   appropriate types parameters, not `AndThen<Bundle<..>, ..>`.
//! - The actual DAG / IR is type-erased inside `GraphInner`, behind
//!   `Arc<GraphInner>`. Real implementation will store the assembled
//!   ops + the cached CB slot here.
//!
//! Trade-off: the type-erased DAG means runtime work to dispatch args
//! to the right node — but that work is dominated by the cost of
//! enqueuing kernels in the first place, so it's not on a hot path.

// Spike: many fields exist as documentation of the future impl shape
// but aren't read by the type-plumbing tests. Allow at the file level
// rather than scattering attrs through structs.
#![allow(dead_code)]

use std::marker::PhantomData;
use std::sync::Arc;

// ─── Graph ─────────────────────────────────────────────────────────

/// A typed compute graph. `I` is the tuple of input types; `O` is the
/// tuple of output types. Both must be tuples (single-output graphs
/// use `(T,)`).
///
/// `PhantomData<fn(I) -> O>` carries the type parameters without
/// affecting variance or auto-traits in surprising ways (contravariant
/// in `I`, covariant in `O`, always `Send + Sync` when the inner is).
pub struct Graph<I, O> {
    inner: Arc<GraphInner>,
    _phantom: PhantomData<fn(I) -> O>,
}

/// `Clone` is by-Arc — graphs are cheaply re-usable as sub-graphs.
/// Required for `and_then`-reuse patterns (`G.and_then(G.clone())`).
impl<I, O> Clone for Graph<I, O> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            _phantom: PhantomData,
        }
    }
}

/// Type-erased graph IR. In the real implementation this would carry
/// the DAG of `DeviceOperation` nodes, the cached CB slot
/// (`Mutex<Option<CachedCB>>`), the recordability bit, etc. Here it's
/// just a debug string + a chain of child graphs for `and_then`.
struct GraphInner {
    description: String,
    children: Vec<Arc<GraphInner>>,
}

impl<I, O> Graph<I, O> {
    /// Build a leaf graph from an op name. Real impl would take a
    /// `DeviceOperation` (or equivalent IR node).
    pub fn leaf(name: &str) -> Self {
        Self {
            inner: Arc::new(GraphInner {
                description: name.into(),
                children: vec![],
            }),
            _phantom: PhantomData,
        }
    }

    /// Compose: `self` produces `Mid`, `next` consumes `Mid` and
    /// produces `O2`. Result has inputs `I` (this graph's inputs) and
    /// outputs `O2` (the next graph's outputs).
    ///
    /// The type bound `Graph<O, O2>` on `next` is what enforces type-
    /// safe composition — passing a graph whose inputs don't match
    /// `self`'s outputs is a compile error.
    pub fn and_then<O2>(self, next: Graph<O, O2>) -> Graph<I, O2> {
        Graph {
            inner: Arc::new(GraphInner {
                description: format!("{} → {}", self.inner.description, next.inner.description),
                children: vec![self.inner, next.inner],
            }),
            _phantom: PhantomData,
        }
    }

    /// Internal dispatch — the per-arity `.call(a, b, c)` shims
    /// (macro-emitted below) bundle their args into the tuple `I` and
    /// hand off here. Real impl would either (a) walk the DAG and
    /// enqueue if no CB, or (b) replay the cached CB if available.
    fn invoke(&self, _args: I) -> Op<O> {
        Op {
            description: self.inner.description.clone(),
            _phantom: PhantomData,
        }
    }
}

// ─── Op ────────────────────────────────────────────────────────────

/// Placeholder for the future async/await-friendly op handle. Real
/// impl returns a `claspr::Op` (or whatever the unified Tier 2 op
/// type ends up being).
pub struct Op<O> {
    description: String,
    _phantom: PhantomData<fn() -> O>,
}

impl<O: Default> Op<O> {
    /// Spike-only: stand-in for `.wait()` / `.await` — fakes a
    /// completed op by producing default outputs. Lets us write
    /// `let (out,) = op.into_outputs()` in tests.
    pub fn into_outputs(self) -> O {
        let _ = self.description;
        O::default()
    }
}

// ─── Per-arity .call(...) inherent impls ──────────────────────────
//
// Macro emits one impl block per arity. Each impl exists only on
// `Graph<TupleN, O>`, so calling with the wrong arity is a "no method
// found" compile error on the wrong concrete type.

macro_rules! impl_call_arity {
    ($($arg:ident: $ty:ident),+) => {
        impl<$($ty),+, O> Graph<($($ty,)+), O> {
            #[allow(clippy::too_many_arguments)]
            pub fn call(&self, $($arg: $ty),+) -> Op<O> {
                self.invoke(($($arg,)+))
            }
        }
    };
}

impl_call_arity!(a: A);
impl_call_arity!(a: A, b: B);
impl_call_arity!(a: A, b: B, c: C);
impl_call_arity!(a: A, b: B, c: C, d: D);
impl_call_arity!(a: A, b: B, c: C, d: D, e: E);
impl_call_arity!(a: A, b: B, c: C, d: D, e: E, f: F);
impl_call_arity!(a: A, b: B, c: C, d: D, e: E, f: F, g: G);
impl_call_arity!(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H);
