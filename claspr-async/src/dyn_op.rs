//! [`DynOp<T>`] — type-erased [`DeviceOperation`] for conditional
//! graphs and any case where different branches produce different
//! concrete op types but share an `Output`.
//!
//! Spike scenario 9 (conditional graph) and cuda-oxide's equivalent
//! `DynOp` pattern. One `Box` per erased op — tolerable cost for the
//! one case where static typing genuinely doesn't work.
//!
//! ## Example
//!
//! ```ignore
//! use claspr_async::{DeviceOperation, DynOp, value};
//!
//! let chain = if some_condition {
//!     DynOp::new(value(42u32))
//! } else {
//!     // Different concrete type — would otherwise be an `if` arm
//!     // type-mismatch error.
//!     DynOp::new(value(1u32).and_then(|n| value(n + 1)))
//! };
//! let result = chain.sync(&ctx)?;
//! ```

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation};
use claspr::Result;

// ── Object-safe inner trait ─────────────────────────────────────────

/// Object-safe version of [`DeviceOperation`]. Differs from the
/// associated-type form only in taking `self: Box<Self>` so the trait
/// can be dyn-dispatched. Crate-internal — users see [`DynOp<T>`].
trait BoxedDeviceOp<T>: Send {
    fn execute_boxed(self: Box<Self>, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(T, Deps)>;
}

impl<O> BoxedDeviceOp<O::Output> for O
where
    O: DeviceOperation,
{
    fn execute_boxed(
        self: Box<Self>,
        ctx: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(O::Output, Deps)> {
        (*self).execute(ctx, deps)
    }
}

// ── DynOp ───────────────────────────────────────────────────────────

/// Type-erased [`DeviceOperation`] yielding `T`. Lets `if` /
/// `match` arms produce different concrete op types as long as
/// they agree on `Output`.
///
/// Construct with [`DynOp::new`]; treat as any other DeviceOperation
/// after that — composes with `.and_then`, `bundle!`, `fan_out`, etc.
///
/// One heap allocation per erased op.
///
/// The lifetime parameter `'op` lets the boxed op borrow from outer
/// scope (typically `&Kernels` for kernel launches inside the chain).
/// `DynOp<'static, T>` is the simplest case and infers naturally for
/// chains built from only owned data. For chains that reference
/// borrowed `Kernels`, the lifetime ties the `DynOp` back to that
/// borrow.
pub struct DynOp<'op, T> {
    inner: Box<dyn BoxedDeviceOp<T> + 'op>,
}

impl<'op, T: Send> DynOp<'op, T> {
    /// Box a concrete op into a type-erased one. Both arms of an
    /// `if`/`match` can produce `DynOp::new(...)` of the same `T`
    /// without their concrete types having to match.
    pub fn new<O>(op: O) -> Self
    where
        O: DeviceOperation<Output = T> + 'op,
    {
        DynOp {
            inner: Box::new(op),
        }
    }
}

impl<'op, T: Send> DeviceOperation for DynOp<'op, T> {
    type Output = T;

    fn execute(self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(T, Deps)> {
        self.inner.execute_boxed(ctx, deps)
    }
}
