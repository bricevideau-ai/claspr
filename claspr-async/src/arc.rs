//! [`ArcSplit::split`] — fan out an `Arc<T>` to N independent leaf ops.
//!
//! The shared-input fan-out pattern: an upstream op produces some
//! immutable state (model weights, look-up table, etc.), wraps it in
//! `Arc` via [`.arc()`](crate::DeviceOperation::arc), and then N
//! downstream branches each want a clone of the Arc to use as their
//! starting value. `arc.split::<N>()` returns an `[Value<Arc<T>>; N]`
//! array of leaf ops, each carrying its own `Arc::clone` of the
//! original.
//!
//! ## Example — share inputs across a `Bundle`
//!
//! ```ignore
//! use claspr_async::{ArcSplit, DeviceOperation, bundle};
//!
//! let chain = upload_inputs()
//!     .arc()
//!     .and_then(|inputs| {
//!         let [a, b, c] = inputs.split::<3>();
//!         bundle!(
//!             a.and_then(|inp| kernel_a_op(inp)),
//!             b.and_then(|inp| kernel_b_op(inp)),
//!             c.and_then(|inp| kernel_c_op(inp)),
//!         )
//!     });
//! ```
//!
//! The trait is implemented for [`Arc<T>`] directly (not just for
//! `Arc`s coming from [`.arc()`](crate::DeviceOperation::arc)) — any
//! [`Arc<T>`] you produce inside an [`and_then`](crate::DeviceOperation::and_then)
//! closure can be split.

use crate::op::{Value, value};
use std::sync::Arc;

/// Fan-out helper on [`Arc<T>`]. See module docs.
///
/// `T: Send + Sync` is required: each split leaf carries an
/// `Arc<T>`, and `Arc<T>: Send` only when `T: Send + Sync`. The
/// leaves are `Value<Arc<T>>`, and `Value` itself requires `Send`.
pub trait ArcSplit<T: Send + Sync>: Sized {
    /// Produce `N` independent leaf ops, each holding an
    /// `Arc::clone` of the original.
    fn split<const N: usize>(self) -> [Value<Arc<T>>; N];
}

impl<T> ArcSplit<T> for Arc<T>
where
    T: Send + Sync,
{
    fn split<const N: usize>(self) -> [Value<Arc<T>>; N] {
        std::array::from_fn(|_| value(Arc::clone(&self)))
    }
}
