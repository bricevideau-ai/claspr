//! [`usm_slice`] — Tier 2 lazy wrapper for [`USMSlice<T>`].
//!
//! Construction is pure host code (no enqueue), so `execute` just
//! wraps the host Vec via [`USMSlice::new`] and passes `deps`
//! through unchanged.

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation};
use claspr::{Result, USMSlice};

/// Combinator built by [`usm_slice`].
pub struct UsmSliceOp<T> {
    data: Option<Vec<T>>,
}

/// Wrap a host `Vec<T>` as a [`USMSlice<T>`] at chain execute time.
/// Errors at execute with `Error::NotSupported` if the chain's
/// device doesn't advertise fine-grain system SVM.
///
/// No enqueue: the USM slice IS the host Vec's memory, so there's
/// nothing to wait on for "construction." Downstream chain stages
/// receive the slice immediately and inherit the upstream `deps`
/// unchanged.
pub fn usm_slice<T>(data: Vec<T>) -> UsmSliceOp<T>
where
    T: Send + 'static,
{
    UsmSliceOp { data: Some(data) }
}

impl<T> DeviceOperation for UsmSliceOp<T>
where
    T: Send + 'static,
{
    type Output = USMSlice<T>;

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(USMSlice<T>, Deps)> {
        let data = self
            .data
            .take()
            .expect("UsmSliceOp::execute called twice — internal claspr-async bug");
        let slice = USMSlice::new(ec.context(), data)?;
        Ok((slice, deps))
    }
}
