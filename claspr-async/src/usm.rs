//! [`usm_slice`] — Tier 2 lazy wrapper for [`USMSlice<T>`].
//!
//! Construction is pure host code (no enqueue), so `execute` just
//! wraps the host Vec via [`USMSlice::new`] and passes `deps`
//! through unchanged.

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation};
use claspr::{Result, USMSlice};
use std::marker::PhantomData;

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

/// Lazy [`USMSlice<T>`] alloc, symmetric with
/// [`device_slice_alloc`](crate::device_slice_alloc) /
/// [`mapped_slice_alloc`](crate::mapped_slice_alloc). Allocates a
/// host `Vec<T>` of `len` elements initialised to `T::default()` and
/// wraps it via [`USMSlice::alloc`].
///
/// Same shape as the existing [`usm_slice`] op, but skips the
/// per-call-site `vec![v; N]` boilerplate when the caller doesn't
/// care about the initial pattern (kernel will overwrite). No perf
/// win over the explicit form — the Vec still needs to be initialised
/// before construction since USMSlice derefs to `&[T]`.
pub struct UsmSliceAlloc<T> {
    len: usize,
    _phantom: PhantomData<fn() -> T>,
}

/// Build a [`UsmSliceAlloc`] op for the chain. Errors at execute time
/// with [`Error::NotSupported`](claspr::Error::NotSupported) when the
/// running device doesn't advertise fine-grain system SVM (same gate
/// as [`usm_slice`]).
pub fn usm_slice_alloc<T>(len: usize) -> UsmSliceAlloc<T>
where
    T: Default + Copy + Send + 'static,
{
    UsmSliceAlloc {
        len,
        _phantom: PhantomData,
    }
}

impl<T> DeviceOperation for UsmSliceAlloc<T>
where
    T: Default + Copy + Send + 'static,
{
    type Output = USMSlice<T>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(USMSlice<T>, Deps)> {
        let slice = USMSlice::<T>::alloc(ec.context(), self.len)?;
        Ok((slice, deps))
    }
}
