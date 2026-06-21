//! Tier 2 lazy wrappers for [`USMSlice<T, M>`].
//!
//! Construction is pure host code (no enqueue), so `execute` just
//! wraps the host Vec via [`USMSlice::new`] / [`USMSlice::alloc_zero`]
//! and passes `deps` through unchanged.

use crate::exec_ctx::ExecutionContext;
use crate::device_op::{Deps, DeviceOperation};
use crate::{MemMode, ReadWrite, Result, USMSlice, USMSliceUninit};
use std::marker::PhantomData;

/// Wrap a host `Vec<T>` as a [`USMSlice<T, M>`] at chain execute
/// time. Built by the [`usm_slice!`](crate::usm_slice!) macro.
///
/// **No marker bound** — USM is host memory, any marker works.
pub struct UsmSliceOp<T, M: MemMode = ReadWrite> {
    data: Option<Vec<T>>,
    _phantom: PhantomData<fn() -> M>,
}

impl<T, M> UsmSliceOp<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data: Some(data),
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for UsmSliceOp<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSlice<T, M>;

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(USMSlice<T, M>, Deps)> {
        let data = self
            .data
            .take()
            .expect("UsmSliceOp::execute called twice — internal claspr-async bug");
        let slice = USMSlice::new(ec.context(), data)?;
        Ok((slice, deps))
    }
}

/// Lazy [`USMSliceUninit<T, M>`] alloc symmetric with
/// [`DeviceSliceAllocUninit`](crate::DeviceSliceAllocUninit) /
/// [`MappedSliceAllocUninit`](crate::MappedSliceAllocUninit). Built
/// by the [`usm_slice_alloc_uninit!`](crate::usm_slice_alloc_uninit!)
/// macro.
///
/// Higher-level fill / write paths (`usm_slice_alloc_zero!`,
/// `usm_slice!(vec)`) are sugar on top of this op composed with
/// [`crate::FillUninit::fill`] / [`crate::WriteUninit::write`].
///
/// **No marker bound** — USM is host memory; allocation works
/// regardless of the kernel-side marker.
pub struct UsmSliceAllocUninit<T, M: MemMode = ReadWrite> {
    len: usize,
    _phantom: PhantomData<fn() -> (T, M)>,
}

impl<T, M> UsmSliceAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    pub fn new(len: usize) -> Self {
        Self {
            len,
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for UsmSliceAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSliceUninit<T, M>;

    fn execute(
        self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(USMSliceUninit<T, M>, Deps)> {
        let uninit = USMSlice::<T, M>::alloc_uninit(ec.context(), self.len)?;
        Ok((uninit, deps))
    }
}
