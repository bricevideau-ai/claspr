//! Lazy buffer alloc primitives — [`DeviceSliceAllocUninit`],
//! [`MappedSliceAllocUninit`], and the construction-time
//! [`DeviceSliceFromSlice`] / [`MappedSliceFromSlice`] (which bake
//! initial data in via `CL_MEM_COPY_HOST_PTR`).
//!
//! `*_alloc_uninit!` is the canonical alloc primitive. Higher-level
//! init paths (`alloc_zero!`, `filled!`, `upload!`) are macro sugar
//! that expands to `alloc_uninit + .fill()/.write()` via the
//! [`crate::FillUninit`] / [`crate::WriteUninit`] traits — see the
//! macro definitions in `lib.rs`.
//!
//! `*_from_slice` stays as its own op because it uses a different
//! mechanism (`CL_MEM_COPY_HOST_PTR` at `clCreateBuffer` time, not
//! a post-alloc enqueued write) — works for any marker, no
//! `Fillable` / `HostWritable` gate.

use crate::exec_ctx::ExecutionContext;
use crate::device_op::{Deps, DeviceOperation};
use crate::transfer::UploadSource;
use crate::{
    DeviceSlice, DeviceSliceUninit, MappedSlice, MappedSliceUninit, MemMode, ReadWrite, Result,
};
use std::marker::PhantomData;

// ── DeviceSlice alloc_uninit ───────────────────────────────────────

/// Lazy [`DeviceSliceUninit<T, M>`] alloc. Built by the
/// [`device_slice_alloc_uninit!`](crate::device_slice_alloc_uninit!)
/// macro or directly via [`Self::new`]. Downstream chain stages
/// transition the wrapper via [`crate::FillUninit::fill`] /
/// [`crate::WriteUninit::write`] or `unsafe { uninit.assume_init() }`.
pub struct DeviceSliceAllocUninit<T, M: MemMode = ReadWrite> {
    len: usize,
    _phantom: PhantomData<fn() -> (T, M)>,
}

impl<T, M> DeviceSliceAllocUninit<T, M>
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

impl<T, M> DeviceOperation for DeviceSliceAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = DeviceSliceUninit<T, M>;

    fn execute(
        self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(DeviceSliceUninit<T, M>, Deps)> {
        let uninit = DeviceSlice::<T, M>::alloc_uninit(ec.context(), self.len)?;
        Ok((uninit, deps))
    }
}

// ── DeviceSlice from_slice (CL_MEM_COPY_HOST_PTR, any marker) ─────

/// Lazy alloc + bake initial data via `CL_MEM_COPY_HOST_PTR`.
/// Output is a fully-initialised [`DeviceSlice<T, M>`]. **Works for
/// any marker** (including `Frozen` / `ReadOnly` / `HostReadOnly`)
/// because the data is copied at creation time — no post-creation
/// runtime write needed.
pub struct DeviceSliceFromSlice<T, M: MemMode = ReadWrite> {
    data: Option<UploadSource<T>>,
    _phantom: PhantomData<fn() -> M>,
}

impl<T, M> DeviceSliceFromSlice<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    pub fn new<S>(data: S) -> Self
    where
        S: Into<UploadSource<T>>,
    {
        Self {
            data: Some(data.into()),
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for DeviceSliceFromSlice<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn execute(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(DeviceSlice<T, M>, Deps)> {
        let source = self
            .data
            .take()
            .expect("DeviceSliceFromSlice::execute called twice");
        let buf = DeviceSlice::<T, M>::from_slice(ec.context(), source.as_slice())?;
        Ok((buf, deps))
    }
}

// ── MappedSlice alloc_uninit ───────────────────────────────────────

pub struct MappedSliceAllocUninit<T, M: MemMode = ReadWrite> {
    len: usize,
    _phantom: PhantomData<fn() -> (T, M)>,
}

impl<T, M> MappedSliceAllocUninit<T, M>
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

impl<T, M> DeviceOperation for MappedSliceAllocUninit<T, M>
where
    T: Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = MappedSliceUninit<T, M>;

    fn execute(
        self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(MappedSliceUninit<T, M>, Deps)> {
        let uninit = MappedSlice::<T, M>::alloc_uninit(ec.context(), self.len)?;
        Ok((uninit, deps))
    }
}

// ── MappedSlice from_slice ─────────────────────────────────────────

pub struct MappedSliceFromSlice<T, M: MemMode = ReadWrite> {
    data: Option<UploadSource<T>>,
    _phantom: PhantomData<fn() -> M>,
}

impl<T, M> MappedSliceFromSlice<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    pub fn new<S>(data: S) -> Self
    where
        S: Into<UploadSource<T>>,
    {
        Self {
            data: Some(data.into()),
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for MappedSliceFromSlice<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn execute(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(MappedSlice<T, M>, Deps)> {
        let source = self
            .data
            .take()
            .expect("MappedSliceFromSlice::execute called twice");
        let buf = MappedSlice::<T, M>::from_slice(ec.context(), source.as_slice())?;
        Ok((buf, deps))
    }
}
