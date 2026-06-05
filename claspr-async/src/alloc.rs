//! Lazy buffer allocs + constructors as [`DeviceOperation`]s.
//!
//! Where the Tier 1 constructors are synchronous fallible host calls
//! that need a `Context` (or `Launcher`) in scope, these defer the
//! alloc until execute time. Two ergonomic wins:
//!
//! 1. **Hoist allocs to the top of a chain** via [`bundle!`](crate::bundle!).
//! 2. **Compose with [`and_then_with_context`](crate::DeviceOperation::and_then_with_context)**.
//!
//! All op structs carry an `M: MemMode = ReadWrite` default — the
//! Tier 2 macros ([`device_slice_alloc_zero!`](crate::device_slice_alloc_zero!),
//! etc.) expand to `Foo::<T>::new(N)` (default marker) or
//! `Foo::<T, M>::new(N)` (explicit marker). Users do not construct
//! these ops directly; the macros are the entry point.
//!
//! ## What these don't buy
//!
//! GPU-side parallelism. `clCreateBuffer` / `clSVMAlloc` are
//! synchronous host calls; a `bundle!` of three allocs runs the
//! three constructors sequentially on the executing thread.

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use crate::transfer::UploadSource;
use claspr::{
    DeviceSlice, DeviceSliceUninit, Fillable, MappedSlice, MappedSliceUninit, MemMode, ReadWrite,
    Result, register_drop_callback,
};
use std::marker::PhantomData;

// ── DeviceSlice alloc_zero ─────────────────────────────────────────

/// Lazy [`DeviceSlice<T, M>`] zero-init alloc. Built by the
/// [`device_slice_alloc_zero!`](crate::device_slice_alloc_zero!) macro
/// or directly via [`Self::new`].
pub struct DeviceSliceAllocZero<T, M: MemMode = ReadWrite> {
    len: usize,
    _phantom: PhantomData<fn() -> (T, M)>,
}

impl<T, M> DeviceSliceAllocZero<T, M>
where
    T: Default + Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    pub fn new(len: usize) -> Self {
        Self {
            len,
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for DeviceSliceAllocZero<T, M>
where
    T: Default + Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(DeviceSlice<T, M>, Deps)> {
        // alloc_zero zero-fills synchronously on the context's
        // default queue and blocks until done; bytes are valid when
        // this returns.
        let buf = DeviceSlice::<T, M>::alloc_zero(ec.context(), self.len)?;
        Ok((buf, deps))
    }
}

// ── DeviceSlice alloc_uninit ───────────────────────────────────────

/// Lazy [`DeviceSliceUninit<T, M>`] alloc. Built by the
/// [`device_slice_alloc_uninit!`](crate::device_slice_alloc_uninit!) macro
/// or directly via [`Self::new`]. Output is the type-stated
/// [`DeviceSliceUninit`] wrapper; downstream chain stages must
/// transition it via `unsafe { uninit.assume_init() }` (then fill /
/// write / kernel-pass) before any host read.
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

// ── DeviceSlice filled (alloc + fill) ──────────────────────────────

/// Lazy alloc + fill — produces a [`DeviceSlice<T, M>`] of `len`
/// elements all set to `value`. Dispatches between runtime
/// `clEnqueueFillBuffer` and the built-in device-kernel fill based
/// on the marker's [`FillStrategy`](claspr::FillStrategy).
pub struct DeviceSliceFilled<T: Copy, M: MemMode = ReadWrite> {
    value: T,
    len: usize,
    _phantom: PhantomData<fn() -> M>,
}

impl<T, M> DeviceSliceFilled<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    pub fn new(value: T, len: usize) -> Self {
        Self {
            value,
            len,
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for DeviceSliceFilled<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(DeviceSlice<T, M>, Deps)> {
        let uninit = DeviceSlice::<T, M>::alloc_uninit(ec.context(), self.len)?;
        // SAFETY: fill below overwrites every byte before any read;
        // downstream gates on the returned fill event.
        let mut buf = unsafe { uninit.assume_init() };
        let event = buf
            .fill(self.value)
            .after_all(deps_as_events(&deps))
            .submit(ec)?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// ── DeviceSlice from_slice (NEW — CL_MEM_COPY_HOST_PTR, any marker) ─

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

// ── MappedSlice alloc_zero ─────────────────────────────────────────

pub struct MappedSliceAllocZero<T, M: MemMode = ReadWrite> {
    len: usize,
    _phantom: PhantomData<fn() -> (T, M)>,
}

impl<T, M> MappedSliceAllocZero<T, M>
where
    T: Default + Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    pub fn new(len: usize) -> Self {
        Self {
            len,
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for MappedSliceAllocZero<T, M>
where
    T: Default + Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(MappedSlice<T, M>, Deps)> {
        let buf = MappedSlice::<T, M>::alloc_zero(ec.context(), self.len)?;
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

// ── MappedSlice filled ─────────────────────────────────────────────

pub struct MappedSliceFilled<T: Copy, M: MemMode = ReadWrite> {
    value: T,
    len: usize,
    _phantom: PhantomData<fn() -> M>,
}

impl<T, M> MappedSliceFilled<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    pub fn new(value: T, len: usize) -> Self {
        Self {
            value,
            len,
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for MappedSliceFilled<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn execute(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(MappedSlice<T, M>, Deps)> {
        let uninit = MappedSlice::<T, M>::alloc_uninit(ec.context(), self.len)?;
        // SAFETY: fill below overwrites every byte before any read.
        let buf = unsafe { uninit.assume_init() };
        let event = buf
            .fill(self.value)
            .after_all(deps_as_events(&deps))
            .submit(ec)?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// ── MappedSlice upload (alloc + .write) ───────────────────────────

pub struct MappedSliceUpload<T, M: MemMode = ReadWrite> {
    source: Option<UploadSource<T>>,
    _phantom: PhantomData<fn() -> M>,
}

impl<T, M> MappedSliceUpload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + claspr::HostWritable + Fillable + Send + 'static,
{
    pub fn new<S>(source: S) -> Self
    where
        S: Into<UploadSource<T>>,
    {
        Self {
            source: Some(source.into()),
            _phantom: PhantomData,
        }
    }
}

impl<T, M> DeviceOperation for MappedSliceUpload<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + claspr::HostWritable + Fillable + Send + 'static,
{
    type Output = MappedSlice<T, M>;

    fn execute(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> Result<(MappedSlice<T, M>, Deps)> {
        let source = self
            .source
            .take()
            .expect("MappedSliceUpload::execute called twice");
        let len = source.len();
        let uninit = MappedSlice::<T, M>::alloc_uninit(ec.context(), len)?;
        // SAFETY: write below overwrites every byte; downstream
        // stages gate on the returned write event.
        let buf = unsafe { uninit.assume_init() };
        let event = buf
            .write(source.as_slice())
            .after_all(deps_as_events(&deps))
            .submit(ec)?;
        register_drop_callback(&event, Box::new(source))?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// ── MappedSlice from_slice (NEW) ───────────────────────────────────

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
