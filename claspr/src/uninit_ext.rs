//! [`FillUninit`] / [`WriteUninit`] — `.fill()` / `.write()`
//! consuming methods on the `*Uninit` wrapper types from claspr.
//!
//! These are the compositional primitives that the higher-level
//! macros (`device_slice_alloc_zero!`, `device_slice_filled!`,
//! `upload!`, …) expand to. Pattern:
//!
//! ```ignore
//! device_slice_alloc_uninit!(u32, N)        // → DeviceSliceUninit
//!     .and_then(|u| u.fill(0u32))           // → DeviceSlice (init)
//!     .and_then(|buf| kernel.do(buf))       // → ...
//! ```
//!
//! `.fill` / `.write` consume the Uninit wrapper and return a
//! [`DeviceOperation`] whose Output is the now-initialised buffer.
//! No `unsafe { assume_init() }` at the call site — the trait
//! impls own the safety contract (each fill/write op writes every
//! byte of the buffer before the chain advances).
//!
//! ## Why a trait
//!
//! Three Uninit wrappers (DeviceSliceUninit, MappedSliceUninit,
//! USMSliceUninit) need the same verb (`.fill`, `.write`) with
//! type-dependent dispatch:
//! - DeviceSliceUninit → enqueues via the Fillable strategy (runtime
//!   `clEnqueueFillBuffer` or built-in fill kernel)
//! - MappedSliceUninit → SVM analog (`clEnqueueSVMMemFill` etc.)
//! - USMSliceUninit → pure host operation (the wrapped Vec)
//!
//! Single trait per verb with an associated `Op` type lets one
//! user-facing call shape cover all three.

use crate::device_op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use crate::exec_ctx::ExecutionContext;
use crate::transfer::UploadSource;
use crate::{
    DeviceSlice, DeviceSliceUninit, Fillable, HostUploadable, HostWritable, MappedSlice,
    MappedSliceUninit, MemMode, Result, USMSlice, USMSliceUninit, register_drop_callback,
};

// ── FillUninit trait ───────────────────────────────────────────────

/// Consume an Uninit wrapper and produce a [`DeviceOperation`] that
/// fills it with `value`, yielding the initialised buffer.
pub trait FillUninit<T>: Sized {
    type Op: DeviceOperation;
    fn fill(self, value: T) -> Self::Op;
}

/// Shared op struct for the .fill() trait. One struct, three
/// [`DeviceOperation`] impls (one per Uninit wrapper type).
pub struct FillFromUninitOp<U, T> {
    state: Option<(U, T)>,
}

impl<U, T> FillFromUninitOp<U, T> {
    fn new(uninit: U, value: T) -> Self {
        Self {
            state: Some((uninit, value)),
        }
    }
    fn take(&mut self) -> (U, T) {
        self.state
            .take()
            .expect("FillFromUninitOp::execute called twice — internal claspr-async bug")
    }
}

// DeviceSliceUninit.fill — async, via Tier 1 .fill() + Fillable dispatch.

impl<T, M> FillUninit<T> for DeviceSliceUninit<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Op = FillFromUninitOp<DeviceSliceUninit<T, M>, T>;
    fn fill(self, value: T) -> Self::Op {
        FillFromUninitOp::new(self, value)
    }
}

impl<T, M> DeviceOperation for FillFromUninitOp<DeviceSliceUninit<T, M>, T>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;
    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (uninit, value) = self.take();
        // SAFETY: fill below writes every byte; downstream gates on
        // the returned fill event.
        let mut buf = unsafe { uninit.assume_init() };
        let event = buf
            .fill(value)
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// MappedSliceUninit.fill — same shape, SVM Tier 1 .fill().

impl<T, M> FillUninit<T> for MappedSliceUninit<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Op = FillFromUninitOp<MappedSliceUninit<T, M>, T>;
    fn fill(self, value: T) -> Self::Op {
        FillFromUninitOp::new(self, value)
    }
}

impl<T, M> DeviceOperation for FillFromUninitOp<MappedSliceUninit<T, M>, T>
where
    T: Copy + Send + 'static,
    M: MemMode + Fillable + Send + 'static,
{
    type Output = MappedSlice<T, M>;
    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (uninit, value) = self.take();
        // SAFETY: fill below writes every byte.
        let buf = unsafe { uninit.assume_init() };
        let event = buf
            .fill(value)
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// USMSliceUninit.fill — pure host op via Tier 1 `fill_into` helper.

impl<T, M> FillUninit<T> for USMSliceUninit<T, M>
where
    T: Copy + Send + 'static,
    M: MemMode + Send + 'static,
{
    type Op = FillFromUninitOp<USMSliceUninit<T, M>, T>;
    fn fill(self, value: T) -> Self::Op {
        FillFromUninitOp::new(self, value)
    }
}

impl<T, M> DeviceOperation for FillFromUninitOp<USMSliceUninit<T, M>, T>
where
    T: Copy + Send + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSlice<T, M>;
    fn execute(mut self, _ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (uninit, value) = self.take();
        // Pure host op — no event, deps pass through unchanged.
        let buf = uninit.fill_into(value);
        Ok((buf, deps))
    }
}

// ── WriteUninit trait ──────────────────────────────────────────────

/// Consume an Uninit wrapper and produce a [`DeviceOperation`] that
/// writes host-side data into it, yielding the initialised buffer.
pub trait WriteUninit<T>: Sized {
    type Op: DeviceOperation;
    fn write<S>(self, src: S) -> Self::Op
    where
        S: Into<UploadSource<T>>;
}

pub struct WriteFromUninitOp<U, T> {
    state: Option<(U, UploadSource<T>)>,
}

impl<U, T> WriteFromUninitOp<U, T> {
    fn new(uninit: U, src: UploadSource<T>) -> Self {
        Self {
            state: Some((uninit, src)),
        }
    }
    fn take(&mut self) -> (U, UploadSource<T>) {
        self.state
            .take()
            .expect("WriteFromUninitOp::execute called twice — internal claspr-async bug")
    }
}

// DeviceSliceUninit.write — Tier 1 .write() (non-blocking) + drop callback.

impl<T, M> WriteUninit<T> for DeviceSliceUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostUploadable + HostWritable + Send + 'static,
{
    type Op = WriteFromUninitOp<DeviceSliceUninit<T, M>, T>;
    fn write<S>(self, src: S) -> Self::Op
    where
        S: Into<UploadSource<T>>,
    {
        WriteFromUninitOp::new(self, src.into())
    }
}

impl<T, M> DeviceOperation for WriteFromUninitOp<DeviceSliceUninit<T, M>, T>
where
    T: Send + Sync + 'static,
    M: MemMode + HostUploadable + HostWritable + Send + 'static,
{
    type Output = DeviceSlice<T, M>;
    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (uninit, src) = self.take();
        // SAFETY: non-blocking write below covers every byte of the
        // buffer; downstream gates on the returned write event.
        let mut buf = unsafe { uninit.assume_init() };
        let event = buf
            .write(src.as_slice())
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        // Keep-alive: drop the host source when CL_COMPLETE fires.
        register_drop_callback(&event, Box::new(src))?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// MappedSliceUninit.write — SVM analog via Tier 1 .write() (SvmWriteOp).

impl<T, M> WriteUninit<T> for MappedSliceUninit<T, M>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    type Op = WriteFromUninitOp<MappedSliceUninit<T, M>, T>;
    fn write<S>(self, src: S) -> Self::Op
    where
        S: Into<UploadSource<T>>,
    {
        WriteFromUninitOp::new(self, src.into())
    }
}

impl<T, M> DeviceOperation for WriteFromUninitOp<MappedSliceUninit<T, M>, T>
where
    T: Send + Sync + 'static,
    M: MemMode + HostWritable + Send + 'static,
{
    type Output = MappedSlice<T, M>;
    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (uninit, src) = self.take();
        // SAFETY: SVM write below covers every byte.
        let buf = unsafe { uninit.assume_init() };
        let event = buf
            .write(src.as_slice())
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        register_drop_callback(&event, Box::new(src))?;
        Ok((buf, vec![wrap_event(event)]))
    }
}

// USMSliceUninit.write — pure host op via Tier 1 `write_from` helper.
// Surfaces LengthMismatch from write_from at execute time.

impl<T, M> WriteUninit<T> for USMSliceUninit<T, M>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    type Op = WriteFromUninitOp<USMSliceUninit<T, M>, T>;
    fn write<S>(self, src: S) -> Self::Op
    where
        S: Into<UploadSource<T>>,
    {
        WriteFromUninitOp::new(self, src.into())
    }
}

impl<T, M> DeviceOperation for WriteFromUninitOp<USMSliceUninit<T, M>, T>
where
    T: Copy + Send + Sync + 'static,
    M: MemMode + Send + 'static,
{
    type Output = USMSlice<T, M>;
    fn execute(mut self, _ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (uninit, src) = self.take();
        // Host memcpy via Tier 1 helper. Returns Err on length
        // mismatch — propagates as the op's Err.
        let buf = uninit.write_from(src.as_slice())?;
        // src dropped at end of this function (no async work to keep
        // it alive for — the memcpy is done).
        Ok((buf, deps))
    }
}
