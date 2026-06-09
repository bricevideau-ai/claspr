//! [`CopyTo`] — single polymorphic copy verb whose behavior is
//! picked by the runtime based on the source + destination types.
//!
//! ## Motivation
//!
//! Earlier the library exposed `device_slice_copy(src, dst)` and
//! `mapped_slice_copy(src, dst)` as separate free functions, each
//! type-locked to one (src, dst) pair. Adding cross-type copies
//! (e.g. `MappedSlice → USMSlice` via SVM memcpy) or Uninit-dst
//! variants (`copy_to(uninit_dst)` that transitions Uninit → Init
//! safely without an `unsafe { assume_init() }` at the call site)
//! meant scaling that verb table along the cross-product of
//! `src type × dst type × init-state`. Bad shape.
//!
//! [`CopyTo`] collapses the verb set: every supported (src, dst)
//! pair gets a trait impl that knows the right OpenCL primitive to
//! enqueue. User writes `src.copy_to(dst).and_then(...)` regardless
//! of the buffer kinds. The op's Output type encodes the post-copy
//! state — an `Uninit` dst comes back fully initialised because the
//! copy wrote every byte.
//!
//! ## Supported pairs (today)
//!
//! | Source | Destination | Primitive |
//! |---|---|---|
//! | `DeviceSlice<T, M1>` | `DeviceSlice<T, M2>` | `clEnqueueCopyBuffer` |
//! | `DeviceSlice<T, M1>` | `DeviceSliceUninit<T, M2>` | same; dst Output: Init |
//! | `MappedSlice<T, M1>` | `MappedSlice<T, M2>` | `clEnqueueSVMMemcpy` (SVM↔SVM) |
//! | `MappedSlice<T, M1>` | `MappedSliceUninit<T, M2>` | same; Output: Init |
//! | `MappedSlice<T, M1>` | `USMSlice<T, M2>` | `clEnqueueSVMMemcpy` (SVM→host-ptr, fine-grain) |
//! | `MappedSlice<T, M1>` | `USMSliceUninit<T, M2>` | same; Output: Init |
//! | `USMSlice<T, M1>` | `MappedSlice<T, M2>` | `clEnqueueSVMMemcpy` (host-ptr→SVM) |
//! | `USMSlice<T, M1>` | `MappedSliceUninit<T, M2>` | same; Output: Init |
//! | `USMSlice<T, M1>` | `USMSlice<T, M2>` | `clEnqueueSVMMemcpy` (host-ptr↔host-ptr) |
//! | `USMSlice<T, M1>` | `USMSliceUninit<T, M2>` | same; Output: Init |
//!
//! **Cross-type DeviceSlice↔SVM** is not yet implemented (would
//! need `clEnqueueReadBuffer` / `clEnqueueWriteBuffer` with an SVM
//! pointer as the host arg). Add when needed.
//!
//! ## Synchronicity
//!
//! All copies go through `clEnqueueSVMMemcpy` / `clEnqueueCopyBuffer`
//! — async OpenCL commands with proper event chaining. **No host
//! memcpy**, even for USM↔USM where both pointers are host-side
//! addressable: the runtime path keeps the chain's event graph
//! consistent and lets downstream ops gate via OpenCL's wait-list
//! semantics. (Per memory `[[arc-deviceslice-readonly]]` and the
//! late-bind-launcher pattern, every Tier 2 op should produce an
//! Event that downstream stages can wait on.)

use crate::exec_ctx::ExecutionContext;
use crate::op::{Deps, DeviceOperation, deps_as_events, wrap_event};
use claspr::{
    Buffer, DeviceSlice, DeviceSliceUninit, Launcher, MappedSlice, MappedSliceUninit, MemMode,
    Result, USMSlice, USMSliceUninit,
};
use opencl3::event::{Event, retain_event};
use opencl3::types::{CL_NON_BLOCKING, cl_event};
use std::ffi::c_void;

/// Polymorphic copy: `src.copy_to(dst).and_then(...)`. The
/// associated `Op` type is the [`DeviceOperation`] that knows how
/// to perform the right runtime copy for the (src, dst) type pair.
/// See the module rustdoc for the supported pairs.
pub trait CopyTo<Dst>: Sized {
    type Op: DeviceOperation;
    fn copy_to(self, dst: Dst) -> Self::Op;
}

/// Shared op-state container — one struct, many [`DeviceOperation`]
/// impls (one per (src, dst) pair).
pub struct CopyToOp<S, D> {
    state: Option<(S, D)>,
}

impl<S, D> CopyToOp<S, D> {
    fn new(src: S, dst: D) -> Self {
        Self {
            state: Some((src, dst)),
        }
    }

    fn take(&mut self) -> (S, D) {
        self.state
            .take()
            .expect("CopyToOp::execute called twice — internal claspr-async bug")
    }
}

// ── (DeviceSlice, DeviceSlice) — clEnqueueCopyBuffer ───────────────

impl<T, M1, M2> CopyTo<DeviceSlice<T, M2>> for DeviceSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<DeviceSlice<T, M1>, DeviceSlice<T, M2>>;
    fn copy_to(self, dst: DeviceSlice<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<DeviceSlice<T, M1>, DeviceSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (DeviceSlice<T, M1>, DeviceSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, mut dst) = self.take();
        let event = DeviceSlice::copy_to(&src, &mut dst)
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}

// ── (DeviceSlice, DeviceSliceUninit) — copy + Uninit→Init ─────────

impl<T, M1, M2> CopyTo<DeviceSliceUninit<T, M2>> for DeviceSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<DeviceSlice<T, M1>, DeviceSliceUninit<T, M2>>;
    fn copy_to(self, dst: DeviceSliceUninit<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<DeviceSlice<T, M1>, DeviceSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    // dst transitions Uninit → Init because the copy writes every byte.
    type Output = (DeviceSlice<T, M1>, DeviceSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the copy_to enqueue below writes every byte of dst
        // before the chain's downstream stages observe the buffer
        // (they wait on the returned event via Deps). No read can
        // observe uninit bytes.
        let mut dst = unsafe { uninit_dst.assume_init() };
        let event = DeviceSlice::copy_to(&src, &mut dst)
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}

// ── (MappedSlice, MappedSlice) — clEnqueueSVMMemcpy via Tier 1 ────

impl<T, M1, M2> CopyTo<MappedSlice<T, M2>> for MappedSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<MappedSlice<T, M1>, MappedSlice<T, M2>>;
    fn copy_to(self, dst: MappedSlice<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<MappedSlice<T, M1>, MappedSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (MappedSlice<T, M1>, MappedSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, dst) = self.take();
        let event = MappedSlice::copy_to(&src, &dst)
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}

// ── (MappedSlice, MappedSliceUninit) — same + Uninit→Init ─────────

impl<T, M1, M2> CopyTo<MappedSliceUninit<T, M2>> for MappedSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<MappedSlice<T, M1>, MappedSliceUninit<T, M2>>;
    fn copy_to(self, dst: MappedSliceUninit<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<MappedSlice<T, M1>, MappedSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (MappedSlice<T, M1>, MappedSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: copy_to below writes every byte of dst.
        let dst = unsafe { uninit_dst.assume_init() };
        let event = MappedSlice::copy_to(&src, &dst)
            .after_all(deps_as_events(&deps))
            .submit_on(ec)?;
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}

// ── Cross-type SVM copies (MappedSlice ↔ USMSlice) ─────────────────
//
// No Tier 1 builder for these — we issue `clEnqueueSVMMemcpy`
// directly with mixed SVM/host pointers (legal on fine-grain-system,
// which USMSlice guarantees via its construction-time check). The
// op-side bookkeeping (event retain + register_use on both buffers)
// mirrors `SvmCopyOp`'s pattern.

/// Common path for the four SVM-memcpy cross-type variants.
/// `register_src` and `register_dst` are closures the caller supplies
/// to push the event onto each buffer's last-use / in-flight list.
unsafe fn svm_memcpy_async<L: Launcher + ?Sized>(
    launcher: &L,
    dst_ptr: *mut c_void,
    src_ptr: *const c_void,
    size: usize,
    deps: &[cl_event],
) -> Result<Event> {
    // SAFETY: caller has vouched that dst_ptr / src_ptr are valid
    // SVM-or-host pointers in the queue's context, that the device
    // supports fine-grain-system SVM (USMSlice's construction
    // guarantees this), and that `size` doesn't overrun either
    // allocation. CL_NON_BLOCKING — caller is responsible for the
    // event-side wait chain.
    let event = unsafe {
        launcher
            .cl_queue()
            .enqueue_svm_mem_cpy(CL_NON_BLOCKING, dst_ptr, src_ptr, size, deps)?
    };
    Ok(event)
}

/// Retain the event once and wrap it as an Arc<Event> for
/// registration on a buffer's in-flight / last-use list. Matches
/// the SvmCopyOp pattern: the returned `event` keeps its original
/// refcount; the returned Arc holds a second refcount paired with
/// `Event::drop` inside the Arc.
unsafe fn retain_for_register(event: &Event) -> Result<std::sync::Arc<Event>> {
    // SAFETY: event.get() is live; retain pairs with the Event::drop
    // inside the Arc returned here.
    unsafe {
        retain_event(event.get())
            .map_err(|code| claspr::Error::OpenCl(opencl3::error_codes::ClError(code)))?;
    }
    Ok(std::sync::Arc::new(Event::new(event.get())))
}

fn cross_type_byte_count<T>(len: usize) -> usize {
    len * std::mem::size_of::<T>()
}

// (MappedSlice, USMSlice)

impl<T, M1, M2> CopyTo<USMSlice<T, M2>> for MappedSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<MappedSlice<T, M1>, USMSlice<T, M2>>;
    fn copy_to(self, dst: USMSlice<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<MappedSlice<T, M1>, USMSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (MappedSlice<T, M1>, USMSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, dst) = self.take();
        if src.len() != dst.len() {
            return Err(claspr::Error::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        let size = cross_type_byte_count::<T>(src.len());
        let raw_deps: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SAFETY: src.ptr() is a live SVM allocation in ec's context;
        // dst.ptr() is a live host pointer (USMSlice requires
        // fine-grain-system SVM, where host pointers are valid SVM
        // arguments to clEnqueueSVMMemcpy).
        let event = unsafe {
            svm_memcpy_async(
                ec,
                dst.ptr() as *mut c_void,
                src.ptr() as *const c_void,
                size,
                &raw_deps,
            )?
        };
        // Register on both buffers' in-flight lists so Drop is
        // queue-ordered after this copy.
        let src_arc = unsafe { retain_for_register(&event)? };
        let dst_arc = unsafe { retain_for_register(&event)? };
        src.register_use(src_arc);
        dst.register_use(dst_arc);
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}

// (MappedSlice, USMSliceUninit)

impl<T, M1, M2> CopyTo<USMSliceUninit<T, M2>> for MappedSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<MappedSlice<T, M1>, USMSliceUninit<T, M2>>;
    fn copy_to(self, dst: USMSliceUninit<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<MappedSlice<T, M1>, USMSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (MappedSlice<T, M1>, USMSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, uninit_dst) = self.take();
        if src.len() != uninit_dst.len() {
            return Err(claspr::Error::LengthMismatch {
                src: src.len(),
                dst: uninit_dst.len(),
            });
        }
        // SAFETY: the SVM memcpy below writes every byte of dst.
        let dst = unsafe { uninit_dst.assume_init() };
        let size = cross_type_byte_count::<T>(src.len());
        let raw_deps: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        // SAFETY: same as the Init variant above.
        let event = unsafe {
            svm_memcpy_async(
                ec,
                dst.ptr() as *mut c_void,
                src.ptr() as *const c_void,
                size,
                &raw_deps,
            )?
        };
        let src_arc = unsafe { retain_for_register(&event)? };
        let dst_arc = unsafe { retain_for_register(&event)? };
        src.register_use(src_arc);
        dst.register_use(dst_arc);
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}

// (USMSlice, MappedSlice)

impl<T, M1, M2> CopyTo<MappedSlice<T, M2>> for USMSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<USMSlice<T, M1>, MappedSlice<T, M2>>;
    fn copy_to(self, dst: MappedSlice<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<USMSlice<T, M1>, MappedSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (USMSlice<T, M1>, MappedSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, dst) = self.take();
        if src.len() != dst.len() {
            return Err(claspr::Error::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        let size = cross_type_byte_count::<T>(src.len());
        let raw_deps: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let event = unsafe {
            svm_memcpy_async(
                ec,
                dst.ptr() as *mut c_void,
                src.ptr() as *const c_void,
                size,
                &raw_deps,
            )?
        };
        let src_arc = unsafe { retain_for_register(&event)? };
        let dst_arc = unsafe { retain_for_register(&event)? };
        src.register_use(src_arc);
        dst.register_use(dst_arc);
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}

// (USMSlice, MappedSliceUninit)

impl<T, M1, M2> CopyTo<MappedSliceUninit<T, M2>> for USMSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<USMSlice<T, M1>, MappedSliceUninit<T, M2>>;
    fn copy_to(self, dst: MappedSliceUninit<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<USMSlice<T, M1>, MappedSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (USMSlice<T, M1>, MappedSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, uninit_dst) = self.take();
        if src.len() != uninit_dst.len() {
            return Err(claspr::Error::LengthMismatch {
                src: src.len(),
                dst: uninit_dst.len(),
            });
        }
        // SAFETY: the SVM memcpy below writes every byte of dst.
        let dst = unsafe { uninit_dst.assume_init() };
        let size = cross_type_byte_count::<T>(src.len());
        let raw_deps: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let event = unsafe {
            svm_memcpy_async(
                ec,
                dst.ptr() as *mut c_void,
                src.ptr() as *const c_void,
                size,
                &raw_deps,
            )?
        };
        let src_arc = unsafe { retain_for_register(&event)? };
        let dst_arc = unsafe { retain_for_register(&event)? };
        src.register_use(src_arc);
        dst.register_use(dst_arc);
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}

// (USMSlice, USMSlice)

impl<T, M1, M2> CopyTo<USMSlice<T, M2>> for USMSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<USMSlice<T, M1>, USMSlice<T, M2>>;
    fn copy_to(self, dst: USMSlice<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<USMSlice<T, M1>, USMSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (USMSlice<T, M1>, USMSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, dst) = self.take();
        if src.len() != dst.len() {
            return Err(claspr::Error::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        let size = cross_type_byte_count::<T>(src.len());
        let raw_deps: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let event = unsafe {
            svm_memcpy_async(
                ec,
                dst.ptr() as *mut c_void,
                src.ptr() as *const c_void,
                size,
                &raw_deps,
            )?
        };
        let src_arc = unsafe { retain_for_register(&event)? };
        let dst_arc = unsafe { retain_for_register(&event)? };
        src.register_use(src_arc);
        dst.register_use(dst_arc);
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}

// (USMSlice, USMSliceUninit)

impl<T, M1, M2> CopyTo<USMSliceUninit<T, M2>> for USMSlice<T, M1>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Op = CopyToOp<USMSlice<T, M1>, USMSliceUninit<T, M2>>;
    fn copy_to(self, dst: USMSliceUninit<T, M2>) -> Self::Op {
        CopyToOp::new(self, dst)
    }
}

impl<T, M1, M2> DeviceOperation for CopyToOp<USMSlice<T, M1>, USMSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (USMSlice<T, M1>, USMSlice<T, M2>);

    fn execute(mut self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        let (src, uninit_dst) = self.take();
        if src.len() != uninit_dst.len() {
            return Err(claspr::Error::LengthMismatch {
                src: src.len(),
                dst: uninit_dst.len(),
            });
        }
        // SAFETY: the SVM memcpy below writes every byte of dst.
        let dst = unsafe { uninit_dst.assume_init() };
        let size = cross_type_byte_count::<T>(src.len());
        let raw_deps: Vec<cl_event> = deps.iter().map(|d| d.as_ref().get()).collect();
        let event = unsafe {
            svm_memcpy_async(
                ec,
                dst.ptr() as *mut c_void,
                src.ptr() as *const c_void,
                size,
                &raw_deps,
            )?
        };
        let src_arc = unsafe { retain_for_register(&event)? };
        let dst_arc = unsafe { retain_for_register(&event)? };
        src.register_use(src_arc);
        dst.register_use(dst_arc);
        Ok(((src, dst), vec![wrap_event(event)]))
    }
}
