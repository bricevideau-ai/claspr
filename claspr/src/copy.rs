//! [`CopyTo`] — single polymorphic copy verb whose behavior is
//! picked by the runtime based on the source + destination types.
//!
//! One polymorphic verb instead of a per-(src, dst)-pair free-function table (which
//! would scale along `src type × dst type × init-state`): every supported (src, dst)
//! pair gets a trait impl that knows the right OpenCL primitive to enqueue. User
//! writes `src.copy_to(dst).and_then(...)` regardless of the buffer kinds. The op's
//! Output type encodes the post-copy state — an `Uninit` dst comes back fully
//! initialised because the copy wrote every byte (no `unsafe { assume_init() }` at
//! the call site).
//!
//! ## Supported pairs
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
//! pointer as the host arg).
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

use crate::eager::{Deps, DeviceEnqueue, deps_to_wait_list, single_dep};
use crate::exec_ctx::ExecutionContext;
use crate::record::RecordableBuffer;
use crate::{
    Buffer, DeviceSlice, DeviceSliceUninit, Launcher, MappedSlice, MappedSliceUninit, MemMode,
    Result, USMSlice, USMSliceUninit,
};

/// Shared CB-record helper for the copy ops: record a copy of `src → dst` (both
/// `RecordableBuffer`, so cl_mem or SVM) into `builder` and return its sync point.
/// `None` when the builder lacks the command / the pair is ineligible (a mixed
/// cl_mem/SVM copy) — `CbBuilder::copy_buffer` marks itself ineligible then — OR
/// when the two operands differ in byte length: rather than silently truncating to
/// `min(src, dst)` (which the CB path used to do, dropping data where the per-op
/// path returns `LengthMismatch`), mark the builder ineligible and return `None` so
/// the boundary discards the CB and re-runs on the per-op path — whose
/// `copy_buffer_enqueue` surfaces the real `Err(LengthMismatch)`. This keeps the
/// CB-accelerated and per-op copies behaviourally identical on a size mismatch.
fn record_copy_cmd<S: RecordableBuffer, D: RecordableBuffer>(
    builder: &crate::record::CbBuilder,
    src: &S,
    dst: &D,
    waits: &crate::exec_ctx::SyncPoints,
) -> Option<crate::cl_sync_point_khr> {
    let sh = src.record_handle();
    let dh = dst.record_handle();
    if sh.byte_len != dh.byte_len {
        // Defer to the per-op path (which errors) instead of truncating.
        builder.mark_ineligible();
        return None;
    }
    builder.copy_buffer(sh.mem, dh.mem, 0, 0, sh.byte_len, waits)
}
use opencl3::event::{Event, retain_event};
use opencl3::types::{CL_NON_BLOCKING, cl_event};
use std::ffi::c_void;

/// Polymorphic copy: `src.copy_to(dst)`. The associated `Op` type is the
/// [`DeviceEnqueue`] op that knows how to perform the right runtime copy for the
/// (src, dst) type pair. The eager `copy_to` graph leaf
/// ([`CopyTo2`](crate::eager::CopyTo2)) drives it. See the module rustdoc for
/// the supported pairs.
pub trait CopyTo<Dst>: Sized {
    type Op: CopyRun;
    fn copy_to(self, dst: Dst) -> Self::Op;
}

/// The copy ops' fallible enqueue with BUFFER RECOVERY on error — the
/// copy-specific superset of [`DeviceEnqueue::run`]. A copy CONSUMES its buffers
/// (the `Uninit → Init` `assume_init` is by-value), and `CopyTo2` holds their
/// return homes, so a failed copy that just dropped the buffers would strand the
/// origin cells in `Lent` (review finding #1). This method does the by-value
/// transition FIRST, then the enqueue, so on failure it hands the recovered,
/// OUTPUT-typed buffers back — `Err((error, output))` — for `CopyTo2` to rehome.
///
/// It lives on its own trait (not on [`DeviceEnqueue`]) so ONLY the copy ops that
/// hold homes pay for recovery; the host-view acquire/release ops (the other
/// `DeviceEnqueue` impls) resolve homeless and keep the plain `run`. The blanket
/// `DeviceEnqueue::run` below delegates here and discards the recovered buffers,
/// so the non-recovering callers are unchanged.
pub trait CopyRun: Send + Sized {
    /// Copy output — the `(src, dst)` pair, dst possibly retyped `Uninit → Init`.
    type Output: Send;
    /// See the trait docs. `Ok((output, deps))` or `Err((error, output))`.
    fn run_recover(
        self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)>;
    /// CB twin — identical to [`DeviceEnqueue::record_cb`]; copy overrides it.
    #[allow(clippy::type_complexity)]
    fn record_cb_recover(
        self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)>;
}

// Every `CopyRun` op is also a plain `DeviceEnqueue` (so it still composes in the
// host-view/copy call sites that want the simple contract): `run` delegates to
// `run_recover` and drops the recovered buffers on error; `record_cb` forwards.
impl<O: CopyRun> DeviceEnqueue for O {
    type Output = <O as CopyRun>::Output;
    fn run(self, ec: &ExecutionContext<'_>, deps: Deps) -> Result<(Self::Output, Deps)> {
        self.run_recover(ec, deps).map_err(|(e, _buffers)| e)
    }
    fn record_cb(
        self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        self.record_cb_recover(builder, waits)
    }
}

/// Shared op-state container — one struct, many [`DeviceEnqueue`]
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
            .expect("CopyToOp::run called twice — internal claspr bug")
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

impl<T, M1, M2> CopyRun for CopyToOp<DeviceSlice<T, M1>, DeviceSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (DeviceSlice<T, M1>, DeviceSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, mut dst) = self.take();
        let raw = deps_to_wait_list(&deps);
        match crate::buffer::copy_buffer_enqueue(&src, &mut dst, ec, &raw) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, dst) = self.take();
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
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

impl<T, M1, M2> CopyRun for CopyToOp<DeviceSlice<T, M1>, DeviceSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    // dst transitions Uninit → Init because the copy writes every byte.
    type Output = (DeviceSlice<T, M1>, DeviceSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the copy_to enqueue below writes every byte of dst
        // before the chain's downstream stages observe the buffer
        // (they wait on the returned event via Deps). No read can
        // observe uninit bytes.
        let mut dst = unsafe { uninit_dst.assume_init() };
        let raw = deps_to_wait_list(&deps);
        match crate::buffer::copy_buffer_enqueue(&src, &mut dst, ec, &raw) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the recorded copy writes every byte before any downstream stage
        // observes the buffer (they wait on the CB's completion). Same soundness as
        // the `run` path's `assume_init`.
        let dst = unsafe { uninit_dst.assume_init() };
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
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

impl<T, M1, M2> CopyRun for CopyToOp<MappedSlice<T, M1>, MappedSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (MappedSlice<T, M1>, MappedSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, dst) = self.take();
        let raw = deps_to_wait_list(&deps);
        match crate::mapped::svm_copy_enqueue(&src, &dst, ec, &raw) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, dst) = self.take();
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
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

impl<T, M1, M2> CopyRun for CopyToOp<MappedSlice<T, M1>, MappedSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (MappedSlice<T, M1>, MappedSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: copy_to below writes every byte of dst.
        let dst = unsafe { uninit_dst.assume_init() };
        let raw = deps_to_wait_list(&deps);
        match crate::mapped::svm_copy_enqueue(&src, &dst, ec, &raw) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the recorded copy writes every byte before downstream observes it.
        let dst = unsafe { uninit_dst.assume_init() };
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
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
            .map_err(|code| crate::Error::OpenCl(opencl3::error_codes::ClError(code)))?;
    }
    Ok(std::sync::Arc::new(Event::new(event.get())))
}

fn cross_type_byte_count<T>(len: usize) -> usize {
    len * std::mem::size_of::<T>()
}

/// SVM-backed slices that record an in-flight event so their `Drop` is
/// queue-ordered after a copy touching them. Both kinds have the same inherent
/// `register_use` + `ptr` + `len`; this trait lets [`svm_memcpy_copy`] drive them
/// generically (the six cross-type SVM copy bodies were byte-identical apart from
/// operand types).
trait SvmCopyOperand {
    fn register_use(&self, event: std::sync::Arc<Event>);
    fn svm_ptr(&self) -> *mut c_void;
    fn elem_len(&self) -> usize;
}

impl<T, M: MemMode> SvmCopyOperand for MappedSlice<T, M> {
    fn register_use(&self, event: std::sync::Arc<Event>) {
        MappedSlice::register_use(self, event);
    }
    fn svm_ptr(&self) -> *mut c_void {
        self.ptr() as *mut c_void
    }
    fn elem_len(&self) -> usize {
        self.len()
    }
}

impl<T, M: MemMode> SvmCopyOperand for USMSlice<T, M> {
    fn register_use(&self, event: std::sync::Arc<Event>) {
        USMSlice::register_use(self, event);
    }
    fn svm_ptr(&self) -> *mut c_void {
        self.ptr() as *mut c_void
    }
    fn elem_len(&self) -> usize {
        self.len()
    }
}

/// The one cross-type SVM copy body, factored out of the six `(Mapped|USM) →
/// (Mapped|USM)` `run_recover` impls that were byte-identical apart from operand
/// types. Length-checks, issues the memcpy, then retains + registers the event on
/// both operands. On ANY failure returns the error WITHOUT registering anything —
/// the caller still owns `src`/`dst` intact to hand back for rehoming.
fn svm_memcpy_copy<S: SvmCopyOperand, D: SvmCopyOperand, T>(
    ec: &ExecutionContext<'_>,
    src: &S,
    dst: &D,
    deps: &Deps,
) -> Result<Event> {
    if src.elem_len() != dst.elem_len() {
        return Err(crate::Error::LengthMismatch {
            src: src.elem_len(),
            dst: dst.elem_len(),
        });
    }
    let size = cross_type_byte_count::<T>(src.elem_len());
    let raw = deps_to_wait_list(deps);
    // SAFETY: both pointers are live SVM/host pointers in ec's context on a
    // fine-grain-system device (USMSlice's construction guarantees it); `size` fits
    // both (checked equal above). CL_NON_BLOCKING — event-side wait chain.
    let event = unsafe { svm_memcpy_async(ec, dst.svm_ptr(), src.svm_ptr(), size, &raw)? };
    // Register only after a successful enqueue (nothing to unwind on the error
    // path — the buffers are handed back unregistered).
    let src_arc = unsafe { retain_for_register(&event)? };
    let dst_arc = unsafe { retain_for_register(&event)? };
    src.register_use(src_arc);
    dst.register_use(dst_arc);
    Ok(event)
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

impl<T, M1, M2> CopyRun for CopyToOp<MappedSlice<T, M1>, USMSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (MappedSlice<T, M1>, USMSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, dst) = self.take();
        match svm_memcpy_copy::<_, _, T>(ec, &src, &dst, &deps) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, dst) = self.take();
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
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

impl<T, M1, M2> CopyRun for CopyToOp<MappedSlice<T, M1>, USMSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (MappedSlice<T, M1>, USMSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the SVM memcpy writes every byte of dst before any downstream
        // stage observes it (they gate on the returned event).
        let dst = unsafe { uninit_dst.assume_init() };
        match svm_memcpy_copy::<_, _, T>(ec, &src, &dst, &deps) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the recorded copy writes every byte before downstream observes it.
        let dst = unsafe { uninit_dst.assume_init() };
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
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

impl<T, M1, M2> CopyRun for CopyToOp<USMSlice<T, M1>, MappedSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (USMSlice<T, M1>, MappedSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, dst) = self.take();
        match svm_memcpy_copy::<_, _, T>(ec, &src, &dst, &deps) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, dst) = self.take();
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
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

impl<T, M1, M2> CopyRun for CopyToOp<USMSlice<T, M1>, MappedSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (USMSlice<T, M1>, MappedSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the SVM memcpy writes every byte of dst before any downstream
        // stage observes it (they gate on the returned event).
        let dst = unsafe { uninit_dst.assume_init() };
        match svm_memcpy_copy::<_, _, T>(ec, &src, &dst, &deps) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the recorded copy writes every byte before downstream observes it.
        let dst = unsafe { uninit_dst.assume_init() };
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
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

impl<T, M1, M2> CopyRun for CopyToOp<USMSlice<T, M1>, USMSlice<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (USMSlice<T, M1>, USMSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, dst) = self.take();
        match svm_memcpy_copy::<_, _, T>(ec, &src, &dst, &deps) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, dst) = self.take();
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
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

impl<T, M1, M2> CopyRun for CopyToOp<USMSlice<T, M1>, USMSliceUninit<T, M2>>
where
    T: Send + 'static,
    M1: MemMode + Send + 'static,
    M2: MemMode + Send + 'static,
{
    type Output = (USMSlice<T, M1>, USMSlice<T, M2>);

    fn run_recover(
        mut self,
        ec: &ExecutionContext<'_>,
        deps: Deps,
    ) -> std::result::Result<(Self::Output, Deps), (crate::Error, Self::Output)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the SVM memcpy writes every byte of dst before any downstream
        // stage observes it (they gate on the returned event).
        let dst = unsafe { uninit_dst.assume_init() };
        match svm_memcpy_copy::<_, _, T>(ec, &src, &dst, &deps) {
            Ok(event) => Ok(((src, dst), single_dep(event))),
            Err(e) => Err((e, (src, dst))),
        }
    }

    fn record_cb_recover(
        mut self,
        builder: Option<&crate::record::CbBuilder>,
        waits: &crate::exec_ctx::SyncPoints,
    ) -> Option<(Self::Output, Option<crate::cl_sync_point_khr>)> {
        let (src, uninit_dst) = self.take();
        // SAFETY: the recorded copy writes every byte before downstream observes it.
        let dst = unsafe { uninit_dst.assume_init() };
        let sp = builder.and_then(|b| record_copy_cmd(b, &src, &dst, waits));
        Some(((src, dst), sp))
    }
}
