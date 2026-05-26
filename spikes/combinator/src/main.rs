//! Combinator-API spike for the claspr execution-model Tier 2 design.
//!
//! Inspired by cuda-oxide's `DeviceOperation` + `and_then` + `bundle!` pattern,
//! adapted for OpenCL's queue/event model. The goal of this spike is to
//! validate that closure-based composition with move semantics actually
//! resolves the borrow-checker tensions we hit with the explicit handle-
//! based design (`Pending<T>`, `BorrowHandle<'a, T>`, etc.).
//!
//! The execution machinery is faked — everything runs synchronously in
//! `.sync()`. The point is to validate the type structure, not the runtime.
//!
//! Scenarios covered (in order of complexity):
//!   1. Linear chain (producer/consumer pipeline)
//!   2. Independent parallel branches via bundle!
//!   3. Diamond (fan-out then fan-in via Arc)
//!   4. ML forward pass (state carried through stages)
//!   5. In-place mutation chain (kernel mutates the buffer it receives)
//!   6. N-ary fan-out via fan_out (variadic, beyond bundle's arity)
//!   7. Multi-producer, single consumer
//!   8. Mixed sync/async (split await with host work in between)
//!   9. Conditional graph shape (dynamic DAG)
//!  10. Error propagation through and_then
//!  11. Buffer round-trip (pass in, get back out at the end)
//!  12. Profiling via callback (event-completion-driven, not wrapped Output)
//!  13. Batch parallelism via fan_out + implicit marker (no spawn needed)
//!  14. Cross-device pipeline (single context spans devices)
//!  15. .and_then_host(...) for in-queue host work without split-await
//!  16. HostAccessible trait — three-stage acquire / host / release

use std::fmt;
use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error: {}", self.0)
    }
}

impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error(s.to_string())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error(s)
    }
}

// ── Device (fake, models claspr::Device which wraps cl_device_id) ────
//
// In real claspr, `Device` is a typed opaque handle around `cl_device_id`
// (which itself is an opaque pointer). Sub-devices (created via
// `clCreateSubDevices` — partitioning a parent into equally-sized or
// custom slices for fine-grained scheduling) are real `cl_device_id`s
// in their own right, with a parent relationship visible via
// `clGetDeviceInfo(CL_DEVICE_PARENT_DEVICE)`.
//
// The spike fakes the handle as an Arc<DeviceInner> with a unique id,
// so Device is Clone (cheap refcount) and PartialEq (handle identity).

#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

struct DeviceInner {
    name: String,
    // Real claspr would carry cl_device_id, Platform, parent (for sub-devices), etc.
}

impl Device {
    pub fn new(name: impl Into<String>) -> Self {
        Device {
            inner: Arc::new(DeviceInner { name: name.into() }),
        }
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Device {}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device({})", self.inner.name)
    }
}

// ── ExecutionContext (fake) ──────────────────────────────────────────

/// What an op sees when it runs. Real version carries a `CommandQueue`
/// + a reference to the per-device queue table so cross-device routing
/// (`.on_device(&dev)`) can pick the right queue at op-execution time.
pub struct ExecutionContext {
    device: Device,
    log: Arc<Mutex<Vec<String>>>,
}

impl ExecutionContext {
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn log(&self, msg: impl Into<String>) {
        self.log.lock().unwrap().push(msg.into());
    }
}

/// Fake scheduling policy — always returns a fresh context bound to
/// the default device. Real version picks the per-device default OOO
/// queue from the Context.
fn make_ctx(log: Arc<Mutex<Vec<String>>>) -> ExecutionContext {
    ExecutionContext {
        device: Device::new("default"),
        log,
    }
}

// ── DeviceOperation trait ────────────────────────────────────────────

/// The core trait. Anything that describes lazy GPU work implements it.
pub trait DeviceOperation: Send + Sized {
    type Output: Send;

    fn execute(self, ctx: &ExecutionContext) -> Result<Self::Output>;

    /// Run synchronously via the default scheduler — for scripts.
    fn sync(self, log: Arc<Mutex<Vec<String>>>) -> Result<Self::Output> {
        let ctx = make_ctx(log);
        self.execute(&ctx)
    }

    /// Sequential dependency: when self completes, run f on its output.
    fn and_then<F, U>(self, f: F) -> AndThen<Self, F>
    where
        F: FnOnce(Self::Output) -> U + Send,
        U: DeviceOperation,
    {
        AndThen {
            source: self,
            f: Some(f),
        }
    }

    /// Wrap output in Arc for sharing across pipelines / readers.
    fn arc(self) -> Arced<Self> {
        Arced { source: self }
    }
}

// ── Value: lift a host value into the chain ──────────────────────────

pub struct Value<T: Send> {
    v: Option<T>,
}

pub fn value<T: Send>(v: T) -> Value<T> {
    Value { v: Some(v) }
}

impl<T: Send> DeviceOperation for Value<T> {
    type Output = T;
    fn execute(mut self, _ctx: &ExecutionContext) -> Result<T> {
        Ok(self.v.take().expect("Value executed twice"))
    }
}

// ── WithContext: defer construction until ctx is known ───────────────

pub struct WithContext<F> {
    f: Option<F>,
}

pub fn with_context<F, O>(f: F) -> WithContext<F>
where
    F: FnOnce(&ExecutionContext) -> Result<O> + Send,
    O: Send,
{
    WithContext { f: Some(f) }
}

impl<F, O> DeviceOperation for WithContext<F>
where
    F: FnOnce(&ExecutionContext) -> Result<O> + Send,
    O: Send,
{
    type Output = O;
    fn execute(mut self, ctx: &ExecutionContext) -> Result<O> {
        (self.f.take().unwrap())(ctx)
    }
}

// ── AndThen ───────────────────────────────────────────────────────────

pub struct AndThen<S, F> {
    source: S,
    f: Option<F>,
}

impl<S, F, U> DeviceOperation for AndThen<S, F>
where
    S: DeviceOperation,
    F: FnOnce(S::Output) -> U + Send,
    U: DeviceOperation,
{
    type Output = U::Output;
    fn execute(mut self, ctx: &ExecutionContext) -> Result<U::Output> {
        let out = self.source.execute(ctx)?;
        let next = (self.f.take().unwrap())(out);
        next.execute(ctx)
    }
}

// ── Arced: wrap output in Arc ────────────────────────────────────────

pub struct Arced<S> {
    source: S,
}

// SPIKE FINDING #2: `Arc<T>` is only Send when T: Send + Sync. Our
// DeviceOperation::Output must be Send (for tokio::spawn), so
// `.arc()` adds a Sync requirement on the wrapped output. In real
// claspr, DeviceSlice<T> would need to be Send + Sync (it is — opencl3's
// cl_mem is thread-safe per CL spec).
impl<S> DeviceOperation for Arced<S>
where
    S: DeviceOperation,
    S::Output: Sync,
{
    type Output = Arc<S::Output>;
    fn execute(self, ctx: &ExecutionContext) -> Result<Arc<S::Output>> {
        Ok(Arc::new(self.source.execute(ctx)?))
    }
}

// ── AndThenHost: in-queue host computation ───────────────────────────
//
// Pure-combinator emulation of `clEnqueueNativeKernel` (which most
// GPU drivers don't support). The closure is FnOnce(Self::Output) -> Result<U>
// — note: returns a Result<host value>, NOT another DeviceOperation.
// The chain's execute order ensures it runs between the prior and next
// ops without needing the CL native-kernel feature.
//
// Beats split-await in two ways:
//   1. Buffer never escapes the chain as a long-lived host binding
//      (lifetime-bounded to closure scope)
//   2. The pipeline structure is one chain, easier to reason about
//      than awaitN → host → awaitN+1

pub struct AndThenHost<S, F> {
    source: S,
    f: Option<F>,
}

pub trait DeviceOperationHostExt: DeviceOperation {
    fn and_then_host<F, U>(self, f: F) -> AndThenHost<Self, F>
    where
        F: FnOnce(Self::Output) -> Result<U> + Send,
        U: Send,
    {
        AndThenHost {
            source: self,
            f: Some(f),
        }
    }
}
impl<S: DeviceOperation> DeviceOperationHostExt for S {}

impl<S, F, U> DeviceOperation for AndThenHost<S, F>
where
    S: DeviceOperation,
    F: FnOnce(S::Output) -> Result<U> + Send,
    U: Send,
{
    type Output = U;
    fn execute(mut self, ctx: &ExecutionContext) -> Result<U> {
        let prior = self.source.execute(ctx)?;
        (self.f.take().unwrap())(prior)
    }
}

// ── HostAccessible trait — three-stage acquire / host / release ──────
//
// The right shape for "host wants to read/write device data" — see the
// design doc. Per buffer type:
//   - DeviceSlice<T>: acquire = d2h to scratch; release = h2d back
//   - SharedBuffer<T>: acquire = clEnqueueSVMMap; release = clEnqueueSVMUnmap
//   - HostBuffer<T>: acquire/release = no-op (already mapped)
//   - Fine-grain SVM: same — no-op
//
// Splitting acquire/release into chain stages (vs hiding them inside a
// host closure with CL_TRUE blocking) makes them proper queue-ordered
// ops that pipeline with other in-flight work.

pub trait HostAccessible<T>: Sized + Send {
    /// Acquire host access. The returned op, when executed, enqueues
    /// the necessary CL command (d2h, map, or no-op) and yields a
    /// HostView that the host can deref.
    fn acquire_host_view(self) -> AcquireOp<Self, T>;
}

pub trait HostViewable<T>: Sized + Send {
    type DeviceBuf;
    /// Release host access. The returned op enqueues the inverse
    /// command (h2d, unmap, or no-op) and yields the device-side buffer.
    fn release_to_device(self) -> ReleaseOp<Self, T>;
    /// Internal: extract the device buf when the release op executes.
    fn release_inner(self) -> Result<Self::DeviceBuf>;
}

/// Acquire op — a DeviceOperation whose Output is the HostView.
/// For the spike, just synthesizes a HostView from the inner buffer.
pub struct AcquireOp<B, T> {
    buf: Option<B>,
    _phantom: PhantomData<T>,
}

impl<B: HostAccessible<T>, T: Send> DeviceOperation for AcquireOp<B, T> {
    type Output = HostView<B, T>;
    fn execute(mut self, ctx: &ExecutionContext) -> Result<HostView<B, T>> {
        let buf = self.buf.take().expect("acquire executed twice");
        ctx.log("acquire_host_view: enqueued (would be d2h/map/no-op per buffer type)");
        Ok(HostView::new(buf))
    }
}

pub struct HostView<B, T> {
    inner: B,
    _phantom: PhantomData<T>,
}

impl<B, T> HostView<B, T> {
    fn new(inner: B) -> Self {
        HostView {
            inner,
            _phantom: PhantomData,
        }
    }
}

/// Release op — yields the buffer back.
pub struct ReleaseOp<V, T> {
    view: Option<V>,
    _phantom: PhantomData<T>,
}

impl<V, T> DeviceOperation for ReleaseOp<V, T>
where
    V: HostViewable<T>,
    T: Send,
    V::DeviceBuf: Send,
{
    type Output = V::DeviceBuf;
    fn execute(mut self, ctx: &ExecutionContext) -> Result<V::DeviceBuf> {
        let view = self.view.take().expect("release executed twice");
        ctx.log("release_to_device: enqueued (would be h2d/unmap/no-op per buffer type)");
        view.release_inner()
    }
}

// Spike impls — fake everything as a roundtrip from inner buffer.
impl<T> HostViewable<T> for HostView<DeviceSlice<T>, T>
where
    T: Clone + Send + 'static,
{
    type DeviceBuf = DeviceSlice<T>;
    fn release_to_device(self) -> ReleaseOp<Self, T> {
        ReleaseOp {
            view: Some(self),
            _phantom: PhantomData,
        }
    }
    fn release_inner(self) -> Result<DeviceSlice<T>> {
        Ok(self.inner)
    }
}

impl<T: Clone + Send + 'static> HostAccessible<T> for DeviceSlice<T> {
    fn acquire_host_view(self) -> AcquireOp<Self, T> {
        AcquireOp {
            buf: Some(self),
            _phantom: PhantomData,
        }
    }
}

// Allow the HostView to be derefed as &[T] / &mut [T] for the spike.
// Real impl would route to the scratch Vec (DeviceSlice case), the
// mapped region (SVM case), or the persistent map (HostBuffer case).
impl<T> std::ops::Deref for HostView<DeviceSlice<T>, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.inner.data
    }
}
impl<T> std::ops::DerefMut for HostView<DeviceSlice<T>, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.inner.data
    }
}

// ── bundle macros (named bundle! to avoid iterator `zip` connotation
//    and to leave `join!` free for a future spawn/join task API). ─────

pub struct Bundle2<A, B> {
    a: A,
    b: B,
}

pub fn bundle2<A, B>(a: A, b: B) -> Bundle2<A, B>
where
    A: DeviceOperation,
    B: DeviceOperation,
{
    Bundle2 { a, b }
}

impl<A, B> DeviceOperation for Bundle2<A, B>
where
    A: DeviceOperation,
    B: DeviceOperation,
{
    type Output = (A::Output, B::Output);
    fn execute(self, ctx: &ExecutionContext) -> Result<Self::Output> {
        let a = self.a.execute(ctx)?;
        let b = self.b.execute(ctx)?;
        Ok((a, b))
    }
}

pub struct Bundle3<A, B, C> {
    a: A,
    b: B,
    c: C,
}

pub fn bundle3<A, B, C>(a: A, b: B, c: C) -> Bundle3<A, B, C>
where
    A: DeviceOperation,
    B: DeviceOperation,
    C: DeviceOperation,
{
    Bundle3 { a, b, c }
}

impl<A, B, C> DeviceOperation for Bundle3<A, B, C>
where
    A: DeviceOperation,
    B: DeviceOperation,
    C: DeviceOperation,
{
    type Output = (A::Output, B::Output, C::Output);
    fn execute(self, ctx: &ExecutionContext) -> Result<Self::Output> {
        let a = self.a.execute(ctx)?;
        let b = self.b.execute(ctx)?;
        let c = self.c.execute(ctx)?;
        Ok((a, b, c))
    }
}

#[macro_export]
macro_rules! bundle {
    ($a:expr, $b:expr $(,)?) => {
        $crate::bundle2($a, $b)
    };
    ($a:expr, $b:expr, $c:expr $(,)?) => {
        $crate::bundle3($a, $b, $c)
    };
}

// ── FanOut: N-ary parallel from a single Vec ─────────────────────────

/// Apply `f` to each element of `inputs`, returning a Vec of outputs.
/// Models tile-parallel processing.
pub struct FanOut<I, F> {
    inputs: Vec<I>,
    f: Option<F>,
}

pub fn fan_out<I, F, U>(inputs: Vec<I>, f: F) -> FanOut<I, F>
where
    I: Send,
    F: FnMut(I) -> U + Send,
    U: DeviceOperation,
{
    FanOut { inputs, f: Some(f) }
}

impl<I, F, U> DeviceOperation for FanOut<I, F>
where
    I: Send,
    F: FnMut(I) -> U + Send,
    U: DeviceOperation,
{
    type Output = Vec<U::Output>;
    fn execute(mut self, ctx: &ExecutionContext) -> Result<Vec<U::Output>> {
        let mut f = self.f.take().unwrap();
        let mut out = Vec::with_capacity(self.inputs.len());
        for input in self.inputs {
            let op = f(input);
            out.push(op.execute(ctx)?);
        }
        Ok(out)
    }
}

// ── Fake DeviceSlice (the GPU buffer handle) ────────────────────────────
//
// Naming aligns with claspr's existing `DeviceSlice<T>` (typed device
// buffer + element count). The cuda-oxide name `DeviceBox<T>` was an
// earlier spike artifact; corrected to use claspr's vocabulary.

pub struct DeviceSlice<T> {
    pub id: u64,
    pub data: Vec<T>,
    pub device: Device,
}

impl<T: Clone> DeviceSlice<T> {
    pub fn len(self) -> usize {
        self.data.len()
    }
}

unsafe impl<T: Send> Send for DeviceSlice<T> {}
unsafe impl<T: Sync> Sync for DeviceSlice<T> {}

// ── Helper ops: h2d, d2h, zeros ──────────────────────────────────────

pub fn h2d<T: Send + Clone + 'static>(host: Vec<T>) -> impl DeviceOperation<Output = DeviceSlice<T>> {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    with_context(move |ctx| {
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ctx.log(format!(
            "h2d: id={} len={} -> {:?}",
            id,
            host.len(),
            ctx.device()
        ));
        Ok(DeviceSlice {
            id,
            data: host,
            device: ctx.device().clone(),
        })
    })
}

pub fn d2h<T: Send + Clone + 'static>(buf: DeviceSlice<T>) -> impl DeviceOperation<Output = Vec<T>> {
    with_context(move |ctx| {
        ctx.log(format!(
            "d2h: id={} len={} <- {:?}",
            buf.id,
            buf.data.len(),
            buf.device
        ));
        let _ = ctx;
        Ok(buf.data)
    })
}

pub fn zeros<T: Send + Default + Clone + 'static>(
    n: usize,
) -> impl DeviceOperation<Output = DeviceSlice<T>> {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(10_000);
    with_context(move |ctx| {
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ctx.log(format!(
            "zeros: id={} n={} -> {:?}",
            id,
            n,
            ctx.device()
        ));
        Ok(DeviceSlice {
            id,
            data: vec![T::default(); n],
            device: ctx.device().clone(),
        })
    })
}

// ── Fake kernel module (modeled after `#[claspr::device]` output) ────
//
// SPIKE FINDING #1: Rust 2024's `impl Trait` capture rules treat `&self`
// methods as capturing the receiver lifetime in the returned `impl Op`.
// Inside a closure body, returning `self.kernel(arg)` then fails because
// `self` (= the closure-local `kernels` binding) goes out of scope.
//
// Two workarounds:
//   (a) Make the kernel-handle Copy + 'static, take `self` by value.
//       Cheap since the handle is ZST or contains only Arc/refs.
//   (b) Add explicit `+ use<>` precise-capture syntax. Verbose at
//       every method signature.
//
// We go with (a). Real claspr would generate `Kernels: Clone` (it
// holds `Arc<Program>` etc.) and methods take `self` by value (cheap
// Arc clone). Equivalent to cuda-oxide's `Arc<CudaModule>` pattern.

#[derive(Clone, Copy)]
pub struct Kernels;

impl Kernels {
    /// In-place "transform" kernel — takes ownership of buf, mutates,
    /// returns it.
    pub fn transform(self, buf: DeviceSlice<f32>) -> impl DeviceOperation<Output = DeviceSlice<f32>> {
        with_context(move |ctx| {
            ctx.log(format!(
                "kernel.transform: id={} on {:?}",
                buf.id,
                buf.device
            ));
            let mut buf = buf;
            for x in &mut buf.data {
                *x *= 2.0;
            }
            Ok(buf)
        })
    }

    pub fn scale(
        self,
        buf: DeviceSlice<f32>,
        factor: f32,
    ) -> impl DeviceOperation<Output = DeviceSlice<f32>> {
        with_context(move |ctx| {
            ctx.log(format!(
                "kernel.scale: id={} factor={} on {:?}",
                buf.id,
                factor,
                buf.device
            ));
            let mut buf = buf;
            for x in &mut buf.data {
                *x *= factor;
            }
            Ok(buf)
        })
    }

    /// "Process A" — distinct named kernel for fan-out testing.
    pub fn process_a(
        self,
        buf: DeviceSlice<f32>,
    ) -> impl DeviceOperation<Output = DeviceSlice<f32>> {
        with_context(move |ctx| {
            ctx.log(format!("kernel.process_a: id={}", buf.id));
            let mut buf = buf;
            for x in &mut buf.data {
                *x += 1.0;
            }
            Ok(buf)
        })
    }

    pub fn process_b(
        self,
        buf: DeviceSlice<f32>,
    ) -> impl DeviceOperation<Output = DeviceSlice<f32>> {
        with_context(move |ctx| {
            ctx.log(format!("kernel.process_b: id={}", buf.id));
            let mut buf = buf;
            for x in &mut buf.data {
                *x -= 1.0;
            }
            Ok(buf)
        })
    }

    /// Read-only kernel — takes Arc<DeviceSlice> (shared read), writes
    /// into a new buffer.
    pub fn read_shared(
        self,
        shared: Arc<DeviceSlice<f32>>,
        out: DeviceSlice<f32>,
    ) -> impl DeviceOperation<Output = DeviceSlice<f32>> {
        with_context(move |ctx| {
            ctx.log(format!(
                "kernel.read_shared: shared_id={} out_id={}",
                shared.id, out.id
            ));
            let mut out = out;
            for (i, x) in out.data.iter_mut().enumerate() {
                *x = shared.data[i % shared.data.len()] + 100.0;
            }
            Ok(out)
        })
    }

    /// Combine two buffers into a third.
    pub fn combine(
        self,
        a: DeviceSlice<f32>,
        b: DeviceSlice<f32>,
    ) -> impl DeviceOperation<Output = DeviceSlice<f32>> {
        with_context(move |ctx| {
            ctx.log(format!("kernel.combine: a={} b={}", a.id, b.id));
            let mut out = a;
            for (i, x) in out.data.iter_mut().enumerate() {
                *x += b.data[i];
            }
            Ok(out)
        })
    }

    /// Fuse three buffers (multi-producer to single consumer).
    pub fn fuse3(
        self,
        a: DeviceSlice<f32>,
        b: DeviceSlice<f32>,
        c: DeviceSlice<f32>,
    ) -> impl DeviceOperation<Output = DeviceSlice<f32>> {
        with_context(move |ctx| {
            ctx.log(format!("kernel.fuse3: a={} b={} c={}", a.id, b.id, c.id));
            let mut out = a;
            for (i, x) in out.data.iter_mut().enumerate() {
                *x = (*x + b.data[i] + c.data[i]) / 3.0;
            }
            Ok(out)
        })
    }

    /// Combine N tiles (variadic fan-in).
    pub fn combine_tiles(
        self,
        tiles: Vec<DeviceSlice<f32>>,
    ) -> impl DeviceOperation<Output = DeviceSlice<f32>> {
        with_context(move |ctx| {
            ctx.log(format!(
                "kernel.combine_tiles: n_tiles={}",
                tiles.len()
            ));
            // Flatten into one buffer
            let mut data = Vec::new();
            let device = tiles[0].device.clone();
            let id = tiles[0].id;
            for tile in tiles {
                data.extend(tile.data);
            }
            Ok(DeviceSlice { id, data, device })
        })
    }

    /// A kernel that might fail (returns Err on bad input).
    pub fn maybe_fails(
        self,
        buf: DeviceSlice<f32>,
        fail: bool,
    ) -> impl DeviceOperation<Output = DeviceSlice<f32>> {
        with_context(move |ctx| {
            if fail {
                ctx.log(format!("kernel.maybe_fails: id={} FAIL", buf.id));
                Err(Error::from("simulated kernel failure"))
            } else {
                ctx.log(format!("kernel.maybe_fails: id={} ok", buf.id));
                Ok(buf)
            }
        })
    }
}

// ── Sub-buffer modeling for fan-out tiling ───────────────────────────

/// Split a buffer into N equal-sized sub-buffers (for tile-parallel work).
/// Each sub-buffer owns its slice of the data.
fn split_into<T: Clone + Send + 'static>(buf: DeviceSlice<T>, n: usize) -> Vec<DeviceSlice<T>> {
    let chunk_size = buf.data.len() / n;
    let mut chunks = Vec::with_capacity(n);
    let mut data_iter = buf.data.into_iter();
    for i in 0..n {
        let chunk: Vec<T> = (&mut data_iter).take(chunk_size).collect();
        chunks.push(DeviceSlice {
            id: buf.id * 100 + i as u64,
            data: chunk,
            device: buf.device.clone(),
        });
    }
    chunks
}

// ── Cross-device transfer (fake) ─────────────────────────────────────
//
// Takes `&Device` (the target). Real claspr would use the shared
// `cl_context` spanning both devices + `clEnqueueMigrateMemObjects`
// (or `clEnqueueCopyBuffer` between buffers on different queues). With
// sub-devices, the target may itself be a partition of a parent —
// `Device` is one type covering both cases.

pub fn transfer_to_device<T: Send + Clone + 'static>(
    buf: DeviceSlice<T>,
    target_device: &Device,
) -> impl DeviceOperation<Output = DeviceSlice<T>> + use<T> {
    let target_device = target_device.clone();
    with_context(move |ctx| {
        ctx.log(format!(
            "transfer: id={} from {:?} to {:?}",
            buf.id, buf.device, target_device
        ));
        Ok(DeviceSlice {
            id: buf.id,
            data: buf.data,
            device: target_device,
        })
    })
}

// ── Profiling via callback (CORRECTED from earlier wrapped-Output) ───
//
// The earlier `.profiled() -> Profiled<T>` shape was wrong: in real
// async claspr, by the time the next chain stage runs, the cl_event
// hasn't fired yet, so clGetEventProfilingInfo would return
// CL_PROFILING_INFO_NOT_AVAILABLE.
//
// Correct shape: `.profiled(|info| ...)` registers a callback via
// clSetEventCallback(event, CL_COMPLETE, thunk). The callback fires
// when the GPU finishes; the user closure runs on a CL driver thread
// and receives the timestamps. Chain doesn't block; Output type stays
// unchanged (profiling is side-effect, not data flow).
//
// FFI safety in real impl:
//   - User closure boxed as FnOnce(ProfilingInfo) + Send + 'static
//   - Thunk uses catch_unwind to prevent panics across FFI
//   - Errors from clGetEventProfilingInfo logged, not propagated

#[derive(Debug, Clone, Copy)]
pub struct ProfilingInfo {
    pub queued_ns: u64,
    pub submit_ns: u64,
    pub start_ns: u64,
    pub end_ns: u64,
}

impl ProfilingInfo {
    pub fn duration_ns(&self) -> u64 {
        self.end_ns - self.start_ns
    }
}

pub struct WithProfile<S, F> {
    source: S,
    callback: Option<F>,
}

pub trait DeviceOperationProfileExt: DeviceOperation {
    fn profiled<F>(self, callback: F) -> WithProfile<Self, F>
    where
        F: FnOnce(ProfilingInfo) + Send + 'static,
    {
        WithProfile {
            source: self,
            callback: Some(callback),
        }
    }
}
impl<S: DeviceOperation> DeviceOperationProfileExt for S {}

impl<S, F> DeviceOperation for WithProfile<S, F>
where
    S: DeviceOperation,
    F: FnOnce(ProfilingInfo) + Send + 'static,
{
    type Output = S::Output;   // unchanged — profiling is side-effect
    fn execute(mut self, ctx: &ExecutionContext) -> Result<S::Output> {
        // Spike: run the inner op synchronously, then synthesize fake
        // ProfilingInfo and call the user closure immediately. In real
        // impl: register the closure as a clSetEventCallback so it
        // fires on the CL driver thread when the kernel completes.
        let value = self.source.execute(ctx)?;
        let info = ProfilingInfo {
            queued_ns: 100_000,
            submit_ns: 200_000,
            start_ns: 1_000_000,
            end_ns: 2_000_000,
        };
        if let Some(cb) = self.callback.take() {
            cb(info);   // In real impl: would happen on CL driver thread later
        }
        Ok(value)
    }
}

// ── Future bridge ────────────────────────────────────────────────────

pub struct DeviceFuture<O: DeviceOperation> {
    op: Option<O>,
    log: Arc<Mutex<Vec<String>>>,
    _pin: PhantomData<*const ()>, // not Unpin — fake
}

impl<O: DeviceOperation> Future for DeviceFuture<O>
where
    O::Output: Unpin,
{
    type Output = Result<O::Output>;
    fn poll(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        // Spike: execute synchronously on first poll
        let this = unsafe { self.get_unchecked_mut() };
        let op = this.op.take().expect("polled after completion");
        let log = this.log.clone();
        Poll::Ready(op.sync(log))
    }
}

// Helper to await an op with a known log handle
fn await_op<O: DeviceOperation>(op: O, log: Arc<Mutex<Vec<String>>>) -> Result<O::Output>
where
    O::Output: Unpin,
{
    let fut = DeviceFuture {
        op: Some(op),
        log,
        _pin: PhantomData,
    };
    pollster::block_on(fut)
}

// ═════════════════════════════════════════════════════════════════════
// SCENARIOS
// ═════════════════════════════════════════════════════════════════════

fn scenario_1_linear_chain(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 1: linear chain (producer/consumer) ===");
    let kernels = Kernels;
    let host_input = vec![1.0f32; 8];
    let pipeline = h2d(host_input)
        .and_then(move |buf| kernels.transform(buf))
        .and_then(|buf| {
            let kernels = Kernels;
            kernels.scale(buf, 0.5)
        })
        .and_then(d2h);
    let result = pipeline.sync(log)?;
    println!("  result[0..4] = {:?}", &result[..4]);
    assert_eq!(result[0], 1.0); // 1.0 * 2.0 * 0.5
    Ok(())
}

fn scenario_2_bundle_parallel(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 2: independent parallel branches via bundle! ===");
    let pipeline = bundle!(
        h2d(vec![1.0f32; 4]).and_then(|buf| Kernels.process_a(buf)),
        h2d(vec![10.0f32; 4]).and_then(|buf| Kernels.process_b(buf)),
    );
    let (a_buf, b_buf) = pipeline.sync(log)?;
    println!("  a[0]={} b[0]={}", a_buf.data[0], b_buf.data[0]);
    assert_eq!(a_buf.data[0], 2.0); // 1 + 1
    assert_eq!(b_buf.data[0], 9.0); // 10 - 1
    Ok(())
}

fn scenario_3_diamond(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 3: diamond (fan-out + fan-in via Arc) ===");
    // Load shared input once. Two independent kernels read it. Combine outputs.
    let pipeline = h2d(vec![5.0f32; 4]).arc().and_then(|shared| {
        // Now `shared: Arc<DeviceSlice<f32>>`. Clone into both branches.
        let s1 = shared.clone();
        let s2 = shared.clone();
        bundle!(
            zeros::<f32>(4).and_then(move |out| Kernels.read_shared(s1, out)),
            zeros::<f32>(4).and_then(move |out| Kernels.read_shared(s2, out)),
        )
        .and_then(|(a, b)| Kernels.combine(a, b))
        .and_then(d2h)
    });
    let result = pipeline.sync(log)?;
    println!("  combined[0..4] = {:?}", &result[..4]);
    Ok(())
}

fn scenario_4_ml_forward_pass(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 4: ML-pass-style multi-stage with state ===");
    // GEMM-shaped pipeline: 3 buffers allocated up front, threaded through
    // 3 kernel stages.
    let w0 = vec![0.1f32; 16];
    let w1 = vec![0.2f32; 16];

    let weights = bundle!(h2d(w0).arc(), h2d(w1).arc()).sync(log.clone())?;
    let (w0, w1) = weights;

    let pipeline = bundle!(h2d(vec![1.0f32; 16]), zeros::<f32>(16), zeros::<f32>(16))
        .and_then(move |(input, hidden, output)| {
            // Stage 1: input -> hidden via w0
            let w0 = w0.clone();
            Kernels.read_shared(w0, hidden).and_then(move |hidden| {
                value((input, hidden, output)) // carry forward
            })
        })
        .and_then(move |(input, hidden, output)| {
            // Stage 2: hidden -> output via w1
            let w1 = w1.clone();
            Kernels.read_shared(w1, output).and_then(move |output| {
                value((input, hidden, output))
            })
        })
        .and_then(|(_input, _hidden, output)| {
            // Stage 3: download just the final output
            d2h(output)
        });
    let result = pipeline.sync(log)?;
    println!("  output[0..4] = {:?}", &result[..4]);
    Ok(())
}

fn scenario_5_in_place_mutation(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 5: in-place mutation chain ===");
    // Each kernel takes ownership of buf, mutates it, returns it.
    let pipeline = h2d(vec![1.0f32; 4])
        .and_then(|buf| Kernels.transform(buf)) // *2
        .and_then(|buf| Kernels.transform(buf)) // *2
        .and_then(|buf| Kernels.scale(buf, 0.25))
        .and_then(d2h);
    let result = pipeline.sync(log)?;
    println!("  result = {:?}", result);
    assert_eq!(result[0], 1.0); // 1 * 2 * 2 * 0.25
    Ok(())
}

fn scenario_6_n_ary_fan_out(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 6: N-ary fan-out (tile-parallel) ===");
    // Split buf into 4 sub-buffers, process each independently, combine.
    let pipeline = h2d(vec![1.0f32; 16])
        .and_then(|buf| {
            let tiles = split_into(buf, 4);
            fan_out(tiles, |tile| Kernels.transform(tile))
        })
        .and_then(|tiles| Kernels.combine_tiles(tiles))
        .and_then(d2h);
    let result = pipeline.sync(log)?;
    println!("  combined len = {}", result.len());
    assert_eq!(result.len(), 16);
    assert_eq!(result[0], 2.0); // 1 * 2
    Ok(())
}

fn scenario_7_multi_producer_single_consumer(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 7: multi-producer, single consumer ===");
    let pipeline = bundle!(
        h2d(vec![1.0f32; 4]),
        h2d(vec![10.0f32; 4]),
        h2d(vec![100.0f32; 4]),
    )
    .and_then(|(a, b, c)| Kernels.fuse3(a, b, c))
    .and_then(d2h);
    let result = pipeline.sync(log)?;
    println!("  fused = {:?}", result);
    Ok(())
}

fn scenario_8_split_await(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 8: mixed sync/async (split await with host work) ===");
    // Build the first part, await it, do host work, then build a second part.
    let part1 = h2d(vec![1.0f32; 4]).and_then(|buf| Kernels.transform(buf));
    let buf_a = await_op(part1, log.clone())?;
    println!("  intermediate buf_a[0] = {}", buf_a.data[0]);

    // Host-only work between the two await points.
    let host_factor = 3.0f32;

    let part2 = Kernels.scale(buf_a, host_factor).and_then(d2h);
    let result = await_op(part2, log)?;
    println!("  final = {:?}", result);
    assert_eq!(result[0], 6.0); // 1 * 2 * 3
    Ok(())
}

fn scenario_9_conditional_graph(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 9: conditional graph shape (dynamic DAG) ===");
    // The graph topology depends on runtime data. Demonstrates the
    // boxing-required-for-type-erasure tax.
    let use_expensive = true;

    let pipeline = h2d(vec![1.0f32; 4]).and_then(move |buf| {
        // Both branches must have the same output type.
        // Without boxing, the closure return type would differ — Rust
        // can't unify `AndThen<impl Op, ...>` with itself parameterised
        // differently.
        if use_expensive {
            // Box::new(...) as Box<dyn ...> — type erasure cost
            DynOp::new(Kernels.transform(buf).and_then(|buf| Kernels.transform(buf)))
        } else {
            DynOp::new(Kernels.scale(buf, 4.0))
        }
    });
    let result = pipeline.and_then(d2h).sync(log)?;
    println!("  result = {:?}", result);
    Ok(())
}

/// Type-erasing wrapper for DeviceOperations with the same Output type.
/// Pays a Box allocation per conditional branch.
pub struct DynOp<O> {
    inner: Box<dyn DynDeviceOperation<Output = O> + Send>,
}

impl<O: Send> DynOp<O> {
    pub fn new<S>(op: S) -> Self
    where
        S: DeviceOperation<Output = O> + 'static,
    {
        Self {
            inner: Box::new(Some(op)),
        }
    }
}

trait DynDeviceOperation: Send {
    type Output: Send;
    fn execute_dyn(&mut self, ctx: &ExecutionContext) -> Result<Self::Output>;
}

impl<S: DeviceOperation + 'static> DynDeviceOperation for Option<S> {
    type Output = S::Output;
    fn execute_dyn(&mut self, ctx: &ExecutionContext) -> Result<S::Output> {
        let op = self.take().expect("DynOp executed twice");
        op.execute(ctx)
    }
}

impl<O: Send> DeviceOperation for DynOp<O> {
    type Output = O;
    fn execute(mut self, ctx: &ExecutionContext) -> Result<O> {
        self.inner.execute_dyn(ctx)
    }
}

fn scenario_10_error_propagation(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 10: error propagation through and_then ===");
    let pipeline = h2d(vec![1.0f32; 4])
        .and_then(|buf| Kernels.transform(buf))
        .and_then(|buf| Kernels.maybe_fails(buf, true)) // ← fails here
        .and_then(|buf| {
            println!("  this stage SHOULD NOT RUN");
            Kernels.transform(buf)
        })
        .and_then(d2h);

    match pipeline.sync(log) {
        Ok(_) => panic!("should have errored"),
        Err(e) => println!("  got expected error: {}", e),
    }
    Ok(())
}

fn scenario_11_buffer_round_trip(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 11: buffer round-trip (pass in, get back) ===");
    let buf = await_op(h2d(vec![1.0f32; 4]), log.clone())?;

    // Run a pipeline that ends with returning the buffer.
    let pipeline = value(buf)
        .and_then(|buf| Kernels.transform(buf))
        .and_then(|buf| Kernels.scale(buf, 3.0));
    let buf_back = pipeline.sync(log.clone())?;
    println!("  buf_back[0] = {}", buf_back.data[0]);
    assert_eq!(buf_back.data[0], 6.0); // 1 * 2 * 3

    // Reuse buf for another pipeline.
    let result = value(buf_back).and_then(d2h).sync(log)?;
    println!("  final reused = {:?}", result);
    Ok(())
}

fn scenario_12_profiling(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 12: profiling via callback (.profiled(|info| ...)) ===");
    // The callback fires when the kernel's event completes. Output type
    // of `.profiled(...)` is unchanged from the inner op (DeviceSlice<f32>);
    // profiling is purely a side-effect. Chain doesn't block.
    let pipeline = h2d(vec![1.0f32; 4])
        .and_then(|buf| {
            Kernels.transform(buf).profiled(|info| {
                eprintln!("  transform took {} ns", info.duration_ns());
            })
        })
        .and_then(|buf| Kernels.scale(buf, 0.5))
        .and_then(d2h);
    let result = pipeline.sync(log)?;
    println!("  result = {:?}", result);
    Ok(())
}

fn scenario_13_batch_parallelism(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 13: batch parallelism via fan_out + marker (no spawn) ===");
    // Load shared weights once.
    let weights = await_op(h2d(vec![0.5f32; 4]).arc(), log.clone())?;

    // fan_out builds N independent pipelines (one per batch index) and
    // bundles them into a single op. The "marker" pattern is implicit:
    // when execute() runs, it submits all sub-ops to the OOO queue with
    // no inter-batch deps, then conceptually inserts a marker to wait on
    // all events. The .sync() blocks on that marker.
    //
    // No tokio. No spawn. No per-batch JoinHandle. Just one chain. The
    // OOO queue + event scheduling does the cross-batch parallelism on
    // the hardware. This is the meaningful improvement over cuda-oxide,
    // which forces tokio::spawn for this pattern because their chains
    // lock to a single stream.
    let pipeline = fan_out((0..3).collect(), move |batch_idx: i32| {
        let w = weights.clone();
        h2d(vec![batch_idx as f32; 4])
            .and_then(move |buf| {
                let w = w.clone();
                zeros::<f32>(4)
                    .and_then(move |out| Kernels.read_shared(w, out))
                    .and_then(move |out| Kernels.combine(buf, out))
            })
            .and_then(d2h)
    });
    let results: Vec<Vec<f32>> = pipeline.sync(log)?;
    for (i, h) in results.iter().enumerate() {
        println!("  batch {}: {:?}", i, &h[..4]);
    }
    Ok(())
}

fn scenario_14_cross_device(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 14: cross-device pipeline ===");
    // Two named Device handles. In real claspr these would come from
    // Context::for_devices(&[dev_a, dev_b])? — a single cl_context
    // spanning both devices, with sub-device support if needed.
    let dev_a = Device::new("dev_a");
    let dev_b = Device::new("dev_b");

    // Note: the fake ExecutionContext always reports the "default" device,
    // but the buffer carries its own Device handle through, and
    // transfer_to_device sets the target.
    let dev_a_for_pipeline = dev_a.clone();
    let pipeline = h2d(vec![1.0f32; 4])
        .and_then(|buf| Kernels.transform(buf))
        .and_then(move |buf| transfer_to_device(buf, &dev_b))
        .and_then(|buf| {
            println!("  buf is now on {:?}", buf.device);
            Kernels.scale(buf, 10.0)
        })
        .and_then(move |buf| transfer_to_device(buf, &dev_a_for_pipeline))
        .and_then(d2h);
    let result = pipeline.sync(log)?;
    let _ = dev_a; // keep alive
    println!("  result = {:?}", result);
    Ok(())
}

fn scenario_15_and_then_host(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 15: .and_then_host(|x| ...) — in-queue host work ===");
    // Pure-combinator emulation of clEnqueueNativeKernel. Host closure
    // runs between GPU stages without split-await. Buffer flows through
    // the closure without ever escaping into a long-lived host binding.
    let pipeline = h2d(vec![1.0f32; 4])
        .and_then(|buf| Kernels.transform(buf))           // GPU: *2 = [2;4]
        .and_then_host(|buf| {
            // Host: do some computation based on the GPU result.
            // Note: we cheat in the spike by reading buf.data; real
            // would acquire_host_view first (see scenario 16).
            let factor = if buf.data[0] > 1.5 { 3.0 } else { 1.0 };
            eprintln!("  host computed factor = {}", factor);
            Ok((buf, factor))
        })
        .and_then(|(buf, factor)| Kernels.scale(buf, factor))   // GPU: *3 = [6;4]
        .and_then(d2h);
    let result = pipeline.sync(log)?;
    println!("  result = {:?}", result);
    assert_eq!(result[0], 6.0); // 1 * 2 * 3
    Ok(())
}

fn scenario_16_host_accessible(log: Arc<Mutex<Vec<String>>>) -> Result<()> {
    println!("\n=== Scenario 16: HostAccessible — acquire/host/release three-stage ===");
    // The clean shape for host code that needs to access device data:
    //   1. acquire_host_view  → enqueues d2h (or map, or no-op)
    //   2. and_then_host       → host work via DerefMut on the view
    //   3. release_to_device   → enqueues h2d (or unmap, or no-op)
    //
    // For DeviceSlice: d2h + h2d (heavy)
    // For SharedBuffer (coarse SVM): clEnqueueSVMMap + Unmap
    // For HostBuffer / fine-grain SVM: no-op (already accessible)
    //
    // Splitting into three chain stages (vs hiding inside one closure)
    // means acquire/release are real, queue-ordered ops that can pipeline
    // with other in-flight work.
    let pipeline = h2d(vec![1.0f32; 4])
        .and_then(|buf| Kernels.transform(buf))      // GPU work, [2;4]
        .and_then(|buf| buf.acquire_host_view())     // → HostView<DeviceSlice<f32>, f32>
        .and_then_host(|mut view| {
            // view: HostView<DeviceSlice<f32>, f32> — DerefMut to [f32]
            view[0] += 100.0;
            eprintln!("  host modified view[0] = {}", view[0]);
            Ok(view)
        })
        .and_then(|view| view.release_to_device())    // → DeviceSlice<f32>
        .and_then(|buf| Kernels.scale(buf, 0.5))     // GPU again
        .and_then(d2h);
    let result = pipeline.sync(log)?;
    println!("  result = {:?}", result);
    // (2.0 + 100.0) * 0.5 = 51.0 for index 0; (2.0) * 0.5 = 1.0 for others
    assert_eq!(result[0], 51.0);
    assert_eq!(result[1], 1.0);
    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
// MAIN
// ═════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let log = Arc::new(Mutex::new(Vec::new()));

    scenario_1_linear_chain(log.clone())?;
    scenario_2_bundle_parallel(log.clone())?;
    scenario_3_diamond(log.clone())?;
    scenario_4_ml_forward_pass(log.clone())?;
    scenario_5_in_place_mutation(log.clone())?;
    scenario_6_n_ary_fan_out(log.clone())?;
    scenario_7_multi_producer_single_consumer(log.clone())?;
    scenario_8_split_await(log.clone())?;
    scenario_9_conditional_graph(log.clone())?;
    scenario_10_error_propagation(log.clone())?;
    scenario_11_buffer_round_trip(log.clone())?;
    scenario_12_profiling(log.clone())?;
    scenario_13_batch_parallelism(log.clone())?;
    scenario_14_cross_device(log.clone())?;
    scenario_15_and_then_host(log.clone())?;
    scenario_16_host_accessible(log.clone())?;

    println!("\n=== ALL SCENARIOS PASSED ===");
    println!("Log entries: {}", log.lock().unwrap().len());
    Ok(())
}
