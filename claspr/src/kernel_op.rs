//! [`KernelOp`] — the host-side enqueue contract emitted by
//! [`claspr_macros::kernel`] on every generated kernel `Op`.
//!
//! This trait exists so the proc-macro can stay free of any
//! `claspr-async` reference. The macro emits one inherent terminal
//! (`.submit`) and one [`KernelOp`] impl; nothing else. Tier 2 chain
//! composition is then added downstream by `claspr-async`'s blanket
//! `impl<O: KernelOp + 'static> DeviceOperation for O`, which is the
//! single place that knows about `Deps` / `Event` wrapping.
//!
//! Concretely:
//!
//! - Tier 1-only consumers depend on `claspr` and never compile
//!   `claspr-async`. They use `.wait(&ctx)` / `.submit(&ctx)`.
//! - Tier 2 consumers add `claspr-async` to their own `Cargo.toml`.
//!   Any `O: KernelOp` automatically composes via `and_then` /
//!   `bundle!` / `fan_out` — no per-kernel feature flag, no
//!   `#[cfg]`-gated macro emission.
//!
//! See `claspr-macros/src/lib.rs` for the emission site.

use crate::queue::Launcher;
use crate::{Event, Result, cl_event};

/// Sealed-trait marker for [`KernelOp`]. The
/// [`#[claspr::kernel]`](claspr_macros::kernel) proc-macro emits an
/// `impl ::claspr::__seal::Sealed for Op {}` alongside the
/// `KernelOp` impl on each generated Op struct; the supertrait
/// reference on [`KernelOp`] is the seal.
///
/// This module is `#[doc(hidden)]` and is not part of the crate's
/// stable surface. Code outside the proc-macro should not impl
/// [`Sealed`][__seal::Sealed]; if a real external use case for
/// extending [`KernelOp`] appears, open an issue so the invariants
/// in [`enqueue_into`][KernelOp::enqueue_into] can be documented
/// and the trait properly unsealed in the same release.
#[doc(hidden)]
pub mod __seal {
    /// Sealed-trait witness. See module docs.
    pub trait Sealed {}
}

/// Enqueue contract for proc-macro-generated kernel Ops.
///
/// **Sealed.** Implementations are emitted only by
/// `#[claspr::kernel]` on its hidden per-kernel `Op` struct — adding
/// a `KernelOp` impl from outside the proc-macro will fail because
/// the [`__seal::Sealed`] supertrait is not implementable there.
/// This is deliberate: a wrong `KernelOp` impl silently corrupts
/// event-graph dependencies and `Arc<DeviceSlice>` Drop ordering
/// (the `last_use` registration that gates `clEnqueueSVMFree` /
/// `clReleaseMemObject` against in-flight enqueues), and the
/// integration is one-place enough that hand-rolled impls aren't
/// load-bearing today.
///
/// The two terminals on `Op` (`.submit`, `.wait`) and the blanket
/// `DeviceOperation` impl in `claspr-async` both go through
/// [`enqueue_into`][Self::enqueue_into], so there is exactly one
/// enqueue body per kernel.
///
/// `extra_deps` is a slice of raw `cl_event` handles that must be
/// merged into the kernel launch's wait list **on top of** the deps
/// the Op already carries (caller-added via `.after()` /
/// `.after_all()`). The async path uses it to inject chain-supplied
/// upstream events. Raw `cl_event` is intentional — both producers
/// (owned `Event`s and `Arc<Event>`s) flatten to the same OpenCL
/// handle type, and OpenCL retains the events internally during
/// enqueue, so the wrappers only need to outlive the call.
///
/// The `'static` bound on the impl side is what lets `claspr-async`
/// move Ops into futures and across executor threads; the trait
/// itself does not require it, so Tier 1-only consumers aren't
/// burdened.
///
/// ## The seal in action
///
/// Trying to impl [`KernelOp`] from outside the proc-macro fails
/// to compile because [`__seal::Sealed`] isn't satisfied:
///
/// ```compile_fail
/// use claspr::{KernelOp, Launcher, Event, Result, cl_event};
///
/// struct MyOp;
/// impl KernelOp for MyOp {
///     type Output = ();
///     fn enqueue_into<L: Launcher>(
///         self,
///         _launcher: &L,
///         _extra_deps: &[cl_event],
///     ) -> Result<((), Event)> {
///         unimplemented!()
///     }
/// }
/// ```
pub trait KernelOp: __seal::Sealed + Send + Sized {
    /// Slice arg(s) the kernel touches — bare for one slice, tuple
    /// for many, `()` for none. Mirrors the macro's existing
    /// `Output` shape on `Op`.
    type Output: Send;

    /// Enqueue this Op on `launcher`'s queue, merging `extra_deps`
    /// into the wait list, and return `(Output, completion Event)`.
    ///
    /// Implementations:
    ///
    /// - Validate caller-added deps against `launcher`'s context.
    /// - Set kernel args, call `clEnqueueNDRangeKernel`, register
    ///   any profiling callback.
    /// - Drop their owned dep events and the kernel handle *after*
    ///   enqueue returns (OpenCL has retained both internally by
    ///   then).
    fn enqueue_into<L: Launcher>(
        self,
        launcher: &L,
        extra_deps: &[cl_event],
    ) -> Result<(Self::Output, Event)>;
}
