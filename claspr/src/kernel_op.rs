//! [`KernelOp`] — the host-side enqueue contract emitted by
//! [`claspr_macros::kernel`][k] on every generated kernel `Op`.
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
//!
//! [k]: claspr_macros::kernel

use crate::queue::Launcher;
use crate::{Event, Result, cl_event};

/// Enqueue contract for proc-macro-generated kernel Ops.
///
/// Implementations are emitted by `#[claspr::kernel]` on its hidden
/// per-kernel `Op` struct. The two terminals on `Op` (`.submit`,
/// `.wait`) and the blanket `DeviceOperation` impl in `claspr-async`
/// both go through [`enqueue_into`][Self::enqueue_into], so there is
/// exactly one enqueue body per kernel.
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
pub trait KernelOp: Send + Sized {
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
