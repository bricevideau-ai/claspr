//! claspr-async — the Tier 2 combinator layer on top of claspr.
//!
//! Where [claspr]'s [`LaunchOp`] surfaces one explicit queue per call
//! (Tier 1: `.wait()` / `.submit()` / `.await`), claspr-async lets you
//! compose lazy [`DeviceOperation`]s into a single chain that runs
//! end-to-end, with the per-device default out-of-order queue picked
//! automatically and event dependencies threaded through behind the
//! scenes.
//!
//! ## At a glance
//!
//! ```ignore
//! use claspr::Context;
//! use claspr_async::{DeviceOperation, value, with_context};
//!
//! let ctx = Context::any()?;
//!
//! // A chain: lift a Vec to device, run a kernel, then download.
//! let result: Vec<u32> = value(input_vec)
//!     .and_then(|v| with_context(move |c| Ok(claspr::DeviceSlice::upload(c, &v)?)))
//!     .and_then(|buf| with_context(move |c| {
//!         kernels.foo_op(c, [N], &buf)?;  // proc-macro-emitted Tier 2 op
//!         Ok(buf)
//!     }))
//!     .and_then(|buf| with_context(move |c| {
//!         let mut out = vec![0u32; buf.len()];
//!         Ok(buf.download(c)?)
//!         Ok(out)
//!     }))
//!     .sync(&ctx)?;
//! ```
//!
//! The proc-macro-emitted Tier 2 wrappers (landing in Phase 4) will
//! reduce that to a single `.foo_op(...).and_then(...)` chain.
//!
//! ## Crate layout (mirrors [`IMPLEMENTATION-PLAN.md`])
//!
//! - [`op`] — [`DeviceOperation`] trait + the core combinators
//!   ([`AndThen`], [`Arced`], [`Value`], [`WithContext`]).
//! - [`exec_ctx`] — [`ExecutionContext`] (passed to each op's
//!   `execute`; implements [`claspr::Launcher`] so existing Tier 1
//!   ops compose into the chain).
//!
//! Later phases add: `bundle` / `fan_out` / `arc` / `future` /
//! `and_then_host` / `host_view` / `profile`.
//!
//! [claspr]: https://docs.rs/claspr
//! [`LaunchOp`]: claspr::LaunchOp

pub mod and_then_host;
pub mod arc;
pub mod bundle;
pub mod exec_ctx;
pub mod fan_out;
pub mod future;
pub mod host_view;
pub mod op;
pub mod profile;
pub mod transfer;

pub use and_then_host::{AndThenHost, DeviceOperationHostExt};
pub use arc::ArcSplit;
pub use bundle::{
    Bundle2, Bundle3, Bundle4, Bundle5, Bundle6, Bundle7, Bundle8, Bundle9, Bundle10, Bundle11,
    Bundle12, Bundle13, Bundle14, Bundle15, Bundle16,
};
pub use exec_ctx::ExecutionContext;
pub use fan_out::{FanOut, fan_out};
pub use future::ChainFuture;
pub use host_view::{
    AcquireDeviceSliceOp, AcquireHostBufferOp, AcquireSharedBufferOp, DeviceSliceHostView,
    HostAccessibleExt, HostBufferHostView, ReleaseDeviceSliceOp, ReleaseHostBufferOp,
    ReleaseSharedBufferOp, SharedBufferHostView,
};
pub use op::{AndThen, Arced, DeviceOperation, Value, WithContext, value, with_context};
pub use profile::{DeviceOperationProfileExt, Profiled};
pub use transfer::{Download, Upload, UploadSource, download, upload};
