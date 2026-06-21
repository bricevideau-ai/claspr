//! claspr-async — **compatibility shim.**
//!
//! The Tier 2 combinator layer (`DeviceOperation`, `and_then`, `bundle`,
//! `fan_out`, the eager struct-graph core, …) has been folded into [`claspr`]
//! itself, so the proc-macro-emitted ops and the graph traits are co-located
//! (the macro emits `::claspr::` paths and can't name a separate crate's
//! types — see `NOTES.md`).
//!
//! This crate now just re-exports `claspr` so existing `claspr_async::…` paths
//! keep resolving. **New code should depend on `claspr` directly.**

pub use claspr::*;

// Glob re-export doesn't carry `#[macro_export]` macros — re-export them by
// path so `claspr_async::{upload, download, slots, …}` keep working.
pub use claspr::{
    device_slice, device_slice_alloc_uninit, device_slice_alloc_zero, device_slice_filled,
    device_slice_from_slice, download, mapped_slice, mapped_slice_alloc_uninit,
    mapped_slice_alloc_zero, mapped_slice_filled, mapped_slice_from_slice, mapped_slice_upload,
    upload, usm_slice, usm_slice_alloc_uninit, usm_slice_alloc_zero,
};
