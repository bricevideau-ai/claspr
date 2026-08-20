//! Compile-fail surface for claspr's unified `DeviceOp` API, driven by
//! [`ui_test`] via the shared harness in `claspr_test_support::ui` (rlib
//! discovery, rustc wiring, and the trybuild-vs-ui_test rationale live
//! there).
//!
//! After the Tier-1/Tier-2 reunification there is one `DeviceOp` trait and
//! the old closure layer is gone — but a couple of type-system safety
//! invariants must survive the fold. Each fixture in
//! `tests/tier2/compile_fail/` deliberately violates one and is expected to
//! fail to compile; the captured stderr is diffed against the golden
//! `.stderr` files committed alongside.
//!
//! Coverage:
//!
//! - `fill_on_frozen` — `DeviceSlice::fill` (now a `DeviceOp`-returning verb)
//!   requires `M: Fillable`; `Frozen` isn't `Fillable`, so the fill must be
//!   rejected. Restatement of the deleted `buffer_ops_fill_on_frozen`.
//! - `arc_to_writable_arg` — `Arc<DeviceSlice<T, M>>` impls only
//!   `KernelSliceReadArg`, so a writable kernel slot must reject it.
//!   Restatement of the deleted `arc_to_writable_arg`.
//! - `buffer_ops_write_on_host_read_only` — `DeviceSlice::write` requires
//!   `M: HostWritable`; `HostReadOnly` isn't, so the write must be rejected.
//! - `bundle_aliased_owned_writes` — two `bundle!` arms can't both move-write
//!   the same owned buffer (the second arm is a use-after-move).
//! - `sequential_use_after_move` — an `.and_then` follow-up can't reach a
//!   moved outer `buf` instead of the upstream handle it is handed.
//! - `and_then_host_escapes_buffer` — the mapped view can't escape the
//!   `and_then_host` closure (HRTB `for<'a> FnOnce(View<'a>) -> Result<()>`).
//! - `host_view_outlives_release` — use-after-move on
//!   `DeviceSliceHostView::release_to_device(self)`.
//! - `scalar_slot_fed_pipe` — a SCALAR slot tag (`F: f32`) fed a `Pipe`
//!   (`F(pipe)`) must be rejected: the unified `Tag(pipe)` pipe-feed `CallArg`
//!   arm is gated to buffer-valued tags (`RecordableBuffer`), so `F<Pipe<f32>>`
//!   has no `CallArg`. Guards the buffer/scalar asymmetry of the feed unify.
//!
//! A second `compile_pass` config (`Mode::Pass`) runs `compile_pass/sanity.rs`:
//! a known-good unified-API `DeviceOp` chain that MUST compile. It is the
//! harness-integrity guard — if the `--extern`/`-L` wiring were misconfigured,
//! every compile-fail fixture would fail for the wrong reason and the suite
//! would pass spuriously; the pass fixture catches that.
//!
//! ## Running and re-blessing
//!
//! ```text
//! cargo test -p claspr-tier2-tests --test safety_compile_fail
//! cargo test -p claspr-tier2-tests --test safety_compile_fail -- --bless
//! ```
//!
//! No OpenCL device needed — these are pure compile-time checks.

use claspr_test_support::ui::{Mode, Result, run_compile_tests};

fn main() -> Result<()> {
    run_compile_tests(
        &["claspr", "claspr_test_kernels"],
        "cargo test -p claspr-tier2-tests --test safety_compile_fail -- --bless",
        &[("compile_fail", Mode::Fail), ("compile_pass", Mode::Pass)],
    )
}
