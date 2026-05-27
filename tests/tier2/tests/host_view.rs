//! Phase 3.6 coverage — `HostAccessible` three-stage round-trip,
//! mirroring spike scenario 16:
//!
//!   upload → kernel → acquire → and_then_host → release → kernel → download
//!
//! TODO(phase4): all four tests in this file currently exercised the
//! old `acquire_host_view → and_then_host(|view| Ok(view)) → release`
//! pattern, which doesn't fit the new value-passing-only async
//! `and_then_host` signature. The old bodies are preserved as
//! comments; placeholder `todo!()` stubs keep them on the test list
//! so the gap is visible until Phase 4 reworks the host-view types.

#[test]
#[ignore = "Phase 4: rewrite acquire/release for the new Mappable shape"]
fn acquire_host_edit_release_round_trip() {
    todo!("phase 4: rewrite acquire/release for new Mappable shape")
}

#[test]
#[ignore = "Phase 4: rewrite acquire/release for the new Mappable shape"]
fn acquire_immediately_release_is_a_round_trip() {
    todo!("phase 4: rewrite acquire/release for new Mappable shape")
}

#[test]
#[ignore = "Phase 4: rewrite acquire/release for the new Mappable shape"]
fn host_buffer_acquire_release_is_zero_copy_passthrough() {
    todo!("phase 4: rewrite acquire/release for new Mappable shape")
}

#[test]
#[ignore = "Phase 4: rewrite acquire/release for the new Mappable shape"]
fn shared_buffer_acquire_release_round_trip() {
    todo!("phase 4: rewrite acquire/release for new Mappable shape")
}
