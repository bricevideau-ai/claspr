//! Eager-API port of `transfer_to_device.rs`.
//!
//! BLOCKED: `transfer_to_device(buf, &dev)` — needs an eager `transfer_to_device`
//! primitive. The closure layer exposes `claspr_async::transfer_to_device`, a
//! `DeviceOperation` that enqueues an explicit cross-device `DeviceSlice`
//! migration. The eager API (`claspr::eager`) has NO equivalent op: there is no
//! `transfer_to_device` / migrate leaf, and `on_device` only ROUTES an op's
//! enqueue to a chosen queue — it does not move an already-resident buffer to
//! another device. Both tests in this file are the cross-device transfer shape
//! (upload → transfer → on_device kernel → … ), so neither is expressible until
//! an eager `transfer_to_device` op exists.
//!
//! NOTE: both tests are also gated on a two-device context and skip on the
//! single-device CI runners — but the missing primitive blocks them even where a
//! second device exists. Reported to the cutover owner; do NOT add the primitive
//! here (source change is the owner's call).
//!
//! Both `transfer_to_device_completes_in_chain` and
//! `transfer_then_on_device_matches_scenario_14_shape` are blocked below.

// BLOCKED: cross-device transfer shape — needs an eager `transfer_to_device` op.
// Originals: tests/tier2/tests/transfer_to_device.rs
//   - transfer_to_device_completes_in_chain
//   - transfer_then_on_device_matches_scenario_14_shape
//
// Faithful translation requires, per stage:
//   .and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(i)))
// where `transfer_to_device` would be an `EagerOp` producing the migrated
// `DeviceSlice`. No such op is exported from `claspr::eager`.

#[test]
fn transfer_to_device_eager_port_is_blocked() {
    // Placeholder so this file compiles and the suite stays green. The two real
    // tests above are BLOCKED on the missing eager `transfer_to_device`
    // primitive (see module docs). Remove this once the primitive lands and the
    // two tests are ported 1:1.
    eprintln!(
        "BLOCKED: eager transfer_to_device port — no eager `transfer_to_device` op; \
         see module docs"
    );
}
