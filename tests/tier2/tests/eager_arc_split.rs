//! Eager-API port of `arc_split.rs`: `.arc()` + `ArcSplit::split::<N>()` — the
//! shared-input fan-out pattern.
//!
//! ALL THREE tests in `arc_split.rs` are BLOCKED in the eager API. Every test
//! here splits a HOST VALUE (`value(vec![..]).arc()` / `value(String).arc()`)
//! to N branches and then does HOST ARITHMETIC on each branch's clone
//! (`arc.iter().sum()`, `arc.iter().product()`, `arc.len()`, `s.len()`).
//!
//! Two independent gaps block this in eager:
//!   1. The eager `arc_split::<N>` slot is a `Pipe<Arc<T>>`, and an `and_then`
//!      closure receives that PIPE, not the `Arc<T>` value — so `arc.iter()`
//!      etc. cannot be called in-graph (same class as the `value_passthrough`
//!      deviation in `eager_chain.rs`: `and_then` hands a pipe, not the host
//!      value).
//!   2. The only host-value seam, `and_then_host`, requires the upstream
//!      `Output: Mappable`. `Arc<Vec<u32>>` / `Vec<u32>` / `String` are NOT
//!      `Mappable` (only `DeviceSlice`, scalars, `()`, and tuples of those
//!      are) — so there is no host seam that hands back the arc'd value either.
//!
//! The eager `arc_split` is designed for DEVICE-buffer Arc fan-out (each clone
//! fed as a read-only kernel arg — see `eager_cutover::arc_split_read_only_fan_out`
//! and `eager_diamond.rs`), not for host-side reductions over a shared lifted
//! host value. See report for the needed primitive.

// BLOCKED: arc_split_into_three_branches_share_value — split host value + host
// reduce per branch (arc.iter().sum/product/len). Needs either an eager seam
// that yields the arc'd value to a host closure (Arc/Vec Mappable in the host
// seam), or a value-receiving `and_then` (the host-scalar-passthrough gap).

// BLOCKED: arc_split_propagates_branch_error — same host-value-reduction shape;
// branch A does `arc.iter().sum()` on the split host value. Blocked for the
// identical reason. (The error injection via `and_then_host(|()| Err(..))` IS
// expressible — `()` is Mappable — but the surrounding host reduction is not.)

// BLOCKED: arc_split_single_does_not_panic — `value(String).arc().split::<1>()`
// then `s.len()` on the host. `String` is not Mappable and `and_then` yields a
// `Pipe<Arc<String>>`, so the host `.len()` on the value is not expressible.

/// All three originals are blocked (see module doc). Placeholder keeps the test
/// binary compiling without fake-passing any blocked shape.
#[test]
fn _placeholder() {}
