//! Eager-API port of `run_await.rs`: the `.run(&ctx).await` async terminal.
//!
//! ALL THREE test fns are BLOCKED on the same KNOWN gap: the eager API has no
//! async terminal. `EagerOpExt` exposes only the synchronous `.sync(&ctx)`
//! terminal — there is no `.run(&ctx) -> ChainFuture`, no `Future`/`poll`
//! implementation, and no `chain_future` in `claspr::eager` (verified by grep).
//! The whole point of these three tests is the `.await` mechanism (completion
//! signaled by an `clEnqueueMarkerWithWaitList` callback waking the waker), so
//! none can be expressed against the eager API without a new async-terminal
//! primitive.
//!
//! BLOCKED tests (each needs an eager `.run()`/`ChainFuture` async terminal):
//!   - await_simple_chain          — upload → fill kernel → download via .await
//!   - await_pure_value_chain      — pure value chain resolved via .await
//!   - await_propagates_chain_error — chain Err surfaces at `.run().await`
//!
//! (The pure-value and error-propagation *values* are covered synchronously in
//! eager_chain.rs / eager_error.rs; only the async terminal itself is missing.)

// No runnable tests: the async `.run().await` terminal does not exist in the
// eager API. File intentionally test-free until an eager async terminal lands.
