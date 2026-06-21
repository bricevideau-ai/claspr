//! Eager-API port of `conditional.rs`: conditional graphs via `DynOp` type
//! erasure.
//!
//! `conditional.rs` is built entirely around `DynOp` — a boxed, type-erased
//! op so that `if`/`match` arms producing DIFFERENT concrete op types can share
//! one `Output` and compose. The eager API (`claspr::eager`) has NO equivalent:
//! there is no `Box<dyn EagerOp>`, no `DynOp`/erasure constructor, and `EagerOp`
//! is not object-safe as exposed (associated `Handle` type + `into_output`
//! consuming `self`). So every `DynOp::new(...)` site is unportable.
//!
//! Only `baseline_kernel_chain_without_dynop` uses NO `DynOp` — it is ported
//! 1:1 below. All seven `DynOp` tests are BLOCKED on an eager dyn-op erasure
//! primitive (see report).
//!
//!   `upload!(v)`        → `upload::<u32, claspr::ReadWrite, _>(v)`
//!   `.and_then_host(f)` → `.and_then_host(f)` (DeviceSlice View is `&mut [u32]`)

use claspr::Context;
use claspr::eager::{EagerOpExt, upload};
use claspr_test_kernels::kernels;
use std::sync::{Arc, Mutex};

const N: usize = 32;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

// BLOCKED: dyn_op_lets_if_arms_have_different_concrete_types — `if cond { DynOp::new(a) }
// else { DynOp::new(b) }` with two DIFFERENT concrete op types unified via
// erasure. Needs an eager dyn-op/box primitive (no `Box<dyn EagerOp>` exists).

// BLOCKED: dyn_op_wraps_simple_value — `DynOp::new(value(42))`. The value/assert
// is trivially reproducible as a plain eager chain, but the test exists to
// exercise the DynOp wrapper itself; no eager erasure primitive to wrap.

// BLOCKED: dyn_op_wraps_value_chain — `DynOp::new(value(1).and_then(...))`. Same
// as above: tests the wrapper, which has no eager equivalent.

// BLOCKED: dyn_op_wraps_upload_download — `DynOp::new(upload.and_then(download))`.
// Tests DynOp over a transfer chain; no eager erasure primitive.

#[test]
fn baseline_kernel_chain_without_dynop() {
    // Sanity baseline: kernel chain OUTSIDE DynOp — the one test in
    // conditional.rs that uses no erasure, so it ports 1:1. and_then_host's
    // closure returns Result<()>; the reduction value flows out via
    // Arc<Mutex<_>>.
    let Some(ctx) = ctx() else { return };
    let kernels = kernels::kernels(&ctx).expect("load kernels");
    let sum_cell = Arc::new(Mutex::new(0u32));
    let cell = Arc::clone(&sum_cell);
    let _final_buf = upload::<u32, claspr::ReadWrite, _>(vec![3u32; N])
        .and_then(|buf| kernels.fill_u32([N], buf, 9))
        .and_then_host(move |slice: &mut [u32]| {
            *cell.lock().unwrap() = slice.iter().sum();
            Ok(())
        })
        .sync(&ctx)
        .expect("baseline");
    assert_eq!(*sum_cell.lock().unwrap(), 9 * N as u32);
}

// BLOCKED: dyn_op_wraps_bare_kernel_op — `DynOp::new(kernels.fill_u32(...))`.
// Tests DynOp over a bare kernel op; no eager erasure primitive.

// BLOCKED: dyn_op_minimal_kernel_chain — `DynOp::new(upload.and_then(fill).and_then_host(..))`.
// Tests DynOp over a kernel+host chain; no eager erasure primitive.

// BLOCKED: dyn_op_picks_branch_with_or_without_kernel — the core conditional:
// two `if` arms of different concrete types (kernel chain vs `value(0)`) unified
// via DynOp. Needs an eager dyn-op/box primitive.

// BLOCKED: non_taken_branch_closure_does_not_fire — laziness guarantee on
// `DynOp` arm construction; pins that the not-taken `DynOp::new(...)` arm is
// never built. Needs the eager dyn-op primitive to express the two-arm shape.
