//! REQUIREMENT LOCK — heterogeneous carry: pipes (resolved at enqueue) and host
//! values (known at build) flowing through ONE graph, with downstream
//! computation on the carried host value.
//!
//! These tests exist to make a specific capability a STRONG, EXPLICIT
//! requirement so that any future re-implementation of the eager graph CANNOT
//! silently drop it — it will fail to compile or fail at runtime here, at the
//! early design phase, rather than being rediscovered later as a "gap".
//!
//! The capability, precisely:
//!   1. A bundle can mix a `Pipe<device buffer>` branch and a `value(scalar)`
//!      branch, and the downstream closure receives the scalar BY VALUE (a
//!      `u32`), NOT a `Pipe<u32>`. This is the bundle-handle composition: each
//!      branch exposes its OWN handle (pipe for buffers, value for `value`),
//!      instead of everything being flattened to pipes.
//!   2. Because the carried scalar arrives by value, it can be COMPUTED ON at
//!      build time downstream (`step + 1`) and re-carried — multi-stage, in
//!      chain, with no host-side hand-tracking of the counter.
//!   3. A bare `Pipe<T>` is itself an `EagerOp` (the identity node), so a buffer
//!      branch passes into a bundle WITHOUT a `forward(..)` wrapper.
//!
//! If a redesign makes `value` hand a pipe again (losing by-value), or flattens
//! bundle branch handles to pipes, or drops `Pipe: EagerOp`, one or more of
//! these stops compiling — which is the point.

use claspr::Context;
use claspr::eager::{EagerOpExt, bundle2, download, upload, value};
use claspr::eager_bundle;
use claspr_test_kernels::kernels;

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

/// REQUIREMENT 1 + 3: a bundle mixing a kernel-output PIPE and a `value` SCALAR;
/// the downstream closure destructures `(Pipe<DeviceSlice>, u32)`. The kernel
/// branch is passed bare (no `forward`); the scalar arrives by value and is
/// asserted as a real `u32` (used in `assert_eq!`, not just threaded).
#[test]
fn bundle_mixes_pipe_and_value_scalar_arrives_by_value() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");

    let (out, carried): (Vec<u32>, u32) = upload::<u32, claspr::ReadWrite, _>(vec![0u32; N])
        .and_then(|buf| eager_bundle!(ks.fill_u32([N], buf, 7), value(42u32)))
        // `buf` is a Pipe<DeviceSlice>, `scalar` is a u32 BY VALUE — this binding
        // would not type-check if the bundle handed a Pipe<u32> here.
        .and_then(|(buf, scalar): (_, u32)| {
            // Use the scalar as a value to prove it's not a pipe: pick the kernel
            // factor from it, and carry it onward unchanged.
            let factor = if scalar == 42 { 1u32 } else { 0 };
            eager_bundle!(ks.scale_u32([N], buf, factor), value(scalar))
        })
        .and_then(|(buf, scalar)| eager_bundle!(download(buf), value(scalar)))
        .sync(&ctx)
        .expect("mixed pipe+scalar bundle");

    assert!(out.iter().all(|&v| v == 7), "fill 7 * scale 1 = 7");
    assert_eq!(carried, 42, "scalar carried through three stages unchanged");
}

/// REQUIREMENT 2: a carried scalar is COMPUTED ON at each stage (`step + 1`),
/// in-graph, across multiple stages — the ml_pass counter shape, distilled. If
/// the scalar were a `Pipe<u32>` downstream, `step + 1` would not compile.
#[test]
fn carried_scalar_is_computed_on_in_chain() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");

    let (out, step): (Vec<u32>, u32) = upload::<u32, claspr::ReadWrite, _>(vec![1u32; N])
        .and_then(|buf| eager_bundle!(ks.scale_u32([N], buf, 2), value(0u32)))
        .and_then(|(buf, step): (_, u32)| {
            // step: u32 — `step + 1` is a build-time host computation.
            eager_bundle!(ks.scale_u32([N], buf, 2), value(step + 1))
        })
        .and_then(|(buf, step): (_, u32)| eager_bundle!(ks.scale_u32([N], buf, 2), value(step + 1)))
        .and_then(|(buf, step)| eager_bundle!(download(buf), value(step)))
        .sync(&ctx)
        .expect("counter chain");

    assert!(out.iter().all(|&v| v == 8), "1 * 2 * 2 * 2 = 8");
    assert_eq!(step, 2, "0 -> +1 -> +1 = 2, computed in-chain");
}

/// REQUIREMENT: a pure host-value chain transforms the value at each stage —
/// `value(n).and_then(|n| value(n + k))`, the shape that was a permanent
/// DEVIATION before the by-value handle. `n` must be the value, not a pipe.
#[test]
fn value_chain_transforms_host_scalar() {
    let Some(ctx) = ctx() else { return };
    let result: u32 = value(1u32)
        .and_then(|n| value(n + 41))
        .and_then(|n| value(n * 2))
        .sync(&ctx)
        .expect("value transform chain");
    assert_eq!(result, 84, "(1 + 41) * 2");
}

/// REQUIREMENT 3 (isolated): a bare `Pipe<T>` IS an op. Take one branch's pipe
/// from a multi-output kernel handle and pass it directly into a `bundle2`
/// alongside a value — no `forward(..)`.
#[test]
fn bare_pipe_is_an_op_in_a_bundle() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");

    // Three producer branches (each upload→fill), bundled; then add_u32's
    // 3-pipe handle. Take `out` BARE into a bundle alongside a value.
    let (summed, tag): (Vec<u32>, u32) = eager_bundle!(
        upload::<u32, claspr::ReadWrite, _>(vec![0u32; N]).and_then(|buf| ks.fill_u32([N], buf, 3)),
        upload::<u32, claspr::ReadWrite, _>(vec![0u32; N]).and_then(|buf| ks.fill_u32([N], buf, 4)),
        upload::<u32, claspr::ReadWrite, _>(vec![0u32; N])
    )
    .and_then(|(a, b, out)| ks.add_u32([N], a, b, out))
    .and_then(|(_a, _b, out)| eager_bundle!(out, value(99u32))) // `out`: bare Pipe
    .and_then(|(out, tag)| eager_bundle!(download(out), value(tag)))
    .sync(&ctx)
    .expect("bare-pipe-in-bundle");

    assert!(summed.iter().all(|&v| v == 7), "3 + 4 = 7");
    assert_eq!(tag, 99);
    // bundle2 also reachable directly (arity-2 path):
    let _ = bundle2(value(1u32), value(2u32));
}
