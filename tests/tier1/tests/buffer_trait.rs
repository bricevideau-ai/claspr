//! `Buffer<T>` plumbing — validates the trait's documented contract:
//! a polymorphic accessor for `len`/`is_empty`/`ctx` that works across
//! `DeviceSlice<T>` and `MappedSlice<T>`.
//!
//! Per the trait's rustdoc this is **explicitly not** a tier-
//! polymorphism point — there is no uniform `upload` verb across the
//! tiers. Code that needs the inspect-the-buffer accessors without
//! committing to a tier is exactly the use case the trait exists
//! for; this file proves that use case compiles and runs.

use claspr::{Buffer, Context, DeviceSlice, MappedSlice, SvmLevel};

const N: usize = 128;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

/// The polymorphic accessor — the documented use case for `Buffer<T>`.
/// `B` only needs to expose `len`/`is_empty`/`ctx`; we deliberately
/// don't ask for an upload/download method because those genuinely
/// differ per tier.
fn describe<T, B: Buffer<T>>(label: &str, b: &B) -> String {
    format!(
        "{label}: {} elements, empty? {}, device {}",
        b.len(),
        b.is_empty(),
        b.ctx().device().name().unwrap_or_default(),
    )
}

#[test]
fn buffer_accessor_works_uniformly_across_tiers() {
    let Some(ctx) = ctx() else { return };

    let device_slice = DeviceSlice::<u32>::alloc(&ctx, N).expect("DeviceSlice alloc");

    let d = describe("DeviceSlice", &device_slice);
    println!("{d}");
    assert!(d.contains(&format!("{N} elements")));

    // MappedSlice when available — the other tier currently covered.
    if ctx.svm_capability() != SvmLevel::None {
        let shared = MappedSlice::<u32>::alloc(&ctx, N).expect("MappedSlice alloc");
        let d = describe("MappedSlice", &shared);
        println!("{d}");
        assert!(d.contains(&format!("{N} elements")));
    }
}

#[test]
fn buffer_is_empty_matches_zero_len_alloc() {
    let Some(ctx) = ctx() else { return };
    // Zero-length allocations on every tier where they're legal.
    // OpenCL accepts size=0 for clCreateBuffer in practice (returns
    // a valid mem object); some drivers reject it. Use this test as
    // a soft check — skip the assertion if alloc errors.
    if let Ok(b) = DeviceSlice::<u32>::alloc(&ctx, 0) {
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
    }
}

#[test]
fn buffer_ctx_round_trips() {
    // `Buffer::ctx()` must return a reference to the same context the
    // buffer was allocated on. Trivial but the contract is load-bearing
    // (drop ordering, queue lookups go through this).
    let Some(ctx) = ctx() else { return };
    let buf = DeviceSlice::<u32>::alloc(&ctx, N).expect("alloc");
    assert!(std::ptr::eq(buf.ctx().raw_context(), ctx.raw_context()));
}
