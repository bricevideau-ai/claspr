//! Eager port of `usm.rs`: USMSlice (fine-grain-system SVM wrapper around a
//! host `Vec<T>`) threaded through the eager graph API.
//!
//! Old → new mapping:
//!   `usm_slice!(v)`              → `claspr::eager::usm_slice(v)` (Vec → USMSlice leaf)
//!   `usm_slice![v; N]`          → `usm_slice(vec![v; N])` (no eager macro)
//!   `usm_slice![a, b, c]`       → `usm_slice(vec![a, b, c])`
//!   `usm_slice_alloc_zero!(T,N)`→ `usm_alloc_uninit::<T, _>(N).and_then(|u| fill_usm_uninit(u, T::default))`
//!                                 (no eager `usm_slice_alloc_zero`; for u32,
//!                                  default == 0, so fill-with-0 is equivalent —
//!                                  same pattern eager_alloc_ops used for mapped)
//!   `.and_then(|s| kernel(...))`→ same; the kernel op accepts the USMSlice pipe
//!                                 and yields a USMSlice (impl_to_input_concrete)
//!
//! Tier-1-only tests (capability gate, drop-safety, DerefMut visibility, the
//! uninit/assume_init shape) have no Tier-2 chain to port and are reproduced
//! verbatim — they're part of the same suite's coverage.
//!
//! Both pocl 7.2-pre on aarch64 and rusticl on llvmpipe report
//! `SvmLevel::FineSystem`, so these run on both. The FineSystem guard is
//! preserved exactly — a SKIP on a non-FineSystem device is a PASS.

use claspr::eager::{
    DeviceOpExt, bundle2, fill_usm_uninit, usm_alloc_uninit, usm_slice, usm_slice_as,
};
use claspr::{Buffer, Context, Frozen, MemRef, RecordableBuffer, SvmLevel, USMSliceUninit};
use claspr_test_kernels::kernels;

const N: usize = 64;

/// Stable identity of a USM slice's backing SVM pointer (as `usize`) for `==`
/// across replays. Works on a `USMSlice` and (via `Deref`/`Checkout`) on a
/// `Checkout<USMSlice>`.
fn svm_ptr_of<B: RecordableBuffer>(b: &B) -> usize {
    match b.record_handle().mem {
        MemRef::Svm(p) => p as usize,
        MemRef::Buffer(m) => m as usize,
    }
}

fn ctx_with_fine_system() -> Option<Context> {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return None;
    };
    if ctx.svm_capability() != SvmLevel::FineSystem {
        eprintln!(
            "SKIP: device reports {:?}, USMSlice needs FineSystem",
            ctx.svm_capability()
        );
        return None;
    }
    Some(ctx)
}

/// usm.rs::usm_slice_capability_gate — Tier-1 construction gate (no chain).
/// Reproduced verbatim.
#[test]
fn usm_slice_capability_gate() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    match ctx.svm_capability() {
        SvmLevel::FineSystem => {
            let s = claspr::USMSlice::<u32>::new(&ctx, vec![1u32, 2, 3])
                .expect("FineSystem device should construct USMSlice");
            assert_eq!(s.len(), 3);
        }
        _ => {
            let err = claspr::USMSlice::<u32>::new(&ctx, vec![1u32, 2, 3])
                .expect_err("non-FineSystem device should error");
            assert!(matches!(err, claspr::Error::NotSupported(_)), "got {err:?}",);
        }
    }
}

/// usm.rs::usm_slice_threads_into_kernel — the crucial test. Host writes via Vec,
/// kernel reads via SVM pointer, kernel writes visible back to host.
/// usm_slice(5s) → scale 7 → 35.
#[test]
fn usm_slice_threads_into_kernel() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let host_data = vec![5u32; N];
    let buf = usm_slice(host_data)
        .and_then(|s| kernels.scale_u32([N], s, 7))
        .sync(&ctx)
        .expect("usm slice chain");

    assert_eq!(buf.len(), N);
    // Host reads kernel's writes directly through Deref.
    assert!(buf.iter().all(|&v| v == 35), "first: {}", buf[0]);
}

/// usm.rs::usm_slice_drop_waits_for_in_flight_kernel — Tier-1 submit + immediate
/// drop; the USMSlice's Drop must block on the in-flight event. Reproduced
/// verbatim (no Tier-2 chain shape).
#[test]
fn usm_slice_drop_waits_for_in_flight_kernel() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = claspr::USMSlice::<u32>::new(&ctx, vec![3u32; N]).expect("USM new");
    let (buf, _evt) = kernels.scale_u32([N], buf, 4).submit().expect("submit");
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

/// usm.rs::usm_slice_alloc_produces_zero_initialised_buffer — eager equivalent of
/// `usm_slice_alloc_zero!(u32, N)` via `usm_alloc_uninit` + `fill_usm_uninit`
/// with u32's default (0). Every element zero before any kernel runs.
#[test]
fn usm_slice_alloc_produces_zero_initialised_buffer() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let buf = usm_alloc_uninit::<u32>(N)
        .and_then(|u| fill_usm_uninit(u, 0u32))
        .sync(&ctx)
        .expect("alloc");
    assert_eq!(buf.len(), N);
    assert!(buf.iter().all(|&v| v == 0), "alloc should zero-init");
}

/// usm.rs::macro_usm_slice_repeat_arm — `usm_slice![v; N]` → `usm_slice(vec![v; N])`.
#[test]
fn macro_usm_slice_repeat_arm() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = usm_slice(vec![6u32; N])
        .and_then(|s| kernels.scale_u32([N], s, 5))
        .sync(&ctx)
        .expect("macro repeat");
    assert!(buf.iter().all(|&v| v == 30));
}

/// usm.rs::macro_usm_slice_literal_arm — `usm_slice![a, b, c]` → `usm_slice(vec![a, b, c])`.
#[test]
fn macro_usm_slice_literal_arm() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = usm_slice(vec![10u32, 20, 30, 40])
        .and_then(|s| kernels.scale_u32([4], s, 2))
        .sync(&ctx)
        .expect("macro literal");
    assert_eq!(&buf[..], &[20u32, 40, 60, 80]);
}

/// usm.rs::usm_slice_host_writes_visible_to_kernel_via_deref_mut — Tier-1
/// DerefMut host write then launch. Reproduced verbatim (no Tier-2 chain shape).
#[test]
fn usm_slice_host_writes_visible_to_kernel_via_deref_mut() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let mut buf = claspr::USMSlice::<u32>::new(&ctx, vec![0u32; N]).expect("USM new");
    for (i, slot) in buf.iter_mut().enumerate() {
        *slot = i as u32;
    }
    let buf = kernels.scale_u32([N], buf, 2).wait().expect("scale");
    for (i, &v) in buf.iter().enumerate() {
        assert_eq!(v, (i as u32) * 2, "element {i}");
    }
}

/// usm.rs::usm_slice_uninit_returns_wrapper_assume_init_writes_via_kernel —
/// Tier-1 alloc_uninit + assume_init + kernel fill. Reproduced verbatim (the
/// eager `usm_alloc_uninit` covers the producing leaf; this test exercises the
/// Tier-1 type-state + assume_init shape, which has no eager analog).
#[test]
fn usm_slice_uninit_returns_wrapper_assume_init_writes_via_kernel() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let uninit: USMSliceUninit<u32> =
        claspr::USMSlice::<u32>::alloc_uninit(&ctx, N).expect("USM alloc_uninit");
    assert_eq!(uninit.len(), N);
    let _ = format!("{uninit:?}");
    // SAFETY: kernel fills every slot before any read below.
    let buf = unsafe { uninit.assume_init() };
    let buf = kernels.fill_u32([N], buf, 77).wait().expect("kernel fill");
    assert!(buf.iter().all(|&v| v == 77));
}

// ── #215: usm_slice is REUSABLE (Upload-shaped self-rehoming) ────────────────
//
// `usm_slice` was a move-in-once chain head (`Mutex<Option<Vec<T>>>` drained on
// run 1). It now mirrors `Upload`: the host source is retained, the USMSlice is
// allocated ONCE into a home cell, re-lent across replays over a STABLE SVM
// pointer, and re-seeded on replay iff the marker is kernel-writable. These tests
// mirror the Upload replay proofs (graph_reuse.rs) for the USM tier.

/// The core replay proof: a `usm_slice(RW) → scale` graph `sync`'d twice gives the
/// SAME result both runs (idempotent reseed — the RW buffer was mutated in place on
/// run 1, so run 2 re-seeds the host source, not compounds), AND the USM chain-head
/// Checkout re-arms so the graph is re-runnable. (USM is host memory, so the result
/// is read directly through the Checkout's `Deref` — no `download`.)
#[test]
fn usm_slice_reused_graph_is_idempotent() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // usm_slice(vec![1; N]) -> scale ×2. Expect all 2s on BOTH runs.
    let g = usm_slice(vec![1u32; N]).and_then(|s| kernels.scale_u32([N], s, 2u32));

    let run1 = g.sync(&ctx).expect("run 1");
    assert!(run1.iter().all(|&v| v == 2), "run1 got {}", run1[0]);
    drop(run1); // re-arm g

    // Run 2 must MATCH run 1 (reseed-on-replay, not compounding — a move-in-once
    // usm_slice would have errored "already consumed" here instead).
    let run2 = g.sync(&ctx).expect("run 2 (replay)");
    assert!(
        run2.iter().all(|&v| v == 2),
        "run2 must match run1 (idempotent reseed), got {}",
        run2[0]
    );
}

/// Stable SVM pointer across replays + explicit reseed-after-kernel-mutation. The
/// RW buffer is scaled in place (mutated) each run; a borrowing map reads it, then
/// the Checkout drops → the SAME USM allocation re-homes and is re-seeded from the
/// host source, so the next run starts from the same seed over the SAME pointer.
#[test]
fn usm_slice_rw_stable_pointer_and_reseed_after_mutation() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // Keep the USMSlice on the graph (no download) so we can read it via the
    // Checkout and observe the SVM pointer + the in-place mutation each run.
    let g = usm_slice(vec![10u32; N]).and_then(|s| kernels.scale_u32([N], s, 3u32));

    // Run 1: 10 × 3 = 30.
    let co1 = g.sync(&ctx).expect("run 1");
    let ptr1 = svm_ptr_of(&*co1);
    assert!(co1.iter().all(|&v| v == 30), "run1 got {}", co1[0]);
    drop(co1); // re-home the USM slice (reseed happens on next execute)

    // Run 2 (replay over the SAME SVM pointer): the buffer was reseeded to 10, so
    // 10 × 3 = 30 again — NOT 30 × 3 = 90 (would be compounding without reseed).
    let co2 = g.sync(&ctx).expect("run 2 (replay)");
    let ptr2 = svm_ptr_of(&*co2);
    assert_eq!(ptr1, ptr2, "USM pointer must be stable across replays");
    assert!(
        co2.iter().all(|&v| v == 30),
        "run2 must reseed to 10 then ×3 = 30 (not compound to 90), got {}",
        co2[0]
    );
    drop(co2);
}

/// Marker generality: a `Frozen` (kernel read-only) `usm_slice` seeds ONCE and is
/// NOT reseeded on replay (RESEED_ON_REPLAY = false). The kernel only reads it, so
/// its contents persist unchanged; the graph still replays over a stable pointer.
/// `add_u32(a, b, out)` reads the frozen buffer as one addend `b` and writes a
/// separate RW `out` — nothing writes the frozen buffer.
#[test]
fn usm_slice_frozen_seeds_once_and_replays() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    // out = a + b, with `a` a fresh RW seed (100s) and `b` a Frozen buffer (4s,
    // read-only, seeded once). `out` is a fresh RW USM buffer, threaded to the
    // terminal and read via Deref (USM is host memory). Frozen ⇒ `b` is NOT
    // reseeded on replay; `a`/`out` are RW ⇒ reseeded.
    let g = bundle2(
        usm_slice(vec![100u32; N]),
        usm_slice_as(vec![4u32; N], Frozen),
    )
    .and_then(move |(a, frozen)| {
        usm_slice(vec![0u32; N]).and_then(move |out| {
            kernels
                .add_u32([N], a, frozen, out)
                .and_then(|(_a, _frozen, out)| claspr::eager::forward(out))
        })
    });

    let out1 = g.sync(&ctx).expect("run 1");
    assert!(
        out1.iter().all(|&v| v == 104),
        "run1: 100 + 4 = 104, got {}",
        out1[0]
    );
    drop(out1);

    // Run 2 (replay): frozen buffer NOT reseeded (unchanged 4s), a/out reseeded ⇒
    // still 104. A move-in-once usm_slice would have errored "already consumed".
    let out2 = g.sync(&ctx).expect("run 2 (replay)");
    assert!(
        out2.iter().all(|&v| v == 104),
        "run2 must match run1 (frozen seed-once + rw reseed), got {}",
        out2[0]
    );
}

/// Generality: `usm_slice` as a BUNDLE BRANCH re-arms across replays (the branch's
/// per-buffer home rides its own Checkout). Two USM branches scaled independently,
/// bundled, `sync`'d twice — both re-home over stable pointers with idempotent
/// results.
#[test]
fn usm_slice_as_bundle_branch_replays() {
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let g = bundle2(
        usm_slice(vec![2u32; N]).and_then(|s| kernels.scale_u32([N], s, 5u32)),
        usm_slice(vec![3u32; N]).and_then(|s| kernels.scale_u32([N], s, 6u32)),
    );

    // Run 1: a = 2×5 = 10, b = 3×6 = 18.
    let (a1, b1) = g.sync(&ctx).expect("run 1");
    let (pa1, pb1) = (svm_ptr_of(&*a1), svm_ptr_of(&*b1));
    assert!(a1.iter().all(|&v| v == 10), "run1 a got {}", a1[0]);
    assert!(b1.iter().all(|&v| v == 18), "run1 b got {}", b1[0]);
    drop((a1, b1)); // both branches re-home

    // Run 2 (replay, same pointers, idempotent reseed): 10 and 18 again.
    let (a2, b2) = g.sync(&ctx).expect("run 2 (replay)");
    assert_eq!(svm_ptr_of(&*a2), pa1, "branch a pointer stable");
    assert_eq!(svm_ptr_of(&*b2), pb1, "branch b pointer stable");
    assert!(a2.iter().all(|&v| v == 10), "run2 a got {}", a2[0]);
    assert!(b2.iter().all(|&v| v == 18), "run2 b got {}", b2[0]);
    drop((a2, b2));
}
