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

use claspr::eager::{EagerOpExt, fill_usm_uninit, usm_alloc_uninit, usm_slice};
use claspr::{Buffer, Context, SvmLevel, USMSliceUninit};
use claspr_test_kernels::kernels;

const N: usize = 64;

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
    let buf = usm_slice::<u32, claspr::ReadWrite>(host_data)
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
    let buf = usm_alloc_uninit::<u32, claspr::ReadWrite>(N)
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

    let buf = usm_slice::<u32, claspr::ReadWrite>(vec![6u32; N])
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

    let buf = usm_slice::<u32, claspr::ReadWrite>(vec![10u32, 20, 30, 40])
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
