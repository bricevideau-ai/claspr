//! USMSlice — fine-grain-system SVM wrapper around a host `Vec<T>`.
//! Designed to be the spec-correct replacement for HostBuffer (which
//! has UB per the spec when passed to a kernel; pocl is permissive,
//! rusticl is strict).
//!
//! Both pocl 7.2-pre on aarch64 and rusticl on llvmpipe report
//! `SvmLevel::FineSystem`, so these tests are expected to run on
//! both. The crucial test — `usm_slice_threads_into_kernel` — is the
//! exact scenario HostBuffer failed on rusticl ("host wrote N, kernel
//! saw 0"); it must pass here.

use claspr::{Buffer, Context, SvmLevel, USMSliceUninit};
use claspr_async::{DeviceOperation, usm_slice, usm_slice_alloc_zero};
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

#[test]
fn usm_slice_capability_gate() {
    // On a device WITHOUT FineSystem, USMSlice::new errors with
    // NotSupported. On FineSystem-capable devices, this test
    // gracefully skips its negative-path assertion and just
    // confirms construction succeeds.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    match ctx.svm_capability() {
        SvmLevel::FineSystem => {
            // Positive path: construction succeeds.
            let s = claspr::USMSlice::<u32>::new(&ctx, vec![1u32, 2, 3])
                .expect("FineSystem device should construct USMSlice");
            assert_eq!(s.len(), 3);
        }
        _ => {
            // Negative path: NotSupported.
            let err = claspr::USMSlice::<u32>::new(&ctx, vec![1u32, 2, 3])
                .expect_err("non-FineSystem device should error");
            assert!(matches!(err, claspr::Error::NotSupported(_)), "got {err:?}",);
        }
    }
}

#[test]
fn usm_slice_threads_into_kernel() {
    // The crucial test: host writes via Vec, kernel reads via SVM
    // pointer, kernel writes visible back to host. HostBuffer
    // failed this on rusticl ("host wrote N, kernel saw 0"); USMSlice
    // is the spec-correct replacement.
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let host_data = vec![5u32; N];
    let buf = usm_slice!(host_data)
        .and_then(|s| kernels.scale_u32([N], s, 7))
        .sync(&ctx)
        .expect("usm slice chain");

    assert_eq!(buf.len(), N);
    // Host reads kernel's writes directly through Deref.
    assert!(buf.iter().all(|&v| v == 35), "first: {}", buf[0]);
}

#[test]
fn usm_slice_drop_waits_for_in_flight_kernel() {
    // Submit a kernel via Tier 1 .submit() (returns the Event but
    // doesn't wait), drop USMSlice immediately, assert error_count
    // stays clean. The USMSlice's Drop must block on the in-flight
    // event before letting the Vec free; otherwise the kernel would
    // read freed memory.
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = claspr::USMSlice::<u32>::new(&ctx, vec![3u32; N]).expect("USM new");
    // Tier 1 submit — non-blocking, returns Event.
    let (buf, _evt) = kernels.scale_u32([N], buf, 4).submit().expect("submit");
    // Drop immediately. The Vec must NOT free until the kernel
    // completes (which the Drop's wait loop ensures).
    drop(buf);
    assert_eq!(ctx.error_count(), 0);
}

#[test]
fn usm_slice_alloc_produces_zero_initialised_buffer() {
    // Symmetric with device_slice_alloc / mapped_slice_alloc:
    // `usm_slice_alloc_zero!(T, N)` allocates a host Vec of length N
    // initialised to T::default(). Before any kernel runs, every
    // element is T::default() (zero for u32).
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let buf = usm_slice_alloc_zero!(u32, N).sync(&ctx).expect("alloc");
    assert_eq!(buf.len(), N);
    assert!(buf.iter().all(|&v| v == 0), "alloc should zero-init");
}

#[test]
fn macro_usm_slice_repeat_arm() {
    // `usm_slice![v; N]` → usm_slice!(vec![v; N]).
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = usm_slice![6u32; N]
        .and_then(|s| kernels.scale_u32([N], s, 5))
        .sync(&ctx)
        .expect("macro repeat");
    assert!(buf.iter().all(|&v| v == 30));
}

#[test]
fn macro_usm_slice_literal_arm() {
    // `usm_slice![a, b, c]` → usm_slice!(vec![a, b, c]).
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let buf = usm_slice![10u32, 20, 30, 40]
        .and_then(|s| kernels.scale_u32([4], s, 2))
        .sync(&ctx)
        .expect("macro literal");
    assert_eq!(&buf[..], &[20u32, 40, 60, 80]);
}

#[test]
fn usm_slice_host_writes_visible_to_kernel_via_deref_mut() {
    // Mutate via DerefMut, then launch. With fine-grain system,
    // host writes are visible to the kernel without any sync.
    let Some(ctx) = ctx_with_fine_system() else {
        return;
    };
    let kernels = kernels::kernels(&ctx).expect("load kernels");

    let mut buf = claspr::USMSlice::<u32>::new(&ctx, vec![0u32; N]).expect("USM new");
    // Host write via DerefMut.
    for (i, slot) in buf.iter_mut().enumerate() {
        *slot = i as u32;
    }
    // Kernel scales each element by 2.
    let buf = kernels.scale_u32([N], buf, 2).wait().expect("scale");
    // Host reads the scaled values back through Deref.
    for (i, &v) in buf.iter().enumerate() {
        assert_eq!(v, (i as u32) * 2, "element {i}");
    }
}

#[test]
fn usm_slice_uninit_returns_wrapper_assume_init_writes_via_kernel() {
    // alloc_uninit returns USMSliceUninit (type-state). assume_init
    // hands back the USMSlice; the kernel writes every slot before
    // any host read. Mirrors DeviceSlice/MappedSlice alloc_uninit
    // shape.
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
    // Host reads kernel's writes directly via Deref (fine-grain SVM).
    assert!(buf.iter().all(|&v| v == 77));
}
