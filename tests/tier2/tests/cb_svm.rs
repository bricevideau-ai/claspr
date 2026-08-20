//! SVM buffers in a command buffer (design v2). SVM kernel args already record
//! (via clSetKernelArgSVMPointer); this covers the SVM device-command leaves —
//! `clCommandSVMMemFillKHR` for an SVM fill — which need the extension's SVM command
//! variants (cl_khr_command_buffer >= 0.9.4, OpenCL 2.0+).
//!
//! Where the driver's command buffer lacks the SVM commands, `CbBuilder` marks the
//! build ineligible and the boundary falls back to the per-op software path — still
//! correct, just not CB-accelerated. So these tests assert RESULTS unconditionally
//! and only note whether a CB was homed.

use claspr::MappedSlice;
use claspr::eager::{DeviceOp, DeviceOpExt};
use claspr_test_kernels::kernels;
use claspr_test_support::{ctx_with_svm, homed_cb};

const N: usize = 64;

/// An SVM `fill` then a kernel over the same SVM buffer is a weight-2 all-device
/// span: `clCommandSVMMemFillKHR` + `clCommandNDRangeKernelKHR` in ONE command
/// buffer (where the driver supports SVM CB commands; else software fallback). The
/// fill sets 2, the kernel scales ×5 → every element 10, idempotent across replays.
#[test]
fn svm_fill_then_kernel_runs_as_command_buffer() {
    let Some(ctx) = ctx_with_svm() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let buf = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc svm");

    // MappedSlice::fill -> FillMapped (records clCommandSVMMemFillKHR); then scale_u32
    // over the same SVM buffer (SVM kernel arg). Two commands → weight 2 → CB.
    let g = buf.fill(2u32).and_then(move |b| ks.scale_u32([N], b, 5u32));
    assert_eq!(g.cbable_weight(), 2, "svm fill + kernel = two commands");

    for i in 0..3 {
        let co = g.sync(&ctx).unwrap_or_else(|e| panic!("sync {i}: {e:?}"));
        let guard = co.map().wait().expect("map");
        assert!(
            guard.iter().all(|&v| v == 10),
            "iter {i}: fill(2)*5 == 10; got {:?}",
            &guard[..8]
        );
        drop(guard);
        drop(co);
    }

    if ctx.has_cl_khr_command_buffer() && homed_cb(&g) {
        eprintln!("svm fill+kernel: recorded a command buffer (SVM CB commands present)");
    } else {
        eprintln!("svm fill+kernel: software fallback (SVM CB commands unavailable)");
    }
}

/// An SVM fill then an SVM→SVM copy is a weight-2 all-device span:
/// `clCommandSVMMemFillKHR` + `clCommandSVMMemcpyKHR` in ONE command buffer (where
/// the driver supports the SVM commands; else software fallback). fill src=9, copy
/// src→dst, read dst → every element 9.
#[test]
fn svm_fill_then_copy_runs_as_command_buffer() {
    use claspr::eager_copy_to;
    let Some(ctx) = ctx_with_svm() else { return };
    let src = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc src");
    let dst = MappedSlice::<u32>::alloc_zero(&ctx, N).expect("alloc dst");

    // fill(src=9) -> eager_copy_to(src -> dst): SVM fill + SVM copy → weight 2 → CB.
    let g = src.fill(9u32).and_then(move |s| eager_copy_to(s, dst));
    assert_eq!(g.cbable_weight(), 2, "svm fill + copy = two commands");

    for i in 0..3 {
        let (_src_co, dst_co): (
            claspr::eager::Checkout<MappedSlice<u32>>,
            claspr::eager::Checkout<MappedSlice<u32>>,
        ) = g.sync(&ctx).unwrap_or_else(|e| panic!("sync {i}: {e:?}"));
        let guard = dst_co.map().wait().expect("map dst");
        assert!(
            guard.iter().all(|&v| v == 9),
            "iter {i}: copy of fill(9); got {:?}",
            &guard[..8]
        );
        drop(guard);
        drop((_src_co, dst_co));
    }

    if ctx.has_cl_khr_command_buffer() && homed_cb(&g) {
        eprintln!("svm fill+copy: recorded a command buffer");
    } else {
        eprintln!("svm fill+copy: software fallback");
    }
}
