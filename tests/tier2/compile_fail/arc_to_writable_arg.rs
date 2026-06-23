//! Regression guard for the Arc-as-writable hole (spike fix `c4d1711`).
//!
//! `Arc<DeviceSlice<T, M>>` implements ONLY `KernelSliceReadArg<T>`, never
//! the read-write variant — so two kernels can never write the same `cl_mem`
//! through cloned `Arc`s, preserving the move-semantics safety story. A
//! writable kernel slot (`scale_u32`'s `data: &mut [u32]`) must reject an
//! `Arc<DeviceSlice<…>>`.
//!
//! Unified-API restatement of the deleted `arc_to_writable_arg` fixture: the
//! kernel launcher is now a `DeviceOp`, but the kernel-arg traits are
//! unchanged.

use claspr::{Context, DeviceSlice};
use claspr_test_kernels::kernels;
use std::sync::Arc;

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = kernels::kernels(&ctx).unwrap();
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, 4).unwrap();
    let shared = Arc::new(buf);
    // `scale_u32`'s `data: &mut [u32]` slot wants `KernelSliceReadWriteArg`.
    // `Arc<DeviceSlice<u32>>` only impls the Read variant — should reject.
    let _ = kernels.scale_u32([4usize], shared, 2u32);
}
