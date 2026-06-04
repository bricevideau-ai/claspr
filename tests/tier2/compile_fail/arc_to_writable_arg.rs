//! Regression guard for commit `c4d1711` — the Arc-as-writable hole.
//!
//! Before that commit, `Arc<DeviceSlice<T, M>>` implemented BOTH
//! `KernelSliceReadArg<T>` and `KernelSliceReadWriteArg<T>` — meaning
//! `bundle!(scale_u32(buf.clone()), scale_u32(buf.clone()))` would
//! happily type-check and let two kernels write the same `cl_mem`
//! through cloned `Arc`s, defeating the move-semantics safety story.
//!
//! After the fix `Arc` impls only the read variant. This fixture
//! confirms a writable kernel slot rejects an `Arc<DeviceSlice<…>>`
//! — turning the spike fix into a type-system invariant.

use claspr::{Context, DeviceSlice};
use claspr_test_kernels::kernels;
use std::sync::Arc;

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = kernels::kernels(&ctx).unwrap();
    let buf = DeviceSlice::<u32>::alloc(&ctx, 4).unwrap();
    let shared = Arc::new(buf);
    // `scale_u32`'s `data: &mut [u32]` slot wants `KernelSliceReadWriteArg`.
    // `Arc<DeviceSlice<u32>>` only impls the Read variant — should reject.
    let _ = kernels.scale_u32([4usize], shared, 2u32);
}
