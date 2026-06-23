//! Two `bundle!` arms that write the same owned `DeviceSlice` must not
//! type-check. The first arm moves `buf`; the second sees a "used after move"
//! error. Confirms move semantics catch parallel-write aliasing on owned
//! buffers (`bundle!` at claspr/src/eager.rs).
//!
//! Unified-API restatement of the deleted `bundle_aliased_owned_writes`
//! fixture: `bundle!` now lives at the `claspr` crate root and each arm is a
//! `DeviceOp`, but the writable kernel slots still take the buffer by value, so
//! aliasing one owned buffer across two arms is a use-after-move.

use claspr::bundle;
use claspr::{Context, DeviceSlice};
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = kernels::kernels(&ctx).unwrap();
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, 4).unwrap();
    // First arm moves `buf`; second arm tries to use it again.
    let _ = bundle!(
        kernels.scale_u32([4usize], buf, 2u32),
        kernels.fill_u32([4usize], buf, 42u32),
    );
}
