//! Two `bundle!` arms that write the same owned `DeviceSlice` must
//! not type-check. The first arm moves `buf`; the second sees a
//! "used after move" error. Confirms move semantics catch
//! parallel-write aliasing on owned buffers.

use claspr::{Context, DeviceSlice};
use claspr_async::bundle;
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let kernels = kernels::kernels(&ctx).unwrap();
    let buf = DeviceSlice::<u32>::alloc(&ctx, 4).unwrap();
    // First arm moves `buf`; second arm tries to use it again.
    let _ = bundle!(
        kernels.scale_u32([4usize], buf, 2u32),
        kernels.fill_u32([4usize], buf, 42u32),
    );
}
