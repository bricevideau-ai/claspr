//! An uninitialised buffer is accepted only by a write-only `PerItemOut<T>`
//! parameter, which initialises it. Passing one to a parameter the kernel may
//! READ (`&[T]` / `&mut [T]`) must not compile: the kernel would read
//! uninitialised bytes.

use claspr::prelude::*;
use claspr_test_kernels::kernels;

fn main() {
    let ctx = Context::any().unwrap();
    let ks = kernels::kernels(&ctx).unwrap();
    let uninit = DeviceSlice::<u32>::alloc_uninit(&ctx, 4).unwrap();
    let b = DeviceSlice::<u32>::from_slice(&ctx, &[1; 4]).unwrap();
    let out = DeviceSlice::<u32>::alloc_zero(&ctx, 4).unwrap();
    let _ = ks.add_u32([4usize], uninit, b, out);
}
