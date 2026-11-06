//! Marking an uninitialised buffer initialised is the launch's job: it calls
//! `KernelSliceWriteArg::__claspr_into_init` only after its grid check passed.
//! The method is `unsafe`, so safe code can't use it to read uninitialised
//! memory through a `DeviceSlice`.

use claspr::{Context, DeviceSlice, KernelSliceWriteArg};

fn main() {
    let ctx = Context::any().unwrap();
    let uninit = DeviceSlice::<u32>::alloc_uninit(&ctx, 4).unwrap();
    let _init = KernelSliceWriteArg::<u32>::__claspr_into_init(uninit);
}
