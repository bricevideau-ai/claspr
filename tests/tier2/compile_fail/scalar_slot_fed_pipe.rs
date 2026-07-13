//! A SCALAR slot tag must stay VALUE-ONLY: feeding it a `Pipe` (`F(pipe)`) must
//! be rejected at compile time. The unified `Tag(value)`/`Tag(pipe)` constructor
//! only grows a pipe-feed `CallArg` arm for BUFFER-valued tags (gated on
//! `RecordableBuffer`, which scalars like `f32` never impl), so `F<Pipe<f32>>`
//! has no `CallArg` — `call((F(pipe),))` can't be applied.
//!
//! This is the negative guard for the buffer/scalar asymmetry: pipe-acceptance is
//! type-driven and never leaks onto scalar/launch slots.

use claspr::eager::{DeviceOpExt, Pipe};
use claspr::{slot, slots, DeviceSlice};
use claspr_test_kernels::kernels;

slots! {
    // A buffer tag (pipe-feedable) and a SCALAR tag (value-only) in ONE block.
    Buf: DeviceSlice<u32>,
    F: f32,
}

fn main() {
    let ctx = claspr::Context::any().unwrap();
    let ks = kernels::kernels(&ctx).unwrap();

    // `F` is a scalar slot: it must NOT accept a pipe source. `F(scalar_pipe)`
    // constructs `F<Pipe<f32>>`, which has no `CallArg` (the pipe arm is gated to
    // buffer values), so `call` cannot apply it.
    let scalar_pipe: Pipe<f32> = Pipe::new();

    let _ = ks.scale_u32([4], slot!(Buf), 2u32).call((F(scalar_pipe),));
}
