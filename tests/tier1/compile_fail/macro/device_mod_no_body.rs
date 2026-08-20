//! `#[claspr::device] mod gpu;` (a file module, no inline body) can't
//! work: the macro has nowhere to inject the include! + `kernels()`
//! items and the build script has no body to lift. Both sides
//! historically did nothing, silently. The macro must error at the
//! module with the inline-body fix spelled out.

#![feature(proc_macro_hygiene)]

#[claspr::device]
mod gpu;

fn main() {}
