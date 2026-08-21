//! `instantiate(...)` only makes sense on a device module — on the
//! fn form of `#[claspr::device]` there is nothing to stamp.

#[claspr::device(instantiate(Real = [f32]))]
fn helper(x: f32) -> f32 {
    x + 1.0
}

fn main() {}
