//! `clEnqueueFillImage` reads four 32-bit components, typed by the format's
//! sampled family. A `Uint` image takes `[u32; 4]`: a `[u8; 4]` color (which
//! the driver would read 12 bytes past) is rejected.

use claspr::{image::format::R32Uint, Context, ReadWrite};

fn main() {
    let ctx = Context::any().unwrap();
    let img = claspr::Image2D::<ReadWrite, R32Uint>::alloc(&ctx, 4, 4).unwrap();
    let color: [u8; 4] = [0, 0, 0, 255];
    let _ = img.fill(color);
}
