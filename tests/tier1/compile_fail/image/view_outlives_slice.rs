//! `Image1DBufferView` borrows the underlying `DeviceSlice` for
//! `'a`. Dropping the slice while the view is still in scope must
//! be a borrow-checker error — otherwise the view's cl_mem ref
//! would point at released storage at use time.

use claspr::{Context, DeviceSlice, Image1DBufferView, ReadWrite, image::format::R32Uint};

fn main() {
    let ctx = Context::any().unwrap();
    let view = {
        let slice = DeviceSlice::<u32>::from_slice(&ctx, &[0u32; 16]).unwrap();
        Image1DBufferView::<ReadWrite, R32Uint>::view_of(&slice).unwrap()
    };
    // `slice` is gone here — the view's `'a` borrow can't be honoured.
    let _ = view.width();
}
