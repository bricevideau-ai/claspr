//! `DeviceSliceHostView::release_to_device(self)` consumes the view.
//! Touching the view after release must not compile, because
//! `release_to_device(self)` moves `self` by value — a second use
//! is a use-after-move error.
//!
//! This is the borrow-check half of the host-view safety story:
//! once the view has been handed back to the device for downstream
//! stages, the host can no longer read it.
//!
//! The fixture exercises the type with a synthetic `fn` that takes
//! the view by value — `release_to_device(self)` consumes it, the
//! second use should fail use-after-move. Doing it this way avoids
//! the type-inference noise from materialising a real view through
//! the chain.

use claspr::ReadWrite;
use claspr_async::host_view::MapReadWrite;
use claspr_async::DeviceSliceHostView;

#[allow(dead_code)]
fn forbidden(view: DeviceSliceHostView<u32, ReadWrite, MapReadWrite>) {
    let _released = view.release_to_device();
    // `view` has been moved into `release_to_device`. Touching it
    // again must trigger a use-after-move error.
    let _peek: &[u32] = &view;
}

fn main() {}
