//@no-rustfix
//! The `and_then_host` closure must not be able to stash a borrow
//! of the mapped view into outer state — the view's lifetime is
//! confined to the closure body by the
//! `for<'a> FnOnce(View<'a>) -> Result<()>` HRTB.
//!
//! This fixture tries to push the `&mut [u32]` view into an outer
//! `Vec<&mut [u32]>`. It must not type-check, because `'a` is
//! universally quantified — no concrete outer lifetime can satisfy
//! the HRTB.
//!
//! The `no-rustfix` directive above suppresses ui_test's auto-fix
//! attempt: rustc emits a `MaybeIncorrect` suggestion to add `move`
//! to the closure, but that fix doesn't actually help — `view`'s
//! lifetime still can't escape. Without the directive, ui_test
//! would apply the suggestion, see the rewritten file still fail,
//! and report a spurious test failure.

use claspr::{Context, DeviceSlice};
use claspr_async::{upload, DeviceOperation, DeviceOperationHostExt};

fn main() {
    let ctx = Context::any().unwrap();
    let mut escaped: Vec<&mut [u32]> = Vec::new();
    let _ = upload(vec![1u32, 2, 3, 4])
        .and_then_host(|view| {
            // Try to leak the borrow past the closure body — must
            // be rejected.
            escaped.push(view);
            Ok(())
        })
        .sync(&ctx);
    let _ = escaped; // suppress unused warning
    let _ = ctx;
    let _: Option<DeviceSlice<u32>> = None;
}
