//! Sub-step-3 coverage: `.arc()` + `ArcSplit::split::<N>()`.
//!
//! Builds the shared-input fan-out pattern: produce some immutable
//! state on device, wrap in Arc, hand out N clones to N branches,
//! verify each branch saw the same input.

use claspr::Context;
use claspr_async::{ArcSplit, DeviceOperation, bundle, value};

#[test]
fn arc_split_into_three_branches_share_value() {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };

    let chain = value(vec![1u32, 2, 3, 4]).arc().and_then(|arc_vec| {
        let [a, b, c] = arc_vec.split::<3>();
        bundle!(
            a.and_then(|arc| value(arc.iter().sum::<u32>())),
            b.and_then(|arc| value(arc.iter().product::<u32>())),
            c.and_then(|arc| value(arc.len() as u32)),
        )
    });

    let (sum, product, len) = chain.sync(&ctx).expect("arc-split chain");
    assert_eq!(sum, 10);
    assert_eq!(product, 24);
    assert_eq!(len, 4);
}

#[test]
fn arc_split_single_does_not_panic() {
    // Edge case: N = 1 returns a one-element array. The Arc was
    // already an Arc; the split is just a no-op clone.
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let chain = value("only-input".to_string()).arc().and_then(|arc| {
        let [one] = arc.split::<1>();
        one.and_then(|s| value(s.len()))
    });
    assert_eq!(chain.sync(&ctx).expect("split 1"), 10);
}
