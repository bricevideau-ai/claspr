//! Regression: slots inside `fan_out` branches must be bindable.
use claspr::eager::{DeviceOp, DeviceOpExt, fan_out, fill};
use claspr::{slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::{N, ctx, seeded};

slots! { Factor: u32 }

/// `FanOut` used to inherit the no-op `bind_slots` default, so a `slot!(Tag)` inside
/// a fan-out branch was invisible: `bind` reported `SlotNoSuchTag` for a tag the
/// graph plainly contained. Every branch must see the bind (a scalar fans out).
#[test]
fn fan_out_branches_bind_slots() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    // Each branch RESETS its buffer (fill) before scaling, so a replay is
    // idempotent and the two syncs below are directly comparable. Composite
    // branches also prove the walk descends past the branch's own root.
    let bufs: Vec<_> = (0..3).map(|_| seeded(&ctx, 0)).collect();
    let g = fan_out(bufs, |b| {
        fill(b, 2u32).and_then(|b| ks.scale_u32([N], b, slot!(Factor)))
    })
    // Opt out of static tracking: this test exercises the RUN-TIME checks.
    .erase_unchecked();

    // The bug: this returned Err(SlotNoSuchTag("Factor")).
    g.mutate_bind(Factor(5u32))
        .expect("bind reaches fan-out branches");
    let cos = g.sync(&ctx).expect("sync");
    for (i, co) in cos.iter().enumerate() {
        let v = co.map().wait().expect("read");
        assert!(v.iter().all(|&x| x == 10), "branch {i}: {:?}", &v[..4]);
    }
    drop(cos);

    // And a re-bind must reach every branch too (replay path).
    g.mutate_bind(Factor(3u32)).expect("re-bind");
    let cos = g.sync(&ctx).expect("sync 2");
    for (i, co) in cos.iter().enumerate() {
        let v = co.map().wait().expect("read 2");
        assert!(
            v.iter().all(|&x| x == 6),
            "branch {i} rebind: {:?}",
            &v[..4]
        );
    }
}

/// `describe` must show what the branches ARE, like the bundles do.
#[test]
fn fan_out_describe_recurses() {
    let Some(ctx) = ctx() else { return };
    let bufs: Vec<_> = (0..2).map(|_| seeded(&ctx, 0)).collect();
    let g = fan_out(bufs, |b| claspr::eager::fill(b, 1u32));
    let mut out = Vec::new();
    g.describe(&mut out);
    assert!(
        out.len() > 2,
        "describe should recurse into branches: {out:?}"
    );
    assert_eq!(
        out.iter().filter(|s| s.contains("fill")).count(),
        2,
        "{out:?}"
    );
}
