//! Nesting shape is not topology: `graph!` (right-nested `and_then`) and a
//! hand-written left-nested chain over the same steps run the same commands
//! and segment into the same command buffers — here a device span, a host
//! seam, and a second device span.

use claspr::eager::{DeviceOp, DeviceOpExt, upload};
use claspr::graph;
use claspr_test_kernels::kernels;
use claspr_test_support::{N, ctx};

fn plus_100(v: &mut [u32]) -> claspr::Result<()> {
    v.iter_mut().for_each(|x| *x += 100);
    Ok(())
}

fn cb_ids<O: DeviceOp>(g: &O) -> Vec<usize> {
    let mut ids = Vec::new();
    g.collect_cb_ids(&mut ids);
    ids
}

/// scale*3 → scale*5 → host(+100) → scale*2 → scale*7 over an upload of 1s
/// (re-seeded each replay):
/// ((1*3*5)+100)*2*7 = 1610, in both shapes, across replays; and both shapes
/// home the same number of command buffers (2 on a CB-capable device — one per
/// device span — 0 where command buffers are unsupported).
#[test]
fn left_and_right_nesting_segment_the_same() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let left = upload(vec![1u32; N])
        .and_then(|b| ks.scale_u32([N], b, 3u32))
        .and_then(|b| ks.scale_u32([N], b, 5u32))
        .and_then_host(plus_100)
        .and_then(|b| ks.scale_u32([N], b, 2u32))
        .and_then(|b| ks.scale_u32([N], b, 7u32));
    let right = graph! {
        let b = upload(vec![1u32; N]);
        let b = ks.scale_u32([N], b, 3u32);
        let b = ks.scale_u32([N], b, 5u32);
        let b = b.and_then_host(plus_100);
        let b = ks.scale_u32([N], b, 2u32);
        ks.scale_u32([N], b, 7u32)
    };

    for run in 1..=3 {
        for (name, got) in [("left", left.sync(&ctx)), ("right", right.sync(&ctx))] {
            let got = got.unwrap_or_else(|e| panic!("{name} run {run}: {e:?}"));
            let v = got.map().wait().expect("read");
            assert!(
                v.iter().all(|&x| x == 1610),
                "{name} run {run}: {:?}",
                &v[..4]
            );
        }
    }
    let (l, r) = (cb_ids(&left), cb_ids(&right));
    assert_eq!(l.len(), r.len(), "left homes {l:?}, right homes {r:?}");
    assert!(
        l.len() == 2 || l.is_empty(),
        "one CB per device span, or none: {l:?}"
    );
}

/// The common `upload → kernels` chain written fluently (left-nested): the
/// kernels after the non-recordable `upload` head are the `next`s of nested
/// `AndThen`s, yet they record into one command buffer like the right-nested
/// form — the upload runs outside it. 1 * 3 * 5 * 7 = 105 on every replay.
#[test]
fn a_left_nested_upload_chain_records_its_kernels() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let left = upload(vec![1u32; N])
        .and_then(|b| ks.scale_u32([N], b, 3u32))
        .and_then(|b| ks.scale_u32([N], b, 5u32))
        .and_then(|b| ks.scale_u32([N], b, 7u32));
    let right = graph! {
        let b = upload(vec![1u32; N]);
        let b = ks.scale_u32([N], b, 3u32);
        let b = ks.scale_u32([N], b, 5u32);
        ks.scale_u32([N], b, 7u32)
    };
    for run in 1..=3 {
        for (name, got) in [("left", left.sync(&ctx)), ("right", right.sync(&ctx))] {
            let got = got.unwrap_or_else(|e| panic!("{name} run {run}: {e:?}"));
            let v = got.map().wait().expect("read");
            assert!(
                v.iter().all(|&x| x == 105),
                "{name} run {run}: {:?}",
                &v[..4]
            );
        }
    }
    let (l, r) = (cb_ids(&left), cb_ids(&right));
    assert_eq!(l.len(), r.len(), "left homes {l:?}, right homes {r:?}");
    assert!(
        l.len() == 1 || l.is_empty(),
        "one CB for the three kernels, or none: {l:?}"
    );
}
