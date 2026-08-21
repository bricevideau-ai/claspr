//! GENERIC-VALUE slot tags — the `slots! { Tag<R>: <value> }` arm.
//!
//! One module-level declaration serves every width a driver instantiates
//! (the `#[claspr::device(instantiate(...))]` companion): `Amt<u32>` and
//! `Amt<f64>` are independent slot identities (`Tag<KeyFor<Value>>`), so two
//! graphs stamped at different widths can never cross-match the "same" tag.
//!
//! Locks in:
//! - a generic scalar tag binds (`bind` and `mutate_bind`) and replays exactly
//!   like a concrete one, including CB-replayed graphs;
//! - two instantiations of ONE tag ident coexist in one process, each bound
//!   independently, with no cross-talk between their graphs;
//! - a generic BUFFER-valued tag (`GBuf<R>: DeviceSlice<R>`) works through the
//!   identity source (raw-value bind — `Checkout`/`Pipe` sources are
//!   deliberately unsupported on generic tags, see the macro arm docs).

use claspr::eager::DeviceOpExt;
use claspr::{DeviceSlice, slot, slots};
use claspr_test_kernels::{kernels, kernels_f64};
use claspr_test_support::{N, ctx, seeded};

// ONE declaration, every width: the whole point.
slots! {
    Amt<R>: R,
    GBuf<R>: DeviceSlice<R>,
}

/// Generic scalar tag: bind, sync, mutate, replay — the concrete-tag contract,
/// on a `u32` instantiation.
#[test]
fn generic_scalar_tag_binds_and_replays() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let buf = seeded(&ctx, 1);
    let g = ks.scale_u32([N], buf, slot!(Amt<u32>));

    let g = g.bind(Amt(3u32));
    let co = g.sync(&ctx).expect("sync 1");
    let v = co.map().wait().expect("read 1");
    assert!(v.iter().all(|&x| x == 3), "first sync: 1 * 3");
    drop(v);
    drop(co);

    g.mutate_call((Amt(5u32),)).expect("mutate");
    let co = g.sync(&ctx).expect("sync 2");
    let v = co.map().wait().expect("read 2");
    assert!(v.iter().all(|&x| x == 15), "replay after mutate: * 3 * 5");
}

/// Two instantiations of the SAME tag ident (`Amt<u32>`, `Amt<f64>`) drive two
/// graphs in one process — independent identities, independent binds.
#[test]
fn same_ident_two_widths_no_cross_talk() {
    let Some(ctx) = ctx() else { return };
    let has_f64 = ctx
        .device()
        .cl3()
        .double_fp_config()
        .map(|v| v != 0)
        .unwrap_or(false);
    if !has_f64 {
        eprintln!("SKIP: device has no Float64 capability");
        return;
    }
    let ks = kernels::kernels(&ctx).expect("load u32 kernels");
    let kf = kernels_f64::kernels(&ctx).expect("load f64 kernels");

    let bu = seeded(&ctx, 1);
    let bf = DeviceSlice::<f64>::alloc_zero(&ctx, N)
        .expect("alloc f64")
        .fill(1.0f64)
        .wait()
        .expect("seed f64");

    let gu = ks.scale_u32([N], bu, slot!(Amt<u32>)).bind(Amt(7u32));
    let gf = kf.scale_f64([N], bf, slot!(Amt<f64>)).bind(Amt(0.5f64));

    let cu = gu.sync(&ctx).expect("sync u32");
    let cf = gf.sync(&ctx).expect("sync f64");
    let vu = cu.map().wait().expect("read u32");
    let vf = cf.map().wait().expect("read f64");
    assert!(vu.iter().all(|&x| x == 7), "u32 graph saw its own bind");
    assert!(vf.iter().all(|&x| x == 0.5), "f64 graph saw its own bind");
}

/// Generic BUFFER-valued tag: the value type mentions the parameter
/// (`DeviceSlice<R>`); raw-value bind moves the buffer into the slot.
#[test]
fn generic_buffer_tag_binds() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");

    let buf = seeded(&ctx, 21);
    let g = ks.scale_u32([N], slot!(GBuf<u32>), 2u32);

    let g = g.bind(GBuf(buf));
    let co = g.sync(&ctx).expect("sync");
    let v = co.map().wait().expect("read");
    assert!(v.iter().all(|&x| x == 42), "buffer slot bound and scaled");
}
