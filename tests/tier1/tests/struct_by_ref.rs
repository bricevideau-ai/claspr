//! A user `#[repr(C)]` struct passed to kernels BY REFERENCE (`&Counter` /
//! `&mut Counter`), backed on the host by a `DeviceScalar<Counter>`.
//!
//! The kernel macro's scalar-by-reference role and the `DeviceScalar` bounds
//! are generic over the element type; these tests pin that a struct (not only
//! `u32`/`f32`) binds, reads and writes correctly through the Tier-1 terminals.

use claspr::{DeviceScalar, DeviceSlice};
use claspr_test_kernels::kernels::{self, Counter};
use claspr_test_support::ctx;

const N: usize = 256;

/// `&Counter`: a kernel reads a struct field.
#[test]
fn kernel_reads_struct_by_reference() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let counter =
        DeviceScalar::<Counter>::new(&ctx, Counter { value: 5, step: 0 }).expect("counter");
    let data = DeviceSlice::<u32>::from_slice(&ctx, &[10; N]).expect("data");

    let (data, _counter) = ks
        .add_counter_u32([N], data, counter)
        .wait()
        .expect("launch");

    let mut out = vec![0u32; N];
    data.read(&mut out).wait().expect("read");
    assert!(out.iter().all(|&v| v == 15), "got {:?}", &out[..4]);
}

/// `&mut Counter`: a kernel updates a struct field in place; the host reads the
/// whole struct back.
#[test]
fn kernel_writes_struct_by_reference() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("load kernels");
    let counter =
        DeviceScalar::<Counter>::new(&ctx, Counter { value: 1, step: 3 }).expect("counter");

    let counter = ks.advance_counter([1], counter).wait().expect("advance");
    let counter = ks
        .advance_counter([1], counter)
        .wait()
        .expect("advance again");

    assert_eq!(
        counter.read_value().expect("read"),
        Counter { value: 7, step: 3 }
    );
}
