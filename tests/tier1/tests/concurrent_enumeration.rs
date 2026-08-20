//! Concurrent device enumeration — regression for the PoCL init race.
//!
//! PoCL's lazy device initialization races concurrent
//! `clGetDeviceIDs`: threads whose first call landed while another
//! thread's first call was still mid-init transiently saw ZERO
//! devices (recovering only ~hundreds of ms later). Before the
//! test-suite skips became loud, this presented as tests silently
//! "passing" by skipping. claspr serializes platform + device
//! enumeration behind `CL_ENUM_LOCK` (see `device.rs`); this pins it.
//!
//! Order matters: the threads spawn FIRST so the process's very first
//! enumeration is the concurrent one — a prior serial probe would
//! warm the runtime's init and mask a lock regression.

use claspr_test_support::ctx;

#[test]
fn concurrent_first_touch_sees_devices() {
    let handles: Vec<_> = (0..8)
        .map(|i| std::thread::spawn(move || (i, claspr::Device::all().map(|v| v.len()))))
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("join"))
        .collect();

    // Serial probe AFTER the race: the ground truth, and the skip /
    // CLASPR_REQUIRE_DEVICE gate for machines with no ICD.
    let Some(_ctx) = ctx() else { return };
    let expected = claspr::Device::all().expect("serial enumeration").len();
    assert!(expected >= 1);

    for (i, n) in results {
        let n = n.expect("threaded enumeration");
        assert_eq!(
            n, expected,
            "thread {i} saw {n} devices, serial probe saw {expected} — \
             enumeration raced the runtime's init"
        );
    }
}
