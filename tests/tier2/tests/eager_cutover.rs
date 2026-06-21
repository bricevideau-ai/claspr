//! Cutover step 1a: the eager struct-graph core ported into claspr-async,
//! exercised through the REAL runtime (non-kernel leaves: alloc + fill).
//! Proves the closure-free model + event-threaded pipes work on the actual
//! claspr-async `ExecutionContext`/queue before the macro change (step 1b).

use claspr::Context;
use claspr_async::eager::{EagerOpExt, alloc_zero, fill};

const N: usize = 256;

fn ctx() -> Option<Context> {
    match Context::any() {
        Ok(c) => Some(c),
        Err(_) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
    }
}

/// The graph is a closure-free struct: `description()` lists nodes with no
/// Context and no execution.
#[test]
fn describable_without_executing() {
    let g = alloc_zero::<u32, claspr::ReadWrite>(N).and_then(|b| fill(b, 7u32));
    assert_eq!(
        g.description(),
        vec!["alloc_zero(len=256)".to_string(), "fill".to_string()]
    );
}

/// alloc → fill, event-threaded, one terminal wait, on real hardware.
#[test]
fn alloc_then_fill_syncs() {
    let Some(ctx) = ctx() else { return };

    let buf = alloc_zero::<u32, claspr::ReadWrite>(N)
        .and_then(|b| fill(b, 0x55u32))
        .sync(&ctx)
        .expect("sync");

    let mut host = vec![0u32; N];
    buf.read(&mut host).wait().expect("read");
    assert!(
        host.iter().all(|&v| v == 0x55),
        "eager fill; got {:?}",
        &host[..8]
    );
}

/// A longer fill chain: each fill threads the prior fill's event as its
/// wait-list (non-blocking), final value wins. Ordering via threaded events.
#[test]
fn chained_fills_order_via_events() {
    let Some(ctx) = ctx() else { return };

    let buf = alloc_zero::<u32, claspr::ReadWrite>(N)
        .and_then(|b| fill(b, 1u32))
        .and_then(|b| fill(b, 2u32))
        .and_then(|b| fill(b, 3u32))
        .sync(&ctx)
        .expect("sync");

    let mut host = vec![0u32; N];
    buf.read(&mut host).wait().expect("read");
    assert!(
        host.iter().all(|&v| v == 3),
        "last fill wins via event ordering; got {:?}",
        &host[..8]
    );
}
