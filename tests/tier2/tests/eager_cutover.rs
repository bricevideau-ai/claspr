//! Cutover step 1a: the eager struct-graph core ported into claspr-async,
//! exercised through the REAL runtime (non-kernel leaves: alloc + fill).
//! Proves the closure-free model + event-threaded pipes work on the actual
//! claspr-async `ExecutionContext`/queue before the macro change (step 1b).

use claspr::Context;
use claspr::eager::{EagerOpExt, alloc_zero, download, fill, upload};

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

/// The canonical chain shape minus the kernel: upload → fill → download,
/// round-tripping host data through the eager graph. Proves the transfer
/// leaves (Upload/Download) port + event-thread correctly.
#[test]
fn upload_fill_download_roundtrip() {
    let Some(ctx) = ctx() else { return };

    let out: Vec<u32> = upload::<u32, claspr::ReadWrite, _>(vec![1u32; N])
        .and_then(|b| fill(b, 9u32))
        .and_then(|b| download(b))
        .sync(&ctx)
        .expect("sync");

    assert_eq!(out.len(), N);
    assert!(out.iter().all(|&v| v == 9), "fill then download; got {:?}", &out[..8]);
}

/// Upload host data and read it straight back (no transform) — the upload
/// leaf's CL_MEM_COPY_HOST_PTR path preserves contents.
#[test]
fn upload_download_preserves_data() {
    let Some(ctx) = ctx() else { return };

    let src: Vec<u32> = (0..N as u32).collect();
    let out: Vec<u32> = upload::<u32, claspr::ReadWrite, _>(src.clone())
        .and_then(|b| download(b))
        .sync(&ctx)
        .expect("sync");

    assert_eq!(out, src, "round-trip preserves data");
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
