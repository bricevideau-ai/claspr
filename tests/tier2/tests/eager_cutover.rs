//! Cutover step 1a: the eager struct-graph core (now in `claspr`),
//! exercised through the REAL runtime (non-kernel leaves: alloc + fill).
//! Proves the closure-free model + event-threaded pipes work on the actual
//! claspr `ExecutionContext`/queue before the macro change (step 1b).

use claspr::eager::eager_copy_to;
use claspr::prelude::*;
use claspr_test_kernels::kernels;

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
    let g = alloc_zero::<u32>(N).and_then(|b| fill(b, 7u32));
    assert_eq!(
        g.description(),
        vec!["alloc_zero(len=256)".to_string(), "fill".to_string()]
    );
}

/// alloc → fill, event-threaded, one terminal wait, on real hardware.
#[test]
fn alloc_then_fill_syncs() {
    let Some(ctx) = ctx() else { return };

    let buf = alloc_zero::<u32>(N)
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

    let out: Vec<u32> = upload(vec![1u32; N])
        .and_then(|b| fill(b, 9u32))
        .and_then(download)
        .sync(&ctx)
        .expect("sync");

    assert_eq!(out.len(), N);
    assert!(
        out.iter().all(|&v| v == 9),
        "fill then download; got {:?}",
        &out[..8]
    );
}

/// **The headline: a KERNEL composes in an eager graph.** `kernels.fill_u32`
/// is now an `DeviceOp` — its buffer arg accepts the upstream `Pipe`, and it
/// deposits the buffer into its output pipe for the next stage. upload → kernel
/// (fill_u32 = 7) → kernel (scale_u32 ×3) → download = 21.
#[test]
fn kernel_composes_in_eager_graph() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");

    let out: Vec<u32> = upload(vec![0u32; N])
        .and_then(|b| ks.fill_u32([N], b, 7u32))
        .and_then(|b| ks.scale_u32([N], b, 3u32))
        .and_then(download)
        .sync(&ctx)
        .expect("sync");

    assert!(
        out.iter().all(|&v| v == 21),
        "fill 7 then ×3 = 21 via eager kernel chain; got {:?}",
        &out[..8]
    );
}

/// All-concrete eager kernel: pass a real buffer (not a pipe) straight into a
/// kernel as the chain head, proving `ToInput` accepts concrete in eager too.
#[test]
fn eager_kernel_concrete_head() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");
    let buf = claspr::DeviceSlice::<u32>::alloc_zero(&ctx, N).expect("buf");

    let out: Vec<u32> = ks
        .fill_u32([N], buf, 5u32)
        .and_then(download)
        .sync(&ctx)
        .expect("sync");
    assert!(
        out.iter().all(|&v| v == 5),
        "concrete-head kernel; got {:?}",
        &out[..8]
    );
}

/// Upload host data and read it straight back (no transform) — the upload
/// leaf's CL_MEM_COPY_HOST_PTR path preserves contents.
#[test]
fn upload_download_preserves_data() {
    let Some(ctx) = ctx() else { return };

    let src: Vec<u32> = (0..N as u32).collect();
    let out: Vec<u32> = upload(src.clone())
        .and_then(download)
        .sync(&ctx)
        .expect("sync");

    assert_eq!(out, src, "round-trip preserves data");
}

/// A longer fill chain: each fill threads the prior fill's event as its
/// wait-list (non-blocking), final value wins. Ordering via threaded events.
#[test]
fn chained_fills_order_via_events() {
    let Some(ctx) = ctx() else { return };

    let buf = alloc_zero::<u32>(N)
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

/// `value` lifts a host value; `arced` wraps an output in `Arc` for sharing.
#[test]
fn value_and_arced() {
    let Some(ctx) = ctx() else { return };

    // value: pure host value through the graph.
    let n: u32 = value(42u32).sync(&ctx).expect("value");
    assert_eq!(n, 42);

    // arced: wrap an uploaded buffer in Arc.
    let shared = arced(upload(vec![5u32; N])).sync(&ctx).expect("arced");
    // `shared` is an `Arc<DeviceSlice>` — a buffer verb would consume it, which
    // can't move out of the Arc. Inspect via a read map guard (borrows `&self`
    // through the Arc) instead.
    let g = shared.map().wait().expect("map");
    assert!(g.iter().all(|&v| v == 5), "arced buffer; got {:?}", &g[..8]);

    // `.arc()` method is the cuda-oxide-style alias for `arced(self)` — must be
    // equivalent. Same buffer, built via the method spelling.
    let shared2 = upload(vec![5u32; N]).arc().sync(&ctx).expect("arc method");
    let g2 = shared2.map().wait().expect("map");
    assert!(
        g2.iter().all(|&v| v == 5),
        "arc() buffer; got {:?}",
        &g2[..8]
    );

    // And it composes into `arc_split` exactly like `arced` does (one producer,
    // 2 read-only branches), proving the method drops into the same graph slot.
    // Both array slots are `Arc::clone`s of the SAME `cl_mem`.
    let [s1, s2] = arc_split::<2, _>(upload(vec![6u32; N]).arc())
        .sync(&ctx)
        .expect("arc() into arc_split");
    let (a, b) = (s1.map().wait().expect("map"), s2.map().wait().expect("map"));
    assert!(a.iter().all(|&v| v == 6) && b.iter().all(|&v| v == 6));
}

/// `bundle2`/`bundle3`: independent branches run and join; outputs tuple.
#[test]
fn bundles_join_branches() {
    let Some(ctx) = ctx() else { return };

    // Two independent download branches join into a tuple of Vecs.
    let (a, b) = bundle2(
        upload(vec![1u32; N])
            .and_then(|x| fill(x, 11u32))
            .and_then(download),
        upload(vec![2u32; N])
            .and_then(|x| fill(x, 22u32))
            .and_then(download),
    )
    .sync(&ctx)
    .expect("bundle2");
    assert!(a.iter().all(|&v| v == 11), "branch a; got {:?}", &a[..8]);
    assert!(b.iter().all(|&v| v == 22), "branch b; got {:?}", &b[..8]);

    let (x, y, z) = bundle3(value(1u32), value(2u32), value(3u32))
        .sync(&ctx)
        .expect("bundle3");
    assert_eq!((x, y, z), (1, 2, 3));
}

/// `fan_out`: one op per input (builder runs eagerly over the inputs), joined.
#[test]
fn fan_out_homogeneous() {
    let Some(ctx) = ctx() else { return };

    let vals: Vec<u32> = fan_out(vec![10u32, 20, 30], value)
        .sync(&ctx)
        .expect("fan_out");
    assert_eq!(vals, vec![10, 20, 30]);
}

/// fan_out of real device work: fill N buffers to distinct values, download all.
#[test]
fn fan_out_device_work() {
    let Some(ctx) = ctx() else { return };

    let outs: Vec<Vec<u32>> = fan_out(vec![1u32, 2u32, 3u32], |v| {
        upload(vec![0u32; 8])
            .and_then(move |b| fill(b, v))
            .and_then(download)
    })
    .sync(&ctx)
    .expect("fan_out device");
    assert_eq!(outs.len(), 3);
    assert!(outs[0].iter().all(|&v| v == 1));
    assert!(outs[1].iter().all(|&v| v == 2));
    assert!(outs[2].iter().all(|&v| v == 3));
}

/// **The keystone: a MULTI-OUTPUT kernel composes in an eager graph.**
/// `add_u32(a, b, out)` has `Output = (DeviceSlice, DeviceSlice, DeviceSlice)`,
/// so its `Handle` is a TUPLE OF PIPES `(Pipe<a>, Pipe<b>, Pipe<out>)`. The
/// downstream `and_then(|(_a, _b, out)| download(out))` selects the `out` pipe
/// and drops the other two (move-once: the dropped element pipes are never
/// `take`n). Proves both halves of the contract: per-element selection in
/// `and_then` AND terminal reconstruct via the overridden `into_output`.
#[test]
fn multi_output_kernel_element_select() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");

    // Concrete buffers as the kernel's chain head (ToInput accepts concrete in
    // eager too). `a = 3`, `b = 4` → `out = 7` element-wise.
    let a = upload(vec![3u32; N]).sync(&ctx).expect("upload a");
    let b = upload(vec![4u32; N]).sync(&ctx).expect("upload b");
    let out = alloc_zero::<u32>(N).sync(&ctx).expect("alloc out");

    let result: Vec<u32> = ks
        .add_u32([N], a, b, out)
        .and_then(|(_a, _b, out)| download(out))
        .sync(&ctx)
        .expect("sync");
    assert!(
        result.iter().all(|&v| v == 7),
        "multi-output add_u32 (3+4); got {:?}",
        &result[..8]
    );
}

/// Multi-output kernel as a TERMINAL: `.sync()` reconstructs the full
/// `(a, b, out)` tuple by draining all three element pipes (the
/// `into_output` override), proving the Tier-1 whole-tuple contract.
#[test]
fn multi_output_kernel_terminal_tuple() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");

    let a = upload(vec![5u32; N]).sync(&ctx).expect("upload a");
    let b = upload(vec![6u32; N]).sync(&ctx).expect("upload b");
    let out = alloc_zero::<u32>(N).sync(&ctx).expect("alloc out");

    // No downstream and_then — the kernel itself is the terminal; sync must
    // reconstruct the (a, b, out) tuple.
    let (_a, _b, out) = ks.add_u32([N], a, b, out).sync(&ctx).expect("sync tuple");
    let result = download(out).sync(&ctx).expect("download");
    assert!(
        result.iter().all(|&v| v == 11),
        "multi-output terminal tuple (5+6); got {:?}",
        &result[..8]
    );
}

/// **Arc fan-out: one shared buffer, N read-only consumers.** `arced` an
/// uploaded buffer, then `arc_split::<2>` fans it to two branches. Each branch
/// gets its OWN `Arc::clone` of the same device buffer (a cheap refcount bump)
/// via a `Pipe<Arc<DeviceSlice>>`, feeds it as the read-only source to a
/// `copy_u32` kernel into its own fresh destination, and downloads. Both
/// branches must observe the identical shared input — proving N consumers each
/// receive a usable clone of one producer's output. Mirrors the closure layer's
/// `value(v).arc().and_then(|a| { let [a,b] = a.split::<2>(); … })`.
#[test]
fn arc_split_read_only_fan_out() {
    let Some(ctx) = ctx() else { return };
    let ks = kernels::kernels(&ctx).expect("kernels");

    // Shared, read-only input: 0,1,2,…,N-1.
    let src: Vec<u32> = (0..N as u32).collect();

    let (out_a, out_b) = arc_split::<2, _>(arced(upload(src.clone())))
        .and_then(|[a, b]| {
            // Each branch owns one Arc clone of the SAME device buffer and reads
            // it (read-only kernel arg) into its own private destination.
            let ks = &ks;
            bundle2(
                alloc_zero::<u32>(N)
                    .and_then(move |dst| ks.copy_u32([N], a, dst))
                    .and_then(|(_src, dst)| download(dst)),
                alloc_zero::<u32>(N)
                    .and_then(move |dst| ks.copy_u32([N], b, dst))
                    .and_then(|(_src, dst)| download(dst)),
            )
        })
        .sync(&ctx)
        .expect("arc_split fan-out");

    assert_eq!(out_a, src, "branch a saw the shared input");
    assert_eq!(out_b, src, "branch b saw the shared input");
}

/// `arc_split` as the TERMINAL: with no downstream `and_then`, `.sync()`
/// reconstructs the `[Arc<T>; N]` array (the `into_output` override drains all
/// N element pipes). Each array element is a clone of the same producer output;
/// `Arc::ptr_eq` confirms they point at one shared allocation, not copies.
#[test]
fn arc_split_terminal_array() {
    let Some(ctx) = ctx() else { return };

    let [a, b, c] = arc_split::<3, _>(arced(upload(vec![9u32; N])))
        .sync(&ctx)
        .expect("arc_split terminal");

    // All three are clones of the one Arc the source produced.
    assert!(
        std::sync::Arc::ptr_eq(&a, &b),
        "a and b share one allocation"
    );
    assert!(
        std::sync::Arc::ptr_eq(&a, &c),
        "a and c share one allocation"
    );

    // `a` is an `Arc<DeviceSlice>` (shared) — inspect via a read map guard
    // (borrows `&self` through the Arc) rather than a consuming buffer verb.
    let g = a.map().wait().expect("map shared buffer");
    assert!(
        g.iter().all(|&v| v == 9),
        "shared buffer contents; got {:?}",
        &g[..8]
    );
}

/// Eager `copy_to` is a TWO-output op: `eager_copy_to(src, dst)` has
/// `Output = (DeviceSlice, DeviceSlice)`, so its `Handle` is `(Pipe<src>,
/// Pipe<dst>)` — the same per-element scatter the multi-output kernel uses. The
/// downstream `and_then(|(_src, dst)| download(dst))` selects the `dst` pipe and
/// drops the `src` pipe (move-once). Proves the copy port event-threads + the
/// copy actually moved the bytes (dst == src == all 7s).
#[test]
fn device_copy_eager() {
    let Some(ctx) = ctx() else { return };

    let src = upload(vec![7u32; N]).sync(&ctx).expect("upload src");
    let dst = alloc_zero::<u32>(N).sync(&ctx).expect("alloc dst");

    let result: Vec<u32> = eager_copy_to(src, dst)
        .and_then(|(_src, dst)| download(dst))
        .sync(&ctx)
        .expect("sync");
    assert!(
        result.iter().all(|&v| v == 7),
        "eager copy_to (DeviceSlice→DeviceSlice); got {:?}",
        &result[..8]
    );
}

// ── Execute-time closure nodes ─────────────────────────────────────────

/// `and_then_with_context`: the builder runs at EXECUTE with the live `ec`.
/// The closure reads `ec.device()` (proving the EC is in scope) then builds the
/// downstream `fill` over the upstream buffer. upload([1;N]) →
/// with_context(fill 9) → download = 9.
#[test]
fn eager_and_then_with_context() {
    let Some(ctx) = ctx() else { return };

    let out: Vec<u32> = upload(vec![1u32; N])
        .and_then_with_context(|ec, buf| {
            // Touch the live ExecutionContext to prove it's in scope at execute.
            let _dev = ec.device();
            fill(buf, 9u32)
        })
        .and_then(download)
        .sync(&ctx)
        .expect("sync");

    assert_eq!(out.len(), N);
    assert!(
        out.iter().all(|&v| v == 9),
        "and_then_with_context fill via live ec; got {:?}",
        &out[..8]
    );
}

/// `and_then_host`: the host seam. upload([1;N]) → host closure (`+= 1` over a
/// `&mut [u32]` view) → download. Proves the host saw the real data, mutations
/// persisted through the unmap, and the buffer forwards downstream unchanged in
/// identity. DeviceSlice's `Mappable::View` is `&mut [u32]` (read-write map).
#[test]
fn eager_and_then_host() {
    let Some(ctx) = ctx() else { return };

    let out: Vec<u32> = upload(vec![1u32; N])
        .and_then_host(|slice: &mut [u32]| {
            for x in slice.iter_mut() {
                *x += 1;
            }
            Ok(())
        })
        .and_then(download)
        .sync(&ctx)
        .expect("sync");

    assert_eq!(out.len(), N);
    assert!(
        out.iter().all(|&v| v == 2),
        "and_then_host mutation (1 + 1) persisted; got {:?}",
        &out[..8]
    );
}

/// `and_then_host` error propagation: the closure returns `Err`, which must
/// surface at the terminal as that error (no silent success).
#[test]
fn eager_and_then_host_error_propagates() {
    let Some(ctx) = ctx() else { return };

    let res = upload(vec![1u32; N])
        .and_then_host(|_slice: &mut [u32]| Err(claspr::Error::SvmNotAvailable))
        .and_then(download)
        .sync(&ctx);

    assert!(
        matches!(res, Err(claspr::Error::SvmNotAvailable)),
        "host closure Err must surface at the terminal; got {res:?}"
    );
}

/// `on_device`: route an op to a different device's queue. Only runs with a
/// two-device context (real multi-device or sub-device partition); single-device
/// runners skip it — there is no second queue to route to.
#[test]
fn eager_on_device() {
    use claspr::device::Platform;

    // Discover a two-device context: real multi-device → sub-device partition →
    // skip. Mirrors tests/tier2/tests/on_device.rs.
    fn ctx_two_devices() -> Option<Context> {
        if let Ok(platforms) = Platform::all() {
            for p in platforms {
                if let Ok(devs) = p.devices()
                    && devs.len() >= 2
                {
                    return Context::builder()
                        .devices(&[devs[0].clone(), devs[1].clone()])
                        .build()
                        .ok();
                }
            }
        }
        if let Ok(devs) = Device::all() {
            for parent in devs {
                if parent.partition_max_sub_devices().unwrap_or(0) < 2 {
                    continue;
                }
                let cu = parent.max_compute_units().unwrap_or(0);
                if cu < 2 {
                    continue;
                }
                let Ok(subs) = parent.partition_equally(cu / 2) else {
                    continue;
                };
                if subs.len() < 2 {
                    continue;
                }
                return Context::builder()
                    .devices(&[subs[0].clone(), subs[1].clone()])
                    .build()
                    .ok();
            }
        }
        eprintln!("SKIP: no two-device context for eager_on_device");
        None
    }

    let Some(ctx) = ctx_two_devices() else { return };

    // Two fill stages, each routed to a distinct device from the context, then
    // download. Device identity resolved from `ec` each stage (portable idiom).
    let out: Vec<u32> = upload(vec![0u32; N])
        .and_then_with_context(|ec, buf| fill(buf, 3u32).on_device(ec.device_at(0)))
        .and_then_with_context(|ec, buf| fill(buf, 7u32).on_device(ec.device_at(1)))
        .and_then(download)
        .sync(&ctx)
        .expect("on_device chain");

    assert_eq!(out.len(), N);
    assert!(
        out.iter().all(|&v| v == 7),
        "last routed fill wins; got {:?}",
        &out[..8]
    );
}
