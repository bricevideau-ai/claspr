//! Combinator-API spike — **now rebased onto real claspr + claspr-async**.
//!
//! Originally (2026-05-12) this was a 1300-line single-file prototype
//! that faked every Tier 2 primitive (`DeviceOperation`, `AndThen`,
//! `Bundle`, `FanOut`, `DynOp`, `HostAccessible`, ...) to validate
//! the type structure before any of it shipped. The design those 16
//! scenarios validated is now in the real `claspr` + `claspr-async`
//! crates; this file is the same 16 scenarios re-expressed against
//! the production API, running on actual OpenCL.
//!
//! It's a reference / regression program — `cargo run` from
//! `spikes/combinator/` exercises every combinator shape end-to-end.
//! The standalone `[workspace]` keeps it out of the main claspr
//! workspace per the spikes/README.md convention.
//!
//! Scenarios:
//!   1. Linear chain (producer/consumer pipeline)
//!   2. Independent parallel branches via `bundle!`
//!   3. Diamond (fan-out + fan-in via `Arc<[T]>`)
//!   4. ML forward pass (state carried through stages)
//!   5. In-place mutation chain
//!   6. N-ary fan-out via `fan_out` (variadic)
//!   7. Multi-producer, single consumer
//!   8. Mixed sync/async (split with host work between)
//!   9. Conditional graph via `DynOp` type erasure
//!  10. Error propagation through `and_then`
//!  11. Buffer round-trip (pass into chain, get back out)
//!  12. Profiling via `.profiled(|info| ...)` callback
//!  13. Batch parallelism via `fan_out` + implicit marker
//!  14. Cross-device pipeline (single context spans devices)
//!  15. `.and_then_host(|x| ...)` for in-queue host work
//!  16. `HostAccessible` — three-stage acquire / host / release
//!
//! Where the spike's original semantics needed faking that claspr
//! doesn't yet expose (sub-buffers from `split_into`, explicit
//! `transfer_to_device`), the scenario uses the closest production-
//! API equivalent and notes the gap in a comment.

use claspr::{Context, Device};
use claspr_async::{
    DeviceOperation, DeviceOperationHostExt, DeviceOperationProfileExt, DynOp, bundle, download,
    fan_out, transfer_to_device, upload, value,
};
use std::sync::Arc;

// ── Kernels (single device module) ──────────────────────────────────

#[claspr::device]
pub mod gpu {
    /// In-place elementwise multiply: `data[i] *= factor`.
    #[claspr::kernel]
    pub fn scale(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [f32],
        factor: f32,
    ) {
        let i = id.x;
        data[i] = data[i] * factor;
    }

    /// In-place scalar add: `data[i] += bias`.
    #[claspr::kernel]
    pub fn add_bias(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [f32],
        bias: f32,
    ) {
        let i = id.x;
        data[i] = data[i] + bias;
    }

    /// Elementwise add: `out[i] += b[i]`.
    #[claspr::kernel]
    pub fn add_inplace(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [f32],
        #[spirv(cross_workgroup)] b: &[f32],
    ) {
        let i = id.x;
        out[i] = out[i] + b[i];
    }

    /// Three-way mean: `out[i] = (out[i] + a[i] + b[i]) / 3`. `out`
    /// is pre-initialised to `c` on the host side before launch; the
    /// kernel folds in `a` and `b`. Used by scenario 7.
    #[claspr::kernel]
    pub fn mean3(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [f32],
        #[spirv(cross_workgroup)] a: &[f32],
        #[spirv(cross_workgroup)] b: &[f32],
    ) {
        let i = id.x;
        out[i] = (out[i] + a[i] + b[i]) / 3.0;
    }

    /// `out[i] = shared[i % shared_len] + bias`. Models the
    /// shared-input read pattern in scenarios 3 (diamond) and 4 (ML
    /// pass).
    #[claspr::kernel]
    pub fn add_shared_bias(
        #[spirv(global_invocation_id)] id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] out: &mut [f32],
        #[spirv(cross_workgroup)] shared: &[f32],
        shared_len: u32,
        bias: f32,
    ) {
        let i = id.x;
        let s = shared[(i as u32 % shared_len) as usize];
        out[i] = s + bias;
    }
}

const N: usize = 16;
const EPS: f32 = 1e-5;

// ── Scenarios ───────────────────────────────────────────────────────

fn scenario_1_linear_chain(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 1: linear chain (producer/consumer) ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let result: Vec<f32> = upload(vec![1.0f32; N])
        .and_then(|buf| kernels_ref.scale([N], buf, 2.0))
        .and_then(|buf| kernels_ref.scale([N], buf, 0.5))
        .and_then(download)
        .sync(ctx)?;
    println!("  result[0..4] = {:?}", &result[..4]);
    assert!((result[0] - 1.0).abs() < EPS, "1 * 2 * 0.5 = 1");
    Ok(())
}

fn scenario_2_bundle_parallel(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 2: independent parallel branches via bundle! ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let (a, b): (Vec<f32>, Vec<f32>) = bundle!(
        upload(vec![1.0f32; N])
            .and_then(|buf| kernels_ref.add_bias([N], buf, 1.0))
            .and_then(download),
        upload(vec![10.0f32; N])
            .and_then(|buf| kernels_ref.add_bias([N], buf, -1.0))
            .and_then(download),
    )
    .sync(ctx)?;
    println!("  a[0]={} b[0]={}", a[0], b[0]);
    assert!((a[0] - 2.0).abs() < EPS);
    assert!((b[0] - 9.0).abs() < EPS);
    Ok(())
}

fn scenario_3_diamond(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 3: diamond (fan-out + fan-in via Arc<[f32]>) ===");
    // Two branches read the same shared input (each uploads its own
    // copy from a shared `Arc<[f32]>`); their outputs are combined.
    //
    // claspr's per-call Op consumes slice args by value, so we can't
    // share a single DeviceSlice across branches. Host-side Arc-share
    // (Arc<[f32]>) lets both branches' uploads borrow the same heap
    // allocation; the upload's keep-alive callback drops each clone
    // when its write completes.
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let shared: Arc<[f32]> = vec![5.0f32; N].into();
    let len = N as u32;
    let s1 = Arc::clone(&shared);
    let s2 = Arc::clone(&shared);

    let result: Vec<f32> = bundle!(
        bundle!(upload(s1), upload(vec![0.0f32; N])).and_then(move |(sh_buf, out)| kernels_ref
            .add_shared_bias([N], out, sh_buf, len, 100.0)
            .and_then(|(out, _sh)| value(out))),
        bundle!(upload(s2), upload(vec![0.0f32; N])).and_then(move |(sh_buf, out)| kernels_ref
            .add_shared_bias([N], out, sh_buf, len, 200.0)
            .and_then(|(out, _sh)| value(out))),
    )
    .and_then(move |(a, b)| {
        kernels_ref
            .add_inplace([N], a, b)
            .and_then(|(out, _b)| value(out))
    })
    .and_then(download)
    .sync(ctx)?;
    println!("  combined[0..4] = {:?}", &result[..4]);
    // (5 + 100) + (5 + 200) = 310
    assert!((result[0] - 310.0).abs() < EPS);
    Ok(())
}

fn scenario_4_ml_forward_pass(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 4: ML forward pass (state carried through stages) ===");
    // Two layers of "weights" (w0, w1), an input, and a hidden/output
    // pair threaded through two stages. Each stage tuple-repacks the
    // surviving state for the next.
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let len = N as u32;

    let result: Vec<f32> = bundle!(
        upload(vec![0.1f32; N]), // w0
        upload(vec![0.2f32; N]), // w1
        upload(vec![1.0f32; N]), // input (kept for symmetry)
        upload(vec![0.0f32; N]), // hidden
    )
    .and_then(move |(w0, w1, _input, hidden)| {
        // Stage 1: hidden = w0 + 0.0
        kernels_ref
            .add_shared_bias([N], hidden, w0, len, 0.0)
            .and_then(move |(hidden, w0)| value((w0, w1, hidden)))
    })
    .and_then(move |(_w0, w1, hidden)| {
        // Stage 2: output = w1 + 0.5 (using a fresh output buf)
        upload(vec![0.0f32; N]).and_then(move |output_buf| {
            kernels_ref
                .add_shared_bias([N], output_buf, w1, len, 0.5)
                .and_then(move |(out, _w1)| value((hidden, out)))
        })
    })
    .and_then(|(_hidden, out)| download(out))
    .sync(ctx)?;
    println!("  output[0..4] = {:?}", &result[..4]);
    // w1 was 0.2; add_shared_bias writes shared[i] + 0.5 = 0.7
    assert!((result[0] - 0.7).abs() < EPS);
    Ok(())
}

fn scenario_5_in_place_mutation(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 5: in-place mutation chain ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let result: Vec<f32> = upload(vec![1.0f32; N])
        .and_then(|buf| kernels_ref.scale([N], buf, 2.0))
        .and_then(|buf| kernels_ref.scale([N], buf, 2.0))
        .and_then(|buf| kernels_ref.scale([N], buf, 0.25))
        .and_then(download)
        .sync(ctx)?;
    println!("  result[0..4] = {:?}", &result[..4]);
    assert!((result[0] - 1.0).abs() < EPS, "1 * 2 * 2 * 0.25 = 1");
    Ok(())
}

fn scenario_6_n_ary_fan_out(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 6: N-ary fan-out (tile-parallel) ===");
    // Spike used `split_into()` to chunk one buffer into 4
    // sub-buffers. claspr doesn't expose `clCreateSubBuffer` yet —
    // we model the shape with 4 separately-uploaded tile buffers.
    // Same topology (fan_out + per-tile op + collect), different
    // ownership story.
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let tiles_per: usize = 4;
    let inputs: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0f32; tiles_per]).collect();

    let tile_results: Vec<Vec<f32>> = fan_out(inputs, move |tile_input| {
        upload(tile_input)
            .and_then(move |buf| kernels_ref.scale([tiles_per], buf, 2.0))
            .and_then(download)
    })
    .sync(ctx)?;

    let combined: Vec<f32> = tile_results.into_iter().flatten().collect();
    println!("  combined len = {}", combined.len());
    assert_eq!(combined.len(), 4 * tiles_per);
    assert!((combined[0] - 2.0).abs() < EPS);
    Ok(())
}

fn scenario_7_multi_producer_single_consumer(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 7: multi-producer, single consumer ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let result: Vec<f32> = bundle!(
        upload(vec![1.0f32; N]),
        upload(vec![10.0f32; N]),
        upload(vec![100.0f32; N]), // c — becomes mean3's `out` input
    )
    .and_then(move |(a, b, out)| {
        kernels_ref
            .mean3([N], out, a, b)
            .and_then(|(out, _a, _b)| value(out))
    })
    .and_then(download)
    .sync(ctx)?;
    println!("  fused[0..4] = {:?}", &result[..4]);
    assert!((result[0] - 37.0).abs() < 1e-3, "(1+10+100)/3 ~ 37");
    Ok(())
}

fn scenario_8_split_await(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 8: mixed sync/async (split with host work between) ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    // First half via .sync() — caller holds buf back.
    let buf_a = upload(vec![1.0f32; N])
        .and_then(|buf| kernels_ref.scale([N], buf, 2.0))
        .sync(ctx)?;
    let host_factor = 3.0f32;
    // Second half resumes with the same buffer.
    let result: Vec<f32> = kernels_ref
        .scale([N], buf_a, host_factor)
        .and_then(download)
        .sync(ctx)?;
    println!("  result[0..4] = {:?}", &result[..4]);
    assert!((result[0] - 6.0).abs() < EPS, "1 * 2 * 3 = 6");
    Ok(())
}

fn scenario_9_conditional_graph(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 9: conditional graph via DynOp ===");
    // Two arms with different concrete op types — DynOp::new erases
    // both into a common DynOp<Vec<f32>>.
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let use_expensive = true;
    let chain: DynOp<Vec<f32>> = if use_expensive {
        DynOp::new(
            upload(vec![1.0f32; N])
                .and_then(|buf| kernels_ref.scale([N], buf, 2.0))
                .and_then(|buf| kernels_ref.scale([N], buf, 2.0))
                .and_then(download),
        )
    } else {
        DynOp::new(
            upload(vec![1.0f32; N])
                .and_then(|buf| kernels_ref.scale([N], buf, 4.0))
                .and_then(download),
        )
    };
    let result = chain.sync(ctx)?;
    println!("  result[0..4] = {:?}", &result[..4]);
    assert!((result[0] - 4.0).abs() < EPS);
    Ok(())
}

fn scenario_10_error_propagation(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 10: error propagation through and_then ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    // Under the new async and_then_host, closure error → user event
    // set to negative status → downstream device commands fail with
    // the same negative code. The specific Error::InvalidArgument
    // variant doesn't survive the user-event boundary; the chain
    // surfaces an Error::OpenCl(ClError(neg)) instead. The "should
    // not run" closure technically still runs (it produces an op
    // eagerly during execute()), but the produced kernel command
    // never actually fires — its wait-list events are failed.
    let result = upload(vec![1.0f32; N])
        .and_then(|buf| kernels_ref.scale([N], buf, 2.0))
        .and_then_host(|_slice: &mut [f32]| -> claspr::Result<()> {
            Err(claspr::Error::InvalidArgument("simulated stage failure"))
        })
        .and_then(|buf| kernels_ref.scale([N], buf, 2.0))
        .and_then(download)
        .sync(ctx);
    match result {
        Ok(_) => panic!("should have errored"),
        Err(e) => println!("  got expected error (negative cl status): {e}"),
    }
    Ok(())
}

fn scenario_11_buffer_round_trip(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 11: buffer round-trip (pass in, get back) ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let buf = upload(vec![1.0f32; N]).sync(ctx)?;
    let buf = kernels_ref
        .scale([N], buf, 2.0)
        .and_then(|buf| kernels_ref.scale([N], buf, 3.0))
        .sync(ctx)?;
    let result: Vec<f32> = value(buf).and_then(download).sync(ctx)?;
    println!("  result[0..4] = {:?}", &result[..4]);
    assert!((result[0] - 6.0).abs() < EPS, "1 * 2 * 3 = 6");
    Ok(())
}

fn scenario_12_profiling(ctx_profiling: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 12: profiling via .profiled(|info| ...) callback ===");
    let kernels = gpu::kernels(ctx_profiling)?;
    let kernels_ref = &kernels;
    let (tx, rx) = std::sync::mpsc::channel();
    let _result: Vec<f32> = upload(vec![1.0f32; N])
        .and_then(|buf| kernels_ref.scale([N], buf, 2.0))
        .and_then(|buf| kernels_ref.scale([N], buf, 0.5))
        .profiled(move |info| {
            tx.send(info).expect("send profiling info");
        })
        .and_then(download)
        .sync(ctx_profiling)?;
    let info = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("callback fired")
        .expect("profiling Ok");
    println!(
        "  profiling: queued={} submit={} start={} end={}",
        info.queued, info.submit, info.start, info.end,
    );
    assert!(info.queued <= info.submit && info.submit <= info.start && info.start <= info.end);
    Ok(())
}

fn scenario_13_batch_parallelism(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 13: batch parallelism via fan_out + implicit marker ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    let weights: Arc<[f32]> = vec![0.5f32; N].into();
    let len = N as u32;

    let results: Vec<Vec<f32>> = fan_out((0..3).collect::<Vec<i32>>(), move |batch_idx| {
        let w = Arc::clone(&weights);
        bundle!(
            upload(vec![batch_idx as f32; N]),
            upload(w),
            upload(vec![0.0f32; N]),
        )
        .and_then(move |(input, w_buf, out)| {
            kernels_ref
                .add_shared_bias([N], out, w_buf, len, 0.0)
                .and_then(move |(out, _w)| {
                    kernels_ref
                        .add_inplace([N], out, input)
                        .and_then(|(out, _input)| value(out))
                })
        })
        .and_then(download)
    })
    .sync(ctx)?;

    for (i, r) in results.iter().enumerate() {
        println!("  batch {i}: {:?}", &r[..4]);
        // 0.5 (shared bias) + batch_idx (add_inplace from input)
        assert!((r[0] - (0.5 + i as f32)).abs() < EPS);
    }
    Ok(())
}

fn scenario_14_cross_device(ctx: &Context, devs: &[Device]) -> claspr::Result<()> {
    println!("\n=== Scenario 14: cross-device pipeline ===");
    if devs.len() < 2 {
        eprintln!("  SKIP: needs ≥2 devices/sub-devices");
        return Ok(());
    }
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;

    // Fully non-blocking cross-device pipeline. The two production
    // primitives that map to OpenCL's actual decomposition:
    //
    //   transfer_to_device(buf, &dev) — explicit cl_mem migration
    //       (clEnqueueMigrateMemObjects). May be a no-op or real
    //       data movement depending on topology; either way it's a
    //       queue command, not host-blocking.
    //   .on_device(&dev) — per-op kernel routing (the kernel
    //       enqueues on `dev`'s default OOO queue).
    //
    // Device handles come from `ec.context().devices()` (via
    // `ec.device_at(i)`) inside `.and_then_with_context` closures,
    // not from external captures — the chain is portable across
    // contexts and doesn't assume "upload landed buf on devs[0]"
    // (upload lands on context.device(), which may be either).
    let result: Vec<f32> = upload(vec![1.0f32; N])
        .and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(0)))
        .and_then_with_context(move |ec, buf| {
            kernels_ref.scale([N], buf, 2.0).on_device(ec.device_at(0))
        })
        .and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(1)))
        .and_then_with_context(move |ec, buf| {
            kernels_ref.scale([N], buf, 10.0).on_device(ec.device_at(1))
        })
        // Migrate back before download (mirrors the original spike's
        // terminal transfer).
        .and_then_with_context(|ec, buf| transfer_to_device(buf, ec.device_at(0)))
        .and_then(download)
        .sync(ctx)?;
    println!("  result[0..4] = {:?}", &result[..4]);
    assert!((result[0] - 20.0).abs() < EPS, "1 * 2 * 10 = 20");
    Ok(())
}

fn scenario_15_and_then_host(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 15: .and_then_host(|x| ...) — in-queue host work ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    // Under the async and_then_host: the closure runs on a worker
    // thread, in queue order between two device stages. The mapped
    // view (`&mut [f32]`) is the buffer's mapped memory — writes
    // commit back via the queued unmap. The next kernel sees them.
    //
    // Note: feeding a host-computed value to a *scalar argument* of
    // the next kernel doesn't work, because the next `.and_then`
    // closure runs at execute() time (before the worker fires).
    // Pass host-computed state through device memory instead — i.e.
    // write it into the buffer itself, here just by tripling every
    // element in place.
    let result: Vec<f32> = upload(vec![1.0f32; N])
        .and_then(|buf| kernels_ref.scale([N], buf, 2.0)) // [2; N]
        .and_then_host(|slice: &mut [f32]| {
            // In-queue host work: triple every element in place.
            for x in slice.iter_mut() {
                *x *= 3.0;
            }
            eprintln!("  host modified slice[0] = {}", slice[0]);
            Ok(())
        })
        .and_then(download)
        .sync(ctx)?;
    println!("  result[0..4] = {:?}", &result[..4]);
    assert!((result[0] - 6.0).abs() < EPS, "1 * 2 * 3 = 6");
    Ok(())
}

fn scenario_16_host_accessible(ctx: &Context) -> claspr::Result<()> {
    println!("\n=== Scenario 16: HostAccessible — direct map via and_then_host ===");
    let kernels = gpu::kernels(ctx)?;
    let kernels_ref = &kernels;
    // The old "acquire_host_view → and_then_host(view) → release"
    // three-stage pattern is now subsumed by `.and_then_host` directly
    // on the `DeviceSlice` — the closure receives a mapped `&mut [T]`,
    // mutations are committed back on the next stage via the unmap
    // command that's already queued.
    let result: Vec<f32> = upload(vec![1.0f32; N])
        .and_then(|buf| kernels_ref.scale([N], buf, 2.0)) // [2; N]
        .and_then_host(|slice: &mut [f32]| {
            slice[0] += 100.0;
            eprintln!("  host modified slice[0] = {}", slice[0]);
            Ok(())
        })
        .and_then(|buf| kernels_ref.scale([N], buf, 0.5)) // GPU again
        .and_then(download)
        .sync(ctx)?;
    println!("  result[0..4] = {:?}", &result[..4]);
    assert!((result[0] - 51.0).abs() < EPS, "(2 + 100) * 0.5 = 51");
    assert!((result[1] - 1.0).abs() < EPS, "2 * 0.5 = 1");
    Ok(())
}

// ── main ────────────────────────────────────────────────────────────

fn pick_devices_for_cross_device() -> claspr::Result<Vec<Device>> {
    use claspr::device::Platform;
    // Same 3-stage discovery as the multi_device tests: real ≥2 →
    // sub-devices → fall back to single (scenario 14 will SKIP).
    if let Ok(platforms) = Platform::all() {
        for p in platforms {
            if let Ok(devs) = p.devices()
                && devs.len() >= 2
            {
                return Ok(vec![devs[0].clone(), devs[1].clone()]);
            }
        }
    }
    if let Ok(parent) = Device::any() {
        if parent.partition_max_sub_devices().unwrap_or(0) >= 2 {
            let cu = parent.max_compute_units().unwrap_or(0);
            if cu >= 2
                && let Ok(subs) = parent.partition_equally(cu / 2)
                && subs.len() >= 2
            {
                return Ok(vec![subs[0].clone(), subs[1].clone()]);
            }
        }
        return Ok(vec![parent]);
    }
    Ok(Vec::new())
}

fn main() -> claspr::Result<()> {
    let Ok(ctx) = Context::any() else {
        eprintln!("SKIP: no OpenCL device");
        return Ok(());
    };
    let ctx_profiling = Context::builder()
        .device(ctx.device())
        .profiling(true)
        .build()?;
    let cross_devs = pick_devices_for_cross_device()?;
    let cross_ctx = if cross_devs.len() >= 2 {
        Some(Context::builder().devices(&cross_devs).build()?)
    } else {
        None
    };

    scenario_1_linear_chain(&ctx)?;
    scenario_2_bundle_parallel(&ctx)?;
    scenario_3_diamond(&ctx)?;
    scenario_4_ml_forward_pass(&ctx)?;
    scenario_5_in_place_mutation(&ctx)?;
    scenario_6_n_ary_fan_out(&ctx)?;
    scenario_7_multi_producer_single_consumer(&ctx)?;
    scenario_8_split_await(&ctx)?;
    scenario_9_conditional_graph(&ctx)?;
    scenario_10_error_propagation(&ctx)?;
    scenario_11_buffer_round_trip(&ctx)?;
    scenario_12_profiling(&ctx_profiling)?;
    scenario_13_batch_parallelism(&ctx)?;
    if let Some(ref cross_ctx) = cross_ctx {
        scenario_14_cross_device(cross_ctx, &cross_devs)?;
    } else {
        println!("\n=== Scenario 14: cross-device pipeline ===");
        eprintln!("  SKIP: no usable 2-device configuration");
    }
    scenario_15_and_then_host(&ctx)?;
    scenario_16_host_accessible(&ctx)?;

    println!("\n=== ALL 16 SCENARIOS PASSED on real claspr ===");
    Ok(())
}
