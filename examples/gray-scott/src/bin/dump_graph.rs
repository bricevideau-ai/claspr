//! Read-only graph-structure dump for the Gray-Scott per-step meta-kernel.
//!
//! This is a DEBUG / introspection tool — it builds the SAME three-dispatch DAG
//! `run_swap` builds in `main.rs` (lines ~426-453) and prints its STRUCTURE, so a
//! human can see *why* the naive command-buffer maximal-span walk misbehaves on
//! gray-scott (a genuine pipe-DAG) but is fine on `cg` (a fork-tree). It does NOT
//! run anything on the device and does NOT touch any execution / CB logic.
//!
//! The key thing the struct nesting (the `AndThen` source/next tree) CANNOT show
//! is the SHARED fan-out pipe edges: `bind(Grid(..))` fans one launch-spec to all
//! three dispatch sites, and each `laplacian` is multi-output (field-passthrough
//! pipe + laplacian-scratch pipe) with BOTH outputs threading into `combine`
//! (in-degree 4 from pipes). Those edges exist only as shared `Pipe` cell
//! identities (`Pipe::cell_id()` = `Arc::as_ptr`), which `dump_graph` surfaces as
//! the SAME cell id appearing in a producer's `out_cells` and several consumers'
//! `in_cells`. `graph_edge_table` then flags any producer with out-degree > 1.
//!
//! This binary carries its OWN copy of the `#[claspr::device] mod gpu` kernel
//! signatures (the host-side launcher methods are proc-macro-generated per crate
//! ROOT, and a `[[bin]]` is a separate root from `main.rs`) — it shares this
//! crate's `OUT_DIR/gpu.rs` via the single `build.rs`. The `g` construction below
//! is replicated verbatim from `run_swap`, so the dump reflects the real graph.

use claspr::eager::{DeviceOp, DeviceOpExt, GraphNode, bundle4, graph_edge_table};
use claspr::{Context, DeviceSlice, LaunchSpec};
use claspr::{slot, slots};

#[claspr::device]
mod gpu {
    pub const W: u32 = 256;
    pub const H: u32 = 256;
    pub const DT: f32 = 1.0;
    pub const DU: f32 = 0.16;
    pub const DV: f32 = 0.08;

    pub fn wrap(coord: u32, delta_is_plus: bool, extent: u32) -> u32 {
        if delta_is_plus {
            let c = coord + 1;
            if c == extent { 0 } else { c }
        } else if coord == 0 {
            extent - 1
        } else {
            coord - 1
        }
    }

    pub fn laplacian_at(field: &[f32], x: u32, y: u32, w: u32, h: u32) -> f32 {
        let i = (y * w + x) as usize;
        let left = (y * w + wrap(x, false, w)) as usize;
        let right = (y * w + wrap(x, true, w)) as usize;
        let up = (wrap(y, false, h) * w + x) as usize;
        let down = (wrap(y, true, h) * w + x) as usize;
        field[left] + field[right] + field[up] + field[down] - 4.0 * field[i]
    }

    #[claspr::kernel]
    pub fn laplacian(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] field_in: &[f32],
        #[spirv(cross_workgroup)] lap_out: &mut [f32],
    ) {
        let x = id.x as u32;
        let y = id.y as u32;
        if x >= W || y >= H {
            return;
        }
        let i = (y * W + x) as usize;
        lap_out[i] = laplacian_at(field_in, x, y, W, H);
    }

    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn combine(
        #[spirv(global_invocation_id)] id: spirv_std::glam::USizeVec3,
        #[spirv(cross_workgroup)] u_in: &[f32],
        #[spirv(cross_workgroup)] v_in: &[f32],
        #[spirv(cross_workgroup)] lap_u: &[f32],
        #[spirv(cross_workgroup)] lap_v: &[f32],
        #[spirv(cross_workgroup)] u_out: &mut [f32],
        #[spirv(cross_workgroup)] v_out: &mut [f32],
        feed_rate: f32,
        kill_rate: f32,
    ) {
        let x = id.x as u32;
        let y = id.y as u32;
        if x >= W || y >= H {
            return;
        }
        let i = (y * W + x) as usize;

        let u = u_in[i];
        let v = v_in[i];
        let uvv = u * v * v;

        let u_next = u + DT * (DU * lap_u[i] - uvv + feed_rate * (1.0 - u));
        let v_next = v + DT * (DV * lap_v[i] + uvv - (feed_rate + kill_rate) * v);

        u_out[i] = u_next;
        v_out[i] = v_next;
    }
}

const W: usize = 256;
const H: usize = 256;
const N: usize = W * H;

const F1: f32 = 0.060;
const K1: f32 = 0.062;

slots! {
    Grid: LaunchSpec,
    UIn:  DeviceSlice<f32>,
    UOut: DeviceSlice<f32>,
    VIn:  DeviceSlice<f32>,
    VOut: DeviceSlice<f32>,
    F:  f32,
    K:  f32,
}

/// Allocate a device field and seed it from a host `Vec<f32>` (same as main.rs).
fn seeded(ctx: &Context, data: Vec<f32>) -> claspr::Result<DeviceSlice<f32>> {
    let buf = DeviceSlice::<f32>::alloc_zero(ctx, data.len())?;
    buf.write(data).wait()
}

/// Short cell id — last 4 hex digits — so shared ids are visually obvious.
fn short(id: usize) -> String {
    format!("{:04x}", id & 0xffff)
}

fn short_list(ids: &[usize]) -> String {
    let inner: Vec<String> = ids.iter().map(|&i| short(i)).collect();
    format!("[{}]", inner.join(","))
}

/// Dump one built graph: struct tree (a), pipe-edge table (b), fan-in view (b'),
/// summary (c). Shared by the per-step (swap) and two-step (immutable) dumps.
fn dump<G: DeviceOp>(title: &str, g: &G) {
    let mut nodes: Vec<GraphNode> = Vec::new();
    g.dump_graph(0, &mut nodes);

    println!("== {title}: STRUCT TREE (source→next, depth-nested) ==");
    for n in &nodes {
        println!(
            "{}{}  out={} in={}  cb_addable={} seam={}",
            "  ".repeat(n.depth),
            n.name,
            short_list(&n.out_cells),
            short_list(&n.in_cells),
            n.cb_addable,
            n.seam,
        );
    }

    println!();
    println!("== PIPE EDGE TABLE (producer cell -> consumer nodes) ==");
    let table = graph_edge_table(&nodes);
    let mut shared = 0usize;
    let mut edges = 0usize;
    for (producer, consumers) in &table {
        if consumers.is_empty() {
            continue;
        }
        edges += 1;
        let consumer_names: Vec<String> =
            consumers.iter().map(|&i| nodes[i].name.clone()).collect();
        let flag = if consumers.len() > 1 {
            shared += 1;
            format!("  <-- SHARED FAN-OUT (out-degree {})", consumers.len())
        } else {
            String::new()
        };
        println!(
            "  {} -> [{}]{}",
            short(*producer),
            consumer_names.join(", "),
            flag
        );
    }

    println!();
    println!("== FAN-IN CONVERGENCE (consumer node <- N producer pipes) ==");
    let mut converge = 0usize;
    for n in &nodes {
        if n.in_cells.len() > 1 {
            converge += 1;
            println!(
                "  {} <- {} producer pipes {}  <-- CONVERGENCE (in-degree {})",
                n.name,
                n.in_cells.len(),
                short_list(&n.in_cells),
                n.in_cells.len()
            );
        }
    }
    if converge == 0 {
        println!("  (none — this is a fork-tree, every consumer reads one producer)");
    }

    println!();
    println!(
        "== SUMMARY ==\n{} nodes, {} pipe edges, {} shared (out-degree>1), \
         {} convergence points (in-degree>1)",
        nodes.len(),
        edges,
        shared,
        converge
    );
}

fn main() -> claspr::Result<()> {
    let ctx = match Context::any() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e}) — the dump needs a Context to alloc buffers");
            return Ok(());
        }
    };

    let ks = gpu::kernels(&ctx)?;

    // Concrete buffers, exactly as run_swap seeds them.
    let u_a = seeded(&ctx, vec![1.0f32; N])?;
    let u_b = seeded(&ctx, vec![0.0f32; N])?;
    let v_a = seeded(&ctx, vec![0.0f32; N])?;
    let v_b = seeded(&ctx, vec![0.0f32; N])?;
    let lap_u_buf = seeded(&ctx, vec![0.0f32; N])?;
    let lap_v_buf = seeded(&ctx, vec![0.0f32; N])?;

    // ── The SAME per-step DAG run_swap builds (main.rs ~426-453). ────────────
    // Three dispatches (lap_u, lap_v, combine); `bind(Grid(..))` fans one launch
    // spec to all three; each laplacian threads BOTH outputs (field + scratch)
    // forward as pipes; combine reads four upstream pipes.
    let g = ks
        .laplacian(slot!(Grid), slot!(UIn), lap_u_buf)
        .and_then(move |(u_in, lap_u)| {
            ks.laplacian(slot!(Grid), slot!(VIn), lap_v_buf)
                .and_then(move |(v_in, lap_v)| {
                    ks.combine(
                        slot!(Grid),
                        u_in,
                        v_in,
                        lap_u,
                        lap_v,
                        slot!(UOut),
                        slot!(VOut),
                        slot!(F),
                        slot!(K),
                    )
                })
        })
        .bind(F(F1))
        .bind(K(K1))
        .bind(Grid(LaunchSpec::from([W, H])))
        .bind(UIn(u_a))
        .bind(VIn(v_a))
        .bind(UOut(u_b))
        .bind(VOut(v_b));

    dump("PER-STEP graph (run_swap / single step)", &g);

    // ── The IMMUTABLE unroll-by-2 graph (run_immutable, main.rs ~666-728). ────
    // This is the structure that RACES on pocl. Two full per-step subgraphs are
    // composed with `and_then`: step 1 value-binds concrete field buffers; step 2
    // is FED from step 1's four output pipes (Tag(pipe) = FedByPipe), the crossed
    // A→B→A rotation. The dropped dependency is the cross-step edge: step 2's
    // laplacians read step 1's combine output, but the CB records their
    // sync_point_wait_list as empty. This dump shows whether that cross-step pipe
    // edge is even VISIBLE in the graph's in_cells (i.e. whether the FedByPipe
    // slot binding surfaces the producer's cell id) — the crux of the root cause.
    // Fresh Kernels — the per-step `g` above moved `ks` into its closures.
    let ks_imm = gpu::kernels(&ctx)?;
    let iu_a = seeded(&ctx, vec![1.0f32; N])?;
    let iu_b = seeded(&ctx, vec![0.0f32; N])?;
    let iv_a = seeded(&ctx, vec![0.0f32; N])?;
    let iv_b = seeded(&ctx, vec![0.0f32; N])?;
    let lap_u1 = seeded(&ctx, vec![0.0f32; N])?;
    let lap_v1 = seeded(&ctx, vec![0.0f32; N])?;
    let lap_u2 = seeded(&ctx, vec![0.0f32; N])?;
    let lap_v2 = seeded(&ctx, vec![0.0f32; N])?;

    // Per-step subgraph with the four field slots left OPEN (invariants Grid/F/K
    // curried in), matching run_immutable's `curried_kernel`. `ks` is captured;
    // the launchers clone the context internally, so the returned graph owns
    // everything it needs.
    let curried = |ks: &gpu::Kernels, lap_u, lap_v| {
        ks.laplacian(slot!(Grid), slot!(UIn), lap_u)
            .and_then(move |(u_in, lap_u_pipe)| {
                ks.laplacian(slot!(Grid), slot!(VIn), lap_v)
                    .and_then(move |(v_in, lap_v_pipe)| {
                        ks.combine(
                            slot!(Grid),
                            u_in,
                            v_in,
                            lap_u_pipe,
                            lap_v_pipe,
                            slot!(UOut),
                            slot!(VOut),
                            slot!(F),
                            slot!(K),
                        )
                        .and_then(
                            |(u_in, v_in, _lap_u, _lap_v, u_out, v_out)| {
                                bundle4(u_in, v_in, u_out, v_out)
                            },
                        )
                    })
            })
            .call((F(F1), K(K1), Grid(LaunchSpec::from([W, H]))))
    };

    // Compose step 1 (concrete bufs) THEN step 2 (fed from step 1's output pipes).
    let g_imm = curried(&ks_imm, lap_u1, lap_v1)
        .call((UIn(iu_a), VIn(iv_a), UOut(iu_b), VOut(iv_b)))
        .and_then(move |(u_a, v_a, u_b, v_b)| {
            curried(&ks_imm, lap_u2, lap_v2).call((
                UIn(u_b),  // read B (step-1 UOut pipe)
                VIn(v_b),  // read B (step-1 VOut pipe)
                UOut(u_a), // write A (step-1 UIn pipe)
                VOut(v_a), // write A (step-1 VIn pipe)
            ))
        });

    println!();
    println!("################################################################");
    println!();
    dump(
        "IMMUTABLE unroll-by-2 graph (run_immutable — the RACING one)",
        &g_imm,
    );

    Ok(())
}
