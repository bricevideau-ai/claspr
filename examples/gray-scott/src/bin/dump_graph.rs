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

use claspr::eager::{DeviceOp, DeviceOpExt, GraphNode, graph_edge_table};
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

    // ── (a) Indented struct tree. ───────────────────────────────────────────
    let mut nodes: Vec<GraphNode> = Vec::new();
    g.dump_graph(0, &mut nodes);

    println!("== gray-scott per-step graph: STRUCT TREE (source→next, depth-nested) ==");
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

    // ── (b) Flat pipe-edge table. ───────────────────────────────────────────
    println!();
    println!("== PIPE EDGE TABLE (producer cell -> consumer nodes) ==");
    let table = graph_edge_table(&nodes);
    let mut shared = 0usize;
    let mut edges = 0usize;
    for (producer, consumers) in &table {
        // Only real edges: a producer with at least one consumer.
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

    // ── (b') Fan-IN convergence view. ───────────────────────────────────────
    // In gray-scott the non-tree structure is a fan-IN diamond, not fan-OUT: no
    // single producer pipe feeds two consumers, but `combine` CONVERGES four
    // upstream pipes (both outputs of BOTH laplacians). A fork-tree (`cg`) never
    // has in-degree > 1 — every consumer reads exactly one producer. That
    // convergence across two independent laplacian branches is exactly the shared
    // edge the source→next maximal-span CB walk must order correctly.
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

    // ── (c) Summary. ────────────────────────────────────────────────────────
    println!();
    println!(
        "== SUMMARY ==\n{} nodes, {} pipe edges, {} shared (out-degree>1), \
         {} convergence points (in-degree>1)",
        nodes.len(),
        edges,
        shared,
        converge
    );

    Ok(())
}
