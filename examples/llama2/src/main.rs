//! A llama2.c-format transformer, built at run time from a text model spec.
//!
//! Inputs: a model spec (`stories15M.spec`: config + the per-token computation
//! as a list of ops), a llama2.c checkpoint for the weights, and its tokenizer.
//! The spec is read at run time and composed into ONE claspr graph that
//! computes one token; generation replays that graph once per token.
//!
//! ```text
//! cargo run --release -p llama2-example -- stories15M.bin tokenizer.bin [-n 64]
//! ```
//!
//! The math follows llama2.c's `run.c` step for step, so a greedy run (argmax
//! sampling) reproduces `./run stories15M.bin -t 0` (see the test at the end).
//!
//! ## The graph
//!
//! The spec's ops each map the graph's [`Stream`] — the residual stream (the
//! logits after `classifier`) plus the step state — to itself, so the builder
//! is a `match` per op folded into one graph: the number of layers and the op
//! sequence are data, not code. Weights are concrete buffers fixed at build
//! time; the KV caches are concrete buffers that persist across replays (each
//! token appends its row).
//!
//! ## Per-token state lives on the device
//!
//! The current token and its position are one device value, a
//! [`gpu::Step`] struct held in a `DeviceScalar` and passed to kernels by
//! reference (`&Step` / `&mut Step`), that flows through the graph as data next
//! to the residual stream: `embed` reads the token, every attention kernel reads the position,
//! and the final `argmax` writes the next token and `pos + 1` back. Nothing
//! the host binds changes between tokens, so on a device with command buffers
//! the whole token is recorded ONCE and every later sync replays it. (Binding
//! the token and position as scalar slots instead bakes them into the kernel
//! arguments, so the command-buffer cache records a new buffer per token.)

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::Arc;

use claspr::eager::{DeviceDynOp, DeviceOpExt, Pipe, bundle2, forward};
use claspr::{Context, DeviceScalar, DeviceSlice, Single, graph};

#[claspr::device]
mod gpu {
    use spirv_std::arch::opencl_std as ocl;
    use spirv_std::glam::USizeVec3;

    // The loops are `while` counters rather than `for j in 0..n`: rust-gpu
    // compiles the range iterator too, but this model then runs ~35% slower on
    // PoCL (measured: 95 vs 150 tok/s).

    /// The per-token state: the current token and its position. One device
    /// value shared by the host (which seeds it and reads it back) and the
    /// kernels (which read it, and `argmax` advances it).
    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    pub struct Step {
        pub token: u32,
        pub pos: u32,
    }

    /// `x = table[token]`: the current token's embedding row.
    #[claspr::kernel]
    pub fn embed(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] table: &[f32],
        #[spirv(cross_workgroup)] x: PerItem<f32>,
        #[spirv(cross_workgroup)] step: &Step,
        dim: u32,
    ) {
        *x = table[step.token as usize * dim as usize + x.index()];
    }

    /// `out = w * x / rms(x)`. Every work-item recomputes the sum of squares (`n`
    /// is the model width, a few hundred), which keeps the kernel free of
    /// work-group reductions.
    #[claspr::kernel]
    pub fn rmsnorm(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] x: &[f32],
        #[spirv(cross_workgroup)] w: PerItemIn<f32>,
        #[spirv(cross_workgroup)] out: PerItem<f32>,
        n: u32,
    ) {
        let i = out.index();
        let mut ss = 0.0f32;
        let mut j = 0usize;
        while j < n as usize {
            ss += x[j] * x[j];
            j += 1;
        }
        let scale = 1.0 / ocl::sqrt(ss / n as f32 + 1e-5);
        *out = *w * (scale * x[i]);
    }

    /// `out = W x` for a row-major `W` of `n` columns (one row per work-item).
    #[claspr::kernel]
    pub fn matmul(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] w: &[f32],
        #[spirv(cross_workgroup)] x: &[f32],
        #[spirv(cross_workgroup)] out: PerItem<f32>,
        n: u32,
    ) {
        let row = out.index() * n as usize;
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < n as usize {
            acc += w[row + j] * x[j];
            j += 1;
        }
        *out = acc;
    }

    /// The key and value projections of `x`, written straight into the current
    /// position's row of this layer's KV caches.
    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn kv(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] wk: &[f32],
        #[spirv(cross_workgroup)] wv: &[f32],
        #[spirv(cross_workgroup)] x: &[f32],
        #[spirv(cross_workgroup)] k_cache: &mut [f32],
        #[spirv(cross_workgroup)] v_cache: &mut [f32],
        #[spirv(cross_workgroup)] step: &Step,
        n: u32,
        kv_dim: u32,
    ) {
        let i = id.x;
        let row = i * n as usize;
        let mut k = 0.0f32;
        let mut v = 0.0f32;
        let mut j = 0usize;
        while j < n as usize {
            k += wk[row + j] * x[j];
            v += wv[row + j] * x[j];
            j += 1;
        }
        let at = step.pos as usize * kv_dim as usize + i;
        k_cache[at] = k;
        v_cache[at] = v;
    }

    /// RoPE: rotate each pair `(2i, 2i+1)` of the query, and of the current
    /// position's key in the cache, by the position's angle for that pair.
    #[claspr::kernel]
    pub fn rope(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] q: &mut [f32],
        #[spirv(cross_workgroup)] k_cache: &mut [f32],
        #[spirv(cross_workgroup)] step: &Step,
        kv_dim: u32,
        head_size: u32,
    ) {
        let i = 2 * id.x;
        let pos = step.pos;
        let head_dim = (i % head_size as usize) as f32;
        let freq = 1.0 / ocl::pow(10000.0f32, head_dim / head_size as f32);
        let val = pos as f32 * freq;
        let (fcr, fci) = (ocl::cos(val), ocl::sin(val));
        let (v0, v1) = (q[i], q[i + 1]);
        q[i] = v0 * fcr - v1 * fci;
        q[i + 1] = v0 * fci + v1 * fcr;
        if i < kv_dim as usize {
            let at = pos as usize * kv_dim as usize + i;
            let (v0, v1) = (k_cache[at], k_cache[at + 1]);
            k_cache[at] = v0 * fcr - v1 * fci;
            k_cache[at + 1] = v0 * fci + v1 * fcr;
        }
    }

    /// Attention scores `att[h, t] = q_h . k_t / sqrt(head_size)` for every head
    /// `h` and every past position `t <= pos` (one work-item per `(h, t)`).
    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn att_scores(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] q: &[f32],
        #[spirv(cross_workgroup)] k_cache: &[f32],
        #[spirv(cross_workgroup)] att: PerItem<f32>,
        #[spirv(cross_workgroup)] step: &Step,
        seq_len: u32,
        head_size: u32,
        kv_dim: u32,
        kv_mul: u32,
    ) {
        let (h, t) = (id.x / seq_len as usize, id.x % seq_len as usize);
        if t > step.pos as usize {
            return;
        }
        let hs = head_size as usize;
        let qo = h * hs;
        let ko = t * kv_dim as usize + (h / kv_mul as usize) * hs;
        let mut score = 0.0f32;
        let mut i = 0usize;
        while i < hs {
            score += q[qo + i] * k_cache[ko + i];
            i += 1;
        }
        *att = score / ocl::sqrt(head_size as f32);
    }

    /// Softmax of each head's scores over `t <= pos` (one work-item per head).
    #[claspr::kernel]
    pub fn softmax(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] att: &mut [f32],
        #[spirv(cross_workgroup)] step: &Step,
        seq_len: u32,
    ) {
        let row = id.x * seq_len as usize;
        let n = step.pos as usize + 1;
        let mut max = att[row];
        let mut t = 1usize;
        while t < n {
            if att[row + t] > max {
                max = att[row + t];
            }
            t += 1;
        }
        let mut sum = 0.0f32;
        t = 0;
        while t < n {
            let e = ocl::exp(att[row + t] - max);
            att[row + t] = e;
            sum += e;
            t += 1;
        }
        t = 0;
        while t < n {
            att[row + t] /= sum;
            t += 1;
        }
    }

    /// Each head's attention-weighted sum of the cached values, for `t <= pos`.
    #[claspr::kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn att_mix(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] att: &[f32],
        #[spirv(cross_workgroup)] v_cache: &[f32],
        #[spirv(cross_workgroup)] out: PerItem<f32>,
        #[spirv(cross_workgroup)] step: &Step,
        seq_len: u32,
        head_size: u32,
        kv_dim: u32,
        kv_mul: u32,
    ) {
        let hs = head_size as usize;
        let (h, j) = (id.x / hs, id.x % hs);
        let vo = (h / kv_mul as usize) * hs + j;
        let mut acc = 0.0f32;
        let mut t = 0usize;
        while t <= step.pos as usize {
            acc += att[h * seq_len as usize + t] * v_cache[t * kv_dim as usize + vo];
            t += 1;
        }
        *out = acc;
    }

    /// `x += y`: the residual connection.
    #[claspr::kernel]
    pub fn add(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] x: PerItem<f32>,
        #[spirv(cross_workgroup)] y: PerItemIn<f32>,
    ) {
        *x += *y;
    }

    /// `h1 = silu(h1) * h3`: the SwiGLU gate.
    #[claspr::kernel]
    pub fn swiglu(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] h1: PerItem<f32>,
        #[spirv(cross_workgroup)] h3: PerItemIn<f32>,
    ) {
        let v = *h1;
        *h1 = v * (1.0 / (1.0 + ocl::exp(-v))) * *h3;
    }

    /// Greedy sampling and the step advance, on the device (one work-item; ties
    /// keep the first index, like llama2.c): the next token is `argmax(logits)`,
    /// at `pos + 1`.
    /// The last kernel of the token, so every reader of `step` has run.
    #[claspr::kernel]
    pub fn argmax(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(cross_workgroup)] logits: &[f32],
        #[spirv(cross_workgroup)] step: &mut Step,
        n: u32,
    ) {
        if id.x != 0 {
            return;
        }
        let mut best = 0usize;
        let mut i = 1usize;
        while i < n as usize {
            if logits[i] > logits[best] {
                best = i;
            }
            i += 1;
        }
        step.token = best as u32;
        step.pos += 1;
    }
}

/// The model graph up to some op: the residual stream (the logits after
/// `classifier`) and the step state. ONE type whatever the spec says.
type Stream = DeviceDynOp<'static, (Single<DeviceSlice<f32>>, Single<DeviceScalar<gpu::Step>>)>;

// ── Model spec ─────────────────────────────────────────────────────────────

/// The model dimensions (the spec's config lines; must match the checkpoint).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
}

impl Config {
    fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }
    fn kv_dim(&self) -> usize {
        self.dim * self.n_kv_heads / self.n_heads
    }
    fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

/// One op of the spec's graph section.
#[derive(Debug)]
enum Op {
    Embed(String),
    /// Tensor names, in spec order: `[rms_att, wq, wk, wv, wo]`.
    Attention([String; 5]),
    /// Tensor names, in spec order: `[rms_ffn, w1, w2, w3]`.
    Ffn([String; 4]),
    RmsNorm(String),
    Classifier(String),
    Argmax,
    /// `repeat <count>` … `end`: the body once per layer.
    Repeat(usize, Vec<Op>),
}

struct Spec {
    config: Config,
    ops: Vec<Op>,
}

impl Spec {
    /// Whether the graph ends with `argmax`, which advances the step state —
    /// what generation needs.
    fn generates(&self) -> bool {
        matches!(self.ops.last(), Some(Op::Argmax))
    }
}

/// The spec's config lines (`<key> <number>`), one per [`Config`] field.
const CONFIG_KEYS: [&str; 7] = [
    "dim",
    "hidden_dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "vocab_size",
    "seq_len",
];

fn parse_spec(text: &str) -> Result<Spec, String> {
    let mut cfg: HashMap<String, usize> = HashMap::new();
    // A stack of op lists: the bottom is the top level, each `repeat` pushes one.
    let mut stack: Vec<(usize, Vec<Op>)> = vec![(0, Vec::new())];
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let err = |m: &str| format!("spec line {}: {m}: {raw:?}", n + 1);
        let names = |k: usize| -> Result<Vec<String>, String> {
            match words.len() == k + 1 {
                true => Ok(words[1..].iter().map(|s| s.to_string()).collect()),
                false => Err(err(&format!("expected {k} tensor name(s)"))),
            }
        };
        let op = match words[0] {
            "embed" => Op::Embed(names(1)?.remove(0)),
            "rmsnorm" => Op::RmsNorm(names(1)?.remove(0)),
            "classifier" => Op::Classifier(names(1)?.remove(0)),
            "argmax" => {
                names(0)?;
                Op::Argmax
            }
            "attention" => Op::Attention(names(5)?.try_into().unwrap()),
            "ffn" => Op::Ffn(names(4)?.try_into().unwrap()),
            "repeat" => {
                if stack.len() > 1 {
                    return Err(err("nested `repeat`"));
                }
                let w = names(1)?.remove(0);
                let count = w
                    .parse()
                    .ok()
                    .or_else(|| cfg.get(&w).copied())
                    .ok_or_else(|| err("unknown repeat count"))?;
                stack.push((count, Vec::new()));
                continue;
            }
            "end" => {
                let (count, body) = stack
                    .pop()
                    .filter(|_| !stack.is_empty())
                    .ok_or_else(|| err("`end` without `repeat`"))?;
                Op::Repeat(count, body)
            }
            key if CONFIG_KEYS.contains(&key) && words.len() == 2 => {
                let v = words[1].parse().map_err(|_| err("expected a number"))?;
                cfg.insert(key.to_string(), v);
                continue;
            }
            _ => return Err(err("unknown op")),
        };
        stack.last_mut().unwrap().1.push(op);
    }
    if stack.len() != 1 {
        return Err("unterminated `repeat`".into());
    }
    let get = |k: &str| {
        cfg.get(k)
            .copied()
            .ok_or_else(|| format!("spec: missing `{k}`"))
    };
    let config = Config {
        dim: get("dim")?,
        hidden_dim: get("hidden_dim")?,
        n_layers: get("n_layers")?,
        n_heads: get("n_heads")?,
        n_kv_heads: get("n_kv_heads")?,
        vocab_size: get("vocab_size")?,
        seq_len: get("seq_len")?,
    };
    let ops = stack.pop().unwrap().1;
    // `embed` opens the graph and `argmax`, if any, closes it; neither repeats.
    let misplaced = |(i, op): (usize, &Op)| match op {
        Op::Embed(_) => i != 0,
        Op::Argmax => i != ops.len() - 1,
        Op::Repeat(_, body) => body.iter().any(|o| matches!(o, Op::Embed(_) | Op::Argmax)),
        _ => false,
    };
    if !matches!(ops.first(), Some(Op::Embed(_))) || ops.iter().enumerate().any(misplaced) {
        return Err(
            "spec: the graph must start with `embed` and may end with `argmax`, \
                    each once and outside `repeat`"
                .into(),
        );
    }
    Ok(Spec { config, ops })
}

// ── Checkpoint ─────────────────────────────────────────────────────────────

/// The tensors of a llama2.c checkpoint, by name. Per-layer tensors hold all
/// layers back to back.
struct Weights {
    config: Config,
    tensors: HashMap<&'static str, Vec<f32>>,
}

/// The tensors stored once per layer (indexed by the `repeat` iteration).
const PER_LAYER: [&str; 9] = [
    "rms_att", "wq", "wk", "wv", "wo", "rms_ffn", "w1", "w2", "w3",
];

impl Weights {
    /// Parse llama2.c's checkpoint format: a 7-int header, then the tensors in
    /// a fixed order. A negative vocab size means a separate classifier
    /// (`wcls`); otherwise the classifier IS the token embedding (the spec then
    /// names `token_embedding` for both).
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let int = |i: usize| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        if bytes.len() < 28 {
            return Err("checkpoint too short".into());
        }
        let shared = int(5) > 0;
        let config = Config {
            dim: int(0) as usize,
            hidden_dim: int(1) as usize,
            n_layers: int(2) as usize,
            n_heads: int(3) as usize,
            n_kv_heads: int(4) as usize,
            vocab_size: int(5).unsigned_abs() as usize,
            seq_len: int(6) as usize,
        };
        let (d, h, l, kv) = (
            config.dim,
            config.hidden_dim,
            config.n_layers,
            config.kv_dim(),
        );
        let mut layout = vec![
            ("token_embedding", config.vocab_size * d),
            ("rms_att", l * d),
            ("wq", l * d * d),
            ("wk", l * d * kv),
            ("wv", l * d * kv),
            ("wo", l * d * d),
            ("rms_ffn", l * d),
            ("w1", l * h * d),
            ("w2", l * d * h),
            ("w3", l * h * d),
            ("rms_final", d),
            // The legacy precomputed RoPE tables, unused (RoPE is computed).
            ("freq_cis", config.seq_len * config.head_size()),
        ];
        if !shared {
            layout.push(("wcls", config.vocab_size * d));
        }
        let floats: Vec<f32> = bytes[28..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut tensors = HashMap::new();
        let mut at = 0;
        for (name, len) in layout {
            let t = floats.get(at..at + len).ok_or("checkpoint truncated")?;
            tensors.insert(name, t.to_vec());
            at += len;
        }
        Ok(Weights { config, tensors })
    }

    /// The tensor `name`, restricted to `layer` if it is a per-layer tensor.
    fn get(&self, name: &str, layer: Option<usize>) -> Result<&[f32], String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("no tensor `{name}`"))?;
        match (PER_LAYER.contains(&name), layer) {
            (true, Some(l)) => {
                let len = t.len() / self.config.n_layers;
                Ok(&t[l * len..(l + 1) * len])
            }
            (true, None) => Err(format!("per-layer tensor `{name}` used outside `repeat`")),
            (false, _) => Ok(t),
        }
    }
}

// ── Graph builder ──────────────────────────────────────────────────────────

/// BOS: llama2.c starts generation from it and stops when it comes back.
const BOS: u32 = 1;

/// Builds the model graph from the spec: device buffers for the weights, KV
/// caches and scratch, then one stage per op.
struct Builder<'a> {
    ctx: &'a Context,
    ks: &'a gpu::Kernels,
    weights: &'a Weights,
    cfg: Config,
    /// Whole-model tensors already on the device, shared by every op that
    /// names them (stories15M's classifier reuses the token embedding).
    shared: std::cell::RefCell<HashMap<String, Arc<DeviceSlice<f32>>>>,
}

impl Builder<'_> {
    fn weight(&self, name: &str, layer: Option<usize>) -> Result<DeviceSlice<f32>, String> {
        let data = self.weights.get(name, layer)?;
        DeviceSlice::from_slice(self.ctx, data).map_err(|e| format!("upload `{name}`: {e:?}"))
    }

    /// The tensors `names` (of `layer`, for per-layer ones), in order.
    fn weights<const N: usize>(
        &self,
        names: &[String; N],
        layer: Option<usize>,
    ) -> Result<[DeviceSlice<f32>; N], String> {
        let bufs: Vec<_> = names
            .iter()
            .map(|n| self.weight(n, layer))
            .collect::<Result<_, _>>()?;
        Ok(bufs.try_into().expect("one buffer per name"))
    }

    /// A whole-model (not per-layer) tensor, uploaded once and shared: each
    /// launch reading it gets an `Arc` of the same device buffer.
    fn shared_weight(&self, name: &str) -> Result<Arc<DeviceSlice<f32>>, String> {
        if let Some(buf) = self.shared.borrow().get(name) {
            return Ok(Arc::clone(buf));
        }
        let buf = Arc::new(self.weight(name, None)?);
        self.shared
            .borrow_mut()
            .insert(name.to_string(), Arc::clone(&buf));
        Ok(buf)
    }

    fn zeros(&self, len: usize) -> Result<DeviceSlice<f32>, String> {
        DeviceSlice::<f32>::alloc_zero(self.ctx, len).map_err(|e| format!("alloc: {e:?}"))
    }

    /// `N` zeroed scratch buffers of `len` floats.
    fn zeros_n<const N: usize>(&self, len: usize) -> Result<[DeviceSlice<f32>; N], String> {
        let bufs: Vec<_> = (0..N).map(|_| self.zeros(len)).collect::<Result<_, _>>()?;
        Ok(bufs.try_into().expect("N buffers"))
    }

    /// The whole graph section (validated by [`parse_spec`]).
    fn build(&self, ops: &[Op]) -> Result<Stream, String> {
        let Some((Op::Embed(table), rest)) = ops.split_first() else {
            return Err("spec: the graph must start with `embed`".into());
        };
        let g = self.embed(table)?;
        self.ops(g, rest, None)
    }

    fn ops(&self, mut g: Stream, ops: &[Op], layer: Option<usize>) -> Result<Stream, String> {
        for op in ops {
            g = match op {
                Op::Embed(_) => unreachable!("parse_spec puts `embed` first, once"),
                Op::Attention(w) => self.attention(g, layer, w)?,
                Op::Ffn(w) => self.ffn(g, layer, w)?,
                Op::RmsNorm(w) => self.rmsnorm(g, w, layer)?,
                Op::Classifier(w) => self.classifier(g, w)?,
                Op::Argmax => self.argmax(g),
                Op::Repeat(count, body) => {
                    for l in 0..*count {
                        g = self.ops(g, body, Some(l))?;
                    }
                    g
                }
            };
        }
        Ok(g)
    }

    /// The first op: the token's embedding, and the step state, starting
    /// generation at BOS, position 0.
    fn embed(&self, table: &str) -> Result<Stream, String> {
        let (table, x) = (self.shared_weight(table)?, self.zeros(self.cfg.dim)?);
        let step = DeviceScalar::new(self.ctx, gpu::Step { token: BOS, pos: 0 })
            .map_err(|e| format!("alloc step: {e:?}"))?;
        let dim = self.cfg.dim as u32;
        Ok(self
            .ks
            .embed([self.cfg.dim], table, x, step, dim)
            .and_then(|(_table, x, step)| bundle2(forward(x), forward(step)))
            .erase())
    }

    fn rmsnorm(&self, g: Stream, w: &str, layer: Option<usize>) -> Result<Stream, String> {
        let (w, out) = (self.weight(w, layer)?, self.zeros(self.cfg.dim)?);
        let (ks, d) = (self.ks, self.cfg.dim);
        Ok(map_x(g, move |x| {
            ks.rmsnorm([d], x, w, out, d as u32)
                .and_then(|(_x, _w, out)| forward(out))
                .erase()
        }))
    }

    fn classifier(&self, g: Stream, w: &str) -> Result<Stream, String> {
        let (w, logits) = (self.shared_weight(w)?, self.zeros(self.cfg.vocab_size)?);
        let (ks, d, v) = (self.ks, self.cfg.dim, self.cfg.vocab_size);
        Ok(map_x(g, move |x| {
            ks.matmul([v], w, x, logits, d as u32)
                .and_then(|(_w, _x, logits)| forward(logits))
                .erase()
        }))
    }

    /// Greedy sampling into the step state: the next token and position.
    fn argmax(&self, g: Stream) -> Stream {
        let (ks, v) = (self.ks, self.cfg.vocab_size as u32);
        g.and_then(move |(logits, step)| {
            ks.argmax([1], logits, step, v)
                .and_then(|(logits, step)| bundle2(forward(logits), forward(step)))
        })
        .erase()
    }

    /// `x += Wo · attention(RoPE(Wq · rmsnorm(x)), KV cache)`, appending this
    /// token's key and value to the layer's caches at the current position.
    fn attention(
        &self,
        g: Stream,
        layer: Option<usize>,
        w: &[String; 5],
    ) -> Result<Stream, String> {
        let c = self.cfg;
        let [rms, wq, wk, wv, wo] = self.weights(w, layer)?;
        let [xb, q, xb2, xo] = self.zeros_n(c.dim)?;
        let [k_cache, v_cache] = self.zeros_n(c.seq_len * c.kv_dim())?;
        let att = self.zeros(c.n_heads * c.seq_len)?;
        let ks = self.ks;
        let (d, kvd, hs) = (c.dim as u32, c.kv_dim() as u32, c.head_size() as u32);
        let (seq, kv_mul) = (c.seq_len as u32, c.kv_mul() as u32);
        Ok(g.and_then(|(x, step)| {
            graph! {
                let (x, _rms, xb) = ks.rmsnorm([c.dim], x, rms, xb, d);
                let (_wq, xb, q) = ks.matmul([c.dim], wq, xb, q, d);
                let (_wk, _wv, _xb, k_cache, v_cache, step) =
                    ks.kv([c.kv_dim()], wk, wv, xb, k_cache, v_cache, step, d, kvd);
                let (q, k_cache, step) = ks.rope([c.dim / 2], q, k_cache, step, kvd, hs);
                let (_q, _k_cache, att, step) = ks.att_scores(
                    [c.n_heads * c.seq_len], q, k_cache, att, step, seq, hs, kvd, kv_mul,
                );
                let (att, step) = ks.softmax([c.n_heads], att, step, seq);
                let (_att, _v_cache, xb2, step) =
                    ks.att_mix([c.dim], att, v_cache, xb2, step, seq, hs, kvd, kv_mul);
                let (_wo, _xb2, xo) = ks.matmul([c.dim], wo, xb2, xo, d);
                let (x, _xo) = ks.add([c.dim], x, xo);
                bundle2(forward(x), forward(step))
            }
        })
        .erase())
    }

    /// `x += W2 · (silu(W1 · rmsnorm(x)) * (W3 · rmsnorm(x)))`.
    fn ffn(&self, g: Stream, layer: Option<usize>, w: &[String; 4]) -> Result<Stream, String> {
        let c = self.cfg;
        let [rms, w1, w2, w3] = self.weights(w, layer)?;
        let [xb, xo] = self.zeros_n(c.dim)?;
        let [h1, h3] = self.zeros_n(c.hidden_dim)?;
        let ks = self.ks;
        let (d, h) = (c.dim as u32, c.hidden_dim as u32);
        Ok(map_x(g, |x| {
            graph! {
                let (x, _rms, xb) = ks.rmsnorm([c.dim], x, rms, xb, d);
                let (_w1, xb, h1) = ks.matmul([c.hidden_dim], w1, xb, h1, d);
                let (_w3, _xb, h3) = ks.matmul([c.hidden_dim], w3, xb, h3, d);
                let (h1, _h3) = ks.swiglu([c.hidden_dim], h1, h3);
                let (_w2, _h1, xo) = ks.matmul([c.dim], w2, h1, xo, h);
                let (x, _xo) = ks.add([c.dim], x, xo);
                forward(x)
            }
            .erase()
        }))
    }
}

/// The residual stream (or the logits) alone: what an op that leaves the step
/// state untouched computes.
type Residual = DeviceDynOp<'static, Single<DeviceSlice<f32>>>;

/// An op on the residual stream alone: `f` maps it, and the step state passes
/// through untouched.
fn map_x(g: Stream, f: impl FnOnce(Pipe<DeviceSlice<f32>>) -> Residual) -> Stream {
    g.and_then(move |(x, step)| f(x).and_then(move |x| bundle2(forward(x), forward(step))))
        .erase()
}

/// Build the model graph described by `spec` over `weights`.
fn build_model(
    ctx: &Context,
    ks: &gpu::Kernels,
    spec: &Spec,
    weights: &Weights,
) -> Result<Stream, String> {
    if spec.config != weights.config {
        return Err(format!(
            "spec config {:?} != checkpoint config {:?}",
            spec.config, weights.config
        ));
    }
    Builder {
        ctx,
        ks,
        weights,
        cfg: spec.config,
        shared: Default::default(),
    }
    .build(&spec.ops)
}

// ── Tokenizer (decode only) ────────────────────────────────────────────────

/// llama2.c's `tokenizer.bin`: a max token length, then per token a score, a
/// length and the bytes.
struct Tokenizer {
    pieces: Vec<Vec<u8>>,
}

impl Tokenizer {
    fn parse(bytes: &[u8], vocab_size: usize) -> Result<Self, String> {
        let mut at = 4;
        let mut pieces = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let len_at = at + 4;
            let len = bytes
                .get(len_at..len_at + 4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as usize)
                .ok_or("tokenizer truncated")?;
            pieces.push(
                bytes
                    .get(len_at + 4..len_at + 4 + len)
                    .ok_or("tokenizer truncated")?
                    .to_vec(),
            );
            at = len_at + 4 + len;
        }
        Ok(Tokenizer { pieces })
    }

    /// The text of `token` following `prev`, as llama2.c's `decode` prints it.
    fn decode(&self, prev: u32, token: u32) -> Vec<u8> {
        let mut piece: &[u8] = &self.pieces[token as usize];
        // After BOS, SentencePiece's leading space is dropped.
        if prev == 1 && piece.first() == Some(&b' ') {
            piece = &piece[1..];
        }
        // Raw bytes are spelled `<0xXX>`.
        if let Some(hex) = std::str::from_utf8(piece)
            .ok()
            .and_then(|s| s.strip_prefix("<0x")?.strip_suffix('>'))
            .filter(|h| h.len() == 2)
            .and_then(|h| u8::from_str_radix(h, 16).ok())
        {
            return vec![hex];
        }
        piece.to_vec()
    }
}

// ── Generation ─────────────────────────────────────────────────────────────

/// Greedy generation for `steps` positions from BOS, as llama2.c's `generate`
/// with temperature 0 and no prompt: one graph sync per token. The graph
/// advances its own step state, so each sync only reads back the [`gpu::Step`].
fn generate(
    ctx: &Context,
    model: Stream,
    steps: usize,
    mut emit: impl FnMut(u32, u32),
) -> claspr::Result<usize> {
    let g = model.and_then(|(_logits, step)| forward(step));
    let mut token = BOS;
    let mut generated = 0;
    for pos in 0..steps {
        let step = g.sync(ctx)?.read_value()?;
        debug_assert_eq!(step.pos as usize, pos + 1);
        let next = step.token;
        if next == BOS {
            break;
        }
        emit(token, next);
        token = next;
        generated += 1;
    }
    Ok(generated)
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: llama2 <checkpoint.bin> <tokenizer.bin> [-n steps] [-s model.spec]";
    let (ckpt, tok) = (args.next().ok_or(usage)?, args.next().ok_or(usage)?);
    let (mut steps, mut spec_path) = (256usize, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-n" => steps = args.next().and_then(|v| v.parse().ok()).ok_or(usage)?,
            "-s" => spec_path = Some(args.next().ok_or(usage)?),
            _ => return Err(usage.into()),
        }
    }
    let spec_text = match spec_path {
        Some(p) => std::fs::read_to_string(&p).map_err(|e| format!("{p}: {e}"))?,
        None => include_str!("../stories15M.spec").to_string(),
    };
    let spec = parse_spec(&spec_text)?;
    let weights = Weights::parse(&std::fs::read(&ckpt).map_err(|e| format!("{ckpt}: {e}"))?)?;
    let tokenizer = Tokenizer::parse(
        &std::fs::read(&tok).map_err(|e| format!("{tok}: {e}"))?,
        weights.config.vocab_size,
    )?;
    let steps = steps.min(weights.config.seq_len);

    let ctx = Context::any().map_err(|e| format!("no OpenCL device: {e:?}"))?;
    let ks = gpu::kernels(&ctx).map_err(|e| format!("kernels: {e:?}"))?;
    if !spec.generates() {
        return Err("spec: generation needs a final `argmax`".into());
    }
    let model = build_model(&ctx, &ks, &spec, &weights)?;

    let t0 = std::time::Instant::now();
    let mut stdout = std::io::stdout();
    let n = generate(&ctx, model, steps, |prev, next| {
        let _ = stdout.write_all(&tokenizer.decode(prev, next));
        let _ = stdout.flush();
    })
    .map_err(|e| format!("generate: {e:?}"))?;
    println!();
    eprintln!(
        "{n} tokens, {:.1} tok/s",
        n as f64 / t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claspr::eager::download;

    fn ctx() -> Option<Context> {
        Context::any().ok()
    }

    /// A tiny random model exercising what stories15M does not: grouped-query
    /// attention (4 heads over 2 KV heads) and a separate classifier.
    const TINY: Config = Config {
        dim: 32,
        hidden_dim: 48,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 40,
        seq_len: 8,
    };

    const TINY_SPEC: &str = "
        dim 32
        hidden_dim 48
        n_layers 2
        n_heads 4
        n_kv_heads 2
        vocab_size 40
        seq_len 8
        embed token_embedding
        repeat n_layers
          attention rms_att wq wk wv wo
          ffn rms_ffn w1 w2 w3
        end
        rmsnorm rms_final
        classifier wcls
        argmax
    ";

    /// A llama2.c checkpoint of `c` with deterministic pseudo-random weights in
    /// `[-0.5, 0.5)` (RMS-norm weights near 1), and an unshared classifier.
    fn checkpoint(c: Config) -> Vec<u8> {
        let mut state = 0x2545_f491_u32;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let (d, h, l, kv) = (c.dim, c.hidden_dim, c.n_layers, c.kv_dim());
        let header = [d, h, l, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len];
        let mut out: Vec<u8> = Vec::new();
        for (i, v) in header.into_iter().enumerate() {
            // A negative vocab size marks the separate classifier.
            let v = if i == 5 { -(v as i32) } else { v as i32 };
            out.extend(v.to_le_bytes());
        }
        let sizes = [
            (c.vocab_size * d, false),
            (l * d, true),
            (l * d * d, false),
            (l * d * kv, false),
            (l * d * kv, false),
            (l * d * d, false),
            (l * d, true),
            (l * h * d, false),
            (l * d * h, false),
            (l * h * d, false),
            (d, true),
            (c.seq_len * c.head_size(), false),
            (c.vocab_size * d, false),
        ];
        for (len, norm) in sizes {
            for _ in 0..len {
                let v = if norm { 1.0 + 0.2 * rand() } else { rand() };
                out.extend(v.to_le_bytes());
            }
        }
        out
    }

    /// The CPU reference: llama2.c's `forward`, ported line for line.
    struct Cpu<'w> {
        w: &'w Weights,
        k_cache: Vec<f32>,
        v_cache: Vec<f32>,
    }

    fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
        let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5;
        let s = 1.0 / ss.sqrt();
        x.iter().zip(w).map(|(x, w)| w * (s * x)).collect()
    }

    fn matmul(w: &[f32], x: &[f32]) -> Vec<f32> {
        w.chunks(x.len())
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    impl<'w> Cpu<'w> {
        fn new(w: &'w Weights) -> Self {
            let c = w.config;
            let len = c.n_layers * c.seq_len * c.kv_dim();
            Cpu {
                w,
                k_cache: vec![0.0; len],
                v_cache: vec![0.0; len],
            }
        }

        fn forward(&mut self, token: usize, pos: usize) -> Vec<f32> {
            let c = self.w.config;
            let (d, kvd, hs) = (c.dim, c.kv_dim(), c.head_size());
            let t = |n: &str, l: usize| self.w.get(n, Some(l)).unwrap().to_vec();
            let mut x =
                self.w.get("token_embedding", None).unwrap()[token * d..(token + 1) * d].to_vec();
            for l in 0..c.n_layers {
                let xb = rmsnorm(&x, &t("rms_att", l));
                let mut q = matmul(&t("wq", l), &xb);
                let mut k = matmul(&t("wk", l), &xb);
                let v = matmul(&t("wv", l), &xb);
                for i in (0..d).step_by(2) {
                    let freq = 1.0 / 10000f32.powf((i % hs) as f32 / hs as f32);
                    let (fcr, fci) = ((pos as f32 * freq).cos(), (pos as f32 * freq).sin());
                    let rot = |v: &mut [f32]| {
                        let (v0, v1) = (v[i], v[i + 1]);
                        v[i] = v0 * fcr - v1 * fci;
                        v[i + 1] = v0 * fci + v1 * fcr;
                    };
                    rot(&mut q);
                    if i < kvd {
                        rot(&mut k);
                    }
                }
                let loff = l * c.seq_len * kvd;
                self.k_cache[loff + pos * kvd..loff + (pos + 1) * kvd].copy_from_slice(&k);
                self.v_cache[loff + pos * kvd..loff + (pos + 1) * kvd].copy_from_slice(&v);
                let mut xb = vec![0.0; d];
                for h in 0..c.n_heads {
                    let kvo = (h / c.kv_mul()) * hs;
                    let mut att: Vec<f32> = (0..=pos)
                        .map(|t| {
                            let k = &self.k_cache[loff + t * kvd + kvo..][..hs];
                            q[h * hs..(h + 1) * hs]
                                .iter()
                                .zip(k)
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                / (hs as f32).sqrt()
                        })
                        .collect();
                    let max = att.iter().cloned().fold(f32::MIN, f32::max);
                    att.iter_mut().for_each(|a| *a = (*a - max).exp());
                    let sum: f32 = att.iter().sum();
                    for (t, a) in att.iter().enumerate() {
                        let v = &self.v_cache[loff + t * kvd + kvo..][..hs];
                        for i in 0..hs {
                            xb[h * hs + i] += a / sum * v[i];
                        }
                    }
                }
                let o = matmul(&t("wo", l), &xb);
                x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
                let xb = rmsnorm(&x, &t("rms_ffn", l));
                let (h1, h3) = (matmul(&t("w1", l), &xb), matmul(&t("w3", l), &xb));
                let g: Vec<f32> = h1
                    .iter()
                    .zip(&h3)
                    .map(|(a, b)| a * (1.0 / (1.0 + (-a).exp())) * b)
                    .collect();
                let o = matmul(&t("w2", l), &g);
                x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
            }
            let x = rmsnorm(&x, self.w.get("rms_final", None).unwrap());
            matmul(self.w.get("wcls", None).unwrap(), &x)
        }
    }

    /// The graph built from the spec computes the same logits as the CPU
    /// reference at every position, replayed once per token. The graph advances
    /// its own step state (`argmax` picks the next token on the device), so the
    /// CPU follows the token the graph chose and checks the position it reports;
    /// the KV caches carry across replays.
    #[test]
    fn tiny_model_logits_match_cpu_reference() {
        let Some(ctx) = ctx() else { return };
        let ks = gpu::kernels(&ctx).expect("kernels");
        let weights = Weights::parse(&checkpoint(TINY)).expect("checkpoint");
        let spec = parse_spec(TINY_SPEC).expect("spec");
        assert!(spec.generates());
        let model = build_model(&ctx, &ks, &spec, &weights).expect("build");
        let g = model.and_then(|(logits, step)| bundle2(download(logits), forward(step)));
        let mut cpu = Cpu::new(&weights);
        let mut token = BOS as usize;
        for pos in 0..TINY.seq_len {
            let (got, step) = g.sync(&ctx).expect("sync");
            let want = cpu.forward(token, pos);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(err < 1e-4, "pos {pos}: max |gpu - cpu| = {err}");
            let step = step.read_value().expect("read step");
            assert_eq!(
                step.pos as usize,
                pos + 1,
                "position advanced on the device"
            );
            token = step.token as usize;
        }
    }

    #[test]
    fn spec_errors_are_reported() {
        let bad = |s: &str| parse_spec(s).err().unwrap_or_default();
        assert!(bad("repeat 2\nffn a b c d\n").contains("unterminated"));
        assert!(bad("end\n").contains("without `repeat`"));
        assert!(bad("frobnicate x\n").contains("unknown op"));
        assert!(bad("dim 4\n").contains("missing"));
        assert!(bad("repeat 1\nrepeat 1\nend\nend\n").contains("nested"));
        let cfg =
            "dim 4\nhidden_dim 4\nn_layers 1\nn_heads 1\nn_kv_heads 1\nvocab_size 4\nseq_len 4\n";
        let misplaced = |ops: &str| bad(&format!("{cfg}{ops}")).contains("must start with `embed`");
        assert!(misplaced("rmsnorm w\n"));
        assert!(misplaced("embed t\nargmax\nrmsnorm w\n"));
        assert!(misplaced("embed t\nrepeat 1\nargmax\nend\n"));
        assert!(parse_spec(&format!("{cfg}embed t\nrmsnorm w\nargmax\n")).is_ok());
    }

    /// With the real stories15M files (`LLAMA2_MODEL_DIR` holding
    /// `stories15M.bin` and `tokenizer.bin`), greedy generation reproduces
    /// llama2.c's `./run stories15M.bin -t 0 -n 64` byte for byte. Skipped
    /// without them (CI does not download the model).
    #[test]
    fn stories15m_greedy_matches_llama2_c() {
        let Some(dir) = std::env::var_os("LLAMA2_MODEL_DIR") else {
            eprintln!("SKIP: set LLAMA2_MODEL_DIR to the stories15M files");
            return;
        };
        let Some(ctx) = ctx() else { return };
        let dir = std::path::PathBuf::from(dir);
        let weights = Weights::parse(&std::fs::read(dir.join("stories15M.bin")).unwrap()).unwrap();
        let tokenizer =
            Tokenizer::parse(&std::fs::read(dir.join("tokenizer.bin")).unwrap(), 32000).unwrap();
        let ks = gpu::kernels(&ctx).unwrap();
        let spec = parse_spec(include_str!("../stories15M.spec")).unwrap();
        let model = build_model(&ctx, &ks, &spec, &weights).unwrap();
        let mut text = Vec::new();
        generate(&ctx, model, 64, |p, n| text.extend(tokenizer.decode(p, n))).unwrap();
        let want = "Once upon a time, there was a little girl named Lily. She loved to play \
                    outside in the sunshine. One day, she saw a big, red ball in the sky. It \
                    was the sun! She thought it was so pretty.\nLily wanted to play with the \
                    ball, but it was";
        assert_eq!(String::from_utf8_lossy(&text), want);
    }
}
