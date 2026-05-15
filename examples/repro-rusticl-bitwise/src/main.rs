//! Bisecting reproducer for the `bitwise_ops_opencl` rusticl crash.
//!
//! The full difftest exercises 12 bitwise / shift / count operations
//! in one kernel. We don't know which one tickles rusticl's
//! SPIR-V → NIR lowering. This binary splits the suspect ops into
//! one device module each — claspr-build produces one SPV blob per
//! `#[claspr::device]` module — and tries to load each in turn from
//! the host. The first failure to load identifies the bad
//! instruction.
//!
//! Standard ops (`& | ^ ! << >>`) are excluded — they're trivially
//! supported everywhere. The interesting ones map to extended
//! instructions or SPIR-V core ops that rusticl might not handle:
//!
//!   count_ones      → OpExtInst <OpenCL.std> popcount
//!   leading_zeros   → OpExtInst <OpenCL.std> clz
//!   trailing_zeros  → OpExtInst <OpenCL.std> ctz
//!   rotate_left     → OpExtInst <OpenCL.std> rotate
//!   rotate_right    → OpExtInst <OpenCL.std> rotate (negated count)
//!   reverse_bits    → OpBitReverse (SPIR-V core)
//!
//! Run on rusticl:
//!   OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd \
//!   RUSTICL_ENABLE=llvmpipe \
//!   cargo run -p repro-rusticl-bitwise

use claspr::Context;

#[claspr::device]
pub mod popcount {
    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
    ) {
        let i = _id.x;
        data[i] = data[i].count_ones();
    }
}

#[claspr::device]
pub mod clz {
    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
    ) {
        let i = _id.x;
        data[i] = data[i].leading_zeros();
    }
}

#[claspr::device]
pub mod ctz {
    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
    ) {
        let i = _id.x;
        data[i] = data[i].trailing_zeros();
    }
}

#[claspr::device]
pub mod rotate_left {
    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
    ) {
        let i = _id.x;
        data[i] = data[i].rotate_left(7);
    }
}

#[claspr::device]
pub mod rotate_right {
    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
    ) {
        let i = _id.x;
        data[i] = data[i].rotate_right(7);
    }
}

#[claspr::device]
pub mod reverse_bits {
    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] data: &mut [u32],
    ) {
        let i = _id.x;
        data[i] = data[i].reverse_bits();
    }
}

/// Reproduces the *original* bitwise_ops_opencl test's exact kernel
/// shape: three slice args, a `match tid % 12` over all 12 ops,
/// dispatched through a helper function (cross-function call). Closer
/// to the actual SPIR-V the failing difftest produces.
#[claspr::device]
pub mod combined {
    fn compute(tid: usize, a: u32, b: u32) -> u32 {
        match tid % 12 {
            0 => a & b,
            1 => a | b,
            2 => a ^ b,
            3 => !a,
            4 => a << (b % 32),
            5 => a >> (b % 32),
            6 => a.rotate_left(b % 32),
            7 => a.rotate_right(b % 32),
            8 => a.count_ones(),
            9 => a.leading_zeros(),
            10 => a.trailing_zeros(),
            11 => a.reverse_bits(),
            _ => 0,
        }
    }

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] input_a: &[u32],
        #[spirv(cross_workgroup)] input_b: &[u32],
        #[spirv(cross_workgroup)] output: &mut [u32],
    ) {
        let tid = _id.x;
        if tid < input_a.len() && tid < input_b.len() && tid < output.len() {
            output[tid] = compute(tid, input_a[tid], input_b[tid]);
        }
    }
}

/// Bisect: three slice args, ONE op, NO helper, NO match. Uses
/// `get_unchecked` to skip implicit per-index bounds-check panics —
/// avoids a separate spirv-opt performance-pass crash that we hit
/// when keeping the panic-emitting `slice[tid]` indexing (under both
/// DebugPrintfThenExit and the UB-via-unreachable strategy this
/// build.rs sets).
#[claspr::device]
pub mod three_slices_one_op_unchecked {
    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] input_a: &[u32],
        #[spirv(cross_workgroup)] input_b: &[u32],
        #[spirv(cross_workgroup)] output: &mut [u32],
    ) {
        let tid = _id.x;
        unsafe {
            *output.get_unchecked_mut(tid) =
                *input_a.get_unchecked(tid) & *input_b.get_unchecked(tid);
        }
    }
}

/// `combined_basic` minus the bounds-checked indexing (uses
/// `get_unchecked`). Still has the helper function with the match
/// inside. If this passes rusticl, the trigger was the panic
/// infrastructure from `slice[tid]` indexing; if it crashes, the
/// helper-function call is.
#[claspr::device]
pub mod combined_basic_helper_unchecked {
    fn compute(tid: usize, a: u32, b: u32) -> u32 {
        match tid % 4 {
            0 => a & b,
            1 => a | b,
            2 => a ^ b,
            3 => !a,
            _ => 0,
        }
    }

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] input_a: &[u32],
        #[spirv(cross_workgroup)] input_b: &[u32],
        #[spirv(cross_workgroup)] output: &mut [u32],
    ) {
        let tid = _id.x;
        unsafe {
            *output.get_unchecked_mut(tid) = compute(
                tid,
                *input_a.get_unchecked(tid),
                *input_b.get_unchecked(tid),
            );
        }
    }
}

/// Smaller bisect: just the 4 plain bitwise ops, helper function +
/// match in the helper. Crashes rusticl. (`combined_basic`)
#[claspr::device]
pub mod combined_basic {
    fn compute(tid: usize, a: u32, b: u32) -> u32 {
        match tid % 4 {
            0 => a & b,
            1 => a | b,
            2 => a ^ b,
            3 => !a,
            _ => 0,
        }
    }

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] input_a: &[u32],
        #[spirv(cross_workgroup)] input_b: &[u32],
        #[spirv(cross_workgroup)] output: &mut [u32],
    ) {
        let tid = _id.x;
        if tid < input_a.len() && tid < input_b.len() && tid < output.len() {
            output[tid] = compute(tid, input_a[tid], input_b[tid]);
        }
    }
}

/// Same as combined_basic but with the match INLINED into the kernel
/// (no helper function), and `get_unchecked` to avoid the spirv-opt
/// perf-pass crash on the panic path. If this passes rusticl, the
/// helper-function call is implicated; if it crashes, the match
/// itself is.
#[claspr::device]
pub mod combined_basic_inlined {
    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] input_a: &[u32],
        #[spirv(cross_workgroup)] input_b: &[u32],
        #[spirv(cross_workgroup)] output: &mut [u32],
    ) {
        let tid = _id.x;
        unsafe {
            let a = *input_a.get_unchecked(tid);
            let b = *input_b.get_unchecked(tid);
            *output.get_unchecked_mut(tid) = match tid % 4 {
                0 => a & b,
                1 => a | b,
                2 => a ^ b,
                3 => !a,
                _ => 0,
            };
        }
    }
}

/// Bisect: shifts only.
#[claspr::device]
pub mod combined_shifts {
    fn compute(tid: usize, a: u32, b: u32) -> u32 {
        match tid % 4 {
            0 => a << (b % 32),
            1 => a >> (b % 32),
            2 => a.rotate_left(b % 32),
            3 => a.rotate_right(b % 32),
            _ => 0,
        }
    }

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] input_a: &[u32],
        #[spirv(cross_workgroup)] input_b: &[u32],
        #[spirv(cross_workgroup)] output: &mut [u32],
    ) {
        let tid = _id.x;
        if tid < input_a.len() && tid < input_b.len() && tid < output.len() {
            output[tid] = compute(tid, input_a[tid], input_b[tid]);
        }
    }
}

/// Bisect: bit-counting only.
#[claspr::device]
pub mod combined_counts {
    fn compute(tid: usize, a: u32, _b: u32) -> u32 {
        match tid % 4 {
            0 => a.count_ones(),
            1 => a.leading_zeros(),
            2 => a.trailing_zeros(),
            3 => a.reverse_bits(),
            _ => 0,
        }
    }

    #[claspr::kernel]
    pub fn run(
        #[spirv(global_invocation_id)] _id: ::glam::USizeVec3,
        #[spirv(cross_workgroup)] input_a: &[u32],
        #[spirv(cross_workgroup)] input_b: &[u32],
        #[spirv(cross_workgroup)] output: &mut [u32],
    ) {
        let tid = _id.x;
        if tid < input_a.len() && tid < input_b.len() && tid < output.len() {
            output[tid] = compute(tid, input_a[tid], input_b[tid]);
        }
    }
}

fn try_step(name: &str, phase: &str, f: impl FnOnce() -> claspr::Result<()>) {
    eprint!("  {name:14} {phase:8} ... ");
    match f() {
        Ok(()) => eprintln!("OK"),
        Err(e) => eprintln!("ERROR: {e}"),
    }
}

const N: usize = 256;

fn main() -> claspr::Result<()> {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no OpenCL device ({e})");
            return Ok(());
        }
    };
    eprintln!(
        "Device: {} ({})",
        ctx.device().name()?,
        ctx.device().vendor()?
    );
    eprintln!("Trying each op as build → launch → readback. The line printed last");
    eprintln!("(without OK/ERROR) identifies which step on which op killed the process.");

    let inputs: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();

    // Each block: build the kernel, allocate, launch, download. Any
    // step crashing the process points the finger.
    macro_rules! try_op {
        ($name:literal, $mod:ident) => {
            try_step($name, "all", || {
                let k = $mod::kernels(&ctx)?;
                let mut data = inputs.clone();
                let buf = ctx.upload(&data)?;
                k.run(&ctx, [N], &buf)?;
                ctx.download(&buf, &mut data)?;
                Ok(())
            });
        };
    }

    try_op!("popcount", popcount);
    try_op!("clz", clz);
    try_op!("ctz", ctz);
    try_op!("rotate_left", rotate_left);
    try_op!("rotate_right", rotate_right);
    try_op!("reverse_bits", reverse_bits);

    // Combined kernels — bisecting which subset crashes rusticl.
    eprintln!();
    eprintln!("Combined kernels (multiple ops in one match):");
    let input_a = inputs.clone();
    let input_b: Vec<u32> = (0..N as u32).collect();
    let mut output = vec![0u32; N];
    macro_rules! try_combined {
        ($name:literal, $mod:ident) => {
            try_step($name, "all", || {
                let k = $mod::kernels(&ctx)?;
                let a_buf = ctx.upload(&input_a)?;
                let b_buf = ctx.upload(&input_b)?;
                let o_buf = ctx.upload(&output)?;
                k.run(&ctx, [N], &a_buf, &b_buf, &o_buf)?;
                ctx.download(&o_buf, &mut output)?;
                Ok(())
            });
        };
    }
    try_combined!("3slc-1op", three_slices_one_op_unchecked);
    try_combined!("basic-inlined", combined_basic_inlined);
    try_combined!("basic-helper-unc", combined_basic_helper_unchecked);
    try_combined!("basic", combined_basic);
    try_combined!("shifts", combined_shifts);
    try_combined!("counts", combined_counts);
    try_combined!("all-12", combined);

    Ok(())
}
