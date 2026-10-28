//! Trace target: the miniWeather shape — a 24-command chain whose scalar slot
//! alternates between two values each sync. Counting `clCreateCommandBufferKHR` /
//! `clCommandNDRangeKernelKHR` / `clSetKernelArg` in an API trace of this is the
//! direct analogue of the port measurement in `NOTES.md`.
use claspr::eager::{DeviceOpExt, fill};
use claspr::{DeviceSlice, slot, slots};
use claspr_test_kernels::kernels;
use claspr_test_support::ctx;

slots! { Factor: u32 }

macro_rules! stack_scale {
    ($head:expr, $ks:ident, $n:expr $(, $tok:tt)*) => {
        $head $( .and_then(|b| { let _ = stringify!($tok); $ks.scale_u32([$n], b, slot!(Factor)) }) )*
    };
}

fn main() -> claspr::Result<()> {
    let Some(ctx) = ctx() else { return Ok(()) };
    if !ctx.has_cl_khr_command_buffer() {
        eprintln!("SKIP");
        return Ok(());
    }
    let n = 64usize;
    let iters: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let ks = kernels::kernels(&ctx)?;
    let buf = DeviceSlice::<u32>::alloc_zero(&ctx, n)?;
    let g = stack_scale!(
        fill(buf, 1u32),
        ks,
        n,
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        i,
        j,
        k,
        l,
        m,
        n2,
        o,
        p,
        q,
        r,
        s,
        t,
        u,
        v,
        w
    )
    .call_all((Factor(1u32),));
    eprintln!("=== STEADY STATE BEGINS ===");
    for i in 0..iters {
        g.mutate_bind(Factor(if i % 2 == 0 { 1u32 } else { 2u32 }))?;
        drop(g.sync(&ctx)?);
    }
    eprintln!("=== DONE ===");
    Ok(())
}
