//! Variadic homogeneous parallel composition — [`FanOut`].
//!
//! Where [`Bundle`](crate::bundle!) is heterogeneous (each child has
//! its own type), `FanOut` takes a `Vec<I>` of inputs and a closure
//! that turns each input into a [`DeviceOperation`]. The N children
//! are all the same op type; the output is `Vec<U::Output>`.
//!
//! Like `Bundle`, children submit to the chain's OOO queue in input
//! order; the runtime overlaps execution on the device per event
//! dependencies.
//!
//! ## Example — tile-parallel processing
//!
//! ```ignore
//! use claspr_async::{DeviceOperation, fan_out, value};
//!
//! let tiles: Vec<u32> = (0..16).collect();
//! let totals: Vec<u32> = fan_out(tiles, |tile| {
//!     value(tile).and_then(|t| value(t.wrapping_mul(2)))
//! })
//! .sync(&ctx)?;
//! ```

use crate::device_op::{Deps, DeviceOperation, wrap_event};
use crate::exec_ctx::ExecutionContext;
use crate::{Error, Launcher, Result};

/// N-ary homogeneous parallel composition. Construct with [`fan_out`].
pub struct FanOut<I, F> {
    inputs: Vec<I>,
    f: Option<F>,
}

/// Apply `f` to each input, executing all resulting ops on the chain's
/// OOO queue, and collect their outputs into a `Vec`. See [`FanOut`].
pub fn fan_out<I, F, U>(inputs: Vec<I>, f: F) -> FanOut<I, F>
where
    I: Send,
    F: FnMut(I) -> U + Send,
    U: DeviceOperation,
{
    FanOut { inputs, f: Some(f) }
}

/// Method-call form of [`fan_out`]: `vec![a, b].fan_out(op)`.
///
/// Reads as data → operation and composes cleanly with downstream
/// `.and_then`, mirroring how tokio / rayon let you chain
/// parallel-map shapes. The free-fn form stays available — use
/// whichever fits the call site.
pub trait FanOutExt<I>: Sized {
    fn fan_out<F, U>(self, f: F) -> FanOut<I, F>
    where
        F: FnMut(I) -> U + Send,
        U: DeviceOperation;
}

impl<I: Send> FanOutExt<I> for Vec<I> {
    fn fan_out<F, U>(self, f: F) -> FanOut<I, F>
    where
        F: FnMut(I) -> U + Send,
        U: DeviceOperation,
    {
        fan_out(self, f)
    }
}

impl<I, F, U> DeviceOperation for FanOut<I, F>
where
    I: Send,
    F: FnMut(I) -> U + Send,
    U: DeviceOperation,
{
    type Output = Vec<U::Output>;

    fn execute(mut self, ctx: &ExecutionContext<'_>, deps: Deps) -> Result<(Vec<U::Output>, Deps)> {
        let mut f = self
            .f
            .take()
            .expect("FanOut::execute called twice — internal claspr-async bug");
        let mut outputs = Vec::with_capacity(self.inputs.len());
        // Hold every child's events until after the marker enqueue.
        // See `bundle.rs` for the same rationale.
        let mut child_evts: Vec<Deps> = Vec::with_capacity(self.inputs.len());
        for input in self.inputs {
            let op = f(input);
            let (out, evts) = op.execute(ctx, deps.clone())?;
            outputs.push(out);
            child_evts.push(evts);
        }
        let all_events: Vec<opencl3::types::cl_event> = child_evts
            .iter()
            .flat_map(|evts| evts.iter().map(|d| d.as_ref().get()))
            .collect();
        // SAFETY: cl_event handles valid for the duration of the call.
        let marker = unsafe { ctx.cl_queue().enqueue_marker_with_wait_list(&all_events) }
            .map_err(Error::OpenCl)?;
        drop(child_evts);
        Ok((outputs, vec![wrap_event(marker)]))
    }
}
