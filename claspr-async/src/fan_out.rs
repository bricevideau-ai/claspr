//! Variadic homogeneous parallel composition — [`FanOut`].
//!
//! Where [`Bundle`](crate::bundle) is heterogeneous (each child has
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

use crate::exec_ctx::ExecutionContext;
use crate::op::DeviceOperation;
use claspr::Result;

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

impl<I, F, U> DeviceOperation for FanOut<I, F>
where
    I: Send,
    F: FnMut(I) -> U + Send,
    U: DeviceOperation,
{
    type Output = Vec<U::Output>;

    fn execute(mut self, ctx: &ExecutionContext<'_>) -> Result<Vec<U::Output>> {
        let mut f = self
            .f
            .take()
            .expect("FanOut::execute called twice — internal claspr-async bug");
        let mut outputs = Vec::with_capacity(self.inputs.len());
        for input in self.inputs {
            let op = f(input);
            outputs.push(op.execute(ctx)?);
        }
        Ok(outputs)
    }
}
