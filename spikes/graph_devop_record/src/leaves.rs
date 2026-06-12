//! Three representative recordable leaf ops + one non-recordable.
//!
//! Stand-ins for real claspr Tier 1 ops, just enough type-shape to
//! exercise the trait extensions.

use crate::device_op::{
    Deps, DeviceOperation, ExecutionContext, RecordContext, RecordableOp, Result, SyncPoints,
};

// ─── Recordable: kernel launch ────────────────────────────────────

pub struct FillKernel {
    pub n: usize,
    pub buf_id: u64,
    pub value: u32,
}

impl DeviceOperation for FillKernel {
    type Output = u64; // buf_id flows through

    fn execute(self, ctx: &ExecutionContext<'_>, _deps: Deps) -> Result<(u64, Deps)> {
        ctx.enqueue_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Real impl: clEnqueueNDRangeKernel + register completion.
        Ok((self.buf_id, vec![1]))
    }
}

impl RecordableOp for FillKernel {
    fn record(
        self,
        ctx: &mut RecordContext<'_>,
        _sync_points: SyncPoints,
    ) -> Result<(u64, SyncPoints)> {
        // Real impl: clCommandNDRangeKernelKHR(cb, queue, props, kernel,
        //   global_work_offset, global_work_size, local_work_size,
        //   sync_point_wait_list, &sync_point).
        let sp = ctx.command_buffer.record(&format!(
            "fill_kernel(n={}, buf={}, value={})",
            self.n, self.buf_id, self.value
        ));
        Ok((self.buf_id, vec![sp]))
    }
}

// ─── Recordable: buffer fill (Tier 1 FillOp analog) ────────────────

pub struct BufferFill {
    pub buf_id: u64,
    pub pattern: u32,
    pub len: usize,
}

impl DeviceOperation for BufferFill {
    type Output = u64;

    fn execute(self, ctx: &ExecutionContext<'_>, _deps: Deps) -> Result<(u64, Deps)> {
        ctx.enqueue_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Real impl: launcher.cl_queue().enqueue_fill_buffer(...).
        Ok((self.buf_id, vec![2]))
    }
}

impl RecordableOp for BufferFill {
    fn record(
        self,
        ctx: &mut RecordContext<'_>,
        _sync_points: SyncPoints,
    ) -> Result<(u64, SyncPoints)> {
        // Real impl: clCommandFillBufferKHR(cb, queue, props, buf,
        //   &pattern, pattern_size, offset, size, sync_point_wait_list,
        //   &sync_point).
        let sp = ctx.command_buffer.record(&format!(
            "buffer_fill(buf={}, pattern={})",
            self.buf_id, self.pattern
        ));
        Ok((self.buf_id, vec![sp]))
    }
}

// ─── Recordable: device-to-device copy (Tier 1 CopyOp analog) ──────

pub struct BufferCopy {
    pub src_id: u64,
    pub dst_id: u64,
    pub len_bytes: usize,
}

impl DeviceOperation for BufferCopy {
    type Output = u64; // dst_id

    fn execute(self, ctx: &ExecutionContext<'_>, _deps: Deps) -> Result<(u64, Deps)> {
        ctx.enqueue_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((self.dst_id, vec![3]))
    }
}

impl RecordableOp for BufferCopy {
    fn record(
        self,
        ctx: &mut RecordContext<'_>,
        _sync_points: SyncPoints,
    ) -> Result<(u64, SyncPoints)> {
        let sp = ctx.command_buffer.record(&format!(
            "buffer_copy(src={}, dst={}, len={})",
            self.src_id, self.dst_id, self.len_bytes
        ));
        Ok((self.dst_id, vec![sp]))
    }
}

// ─── Non-recordable: host-to-device upload ─────────────────────────
//
// Note the absence of `impl RecordableOp for Upload`. A chain that
// includes an Upload simply cannot be `.record()`'d — trying to do
// so is a compile-time error because the combinator's bound
// (`Source: RecordableOp`) won't be satisfied.

pub struct Upload {
    pub data: Vec<u32>,
    /// Where the upload lands; in real claspr this would be allocated
    /// inside `execute` and returned. Spike fakes that with a static id.
    pub allocated_buf_id: u64,
}

impl DeviceOperation for Upload {
    type Output = u64;

    fn execute(self, ctx: &ExecutionContext<'_>, _deps: Deps) -> Result<(u64, Deps)> {
        ctx.enqueue_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Real impl: clEnqueueWriteBuffer (non-blocking).
        Ok((self.allocated_buf_id, vec![99]))
    }
}

// Deliberately no `impl RecordableOp for Upload`.

// ─── Non-recordable: device-to-host download ───────────────────────

pub struct Download {
    pub buf_id: u64,
    /// Number of elements the spike pretends to read back.
    pub len: usize,
}

impl DeviceOperation for Download {
    type Output = Vec<u32>;

    fn execute(self, ctx: &ExecutionContext<'_>, _deps: Deps) -> Result<(Vec<u32>, Deps)> {
        ctx.enqueue_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Real impl: clEnqueueReadBuffer + wait.
        Ok((vec![self.buf_id as u32; self.len], vec![100]))
    }
}

// No `impl RecordableOp for Download` — same as Upload.

// ─── Recordable: elementwise multiply kernel (batch-inference) ─────

pub struct ElemMul {
    pub n: usize,
    pub a_buf_id: u64,
    /// In the real example this is `Arc<DeviceSlice<u32>>` — the
    /// shared weights buffer.
    pub b_buf_id: u64,
}

impl DeviceOperation for ElemMul {
    type Output = (u64, u64); // (a_buf, b_buf) flow through

    fn execute(self, ctx: &ExecutionContext<'_>, _deps: Deps) -> Result<((u64, u64), Deps)> {
        ctx.enqueue_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(((self.a_buf_id, self.b_buf_id), vec![10]))
    }
}

impl RecordableOp for ElemMul {
    fn record(
        self,
        ctx: &mut RecordContext<'_>,
        _sync_points: SyncPoints,
    ) -> Result<((u64, u64), SyncPoints)> {
        let sp = ctx.command_buffer.record(&format!(
            "elem_mul(a={}, b={}, n={})",
            self.a_buf_id, self.b_buf_id, self.n
        ));
        Ok(((self.a_buf_id, self.b_buf_id), vec![sp]))
    }
}

// ─── Recordable: add-bias kernel (batch-inference) ─────────────────

pub struct AddBias {
    pub n: usize,
    pub buf_id: u64,
    pub bias: u32,
}

impl DeviceOperation for AddBias {
    type Output = u64;

    fn execute(self, ctx: &ExecutionContext<'_>, _deps: Deps) -> Result<(u64, Deps)> {
        ctx.enqueue_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((self.buf_id, vec![11]))
    }
}

impl RecordableOp for AddBias {
    fn record(
        self,
        ctx: &mut RecordContext<'_>,
        _sync_points: SyncPoints,
    ) -> Result<(u64, SyncPoints)> {
        let sp = ctx.command_buffer.record(&format!(
            "add_bias(buf={}, bias={})",
            self.buf_id, self.bias
        ));
        Ok((self.buf_id, vec![sp]))
    }
}
