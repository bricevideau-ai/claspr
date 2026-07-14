//! Built-in fill kernels for the [`FillStrategy::DeviceKernel`] path.
//!
//! When [`DeviceSlice::fill`] / [`MappedSlice::fill`] is called on a
//! marker whose [`FillStrategy`] is `DeviceKernel`
//! (`HostReadOnly` / `DeviceScratch` — buffers the runtime can't
//! directly touch), the dispatch routes through a kernel launch
//! instead of `clEnqueueFillBuffer`. This module owns the OpenCL
//! C source strings for those kernels and the helper that builds
//! the program on a [`Context`](crate::Context).
//!
//! Per-size fast-path kernels for common widths (1, 2, 4, 8, 16
//! bytes) cover primitives + standard OpenCL vector types. For
//! arbitrary `size_of::<T>()` the dispatch falls back to
//! [`KERNEL_BYTES`], a byte-stride kernel that takes the pattern
//! as a small SVM-style buffer argument.
//!
//! Written as OpenCL C source (NOT rust-gpu) so the program builds
//! on any OpenCL device without depending on claspr's rust-gpu
//! infrastructure. The C source compiles in microseconds at first
//! use; subsequent fills reuse the cached `Program` on the Context.
//!
//! [`FillStrategy`]: crate::FillStrategy

/// OpenCL C source for the built-in fill program. Contains one
/// specialized kernel per common pattern size + a byte-generic
/// fallback. All kernels operate on the same memory model so the
/// same program serves both buffer and SVM allocations (the kernel
/// arg is `__global X*` either way).
pub(crate) const FILL_PROGRAM_SOURCE: &str = r#"
__kernel void claspr_fill_1 (__global uchar*  data, uchar  value, uint count) {
    uint id = get_global_id(0);
    if (id < count) data[id] = value;
}

__kernel void claspr_fill_2 (__global ushort* data, ushort value, uint count) {
    uint id = get_global_id(0);
    if (id < count) data[id] = value;
}

__kernel void claspr_fill_4 (__global uint*   data, uint   value, uint count) {
    uint id = get_global_id(0);
    if (id < count) data[id] = value;
}

__kernel void claspr_fill_8 (__global ulong*  data, ulong  value, uint count) {
    uint id = get_global_id(0);
    if (id < count) data[id] = value;
}

__kernel void claspr_fill_16(__global ulong2* data, ulong2 value, uint count) {
    uint id = get_global_id(0);
    if (id < count) data[id] = value;
}

/* Byte-generic fallback: pattern as small global buffer. The host
 * allocates a per-fill pattern buffer holding `pattern_size` bytes,
 * passes it as the second arg. Each work-item writes `pattern_size`
 * bytes from `pattern` to its slot in `data`. */
__kernel void claspr_fill_bytes(
    __global       uchar* data,
    __global const uchar* pattern,
    uint pattern_size,
    uint count
) {
    uint id = get_global_id(0);
    if (id < count) {
        uint base = id * pattern_size;
        for (uint i = 0; i < pattern_size; i++) {
            data[base + i] = pattern[i];
        }
    }
}
"#;

/// Kernel name returned by [`fast_path_kernel_name`] for a given
/// pattern byte size. Returns `None` for sizes that need the
/// byte-generic fallback ([`KERNEL_BYTES`]).
pub(crate) fn fast_path_kernel_name(pattern_size: usize) -> Option<&'static str> {
    match pattern_size {
        1 => Some("claspr_fill_1"),
        2 => Some("claspr_fill_2"),
        4 => Some("claspr_fill_4"),
        8 => Some("claspr_fill_8"),
        16 => Some("claspr_fill_16"),
        _ => None,
    }
}

/// Name of the byte-generic fallback kernel. Used for pattern sizes
/// that aren't in the fast-path set (1/2/4/8/16) — e.g. 12-byte
/// vec3-padded types, 32-byte vec8 types, arbitrary user structs.
pub(crate) const KERNEL_BYTES: &str = "claspr_fill_bytes";

/// Launch the built-in fill kernel: fast-path per-size kernel when the pattern width
/// matches, else the byte-generic [`KERNEL_BYTES`] fallback (which uploads the pattern
/// bytes into a small buffer first). `set_data_arg` sets kernel arg 0 — the ONLY
/// difference between the `cl_mem` (`exec.set_arg(buffer)`) and SVM
/// (`exec.set_arg_svm(ptr)`) callers — so both `DeviceSlice` and `MappedSlice` fills
/// share this one launch path. Non-blocking; returns the launch event.
pub(crate) fn fill_via_kernel<T: Copy, L: crate::Launcher + ?Sized>(
    ctx: &crate::Context,
    launcher: &L,
    set_data_arg: impl Fn(&mut opencl3::kernel::ExecuteKernel),
    pattern: &T,
    count: usize,
    deps: &[opencl3::types::cl_event],
) -> crate::Result<crate::Event> {
    use crate::error::Error;
    use opencl3::kernel::{ExecuteKernel, Kernel};
    use opencl3::memory::{Buffer as ClBuffer, CL_MEM_READ_ONLY};
    use opencl3::types::CL_BLOCKING;

    let pattern_size = std::mem::size_of::<T>();
    let count_u32 =
        u32::try_from(count).map_err(|_| Error::InvalidArgument("fill count exceeds u32::MAX"))?;
    let program = ctx.fill_program()?;

    if let Some(name) = fast_path_kernel_name(pattern_size) {
        let kernel = Kernel::create(program, name)?;
        let mut exec = ExecuteKernel::new(&kernel);
        // SAFETY: arg 0 is the data buffer/SVM pointer (element size `pattern_size`,
        // set by the caller's `set_data_arg`); arg 1 is the pattern by value (size
        // matches); arg 2 is the element count. The kernel writes `count` elements.
        unsafe {
            set_data_arg(&mut exec);
            exec.set_arg(pattern);
            exec.set_arg(&count_u32);
            exec.set_global_work_size(count);
            exec.set_event_wait_list(deps);
            Ok(exec.enqueue_nd_range(launcher.cl_queue())?)
        }
    } else {
        // Byte-generic path: upload the pattern bytes into a tiny read-only buffer,
        // then launch `claspr_fill_bytes` with (data, pattern_buf, pattern_size, count).
        let pattern_size_u32 = u32::try_from(pattern_size)
            .map_err(|_| Error::InvalidArgument("fill pattern size exceeds u32::MAX"))?;
        // SAFETY: `pattern` is a live `&T`; read `pattern_size` bytes of its repr.
        let pattern_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pattern as *const T as *const u8, pattern_size) };
        // SAFETY: fresh buffer in ctx's CL context; the blocking write and the kernel
        // both go through `launcher.cl_queue`, so the launch is serialized after it.
        let mut pattern_buf = unsafe {
            ClBuffer::<u8>::create(
                ctx.raw_context(),
                CL_MEM_READ_ONLY,
                pattern_size,
                std::ptr::null_mut(),
            )?
        };
        let _write_evt = unsafe {
            launcher.cl_queue().enqueue_write_buffer(
                &mut pattern_buf,
                CL_BLOCKING,
                0,
                pattern_bytes,
                &[],
            )?
        };
        let kernel = Kernel::create(program, KERNEL_BYTES)?;
        let mut exec = ExecuteKernel::new(&kernel);
        // SAFETY: arg 0 = data (buffer/SVM, via `set_data_arg`), arg 1 = pattern
        // buffer, arg 2 = pattern byte count, arg 3 = element count.
        let event = unsafe {
            set_data_arg(&mut exec);
            exec.set_arg(&pattern_buf);
            exec.set_arg(&pattern_size_u32);
            exec.set_arg(&count_u32);
            exec.set_global_work_size(count);
            exec.set_event_wait_list(deps);
            exec.enqueue_nd_range(launcher.cl_queue())?
        };
        // `pattern_buf` drops here; OpenCL retains the cl_mem for the in-flight kernel.
        Ok(event)
    }
}
