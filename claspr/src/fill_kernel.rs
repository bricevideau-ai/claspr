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
