//! Device-side typed buffers.

use opencl3::memory::Buffer;

/// A typed device-side buffer paired with its element count.
///
/// Mirrors rust-gpu's slice decomposition for kernel parameters: a
/// `&mut [T]` kernel arg is two `clSetKernelArg` calls (data pointer +
/// `usize` length). When a `DeviceSlice<T>` is passed as a launch
/// argument, claspr sets both — see the [`KernelArg`] impl.
///
/// Construct via [`Context::upload`] (with initial host data) or
/// [`Context::alloc`] (uninitialised).
///
/// [`KernelArg`]: crate::launch::KernelArg
/// [`Context::upload`]: crate::context::Context::upload
/// [`Context::alloc`]: crate::context::Context::alloc
pub struct DeviceSlice<T> {
    pub(crate) buffer: Buffer<T>,
    pub(crate) len: usize,
}

impl<T> DeviceSlice<T> {
    /// Number of `T` elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the buffer has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the underlying opencl3 [`Buffer`] for cases that need
    /// direct OpenCL access (e.g. enqueueing reads/writes manually).
    pub fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }
}
