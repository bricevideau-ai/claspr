//! Device-side typed buffers.

use crate::context::Context;
use crate::error::{Error, Result};
use crate::queue::Launcher;
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::types::CL_BLOCKING;
use std::ptr;

/// A typed device-side buffer paired with its element count.
///
/// Mirrors rust-gpu's slice decomposition for kernel parameters: a
/// `&mut [T]` kernel arg becomes two `clSetKernelArg` calls (data
/// pointer + `usize` length). When passed as a launch argument,
/// claspr sets both — see the [`KernelArg`] impl in
/// [`crate::launch`].
///
/// Construct via [`DeviceSlice::alloc`] (uninitialised) or
/// [`DeviceSlice::upload`] (with initial host data). Read back via
/// [`DeviceSlice::download`].
///
/// [`KernelArg`]: crate::launch::KernelArg
pub struct DeviceSlice<T> {
    pub(crate) buffer: Buffer<T>,
    pub(crate) len: usize,
}

impl<T> DeviceSlice<T> {
    /// Allocate a device buffer of `len` elements, uninitialised.
    ///
    /// Pure context op — no command queue needed (`clCreateBuffer`
    /// doesn't enqueue anything). Pass any `Context` (e.g. from
    /// `Context::any()` or as borrowed from a `Launcher`).
    pub fn alloc(ctx: &Context, len: usize) -> Result<Self> {
        // SAFETY: passing a null host pointer means OpenCL allocates
        // fresh device memory and ignores the host-pointer contract
        // that makes `Buffer::create` generally unsafe.
        let buffer = unsafe {
            Buffer::<T>::create(ctx.raw_context(), CL_MEM_READ_WRITE, len, ptr::null_mut())?
        };
        Ok(DeviceSlice { buffer, len })
    }

    /// Allocate a device buffer and write `data` into it (blocking).
    ///
    /// Needs a [`Launcher`] for the queue side. Pass `&ctx` for the
    /// default queue or `&queue` for an explicit one.
    pub fn upload<L: Launcher>(launcher: &L, data: &[T]) -> Result<Self> {
        let mut slice = Self::alloc(launcher.context(), data.len())?;
        // SAFETY: blocking write into the buffer we just allocated;
        // no aliasing, no concurrent device access.
        unsafe {
            launcher
                .cl_queue()
                .enqueue_write_buffer(&mut slice.buffer, CL_BLOCKING, 0, data, &[])?
                .wait()?;
        }
        Ok(slice)
    }

    /// Read the buffer back into a host slice (blocking).
    ///
    /// `dst` must have the same length as `self`. Returns
    /// [`Error::LengthMismatch`] otherwise.
    pub fn download<L: Launcher>(&self, launcher: &L, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::LengthMismatch {
                src: self.len,
                dst: dst.len(),
            });
        }
        // SAFETY: blocking read; no aliasing of `dst`.
        unsafe {
            launcher
                .cl_queue()
                .enqueue_read_buffer(&self.buffer, CL_BLOCKING, 0, dst, &[])?
                .wait()?;
        }
        Ok(())
    }

    /// Number of `T` elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the buffer has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the underlying opencl3 [`Buffer`] for cases that need
    /// direct OpenCL access.
    pub fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }
}
