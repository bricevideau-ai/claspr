//! OpenCL context, command queue, and the launch entry point.

use crate::Result;
use crate::buffer::DeviceSlice;
use crate::launch::{IntoLaunchSpec, KernelArgs};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_device_id};
use std::ptr;

/// OpenCL context, command queue, and selected device.
///
/// Owns one device, one [`opencl3::context::Context`], and one
/// profiling-enabled [`CommandQueue`]. Kernel launches go through
/// [`Context::launch`].
pub struct Context {
    device_id: cl_device_id,
    context: opencl3::context::Context,
    queue: CommandQueue,
}

impl Context {
    /// Pick the first available OpenCL device of any type.
    ///
    /// Errors if no OpenCL device is reachable. For multi-device
    /// systems where you want a specific vendor or device type, use
    /// [`Context::select`].
    pub fn new() -> Result<Self> {
        Self::from_device_id(
            *get_all_devices(CL_DEVICE_TYPE_ALL)?
                .first()
                .ok_or("no OpenCL devices found")?,
        )
    }

    /// Pick the first device for which `pred` returns `true`.
    ///
    /// `pred` receives an [`opencl3::device::Device`] and can call any
    /// of its query methods (`name`, `vendor`, `device_type`,
    /// `extensions`, …) to make the decision.
    ///
    /// ```ignore
    /// let ctx = claspr::Context::select(|d| {
    ///     d.vendor().map(|v| v == "Intel(R) Corporation").unwrap_or(false)
    /// })?;
    /// ```
    pub fn select<F>(mut pred: F) -> Result<Self>
    where
        F: FnMut(&Device) -> bool,
    {
        let device_id = get_all_devices(CL_DEVICE_TYPE_ALL)?
            .into_iter()
            .find(|id| pred(&Device::new(*id)))
            .ok_or("no OpenCL device matched the predicate")?;
        Self::from_device_id(device_id)
    }

    fn from_device_id(device_id: cl_device_id) -> Result<Self> {
        let device = Device::new(device_id);
        let context = opencl3::context::Context::from_device(&device)?;
        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;
        Ok(Self {
            device_id,
            context,
            queue,
        })
    }

    /// Borrow the [`Device`] backing this context for capability queries.
    pub fn device(&self) -> Device {
        Device::new(self.device_id)
    }

    /// Borrow the underlying [`opencl3::context::Context`].
    pub fn raw_context(&self) -> &opencl3::context::Context {
        &self.context
    }

    /// Borrow the profiling-enabled command queue.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    // ── Buffer management ─────────────────────────────────────────────

    /// Allocate a device buffer and write `data` into it (blocking).
    pub fn upload<T>(&self, data: &[T]) -> Result<DeviceSlice<T>> {
        let mut buffer = unsafe {
            Buffer::<T>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                data.len(),
                ptr::null_mut(),
            )?
        };
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, data, &[])?
                .wait()?;
        }
        Ok(DeviceSlice {
            buffer,
            len: data.len(),
        })
    }

    /// Allocate a device buffer of `len` `T`s without initialising it.
    ///
    /// Wraps `Buffer::create(.., null_mut())` — passing the null host
    /// pointer makes OpenCL allocate fresh device memory and ignore the
    /// host-pointer contract that makes `Buffer::create` generally
    /// unsafe, so the wrapper here is sound.
    pub fn alloc<T>(&self, len: usize) -> Result<DeviceSlice<T>> {
        let buffer =
            unsafe { Buffer::<T>::create(&self.context, CL_MEM_READ_WRITE, len, ptr::null_mut())? };
        Ok(DeviceSlice { buffer, len })
    }

    /// Read a device buffer back into a host slice (blocking).
    ///
    /// `dst` must have the same length as `src`.
    pub fn download<T>(&self, src: &DeviceSlice<T>, dst: &mut [T]) -> Result<()> {
        if dst.len() != src.len() {
            return Err(format!(
                "download length mismatch: src has {} elements, dst has {}",
                src.len(),
                dst.len()
            )
            .into());
        }
        unsafe {
            self.queue
                .enqueue_read_buffer(&src.buffer, CL_BLOCKING, 0, dst, &[])?
                .wait()?;
        }
        Ok(())
    }

    // ── Program / kernel ─────────────────────────────────────────────

    /// Create + build an OpenCL program from raw SPIR-V bytes.
    ///
    /// Returns the build log on failure.
    pub fn build_program(&self, spv_bytes: &[u8]) -> Result<Program> {
        let mut program = Program::create_from_il(&self.context, spv_bytes)
            .map_err(|e| format!("create_from_il: {e}"))?;
        if let Err(e) = program.build(self.context.devices(), "") {
            let log = program
                .get_build_log(self.device_id)
                .unwrap_or_else(|_| "no build log".into());
            return Err(format!("program.build: {e}\nbuild log: {log}").into());
        }
        Ok(program)
    }

    /// Look up a kernel by entry-point name in a built program.
    pub fn kernel(&self, program: &Program, name: &str) -> Result<Kernel> {
        Kernel::create(program, name).map_err(|e| format!("Kernel::create({name}): {e}").into())
    }

    /// Convenience: [`build_program`] + [`kernel`] in one call. The
    /// intermediate `Program` is dropped — OpenCL refcounts it
    /// internally and the kernel keeps it alive.
    ///
    /// [`build_program`]: Self::build_program
    /// [`kernel`]: Self::kernel
    pub fn kernel_from_spv(&self, spv_bytes: &[u8], name: &str) -> Result<Kernel> {
        let program = self.build_program(spv_bytes)?;
        self.kernel(&program, name)
    }

    // ── Launch ───────────────────────────────────────────────────────

    /// Launch a kernel, blocking until it finishes.
    ///
    /// `spec` is the work-item geometry — pass `[N]`, `[W, H]`, or
    /// `[X, Y, Z]` for global-only, or `(global, local)` to control
    /// the workgroup size as well. `args` is a tuple of [`KernelArg`]
    /// values, set in declaration order.
    ///
    /// Returns the [`Event`] for profiling — feed it to
    /// [`profiling_duration`] to get the kernel runtime.
    ///
    /// [`KernelArg`]: crate::launch::KernelArg
    /// [`profiling_duration`]: crate::launch::profiling_duration
    pub fn launch<S, A>(&self, kernel: &Kernel, spec: S, args: A) -> Result<Event>
    where
        S: IntoLaunchSpec,
        A: KernelArgs,
    {
        let spec = spec.into_launch_spec();
        let mut exec = ExecuteKernel::new(kernel);
        args.set_all(&mut exec);
        exec.set_global_work_sizes(spec.global());
        if let Some(local) = spec.local() {
            exec.set_local_work_sizes(local);
        }
        let event = unsafe { exec.enqueue_nd_range(&self.queue)? };
        event.wait()?;
        Ok(event)
    }
}
