//! [`Platform`] + [`Device`] — entry points for selecting where
//! kernels run.
//!
//! Both are `Arc`-wrapped internally so they're cheap to clone and
//! safe to share across threads. The OpenCL ICD runs its own
//! reference count under the hood; this Rust-side `Arc` is the only
//! per-process refcount we add.
//!
//! # Picking a device
//!
//! ```ignore
//! // Trivial: first device of any type
//! let device = claspr::Device::any()?;
//!
//! // Canned selectors
//! let gpu = claspr::Device::any_gpu()?;
//! let cpu = claspr::Device::any_cpu()?;
//!
//! // Custom: SYCL-style scoring closure (highest score wins,
//! // negative excludes)
//! let intel_gpu = claspr::Device::find(|d| match d {
//!     d if !d.is_gpu() => -1,
//!     d if d.vendor().unwrap_or_default().contains("Intel") => 100,
//!     _ => 1,
//! })?;
//! ```
//!
//! # Sub-devices
//!
//! Partition a CPU device by compute units to reserve threads for
//! one workload while another runs:
//!
//! ```ignore
//! let cpu = claspr::Device::any_cpu()?;
//! let halves = cpu.partition_equally(2)?;
//! // each `halves[i]` is a Device backed by half of the CPU's CUs
//! ```

use crate::{Error, Result};
use opencl3::device::{
    CL_DEVICE_PARTITION_BY_COUNTS, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,
    CL_DEVICE_PARTITION_EQUALLY, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL,
    CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_CUSTOM, CL_DEVICE_TYPE_GPU, Device as Cl3Device,
    release_device,
};
use opencl3::platform::{Platform as Cl3Platform, get_platforms};
use opencl3::types::{cl_device_id, cl_device_type, cl_platform_id};
use std::fmt;
use std::sync::Arc;

// ── Platform ─────────────────────────────────────────────────────────

/// An OpenCL platform — one vendor's runtime implementation reachable
/// through the loader (pocl, NVIDIA, Mesa Rusticl, Apple, …).
///
/// Cheap to clone: an `Arc` over the cached ID + name + vendor.
#[derive(Clone)]
pub struct Platform {
    inner: Arc<PlatformInner>,
}

struct PlatformInner {
    id: cl_platform_id,
    name: String,
    vendor: String,
}

// SAFETY: cl_platform_id is an opaque handle to runtime-owned state.
// The OpenCL spec guarantees thread-safety for all its API calls
// (CL 1.2 §3.4.1, CL 3.0 §3.4.1).
unsafe impl Send for PlatformInner {}
unsafe impl Sync for PlatformInner {}

impl Platform {
    /// Every OpenCL platform reachable from the loader.
    pub fn all() -> Result<Vec<Platform>> {
        let cl3_platforms = get_platforms()?;
        let mut out = Vec::with_capacity(cl3_platforms.len());
        for p in cl3_platforms {
            out.push(Platform::from_cl3(p)?);
        }
        Ok(out)
    }

    fn from_cl3(p: Cl3Platform) -> Result<Platform> {
        let name = p.name().unwrap_or_default();
        let vendor = p.vendor().unwrap_or_default();
        Ok(Platform {
            inner: Arc::new(PlatformInner {
                id: p.id(),
                name,
                vendor,
            }),
        })
    }

    /// Devices on this platform of the given type.
    pub fn devices_of_type(&self, kind: DeviceType) -> Result<Vec<Device>> {
        let cl3 = Cl3Platform::new(self.inner.id);
        let ids = cl3.get_devices(kind.as_cl_type())?;
        Ok(ids
            .into_iter()
            .map(|id| Device::from_root(id, self.clone()))
            .collect())
    }

    /// All devices on this platform.
    pub fn devices(&self) -> Result<Vec<Device>> {
        self.devices_of_type(DeviceType::All)
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub fn vendor(&self) -> &str {
        &self.inner.vendor
    }

    pub fn raw_id(&self) -> cl_platform_id {
        self.inner.id
    }
}

impl fmt::Debug for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Platform")
            .field("name", &self.inner.name)
            .field("vendor", &self.inner.vendor)
            .finish()
    }
}

// ── DeviceType ───────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Accelerator,
    Custom,
    All,
}

impl DeviceType {
    fn as_cl_type(self) -> cl_device_type {
        match self {
            DeviceType::Cpu => CL_DEVICE_TYPE_CPU,
            DeviceType::Gpu => CL_DEVICE_TYPE_GPU,
            DeviceType::Accelerator => CL_DEVICE_TYPE_ACCELERATOR,
            DeviceType::Custom => CL_DEVICE_TYPE_CUSTOM,
            DeviceType::All => CL_DEVICE_TYPE_ALL,
        }
    }

    fn from_cl_type(t: cl_device_type) -> DeviceType {
        // Devices may report multiple bits set (DEFAULT plus a real type).
        // Match in priority order, falling back to All for the unrecognised case.
        if t & CL_DEVICE_TYPE_GPU != 0 {
            DeviceType::Gpu
        } else if t & CL_DEVICE_TYPE_CPU != 0 {
            DeviceType::Cpu
        } else if t & CL_DEVICE_TYPE_ACCELERATOR != 0 {
            DeviceType::Accelerator
        } else if t & CL_DEVICE_TYPE_CUSTOM != 0 {
            DeviceType::Custom
        } else {
            DeviceType::All
        }
    }
}

// ── Device ───────────────────────────────────────────────────────────

/// An OpenCL device (or sub-device).
///
/// Cheap to clone: an `Arc` over the device handle + cached
/// platform reference. Sub-devices get released on the last `Arc`
/// drop; root devices are owned by the OpenCL ICD and not released.
#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

struct DeviceInner {
    id: cl_device_id,
    platform: Platform,
    is_sub_device: bool,
}

// SAFETY: cl_device_id is an opaque handle. OpenCL API calls on it
// are thread-safe (CL spec §3.4.1).
unsafe impl Send for DeviceInner {}
unsafe impl Sync for DeviceInner {}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        if self.is_sub_device {
            // SAFETY: id was returned by clCreateSubDevices and we hold
            // the only Arc to it. Errors here can't be propagated; the
            // ICD will leak the sub-device, which is recoverable.
            let _ = unsafe { release_device(self.id) };
        }
    }
}

impl Device {
    /// All devices reachable from any platform.
    pub fn all() -> Result<Vec<Device>> {
        let mut out = Vec::new();
        for p in Platform::all()? {
            out.extend(p.devices()?);
        }
        Ok(out)
    }

    /// First device of any type. Convenience for trivial single-device
    /// setups; for anything multi-device, use [`find`](Self::find).
    pub fn any() -> Result<Device> {
        Self::all()?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Other("no OpenCL devices found".into()))
    }

    /// First GPU device on any platform.
    pub fn any_gpu() -> Result<Device> {
        Self::find(|d| if d.is_gpu() { 100 } else { -1 })
    }

    /// First CPU device on any platform.
    pub fn any_cpu() -> Result<Device> {
        Self::find(|d| if d.is_cpu() { 100 } else { -1 })
    }

    /// Pick the device with the highest score. Negative scores
    /// exclude. Mirrors SYCL 2020 device-selector semantics.
    ///
    /// Returns `Error::NotSupported` if every device gets a negative
    /// score (no match) or there are no devices at all.
    pub fn find<F>(mut score: F) -> Result<Device>
    where
        F: FnMut(&Device) -> i32,
    {
        let mut best: Option<(i32, Device)> = None;
        for d in Self::all()? {
            let s = score(&d);
            if s < 0 {
                continue;
            }
            match &best {
                Some((bs, _)) if *bs >= s => {}
                _ => best = Some((s, d)),
            }
        }
        best.map(|(_, d)| d)
            .ok_or(Error::NotSupported("no device matched the selector"))
    }

    fn from_root(id: cl_device_id, platform: Platform) -> Device {
        Device {
            inner: Arc::new(DeviceInner {
                id,
                platform,
                is_sub_device: false,
            }),
        }
    }

    fn from_sub(id: cl_device_id, platform: Platform) -> Device {
        Device {
            inner: Arc::new(DeviceInner {
                id,
                platform,
                is_sub_device: true,
            }),
        }
    }

    pub fn name(&self) -> Result<String> {
        Ok(self.cl3().name()?)
    }

    pub fn vendor(&self) -> Result<String> {
        Ok(self.cl3().vendor()?)
    }

    pub fn device_type(&self) -> DeviceType {
        self.cl3()
            .dev_type()
            .map(DeviceType::from_cl_type)
            .unwrap_or(DeviceType::All)
    }

    pub fn is_gpu(&self) -> bool {
        self.device_type() == DeviceType::Gpu
    }

    pub fn is_cpu(&self) -> bool {
        self.device_type() == DeviceType::Cpu
    }

    pub fn max_compute_units(&self) -> Result<u32> {
        Ok(self.cl3().max_compute_units()?)
    }

    pub fn max_work_group_size(&self) -> Result<usize> {
        Ok(self.cl3().max_work_group_size()?)
    }

    pub fn platform(&self) -> &Platform {
        &self.inner.platform
    }

    pub fn raw_id(&self) -> cl_device_id {
        self.inner.id
    }

    /// Convert into an opencl3 device wrapper for any uncovered query.
    /// Stable but escape-hatch — prefer adding a method here when a
    /// query becomes load-bearing.
    pub fn cl3(&self) -> Cl3Device {
        Cl3Device::new(self.inner.id)
    }

    // ── Sub-devices ──────────────────────────────────────────────────

    /// Partition this device into `n` equally-sized sub-devices.
    /// Each sub-device gets `max_compute_units / n` compute units.
    pub fn partition_equally(&self, n: u32) -> Result<Vec<Device>> {
        // CL_DEVICE_PARTITION_EQUALLY, n, 0
        let props = [CL_DEVICE_PARTITION_EQUALLY, n as isize, 0];
        self.partition(&props)
    }

    /// Partition this device into sub-devices with the given
    /// per-partition compute-unit counts.
    pub fn partition_by_counts(&self, counts: &[u32]) -> Result<Vec<Device>> {
        // CL_DEVICE_PARTITION_BY_COUNTS, count_0, count_1, ..., 0, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END
        let mut props = Vec::with_capacity(counts.len() + 2);
        props.push(CL_DEVICE_PARTITION_BY_COUNTS);
        props.extend(counts.iter().map(|&c| c as isize));
        props.push(CL_DEVICE_PARTITION_BY_COUNTS_LIST_END);
        self.partition(&props)
    }

    fn partition(&self, props: &[isize]) -> Result<Vec<Device>> {
        // opencl3's `create_sub_devices` returns `Vec<SubDevice>`, each
        // of which calls `clReleaseDevice` on Drop. We re-take ownership
        // of the cl_device_id and `mem::forget` the SubDevice so our
        // own DeviceInner::drop is the single release path. (opencl3's
        // own `From<SubDevice> for cl_device_id` is unsound — it
        // extracts the id without forgetting, so the Drop fires on the
        // moved-from value.)
        let subs = self.cl3().create_sub_devices(props)?;
        Ok(subs
            .into_iter()
            .map(|sub| {
                let id = sub.id();
                std::mem::forget(sub);
                Device::from_sub(id, self.inner.platform.clone())
            })
            .collect())
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name().unwrap_or_default())
            .field("vendor", &self.vendor().unwrap_or_default())
            .field("type", &self.device_type())
            .field("platform", &self.inner.platform.name())
            .field("sub_device", &self.inner.is_sub_device)
            .finish()
    }
}
