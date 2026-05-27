//! claspr's [`Error`] type and [`Result`] alias.
//!
//! Single typed enum so callers can `match` on the failure mode
//! (compile failure, OpenCL status, missing capability, …) instead of
//! string-sniffing a boxed trait object.
//!
//! Variants are added on demand — `#[non_exhaustive]` so a future
//! addition doesn't break `match` arms in downstream code. The audit
//! pass that removed `Error::Other`, `Error::KernelArg`, and
//! `Error::InvalidWorkSize` confirmed every variant below is
//! constructed at at least one in-tree call site.

use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An OpenCL API call returned a non-success status.
    OpenCl(opencl3::error_codes::ClError),
    /// Program build failed; carries the build log from `clGetProgramBuildInfo`.
    Build { log: String },
    /// A buffer download / upload had mismatched lengths.
    LengthMismatch { src: usize, dst: usize },
    /// The selected device or runtime doesn't support a requested feature.
    NotSupported(&'static str),
    /// SVM (shared virtual memory) was requested but not available on this device.
    SvmNotAvailable,
    /// A [`LaunchOp::profiled`](crate::op::LaunchOp::profiled) closure was
    /// registered, but the target queue lacks `CL_QUEUE_PROFILING_ENABLE`.
    /// Build the context with [`ContextBuilder::profiling(true)`](crate::context::ContextBuilder::profiling)
    /// so the per-device default queues — and any
    /// [`Queue::new`](crate::queue::Queue::new) /
    /// [`Queue::on_device`](crate::queue::Queue::on_device) built off it —
    /// inherit profiling.
    ProfilingDisabled,
    /// I/O error (reading a SPIR-V file, writing a PPM, …). Constructed
    /// via the [`From<io::Error>`] impl below — any `?` on a fallible
    /// I/O call inside claspr lands here automatically.
    ///
    /// [`From<io::Error>`]: #impl-From%3CError%3E-for-Error
    Io(io::Error),
    /// A function argument failed a validation check (empty slice,
    /// usize overflow, malformed name, …). The string is a static
    /// description — the offending value is the caller's, not part
    /// of the error.
    InvalidArgument(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OpenCl(e) => write!(f, "OpenCL error: {e}"),
            Error::Build { log } => write!(f, "program build failed:\n{log}"),
            Error::LengthMismatch { src, dst } => {
                write!(f, "length mismatch: src has {src} elements, dst has {dst}")
            }
            Error::NotSupported(what) => write!(f, "not supported on this device: {what}"),
            Error::SvmNotAvailable => {
                f.write_str("SVM (shared virtual memory) not available on this device")
            }
            Error::ProfilingDisabled => f.write_str(
                "queue does not have CL_QUEUE_PROFILING_ENABLE \
                 (build the Context with .profiling(true))",
            ),
            Error::Io(e) => write!(f, "I/O: {e}"),
            Error::InvalidArgument(what) => write!(f, "invalid argument: {what}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::OpenCl(e) => Some(e),
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<opencl3::error_codes::ClError> for Error {
    fn from(e: opencl3::error_codes::ClError) -> Self {
        Error::OpenCl(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
