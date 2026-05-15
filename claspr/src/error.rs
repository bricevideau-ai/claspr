//! claspr's [`Error`] type and [`Result`] alias.
//!
//! Single typed enum so callers can `match` on the failure mode
//! (compile failure, OpenCL status, missing capability) instead of
//! string-sniffing a boxed trait object.

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
    /// A kernel argument couldn't be set (wrong type, wrong index, etc).
    KernelArg(String),
    /// Work-size geometry is invalid (zero dims, mismatched lengths,
    /// exceeds device max, …).
    InvalidWorkSize(String),
    /// A buffer download / upload had mismatched lengths.
    LengthMismatch { src: usize, dst: usize },
    /// The selected device or runtime doesn't support a requested feature.
    NotSupported(&'static str),
    /// SVM (shared virtual memory) was requested but not available on this device.
    SvmNotAvailable,
    /// I/O error (reading a SPIR-V file, writing a PPM, …).
    Io(io::Error),
    /// Free-form message for cases the typed variants don't cover yet.
    /// New strongly-typed variants should subsume these over time.
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OpenCl(e) => write!(f, "OpenCL error: {e}"),
            Error::Build { log } => write!(f, "program build failed:\n{log}"),
            Error::KernelArg(msg) => write!(f, "kernel argument: {msg}"),
            Error::InvalidWorkSize(msg) => write!(f, "invalid work size: {msg}"),
            Error::LengthMismatch { src, dst } => {
                write!(f, "length mismatch: src has {src} elements, dst has {dst}")
            }
            Error::NotSupported(what) => write!(f, "not supported on this device: {what}"),
            Error::SvmNotAvailable => {
                f.write_str("SVM (shared virtual memory) not available on this device")
            }
            Error::Io(e) => write!(f, "I/O: {e}"),
            Error::Other(msg) => f.write_str(msg),
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

// String / &str conversions cover the existing `format!(...).into()`
// and `"...".into()` patterns. New code should prefer the typed
// variants — these shims exist so the migration doesn't require
// touching every fallible call site at once.
impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Other(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Other(s.to_owned())
    }
}
