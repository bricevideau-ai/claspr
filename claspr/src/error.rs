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
    /// A host closure inside `DeviceOpExt::and_then_host` panicked.
    /// The string is the panic payload extracted via `catch_unwind`
    /// then downcast to `&str` / `String`. The backtrace is lost, as
    /// is usual when a panic crosses a `catch_unwind` boundary.
    HostPanic(String),
    /// A reusable graph was `sync`'d while one of its typed slots
    /// (built with `slot!(Tag)`) is still unbound — completeness is
    /// checked at run time. The string is the tag's `type_name`; bind
    /// it with `g.bind(Tag(value))` before `sync`. (Also surfaced if a
    /// bound slot's buffer is still lent to a live `Checkout` — the
    /// graph is busy on that slot.)
    SlotUnbound(&'static str),
    /// A `g.bind(Tag(value))` (the set-once verb) targeted a slot that is
    /// already bound to a **different** buffer. `bind` is idempotent on an
    /// equal binding (same `cl_mem`) but rejects a conflicting one; the
    /// string is the tag's `type_name`. Use
    /// [`mutate_bind`](crate::DeviceOpExt::mutate_bind) to deliberately
    /// change a bound slot's value.
    SlotConflict(&'static str),
    /// A `g.bind` / `g.mutate_bind` targeted a slot whose buffer is currently
    /// **checked out** — lent to a live [`Checkout`](crate::Checkout) from an
    /// in-flight run. The slot's value is in the caller's hands, so re-binding
    /// it would silently clobber it (the Checkout's drop would rehome the old
    /// buffer over the new). The string is the tag's `type_name`; drop the
    /// `Checkout` (re-arming the slot) or call
    /// [`into_inner`](crate::Checkout::into_inner) (severing it) before
    /// re-binding.
    SlotCheckedOut(&'static str),
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
            Error::HostPanic(msg) => write!(f, "host closure panicked: {msg}"),
            Error::SlotUnbound(tag) => write!(
                f,
                "eager graph: slot `{tag}` is unbound — bind it with \
                 `g.bind(Tag(value))` before sync (or a previous Checkout is still \
                 holding its buffer)"
            ),
            Error::SlotConflict(tag) => write!(
                f,
                "eager graph: slot `{tag}` is already bound to a different value; \
                 use `mutate_bind` to change it"
            ),
            Error::SlotCheckedOut(tag) => write!(
                f,
                "eager graph: slot `{tag}` is currently checked out; drop the \
                 Checkout (or call into_inner) before re-binding"
            ),
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
