//! [`UploadSource`] — the polymorphic host-data source shared by the
//! `upload` graph leaf (in `eager.rs`).
//!
//! ## Sharing host data: [`UploadSource`]
//!
//! `upload` accepts any `impl Into<UploadSource<T>>` — currently
//! `Vec<T>`, `Box<[T]>`, and `Arc<[T]>`. The `Arc<[T]>` variant lets
//! the caller keep a clone of the source for their own use or upload
//! the same data to multiple buffers without copying:
//!
//! ```ignore
//! use std::sync::Arc;
//! let weights: Arc<[f32]> = Arc::from(vec![0.1, 0.2, 0.3]);
//! let buf_a = upload(Arc::clone(&weights)).sync(&ctx)?;
//! let buf_b = upload(Arc::clone(&weights)).sync(&ctx)?;
//! // weights still usable here; data heap not freed until all Arcs
//! // (including the ones held by the keep-alive callbacks) drop.
//! ```

use std::sync::Arc;

// ── UploadSource ────────────────────────────────────────────────────

/// Polymorphic host-data source for the [`Upload`](crate::Upload) leaf. Concrete variants
/// cover the common cases — `Vec<T>` (move and forget), `Box<[T]>`
/// (heap-allocated slice), `Arc<[T]>` (shared / caller retains a
/// clone). Construct via [`From`] / [`Into`].
pub enum UploadSource<T> {
    Vec(Vec<T>),
    Box(Box<[T]>),
    Arc(Arc<[T]>),
}

impl<T> UploadSource<T> {
    /// Borrow the underlying slice. Stable address across the
    /// lifetime of the [`UploadSource`] — OpenCL is reading from it
    /// during the non-blocking write.
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Vec(v) => v,
            Self::Box(b) => b,
            Self::Arc(a) => a,
        }
    }

    /// Element count.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// `true` if the source has zero elements.
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}

impl<T> From<Vec<T>> for UploadSource<T> {
    fn from(v: Vec<T>) -> Self {
        Self::Vec(v)
    }
}

impl<T> From<Box<[T]>> for UploadSource<T> {
    fn from(b: Box<[T]>) -> Self {
        Self::Box(b)
    }
}

impl<T> From<Arc<[T]>> for UploadSource<T> {
    fn from(a: Arc<[T]>) -> Self {
        Self::Arc(a)
    }
}

// The host→device upload and device→host download graph leaves live in
// `eager.rs` (`upload` / `download`). This module retains only the
// [`UploadSource`] enum they share.
