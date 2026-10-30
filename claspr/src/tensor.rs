//! [`Shape`]: the strided layout of a `Tensor<T, R>` / `TensorMut<T, R>` kernel
//! argument.
//!
//! A tensor parameter is two kernel args on the wire — its buffer (pointer +
//! length, like any slice) and its `Shape` by value — so the launcher takes the
//! buffer (piped, as usual) followed by a `<name>_shape` scalar (not piped; a
//! literal or a `slot!`, so a replay can re-bind it, e.g. to walk tiles). Every
//! launch first checks the shape stays inside the buffer
//! ([`check_tensor_shape`]).

use crate::eager::{ScalarInput, SlotEq, SlotValue};
use std::any::Any;

/// The layout of a rank-`R` tensor view over a buffer: element `[i0, .., i(R-1)]`
/// (each `i_d < ext[d]`) is buffer element `offset + Σ i_d * stride[d]`.
///
/// [`contiguous`](Self::contiguous) is the row-major layout of a whole buffer;
/// [`slice`](Self::slice) narrows one axis — a sub-view (tile) of the same
/// buffer. `#[repr(C)]` and matched field-for-field by the device-side `Shape`,
/// which a kernel receives by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape<const R: usize> {
    /// Buffer index of element `[0, .., 0]`.
    pub offset: usize,
    /// Extent of each axis.
    pub ext: [usize; R],
    /// Buffer-index step of each axis.
    pub stride: [usize; R],
}

impl<const R: usize> Shape<R> {
    /// Row-major (last axis fastest), dense, from buffer index 0.
    pub fn contiguous(ext: [usize; R]) -> Self {
        let mut stride = [0; R];
        let mut step = 1;
        for d in (0..R).rev() {
            stride[d] = step;
            step *= ext[d];
        }
        Shape {
            offset: 0,
            ext,
            stride,
        }
    }

    /// Number of elements the view covers (`Π ext`).
    pub fn len(&self) -> usize {
        self.ext.iter().product()
    }

    /// Whether the view covers no element.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The view narrowed to `range` along `axis` (same buffer, same strides).
    ///
    /// # Panics
    /// If `axis >= R` or `range` is not within `0..ext[axis]`.
    pub fn slice(mut self, axis: usize, range: std::ops::Range<usize>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.ext[axis],
            "Shape::slice: {range:?} is not within 0..{} on axis {axis}",
            self.ext[axis]
        );
        self.offset += range.start * self.stride[axis];
        self.ext[axis] = range.end - range.start;
        self
    }

    /// The largest buffer index the view touches, or `None` if it touches none
    /// (an empty view) or the index overflows.
    pub fn max_index(&self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        let mut max = self.offset;
        for d in 0..R {
            max = max.checked_add((self.ext[d] - 1).checked_mul(self.stride[d])?)?;
        }
        Some(max)
    }
}

/// Check that `shape` stays inside a tensor argument's buffer of `len`
/// elements: every element it reaches is a buffer element. Called by the code
/// the `#[kernel]` macro emits, before the enqueue. Not API.
#[doc(hidden)]
pub fn check_tensor_shape<const R: usize>(
    shape: &Shape<R>,
    len: usize,
    arg: &'static str,
) -> crate::Result<()> {
    let inside = shape.is_empty() || shape.max_index().is_some_and(|max| max < len);
    if inside {
        Ok(())
    } else {
        Err(crate::Error::TensorShapeMismatch {
            arg,
            offset: shape.offset,
            ext: shape.ext.to_vec(),
            stride: shape.stride.to_vec(),
            len,
        })
    }
}

// A `Shape` is a by-value scalar kernel arg and a clone-able, value-compared
// scalar slot value — like the built-in scalars.
impl<const R: usize> crate::ScalarArg for Shape<R> {}

impl<const R: usize> crate::KernelArg for Shape<R> {
    fn set(&self, exec: &mut crate::launch::__macro_support::ExecuteKernel<'_>) {
        // SAFETY: `Shape` is `#[repr(C)]` plain data matching the device-side
        // struct field for field; passed by value.
        unsafe {
            exec.set_arg(self);
        }
    }
}

impl<const R: usize> SlotValue for Shape<R> {
    fn fill_clone(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(*self))
    }
}

impl<const R: usize> SlotEq for Shape<R> {
    fn slot_eq(&self, other: &Self) -> bool {
        self == other
    }

    fn slot_digest(&self) -> u64 {
        crate::eager::slot_digest_of(self)
    }
}

impl<const R: usize> From<Shape<R>> for ScalarInput<Shape<R>> {
    fn from(shape: Shape<R>) -> Self {
        ScalarInput::Concrete(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_is_row_major() {
        let s = Shape::contiguous([3, 4, 5]);
        assert_eq!((s.offset, s.stride), (0, [20, 5, 1]));
        assert_eq!(s.max_index(), Some(59));
    }

    #[test]
    fn slice_narrows_one_axis() {
        let s = Shape::contiguous([6, 8]).slice(1, 2..5).slice(0, 1..4);
        assert_eq!((s.offset, s.ext, s.stride), (10, [3, 3], [8, 1]));
        assert_eq!(s.max_index(), Some(10 + 2 * 8 + 2));
    }

    #[test]
    fn check_rejects_a_shape_past_the_buffer() {
        let s = Shape::contiguous([4, 4]);
        assert!(check_tensor_shape(&s, 16, "t").is_ok());
        assert!(check_tensor_shape(&s, 15, "t").is_err());
        assert!(check_tensor_shape(&Shape::contiguous([0, 4]), 0, "t").is_ok());
    }
}
