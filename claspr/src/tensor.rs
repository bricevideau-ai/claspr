//! [`Tensor`]: a buffer and its [`Shape`] as ONE host value — what a
//! `Tensor<T, R>` / `TensorMut<T, R>` kernel parameter takes.
//!
//! On the wire a tensor is two kernel args — its buffer (pointer + length, like
//! any slice) and its `Shape` by value — but on the host it is one object: the
//! launcher decomposes it, and the handle hands the same `Tensor` back, so it
//! flows through pipes into the next kernel unchanged. A `Tensor` is also
//! accepted wherever its buffer is (a slice or `PerItem*` param: the shape is
//! then ignored), so a kernel can fill a tensor's buffer directly — including
//! an uninitialised one, as a `PerItemOut` target, which hands back a `Tensor`
//! over the initialised buffer. Every launch
//! with a tensor param first checks the shape stays inside the buffer
//! ([`check_tensor_shape`]).

use crate::eager::{Input, SlotEq, SlotValue, ToInput};
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

/// The geometry of a `clEnqueue{Read,Write}BufferRect` that copies a view
/// to/from a dense host array (all in BYTES except `region[1..]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub buffer_origin: [usize; 3],
    pub region: [usize; 3],
    pub buffer_row_pitch: usize,
    pub buffer_slice_pitch: usize,
    pub host_row_pitch: usize,
    pub host_slice_pitch: usize,
}

impl<const R: usize> Shape<R> {
    /// The rect copy for this view, if OpenCL can express it: rank 1–3, a
    /// contiguous innermost axis, and pitches that don't overlap (the slice
    /// pitch a multiple of the row pitch). `None` otherwise — the caller falls
    /// back (or refuses). An empty view has no rect.
    pub(crate) fn rect(&self, elem: usize) -> Option<Rect> {
        if R == 0 || R > 3 || self.is_empty() {
            return None;
        }
        let ext = |back: usize| if back < R { self.ext[R - 1 - back] } else { 1 };
        let stride = |back: usize| {
            if back < R {
                self.stride[R - 1 - back]
            } else {
                0
            }
        };
        let (e0, e1, e2) = (ext(0), ext(1), ext(2));
        if e0 > 1 && stride(0) != 1 {
            return None;
        }
        // An axis of extent 1 is never stepped: give it the tightest pitch.
        let rp = if e1 > 1 { stride(1) } else { e0 };
        let sp = if e2 > 1 { stride(2) } else { rp * e1 };
        if rp < e0 || rp == 0 || sp < rp * e1 || sp % rp != 0 {
            return None;
        }
        let (z, rem) = (self.offset / sp, self.offset % sp);
        let (y, x) = (rem / rp, rem % rp);
        Some(Rect {
            buffer_origin: [x * elem, y, z],
            region: [e0 * elem, e1, e2],
            buffer_row_pitch: rp * elem,
            buffer_slice_pitch: sp * elem,
            host_row_pitch: e0 * elem,
            host_slice_pitch: e0 * e1 * elem,
        })
    }

    /// The view's elements, row-major and dense, from `span` — the buffer from
    /// element `offset` to the view's last element (the fallback when no rect
    /// copy fits).
    pub(crate) fn gather<T: Copy>(&self, span: &[T]) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len());
        self.for_each_index(|i| out.push(span[i - self.offset]));
        out
    }

    /// Call `f` with the buffer index of every element, in row-major order.
    fn for_each_index(&self, mut f: impl FnMut(usize)) {
        if self.is_empty() {
            return;
        }
        let mut idx = [0usize; R];
        loop {
            f(self.offset + (0..R).map(|d| idx[d] * self.stride[d]).sum::<usize>());
            let mut d = R;
            loop {
                if d == 0 {
                    return;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < self.ext[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
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

/// Bare extents: the [`contiguous`](Shape::contiguous) shape.
impl<const R: usize> From<[usize; R]> for Shape<R> {
    fn from(ext: [usize; R]) -> Self {
        Shape::contiguous(ext)
    }
}

/// A buffer and its [`Shape`]: the host value of a `Tensor<T, R>` /
/// `TensorMut<T, R>` kernel parameter (see the module docs).
#[derive(Debug)]
pub struct Tensor<B, const R: usize> {
    buf: B,
    shape: Shape<R>,
}

impl<B, const R: usize> Tensor<B, R> {
    /// `buf` viewed with `shape` — a [`Shape`], or bare extents for the
    /// contiguous one (`Tensor::new(buf, [rows, cols])`).
    pub fn new(buf: B, shape: impl Into<Shape<R>>) -> Self {
        Tensor {
            buf,
            shape: shape.into(),
        }
    }

    /// The view's layout.
    pub fn shape(&self) -> Shape<R> {
        self.shape
    }

    /// The same buffer viewed with another shape (e.g. the next tile).
    pub fn with_shape(self, shape: impl Into<Shape<R>>) -> Self {
        Tensor {
            buf: self.buf,
            shape: shape.into(),
        }
    }

    /// The underlying buffer.
    pub fn buffer(&self) -> &B {
        &self.buf
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut B {
        &mut self.buf
    }

    /// The underlying buffer, dropping the shape.
    pub fn into_buffer(self) -> B {
        self.buf
    }

    /// The buffer and the shape.
    pub fn into_parts(self) -> (B, Shape<R>) {
        (self.buf, self.shape)
    }
}

/// A host value that carries a rank-`R` [`Shape`] for a tensor kernel param.
/// The `#[kernel]` macro bounds `Tensor<T, R>` params by it (with the slice
/// trait) and passes `shape()` as the arg after the buffer.
pub trait KernelTensorArg<const R: usize> {
    /// The layout passed to the kernel.
    fn shape(&self) -> Shape<R>;
}

impl<B, const R: usize> KernelTensorArg<R> for Tensor<B, R> {
    fn shape(&self) -> Shape<R> {
        self.shape
    }
}

// The buffer traits delegate to the buffer: the shape is a separate kernel arg
// (the macro passes it), so as a slice a tensor is just its buffer.
impl<B: crate::launch::kernel_slice_arg_sealed::Sealed, const R: usize>
    crate::launch::kernel_slice_arg_sealed::Sealed for Tensor<B, R>
{
}

impl<B: crate::KernelArg, const R: usize> crate::KernelArg for Tensor<B, R> {
    fn set(&self, exec: &mut crate::launch::__macro_support::ExecuteKernel<'_>) {
        self.buf.set(exec);
    }
    fn register_completion(&self, event: &::opencl3::event::Event) {
        self.buf.register_completion(event);
    }
}

impl<B: crate::KernelPointerArg, const R: usize> crate::KernelPointerArg for Tensor<B, R> {
    fn set_pointer_only(&self, exec: &mut crate::launch::__macro_support::ExecuteKernel<'_>) {
        self.buf.set_pointer_only(exec);
    }
}

impl<T, B: crate::KernelSliceReadArg<T>, const R: usize> crate::KernelSliceReadArg<T>
    for Tensor<B, R>
{
    fn element_count(&self) -> usize {
        self.buf.element_count()
    }
    fn record_handle(&self) -> crate::Result<crate::record::BufHandle> {
        self.buf.record_handle()
    }
}

impl<T, B: crate::KernelSliceReadWriteArg<T>, const R: usize> crate::KernelSliceReadWriteArg<T>
    for Tensor<B, R>
{
}

impl<B: SlotEq, const R: usize> SlotEq for Tensor<B, R> {
    fn slot_eq(&self, other: &Self) -> bool {
        self.buf.slot_eq(&other.buf) && self.shape == other.shape
    }

    fn slot_digest(&self) -> u64 {
        crate::eager::slot_digest_of((self.buf.slot_digest(), self.shape))
    }
}

impl<B: SlotValue, const R: usize> SlotValue for Tensor<B, R> {
    fn fill_clone(&self) -> Option<Box<dyn Any + Send>> {
        // A clone-able buffer (an `Arc`) fans out with its shape; a move-only
        // one is taken once, like the bare buffer.
        let buf = self.buf.fill_clone()?.downcast::<B>().ok()?;
        Some(Box::new(Tensor {
            buf: *buf,
            shape: self.shape,
        }))
    }
}

impl<E, B, const R: usize> ToInput<E> for Tensor<B, R> {
    type Buf = Tensor<B, R>;
    fn to_input(self) -> Input<Tensor<B, R>> {
        Input::from(self)
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
    fn rect_of_a_tile() {
        // Rows 1..3, cols 2..5 of a 4 x 6 f32 matrix.
        let s = Shape::contiguous([4, 6]).slice(0, 1..3).slice(1, 2..5);
        let r = s.rect(4).expect("rect");
        assert_eq!(r.buffer_origin, [2 * 4, 1, 0]);
        assert_eq!(r.region, [3 * 4, 2, 1]);
        assert_eq!((r.buffer_row_pitch, r.host_row_pitch), (6 * 4, 3 * 4));
        // 3-D: a sub-box of a 2 x 3 x 4 tensor.
        let s = Shape::contiguous([2, 3, 4]).slice(1, 1..3).slice(2, 1..3);
        let r = s.rect(1).expect("rect");
        assert_eq!((r.buffer_origin, r.region), ([1, 1, 0], [2, 2, 2]));
        assert_eq!((r.buffer_row_pitch, r.buffer_slice_pitch), (4, 12));
    }

    #[test]
    fn no_rect_for_a_strided_inner_axis_or_rank_four() {
        let t = Shape {
            offset: 0,
            ext: [3, 2],
            stride: [1, 3],
        }; // a transposed view
        assert_eq!(t.rect(4), None);
        assert_eq!(Shape::contiguous([2, 2, 2, 2]).rect(4), None);
    }

    #[test]
    fn gather_reads_a_transposed_view() {
        let buf: Vec<u32> = (0..6).collect(); // 2 x 3, row-major
        let t = Shape {
            offset: 0,
            ext: [3, 2],
            stride: [1, 3],
        };
        assert_eq!(t.gather(&buf), [0, 3, 1, 4, 2, 5]);
        let tile = Shape::contiguous([4, 4]).slice(0, 1..3).slice(1, 1..3);
        let buf: Vec<u32> = (0..16).collect();
        assert_eq!(tile.gather(&buf[tile.offset..]), [5, 6, 9, 10]);
    }

    #[test]
    fn check_rejects_a_shape_past_the_buffer() {
        let s = Shape::contiguous([4, 4]);
        assert!(check_tensor_shape(&s, 16, "t").is_ok());
        assert!(check_tensor_shape(&s, 15, "t").is_err());
        assert!(check_tensor_shape(&Shape::contiguous([0, 4]), 0, "t").is_ok());
    }
}
