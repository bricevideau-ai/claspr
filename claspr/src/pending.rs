//! Values a non-blocking terminal hands back before the device is done with
//! them.
//!
//! A non-blocking terminal (`submit` / `submit_on` / `submit_value_on`) returns
//! as soon as the work is enqueued. Whatever host memory that work still reads
//! or writes — a downloaded `Vec`, a caller's slice — must stay alive and
//! untouched until it completes. Two ways keep that sound:
//!
//! - [`Pending`]: the value (and anything the transfer borrows) comes back only
//!   through [`wait`](Pending::wait); dropping the guard waits too, so the memory
//!   can never be freed or read while the device still uses it.
//! - [`DeviceResident`]: values whose storage the host can't touch directly
//!   (buffers, images, device scalars, tensors, tiles — and tuples of them) may
//!   be handed back right away with the completion event, Tier-1 style: the
//!   caller orders later work after it with `.after(event)`.

use std::marker::PhantomData;

/// A value the device may still be writing (or a transfer still reading the
/// host memory it borrows): [`wait`](Self::wait) for it. Dropping the guard
/// without waiting blocks until the work completes, so no host memory the work
/// uses is ever freed early. `'a` is the borrow of any caller memory involved
/// (a slice read into / written from), which stays borrowed while the guard
/// lives.
#[must_use = "dropping a `Pending` blocks until the device work completes; call `wait`"]
pub struct Pending<'a, T> {
    value: Option<T>,
    event: crate::Event,
    // Whatever must outlive the work (the op, its borrows): dropped after it.
    keep: Option<Box<dyn Send + 'a>>,
    _borrow: PhantomData<&'a mut ()>,
}

impl<'a, T> Pending<'a, T> {
    pub(crate) fn new(value: T, event: crate::Event, keep: Option<Box<dyn Send + 'a>>) -> Self {
        Pending {
            value: Some(value),
            event,
            keep,
            _borrow: PhantomData,
        }
    }

    /// Wait for the work to complete and take the value.
    pub fn wait(mut self) -> crate::Result<T> {
        self.event.wait().map_err(crate::Error::OpenCl)?;
        self.keep = None;
        Ok(self.value.take().expect("a Pending is waited once"))
    }

    /// The completion event — to order other work after this one
    /// (`.after(..)` takes its own handle: clone it with `retain`).
    pub fn event(&self) -> &crate::Event {
        &self.event
    }

    /// Whether the work has completed (then [`wait`](Self::wait) won't block).
    pub fn is_complete(&self) -> bool {
        self.event
            .command_execution_status()
            .is_ok_and(|s| s.0 == opencl3::event::CL_COMPLETE)
    }
}

impl<T> Drop for Pending<'_, T> {
    fn drop(&mut self) {
        if self.value.is_some() {
            // Not waited: the device may still use the value's (or a borrowed
            // slice's) memory — wait before either is released.
            let _ = self.event.wait();
        }
        // `keep` and `value` drop after this, in field order.
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A value a non-blocking terminal may hand back before its work completes,
/// alongside the completion event: device storage the host can't reach
/// directly (or, for a fine-grain `USMSlice`, whose drop waits for in-flight
/// kernels). Sealed — host memory (a `Vec`) is not, and comes back as a
/// [`Pending`] instead.
pub trait DeviceResident: sealed::Sealed + Send {}

macro_rules! device_resident {
    ($([$($g:tt)*] $t:ty),* $(,)?) => {$(
        impl<$($g)*> sealed::Sealed for $t {}
        impl<$($g)*> DeviceResident for $t where $t: Send {}
    )*};
}

device_resident! {
    [T, M: crate::access::MemMode] crate::DeviceSlice<T, M>,
    [T, M: crate::access::MemMode] std::sync::Arc<crate::DeviceSlice<T, M>>,
    [T, M: crate::access::MemMode] crate::MappedSlice<T, M>,
    [T, M: crate::access::MemMode] crate::USMSlice<T, M>,
    [B: DeviceResident] crate::Scalar<B>,
    [B: DeviceResident, const R: usize] crate::Tensor<B, R>,
    [T, M: crate::access::MemMode, const R: usize] crate::Tile<T, M, R>,
    [A: crate::access::KernelAccess, F: crate::format::Format] crate::Image1D<A, F>,
    [A: crate::access::KernelAccess, F: crate::format::Format] crate::Image2D<A, F>,
    [A: crate::access::KernelAccess, F: crate::format::Format] crate::Image3D<A, F>,
    [A: crate::access::KernelAccess, F: crate::format::Format] crate::Image1DArray<A, F>,
    [A: crate::access::KernelAccess, F: crate::format::Format] crate::Image2DArray<A, F>,
    [A: crate::access::KernelAccess, F: crate::format::Format] crate::Image1DBuffer<A, F>,
}

impl sealed::Sealed for () {}
impl DeviceResident for () {}

macro_rules! device_resident_tuples {
    ($($name:ident)+) => {
        impl<$($name: DeviceResident),+> sealed::Sealed for ($($name,)+) {}
        impl<$($name: DeviceResident),+> DeviceResident for ($($name,)+) {}
    };
}
device_resident_tuples!(A);
device_resident_tuples!(A B);
device_resident_tuples!(A B C);
device_resident_tuples!(A B C D);
device_resident_tuples!(A B C D E);
device_resident_tuples!(A B C D E F);
device_resident_tuples!(A B C D E F G);
device_resident_tuples!(A B C D E F G H);
device_resident_tuples!(A B C D E F G H I);
device_resident_tuples!(A B C D E F G H I J);
device_resident_tuples!(A B C D E F G H I J K);
device_resident_tuples!(A B C D E F G H I J K L);
device_resident_tuples!(A B C D E F G H I J K L M);
device_resident_tuples!(A B C D E F G H I J K L M N);
device_resident_tuples!(A B C D E F G H I J K L M N O);
device_resident_tuples!(A B C D E F G H I J K L M N O P);
device_resident_tuples!(A B C D E F G H I J K L M N O P Q);
device_resident_tuples!(A B C D E F G H I J K L M N O P Q R);
device_resident_tuples!(A B C D E F G H I J K L M N O P Q R S);
device_resident_tuples!(A B C D E F G H I J K L M N O P Q R S T);
device_resident_tuples!(A B C D E F G H I J K L M N O P Q R S T U);
device_resident_tuples!(A B C D E F G H I J K L M N O P Q R S T U V);
device_resident_tuples!(A B C D E F G H I J K L M N O P Q R S T U V W);
device_resident_tuples!(A B C D E F G H I J K L M N O P Q R S T U V W X);
