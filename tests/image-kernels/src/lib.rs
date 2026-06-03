//! Image kernels for claspr's runtime integration tests.
//!
//! Lives in its own crate (separate from `claspr-test-kernels`) because
//! image kernels need the `image()` build preset — no
//! `DebugPrintfThenExit` panic strategy, which conflicts with image
//! emission. See `build.rs`.
//!
//! ## Coverage
//!
//! Each module exercises one (dim, sampled-family, access) corner of
//! the claspr image trait dispatch:
//!
//! - [`mod@dim2_uint`] — 2D / `type=u32` / write-only. Pairs with
//!   non-default formats (`R32Uint`, `R32G32B32A32Uint`) in the
//!   runtime tests. (`ReadWrite` is not exercised here — `OpCapability
//!   ImageReadWrite` requires OpenCL 2.0+ and is rejected by the 1.2
//!   `image()` preset; covered in a separate OCL 2.0 image-kernels
//!   crate when added.)
//! - [`mod@dim2_float`] — 2D / `type=f32` / write-only fill +
//!   read-only image→buffer copy. Float family.
//! - [`mod@dim2_sint`] — 2D / `type=i32` / write-only fill +
//!   read-only image→buffer copy. Sint family.
//!
//! - [`mod@dim1_uint`] — 1D / `type=u32` / write-only fill.
//!   Proves the `Image1D` runtime + `KernelImage1D*Arg` trait
//!   family round-trip.
//! - [`mod@dim3_uint`] — 3D / `type=u32` / write-only fill.
//!   Proves the `Image3D` runtime + `KernelImage3D*Arg` trait
//!   family round-trip.
//! - [`mod@dim_buffer_uint`] — 1D-buffer / `type=u32` /
//!   write-only fill + read-only image→buffer copy. Proves the
//!   `Image1DBuffer` runtime + `KernelImageBuffer*Arg` trait
//!   family + the rust-gpu auto-declare of `OpCapability
//!   ImageBuffer` for `OpTypeImage Dim=Buffer`.
//! - [`mod@dim1_array_uint`] — 1D-array / `type=u32` /
//!   write-only fill. Coord is `IVec2(x, layer)`. Proves the
//!   `Image1DArray` runtime + `KernelImage1DArray*Arg` trait
//!   family + the spirv-std `(OneD, Arrayed::True)` coord impl.
//! - [`mod@dim2_array_uint`] — 2D-array / `type=u32` /
//!   write-only fill. Coord is `IVec3(x, y, layer)`. Proves
//!   the `Image2DArray` runtime + `KernelImage2DArray*Arg`
//!   trait family.

#[claspr::device]
pub mod dim2_uint {
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        cl::Int2,
        glam::{USizeVec3, UVec4},
    };

    /// Write `(x + y*width, 0, 0, 0xFFFF_FFFF)` at every pixel.
    /// Default access (write-only via explicit attribute).
    #[claspr::kernel]
    pub fn fill_pattern(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(image_access = "write_only")] image: &mut Image!(2D, type=u32, sampled=false),
        width: u32,
        height: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        if px >= width || py >= height {
            return;
        }
        let v = px + py * width;
        unsafe {
            image.write(
                Int2::new(px as i32, py as i32),
                UVec4::new(v, 0, 0, 0xFFFF_FFFF),
            );
        }
    }
}

#[claspr::device]
pub mod dim2_float {
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        cl::Int2,
        glam::{USizeVec3, Vec4},
    };

    /// Write `(x as f32, y as f32, 0.0, 1.0)` per pixel.
    #[claspr::kernel]
    pub fn fill_pattern(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(image_access = "write_only")] image: &mut Image!(2D, type=f32, sampled=false),
        width: u32,
        height: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        if px >= width || py >= height {
            return;
        }
        unsafe {
            image.write(
                Int2::new(px as i32, py as i32),
                Vec4::new(px as f32, py as f32, 0.0, 1.0),
            );
        }
    }

    /// Read every pixel from the image; write the .x component to a
    /// linear buffer. Proves read-only access + cross-arg
    /// (image + slice) interaction.
    #[claspr::kernel]
    pub fn copy_to_buffer(
        #[spirv(global_invocation_id)] id: USizeVec3,
        image: &Image!(2D, type=f32, sampled=false),
        #[spirv(cross_workgroup)] out: &mut [f32],
        width: u32,
        height: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        if px >= width || py >= height {
            return;
        }
        let pixel: Vec4 = unsafe { image.read(Int2::new(px as i32, py as i32)) };
        let i = (py * width + px) as usize;
        out[i] = out[i] * 0.0 + pixel.x;
    }
}

#[claspr::device]
pub mod dim2_sint {
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        cl::Int2,
        glam::{IVec4, USizeVec3},
    };

    #[claspr::kernel]
    pub fn fill_pattern(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(image_access = "write_only")] image: &mut Image!(2D, type=i32, sampled=false),
        width: u32,
        height: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        if px >= width || py >= height {
            return;
        }
        let v = (px as i32) - (py as i32);
        unsafe {
            image.write(Int2::new(px as i32, py as i32), IVec4::new(v, -v, 0, 1));
        }
    }

    #[claspr::kernel]
    pub fn copy_to_buffer(
        #[spirv(global_invocation_id)] id: USizeVec3,
        image: &Image!(2D, type=i32, sampled=false),
        #[spirv(cross_workgroup)] out: &mut [i32],
        width: u32,
        height: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        if px >= width || py >= height {
            return;
        }
        let pixel: IVec4 = unsafe { image.read(Int2::new(px as i32, py as i32)) };
        let i = (py * width + px) as usize;
        out[i] = out[i].wrapping_mul(0).wrapping_add(pixel.x);
    }
}

#[claspr::device]
pub mod dim1_uint {
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        glam::{USizeVec3, UVec4},
    };

    /// 1D image fill. Coord is a bare `i32` per spirv-std's
    /// `ImageCoordinate<S: Scalar, Dim::OneD, Arrayed::False>`
    /// scalar impl.
    #[claspr::kernel]
    pub fn fill_pattern(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(image_access = "write_only")] image: &mut Image!(1D, type=u32, sampled=false),
        width: u32,
    ) {
        let px = id.x as u32;
        if px >= width {
            return;
        }
        unsafe {
            image.write(px as i32, UVec4::new(px, 0, 0, 0));
        }
    }
}

#[claspr::device]
pub mod dim3_uint {
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        glam::{IVec3, USizeVec3, UVec4},
    };

    /// 3D image fill. Coord is `IVec3` per spirv-std's
    /// `ImageCoordinate<S, Dim::ThreeD, Arrayed::False>` impl
    /// (a 3-component integer vector).
    #[claspr::kernel]
    pub fn fill_pattern(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(image_access = "write_only")] image: &mut Image!(3D, type=u32, sampled=false),
        width: u32,
        height: u32,
        depth: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        let pz = id.z as u32;
        if px >= width || py >= height || pz >= depth {
            return;
        }
        let v = px + py * width + pz * width * height;
        unsafe {
            image.write(
                IVec3::new(px as i32, py as i32, pz as i32),
                UVec4::new(v, 0, 0, 0),
            );
        }
    }
}

#[claspr::device]
pub mod dim_buffer_uint {
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        glam::{USizeVec3, UVec4},
    };

    /// Image-buffer write fill — same shape as `dim1_uint` since
    /// 1D and buffer images both take a scalar `i32` coordinate
    /// and write `UVec4` typed pixels.
    #[claspr::kernel]
    pub fn fill_pattern(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(image_access = "write_only")] image: &mut Image!(buffer, type=u32, sampled=false),
        width: u32,
    ) {
        let px = id.x as u32;
        if px >= width {
            return;
        }
        unsafe {
            image.write(px as i32, UVec4::new(px.wrapping_mul(3), 0, 0, 0));
        }
    }

    /// Read every image-buffer pixel and copy the .x channel to a
    /// linear slice. Proves the read-only image-buffer path +
    /// kernel can pair image-buffer with a regular `&mut [T]`
    /// slice arg in the same launch.
    #[claspr::kernel]
    pub fn copy_to_buffer(
        #[spirv(global_invocation_id)] id: USizeVec3,
        image: &Image!(buffer, type=u32, sampled=false),
        #[spirv(cross_workgroup)] out: &mut [u32],
        width: u32,
    ) {
        let px = id.x as u32;
        if px >= width {
            return;
        }
        let pixel: UVec4 = unsafe { image.read(px as i32) };
        let i = px as usize;
        out[i] = out[i].wrapping_mul(0).wrapping_add(pixel.x);
    }
}

#[claspr::device]
pub mod dim1_array_uint {
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        glam::{IVec2, USizeVec3, UVec4},
    };

    /// 1D-array image fill. Coord is `IVec2(x, layer)`. Per-layer
    /// pattern: `pixel = layer * width + x` so we can verify each
    /// layer independently from the host side after download.
    #[claspr::kernel]
    pub fn fill_pattern(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(image_access = "write_only")] image: &mut Image!(1D, arrayed=true, type=u32, sampled=false),
        width: u32,
        array_size: u32,
    ) {
        let px = id.x as u32;
        let layer = id.y as u32;
        if px >= width || layer >= array_size {
            return;
        }
        let v = layer.wrapping_mul(width).wrapping_add(px);
        unsafe {
            image.write(IVec2::new(px as i32, layer as i32), UVec4::new(v, 0, 0, 0));
        }
    }
}

#[claspr::device]
pub mod dim2_array_uint {
    #[cfg(target_arch = "spirv")]
    use spirv_std::{
        Image,
        glam::{IVec3, USizeVec3, UVec4},
    };

    /// 2D-array image fill. Coord is `IVec3(x, y, layer)`.
    /// Per-layer pattern packs the layer into the .x channel so
    /// host-side verification can distinguish layers.
    #[claspr::kernel]
    pub fn fill_pattern(
        #[spirv(global_invocation_id)] id: USizeVec3,
        #[spirv(image_access = "write_only")] image: &mut Image!(2D, arrayed=true, type=u32, sampled=false),
        width: u32,
        height: u32,
        array_size: u32,
    ) {
        let px = id.x as u32;
        let py = id.y as u32;
        let layer = id.z as u32;
        if px >= width || py >= height || layer >= array_size {
            return;
        }
        let v = layer
            .wrapping_mul(width)
            .wrapping_mul(height)
            .wrapping_add(py.wrapping_mul(width))
            .wrapping_add(px);
        unsafe {
            image.write(
                IVec3::new(px as i32, py as i32, layer as i32),
                UVec4::new(v, 0, 0, 0),
            );
        }
    }
}
