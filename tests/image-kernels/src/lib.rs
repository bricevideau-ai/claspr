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
//! - [`mod@dim3_uint`] — 3D / `type=u32` / write-only fill.
//!   Proves the `Image3D` runtime + `KernelImage3D*Arg` trait
//!   family round-trip.
//!
//! 1D image kernels are intentionally absent — rust-gpu is missing
//! the auto-declare for `OpCapability Image1D` when an
//! `OpTypeImage Dim=1D` is emitted on Kernel; spirv-val rejects.
//! 3D needs no extra capability (Dim=3D carries no per-Dim cap
//! requirement in the SPIR-V core spec) so it is exercised here.

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

// Dim1D kernels are intentionally absent — rust-gpu does not yet
// auto-declare `OpCapability Image1D` when an `OpTypeImage Dim=1D`
// is emitted on Kernel; spirv-val then rejects the module.
// Once rust-gpu auto-declares the dim capability, add a
// `dim1_uint` module mirroring `dim3_uint`. The Image1D runtime +
// KernelImage1D*Arg traits already exist in claspr; only the
// kernel side is gated.

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
