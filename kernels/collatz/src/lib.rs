#![cfg_attr(target_arch = "spirv", no_std)]

use glam::USizeVec3;
use spirv_std::{glam, spirv};

/// Returns the length of the Collatz sequence for `n`, or `None` if
/// `n` is zero or the sequence overflows a `u32`.
pub fn collatz(mut n: u32) -> Option<u32> {
    let mut i = 0;
    if n == 0 {
        return None;
    }
    while n != 1 {
        n = if n.is_multiple_of(2) {
            n / 2
        } else {
            if n >= 0x5555_5555 {
                return None;
            }
            3 * n + 1
        };
        i += 1;
    }
    Some(i)
}

#[spirv(kernel)]
pub fn collatz_kernel(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
) {
    let index = id.x;
    data[index] = collatz(data[index]).unwrap_or(u32::MAX);
}
