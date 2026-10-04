//! Allocation-free vector integer checking; the cold caller proves ISA and native-width profile.
//! Carry/borrow/sign identities depend only on lane bit patterns and wrapping lane arithmetic.
//! Reject validates the whole call before publication; Clamp has only useful range faults.
#[rustfmt::skip]
use fusion_pcu::{PcuCheckedInteger, PcuDispatchIntegerBinaryOp, PcuRangePolicy, PcuExecutionFault};
use super::{Executable, PcuCpuCheckedIntegerError};
use crate::PcuCpuImplementation;
#[cfg(target_arch = "aarch64")]
#[path = "neon/neon.rs"]
mod neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[path = "x86/x86.rs"]
mod x86;

/// Native vector methods are static monomorphized calls, never a warm trait-object table.
trait Vector {
    type Bits: Copy;
    const BYTES: usize;
    unsafe fn load(pointer: *const u8) -> Self::Bits;
    unsafe fn store(pointer: *mut u8, value: Self::Bits);
    unsafe fn add<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits;
    unsafe fn sub<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits;
    unsafe fn and(a: Self::Bits, b: Self::Bits) -> Self::Bits;
    unsafe fn or(a: Self::Bits, b: Self::Bits) -> Self::Bits;
    unsafe fn xor(a: Self::Bits, b: Self::Bits) -> Self::Bits;
    unsafe fn and_not(a: Self::Bits, b: Self::Bits) -> Self::Bits;
    /// All bits of each lane become its original sign bit.
    unsafe fn signs<const SIZE: usize>(value: Self::Bits) -> Self::Bits;
    /// One bit per byte. Only the high byte of each masked native lane is nonzero.
    unsafe fn mask(value: Self::Bits) -> u64;
    /// Enters this vector family's guarded target-feature loop.
    #[allow(clippy::too_many_arguments)] // Same complete transaction boundary for each ISA.
    unsafe fn execute<
        T: PcuCheckedInteger,
        const OP: u8,
        const CLAMP: bool,
        const LB: bool,
        const RB: bool,
    >(
        left: &[u8],
        right: &[u8],
        output: &mut [u8],
        extent: usize,
        broadcast_left: &[u8; 64],
        broadcast_right: &[u8; 64],
    ) -> Result<(), PcuCpuCheckedIntegerError>;
}

pub(super) fn prepare<T: PcuCheckedInteger>(
    implementation: PcuCpuImplementation,
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    left: bool,
    right: bool,
) -> Result<Executable, PcuCpuCheckedIntegerError> {
    macro_rules! layouts {
        ($arch:ty,$op:expr,$clamp:expr) => {
            match (left, right) {
                (false, false) => run::<T, $arch, $op, $clamp, false, false>,
                (false, true) => run::<T, $arch, $op, $clamp, false, true>,
                (true, false) => run::<T, $arch, $op, $clamp, true, false>,
                (true, true) => run::<T, $arch, $op, $clamp, true, true>,
            }
        };
    }
    macro_rules! operation {
        ($arch:ty) => {
            match (op, range) {
                (PcuDispatchIntegerBinaryOp::Add, PcuRangePolicy::Reject) => {
                    layouts!($arch, 0, false)
                }
                (PcuDispatchIntegerBinaryOp::Sub, PcuRangePolicy::Reject) => {
                    layouts!($arch, 1, false)
                }
                (PcuDispatchIntegerBinaryOp::Add, PcuRangePolicy::Clamp) => {
                    layouts!($arch, 0, true)
                }
                (PcuDispatchIntegerBinaryOp::Sub, PcuRangePolicy::Clamp) => {
                    layouts!($arch, 1, true)
                }
                _ => return Err(PcuCpuCheckedIntegerError::UnsupportedProfile),
            }
        };
    }
    if !matches!(T::HOST_SIZE, 1 | 2 | 4 | 8) || !cfg!(target_endian = "little") {
        return Err(PcuCpuCheckedIntegerError::UnsupportedProfile);
    }
    let selected = match implementation {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Sse2 => operation!(x86::Sse2),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Avx2 => operation!(x86::Avx2),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Avx512 => operation!(x86::Avx512),
        #[cfg(target_arch = "aarch64")]
        PcuCpuImplementation::Neon => operation!(neon::Neon),
        _ => return Err(PcuCpuCheckedIntegerError::UnsupportedProfile),
    };
    Ok(selected)
}

fn run<
    T: PcuCheckedInteger,
    V: Vector,
    const OP: u8,
    const CLAMP: bool,
    const LB: bool,
    const RB: bool,
>(
    left: &[u8],
    right: &[u8],
    output: &mut [u8],
    extent: usize,
) -> Result<(), PcuCpuCheckedIntegerError> {
    // Cold admission plus public host-schema checks prove width/ISA/full prefix. Unaligned
    // vector loads never overread a scalar broadcast, odd tail or caller padding.
    let size = T::HOST_SIZE;
    let left = &left[..if LB { size } else { extent * size }];
    let right = &right[..if RB { size } else { extent * size }];
    let output = &mut output[..extent * size];
    let mut a = [0u8; 64];
    let mut b = [0u8; 64];
    if LB {
        for chunk in a.chunks_exact_mut(size) {
            chunk.copy_from_slice(left);
        }
    }
    if RB {
        for chunk in b.chunks_exact_mut(size) {
            chunk.copy_from_slice(right);
        }
    }
    // SAFETY: Private prepared function pointer is chosen only after detected ISA admission;
    // T is sealed, byte spans are complete/disjoint, broadcasts copied to initialized64-byte storage.
    unsafe { V::execute::<T, OP, CLAMP, LB, RB>(left, right, output, extent, &a, &b) }
}

fn scalar<T: PcuCheckedInteger, const OP: u8>(
    a: T,
    b: T,
) -> Result<T, fusion_pcu::PcuExecutionFaultKind> {
    if OP == 0 {
        a.pcu_checked_add(b)
    } else {
        a.pcu_checked_sub(b)
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // The vector transaction and scalar exact fault/tail bridge remain one guarded loop law.
#[allow(clippy::inline_always)] // Inline the shared loop into its ISA wrapper so target-feature calls vanish per vector.
#[inline(always)]
unsafe fn execute<
    T: PcuCheckedInteger,
    V: Vector,
    const OP: u8,
    const CLAMP: bool,
    const LB: bool,
    const RB: bool,
>(
    left: &[u8],
    right: &[u8],
    output: &mut [u8],
    extent: usize,
    broadcast_left: &[u8; 64],
    broadcast_right: &[u8; 64],
) -> Result<(), PcuCpuCheckedIntegerError> {
    let size = T::HOST_SIZE;
    let lanes = V::BYTES / size;
    let vectors = extent / lanes;
    let prefix = vectors * lanes;
    let signed = matches!(
        T::TYPE,
        fusion_pcu::PcuScalarType::I8
            | fusion_pcu::PcuScalarType::I16
            | fusion_pcu::PcuScalarType::I32
            | fusion_pcu::PcuScalarType::I64
    );
    let mut sign = [0u8; 64];
    let mut maximum = [255u8; 64];
    let mut minimum = [0u8; 64];
    for i in (size - 1..V::BYTES).step_by(size) {
        sign[i] = 128;
        if signed {
            maximum[i] = 127;
            minimum[i] = 128;
        }
    }
    // SAFETY: Every method is guarded by the cold ISA proof. Local masks contain64 initialized
    // bytes; native spans cover every complete vector below prefix. No typed alignment required.
    unsafe {
        let sign = V::load(sign.as_ptr());
        let maximum = V::load(maximum.as_ptr());
        let minimum = V::load(minimum.as_ptr());
        let load = |index: usize| {
            let a = if LB {
                broadcast_left.as_ptr()
            } else {
                left.as_ptr().add(index * size)
            };
            let b = if RB {
                broadcast_right.as_ptr()
            } else {
                right.as_ptr().add(index * size)
            };
            (V::load(a), V::load(b))
        };
        let lane = |index: usize| {
            (
                left.as_ptr()
                    .add(if LB { 0 } else { index * size })
                    .cast::<T>()
                    .read_unaligned(),
                right
                    .as_ptr()
                    .add(if RB { 0 } else { index * size })
                    .cast::<T>()
                    .read_unaligned(),
            )
        };
        let calculate = |a, b| match size {
            1 => {
                if OP == 0 {
                    V::add::<1>(a, b)
                } else {
                    V::sub::<1>(a, b)
                }
            }
            2 => {
                if OP == 0 {
                    V::add::<2>(a, b)
                } else {
                    V::sub::<2>(a, b)
                }
            }
            4 => {
                if OP == 0 {
                    V::add::<4>(a, b)
                } else {
                    V::sub::<4>(a, b)
                }
            }
            _ => {
                if OP == 0 {
                    V::add::<8>(a, b)
                } else {
                    V::sub::<8>(a, b)
                }
            }
        };
        let signs = |value| match size {
            1 => V::signs::<1>(value),
            2 => V::signs::<2>(value),
            4 => V::signs::<4>(value),
            _ => V::signs::<8>(value),
        };
        let overflow = |a, b, r| {
            let bits = if signed {
                if OP == 0 {
                    V::and(V::xor(a, r), V::xor(b, r))
                } else {
                    V::and(V::xor(a, b), V::xor(a, r))
                }
            } else if OP == 0 {
                V::or(V::and(a, b), V::and_not(r, V::or(a, b)))
            } else {
                V::or(V::and_not(a, b), V::and_not(V::xor(a, b), r))
            };
            V::and(bits, sign)
        };
        let fault = |index: usize| {
            let (a, b) = lane(index);
            let Err(kind) = scalar::<T, OP>(a, b) else {
                unreachable!("exact carry/sign flag proves sealed checked range fault")
            };
            PcuExecutionFault {
                kind,
                invocation_id: u64::try_from(index).expect("cold u32 extent"),
                recovered: CLAMP,
            }
        };
        let mut first = None;
        for vector in 0..vectors {
            let index = vector * lanes;
            let (a, b) = load(index);
            let r = calculate(a, b);
            let flag = overflow(a, b, r);
            let mask = V::mask(flag);
            if mask != 0 {
                let index = index + (mask.trailing_zeros() as usize) / size;
                let f = fault(index);
                if !CLAMP {
                    return Err(PcuCpuCheckedIntegerError::Fault(f));
                }
                first.get_or_insert(f);
            }
            if CLAMP {
                let bounds = if signed {
                    let negative = signs(a);
                    V::or(V::and(negative, minimum), V::and_not(negative, maximum))
                } else if OP == 0 {
                    maximum
                } else {
                    minimum
                };
                let flag = signs(flag);
                let useful = V::or(V::and(flag, bounds), V::and_not(flag, r));
                V::store(output.as_mut_ptr().add(index * size), useful);
            }
        }
        for index in prefix..extent {
            let (a, b) = lane(index);
            if CLAMP {
                let value = match if OP == 0 {
                    a.pcu_clamped_add(b)
                } else {
                    a.pcu_clamped_sub(b)
                } {
                    Ok(v) => v,
                    Err(f) => {
                        first.get_or_insert_with(|| PcuExecutionFault {
                            kind: f.kind(),
                            invocation_id: u64::try_from(index).expect("cold u32 extent"),
                            recovered: true,
                        });
                        f.clamped_value()
                    }
                };
                output
                    .as_mut_ptr()
                    .add(index * size)
                    .cast::<T>()
                    .write_unaligned(value);
            } else {
                scalar::<T, OP>(a, b).map_err(|kind| super::super::fault(index, kind))?;
            }
        }
        if CLAMP {
            return first.map_or(Ok(()), |f| Err(PcuCpuCheckedIntegerError::Fault(f)));
        }
        // Input values cannot change across this synchronous safe exclusive host call. Whole-call
        // preflight above proves wrapping lane results exactly equal the accepted scalar arithmetic.
        for vector in 0..vectors {
            let index = vector * lanes;
            let (a, b) = load(index);
            V::store(output.as_mut_ptr().add(index * size), calculate(a, b));
        }
        for index in prefix..extent {
            let (a, b) = lane(index);
            let value = scalar::<T, OP>(a, b).expect("whole-call checked preflight");
            output
                .as_mut_ptr()
                .add(index * size)
                .cast::<T>()
                .write_unaligned(value);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
