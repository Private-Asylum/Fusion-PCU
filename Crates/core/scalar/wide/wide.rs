//! Allocation-free wide integer references and canonical little-endian carriers.
//!
//! Limbs are least-significant first. Signed carriers use two's complement.
//! No operator overload silently wraps or panics: exceptional arithmetic returns
//! a PCU fault independent of optimization level. These are reference/storage
//! contracts, not provider admission or constant-time cryptographic primitives.
#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuScalar,
    PcuScalarType,
    PcuCheckedInteger,
};
use core::cmp::Ordering;

fn compare<const N: usize>(left: &[u64; N], right: &[u64; N]) -> Ordering {
    for index in (0..N).rev() {
        let order = left[index].cmp(&right[index]);
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}

fn add<const N: usize>(left: [u64; N], right: [u64; N]) -> ([u64; N], bool) {
    let mut output = [0; N];
    let mut carry = false;
    for index in 0..N {
        let (sum, first) = left[index].overflowing_add(right[index]);
        let (sum, second) = sum.overflowing_add(u64::from(carry));
        output[index] = sum;
        carry = first || second;
    }
    (output, carry)
}

fn subtract<const N: usize>(left: [u64; N], right: [u64; N]) -> ([u64; N], bool) {
    let mut output = [0; N];
    let mut borrow = false;
    for index in 0..N {
        let (difference, first) = left[index].overflowing_sub(right[index]);
        let (difference, second) = difference.overflowing_sub(u64::from(borrow));
        output[index] = difference;
        borrow = first || second;
    }
    (output, borrow)
}

fn negate<const N: usize>(value: [u64; N]) -> [u64; N] {
    subtract([0; N], value).0
}

// A limb product plus a destination word plus carry is at most 2^128-1.
// The fixed 16-word temporary accommodates the largest admitted carrier (512 bits).
#[allow(clippy::cast_possible_truncation)] // Low limbs intentionally retain the low 64 bits.
fn multiply<const N: usize>(left: [u64; N], right: [u64; N]) -> ([u64; N], bool) {
    let mut full = [0_u64; 16];
    for (left_index, left_word) in left.iter().enumerate() {
        let mut carry = 0_u128;
        for (right_index, right_word) in right.iter().enumerate() {
            let index = left_index + right_index;
            let product =
                u128::from(*left_word) * u128::from(*right_word) + u128::from(full[index]) + carry;
            full[index] = product as u64;
            carry = product >> 64;
        }
        full[left_index + N] = carry as u64;
    }
    let mut output = [0; N];
    output.copy_from_slice(&full[..N]);
    (output, full[N..2 * N].iter().any(|word| *word != 0))
}

fn divide<const N: usize>(
    dividend: [u64; N],
    divisor: [u64; N],
) -> Result<([u64; N], [u64; N]), PcuExecutionFaultKind> {
    if divisor == [0; N] {
        return Err(PcuExecutionFaultKind::DivideByZero);
    }
    let mut quotient = [0; N];
    let mut remainder = [0; N];
    for bit in (0..N * 64).rev() {
        let overflow = remainder[N - 1] >> 63 != 0;
        let mut carry = (dividend[bit / 64] >> (bit % 64)) & 1;
        for word in &mut remainder {
            let next = *word >> 63;
            *word = (*word << 1) | carry;
            carry = next;
        }
        if overflow || compare(&remainder, &divisor) != Ordering::Less {
            // On carry, the implicit high bit makes this subtraction valid;
            // modular low-word subtraction recovers the exact bounded remainder.
            remainder = subtract(remainder, divisor).0;
            quotient[bit / 64] |= 1 << (bit % 64);
        }
    }
    Ok((quotient, remainder))
}

macro_rules! carrier {
    ($name:ident, $kind:ident, $n:literal, $bytes:literal) => {
        #[doc = concat!("Padding-free ", stringify!($kind), " carrier with little-endian limbs.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name([u64; $n]);
        impl $name {
            #[must_use]
            pub const fn from_limbs_le(limbs: [u64; $n]) -> Self {
                Self(limbs)
            }
            #[must_use]
            pub const fn to_limbs_le(self) -> [u64; $n] {
                self.0
            }
        }
        impl super::sealed::Sealed for $name {}
        impl PcuScalar for $name {
            const TYPE: PcuScalarType = PcuScalarType::$kind;
            const HOST_SIZE: usize = core::mem::size_of::<Self>();
            const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
            const ENCODED_SIZE: usize = $bytes;
            type Encoded = [u8; $bytes];
            fn encode_le(self) -> Self::Encoded {
                let mut output = [0; $bytes];
                for (index, word) in self.0.iter().enumerate() {
                    output[index * 8..index * 8 + 8].copy_from_slice(&word.to_le_bytes());
                }
                output
            }
            fn decode_le(bytes: Self::Encoded) -> Self {
                let mut limbs = [0; $n];
                for (index, word) in limbs.iter_mut().enumerate() {
                    let mut encoded = [0; 8];
                    encoded.copy_from_slice(&bytes[index * 8..index * 8 + 8]);
                    *word = u64::from_le_bytes(encoded);
                }
                Self(limbs)
            }
        }
    };
}
carrier!(PcuU256, U256, 4, 32);
carrier!(PcuI256, I256, 4, 32);
carrier!(PcuU512, U512, 8, 64);
carrier!(PcuI512, I512, 8, 64);
carrier!(PcuF128Bits, F128, 2, 16);
carrier!(PcuF256Bits, F256, 4, 32);

macro_rules! unsigned {
    ($name:ident, $n:literal) => {
        impl $name {
            pub const ZERO: Self = Self([0; $n]);
            pub const MAX: Self = Self([u64::MAX; $n]);
            /// Computes an exact quotient/remainder, rejecting a zero divisor.
            /// # Errors
            /// Returns `DivideByZero` without executing invalid arithmetic.
            pub fn checked_div_rem(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind> {
                divide(self.0, rhs.0).map(|(q, r)| (Self(q), Self(r)))
            }
        }
        impl crate::scalar_checked::sealed::Sealed for $name {}
        impl PcuCheckedInteger for $name {
            fn pcu_checked_add(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                let (value, overflow) = add(self.0, rhs.0);
                if overflow {
                    Err(PcuExecutionFaultKind::ArithmeticOverflow)
                } else {
                    Ok(Self(value))
                }
            }
            fn pcu_checked_sub(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                let (value, underflow) = subtract(self.0, rhs.0);
                if underflow {
                    Err(PcuExecutionFaultKind::ArithmeticUnderflow)
                } else {
                    Ok(Self(value))
                }
            }
            fn pcu_checked_mul(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                let (value, overflow) = multiply(self.0, rhs.0);
                if overflow {
                    Err(PcuExecutionFaultKind::ArithmeticOverflow)
                } else {
                    Ok(Self(value))
                }
            }
        }
    };
}
unsigned!(PcuU256, 4);
unsigned!(PcuU512, 8);

macro_rules! signed {
    ($name:ident, $n:literal) => {
        impl $name {
            pub const ZERO: Self = Self([0; $n]);
            pub const MIN: Self = {
                let mut words = [0; $n];
                words[$n - 1] = 1 << 63;
                Self(words)
            };
            pub const MAX: Self = {
                let mut words = [u64::MAX; $n];
                words[$n - 1] >>= 1;
                Self(words)
            };
            const fn negative(self) -> bool {
                self.0[$n - 1] >> 63 != 0
            }
            /// Truncates toward zero; remainder follows the dividend's sign.
            /// # Errors
            /// Rejects zero and MIN / -1 (also for remainder).
            pub fn checked_div_rem(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind> {
                if self == Self::MIN && rhs.0 == [u64::MAX; $n] {
                    return Err(PcuExecutionFaultKind::SignedDivisionOverflow);
                }
                let left = if self.negative() {
                    negate(self.0)
                } else {
                    self.0
                };
                let right = if rhs.negative() { negate(rhs.0) } else { rhs.0 };
                let (q, r) = divide(left, right)?;
                Ok((
                    Self(if self.negative() != rhs.negative() {
                        negate(q)
                    } else {
                        q
                    }),
                    Self(if self.negative() { negate(r) } else { r }),
                ))
            }
        }
        impl crate::scalar_checked::sealed::Sealed for $name {}
        impl PcuCheckedInteger for $name {
            fn pcu_checked_add(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                let result = Self(add(self.0, rhs.0).0);
                if self.negative() == rhs.negative() && result.negative() != self.negative() {
                    Err(if self.negative() {
                        PcuExecutionFaultKind::ArithmeticUnderflow
                    } else {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    })
                } else {
                    Ok(result)
                }
            }
            fn pcu_checked_sub(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                let result = Self(subtract(self.0, rhs.0).0);
                if self.negative() != rhs.negative() && result.negative() != self.negative() {
                    Err(if self.negative() {
                        PcuExecutionFaultKind::ArithmeticUnderflow
                    } else {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    })
                } else {
                    Ok(result)
                }
            }
            fn pcu_checked_mul(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                let negative = self.negative() != rhs.negative();
                let left = if self.negative() {
                    negate(self.0)
                } else {
                    self.0
                };
                let right = if rhs.negative() { negate(rhs.0) } else { rhs.0 };
                let (magnitude, overflow) = multiply(left, right);
                let limit = if negative { Self::MIN.0 } else { Self::MAX.0 };
                if overflow || compare(&magnitude, &limit) == Ordering::Greater {
                    Err(if negative {
                        PcuExecutionFaultKind::ArithmeticUnderflow
                    } else {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    })
                } else {
                    Ok(Self(if negative {
                        negate(magnitude)
                    } else {
                        magnitude
                    }))
                }
            }
        }
    };
}
signed!(PcuI256, 4);
signed!(PcuI512, 8);

#[cfg(test)]
mod tests;

// All integer policies share one canonical carrier; permission changes arithmetic only.
// Wrapping is modulo 2^width, including signed two's-complement representations.
macro_rules! integer_policies {
    ($name:ident, $minimum:ident) => {
        impl crate::scalar_wrapping::sealed::Sealed for $name {}
        impl crate::PcuWrappingInteger for $name {
            fn wrapping_add(self, rhs: Self) -> Self {
                Self(add(self.0, rhs.0).0)
            }
            fn wrapping_sub(self, rhs: Self) -> Self {
                Self(subtract(self.0, rhs.0).0)
            }
            fn wrapping_mul(self, rhs: Self) -> Self {
                Self(multiply(self.0, rhs.0).0)
            }
        }
        impl crate::scalar_clamped::sealed::Sealed for $name {}
        impl crate::PcuClampedInteger for $name {
            fn pcu_clamped_add(self, rhs: Self) -> Result<Self, crate::PcuClampedFault<Self>> {
                self.pcu_checked_add(rhs).map_err(|kind| {
                    crate::PcuClampedFault::new(
                        kind,
                        if kind == PcuExecutionFaultKind::ArithmeticUnderflow {
                            Self::$minimum
                        } else {
                            Self::MAX
                        },
                    )
                })
            }
            fn pcu_clamped_sub(self, rhs: Self) -> Result<Self, crate::PcuClampedFault<Self>> {
                self.pcu_checked_sub(rhs).map_err(|kind| {
                    crate::PcuClampedFault::new(
                        kind,
                        if kind == PcuExecutionFaultKind::ArithmeticUnderflow {
                            Self::$minimum
                        } else {
                            Self::MAX
                        },
                    )
                })
            }
            fn pcu_clamped_mul(self, rhs: Self) -> Result<Self, crate::PcuClampedFault<Self>> {
                self.pcu_checked_mul(rhs).map_err(|kind| {
                    crate::PcuClampedFault::new(
                        kind,
                        if kind == PcuExecutionFaultKind::ArithmeticUnderflow {
                            Self::$minimum
                        } else {
                            Self::MAX
                        },
                    )
                })
            }
        }
    };
}
integer_policies!(PcuU256, ZERO);
integer_policies!(PcuU512, ZERO);
integer_policies!(PcuI256, MIN);
integer_policies!(PcuI512, MIN);
