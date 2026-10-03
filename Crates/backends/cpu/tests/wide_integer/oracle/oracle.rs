//! Independent base-256 signed-magnitude arithmetic; no core checked arithmetic is called.
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedInteger,
    PcuDispatchIntegerBinaryOp,
    PcuExecutionFaultKind,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
pub trait Wide: PcuCheckedInteger + core::fmt::Debug + Eq {
    const BYTES: usize;
    const SIGNED: bool;
    fn bytes(self) -> [u8; 64];
    fn from_bytes(bytes: [u8; 64]) -> Self;
}
macro_rules! primitive {
    ($ty:ty, $signed:expr, $n:expr) => {
        impl Wide for $ty {
            const BYTES: usize = $n;
            const SIGNED: bool = $signed;
            fn bytes(self) -> [u8; 64] {
                let mut bytes = [0; 64];
                bytes[..$n].copy_from_slice(&self.to_le_bytes());
                bytes
            }
            fn from_bytes(bytes: [u8; 64]) -> Self {
                let mut value = [0; $n];
                value.copy_from_slice(&bytes[..$n]);
                Self::from_le_bytes(value)
            }
        }
    };
}
primitive!(i8, true, 1);
primitive!(u8, false, 1);
primitive!(i16, true, 2);
primitive!(u16, false, 2);
primitive!(i32, true, 4);
primitive!(u32, false, 4);
primitive!(i64, true, 8);
primitive!(u64, false, 8);
primitive!(i128, true, 16);
primitive!(u128, false, 16);
macro_rules! carrier {
    ($ty:ty, $n:expr, $signed:expr) => {
        impl Wide for $ty {
            const BYTES: usize = $n * 8;
            const SIGNED: bool = $signed;
            fn bytes(self) -> [u8; 64] {
                let mut bytes = [0; 64];
                for (i, limb) in self.to_limbs_le().into_iter().enumerate() {
                    bytes[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
                }
                bytes
            }
            fn from_bytes(bytes: [u8; 64]) -> Self {
                let mut limbs = [0; $n];
                for (i, limb) in limbs.iter_mut().enumerate() {
                    let mut encoded = [0; 8];
                    encoded.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
                    *limb = u64::from_le_bytes(encoded);
                }
                Self::from_limbs_le(limbs)
            }
        }
    };
}
carrier!(PcuI256, 4, true);
carrier!(PcuU256, 4, false);
carrier!(PcuI512, 8, true);
carrier!(PcuU512, 8, false);
fn complement(bytes: &mut [u8]) {
    let mut carry = true;
    for byte in bytes {
        *byte = !*byte;
        if carry {
            let (value, overflow) = byte.overflowing_add(1);
            *byte = value;
            carry = overflow;
        }
    }
}
fn magnitude<T: Wide>(value: T) -> (bool, [u8; 128]) {
    let mut bytes = value.bytes();
    let negative = T::SIGNED && bytes[T::BYTES - 1] >> 7 != 0;
    if negative {
        complement(&mut bytes[..T::BYTES]);
    }
    let mut magnitude = [0; 128];
    magnitude[..T::BYTES].copy_from_slice(&bytes[..T::BYTES]);
    (negative, magnitude)
}
fn compare(left: &[u8; 128], right: &[u8; 128]) -> core::cmp::Ordering {
    for (a, b) in left.iter().zip(right).rev() {
        let order = a.cmp(b);
        if order != core::cmp::Ordering::Equal {
            return order;
        }
    }
    core::cmp::Ordering::Equal
}
fn add(left: [u8; 128], right: [u8; 128]) -> [u8; 128] {
    let mut result = [0; 128];
    let mut carry = 0_u16;
    for i in 0..128 {
        let sum = u16::from(left[i]) + u16::from(right[i]) + carry;
        result[i] = u8::try_from(sum & 255).unwrap();
        carry = sum >> 8;
    }
    assert_eq!(carry, 0);
    result
}
fn sub(left: [u8; 128], right: [u8; 128]) -> [u8; 128] {
    let mut result = [0; 128];
    let mut borrow = 0_i16;
    for i in 0..128 {
        let difference = i16::from(left[i]) - i16::from(right[i]) - borrow;
        result[i] = u8::try_from(difference.rem_euclid(256)).unwrap();
        borrow = i16::from(difference < 0);
    }
    assert_eq!(borrow, 0);
    result
}
fn mul(left: [u8; 128], right: [u8; 128], bytes: usize) -> [u8; 128] {
    let mut result = [0; 128];
    // Each byte product + output + carry <= 255²+255+255=65535. No u16 wrap.
    // Two <=64-byte magnitudes fit the complete fixed 128-byte product without narrowing.
    for i in 0..bytes {
        let mut carry = 0_u16;
        for j in 0..bytes {
            let product =
                u16::from(left[i]) * u16::from(right[j]) + u16::from(result[i + j]) + carry;
            result[i + j] = u8::try_from(product & 255).unwrap();
            carry = product >> 8;
        }
        result[i + bytes] = u8::try_from(carry).unwrap();
    }
    result
}
pub fn evaluate<T: Wide>(
    left: T,
    right: T,
    op: PcuDispatchIntegerBinaryOp,
) -> Result<T, PcuExecutionFaultKind> {
    let (left_negative, left) = magnitude(left);
    let (mut right_negative, right) = magnitude(right);
    if op == PcuDispatchIntegerBinaryOp::Sub {
        right_negative = !right_negative;
    }
    let (mut negative, result) = if op == PcuDispatchIntegerBinaryOp::Mul {
        (left_negative != right_negative, mul(left, right, T::BYTES))
    } else if left_negative == right_negative {
        (left_negative, add(left, right))
    } else if compare(&left, &right) == core::cmp::Ordering::Less {
        (right_negative, sub(right, left))
    } else {
        (left_negative, sub(left, right))
    };
    if result == [0; 128] {
        negative = false;
    }
    let mut limit = [0; 128];
    if !T::SIGNED && negative {
        return Err(PcuExecutionFaultKind::ArithmeticUnderflow);
    }
    if T::SIGNED && negative {
        limit[T::BYTES - 1] = 128;
    } else {
        limit[..T::BYTES].fill(255);
        if T::SIGNED {
            limit[T::BYTES - 1] = 127;
        }
    }
    if compare(&result, &limit) == core::cmp::Ordering::Greater {
        return Err(if negative {
            PcuExecutionFaultKind::ArithmeticUnderflow
        } else {
            PcuExecutionFaultKind::ArithmeticOverflow
        });
    }
    let mut bytes = [0; 64];
    bytes[..T::BYTES].copy_from_slice(&result[..T::BYTES]);
    if negative {
        complement(&mut bytes[..T::BYTES]);
    }
    Ok(T::from_bytes(bytes))
}
pub fn small<T: Wide>(value: u8) -> T {
    let mut bytes = [0; 64];
    bytes[0] = value;
    T::from_bytes(bytes)
}
pub fn minimum<T: Wide>() -> T {
    let mut bytes = [0; 64];
    if T::SIGNED {
        bytes[T::BYTES - 1] = 128;
    }
    T::from_bytes(bytes)
}
pub fn maximum<T: Wide>() -> T {
    let mut bytes = [255; 64];
    if T::SIGNED {
        bytes[T::BYTES - 1] = 127;
    }
    T::from_bytes(bytes)
}
