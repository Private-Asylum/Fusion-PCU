//! Independent base-256 binary long division; no core division method is called.
#[path = "../../wide_integer/oracle/oracle.rs"]
#[allow(dead_code)] // Carrier byte adapters are shared; binary arithmetic is not used here.
mod carriers;
pub use carriers::{Wide, small};
pub fn minimum<T: Wide>() -> T {
    carriers::minimum()
}
pub fn maximum<T: Wide>() -> T {
    carriers::maximum()
}
use pcu_facade::PcuExecutionFaultKind;
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
fn magnitude<T: Wide>(value: T) -> (bool, [u8; 65]) {
    let mut bytes = value.bytes();
    let negative = T::SIGNED && bytes[T::BYTES - 1] & 128 != 0;
    if negative {
        complement(&mut bytes[..T::BYTES]);
    }
    let mut result = [0; 65];
    result[..T::BYTES].copy_from_slice(&bytes[..T::BYTES]);
    (negative, result)
}
fn compare(left: &[u8; 65], right: &[u8; 65]) -> core::cmp::Ordering {
    for (a, b) in left.iter().zip(right).rev() {
        let order = a.cmp(b);
        if order != core::cmp::Ordering::Equal {
            return order;
        }
    }
    core::cmp::Ordering::Equal
}
fn subtract(left: &mut [u8; 65], right: &[u8; 65]) {
    let mut borrow = 0_i16;
    for (a, b) in left.iter_mut().zip(right) {
        let difference = i16::from(*a) - i16::from(*b) - borrow;
        *a = u8::try_from(difference.rem_euclid(256)).unwrap();
        borrow = i16::from(difference < 0);
    }
    assert_eq!(borrow, 0);
}
fn encode<T: Wide>(magnitude: [u8; 65], negative: bool) -> T {
    assert_eq!(magnitude[64], 0);
    let mut bytes = [0; 64];
    bytes[..T::BYTES].copy_from_slice(&magnitude[..T::BYTES]);
    if negative {
        complement(&mut bytes[..T::BYTES]);
    }
    T::from_bytes(bytes)
}
pub fn domain<T: Wide>(left: T, right: T) -> Result<(), PcuExecutionFaultKind> {
    let lhs = left.bytes();
    let rhs = right.bytes();
    if rhs[..T::BYTES].iter().all(|byte| *byte == 0) {
        return Err(PcuExecutionFaultKind::DivideByZero);
    }
    if T::SIGNED
        && lhs[T::BYTES - 1] == 128
        && lhs[..T::BYTES - 1].iter().all(|byte| *byte == 0)
        && rhs[..T::BYTES].iter().all(|byte| *byte == 255)
    {
        return Err(PcuExecutionFaultKind::SignedDivisionOverflow);
    }
    Ok(())
}
pub fn evaluate<T: Wide>(left: T, right: T) -> Result<(T, T), PcuExecutionFaultKind> {
    let (left_negative, numerator) = magnitude(left);
    let (right_negative, denominator) = magnitude(right);
    if denominator == [0; 65] {
        return Err(PcuExecutionFaultKind::DivideByZero);
    }
    let mut minimum = [0; 65];
    minimum[T::BYTES - 1] = 128;
    let mut one = [0; 65];
    one[0] = 1;
    if T::SIGNED && left_negative && right_negative && numerator == minimum && denominator == one {
        return Err(PcuExecutionFaultKind::SignedDivisionOverflow);
    }
    let mut quotient = [0; 65];
    let mut remainder = [0; 65];
    // Remainder before shift is < denominator <= 2^512. One extra byte holds the
    // shift carry exactly; each byte shift+carry <=511, never overflowing u16.
    for bit in (0..T::BYTES * 8).rev() {
        let mut carry = u16::from((numerator[bit / 8] >> (bit % 8)) & 1);
        for byte in &mut remainder {
            let shifted = u16::from(*byte) * 2 + carry;
            *byte = u8::try_from(shifted & 255).unwrap();
            carry = shifted >> 8;
        }
        assert_eq!(carry, 0);
        if compare(&remainder, &denominator) != core::cmp::Ordering::Less {
            subtract(&mut remainder, &denominator);
            quotient[bit / 8] |= 1 << (bit % 8);
        }
    }
    Ok((
        encode(quotient, left_negative != right_negative),
        encode(remainder, left_negative),
    ))
}
pub fn decode<T: Wide>(text: &str) -> T {
    assert_eq!(text.len(), T::BYTES * 2);
    let mut bytes = [0; 64];
    for (i, byte) in bytes[..T::BYTES].iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).unwrap();
    }
    T::from_bytes(bytes)
}
type Golden<T> = (T, T, Result<(T, T), PcuExecutionFaultKind>);
pub fn goldens<T: Wide>() -> Vec<Golden<T>> {
    include_str!("vectors.txt")
        .lines()
        .filter_map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.is_empty()
                || fields[0].starts_with('#')
                || fields[0].parse::<usize>().unwrap() != T::BYTES
                || fields[1] != if T::SIGNED { "s" } else { "u" }
            {
                return None;
            }
            let result = match fields[4] {
                "zero" => Err(PcuExecutionFaultKind::DivideByZero),
                "overflow" => Err(PcuExecutionFaultKind::SignedDivisionOverflow),
                q => Ok((decode(q), decode(fields[5]))),
            };
            Some((decode(fields[2]), decode(fields[3]), result))
        })
        .collect()
}
