//! Independent decoded-dyadic/midpoint oracle; core only cross-checks admitted laws.
#[path = "../../checked_low_precision/oracle/oracle.rs"]
#[allow(dead_code)]
mod shared;
pub use shared::Format;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuScalar,
};
pub fn bits<T: PcuScalar>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
pub fn code(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 3,
        PcuExecutionFaultKind::ArithmeticUnderflow => 4,
        PcuExecutionFaultKind::InvalidFloatingOperand => 5,
        _ => panic!("unproved fault"),
    }
}
pub fn payload<T: Format>(lane: usize) -> T {
    T::from(match lane % 3 {
        0 => T::ONE,
        1 => 1,
        _ => T::SIGN,
    })
}
pub fn factor<T: Format>(half: bool) -> T {
    T::from(if half {
        T::ONE - (1 << T::FRACTION)
    } else {
        T::ONE + (1 << T::FRACTION)
    })
}
pub fn small<T: Format>(value: u8) -> T {
    match value {
        0 => T::zero(),
        1 => T::one(),
        2 => factor(false),
        3 => T::from(T::ONE + (1 << (T::FRACTION - 1))),
        _ => T::sentinel(),
    }
}
#[allow(clippy::float_cmp)] // Exact dyadics and halfway comparisons define the contract.
pub fn operation<T: Format>(
    a: T,
    b: T,
    mul: bool,
    policy: PcuFloatUnderflowPolicy,
) -> Result<T, u32> {
    if a.bits() & (T::SIGN - 1) > T::MAX || b.bits() & (T::SIGN - 1) > T::MAX {
        return Err(5);
    }
    let left = shared::unpack::<T>(a.bits());
    let right = shared::unpack::<T>(b.bits());
    let result = if mul { left * right } else { left + right };
    let magnitude = result.abs();
    let maximum = shared::unpack::<T>(T::MAX);
    let previous = shared::unpack::<T>(T::MAX - 1);
    let boundary = maximum + (maximum - previous) / 2.0;
    if magnitude > boundary || (magnitude == boundary && T::MAX & 1 != 0) {
        return Err(3);
    }
    let encoded = shared::independent(a, b, if mul { 2 } else { 0 });
    let min_normal = shared::unpack::<T>(1 << T::FRACTION);
    let tiny_boundary = min_normal - shared::unpack::<T>(1) / 4.0;
    let tiny_inexact =
        result != 0.0 && magnitude < tiny_boundary && result != shared::unpack::<T>(encoded);
    let subnormal = encoded & (T::SIGN - 1) != 0 && encoded & (T::SIGN - 1) < (1 << T::FRACTION);
    if match policy {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => tiny_inexact,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => tiny_inexact || subnormal,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => false,
    } {
        return Err(4);
    }
    Ok(T::from(encoded))
}
pub fn pipeline<T: Format>(
    input: &[T],
    constant: &[T],
    uniform: &[T],
    policy: PcuFloatUnderflowPolicy,
) -> Result<Vec<T>, (u32, u32, bool)> {
    let mut stage = Vec::with_capacity(input.len());
    for (index, (&a, &b)) in input.iter().zip(constant).enumerate() {
        let value = operation(a, b, false, policy);
        assert_eq!(
            value,
            shared::reference(a, b, 0, policy).map_err(|error| code(error.kind()))
        );
        stage.push(value.map_err(|code| (u32::try_from(index).unwrap(), code, false))?);
    }
    let mut output = Vec::with_capacity(input.len());
    for (index, (&a, &b)) in stage.iter().zip(uniform).enumerate() {
        let value = operation(a, b, true, policy);
        assert_eq!(
            value,
            shared::reference(a, b, 2, policy).map_err(|error| code(error.kind()))
        );
        output.push(value.map_err(|code| (u32::try_from(index).unwrap(), code, false))?);
    }
    Ok(output)
}
