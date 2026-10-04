//! Exact integer literal producers, including every signed minimum bit pattern.

use std::fmt::Write as _;

#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchValueId,
    PcuParameterValue,
};
use super::RocmLowerError;

pub(super) fn emit(
    source: &mut String,
    indent: &str,
    result: PcuDispatchValueId,
    value: PcuParameterValue,
) -> Result<(), RocmLowerError> {
    // Casting a negative literal or an out-of-range unsigned value to signed C++
    // would depend on host/compiler interpretation. Reconstruct the original bits.
    let (cpp_type, unsigned_type, bits) = match value {
        PcuParameterValue::U8(value) => ("unsigned char", "unsigned char", u64::from(value)),
        PcuParameterValue::I8(value) => (
            "signed char",
            "unsigned char",
            u64::from(u8::from_ne_bytes(value.to_ne_bytes())),
        ),
        PcuParameterValue::U16(value) => ("unsigned short", "unsigned short", u64::from(value)),
        PcuParameterValue::I16(value) => (
            "short",
            "unsigned short",
            u64::from(u16::from_ne_bytes(value.to_ne_bytes())),
        ),
        PcuParameterValue::U32(value) => ("unsigned int", "unsigned int", u64::from(value)),
        PcuParameterValue::I32(value) => (
            "int",
            "unsigned int",
            u64::from(u32::from_ne_bytes(value.to_ne_bytes())),
        ),
        PcuParameterValue::U64(value) => ("unsigned long long", "unsigned long long", value),
        PcuParameterValue::I64(value) => (
            "long long",
            "unsigned long long",
            u64::from_ne_bytes(value.to_ne_bytes()),
        ),
        _ => return Err(RocmLowerError::UnsupportedConstant),
    };
    writeln!(
        source,
        "{indent}{cpp_type} v{} = __builtin_bit_cast({cpp_type}, static_cast<{unsigned_type}>(0x{bits:016x}ull));",
        result.0,
    )
    .map_err(|_| RocmLowerError::FormattingFailure)
}
