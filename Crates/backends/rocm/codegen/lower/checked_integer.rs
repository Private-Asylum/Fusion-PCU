//! HIP source emission for checked exact-width integer Add/Sub/Mul.

use std::fmt::Write as _;

#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchValueId,
    PcuScalarType,
};
use fusion_pcu::model::PcuDispatchIntegerBinaryOp;

use super::RocmLowerError;

#[path = "checked_integer/wide/wide.rs"]
mod wide;
pub(super) use wide::emit_helpers;
pub(super) use wide::emit_div_rem;

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keep width-specific fault guards together.
pub(super) fn emit_checked_integer_binary(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    scalar: PcuScalarType,
    op: PcuDispatchIntegerBinaryOp,
    range_policy: fusion_pcu::PcuRangePolicy,
    result: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    if matches!(
        scalar,
        PcuScalarType::I128
            | PcuScalarType::U128
            | PcuScalarType::I256
            | PcuScalarType::U256
            | PcuScalarType::I512
            | PcuScalarType::U512
    ) {
        return wide::emit(
            source,
            indent,
            logical_index,
            scalar,
            op,
            range_policy,
            result,
            lhs,
            rhs,
        );
    }
    let (cpp_type, signed, bits) = match scalar {
        PcuScalarType::U8 => ("unsigned char", false, 8),
        PcuScalarType::I8 => ("signed char", true, 8),
        PcuScalarType::U16 => ("unsigned short", false, 16),
        PcuScalarType::I16 => ("short", true, 16),
        PcuScalarType::U32 => ("unsigned int", false, 32),
        PcuScalarType::I32 => ("int", true, 32),
        PcuScalarType::U64 => ("unsigned long long", false, 64),
        PcuScalarType::I64 => ("long long", true, 64),
        _ => return Err(RocmLowerError::UnsupportedKernelInterface),
    };
    let clamp = range_policy == fusion_pcu::PcuRangePolicy::Clamp;
    let fault = format!("fusion_integer_fault_{}", result.0);
    if clamp {
        writeln!(source, "{indent}unsigned {fault} = 0u;")
            .map_err(|_| RocmLowerError::FormattingFailure)?;
    }
    let lhs_name = format!("fusion_checked_lhs_{}", result.0);
    let rhs_name = format!("fusion_checked_rhs_{}", result.0);
    writeln!(source, "{indent}const {cpp_type} {lhs_name} = v{};", lhs.0)
        .and_then(|()| writeln!(source, "{indent}const {cpp_type} {rhs_name} = v{};", rhs.0))
        .map_err(|_| RocmLowerError::FormattingFailure)?;

    let overflow_tag = if clamp {
        format!("{fault} = 3u;")
    } else {
        format!(
            "atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 3u) | 3ull); return;"
        )
    };
    let underflow_tag = if clamp {
        format!("{fault} = 4u;")
    } else {
        format!(
            "atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 3u) | 4ull); return;"
        )
    };
    if bits == 64 {
        if signed {
            let guard = match op {
                PcuDispatchIntegerBinaryOp::Add => format!(
                    "if ({rhs_name} > 0ll && {lhs_name} > 9223372036854775807ll - {rhs_name}) {{ {overflow_tag} }} else if ({rhs_name} < 0ll && {lhs_name} < (-9223372036854775807ll - 1ll) - {rhs_name}) {{ {underflow_tag} }}"
                ),
                PcuDispatchIntegerBinaryOp::Sub => format!(
                    "if ({rhs_name} < 0ll && {lhs_name} > 9223372036854775807ll + {rhs_name}) {{ {overflow_tag} }} else if ({rhs_name} > 0ll && {lhs_name} < (-9223372036854775807ll - 1ll) + {rhs_name}) {{ {underflow_tag} }}"
                ),
                PcuDispatchIntegerBinaryOp::Mul => format!(
                    "if ({lhs_name} > 0ll) {{ if ({rhs_name} > 0ll && {lhs_name} > 9223372036854775807ll / {rhs_name}) {{ {overflow_tag} }} else if ({rhs_name} < 0ll && {rhs_name} < (-9223372036854775807ll - 1ll) / {lhs_name}) {{ {underflow_tag} }} }} else if ({lhs_name} < 0ll) {{ if ({rhs_name} > 0ll && {lhs_name} < (-9223372036854775807ll - 1ll) / {rhs_name}) {{ {underflow_tag} }} else if ({rhs_name} < 0ll && {lhs_name} < 9223372036854775807ll / {rhs_name}) {{ {overflow_tag} }} }}"
                ),
            };
            writeln!(source, "{indent}{guard}").map_err(|_| RocmLowerError::FormattingFailure)?;
        } else {
            let guard = match op {
                PcuDispatchIntegerBinaryOp::Add => format!(
                    "if ({lhs_name} > 18446744073709551615ull - {rhs_name}) {{ {overflow_tag} }}"
                ),
                PcuDispatchIntegerBinaryOp::Sub => {
                    format!("if ({lhs_name} < {rhs_name}) {{ {underflow_tag} }}")
                }
                PcuDispatchIntegerBinaryOp::Mul => format!(
                    "if ({rhs_name} != 0ull && {lhs_name} > 18446744073709551615ull / {rhs_name}) {{ {overflow_tag} }}"
                ),
            };
            writeln!(source, "{indent}{guard}").map_err(|_| RocmLowerError::FormattingFailure)?;
        }
    } else if signed {
        let wide_op = match op {
            PcuDispatchIntegerBinaryOp::Add => "+",
            PcuDispatchIntegerBinaryOp::Sub => "-",
            PcuDispatchIntegerBinaryOp::Mul => "*",
        };
        let min = -(1_i64 << (bits - 1));
        let max = (1_i64 << (bits - 1)) - 1;
        let wide = format!("fusion_checked_wide_{}", result.0);
        writeln!(
            source,
            "{indent}const long long {wide} = static_cast<long long>({lhs_name}) {wide_op} static_cast<long long>({rhs_name});"
        )
        .and_then(|()| {
            writeln!(source, "{indent}if ({wide} < ({min}ll)) {{ {underflow_tag} }} else if ({wide} > {max}ll) {{ {overflow_tag} }}")
        })
        .map_err(|_| RocmLowerError::FormattingFailure)?;
    } else {
        let wide_op = match op {
            PcuDispatchIntegerBinaryOp::Add => "+",
            PcuDispatchIntegerBinaryOp::Sub => "-",
            PcuDispatchIntegerBinaryOp::Mul => "*",
        };
        let wide = format!("fusion_checked_wide_{}", result.0);
        let max = (1_u64 << bits) - 1;
        if op == PcuDispatchIntegerBinaryOp::Sub {
            writeln!(
                source,
                "{indent}if ({lhs_name} < {rhs_name}) {{ {underflow_tag} }}"
            )
            .map_err(|_| RocmLowerError::FormattingFailure)?;
        } else {
            writeln!(
                source,
                "{indent}const unsigned long long {wide} = static_cast<unsigned long long>({lhs_name}) {wide_op} static_cast<unsigned long long>({rhs_name});"
            )
            .and_then(|()| {
                writeln!(source, "{indent}if ({wide} > {max}ull) {{ {overflow_tag} }}")
            })
            .map_err(|_| RocmLowerError::FormattingFailure)?;
        }
    }

    let operator = match op {
        PcuDispatchIntegerBinaryOp::Add => "+",
        PcuDispatchIntegerBinaryOp::Sub => "-",
        PcuDispatchIntegerBinaryOp::Mul => "*",
    };
    if clamp {
        // Conditional evaluation is essential: an overflowing signed expression is
        // never evaluated, even though the range fault is recovered.
        let (min, max) = if signed {
            if bits == 64 {
                (
                    "(-9223372036854775807ll - 1ll)".to_owned(),
                    "9223372036854775807ll".to_owned(),
                )
            } else {
                (
                    format!("({}ll)", -(1_i64 << (bits - 1))),
                    format!("{}ll", (1_i64 << (bits - 1)) - 1),
                )
            }
        } else if bits == 64 {
            ("0ull".to_owned(), "18446744073709551615ull".to_owned())
        } else {
            ("0ull".to_owned(), format!("{}ull", (1_u64 << bits) - 1))
        };
        writeln!(source,"{indent}{cpp_type} v{} = {fault} == 0u ? static_cast<{cpp_type}>({lhs_name} {operator} {rhs_name}) : static_cast<{cpp_type}>({fault} == 3u ? {max} : {min});",result.0)
            .and_then(|()|writeln!(source,"{indent}if ({fault} != 0u && !fusion_range_fault_recorded) {{ atomicMin(fusion_fault_word, 0x8000000000000000ull | (static_cast<unsigned long long>({logical_index}) << 3u) | {fault}); fusion_range_fault_recorded = true; }}"))
            .map_err(|_| RocmLowerError::FormattingFailure)
    } else {
        writeln!(
            source,
            "{indent}{cpp_type} v{} = static_cast<{cpp_type}>({lhs_name} {operator} {rhs_name});",
            result.0
        )
        .map_err(|_| RocmLowerError::FormattingFailure)
    }
}

#[cfg(test)]
#[path = "checked_integer/tests.rs"]
mod tests;
