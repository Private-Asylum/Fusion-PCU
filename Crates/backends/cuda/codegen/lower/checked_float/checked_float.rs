//! Bit-exact binary32 checked arithmetic used by the CUDA source lowerer.
//!
//! The helper uses fixed-width integer significands throughout. In particular, it never asks
//! the device floating-point unit to perform the operation, so shader FTZ settings cannot
//! silently change either the result or its underflow classification.
//!
//! For the finite binary results implemented here, rounding uses roundTiesToEven (IEEE Std
//! 754-2019 4.3.1 and the binary default in 4.3.3). Packing detects tininess after rounding
//! destination precision and, for `IeeeAfterRounding`, reports underflow only when that final
//! packing is inexact, one of the permitted binary choices in 7.5. `RejectSubnormalResult` is a
//! stricter PCU policy that also faults on exact subnormal results. This is a description of
//! numeric rules, not a claim of full IEEE 754 conformance: PCU rejects non-finite inputs and
//! exposes range conditions as execution faults according to its policies, rather than
//! implementing IEEE default exception flags and handling.

use std::fmt::Write as _;

#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
};
#[rustfmt::skip]
use fusion_pcu::model::{
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
};

use super::CudaLowerError;

pub(super) fn uses_checked_float(kernel: &fusion_pcu::PcuDispatchKernelIr<'_>) -> bool {
    fn contains(ops: &[fusion_pcu::PcuDispatchOp<'_>]) -> bool {
        ops.iter().any(|op| match op {
            fusion_pcu::PcuDispatchOp::Data(
                fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary { .. }
                | fusion_pcu::PcuDispatchDataOp::CheckedFloatUnary { .. }
                | fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert { .. },
            ) => true,
            fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. } => contains(body),
            _ => false,
        })
    }
    contains(kernel.ops)
}

/// Shared integer-only binary helpers for synthesized compound kernels.
#[cfg(feature = "tensor")]
pub(super) fn emit_scalar_helpers(source: &mut String, scalar: fusion_pcu::PcuScalarType) {
    match scalar {
        fusion_pcu::PcuScalarType::F32 => source.push_str(CHECKED_F32_HELPERS),
        fusion_pcu::PcuScalarType::F64 => source.push_str(f64::CHECKED_F64_HELPERS),
        _ => unreachable!("compound checked helpers require an admitted floating scalar"),
    }
}

/// Emits the shared integer-only implementation once per checked-float kernel.
pub(super) fn emit_helpers(source: &mut String, kernel: &fusion_pcu::PcuDispatchKernelIr<'_>) {
    fn inspect(
        ops: &[fusion_pcu::PcuDispatchOp<'_>],
        has_f32: &mut bool,
        has_f64: &mut bool,
        has_narrow: &mut bool,
        has_widen: &mut bool,
    ) {
        for op in ops {
            match op {
                fusion_pcu::PcuDispatchOp::Data(
                    fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary { value_type, .. }
                    | fusion_pcu::PcuDispatchDataOp::CheckedFloatUnary { value_type, .. },
                ) => match value_type {
                    fusion_pcu::PcuValueType::Scalar(fusion_pcu::PcuScalarType::F32) => {
                        *has_f32 = true;
                    }
                    fusion_pcu::PcuValueType::Scalar(fusion_pcu::PcuScalarType::F64) => {
                        *has_f64 = true;
                    }
                    _ => {}
                },
                fusion_pcu::PcuDispatchOp::Data(
                    fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert { conversion, .. },
                ) => match conversion {
                    fusion_pcu::PcuDispatchCheckedFloatConversion::F64ToF32 => *has_narrow = true,
                    fusion_pcu::PcuDispatchCheckedFloatConversion::F32ToF64 => *has_widen = true,
                },
                fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. } => {
                    inspect(body, has_f32, has_f64, has_narrow, has_widen);
                }
                _ => {}
            }
        }
    }
    let mut has_f32 = false;
    let mut has_f64 = false;
    let mut has_narrow = false;
    let mut has_widen = false;
    inspect(
        kernel.ops,
        &mut has_f32,
        &mut has_f64,
        &mut has_narrow,
        &mut has_widen,
    );
    if has_f32 {
        source.push_str(CHECKED_F32_HELPERS);
        source.push_str(CHECKED_F32_RELU_HELPER);
        source.push_str(CHECKED_F32_NEG_HELPER);
    }
    if has_f64 {
        source.push_str(f64::CHECKED_F64_HELPERS);
        source.push_str(CHECKED_F64_RELU_HELPER);
        source.push_str(CHECKED_F64_NEG_HELPER);
    }
    if has_narrow {
        source.push_str(CHECKED_F64_TO_F32_HELPER);
    }
    if has_widen {
        source.push_str(CHECKED_F32_TO_F64_HELPER);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_checked_float_unary(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    value_type: fusion_pcu::PcuValueType,
    result: PcuDispatchValueId,
    value: PcuDispatchValueId,
) -> Result<(), CudaLowerError> {
    let policy_tag = match policy {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => 0u32,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
    };
    let (prefix, upper, ty, bits_type, sign_mask, max_finite) = match value_type {
        fusion_pcu::PcuValueType::Scalar(fusion_pcu::PcuScalarType::F32) => (
            "f32",
            "F32",
            "float",
            "unsigned int",
            "0x80000000u",
            "0x7f7fffffu",
        ),
        fusion_pcu::PcuValueType::Scalar(fusion_pcu::PcuScalarType::F64) => (
            "f64",
            "F64",
            "double",
            "unsigned long long",
            "0x8000000000000000ull",
            "0x7fefffffffffffffull",
        ),
        _ => return Err(CudaLowerError::UnsupportedKernelInterface),
    };
    let operation = match op {
        PcuDispatchFloatUnaryOp::Relu => "relu",
        PcuDispatchFloatUnaryOp::Neg => "neg",
    };
    writeln!(
        source,
        "{indent}const Fusion{upper}CheckedResult fusion_{prefix}_checked_{} = fusion_checked_{prefix}_{operation}(__builtin_bit_cast({bits_type}, v{}), {policy_tag}u);",
        result.0,
        value.0,
    )
    .map_err(|_| CudaLowerError::FormattingFailure)?;
    emit_checked_result(
        source,
        indent,
        logical_index,
        range_policy,
        &format!("fusion_{prefix}_checked_{}", result.0),
        result,
        ty,
        bits_type,
        sign_mask,
        max_finite,
    )
}

#[allow(clippy::too_many_arguments)] // Keep operand and semantic facts explicit at the lowering boundary.
pub(super) fn emit_checked_float_binary(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    value_type: fusion_pcu::PcuValueType,
    result: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), CudaLowerError> {
    let op_tag = match op {
        PcuDispatchFloatBinaryOp::Add => 0u32,
        PcuDispatchFloatBinaryOp::Sub => 1,
        PcuDispatchFloatBinaryOp::Mul => 2,
        PcuDispatchFloatBinaryOp::Div => 3,
    };
    let policy_tag = match policy {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => 0u32,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
    };
    let (prefix, ty, cast) = match value_type {
        fusion_pcu::PcuValueType::Scalar(fusion_pcu::PcuScalarType::F32) => {
            ("f32", "float", ("unsigned int", "unsigned int"))
        }
        fusion_pcu::PcuValueType::Scalar(fusion_pcu::PcuScalarType::F64) => (
            "f64",
            "double",
            ("unsigned long long", "unsigned long long"),
        ),
        _ => return Err(CudaLowerError::UnsupportedKernelInterface),
    };
    writeln!(source,
        "{indent}const Fusion{upper}CheckedResult fusion_{prefix}_checked_{} = fusion_checked_{prefix}_binary(__builtin_bit_cast({lhs_cast}, v{}), __builtin_bit_cast({rhs_cast}, v{}), {op_tag}u, {policy_tag}u);",
        result.0, lhs.0, rhs.0,
        upper = if prefix == "f32" { "F32" } else { "F64" },
        lhs_cast = cast.0,
        rhs_cast = cast.1,
    ).map_err(|_| CudaLowerError::FormattingFailure)?;
    emit_checked_result(
        source,
        indent,
        logical_index,
        range_policy,
        &format!("fusion_{prefix}_checked_{}", result.0),
        result,
        ty,
        cast.0,
        if prefix == "f32" {
            "0x80000000u"
        } else {
            "0x8000000000000000ull"
        },
        if prefix == "f32" {
            "0x7f7fffffu"
        } else {
            "0x7fefffffffffffffull"
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_checked_float_convert(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    conversion: fusion_pcu::PcuDispatchCheckedFloatConversion,
    policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    result: PcuDispatchValueId,
    value: PcuDispatchValueId,
) -> Result<(), CudaLowerError> {
    match conversion {
        fusion_pcu::PcuDispatchCheckedFloatConversion::F64ToF32 => emit_checked_f64_to_f32(
            source,
            indent,
            logical_index,
            policy,
            range_policy,
            result,
            value,
        ),
        fusion_pcu::PcuDispatchCheckedFloatConversion::F32ToF64 => emit_checked_f32_to_f64(
            source,
            indent,
            logical_index,
            policy,
            range_policy,
            result,
            value,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_checked_f64_to_f32(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    result: PcuDispatchValueId,
    value: PcuDispatchValueId,
) -> Result<(), CudaLowerError> {
    let policy_tag = match policy {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => 0u32,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => 1u32,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2u32,
    };
    writeln!(source,
        "{indent}const FusionF64ToF32Result fusion_f64_to_f32_checked_{} = fusion_checked_f64_to_f32(__builtin_bit_cast(unsigned long long, v{}), {policy_tag}u);",
        result.0, value.0
    ).map_err(|_| CudaLowerError::FormattingFailure)?;
    emit_checked_result(
        source,
        indent,
        logical_index,
        range_policy,
        &format!("fusion_f64_to_f32_checked_{}", result.0),
        result,
        "float",
        "unsigned int",
        "0x80000000u",
        "0x7f7fffffu",
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_checked_f32_to_f64(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    _policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    result: PcuDispatchValueId,
    value: PcuDispatchValueId,
) -> Result<(), CudaLowerError> {
    writeln!(source,
        "{indent}const FusionF32ToF64Result fusion_f32_to_f64_checked_{} = fusion_checked_f32_to_f64(__builtin_bit_cast(unsigned int, v{}));",
        result.0, value.0
    ).map_err(|_| CudaLowerError::FormattingFailure)?;
    emit_checked_result(
        source,
        indent,
        logical_index,
        range_policy,
        &format!("fusion_f32_to_f64_checked_{}", result.0),
        result,
        "double",
        "unsigned long long",
        "0x8000000000000000ull",
        "0x7fefffffffffffffull",
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_checked_result(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    range_policy: PcuRangePolicy,
    checked_name: &str,
    result: PcuDispatchValueId,
    value_type: &str,
    bits_type: &str,
    sign_mask: &str,
    max_finite: &str,
) -> Result<(), CudaLowerError> {
    if range_policy == PcuRangePolicy::Clamp {
        writeln!(source,
            "{indent}if ({checked_name}.fault != 0u) {{\n{indent}    if ({checked_name}.fault == 3u || {checked_name}.fault == 4u) {{\n{indent}        if (!fusion_range_fault_recorded) {{ atomicMin(fusion_fault_word, 0x8000000000000000ull | (static_cast<unsigned long long>({logical_index}) << 3u) | {checked_name}.fault); fusion_range_fault_recorded = true; }}\n{indent}    }} else {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 3u) | {checked_name}.fault); return; }}\n{indent}}}"
        ).and_then(|()| writeln!(source,
            "{indent}{bits_type} fusion_recovered_bits_{} = {checked_name}.bits;\n{indent}if ({checked_name}.fault == 3u) fusion_recovered_bits_{} = (fusion_recovered_bits_{} & {sign_mask}) | {max_finite};\n{indent}{value_type} v{} = __builtin_bit_cast({value_type}, fusion_recovered_bits_{});",
            result.0, result.0, result.0, result.0, result.0
        )).map_err(|_| CudaLowerError::FormattingFailure)
    } else {
        writeln!(source,
            "{indent}if ({checked_name}.fault != 0u) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 3u) | {checked_name}.fault); return; }}\n{indent}{value_type} v{} = __builtin_bit_cast({value_type}, {checked_name}.bits);",
            result.0
        ).map_err(|_| CudaLowerError::FormattingFailure)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[path = "f64.rs"]
mod f64;

// IEEE Std 754-2019 7.4 defines overflow against the rounded result with unbounded exponent
// range; 7.5 permits tininess detection after rounding. `unbounded_exp` below is the result
// rounded to destination precision before exponent-range packing, while `inexact` is measured
// from the final packing. Policy 0 faults for tiny-and-inexact results; policy 1 additionally
// faults for exact subnormals; policy 2 permits gradual underflow. These are PCU fault policies,
// not IEEE default exception handling.
const CHECKED_F32_HELPERS: &str = r"
struct FusionF32CheckedResult { unsigned int bits; unsigned int fault; };
__device__ __forceinline__ unsigned long long fusion_f32_shift_jam(unsigned long long x, int distance) {
    if (distance <= 0) return x;
    if (distance < 64) return (x >> distance) | static_cast<unsigned long long>((x & ((1ull << distance) - 1ull)) != 0ull);
    return static_cast<unsigned long long>(x != 0ull);
}
__device__ __forceinline__ bool fusion_f32_round_increment(unsigned long long sig, unsigned long long discarded) {
    return discarded > 4ull || (discarded == 4ull && (sig & 1ull) != 0ull);
}
__device__ __forceinline__ void fusion_f32_normalize(unsigned long long* ext, int* exponent) {
    while (*ext >= (1ull << 27)) { *ext = fusion_f32_shift_jam(*ext, 1); ++*exponent; }
    while (*ext < (1ull << 26)) { *ext <<= 1; --*exponent; }
}
__device__ __forceinline__ int fusion_f32_round_precision(unsigned long long ext, int exponent) {
    fusion_f32_normalize(&ext, &exponent);
    unsigned long long discarded = ext & 7ull;
    unsigned long long sig = ext >> 3;
    if (fusion_f32_round_increment(sig, discarded)) ++sig;
    if (sig == (1ull << 24)) ++exponent;
    return exponent;
}
__device__ __forceinline__ FusionF32CheckedResult fusion_f32_pack(unsigned int sign, unsigned long long ext, int exponent, unsigned int policy) {
    fusion_f32_normalize(&ext, &exponent);
    int unbounded_exp = fusion_f32_round_precision(ext, exponent);
    bool tiny = unbounded_exp < -126;
    if (exponent < -126) {
        ext = fusion_f32_shift_jam(ext, -126 - exponent);
        exponent = -126;
    }
    unsigned long long discarded = ext & 7ull;
    unsigned long long sig = ext >> 3;
    bool inexact = discarded != 0ull;
    if (fusion_f32_round_increment(sig, discarded)) ++sig;
    unsigned int bits;
    bool subnormal;
    if (exponent == -126 && sig < 0x800000ull) {
        bits = (sign << 31) | static_cast<unsigned int>(sig);
        subnormal = sig != 0ull;
    } else {
        if (sig >= (1ull << 24)) { sig >>= 1; ++exponent; }
        if (exponent > 127) return {(sign << 31) | 0x7f800000u, 3u};
        bits = (sign << 31) | (static_cast<unsigned int>(exponent + 127) << 23) | (static_cast<unsigned int>(sig) & 0x7fffffu);
        subnormal = false;
    }
    if ((bits & 0x7fffffffu) == 0u) inexact = discarded != 0ull;
    if ((policy == 0u && tiny && inexact) || (policy == 1u && (subnormal || (tiny && inexact)))) return {bits, 4u};
    return {bits, 0u};
}
__device__ __forceinline__ FusionF32CheckedResult fusion_checked_f32_binary(unsigned int x, unsigned int y, unsigned int op, unsigned int policy) {
    unsigned int xe = (x >> 23) & 0xffu, ye = (y >> 23) & 0xffu;
    if (xe == 0xffu || ye == 0xffu) return {0u, 5u};
    unsigned int xs = x >> 31, ys = y >> 31;
    unsigned int xm = x & 0x7fffffu, ym = y & 0x7fffffu;
    int xexp = -126, yexp = -126;
    if (xe != 0u) { xm |= 0x800000u; xexp = static_cast<int>(xe) - 127; }
    if (ye != 0u) { ym |= 0x800000u; yexp = static_cast<int>(ye) - 127; }
    if (op == 2u) {
        unsigned long long product = static_cast<unsigned long long>(xm) * static_cast<unsigned long long>(ym);
        if (product == 0ull) return {(xs ^ ys) << 31, 0u};
        int top = 63 - __clzll(product);
        int exponent = xexp + yexp + top - 46;
        unsigned long long ext = top > 26 ? fusion_f32_shift_jam(product, top - 26) : product << (26 - top);
        return fusion_f32_pack(xs ^ ys, ext, exponent, policy);
    }
    if (op == 3u) {
        unsigned int sign = xs ^ ys;
        if ((y << 1) == 0u) return {0u, 1u};
        if ((x << 1) == 0u) return {sign << 31, 0u};
        int a_shift = __clz(xm) - 8;
        int b_shift = __clz(ym) - 8;
        xm <<= a_shift; xexp -= a_shift;
        ym <<= b_shift; yexp -= b_shift;
        unsigned long long numerator = xm;
        unsigned long long denominator = ym;
        int exponent = xexp - yexp;
        if (numerator < denominator) { numerator <<= 1; --exponent; }
        unsigned long long remainder = numerator - denominator;
        unsigned long long ext = 1ull;
        for (int i = 0; i < 26; ++i) {
            remainder <<= 1;
            ext <<= 1;
            if (remainder >= denominator) { remainder -= denominator; ext |= 1ull; }
        }
        if (remainder != 0ull) ext |= 1ull;
        return fusion_f32_pack(sign, ext, exponent, policy);
    }
    if (op == 1u) ys ^= 1u;
    if (xm == 0u && ym == 0u) return {(static_cast<unsigned int>(xs == ys) * xs) << 31, 0u};
    if (xm == 0u) return fusion_f32_pack(ys, static_cast<unsigned long long>(ym) << 3, yexp, policy);
    if (ym == 0u) return fusion_f32_pack(xs, static_cast<unsigned long long>(xm) << 3, xexp, policy);
    if (xexp < yexp) {
        unsigned int tm = xm; xm = ym; ym = tm;
        int te = xexp; xexp = yexp; yexp = te;
        unsigned int ts = xs; xs = ys; ys = ts;
    }
    unsigned long long a = static_cast<unsigned long long>(xm) << 3;
    unsigned long long b = fusion_f32_shift_jam(static_cast<unsigned long long>(ym) << 3, xexp - yexp);
    unsigned int sign;
    unsigned long long ext;
    if (xs == ys) { ext = a + b; sign = xs; }
    else if (a >= b) { ext = a - b; sign = xs; }
    else { ext = b - a; sign = ys; }
    if (ext == 0ull) return {0u, 0u};
    return fusion_f32_pack(sign, ext, xexp, policy);
}
";

// The narrowing conversion follows the same roundTiesToEven and after-rounding tininess rules
// as the binary arithmetic helpers (IEEE Std 754-2019 4.3.1, 4.3.3, and 7.5). Its reported
// underflow/overflow conditions are PCU execution faults, not IEEE default status-flag behavior.
const CHECKED_F64_TO_F32_HELPER: &str = r"
struct FusionF64ToF32Result { unsigned int bits; unsigned int fault; };
__device__ __forceinline__ unsigned long long fusion_f64_to_f32_round_right_even(unsigned long long value, unsigned int shift, bool* inexact) {
    if (shift == 0u) { *inexact = false; return value; }
    if (shift > 64u) { *inexact = value != 0ull; return 0ull; }
    unsigned long long quotient = shift == 64u ? 0ull : value >> shift;
    unsigned long long mask = shift == 64u ? ~0ull : ((1ull << shift) - 1ull);
    unsigned long long remainder = value & mask;
    unsigned long long halfway = 1ull << (shift - 1u);
    bool increment = remainder > halfway || (remainder == halfway && (quotient & 1ull) != 0ull);
    *inexact = remainder != 0ull;
    return quotient + static_cast<unsigned long long>(increment);
}
__device__ __forceinline__ FusionF64ToF32Result fusion_checked_f64_to_f32(unsigned long long bits, unsigned int policy) {
    unsigned int sign = static_cast<unsigned int>((bits >> 32u) & 0x80000000ull);
    unsigned int exponent_field = static_cast<unsigned int>((bits >> 52u) & 0x7ffull);
    unsigned long long fraction = bits & 0x000fffffffffffffull;
    if (exponent_field == 0x7ffu) return {0u, 5u};
    if (exponent_field == 0u && fraction == 0ull) return {sign, 0u};
    unsigned long long significand;
    int exponent;
    if (exponent_field == 0u) { significand = fraction; exponent = -1022; }
    else { significand = fraction | (1ull << 52u); exponent = static_cast<int>(exponent_field) - 1023; }
    int top = 63 - __clzll(significand);
    int unbiased = exponent - 52 + top;
    unsigned int precision_shift = top > 23 ? static_cast<unsigned int>(top - 23) : 0u;
    bool precision_inexact = false;
    unsigned long long rounded = fusion_f64_to_f32_round_right_even(significand, precision_shift, &precision_inexact);
    if (top < 23) rounded = significand << static_cast<unsigned int>(23 - top);
    int rounded_exponent = unbiased;
    if (rounded == (1ull << 24u)) { rounded >>= 1u; ++rounded_exponent; }
    bool tiny_after_rounding = rounded_exponent < -126;
    unsigned int magnitude;
    bool inexact;
    bool subnormal;
    if (unbiased >= -126) {
        if (rounded_exponent > 127) return {(sign | 0x7f800000u), 3u};
        unsigned int exp_field = static_cast<unsigned int>(rounded_exponent + 127);
        magnitude = (exp_field << 23u) | (static_cast<unsigned int>(rounded) & 0x007fffffu);
        inexact = precision_inexact;
        subnormal = false;
    } else {
        int sub_shift_signed = -(exponent + 97);
        unsigned int sub_shift = sub_shift_signed > 0 ? static_cast<unsigned int>(sub_shift_signed) : 0u;
        bool sub_inexact = false;
        unsigned long long sub_sig = fusion_f64_to_f32_round_right_even(significand, sub_shift, &sub_inexact);
        inexact = sub_inexact;
        if (sub_sig >= (1ull << 23u)) { magnitude = 0x00800000u; subnormal = false; }
        else { magnitude = static_cast<unsigned int>(sub_sig); subnormal = sub_sig != 0ull; }
    }
    bool rejects = (policy == 0u && tiny_after_rounding && inexact)
        || (policy == 1u && (subnormal || (tiny_after_rounding && inexact)));
    unsigned int result = sign | magnitude;
    return rejects ? FusionF64ToF32Result{result, 4u} : FusionF64ToF32Result{result, 0u};
}
";

// Widening a finite binary32 value to binary64 is exact; the helper's non-finite rejection is
// part of the PCU checked-conversion contract, not IEEE 754's general conversion behavior.
const CHECKED_F32_TO_F64_HELPER: &str = r"
struct FusionF32ToF64Result { unsigned long long bits; unsigned int fault; };
__device__ __forceinline__ FusionF32ToF64Result fusion_checked_f32_to_f64(unsigned int bits) {
    unsigned int sign = bits >> 31u;
    unsigned int exponent_field = (bits >> 23u) & 0xffu;
    unsigned int fraction = bits & 0x007fffffu;
    if (exponent_field == 0xffu) return {0ull, 5u};
    unsigned long long sign64 = static_cast<unsigned long long>(sign) << 63u;
    if (exponent_field == 0u && fraction == 0u) return {sign64, 0u};
    unsigned int exponent64;
    unsigned long long fraction64;
    if (exponent_field != 0u) {
        exponent64 = exponent_field + 896u;
        fraction64 = static_cast<unsigned long long>(fraction) << 29u;
    } else {
        int top = 31 - __clz(fraction);
        exponent64 = static_cast<unsigned int>(top + 874);
        unsigned long long normalized = static_cast<unsigned long long>(fraction) << static_cast<unsigned int>(52 - top);
        fraction64 = normalized & 0x000fffffffffffffull;
    }
    return {sign64 | (static_cast<unsigned long long>(exponent64) << 52u) | fraction64, 0u};
}
";

const CHECKED_F32_RELU_HELPER: &str = r"
__device__ __forceinline__ FusionF32CheckedResult fusion_checked_f32_relu(unsigned int bits, unsigned int policy) {
    if ((bits & 0x7f800000u) == 0x7f800000u) return {bits, 5u};
    if ((bits & 0x80000000u) == 0u) {
        if ((bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u && policy == 1u) return {bits, 4u};
        return {bits, 0u};
    }
    return {0u, 0u};
}
";

const CHECKED_F64_RELU_HELPER: &str = r"
__device__ __forceinline__ FusionF64CheckedResult fusion_checked_f64_relu(unsigned long long bits, unsigned int policy) {
    if ((bits & 0x7ff0000000000000ull) == 0x7ff0000000000000ull) return {bits, 5u};
    if ((bits & 0x8000000000000000ull) == 0ull) {
        if ((bits & 0x7ff0000000000000ull) == 0ull && (bits & 0x000fffffffffffffull) != 0ull && policy == 1u) return {bits, 4u};
        return {bits, 0u};
    }
    return {0ull, 0u};
}
";

// Negation changes only the sign encoding. Finite results are exact, including signed zero
// and subnormals; only the explicit reject-subnormal policy introduces a range fault here.
// The fault retains the negated encoding so explicit clamp can publish the complete result.
const CHECKED_F32_NEG_HELPER: &str = r"
__device__ __forceinline__ FusionF32CheckedResult fusion_checked_f32_neg(unsigned int bits, unsigned int policy) {
    if ((bits & 0x7f800000u) == 0x7f800000u) return {bits, 5u};
    unsigned int result = bits ^ 0x80000000u;
    if ((bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u && policy == 1u) return {result, 4u};
    return {result, 0u};
}
";

const CHECKED_F64_NEG_HELPER: &str = r"
__device__ __forceinline__ FusionF64CheckedResult fusion_checked_f64_neg(unsigned long long bits, unsigned int policy) {
    if ((bits & 0x7ff0000000000000ull) == 0x7ff0000000000000ull) return {bits, 5u};
    unsigned long long result = bits ^ 0x8000000000000000ull;
    if ((bits & 0x7ff0000000000000ull) == 0ull && (bits & 0x000fffffffffffffull) != 0ull && policy == 1u) return {result, 4u};
    return {result, 0u};
}
";
