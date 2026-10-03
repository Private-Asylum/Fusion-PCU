//! Exact finite-input binary16/BF16/OFP8 map implementation using integer arithmetic.
//! No native low-precision arithmetic, intermediate float rounding, FTZ, or CPU fallback.
use std::fmt::Write as _;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuScalarType,
    PcuValueType,
};
use fusion_pcu::model::PcuDispatchFloatBinaryOp;
use super::RocmLowerError;
pub(super) const fn format(value_type: PcuValueType) -> Option<(u32, &'static str)> {
    match value_type {
        PcuValueType::Scalar(PcuScalarType::F16) => Some((0, "unsigned short")),
        PcuValueType::Scalar(PcuScalarType::BF16) => Some((1, "unsigned short")),
        PcuValueType::Scalar(PcuScalarType::F8E4M3FN) => Some((2, "unsigned char")),
        PcuValueType::Scalar(PcuScalarType::F8E5M2) => Some((3, "unsigned char")),
        _ => None,
    }
}
pub(super) fn uses(kernel: &fusion_pcu::PcuDispatchKernelIr<'_>) -> bool {
    fn contains(ops: &[fusion_pcu::PcuDispatchOp<'_>]) -> bool {
        ops.iter().any(|op| match op {
            fusion_pcu::PcuDispatchOp::Data(
                fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary { value_type, .. }
                | fusion_pcu::PcuDispatchDataOp::CheckedFloatUnary { value_type, .. },
            ) => format(*value_type).is_some(),
            fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. } => contains(body),
            _ => false,
        })
    }
    contains(kernel.ops)
}
#[allow(clippy::too_many_arguments)] // Keep source operands and independent policies explicit.
pub(super) fn emit(
    source: &mut String,
    indent: &str,
    index: &str,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    value_type: PcuValueType,
    result: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    let (format, ty) = format(value_type).ok_or(RocmLowerError::UnsupportedKernelInterface)?;
    let operation = match op {
        PcuDispatchFloatBinaryOp::Add => 0u32,
        PcuDispatchFloatBinaryOp::Sub => 1,
        PcuDispatchFloatBinaryOp::Mul => 2,
        PcuDispatchFloatBinaryOp::Div => 3,
    };
    let underflow = match policy {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => 0u32,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
    };
    let name = format!("fusion_low_checked_{}", result.0);
    writeln!(source, "{indent}const FusionLowResult {name} = fusion_checked_low_binary(v{}, v{}, {format}u, {operation}u, {underflow}u);", lhs.0, rhs.0).map_err(|_| RocmLowerError::FormattingFailure)?;
    if range == PcuRangePolicy::Clamp {
        writeln!(source, "{indent}if ({name}.fault != 0u) {{\n{indent}    if ({name}.fault == 3u || {name}.fault == 4u) {{\n{indent}        if (!fusion_range_fault_recorded) {{ atomicMin(fusion_fault_word, 0x8000000000000000ull | (static_cast<unsigned long long>({index}) << 3u) | {name}.fault); fusion_range_fault_recorded = true; }}\n{indent}    }} else {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({index}) << 3u) | {name}.fault); return; }}\n{indent}}}")
    } else {
        writeln!(source, "{indent}if ({name}.fault != 0u) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({index}) << 3u) | {name}.fault); return; }}")
    }.and_then(|()| writeln!(source, "{indent}{ty} v{} = static_cast<{ty}>({name}.bits);", result.0)).map_err(|_| RocmLowerError::FormattingFailure)
}
#[allow(clippy::too_many_arguments)] // Exact unary bytes and independent policies remain explicit.
pub(super) fn emit_unary(
    source: &mut String,
    indent: &str,
    index: &str,
    op: fusion_pcu::PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    value_type: PcuValueType,
    result: PcuDispatchValueId,
    value: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    let (format, ty) = format(value_type).ok_or(RocmLowerError::UnsupportedKernelInterface)?;
    let operation = u32::from(op == fusion_pcu::PcuDispatchFloatUnaryOp::Relu);
    let tight = u32::from(policy == PcuFloatUnderflowPolicy::RejectSubnormalResult);
    let name = format!("fusion_low_unary_{}", result.0);
    writeln!(source, "{indent}const FusionLowResult {name} = fusion_checked_low_unary(v{}, {format}u, {operation}u, {tight}u);", value.0).map_err(|_| RocmLowerError::FormattingFailure)?;
    if range == PcuRangePolicy::Clamp {
        writeln!(source, "{indent}if ({name}.fault != 0u) {{\n{indent}    if ({name}.fault == 4u) {{\n{indent}        if (!fusion_range_fault_recorded) {{ atomicMin(fusion_fault_word, 0x8000000000000000ull | (static_cast<unsigned long long>({index}) << 3u) | {name}.fault); fusion_range_fault_recorded = true; }}\n{indent}    }} else {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({index}) << 3u) | {name}.fault); return; }}\n{indent}}}")
    } else {
        writeln!(source, "{indent}if ({name}.fault != 0u) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({index}) << 3u) | {name}.fault); return; }}")
    }.and_then(|()| writeln!(source, "{indent}{ty} v{} = static_cast<{ty}>({name}.bits);", result.0)).map_err(|_| RocmLowerError::FormattingFailure)
}
pub(super) const HELPERS: &str = r"
struct FusionLowResult { unsigned int bits; unsigned int fault; };
struct FusionLowFinite { bool negative; unsigned long long sig; int scale; };
// Four named encodings; E4M3FN admits final exponent fractions 000 through 110.
struct FusionLowFormat { unsigned int fraction; int bias; unsigned int sign; unsigned int maximum; };
__device__ __forceinline__ FusionLowFormat fusion_low_format(unsigned int format) {
    if (format == 0u) return {10u, 15, 0x8000u, 0x7bffu};
    if (format == 1u) return {7u, 127, 0x8000u, 0x7f7fu};
    if (format == 2u) return {3u, 7, 0x80u, 0x7eu};
    return {2u, 15, 0x80u, 0x7bu};
}
// Exact encoding-only sign/selection. IEEE 754-2019 5.5.1 sign operations are exact;
// PCU rejects nonfinite operands even for the unselected ReLU branch. Tight policy
// rejects only a selected nonzero subnormal; Clamp preserves its exact destination bits.
__device__ __forceinline__ FusionLowResult fusion_checked_low_unary(unsigned int bits, unsigned int format, unsigned int op, unsigned int tight) {
    FusionLowFormat f = fusion_low_format(format);
    unsigned int magnitude = bits & (f.sign - 1u);
    if (magnitude > f.maximum) return {0u, 5u};
    unsigned int result = op == 0u ? bits ^ f.sign : ((bits & f.sign) == 0u && magnitude != 0u ? bits : 0u);
    magnitude = result & (f.sign - 1u);
    return {result, tight != 0u && magnitude != 0u && magnitude < (1u << f.fraction) ? 4u : 0u};
}
__device__ __forceinline__ FusionLowFinite fusion_low_decode(unsigned int bits, FusionLowFormat f) {
    unsigned int exponent = (bits & (f.sign - 1u)) >> f.fraction;
    return {(bits & f.sign) != 0u,
        (bits & ((1u << f.fraction) - 1u)) | (exponent == 0u ? 0ull : (1ull << f.fraction)),
        static_cast<int>(exponent == 0u ? 1u : exponent) - f.bias - static_cast<int>(f.fraction)};
}
// Binary maps have <=44 numerator bits and <=11 denominator bits. Positive
// shifts produce at most 22 bits; a denominator beyond u64 makes rounding zero.
// All shifts are guarded, including the exact -64 boundary, and d is nonzero.
__device__ __forceinline__ unsigned long long fusion_low_round(unsigned long long n, unsigned long long d, int shift, bool* inexact) {
    if (shift >= 0) n <<= shift;
    else {
        int distance = -shift;
        if (distance >= 64 || d > (~0ull >> distance)) { *inexact = n != 0ull; return 0ull; }
        d <<= distance;
    }
    unsigned long long q = n / d;
    unsigned long long r = n % d;
    *inexact = r != 0ull;
    return q + static_cast<unsigned long long>(r > d - r || (r == d - r && (q & 1ull) != 0ull));
}
__device__ __forceinline__ int fusion_low_ratio_exponent(unsigned long long n, unsigned long long d) {
    int estimate = __builtin_clzll(d) - __builtin_clzll(n);
    bool below = estimate >= 0 ? n < (d << estimate) : (n << -estimate) < d;
    return estimate - static_cast<int>(below);
}
// IEEE 754-2019 4.3.1 nearest/ties-even, 7.4 overflow, 7.5(a) tininess
// after destination-precision rounding with unbounded exponent. PCU's faults
// and nonfinite-input rejection are distinct from IEEE default exceptions.
__device__ __forceinline__ FusionLowResult fusion_low_pack(bool negative, unsigned long long n, unsigned long long d, int scale, FusionLowFormat f, unsigned int policy) {
    unsigned int sign = negative ? f.sign : 0u;
    if (n == 0ull) return {sign, 0u};
    int fraction = static_cast<int>(f.fraction);
    int exponent = scale + fusion_low_ratio_exponent(n, d);
    int normal_scale = exponent - fraction;
    bool ignored = false;
    unsigned long long unbounded = fusion_low_round(n, d, scale - normal_scale, &ignored);
    int unbounded_exponent = exponent + static_cast<int>(unbounded == (1ull << (f.fraction + 1u)));
    int minimum_exponent = 1 - f.bias;
    int final_scale = normal_scale > minimum_exponent - fraction ? normal_scale : minimum_exponent - fraction;
    bool inexact = false;
    unsigned long long rounded = fusion_low_round(n, d, scale - final_scale, &inexact);
    int packed_exponent = final_scale + fraction;
    if (rounded == (1ull << (f.fraction + 1u))) { rounded >>= 1; ++packed_exponent; }
    int maximum_exponent = static_cast<int>(f.maximum >> f.fraction) - f.bias;
    unsigned long long maximum_significand = (1ull << f.fraction) | (f.maximum & ((1u << f.fraction) - 1u));
    if (packed_exponent > maximum_exponent || (packed_exponent == maximum_exponent && rounded > maximum_significand)) return {sign | f.maximum, 3u};
    bool subnormal = rounded < (1ull << f.fraction);
    unsigned int magnitude = subnormal ? static_cast<unsigned int>(rounded) :
        (static_cast<unsigned int>(packed_exponent + f.bias) << f.fraction) | (static_cast<unsigned int>(rounded) & ((1u << f.fraction) - 1u));
    unsigned int bits = sign | magnitude;
    bool tiny = unbounded_exponent < minimum_exponent;
    bool rejected = policy == 0u ? tiny && inexact : policy == 1u ? (tiny && inexact) || (subnormal && rounded != 0ull) : false;
    return {bits, rejected ? 4u : 0u};
}
__device__ __forceinline__ FusionLowResult fusion_checked_low_binary(unsigned int left, unsigned int right, unsigned int format, unsigned int op, unsigned int policy) {
    FusionLowFormat f = fusion_low_format(format);
    if ((left & (f.sign - 1u)) > f.maximum || (right & (f.sign - 1u)) > f.maximum) return {0u, 5u};
    FusionLowFinite a = fusion_low_decode(left, f);
    FusionLowFinite b = fusion_low_decode(right, f);
    if (op == 2u) return fusion_low_pack(a.negative != b.negative, a.sig * b.sig, 1ull, a.scale + b.scale, f, policy);
    if (op == 3u) {
        if (b.sig == 0ull) return {0u, 1u};
        return fusion_low_pack(a.negative != b.negative, a.sig, b.sig, a.scale - b.scale, f, policy);
    }
    b.negative = b.negative != (op == 1u);
    if (a.sig == 0ull && b.sig == 0ull) return {a.negative && b.negative ? f.sign : 0u, 0u};
    if (a.sig == 0ull) return fusion_low_pack(b.negative, b.sig, 1ull, b.scale, f, policy);
    if (b.sig == 0ull) return fusion_low_pack(a.negative, a.sig, 1ull, a.scale, f, policy);
    if (a.scale < b.scale) { FusionLowFinite temporary = a; a = b; b = temporary; }
    int gap = a.scale - b.scale;
    // <=11-bit precision: this smaller finite operand cannot reach a midpoint
    // or cancel the larger normal value. Returning it still applies the policy.
    if (gap > 32) return fusion_low_pack(a.negative, a.sig, 1ull, a.scale, f, policy);
    unsigned long long large = a.sig << gap;
    if (a.negative == b.negative) return fusion_low_pack(a.negative, large + b.sig, 1ull, b.scale, f, policy);
    if (large >= b.sig) return fusion_low_pack(a.negative && large != b.sig, large - b.sig, 1ull, b.scale, f, policy);
    return fusion_low_pack(b.negative, b.sig - large, 1ull, b.scale, f, policy);
}
";
