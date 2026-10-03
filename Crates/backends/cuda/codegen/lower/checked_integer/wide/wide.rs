//! Checked wide integer limb lowering, separate from representation-only carriers.
use std::fmt::Write as _;
#[rustfmt::skip]
use fusion_pcu::{PcuScalarType,PcuDispatchKernelIr,PcuDispatchOp,PcuDispatchDataOp,PcuDispatchValueId};
use fusion_pcu::model::PcuDispatchIntegerBinaryOp;
use super::CudaLowerError;
fn uses(ops: &[PcuDispatchOp<'_>]) -> bool {
    ops.iter().any(|op| match op {
        PcuDispatchOp::Data(
            PcuDispatchDataOp::CheckedIntegerBinary { value_type, .. }
            | PcuDispatchDataOp::CheckedDivRem { value_type, .. },
        ) => {
            matches!(
                value_type.scalar_type(),
                PcuScalarType::I128
                    | PcuScalarType::U128
                    | PcuScalarType::I256
                    | PcuScalarType::U256
                    | PcuScalarType::I512
                    | PcuScalarType::U512
            )
        }
        PcuDispatchOp::GridStrideLoop { body, .. } => uses(body),
        _ => false,
    })
}
pub fn emit_helpers(source: &mut String, kernel: &PcuDispatchKernelIr<'_>) {
    if uses(kernel.ops) {
        source.push_str(HELPERS);
        if uses_div_rem(kernel.ops) {
            source.push_str(DIV_REM_HELPERS);
        }
    }
}
fn uses_div_rem(ops: &[PcuDispatchOp<'_>]) -> bool {
    ops.iter().any(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { .. }) => true,
        PcuDispatchOp::GridStrideLoop { body, .. } => uses_div_rem(body),
        _ => false,
    })
}
#[allow(clippy::too_many_arguments)] // Mirror the existing cold typed instruction emitter.
pub(super) fn emit(
    source: &mut String,
    indent: &str,
    index: &str,
    scalar: PcuScalarType,
    op: PcuDispatchIntegerBinaryOp,
    range_policy: fusion_pcu::PcuRangePolicy,
    result: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), CudaLowerError> {
    let cpp_type = match scalar.bit_width() {
        128 => "FusionBits128",
        256 => "FusionBits256",
        512 => "FusionBits512",
        _ => return Err(CudaLowerError::UnsupportedKernelInterface),
    };
    let signed = matches!(
        scalar,
        PcuScalarType::I128 | PcuScalarType::I256 | PcuScalarType::I512
    );
    let operation = match op {
        PcuDispatchIntegerBinaryOp::Add => 0,
        PcuDispatchIntegerBinaryOp::Sub => 1,
        PcuDispatchIntegerBinaryOp::Mul => 2,
    };
    writeln!(source,"{indent}const FusionWideResult<{cpp_type}> fusion_checked_wide_{} = fusion_checked_wide_binary(v{}, v{}, {operation}u, {signed});",result.0,lhs.0,rhs.0).map_err(|_| CudaLowerError::FormattingFailure)?;
    if range_policy == fusion_pcu::PcuRangePolicy::Clamp {
        writeln!(source,"{indent}{cpp_type} v{} = fusion_checked_wide_{}.fault == 0u ? fusion_checked_wide_{}.value : fusion_wide_saturate<{cpp_type}>(fusion_checked_wide_{}.fault, {signed});",result.0,result.0,result.0,result.0)
        .and_then(|()|writeln!(source,"{indent}if (fusion_checked_wide_{}.fault != 0u && !fusion_range_fault_recorded) {{ atomicMin(fusion_fault_word, 0x8000000000000000ull | (static_cast<unsigned long long>({index}) << 3u) | fusion_checked_wide_{}.fault); fusion_range_fault_recorded = true; }}",result.0,result.0))
        .map_err(|_| CudaLowerError::FormattingFailure)
    } else {
        writeln!(source,"{indent}if (fusion_checked_wide_{}.fault != 0u) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({index}) << 3u) | fusion_checked_wide_{}.fault); return; }}",result.0,result.0)
        .and_then(|()|writeln!(source,"{indent}{cpp_type} v{} = fusion_checked_wide_{}.value;",result.0,result.0)).map_err(|_| CudaLowerError::FormattingFailure)
    }
}
const HELPERS: &str = r"// Exact two's-complement range checking using only unsigned 32/64-bit arithmetic.
// No host arithmetic fallback, native 128-bit type, or signed-overflow expression.
template<typename T> struct FusionWideResult { T value; unsigned fault; };
template<typename T> __device__ __forceinline__ T fusion_wide_saturate(unsigned fault, bool signed_type) {
    constexpr unsigned N = sizeof(T) / sizeof(unsigned long long);
    T value = {};
    if (fault == 3u) {
        for (unsigned i=0; i<N; ++i) value.limbs[i] = 0xffffffffffffffffull;
        if (signed_type) value.limbs[N-1] = 0x7fffffffffffffffull;
    } else if (signed_type) value.limbs[N-1] = 0x8000000000000000ull;
    return value;
}
template<typename T> __device__ __forceinline__ T fusion_wide_negate(T value) {
    constexpr unsigned N = sizeof(T) / sizeof(unsigned long long);
    unsigned long long carry = 1ull;
    for (unsigned i=0; i<N; ++i) {
        const unsigned long long flipped = ~value.limbs[i];
        value.limbs[i] = flipped + carry;
        carry = carry && value.limbs[i] == 0ull;
    }
    return value;
}
template<typename T> __device__ __forceinline__ FusionWideResult<T> fusion_checked_wide_binary(T a, T b, unsigned operation, bool signed_type) {
    constexpr unsigned N = sizeof(T) / sizeof(unsigned long long);
    constexpr unsigned D = N * 2u;
    T value = {};
    const bool a_negative = signed_type && (a.limbs[N-1] >> 63u) != 0ull;
    const bool b_negative = signed_type && (b.limbs[N-1] >> 63u) != 0ull;
    if (operation < 2u) {
        unsigned long long carry = 0ull;
        for (unsigned i=0; i<N; ++i) {
            if (operation == 0u) {
                const unsigned long long partial = a.limbs[i] + b.limbs[i];
                const bool first_carry = partial < a.limbs[i];
                value.limbs[i] = partial + carry;
                carry = first_carry || value.limbs[i] < partial;
            } else {
                const unsigned long long partial = a.limbs[i] - b.limbs[i];
                const bool first_borrow = a.limbs[i] < b.limbs[i];
                value.limbs[i] = partial - carry;
                carry = first_borrow || partial < carry;
            }
        }
        const bool result_negative = (value.limbs[N-1] >> 63u) != 0ull;
        const bool signed_range = operation == 0u
            ? a_negative == b_negative && result_negative != a_negative
            : a_negative != b_negative && result_negative != a_negative;
        const unsigned fault = signed_type
            ? (signed_range ? (a_negative ? 4u : 3u) : 0u)
            : (carry ? (operation == 0u ? 3u : 4u) : 0u);
        return {value, fault};
    }
    const bool negative = a_negative != b_negative;
    if (a_negative) a = fusion_wide_negate(a);
    if (b_negative) b = fusion_wide_negate(b);
    unsigned left[D], right[D], product[D*2u] = {};
    for (unsigned i=0; i<N; ++i) {
        left[i*2u] = static_cast<unsigned>(a.limbs[i]);
        left[i*2u+1u] = static_cast<unsigned>(a.limbs[i] >> 32u);
        right[i*2u] = static_cast<unsigned>(b.limbs[i]);
        right[i*2u+1u] = static_cast<unsigned>(b.limbs[i] >> 32u);
    }
    // (2^32-1)^2 + (2^32-1) + (2^32-1) fits exactly in u64.
    for (unsigned i=0; i<D; ++i) {
        unsigned long long carry = 0ull;
        for (unsigned j=0; j<D; ++j) {
            const unsigned long long sum = static_cast<unsigned long long>(left[i]) * right[j] + product[i+j] + carry;
            product[i+j] = static_cast<unsigned>(sum);
            carry = sum >> 32u;
        }
        product[i+D] = static_cast<unsigned>(carry);
    }
    bool outside = false;
    for (unsigned i=D; i<D*2u; ++i) outside = outside || product[i] != 0u;
    if (signed_type) {
        if (!negative) outside = outside || (product[D-1] >> 31u) != 0u;
        else {
            outside = outside || product[D-1] > 0x80000000u;
            if (product[D-1] == 0x80000000u) {
                for (unsigned i=0; i<D-1u; ++i) outside = outside || product[i] != 0u;
            }
        }
    }
    if (outside) return {value, signed_type && negative ? 4u : 3u};
    for (unsigned i=0; i<N; ++i) value.limbs[i] = static_cast<unsigned long long>(product[i*2u]) | (static_cast<unsigned long long>(product[i*2u+1u]) << 32u);
    if (negative) value = fusion_wide_negate(value);
    return {value, 0u};
}
";

#[allow(clippy::too_many_arguments)] // Mirror the checked instruction's joint-output ABI.
pub fn emit_div_rem(
    source: &mut String,
    indent: &str,
    index: &str,
    scalar: PcuScalarType,
    quotient: PcuDispatchValueId,
    remainder: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), CudaLowerError> {
    let cpp_type = match scalar.bit_width() {
        128 => "FusionBits128",
        256 => "FusionBits256",
        512 => "FusionBits512",
        _ => return Err(CudaLowerError::UnsupportedKernelInterface),
    };
    let signed = matches!(
        scalar,
        PcuScalarType::I128 | PcuScalarType::I256 | PcuScalarType::I512
    );
    writeln!(source, "{indent}const FusionWideDivRem<{cpp_type}> fusion_joint_{} = fusion_checked_wide_div_rem(v{}, v{}, {signed});", quotient.0, lhs.0, rhs.0)
        .and_then(|()| writeln!(source, "{indent}if (fusion_joint_{}.fault != 0u) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({index}) << 3u) | fusion_joint_{}.fault); return; }}", quotient.0, quotient.0))
        .and_then(|()| writeln!(source, "{indent}{cpp_type} v{} = fusion_joint_{}.quotient;", quotient.0, quotient.0))
        .and_then(|()| writeln!(source, "{indent}{cpp_type} v{} = fusion_joint_{}.remainder;", remainder.0, quotient.0))
        .map_err(|_| CudaLowerError::FormattingFailure)
}

const DIV_REM_HELPERS: &str = r"// Proposed joint restoring division; raw limbs are little endian two's complement.
template<typename T> struct FusionWideDivRem { T quotient; T remainder; unsigned fault; };
template<typename T> __device__ __forceinline__ int fusion_wide_compare(T a, T b) {
    constexpr unsigned N = sizeof(T) / sizeof(unsigned long long);
    for (unsigned i=N; i>0u; --i) {
        if (a.limbs[i-1u] < b.limbs[i-1u]) return -1;
        if (a.limbs[i-1u] > b.limbs[i-1u]) return 1;
    }
    return 0;
}
template<typename T> __device__ __forceinline__ T fusion_wide_subtract(T a, T b) {
    constexpr unsigned N = sizeof(T) / sizeof(unsigned long long);
    unsigned long long borrow = 0ull;
    for (unsigned i=0; i<N; ++i) {
        const unsigned long long partial = a.limbs[i] - b.limbs[i];
        const bool first_borrow = a.limbs[i] < b.limbs[i];
        a.limbs[i] = partial - borrow;
        borrow = first_borrow || partial < borrow;
    }
    return a;
}
template<typename T> __device__ __forceinline__ FusionWideDivRem<T> fusion_checked_wide_div_rem(T a, T b, bool signed_type) {
    constexpr unsigned N = sizeof(T) / sizeof(unsigned long long);
    T quotient = {}, remainder = {};
    bool zero = true, minimum = a.limbs[N-1u] == 0x8000000000000000ull;
    bool minus_one = true, denominator_one = b.limbs[0] == 1ull;
    for (unsigned i=0; i<N; ++i) {
        zero = zero && b.limbs[i] == 0ull;
        minus_one = minus_one && b.limbs[i] == 0xffffffffffffffffull;
        if (i+1u<N) minimum = minimum && a.limbs[i] == 0ull;
        if (i>0u) denominator_one = denominator_one && b.limbs[i] == 0ull;
    }
    if (zero) return {quotient, remainder, 1u};
    // Shared status protocol: signed division overflow is 2; arithmetic range overflow is 3.
    if (signed_type && minimum && minus_one) return {quotient, remainder, 2u};
    if (denominator_one) return {a, remainder, 0u};
    const bool a_negative = signed_type && (a.limbs[N-1u] >> 63u) != 0ull;
    const bool b_negative = signed_type && (b.limbs[N-1u] >> 63u) != 0ull;
    if (a_negative) a = fusion_wide_negate(a);
    if (b_negative) b = fusion_wide_negate(b);
    const int comparison = fusion_wide_compare(a, b);
    if (comparison < 0) remainder = a;
    else if (comparison == 0) quotient.limbs[0] = 1ull;
    else {
        // One pass yields both outputs. R < denominator before each shift, so
        // the possible W+1-bit carry is consumed by exactly one subtraction.
        for (unsigned bit=N*64u; bit>0u; --bit) {
            unsigned long long carry = (a.limbs[(bit-1u)/64u] >> ((bit-1u)%64u)) & 1ull;
            for (unsigned i=0; i<N; ++i) {
                const unsigned long long next = remainder.limbs[i] >> 63u;
                remainder.limbs[i] = (remainder.limbs[i] << 1u) | carry;
                carry = next;
            }
            if (carry != 0ull || fusion_wide_compare(remainder, b) >= 0) {
                remainder = fusion_wide_subtract(remainder, b);
                quotient.limbs[(bit-1u)/64u] |= 1ull << ((bit-1u)%64u);
            }
        }
    }
    if (a_negative != b_negative) quotient = fusion_wide_negate(quotient);
    if (a_negative) remainder = fusion_wide_negate(remainder);
    return {quotient, remainder, 0u};
}
";
