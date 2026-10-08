//! Register-bounded, integer-only binary64 CUDA helper source.
//!
//! Add/subtract retain a 53-bit significand plus three rounding bits and a sticky bit. Multiply
//! forms its exact 106-bit product from four 32-bit partial products held in two 64-bit limbs.
//! The device floating-point unit is never used for arithmetic or classification.
//!
//! The packer uses roundTiesToEven (IEEE Std 754-2019 4.3.1; the required binary default is
//! specified in 4.3.3), then detects tininess after rounding to destination precision (one of
//! the permitted binary choices in 7.5) and requires final-packing inexactness to report
//! underflow under `IeeeAfterRounding`; `RejectSubnormalResult` additionally faults on exact
//! subnormals. Overflow classification follows 7.4's rounded-result threshold. PCU's range
//! faults and non-finite-input rejection are PCU policy; this helper does not implement IEEE
//! default exception flags or claim full IEEE 754 conformance.

pub(super) const CHECKED_F64_HELPERS: &str = r"
struct FusionF64CheckedResult { unsigned long long bits; unsigned int fault; };
__device__ __forceinline__ unsigned long long fusion_f64_shift_jam(unsigned long long value, unsigned int distance) {
    if (distance == 0u) return value;
    if (distance < 64u) return (value >> distance) | static_cast<unsigned long long>((value & ((1ull << distance) - 1ull)) != 0ull);
    return static_cast<unsigned long long>(value != 0ull);
}
// The exact product is at most 106 bits and is shifted to bit 55, so distance is 0..=50.
__device__ __forceinline__ unsigned long long fusion_f64_pair_shift_jam(unsigned long long lo, unsigned long long hi, unsigned int distance) {
    if (distance == 0u) return lo;
    unsigned long long retained = (lo >> distance) | (hi << (64u - distance));
    bool sticky = distance < 64u ? (lo & ((1ull << distance) - 1ull)) != 0ull : lo != 0ull;
    return retained | static_cast<unsigned long long>(sticky);
}
__device__ __forceinline__ bool fusion_f64_round_increment(unsigned long long significand, unsigned int discarded) {
    return discarded > 4u || (discarded == 4u && (significand & 1ull) != 0ull);
}
// Private to fusion_f64_pack, which has already normalized ext and exponent.
__device__ __forceinline__ void fusion_f64_round_precision(unsigned long long* ext, int* exponent) {
    unsigned int discarded = static_cast<unsigned int>(*ext & 7ull);
    unsigned long long significand = *ext >> 3;
    if (fusion_f64_round_increment(significand, discarded)) ++significand;
    if (significand == 0x0020000000000000ull) { significand >>= 1; ++*exponent; }
    *ext = significand << 3;
}
__device__ __forceinline__ FusionF64CheckedResult fusion_f64_pack(bool sign, unsigned long long ext, int exponent, unsigned int policy) {
    while (ext >= 0x0100000000000000ull) { ext = fusion_f64_shift_jam(ext, 1u); ++exponent; }
    while (ext < 0x0080000000000000ull) { ext <<= 1; --exponent; }
    unsigned long long precision_ext = ext;
    int rounded_exponent = exponent;
    fusion_f64_round_precision(&precision_ext, &rounded_exponent);
    (void)precision_ext;
    bool tiny_after_rounding = rounded_exponent < -1022;
    if (exponent < -1022) {
        ext = fusion_f64_shift_jam(ext, static_cast<unsigned int>(-1022 - exponent));
        exponent = -1022;
    }
    unsigned int discarded = static_cast<unsigned int>(ext & 7ull);
    unsigned long long significand = ext >> 3;
    bool inexact = discarded != 0u;
    if (fusion_f64_round_increment(significand, discarded)) ++significand;
    unsigned long long sign_bits = sign ? 0x8000000000000000ull : 0ull;
    unsigned long long bits;
    bool subnormal;
    if (exponent == -1022 && significand < 0x0010000000000000ull) {
        bits = sign_bits | significand;
        subnormal = significand != 0ull;
    } else {
        if (significand >= 0x0020000000000000ull) { significand >>= 1; ++exponent; }
        if (exponent > 1023) return {sign_bits | 0x7ff0000000000000ull, 3u};
        bits = sign_bits | (static_cast<unsigned long long>(exponent + 1023) << 52) | (significand & 0x000fffffffffffffull);
        subnormal = false;
    }
    if ((policy == 0u && tiny_after_rounding && inexact) || (policy == 1u && (subnormal || (tiny_after_rounding && inexact)))) return {bits, 4u};
    return {bits, 0u};
}
__device__ __forceinline__ FusionF64CheckedResult fusion_checked_f64_binary(unsigned long long x, unsigned long long y, unsigned int op, unsigned int policy) {
    unsigned int xe = static_cast<unsigned int>((x >> 52) & 0x7ffull), ye = static_cast<unsigned int>((y >> 52) & 0x7ffull);
    if (xe == 0x7ffu || ye == 0x7ffu) return {0ull, 5u};
    bool xs = (x >> 63) != 0ull, ys = (y >> 63) != 0ull;
    unsigned long long xm = x & 0x000fffffffffffffull, ym = y & 0x000fffffffffffffull;
    int xexp = -1022, yexp = -1022;
    if (xe != 0u) { xm |= 0x0010000000000000ull; xexp = static_cast<int>(xe) - 1023; }
    if (ye != 0u) { ym |= 0x0010000000000000ull; yexp = static_cast<int>(ye) - 1023; }
    if (op == 2u) {
        if (xm == 0ull || ym == 0ull) return {static_cast<unsigned long long>(xs != ys) << 63, 0u};
        unsigned int av[2] = {static_cast<unsigned int>(xm), static_cast<unsigned int>(xm >> 32)};
        unsigned int bv[2] = {static_cast<unsigned int>(ym), static_cast<unsigned int>(ym >> 32)};
        unsigned int product[4] = {};
        for (int i = 0; i < 2; ++i) {
            unsigned long long carry = 0ull;
            for (int j = 0; j < 2; ++j) {
                unsigned long long t = static_cast<unsigned long long>(av[i]) * bv[j] + product[i + j] + carry;
                product[i + j] = static_cast<unsigned int>(t);
                carry = t >> 32;
            }
            product[i + 2] = static_cast<unsigned int>(carry);
        }
        unsigned long long lo = static_cast<unsigned long long>(product[0]) | (static_cast<unsigned long long>(product[1]) << 32);
        unsigned long long hi = static_cast<unsigned long long>(product[2]) | (static_cast<unsigned long long>(product[3]) << 32);
        int top = hi != 0ull ? 127 - __clzll(hi) : 63 - __clzll(lo);
        unsigned long long ext = top > 55 ? fusion_f64_pair_shift_jam(lo, hi, static_cast<unsigned int>(top - 55)) : lo << static_cast<unsigned int>(55 - top);
        return fusion_f64_pack(xs != ys, ext, xexp + yexp + top - 104, policy);
    }
    if (op == 3u) {
        bool sign = xs != ys;
        if ((y << 1) == 0ull) return {0ull, 1u};
        if ((x << 1) == 0ull) return {static_cast<unsigned long long>(sign) << 63, 0u};
        int a_shift = __clzll(xm) - 11;
        int b_shift = __clzll(ym) - 11;
        xm <<= a_shift; xexp -= a_shift;
        ym <<= b_shift; yexp -= b_shift;
        unsigned long long numerator = xm;
        unsigned long long denominator = ym;
        int exponent = xexp - yexp;
        if (numerator < denominator) { numerator <<= 1; --exponent; }
        unsigned long long remainder = numerator - denominator;
        unsigned long long ext = 1ull;
        for (int i = 0; i < 55; ++i) {
            remainder <<= 1;
            ext <<= 1;
            if (remainder >= denominator) { remainder -= denominator; ext |= 1ull; }
        }
        if (remainder != 0ull) ext |= 1ull;
        return fusion_f64_pack(sign, ext, exponent, policy);
    }
    if (op == 1u) ys = !ys;
    if (xm == 0ull && ym == 0ull) return {static_cast<unsigned long long>(xs == ys && xs) << 63, 0u};
    if (xm == 0ull) return fusion_f64_pack(ys, ym << 3, yexp, policy);
    if (ym == 0ull) return fusion_f64_pack(xs, xm << 3, xexp, policy);
    if (xexp < yexp) {
        unsigned long long magnitude = xm; xm = ym; ym = magnitude;
        int exponent = xexp; xexp = yexp; yexp = exponent;
        bool sign = xs; xs = ys; ys = sign;
    }
    unsigned int distance = static_cast<unsigned int>(xexp - yexp);
    unsigned long long a = xm << 3;
    unsigned long long b = fusion_f64_shift_jam(ym << 3, distance);
    unsigned long long ext;
    bool sign;
    if (xs == ys) { ext = a + b; sign = xs; }
    else if (a >= b) { ext = a - b; sign = xs; }
    else { ext = b - a; sign = ys; }
    if (ext == 0ull) return {0ull, 0u};
    return fusion_f64_pack(sign, ext, xexp, policy);
}
";
