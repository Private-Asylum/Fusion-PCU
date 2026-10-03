#include <metal_stdlib>
using namespace metal;
// Independently compiled/proved PCU binary32 integer realization. No native floating ALU,
// 64-bit arithmetic, contraction or device FTZ/DAZ assumption. IEEE754-2019 4.3.1 nearest
// ties-even and7.5(a) tininess after precision rounding; finite-input faults are PCU policy.
// Exact32x32->64 from bounded16-bit products, all intermediates fit U32.
inline uint2 pcu_mul32(uint x, uint y) {
    uint x0=x&0xffffu, x1=x>>16, y0=y&0xffffu, y1=y>>16;
    uint w0=x0*y0, t=x1*y0+(w0>>16);
    uint w1=(t&0xffffu)+x0*y1;
    return uint2((w1<<16)|(w0&0xffffu),x1*y1+(t>>16)+(w1>>16));
}
struct Result { uint bits; uint status; };
struct Parts { uint sign; uint significand; int exponent; };
// Diagnostic ABI: 0 success, 1 invalid operand, 2 underflow, 3 overflow, 4 zero divisor.
Result ok(uint bits) { return {bits, 0}; }
Result fault(uint status) { return {0, status}; }
Parts decode(uint bits) {
    uint exponent = (bits >> 23) & 255u;
    uint significand = bits & 0x7fffffu;
    if (exponent != 0) significand |= 0x800000u;
    return {bits >> 31, significand, exponent == 0 ? -126 : int(exponent)-127};
}
// Every call guards distance before shifting. Zero and >=32 never execute a shift.
// Values jammed here fit <=28 bits; hence >=32 equals the wider core oracle's zero+sticky.
uint jam(uint value, uint distance) {
    if (distance == 0) return value;
    if (distance >= 32) return uint(value != 0);
    return (value >> distance) | uint((value & ((1u << distance)-1u)) != 0);
}
bool increment(uint significand, uint discarded) {
    return discarded > 4u || (discarded == 4u && (significand & 1u) != 0);
}
struct Binary {
uint POLICY;
bool CLAMP;
Result range_fault(uint bits,uint status) { return {CLAMP ? bits : 0u,status | (CLAMP ? 0x100u : 0u)}; }
Result passthrough(uint bits) {
    uint magnitude = bits & 0x7fffffffu;
    if (POLICY == 1 && magnitude != 0 && magnitude < 0x800000u) return range_fault(bits,2);
    return ok(bits);
}
// Precondition ext!=0. Add supplies <2^28; Mul/Div supply normalized <2^27.
// Left normalization stops at bit26, so shifts/additions never overflow a U32.
// Exponents remain in [-298,277], including gradual packing; I32 never overflows.
Result pack(uint sign, uint ext, int exponent) {
    while (ext >= (1u << 27)) { ext = jam(ext, 1); exponent += 1; }
    while (ext < (1u << 26)) { ext <<= 1; exponent -= 1; }
    uint unbounded_sig = ext >> 3;
    if (increment(unbounded_sig, ext & 7u)) unbounded_sig += 1;
    int unbounded_exp = exponent + int(unbounded_sig == (1u << 24));
    bool tiny = unbounded_exp < -126;
    if (exponent < -126) { ext = jam(ext, uint(-126-exponent)); exponent = -126; }
    uint discarded = ext & 7u;
    uint significand = ext >> 3;
    bool inexact = discarded != 0;
    if (increment(significand, discarded)) significand += 1;
    bool subnormal = false;
    uint bits;
    if (exponent == -126 && significand < 0x800000u) {
        bits = (sign << 31) | significand;
        subnormal = significand != 0;
    } else {
        if (significand >= (1u << 24)) { significand >>= 1; exponent += 1; }
        if (exponent > 127) return range_fault((sign << 31) | 0x7f7fffffu,3);
        bits = (sign << 31) | (uint(exponent+127) << 23) | (significand & 0x7fffffu);
    }
    bool reject = POLICY == 0 ? (tiny && inexact) : POLICY == 1 ? (subnormal || (tiny && inexact)) : false;
    return reject ? range_fault(bits,2) : ok(bits);
}
Result add(uint left, uint right, bool subtract) {
    Parts a = decode(left); Parts b = decode(right); b.sign ^= uint(subtract);
    if (a.significand == 0) {
        if (b.significand != 0) return passthrough((b.sign << 31) | (right & 0x7fffffffu));
        return ok(uint(a.sign == b.sign) * (a.sign << 31));
    }
    if (b.significand == 0) return passthrough(left);
    if (a.exponent < b.exponent) { Parts temporary = a; a = b; b = temporary; }
    uint x = a.significand << 3; // <2^27, maximum sum of both extended inputs <2^28.
    uint y = jam(b.significand << 3, uint(a.exponent-b.exponent));
    uint sign; uint ext;
    if (a.sign == b.sign) { sign = a.sign; ext = x+y; }
    else if (x >= y) { sign = a.sign; ext = x-y; }
    else { sign = b.sign; ext = y-x; }
    return ext == 0 ? ok(0) : pack(sign, ext, a.exponent);
}
Result multiply(uint left, uint right) {
    Parts a = decode(left); Parts b = decode(right);
    uint sign = a.sign ^ b.sign;
    if (a.significand == 0 || b.significand == 0) return ok(sign << 31);
    // Each significand <2^24; umulExtended computes all <=48 product bits exactly.
    uint2 product = pcu_mul32(a.significand, b.significand);
    uint low = product.x, high = product.y;
    int top = high != 0 ? 63-int(clz(high)) : 31-int(clz(low)); // product nonzero: top in0..47.
    int exponent = a.exponent+b.exponent+top-46;
    uint ext;
    if (top > 26) {
        uint distance = uint(top-26); // 1..21: both limb shifts are strictly below32.
        ext = (high << (32u-distance)) | (low >> distance);
        ext |= uint((low & ((1u << distance)-1u)) != 0);
    } else {
        ext = low << uint(26-top); // high=0 and resulting top=26, never overflows.
    }
    return pack(sign, ext, exponent);
}
Result divide(uint left, uint right) {
    Parts a = decode(left); Parts b = decode(right);
    uint sign = (left ^ right) >> 31;
    if (b.significand == 0) return fault(4);
    if (a.significand == 0) return ok(sign << 31);
    uint a_shift = uint(clz(a.significand))-8u; // both nonzero significands: 0..23.
    uint b_shift = uint(clz(b.significand))-8u;
    a.significand <<= a_shift; a.exponent -= int(a_shift);
    b.significand <<= b_shift; b.exponent -= int(b_shift);
    uint remainder = a.significand;
    int exponent = a.exponent-b.exponent;
    if (remainder < b.significand) { remainder <<= 1; exponent -= 1; }
    // Exact quotient of (normalized numerator<<26)/denominator, without 64-bit division.
    // Numerator starts in[d,2*d), d<2^24. After subtraction remainder<d; each next
    // shift keeps remainder<2^25. Quotient has exactly27 bits; no U32 overflow/zero divide.
    uint quotient = 0;
    for (uint bit=0; bit<27; bit++) {
        quotient <<= 1;
        if (remainder >= b.significand) { remainder -= b.significand; quotient |= 1; }
        if (bit != 26) remainder <<= 1;
    }
    quotient |= uint(remainder != 0);
    return pack(sign, quotient, exponent);
}
};

kernel void pcu_checked_f32(device const uint* a [[buffer(0)]],
                            device const uint* b [[buffer(1)]],
                            device uint* output [[buffer(2)]],
                            device uint* records [[buffer(3)]],
                            constant uint2& parameters [[buffer(4)]],
                            uint index [[thread_position_in_grid]]) {
    if (index >= parameters.x) return;
    uint code=(parameters.y & 0xffu)-44u, operation=code&3u;
    uint left=a[(parameters.y & 0x10000u)!=0u ? 0u : index], right=b[(parameters.y & 0x20000u)!=0u ? 0u : index];
    Binary arithmetic={code>>2,(parameters.y & 0x100u)!=0u};
    Result result;
    if (((left>>23)&255u)==255u || ((right>>23)&255u)==255u) result=fault(1);
    else if (operation==0) result=arithmetic.add(left,right,false);
    else if (operation==1) result=arithmetic.add(left,right,true);
    else if (operation==2) result=arithmetic.multiply(left,right);
    else result=arithmetic.divide(left,right);
    // Runtime diagnostic ABI differs from the source algorithm's local result tags.
    uint kind=result.status & 0xffu;
    uint status=(kind==1 ? 4u : kind==2 ? 3u : kind==3 ? 1u : kind==4 ? 2u : 0u) | (result.status & 0x100u);
    output[index]=result.bits;
    records[index]=status;
}
