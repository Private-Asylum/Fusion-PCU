// Audited integer synthesis emitted through MLX fast::metal_kernel.
// Independent MLX numeric qualification is required before admission.
#pragma once
inline constexpr char pcu_mlx_low_binary_header[] = R"PCUMLX(
#include <metal_stdlib>
using namespace metal;
// Integer realization for binary16, bfloat16 and named OFP8 E4M3FN/E5M2.
// Each entry is independently qualified against PCU's rational reference. No half/float ALU.
// IEEE754-2019 4.3.1 ties-even, 7.5(a) destination-precision tininess with unbounded exponent.
// E4M3FN is OFP8 (not IEEE interchange); finite top exponent and reserved NaN are explicit.
struct PcuBinaryResult { uint bits; uint status; };
struct PcuBinaryParts { uint sign; uint significand; int exponent; };
struct PcuBinaryFormat { uint fraction; uint sign_mask; uint max_finite; int bias; };
PcuBinaryResult pcu_ok(uint bits) { return {bits, 0}; }
PcuBinaryResult pcu_fault(uint status) { return {0, status}; }
uint pcu_jam(uint value, uint distance) {
    if (distance == 0) return value;
    if (distance >= 32) return uint(value != 0);
    return (value >> distance) | uint((value & ((1u << distance)-1u)) != 0);
}
bool pcu_increment(uint significand, uint discarded) {
    return discarded > 4u || (discarded == 4u && (significand & 1u) != 0);
}
struct PcuBinaryArithmetic {
    PcuBinaryFormat f;
    uint policy;
    bool clamp;
    PcuBinaryResult range_fault(uint bits,uint status) { return {clamp ? bits : 0u, status | (clamp ? 0x100u : 0u)}; }
    PcuBinaryParts decode(uint bits) {
        uint exponent = (bits & (f.sign_mask-1u)) >> f.fraction;
        uint significand = bits & ((1u << f.fraction)-1u);
        if (exponent != 0) significand |= 1u << f.fraction;
        return {uint((bits & f.sign_mask) != 0), significand,
                exponent == 0 ? 1-f.bias : int(exponent)-f.bias};
    }
    PcuBinaryResult passthrough(uint bits) {
        uint magnitude = bits & (f.sign_mask-1u);
        if (policy == 1 && magnitude != 0 && magnitude < (1u << f.fraction)) return range_fault(bits,2);
        return pcu_ok(bits);
    }
    // Nonzero ext. fraction in{2,3,7,10}, Add/Sub <2^15, product <2^22.
    // Normalization targets fraction+3 (<=13), guarded pcu_jam prevents oversized shifts.
    // Exponents bounded by BF16's [-266,261]; every signed exponent operation fits I32.
    PcuBinaryResult pack(uint sign, uint ext, int exponent) {
        uint normal = 1u << (f.fraction+3u);
        while (ext >= (normal << 1)) { ext = pcu_jam(ext,1); exponent += 1; }
        while (ext < normal) { ext <<= 1; exponent -= 1; }
        uint unbounded = ext >> 3;
        if (pcu_increment(unbounded,ext & 7u)) unbounded += 1;
        int unbounded_exp = exponent + int(unbounded == (1u << (f.fraction+1u)));
        int minimum = 1-f.bias;
        bool tiny = unbounded_exp < minimum;
        if (exponent < minimum) { ext = pcu_jam(ext,uint(minimum-exponent)); exponent = minimum; }
        uint discarded = ext & 7u, significand = ext >> 3;
        bool inexact = discarded != 0;
        if (pcu_increment(significand,discarded)) significand += 1;
        if (significand >= (1u << (f.fraction+1u))) { significand >>= 1; exponent += 1; }
        int maximum = int(f.max_finite >> f.fraction)-f.bias;
        uint max_sig = (1u << f.fraction) | (f.max_finite & ((1u << f.fraction)-1u));
        // Rounded-field comparison admits E4M3FN exp1111 frac000..110, excludes frac111.
        if (exponent > maximum || (exponent == maximum && significand > max_sig)) return range_fault(uint(sign != 0)*f.sign_mask | f.max_finite,3);
        bool subnormal = significand < (1u << f.fraction);
        uint magnitude = subnormal ? significand :
            (uint(exponent+f.bias) << f.fraction) | (significand & ((1u << f.fraction)-1u));
        bool reject = policy == 0 ? (tiny && inexact) :
            policy == 1 ? ((subnormal && significand != 0) || (tiny && inexact)) : false;
        uint bits = uint(sign != 0)*f.sign_mask | magnitude;
        return reject ? range_fault(bits,2) : pcu_ok(bits);
    }
    PcuBinaryResult add(uint left, uint right, bool subtract) {
        PcuBinaryParts a = decode(left), b = decode(right); b.sign ^= uint(subtract);
        if (a.significand == 0) {
            if (b.significand != 0) return passthrough(uint(b.sign != 0)*f.sign_mask | (right & (f.sign_mask-1u)));
            return pcu_ok(uint(a.sign == b.sign && a.sign != 0)*f.sign_mask);
        }
        if (b.significand == 0) return passthrough(left);
        if (a.exponent < b.exponent) { PcuBinaryParts temp = a; a = b; b = temp; }
        uint x = a.significand << 3, y = pcu_jam(b.significand << 3,uint(a.exponent-b.exponent));
        uint sign, ext;
        if (a.sign == b.sign) { sign = a.sign; ext = x+y; }
        else if (x >= y) { sign = a.sign; ext = x-y; }
        else { sign = b.sign; ext = y-x; }
        return ext == 0 ? pcu_ok(0) : pack(sign,ext,a.exponent);
    }
    PcuBinaryResult multiply(uint left, uint right) {
        PcuBinaryParts a = decode(left), b = decode(right); uint sign = a.sign ^ b.sign;
        if (a.significand == 0 || b.significand == 0) return pcu_ok(uint(sign != 0)*f.sign_mask);
        uint product = a.significand*b.significand; // <=22 exact bits, no overflow.
        int top = 31-int(clz(product)), target = int(f.fraction)+3;
        int exponent = a.exponent+b.exponent+top-2*int(f.fraction);
        uint ext = top > target ? pcu_jam(product,uint(top-target)) : product << uint(target-top);
        return pack(sign,ext,exponent);
    }
    PcuBinaryResult divide(uint left, uint right) {
        PcuBinaryParts a = decode(left), b = decode(right); uint sign = a.sign ^ b.sign;
        if (b.significand == 0) return pcu_fault(4);
        if (a.significand == 0) return pcu_ok(uint(sign != 0)*f.sign_mask);
        uint a_shift = uint(clz(a.significand))-(31u-f.fraction);
        uint b_shift = uint(clz(b.significand))-(31u-f.fraction);
        a.significand <<= a_shift; a.exponent -= int(a_shift);
        b.significand <<= b_shift; b.exponent -= int(b_shift);
        uint remainder = a.significand; int exponent = a.exponent-b.exponent;
        if (remainder < b.significand) { remainder <<= 1; exponent -= 1; }
        // d<2^11, remainder<2*d before subtraction, next shift<2^12.
        // Quotient has fraction+4<=14 bits; no U32 overflow or integer division.
        uint quotient = 0, count = f.fraction+4u;
        for (uint bit=0; bit<count; bit++) {
            quotient <<= 1;
            if (remainder >= b.significand) { remainder -= b.significand; quotient |= 1; }
            if (bit+1u != count) remainder <<= 1;
        }
        quotient |= uint(remainder != 0);
        return pack(sign,quotient,exponent);
    }
    PcuBinaryResult evaluate(uint left, uint right, uint operation) {
        if ((left & (f.sign_mask-1u)) > f.max_finite || (right & (f.sign_mask-1u)) > f.max_finite) return pcu_fault(1);
        if (operation == 0) return add(left,right,false);
        if (operation == 1) return add(left,right,true);
        if (operation == 2) return multiply(left,right);
        return divide(left,right);
    }
};
uint pcu_diagnostic(uint status) {
    uint kind=status & 0xffu;
    return (kind==1 ? 4u : kind==2 ? 3u : kind==3 ? 1u : kind==4 ? 2u : 0u) | (status & 0x100u);
}
)PCUMLX";
