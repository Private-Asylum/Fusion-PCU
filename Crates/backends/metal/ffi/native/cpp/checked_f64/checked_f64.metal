#include <metal_stdlib>
using namespace metal;
// IEEE 754-2019 4.3.1 nearest ties even and 7.5(a) tininess after precision rounding.
// PCU additionally rejects nonfinite inputs and reports faults instead of IEEE defaults.
// No floating instructions or ulong division. Unsigned shifts are always guarded.
struct Result { ulong bits; uint status; };
struct Parts { ulong sign; ulong sig; int exp; };
inline Result ok(ulong bits) { return {bits, 0}; }
inline Result fault(uint status) { return {0, status}; }
inline Result range_fault(ulong bits,uint status,uint policy) { return {(policy & 0x100u)!=0u ? bits : 0ul,status | (policy & 0x100u)}; }
inline Parts decode(ulong bits) {
    uint exp = uint((bits >> 52) & 2047ul);
    ulong sig = bits & 0xffffffffffffful;
    if (exp != 0) sig |= 0x10000000000000ul;
    return {bits >> 63, sig, exp == 0 ? -1022 : int(exp) - 1023};
}
inline ulong jam(ulong value, uint distance) {
    if (distance == 0) return value;
    if (distance >= 64) return ulong(value != 0);
    return (value >> distance) | ulong((value & ((1ul << distance)-1ul)) != 0);
}
inline bool increment(ulong sig, ulong discarded) {
    return discarded > 4ul || (discarded == 4ul && (sig & 1ul) != 0);
}
inline Result passthrough(ulong bits, uint policy) {
    ulong magnitude = bits & 0x7ffffffffffffffful;
    if ((policy & 3u) == 1 && magnitude != 0 && magnitude < 0x10000000000000ul) return range_fault(bits,3,policy);
    return ok(bits);
}
// Add extended significands <2^57; normalized Mul/Div <2^56. Exponents remain
// in [-2148,2101], including gradual packing; signed exponent math never overflows.
inline Result pack(ulong sign, ulong ext, int exponent, uint policy) {
    while (ext >= (1ul << 56)) { ext = jam(ext, 1); exponent += 1; }
    while (ext < (1ul << 55)) { ext <<= 1; exponent -= 1; }
    ulong unbounded_sig = ext >> 3;
    if (increment(unbounded_sig, ext & 7ul)) unbounded_sig += 1;
    int unbounded_exp = exponent + int(unbounded_sig == (1ul << 53));
    bool tiny = unbounded_exp < -1022;
    if (exponent < -1022) { ext = jam(ext, uint(-1022-exponent)); exponent = -1022; }
    ulong discarded = ext & 7ul;
    ulong sig = ext >> 3;
    bool inexact = discarded != 0;
    if (increment(sig, discarded)) sig += 1;
    bool subnormal = false;
    ulong bits;
    if (exponent == -1022 && sig < (1ul << 52)) {
        bits = (sign << 63) | sig;
        subnormal = sig != 0;
    } else {
        if (sig >= (1ul << 53)) { sig >>= 1; exponent += 1; }
        if (exponent > 1023) return range_fault((sign << 63) | 0x7feffffffffffffful,1,policy);
        bits = (sign << 63) | (ulong(exponent+1023) << 52) | (sig & 0xffffffffffffful);
    }
    uint underflow=policy & 3u;
    bool reject = underflow == 0 ? (tiny && inexact) : underflow == 1 ? (subnormal || (tiny && inexact)) : false;
    return reject ? range_fault(bits,3,policy) : ok(bits);
}
inline Result add(ulong left, ulong right, bool subtract, uint policy) {
    Parts a = decode(left), b = decode(right); b.sign ^= ulong(subtract);
    if (a.sig == 0) {
        if (b.sig != 0) return passthrough((b.sign << 63) | (right & 0x7ffffffffffffffful), policy);
        return ok(ulong(a.sign == b.sign) * (a.sign << 63));
    }
    if (b.sig == 0) return passthrough(left, policy);
    if (a.exp < b.exp) { Parts t = a; a = b; b = t; }
    ulong x = a.sig << 3, y = jam(b.sig << 3, uint(a.exp-b.exp));
    ulong sign, ext;
    if (a.sign == b.sign) { sign = a.sign; ext = x+y; }
    else if (x >= y) { sign = a.sign; ext = x-y; }
    else { sign = b.sign; ext = y-x; }
    return ext == 0 ? ok(0) : pack(sign, ext, a.exp, policy);
}
inline Result multiply(ulong left, ulong right, uint policy) {
    Parts a = decode(left), b = decode(right);
    ulong sign = a.sign ^ b.sign;
    if (a.sig == 0 || b.sig == 0) return ok(sign << 63);
    // Split each <2^53 significand into 27 low/26 high bits. Every partial product
    // and cross sum fits U64 (<2^54). Unsigned low-limb additions wrap by definition;
    // each carry is retained in high, so all <=106 product bits survive.
    ulong a0 = a.sig & ((1ul << 27)-1ul), a1 = a.sig >> 27;
    ulong b0 = b.sig & ((1ul << 27)-1ul), b1 = b.sig >> 27;
    ulong p00 = a0*b0, p01 = a0*b1+a1*b0, p11 = a1*b1;
    ulong low = p00, high = (p01 >> 37) + (p11 >> 10);
    ulong old = low; low += p01 << 27; high += ulong(low < old);
    old = low; low += p11 << 54; high += ulong(low < old);
    uint top = high != 0 ? 127u-uint(clz(high)) : 63u-uint(clz(low));
    int exponent = a.exp+b.exp+int(top)-104;
    ulong ext;
    if (top > 55) {
        uint distance = top-55; // 1..50: guarded limb shifts strictly below64.
        ext = (high << (64u-distance)) | (low >> distance);
        ext |= ulong((low & ((1ul << distance)-1ul)) != 0);
    } else { ext = low << (55u-top); }
    return pack(sign, ext, exponent, policy);
}
inline Result divide(ulong left, ulong right, uint policy) {
    Parts a = decode(left), b = decode(right);
    ulong sign = a.sign ^ b.sign;
    if (b.sig == 0) return fault(2);
    if (a.sig == 0) return ok(sign << 63);
    uint a_shift = uint(clz(a.sig))-11u, b_shift = uint(clz(b.sig))-11u;
    a.sig <<= a_shift; a.exp -= int(a_shift);
    b.sig <<= b_shift; b.exp -= int(b_shift);
    ulong remainder = a.sig;
    int exponent = a.exp-b.exp;
    if (remainder < b.sig) { remainder <<= 1; exponent -= 1; }
    // Normalized denominator <2^53 and numerator in[d,2*d). After subtraction
    // remainder<d, every next shift<2^54. Exactly56 quotient bits fit U64.
    ulong quotient = 0;
    for (uint bit=0; bit<56; bit++) {
        quotient <<= 1;
        if (remainder >= b.sig) { remainder -= b.sig; quotient |= 1; }
        if (bit != 55) remainder <<= 1;
    }
    quotient |= ulong(remainder != 0);
    return pack(sign, quotient, exponent, policy);
}
kernel void pcu_checked_f64(device const ulong* a [[buffer(0)]],
                            device const ulong* b [[buffer(1)]],
                            device ulong* output [[buffer(2)]],
                            device uint* records [[buffer(3)]],
                            constant uint2& parameters [[buffer(4)]],
                            uint index [[thread_position_in_grid]]) {
    if (index >= parameters.x) return;
    uint code = (parameters.y & 0xffu)-32u, operation = code & 3u, policy = (code >> 2) | (parameters.y & 0x100u);
    ulong left = a[(parameters.y & 0x10000u)!=0u ? 0u : index], right = b[(parameters.y & 0x20000u)!=0u ? 0u : index];
    Result result;
    if (((left >> 52) & 2047ul) == 2047ul || ((right >> 52) & 2047ul) == 2047ul) result = fault(4);
    else if (operation == 0) result = add(left,right,false,policy);
    else if (operation == 1) result = add(left,right,true,policy);
    else if (operation == 2) result = multiply(left,right,policy);
    else result = divide(left,right,policy);
    output[index] = result.bits;
    records[index] = result.status;
}
