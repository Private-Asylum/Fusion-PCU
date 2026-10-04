#include <metal_stdlib>
using namespace metal;
// Exact U32 representation conversion; no native floating operation.
struct Result { uint bits; uint status; };
Result ok(uint bits) { return Result{bits,0}; }
uint jam(uint value, uint distance) {
    if (distance == 0) return value;
    if (distance >= 32) return uint(value != 0);
    return (value >> distance) | uint((value & ((1u << distance)-1u)) != 0);
}
bool increment(uint significand, uint discarded) {
    return discarded > 4u || (discarded == 4u && (significand & 1u) != 0);
}
Result passthrough(uint bits,uint policy) {
    uint magnitude = bits & 0x7fffffffu;
    if (policy == 1 && magnitude != 0 && magnitude < 0x800000u) return Result{bits, 3};
    return ok(bits);
}
// Precondition: ext is nonzero, normalized with its top bit26, and <2^27.
// Left normalization stops at bit26, so shifts/additions never overflow a U32.
// Conversion exponents remain in [-1074,1023], including gradual packing; I32 never overflows.
Result pack(uint sign, uint ext, int exponent,uint policy) {
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
        if (exponent > 127) return Result{(sign << 31) | 0x7f7fffffu, 1};
        bits = (sign << 31) | (uint(exponent+127) << 23) | (significand & 0x7fffffu);
    }
    bool reject = policy == 0 ? (tiny && inexact) : policy == 1 ? (subnormal || (tiny && inexact)) : false;
    return reject ? Result{bits, 3} : ok(bits);
}

// Nonzero significand has highest bit0..52. Its normalized extended F32 value
// has top bit26 and is <2^27. All residual shifts are0..26; zero is branched.
uint precision_jam(uint2 significand,int highest) {
    if(highest<=26) return significand.x << uint(26-highest);
    uint distance=uint(highest-26); //1..26, both complementary shifts strictly below32.
    uint extended=(significand.y << (32-distance)) | (significand.x >> distance);
    return extended | uint((significand.x & ((1u << distance)-1u))!=0);
}
Result narrow(uint2 bits,uint policy) {
    uint exponent=(bits.y>>20)&2047u;
    uint sign=bits.y>>31;
    if(exponent==2047u) return Result{0, 4};
    uint2 significand=uint2(bits.x,bits.y&0xfffffu);
    if(exponent!=0) significand.y |= 0x100000u;
    if((significand.x|significand.y)==0) return ok(sign<<31);
    int highest=significand.y!=0 ? 32+(31-int(clz(significand.y))) : (31-int(clz(significand.x)));
    int unbiased=(exponent==0 ? -1022 : int(exponent)-1023)+highest-52;
    return pack(sign,precision_jam(significand,highest),unbiased,policy);
}
uint3 widen(uint bits) {
    uint exponent=(bits>>23)&255u;
    uint sign=bits>>31;
    uint significand=bits&0x7fffffu;
    if(exponent==255u) return uint3(0,0,4);
    if(exponent==0 && significand==0) return uint3(0,sign<<31,0);
    int unbiased=exponent==0 ? -126 : int(exponent)-127;
    if(exponent==0) {
        // Nonzero F32 subnormal: shift1..23. Top bit stops at23; no U32 overflow.
        uint distance=uint(23-(31-int(clz(significand))));
        significand <<= distance; unbiased-=int(distance);
    }
    significand &= 0x7fffffu;
    // Every nonzero finite F32 value is a normal F64 value: exponent[-149,127].
    return uint3(significand<<29,(sign<<31)|(uint(unbiased+1023)<<20)|(significand>>3),0);
}

kernel void pcu_checked_conversion(device const uint* input [[buffer(0)]],device const uint* unused [[buffer(1)]],device uint* output [[buffer(2)]],device uint* records [[buffer(3)]],constant uint2& profile [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    if(id>=profile.x)return;
    uint operation=profile.y;uint source=(operation&0x10000u)!=0u?0u:id;
    uint status;
    if((operation&1u)==0u){uint3 result=widen(input[source]);output[2u*id]=result.x;output[2u*id+1u]=result.y;status=result.z;}
    else{Result result=narrow(uint2(input[2u*source],input[2u*source+1u]),(operation>>1u)&3u);output[id]=result.bits;status=result.status;}
    if((operation&0x100u)!=0u&&(status==1u||status==3u))status|=0x100u;
    records[id]=status;
}
