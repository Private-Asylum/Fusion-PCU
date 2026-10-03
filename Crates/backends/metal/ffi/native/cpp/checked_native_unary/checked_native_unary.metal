#include <metal_stdlib>
using namespace metal;
// Exact IEEE encoding selection; F64 is two U32 limbs. No floating instruction,
// signed overflow, division or variable shift occurs in this profile.
struct Unary32 { uint bits; uint diagnostic; };
struct Unary64 { uint2 bits; uint diagnostic; };
Unary32 unary32(uint bits, uint operation) {
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f7fffffu) return {0u,4u};
    uint result = (operation & 1u) == 0u ? bits ^ 0x80000000u :
        ((bits & 0x80000000u) != 0u || magnitude == 0u ? 0u : bits);
    uint out_magnitude = result & 0x7fffffffu;
    bool rejected = ((operation >> 1u) & 3u) == 1u &&
        out_magnitude != 0u && out_magnitude < 0x00800000u;
    if (rejected) return (operation & 0x100u) != 0u ? Unary32{result,0x103u} : Unary32{0u,3u};
    return {result,0u};
}
Unary64 unary64(uint2 bits, uint operation) {
    uint high = bits.y & 0x7fffffffu;
    if ((high & 0x7ff00000u) == 0x7ff00000u) return {uint2(0u),4u};
    uint2 result = bits;
    if ((operation & 1u) == 0u) result.y ^= 0x80000000u;
    else if ((bits.y & 0x80000000u) != 0u || (high | bits.x) == 0u) result = uint2(0u);
    uint out_high = result.y & 0x7fffffffu;
    bool rejected = ((operation >> 1u) & 3u) == 1u &&
        (out_high | result.x) != 0u && out_high < 0x00100000u;
    if (rejected) return (operation & 0x100u) != 0u ? Unary64{result,0x103u} : Unary64{uint2(0u),3u};
    return {result,0u};
}
#define PCU_NATIVE_UNARY(NAME, TYPE, RESULT, FUNCTION) \
kernel void NAME(device const TYPE* input [[buffer(0)]], device const TYPE* unused [[buffer(1)]], \
                 device TYPE* output [[buffer(2)]], device uint* records [[buffer(3)]], \
                 constant uint2& parameters [[buffer(4)]], uint index [[thread_position_in_grid]]) { \
    if (index >= parameters.x) return; \
    RESULT result = FUNCTION(input[(parameters.y & 0x10000u) != 0u ? 0u : index],parameters.y); \
    output[index] = result.bits; records[index] = result.diagnostic; \
}
PCU_NATIVE_UNARY(pcu_unary_f32, uint, Unary32, unary32)
PCU_NATIVE_UNARY(pcu_unary_f64, uint2, Unary64, unary64)
