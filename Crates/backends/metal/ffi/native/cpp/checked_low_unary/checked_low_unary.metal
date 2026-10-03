#include <metal_stdlib>
using namespace metal;
// Exact finite encoding selection over binary16/bfloat16 and named OFP8 formats.
// No rounding, native floating arithmetic, signed arithmetic or variable shifts occur.
// Nonfinite operands fault before sign selection; ReLU maps every nonpositive input to +0.
// RejectSubnormalResult examines the output; exact subnormals have no IEEE inexact-tiny fault.
struct UnaryResult { uint bits; uint diagnostic; };
UnaryResult unary(uint bits, uint fraction, uint sign, uint maximum, uint operation) {
    uint magnitude = bits & (sign-1u);
    if (magnitude > maximum) return {0u,4u};
    uint output = (operation & 1u) == 0u ? bits ^ sign :
        ((bits & sign) != 0u || magnitude == 0u ? 0u : bits);
    uint out_magnitude = output & (sign-1u);
    if (((operation >> 1u) & 3u) == 1u && out_magnitude != 0u && out_magnitude < (1u << fraction))
        return (operation & 0x100u) != 0u ? UnaryResult{output,0x103u} : UnaryResult{0u,3u};
    return {output,0u};
}
#define PCU_UNARY(NAME, TYPE, FRACTION, SIGN, MAXIMUM) \
kernel void NAME(device const TYPE* input [[buffer(0)]], device const TYPE* unused [[buffer(1)]], \
                 device TYPE* output [[buffer(2)]], device uint* records [[buffer(3)]], \
                 constant uint2& parameters [[buffer(4)]], uint index [[thread_position_in_grid]]) { \
    if (index >= parameters.x) return; \
    UnaryResult result = unary(uint(input[(parameters.y & 0x10000u) != 0u ? 0u : index]),FRACTION,SIGN,MAXIMUM,parameters.y); \
    output[index] = TYPE(result.bits); records[index] = result.diagnostic; \
}
PCU_UNARY(pcu_unary_f16, ushort, 10u, 0x8000u, 0x7bffu)
PCU_UNARY(pcu_unary_bf16, ushort, 7u, 0x8000u, 0x7f7fu)
PCU_UNARY(pcu_unary_e4m3fn, uchar, 3u, 0x80u, 0x7eu)
PCU_UNARY(pcu_unary_e5m2, uchar, 2u, 0x80u, 0x7bu)
