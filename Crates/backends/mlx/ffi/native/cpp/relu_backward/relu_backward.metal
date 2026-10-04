// Independent MLX custom primitive uses the same finite encoding selection algorithm.
// Checked backward selection is encoding arithmetic; both operands are validated even if masked.
#include <metal_stdlib>
using namespace metal;
inline uint2 backward_bits(uint2 input,uint2 upstream,bool wide,bool reject_tiny,thread uint& status) {
    uint mask=wide?0x7ff00000u:0x7f800000u;
    uint top=wide?input.y:input.x;
    uint gradient=wide?upstream.y:upstream.x;
    if((top&mask)==mask||(gradient&mask)==mask){status=4u;return uint2(0u);}
    bool positive=(top&0x80000000u)==0u&&((top&0x7fffffffu)!=0u||(wide&&input.x!=0u));
    uint2 output=positive?upstream:uint2(0u);
    uint selected=wide?output.y:output.x;
    bool nonzero=(selected&0x7fffffffu)!=0u||(wide&&output.x!=0u);
    status=reject_tiny&&nonzero&&(selected&mask)==0u?3u:0u;
    return output;
}

// Formats 1..4 are binary16, bfloat16, OFP8 E4M3FN and E5M2, carried as integers.
inline uint backward_low(uint input,uint upstream,uint format,bool reject_tiny,thread uint& status) {
    uint sign=format<=2u?0x8000u:0x80u;
    uint exponent=format==1u?0x7c00u:format==2u?0x7f80u:format==3u?0x78u:0x7cu;
    uint magnitude=sign-1u;
    bool invalid=format==3u?((input&magnitude)==0x7fu||(upstream&magnitude)==0x7fu):((input&exponent)==exponent||(upstream&exponent)==exponent);
    if(invalid){status=4u;return 0u;}
    uint value=(input&sign)==0u&&(input&magnitude)!=0u?upstream:0u;
    status=reject_tiny&&(value&magnitude)!=0u&&(value&exponent)==0u?3u:0u;
    return value;
}
