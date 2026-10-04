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
kernel void pcu_relu_backward(device const uint* input [[buffer(0)]],device const uint* upstream [[buffer(1)]],device uint* output [[buffer(2)]],device uint* records [[buffer(3)]],constant uint2& profile [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    if(id>=profile.x)return;
    uint format=(profile.y>>4u)&7u;
    if(format!=0u){
        uint width=format<=2u?2u:1u;
        uint left=(profile.y&0x100u)!=0u?0u:id;
        uint right=(profile.y&0x200u)!=0u?0u:id;
        device const uchar* a_bytes=reinterpret_cast<device const uchar*>(input);
        device const uchar* b_bytes=reinterpret_cast<device const uchar*>(upstream);
        device uchar* out_bytes=reinterpret_cast<device uchar*>(output);
        uint a=uint(a_bytes[width*left]);uint b=uint(b_bytes[width*right]);
        if(width==2u){a|=uint(a_bytes[2u*left+1u])<<8u;b|=uint(b_bytes[2u*right+1u])<<8u;}
        uint status;uint value=backward_low(a,b,format,(profile.y&2u)!=0u,status);
        out_bytes[width*id]=uchar(value&0xffu);
        if(width==2u)out_bytes[2u*id+1u]=uchar(value>>8u);
        records[id]=status;return;
    }
    bool wide=(profile.y&1u)!=0u;
    uint left=(profile.y&0x100u)!=0u?0u:id;
    uint right=(profile.y&0x200u)!=0u?0u:id;
    uint2 a=wide?uint2(input[2u*left],input[2u*left+1u]):uint2(input[left],0u);
    uint2 b=wide?uint2(upstream[2u*right],upstream[2u*right+1u]):uint2(upstream[right],0u);
    uint status;uint2 value=backward_bits(a,b,wide,(profile.y&2u)!=0u,status);
    if(wide){output[2u*id]=value.x;output[2u*id+1u]=value.y;}else output[id]=value.x;
    records[id]=status;
}
