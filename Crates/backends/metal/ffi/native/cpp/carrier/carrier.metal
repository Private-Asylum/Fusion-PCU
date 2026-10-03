#include <metal_stdlib>
using namespace metal;
// No scalar interpretation: one invocation owns all initialized bytes of one logical value.
// Admission bounds count*width <= UINT32_MAX and width in {1,2,4,8,16,32,64}.
kernel void pcu_carrier(device const uchar* input [[buffer(0)]],
    device const uchar* unused [[buffer(1)]], device uchar* output [[buffer(2)]],
    device uint* records [[buffer(3)]], constant uint2& config [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= config.x) return;
    uint width = config.y & 255u;
    uint source = (config.y & 256u) != 0u ? 0u : id * width;
    uint target = id * width;
    for (uint byte = 0u; byte < width; ++byte) output[target + byte] = input[source + byte];
    records[id] = 0u;
}
