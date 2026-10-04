// Two independent phases preserve graph operation order before invocation order.
// Add status is immutable during Mul, so racing Mul faults cannot suppress a lower lane.
extern "C" __global__ void independent_literal_add(
    const unsigned char* input,const unsigned char* constant,const unsigned char* uniform,
    unsigned char* output,U64* status,U32 extent,U32 invocations,U32 clamp,U32 dead) {
    U32 id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=extent) return;
    W value; U32 first=0;
    if(checked(0,load(input,id),load(constant,id),0,first,value)) {report(status,id,first);return;}
    store(output,id,value);
}
extern "C" __global__ void independent_literal_mul(
    const unsigned char* input,const unsigned char* constant,const unsigned char* uniform,
    unsigned char* output,U64* status,U32 extent,U32 invocations,U32 clamp,U32 dead) {
    U32 id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=extent || status[0] != ~U64(0)) return;
    W value; U32 first=0;
    if(checked(1,load(output,id),load(uniform,id),0,first,value)) {report(status+1,id,first);return;}
    store(output,id,value);
}
