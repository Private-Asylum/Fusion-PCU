// Independent decoded dyadics and binary64 midpoint search. No PCU helpers.
using U32=unsigned int;
using U64=unsigned long long;
__device__ U32 load(const unsigned char* bytes,U32 index) {
    bytes+=U64(index)*BYTES; U32 value=bytes[0];
    if(BYTES==2) value|=U32(bytes[1])<<8;
    return value;
}
__device__ void store(unsigned char* bytes,U32 index,U32 value) {
    bytes+=U64(index)*BYTES;bytes[0]=static_cast<unsigned char>(value);
    if(BYTES==2)bytes[1]=static_cast<unsigned char>(value>>8);
}
__device__ double unpack(U32 bits) {
    U32 magnitude=bits&(SIGN-1);U32 exponent=magnitude>>FRACTION;
    U32 significand=(magnitude&((1u<<FRACTION)-1))|(exponent?1u<<FRACTION:0);
    int scale=int(exponent?exponent:1)-BIAS-FRACTION;
    U64 power_bits=U64(scale+1023)<<52;
    double value=double(significand)*__longlong_as_double(static_cast<long long>(power_bits));
    return bits&SIGN?-value:value;
}
__device__ U32 checked(U32 a,U32 b,bool multiply,U32 &result) {
    if((a&(SIGN-1))>MAXIMUM||(b&(SIGN-1))>MAXIMUM)return 5;
    double left=unpack(a),right=unpack(b);
    double exact=multiply?left*right:left+right;
    U64 exact_bits=static_cast<U64>(__double_as_longlong(exact));
    double magnitude=__longlong_as_double(static_cast<long long>(exact_bits&~(U64(1)<<63)));
    double maximum=unpack(MAXIMUM),previous=unpack(MAXIMUM-1);
    double overflow=maximum+(maximum-previous)/2.0;
    if(magnitude>overflow||(magnitude==overflow&&(MAXIMUM&1)))return 3;
    U32 low=0,high=MAXIMUM;
    while(low<high) {U32 middle=low+(high-low+1)/2;if(unpack(middle)<=magnitude)low=middle;else high=middle-1;}
    if(low<MAXIMUM) {double midpoint=(unpack(low)+unpack(low+1))/2.0;
        if(magnitude>midpoint||(magnitude==midpoint&&(low&1)))++low;}
    result=((exact_bits>>63)?SIGN:0)|low;
    double tiny_boundary=unpack(1u<<FRACTION)-unpack(1)/4.0;
    bool tiny_inexact=exact!=0.0&&magnitude<tiny_boundary&&exact!=unpack(result);
    bool subnormal=low!=0&&low<(1u<<FRACTION);
    if((POLICY==0&&tiny_inexact)||(POLICY==2&&(tiny_inexact||subnormal)))return 4;
    return 0;
}
__device__ void report(U64* status,U32 index,U32 fault) {
    if(fault)atomicMin(status,(U64(index)<<32)|fault);
}
extern "C" __global__ void independent_literal_add(
    const unsigned char* input,const unsigned char* constant,const unsigned char* uniform,
    unsigned char* output,U64* status,U32 extent,U32 invocations,U32 clamp,U32 dead) {
    U32 id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=extent)return;
    U32 result=0;U32 fault=checked(load(input,id),load(constant,id),false,result);
    if(fault){report(status,id,fault);return;}store(output,id,result);
}
extern "C" __global__ void independent_literal_mul(
    const unsigned char* input,const unsigned char* constant,const unsigned char* uniform,
    unsigned char* output,U64* status,U32 extent,U32 invocations,U32 clamp,U32 dead) {
    U32 id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=extent||status[0]!=~U64(0))return;
    U32 result=0;U32 fault=checked(load(output,id),load(uniform,id),true,result);
    if(fault){report(status+1,id,fault);return;}store(output,id,result);
}
