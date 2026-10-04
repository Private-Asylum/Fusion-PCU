// Independent fixed-expression control: no PCU IR, generated arithmetic, or header imports.
// All integer bit widths use explicit unsigned32 limbs and a full unsigned product.
#define L ((BITS + 31) / 32)
typedef unsigned int U32;
typedef unsigned long long U64;
struct W { U32 x[L]; };
__device__ U32 mask() { return BITS < 32 ? ((1u << (BITS % 32)) - 1u) : ~0u; }
__device__ W zero() { W a; for (int i=0;i<L;++i) a.x[i]=0; return a; }
__device__ bool negative(W a) { return SIGNED && ((a.x[L-1] >> ((BITS-1)%32)) & 1u); }
__device__ W negate(W a) {
    U64 carry=1;
    for (int i=0;i<L;++i) { U64 n=U64(~a.x[i])+carry; a.x[i]=U32(n); carry=n>>32; }
    a.x[L-1]&=mask(); return a;
}
__device__ int compare(W a,W b) {
    for (int i=L-1;i>=0;--i) if (a.x[i]!=b.x[i]) return a.x[i]>b.x[i] ? 1 : -1;
    return 0;
}
__device__ W limit(bool lower) {
    W a=zero();
    if (!SIGNED) { if (!lower) for(int i=0;i<L;++i) a.x[i]=~0u; }
    else if (lower) a.x[L-1]=1u<<((BITS-1)%32);
    else { for(int i=0;i<L;++i) a.x[i]=~0u; a.x[L-1]&=(1u<<((BITS-1)%32))-1u; }
    a.x[L-1]&=mask(); return a;
}
__device__ W add(W a,W b,U32 &fault) {
    W r; U64 carry=0;
    for(int i=0;i<L;++i) { U64 n=U64(a.x[i])+b.x[i]+carry; r.x[i]=U32(n); carry=n>>32; }
    bool outside=carry || (r.x[L-1]&~mask()); r.x[L-1]&=mask();
    if (SIGNED) { if(negative(a)==negative(b) && negative(r)!=negative(a)) fault=negative(a)?4:3; }
    else if(outside) fault=3;
    return r;
}
__device__ W sub(W a,W b,U32 &fault) {
    W r; U64 borrow=0;
    for(int i=0;i<L;++i) { U64 right=U64(b.x[i])+borrow; r.x[i]=U32(U64(a.x[i])-right); borrow=U64(a.x[i])<right; }
    r.x[L-1]&=mask();
    if (SIGNED) { if(negative(a)!=negative(b) && negative(r)!=negative(a)) fault=negative(a)?4:3; }
    else if(borrow) fault=4;
    return r;
}
__device__ W mul(W a,W b,U32 &fault) {
    bool sign=negative(a)!=negative(b);
    if(SIGNED) { if(negative(a)) a=negate(a); if(negative(b)) b=negate(b); }
    U32 full[2*L]; for(int i=0;i<2*L;++i) full[i]=0;
    for(int i=0;i<L;++i) {
        U64 carry=0;
        for(int j=0;j<L;++j) { U64 n=U64(a.x[i])*b.x[j]+full[i+j]+carry; full[i+j]=U32(n); carry=n>>32; }
        full[i+L]=U32(carry);
    }
    W r; for(int i=0;i<L;++i) r.x[i]=full[i];
    bool outside=(r.x[L-1]&~mask())!=0;
    for(int i=L;i<2*L;++i) outside=outside || full[i]!=0;
    r.x[L-1]&=mask();
    if(SIGNED) {
        W bound=limit(sign); if(sign) bound=negate(bound);
        if(outside || compare(r,bound)>0) fault=sign?4:3;
        if(sign) r=negate(r);
    } else if(outside) fault=3;
    return r;
}
__device__ W load(const volatile unsigned char* p,U32 index) {
    W r=zero(); p+=U64(index)*(BITS/8);
    for(int i=0;i<BITS/8;++i) r.x[i/4]|=U32(p[i])<<(8*(i%4));
    return r;
}
__device__ void store(volatile unsigned char* p,U32 index,W a) {
    p+=U64(index)*(BITS/8);
    for(int i=0;i<BITS/8;++i) p[i]=static_cast<unsigned char>(a.x[i/4]>>(8*(i%4)));
}
__device__ bool checked(U32 op,W a,W b,U32 clamp,U32 &first,W &r) {
    U32 fault=0; r=op==0?add(a,b,fault):op==1?mul(a,b,fault):sub(a,b,fault);
    if(fault) { if(!first) first=fault; if(clamp) r=limit(fault==4); }
    return fault!=0;
}
__device__ void report(U64* status,U32 index,U32 first) { if(first) atomicMin(status,(U64(index)<<32)|first); }
extern "C" __global__ void independent_composed(
    const unsigned char* input,const unsigned char* seed,unsigned char* stage,
    unsigned char* output,U64* status,U32 extent,U32 invocations,U32 clamp,U32 dead) {
    U32 lane=blockIdx.x*blockDim.x+threadIdx.x;
    if(lane>=invocations) return;
    W b=load(seed,0);
    for(U32 id=lane;id<extent;id+=invocations) {
        W original=load(input,id),value; U32 first=0;
        bool fault=checked(0,original,b,clamp,first,value);
        if(fault && !clamp) { report(status,id,first); continue; }
        if(dead) { store(output,id,original); report(status,id,first); continue; }
        store(stage,id,value); value=load(stage,id);
        fault=checked(1,value,original,clamp,first,value);
        if(fault && !clamp) { report(status,id,first); continue; }
        fault=checked(2,value,original,clamp,first,value);
        if(fault && !clamp) { report(status,id,first); continue; }
        store(output,id,value); report(status,id,first);
    }
}
