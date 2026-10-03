#include <metal_stdlib>
using namespace metal;
#define LIMBS ((PCU_WIDTH+31u)/32u)
constant uint BYTES=PCU_WIDTH/8u;
void load_value(device const uchar* input,uint id,thread uint* value){
 for(uint j=0;j<LIMBS;++j){uint word=0u;for(uint k=0;k<4u && j*4u+k<BYTES;++k)
   word|=uint(input[id*BYTES+j*4u+k])<<(k*8u);value[j]=word;}
}
void store_value(device uchar* output,uint id,thread const uint* value){
 for(uint j=0;j<LIMBS;++j)for(uint k=0;k<4u && j*4u+k<BYTES;++k)
   output[id*BYTES+j*4u+k]=uchar(value[j]>>(k*8u));
}
void negate(thread uint* value){uint carry=1u;for(uint j=0;j<LIMBS;++j){uint v=~value[j],n=v+carry;carry=uint(n<v);value[j]=n;}value[LIMBS-1u]&=PCU_MASK;}
uint compare(thread const uint* a,thread const uint* b){
 for(uint j=LIMBS;j>0u;--j){if(a[j-1u]>b[j-1u])return 1u;if(a[j-1u]<b[j-1u])return 0u;}return 1u;
}
void subtract(thread uint* a,thread const uint* b){uint borrow=0u;for(uint j=0;j<LIMBS;++j){uint d=a[j]-b[j],n=d-borrow;borrow=uint(a[j]<b[j]||d<borrow);a[j]=n;}a[LIMBS-1u]&=PCU_MASK;}
kernel void pcu_checked_div_rem(device const uchar* left [[buffer(0)]],device const uchar* right [[buffer(1)]],
 device uchar* packed [[buffer(2)]],device uint* records [[buffer(3)]],constant uint2& profile [[buffer(4)]],uint id [[thread_position_in_grid]]){
 if(id>=profile.x)return;
 uint a[LIMBS],b[LIMBS],q[LIMBS],r[LIMBS];
 load_value(left,PCU_LEFT_SCALAR!=0u?0u:id,a);load_value(right,PCU_RIGHT_SCALAR!=0u?0u:id,b);
 bool an=PCU_SIGNED!=0u&&(a[LIMBS-1u]&PCU_SIGN)!=0u,bn=PCU_SIGNED!=0u&&(b[LIMBS-1u]&PCU_SIGN)!=0u;
 if(an)negate(a);if(bn)negate(b);
 uint divisor=0u;for(uint j=0;j<LIMBS;++j){divisor|=b[j];q[j]=0u;r[j]=0u;}
 uint fault=divisor==0u?2u:0u;
 bool minimum=a[LIMBS-1u]==PCU_SIGN,minus_one=b[0]==1u&&bn;
 for(uint j=0;j<LIMBS;++j){if(j+1u<LIMBS)minimum=minimum&&a[j]==0u;if(j>0u)minus_one=minus_one&&b[j]==0u;}
 if(fault==0u&&an&&minimum&&minus_one)fault=5u;
 if(fault==0u){
   for(uint bit=PCU_WIDTH;bit>0u;--bit){uint index=bit-1u,carry=(a[index/32u]>>(index%32u))&1u;
     for(uint j=0;j<LIMBS;++j){uint next=r[j]>>31u;r[j]=(r[j]<<1u)|carry;carry=next;}
     // Narrow formats retain their bounded logical high bit; the extra carry participates
     // in the unsigned comparison before subtraction, so no shifted bit is lost.
     uint excess=carry | uint((r[LIMBS-1u]&~PCU_MASK)!=0u);r[LIMBS-1u]&=PCU_MASK;
     if(excess!=0u||compare(r,b)!=0u){subtract(r,b);q[index/32u]|=1u<<(index%32u);}
   }
   if(an!=bn)negate(q);if(an)negate(r);
 }
 store_value(packed,id,q);store_value(packed+profile.x*BYTES,id,r);records[id]=fault;
}
