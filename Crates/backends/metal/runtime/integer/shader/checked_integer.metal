#include <metal_stdlib>
using namespace metal;
// WIDTH and SIGNED are bounded literals supplied by the cold Rust dtype gate.
#define LIMBS ((PCU_WIDTH + 31u) / 32u)
constant uint BYTES = PCU_WIDTH / 8u;
constant uint MASK = PCU_MASK;
constant uint SIGN = PCU_SIGN;
void load_value(device const uchar* input,uint index,thread uint* value){
  for(uint limb=0;limb<LIMBS;++limb){
    uint word=0;
    for(uint byte=0;byte<4u && limb*4u+byte<BYTES;++byte)
      word |= uint(input[index*BYTES+limb*4u+byte]) << (byte*8u);
    value[limb]=word;
  }
}
void store_value(device uchar* output,uint index,thread const uint* value){
  for(uint limb=0;limb<LIMBS;++limb)
    for(uint byte=0;byte<4u && limb*4u+byte<BYTES;++byte)
      output[index*BYTES+limb*4u+byte]=uchar(value[limb]>>(byte*8u));
}
void negate(thread uint* value){
  uint carry=1u;
  for(uint limb=0;limb<LIMBS;++limb){uint v=~value[limb];uint next=v+carry;
    carry=uint(next<v);value[limb]=next;}
  value[LIMBS-1u]&=MASK;
}
void multiply_word(uint a,uint b,thread uint& low,thread uint& high){
  // Four exact16-bit partial products; each intermediate is bounded below2^32.
  uint a0=a&0xffffu,a1=a>>16u,b0=b&0xffffu,b1=b>>16u;
  uint w0=a0*b0,t=a1*b0+(w0>>16u),w1=t&0xffffu,w2=t>>16u;
  w1=a0*b1+w1;low=(w1<<16u)|(w0&0xffffu);high=a1*b1+w2+(w1>>16u);
}
void add_product_word(thread uint* product,uint index,uint word){
  while(word!=0u && index<2u*LIMBS){uint previous=product[index];
    product[index]=previous+word;word=uint(product[index]<previous);++index;}
}
kernel void pcu_checked_integer(device const uchar* left [[buffer(0)]],
 device const uchar* right [[buffer(1)]],device uchar* output [[buffer(2)]],
 device uint* records [[buffer(3)]],constant uint2& profile [[buffer(4)]],
 uint id [[thread_position_in_grid]]){
  if(id>=profile.x)return;
  uint a[LIMBS],b[LIMBS],result[LIMBS];
  load_value(left,(profile.y&(1u<<16u))!=0u?0u:id,a);
  load_value(right,(profile.y&(1u<<17u))!=0u?0u:id,b);
  bool an=PCU_SIGNED!=0u && (a[LIMBS-1u]&SIGN)!=0u;
  bool bn=PCU_SIGNED!=0u && (b[LIMBS-1u]&SIGN)!=0u;
  uint operation=profile.y&0xffu,fault=0u;bool negative_limit=false;
  if(operation==3u){for(uint j=0;j<LIMBS;++j)result[j]=a[j];}
  else if(operation<2u){
    uint carry=0u;
    for(uint j=0;j<LIMBS;++j){
      if(operation==0u){uint sum=a[j]+b[j],next=sum+carry;
        carry=uint(sum<a[j] || next<sum);result[j]=next;}
      else {uint delta=a[j]-b[j],next=delta-carry;
        carry=uint(a[j]<b[j] || delta<carry);result[j]=next;}
    }
    bool excess=(result[LIMBS-1u]&~MASK)!=0u;
    result[LIMBS-1u]&=MASK;
    bool rn=(result[LIMBS-1u]&SIGN)!=0u;
    if(PCU_SIGNED!=0u){
      if(operation==0u?an==bn && rn!=an:an!=bn && rn!=an){fault=an?3u:1u;negative_limit=an;}
    }else if(operation==0u?(carry!=0u || excess):carry!=0u){fault=operation==0u?1u:3u;}
  }else{
    uint aa[LIMBS],bb[LIMBS],product[2u*LIMBS];
    for(uint j=0;j<LIMBS;++j){aa[j]=a[j];bb[j]=b[j];}
    if(an)negate(aa);if(bn)negate(bb);
    for(uint j=0;j<2u*LIMBS;++j)product[j]=0u;
    for(uint j=0;j<LIMBS;++j)for(uint k=0;k<LIMBS;++k){
      uint low,high;multiply_word(aa[j],bb[k],low,high);
      add_product_word(product,j+k,low);add_product_word(product,j+k+1u,high);
    }
    bool excess=(product[LIMBS-1u]&~MASK)!=0u;
    for(uint j=LIMBS;j<2u*LIMBS;++j)excess=excess || product[j]!=0u;
    for(uint j=0;j<LIMBS;++j)result[j]=product[j];
    result[LIMBS-1u]&=MASK;
    bool negative=an!=bn;
    if(PCU_SIGNED!=0u){
      uint limit=negative?SIGN:SIGN-1u;
      excess=excess || result[LIMBS-1u]>limit;
      if(negative && result[LIMBS-1u]==SIGN)
        for(uint j=0;j+1u<LIMBS;++j)excess=excess || result[j]!=0u;
      if(negative)negate(result);
      negative_limit=negative;
    }
    if(excess)fault=PCU_SIGNED!=0u && negative?3u:1u;
  }
  bool clamp=(profile.y&(1u<<8u))!=0u;
  if(fault!=0u){
    for(uint j=0;j<LIMBS;++j)result[j]=0u;
    if(clamp){
      if(PCU_SIGNED!=0u){
        if(negative_limit)result[LIMBS-1u]=SIGN;
        else {for(uint j=0;j<LIMBS;++j)result[j]=0xffffffffu;result[LIMBS-1u]=SIGN-1u;}
      }else if(fault==1u){for(uint j=0;j<LIMBS;++j)result[j]=0xffffffffu;result[LIMBS-1u]=MASK;}
    }
  }
  store_value(output,id,result);
  records[id]=fault==0u?0u:(fault|(clamp?0x100u:0u));
}
