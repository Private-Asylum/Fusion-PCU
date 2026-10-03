// Independent MLX-owned integer realization; no MetalSession dispatch or native signed ALU.
#pragma once
inline constexpr char pcu_mlx_integer_header[] = R"PCUMLX(
#include <metal_stdlib>
using namespace metal;
#define LIMBS ((PCU_WIDTH + 31u) / 32u)
constant uint MASK = PCU_MASK;
constant uint SIGN = PCU_SIGN;
template<typename T>
void load_value(device const T* input,uint index,thread uint* value){
  for(uint limb=0;limb<LIMBS;++limb)value[limb]=uint(input[index*LIMBS+limb]);
}
template<typename T>
void load_value(constant const T* input,uint index,thread uint* value){
  // MLX chooses constant storage for a scalar input; retain identical limb reads.
  for(uint limb=0;limb<LIMBS;++limb)value[limb]=uint(input[index*LIMBS+limb]);
}
template<typename T>
void store_value(device T* output,uint index,thread const uint* value){
  for(uint limb=0;limb<LIMBS;++limb)output[index*LIMBS+limb]=T(value[limb]);
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
)PCUMLX";
