// MLX-owned U32 magnitude division; no native signed arithmetic or ulong division.
#pragma once
inline constexpr char pcu_mlx_div_rem_header[] = R"PCUMLX(
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
  for(uint limb=0;limb<LIMBS;++limb)value[limb]=uint(input[index*LIMBS+limb]);
}
template<typename T>
void store_value(device T* output,uint index,thread const uint* value){
  for(uint limb=0;limb<LIMBS;++limb)output[index*LIMBS+limb]=T(value[limb]);
}
void negate(thread uint* value){
  uint carry=1u;
  for(uint limb=0;limb<LIMBS;++limb){uint previous=~value[limb];
    value[limb]=previous+carry;carry=uint(value[limb]<previous);}
  value[LIMBS-1u]&=MASK;
}
bool at_least(thread const uint* remainder,thread const uint* divisor){
  if(remainder[2]!=0u)return true;
  if(remainder[1]!=divisor[1])return remainder[1]>divisor[1];
  return remainder[0]>=divisor[0];
}
void subtract_divisor(thread uint* remainder,thread const uint* divisor){
  uint borrow=0u;
  for(uint limb=0;limb<3u;++limb){uint previous=remainder[limb];
    uint word=limb<2u?divisor[limb]:0u;uint difference=previous-word;
    uint first=uint(previous<word);remainder[limb]=difference-borrow;
    borrow=first|uint(difference<borrow);}
}
)PCUMLX";
