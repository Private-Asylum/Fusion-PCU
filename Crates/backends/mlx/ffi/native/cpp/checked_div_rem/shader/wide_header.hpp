// Private unqualified wide prototype. Primitive-eight dispatch retains its independent header.
#pragma once
inline constexpr char pcu_mlx_div_rem_wide_header[] = R"PCUMLX(
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
  if(remainder[LIMBS]!=0u)return true;
  for(uint remaining=LIMBS;remaining!=0u;--remaining){uint limb=remaining-1u;
    if(remainder[limb]!=divisor[limb])return remainder[limb]>divisor[limb];}
  return true;
}
void subtract_divisor(thread uint* remainder,thread const uint* divisor){
  uint borrow=0u;
  for(uint limb=0;limb<=LIMBS;++limb){uint previous=remainder[limb];
    uint word=limb<LIMBS?divisor[limb]:0u;uint difference=previous-word;
    uint first=uint(previous<word);remainder[limb]=difference-borrow;
    borrow=first|uint(difference<borrow);}
}
)PCUMLX";
