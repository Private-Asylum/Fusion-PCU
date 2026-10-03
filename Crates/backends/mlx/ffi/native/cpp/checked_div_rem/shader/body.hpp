#pragma once
inline constexpr char pcu_mlx_div_rem_body[] = R"PCUMLX(
uint index=thread_position_in_grid.x;
if(index<PCU_COUNT){
  uint a[2]={0u,0u},b[2]={0u,0u},q[2]={0u,0u},r[3]={0u,0u,0u};
  load_value(left,PCU_LEFT_BROADCAST!=0u?0u:index,a);
  load_value(right,PCU_RIGHT_BROADCAST!=0u?0u:index,b);
  bool negative_a=PCU_SIGNED!=0u&&(a[LIMBS-1u]&SIGN)!=0u;
  bool negative_b=PCU_SIGNED!=0u&&(b[LIMBS-1u]&SIGN)!=0u;
  if(negative_a)negate(a);
  if(negative_b)negate(b);
  bool minimum_a=a[LIMBS-1u]==SIGN;
  for(uint limb=0;limb+1u<LIMBS;++limb)minimum_a=minimum_a&&a[limb]==0u;
  bool one_b=b[0]==1u&&b[1]==0u;
  uint fault=(b[0]|b[1])==0u?4u:
    (negative_a&&negative_b&&minimum_a&&one_b?1u:0u);
  if(fault==0u){
    // Before every step r<divisor<=2^64-1. The shift is therefore below2^65;
    // three U32 limbs retain that carry, and subtraction restores r<divisor.
    for(uint remaining=PCU_WIDTH;remaining!=0u;--remaining){
      uint bit=remaining-1u;
      r[2]=(r[2]<<1u)|(r[1]>>31u);
      r[1]=(r[1]<<1u)|(r[0]>>31u);
      r[0]=(r[0]<<1u)|((a[bit/32u]>>(bit%32u))&1u);
      if(at_least(r,b)){subtract_divisor(r,b);q[bit/32u]|=1u<<(bit%32u);}
    }
    if(negative_a!=negative_b)negate(q);
    if(negative_a)negate(r);
  }
  store_value(quotient,index,q);
  store_value(remainder,index,r);
  records[index]=fault;
}
)PCUMLX";
