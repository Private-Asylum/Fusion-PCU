// Private unqualified wide prototype. U32 magnitudes only, bounded to sixteen limbs.
#pragma once
inline constexpr char pcu_mlx_div_rem_wide_body[] = R"PCUMLX(
uint index=thread_position_in_grid.x;
if(index<PCU_COUNT){
  uint a[LIMBS],b[LIMBS],q[LIMBS],r[LIMBS+1u];
  for(uint limb=0;limb<LIMBS;++limb){a[limb]=0u;b[limb]=0u;q[limb]=0u;r[limb]=0u;}
  r[LIMBS]=0u;
  load_value(left,PCU_LEFT_BROADCAST!=0u?0u:index,a);
  load_value(right,PCU_RIGHT_BROADCAST!=0u?0u:index,b);
  bool negative_a=PCU_SIGNED!=0u&&(a[LIMBS-1u]&SIGN)!=0u;
  bool negative_b=PCU_SIGNED!=0u&&(b[LIMBS-1u]&SIGN)!=0u;
  if(negative_a)negate(a);
  if(negative_b)negate(b);
  bool minimum_a=a[LIMBS-1u]==SIGN;
  bool one_b=b[0]==1u;
  uint divisor_or=0u;
  for(uint limb=0;limb<LIMBS;++limb){
    divisor_or|=b[limb];
    if(limb+1u<LIMBS)minimum_a=minimum_a&&a[limb]==0u;
    if(limb!=0u)one_b=one_b&&b[limb]==0u;
  }
  uint fault=divisor_or==0u?4u:(negative_a&&negative_b&&minimum_a&&one_b?1u:0u);
  if(fault==0u){
    // Before each step r<divisor<2^WIDTH. After shifting, r<2^(WIDTH+1).
    // LIMBS+1 words preserve that carry; subtracting once restores r<divisor.
    // All shift operands are 1,31 or bit%32; no shift by the U32 width occurs.
    for(uint remaining=PCU_WIDTH;remaining!=0u;--remaining){
      uint bit=remaining-1u;
      for(uint words=LIMBS;words!=0u;--words)
        r[words]=(r[words]<<1u)|(r[words-1u]>>31u);
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
