#pragma once
inline constexpr char pcu_mlx_integer_body[] = R"PCUMLX(
uint id = thread_position_in_grid.x;
if(id>=PCU_COUNT)return;
  uint a[LIMBS],b[LIMBS],result[LIMBS];
  load_value(left,PCU_LEFT_BROADCAST?0u:id,a);
  load_value(right,PCU_RIGHT_BROADCAST?0u:id,b);
  bool an=PCU_SIGNED!=0u && (a[LIMBS-1u]&SIGN)!=0u;
  bool bn=PCU_SIGNED!=0u && (b[LIMBS-1u]&SIGN)!=0u;
  uint operation=PCU_OPERATION,fault=0u;bool negative_limit=false;
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
  bool clamp=PCU_CLAMP;
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
)PCUMLX";
