//! Independently written workload body; arithmetic helpers and ownership ABI are shared.
use super::Policy;
use fusion_pcu::PcuScalarType;
const INTEGER: &str = r"
if(id>=COUNTu)return;
uint original[LIMBS],seed[LIMBS],first[LIMBS]={0},dead[LIMBS]={0},product[LIMBS]={0},final_value[LIMBS]={0};
load_value(r0,id,original);load_value(r2,0u,seed);
bool failed=false;uint status=pcu_integer(original,seed,first,0u,CLAMP0);
records[id]=status;failed=status!=0u&&(status&0x100u)==0u;
status=failed?0u:pcu_integer(first,seed,dead,0u,CLAMP1);
records[COUNTu+id]=status;failed=failed||(status!=0u&&(status&0x100u)==0u);
status=failed?0u:pcu_integer(first,original,product,2u,CLAMP2);
records[2u*COUNTu+id]=status;failed=failed||(status!=0u&&(status&0x100u)==0u);
status=failed?0u:pcu_integer(product,original,final_value,1u,CLAMP3);
records[3u*COUNTu+id]=status;
store_value(r1,id,original);store_value(r3,id,final_value);
";
const FLOAT: &str = r"
if(id>=COUNTu)return;
LOADS
WORD first=0,dead=0,product=0,final_value=0;
bool failed=false;uint status=0;
{Result step;if(NONFINITE(original)||NONFINITE(seed))step=fault(INVALIDu);else FIRST;
first=step.bits;status=STATUS;records[id]=status;failed=status!=0u&&(status&0x100u)==0u;}
status=0;if(!failed){Result step;if(NONFINITE(first)||NONFINITE(seed))step=fault(INVALIDu);else DEAD;
dead=step.bits;status=STATUS;}records[COUNTu+id]=status;failed=failed||(status!=0u&&(status&0x100u)==0u);
status=0;if(!failed){Result step;if(NONFINITE(first)||NONFINITE(original))step=fault(INVALIDu);else PRODUCT;
product=step.bits;status=STATUS;}records[2u*COUNTu+id]=status;failed=failed||(status!=0u&&(status&0x100u)==0u);
status=0;if(!failed){Result step;if(NONFINITE(product)||NONFINITE(original))step=fault(INVALIDu);else FINAL;
final_value=step.bits;status=STATUS;}records[3u*COUNTu+id]=status;
STORES
";
pub fn source(scalar: PcuScalarType, count: u32, policies: [Policy; 4]) -> String {
    let mut body = if matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64) {
        floating(scalar, policies)
    } else {
        INTEGER.to_owned()
    };
    body = body.replace("COUNT", &count.to_string());
    for (site, policy) in policies.iter().enumerate() {
        body = body.replace(
            &format!("CLAMP{site}"),
            if policy.clamp { "true" } else { "false" },
        );
    }
    format!(
        "kernel void pcu_checked_composed(device uchar* r0 [[buffer(0)]],device uchar* r1 [[buffer(1)]],device uchar* r2 [[buffer(2)]],device uchar* r3 [[buffer(3)]],device uint* records [[buffer(4)]],constant uint2& profile [[buffer(5)]],uint id [[thread_position_in_grid]]) {{\n{body}\n}}"
    )
}
fn floating(scalar: PcuScalarType, policies: [Policy; 4]) -> String {
    let narrow = scalar == PcuScalarType::F32;
    let mut body = FLOAT.replace("WORD", if narrow { "uint" } else { "ulong" })
        .replace("INVALID", if narrow { "1" } else { "4" })
        .replace("LOADS", if narrow { "uint original=reinterpret_cast<device const uint*>(r0)[id],seed=reinterpret_cast<device const uint*>(r2)[0];" } else {
            "ulong original=reinterpret_cast<device const ulong*>(r0)[id],seed=reinterpret_cast<device const ulong*>(r2)[0];"
        })
        .replace("STORES", if narrow { "reinterpret_cast<device uint*>(r1)[id]=original;reinterpret_cast<device uint*>(r3)[id]=final_value;" } else {
            "reinterpret_cast<device ulong*>(r1)[id]=original;reinterpret_cast<device ulong*>(r3)[id]=final_value;"
        })
        .replace("NONFINITE(original)", if narrow { "((original>>23u)&255u)==255u" } else { "((original>>52u)&2047ul)==2047ul" })
        .replace("NONFINITE(seed)", if narrow { "((seed>>23u)&255u)==255u" } else { "((seed>>52u)&2047ul)==2047ul" })
        .replace("NONFINITE(first)", if narrow { "((first>>23u)&255u)==255u" } else { "((first>>52u)&2047ul)==2047ul" })
        .replace("NONFINITE(product)", if narrow { "((product>>23u)&255u)==255u" } else { "((product>>52u)&2047ul)==2047ul" })
        .replace("STATUS", if narrow { "((step.status&0xffu)==1u?4u:(step.status&0xffu)==2u?3u:(step.status&0xffu)==3u?1u:(step.status&0xffu)==4u?2u:0u)|(step.status&0x100u)" } else { "step.status" });
    for (site, (token, method, lhs, rhs)) in [
        ("FIRST", "add", "original", "seed"),
        ("DEAD", "divide", "first", "seed"),
        ("PRODUCT", "multiply", "first", "original"),
        ("FINAL", "add", "product", "original"),
    ]
    .iter()
    .enumerate()
    {
        let policy = policies[site];
        let suffix = if *method == "add" { ",false" } else { "" };
        let call = if narrow {
            format!(
                "{{Binary math={{{}u,{}}};step=math.{method}({lhs},{rhs}{suffix});}}",
                policy.underflow, policy.clamp
            )
        } else {
            format!(
                "step={method}({lhs},{rhs}{suffix},{}u);",
                policy.underflow | if policy.clamp { 0x100 } else { 0 }
            )
        };
        body = body.replace(token, &call);
    }
    body
}
