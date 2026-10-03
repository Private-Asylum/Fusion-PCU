//! Handwritten two's-complement checked Add/Mul with no PCU emitter or signed C++ overflow.
#[rustfmt::skip]
use fusion_pcu::{
    PcuRangePolicy,
    PcuFloatUnderflowPolicy,
};
pub fn source(
    signed: bool,
    extent: usize,
    invocations: u32,
    range: PcuRangePolicy,
    _uf: PcuFloatUnderflowPolicy,
) -> String {
    format!(
        "#define SIGNED {}\n#define CLAMP {}\n#define EXTENT {extent}ull\n#define INVOCATIONS {invocations}u\n{KERNEL}",
        u8::from(signed),
        u8::from(range == PcuRangePolicy::Clamp)
    )
}
const KERNEL: &str = r#"
using U=unsigned long long;
struct Checked { U value; unsigned code; };
__device__ Checked add(U a,U b) {
    U r=a+b;
    if (!SIGNED) return r<a?Checked{~0ull,3}:Checked{r,0};
    U sign=1ull<<63;
    if (((a^b)&sign)==0 && ((a^r)&sign)) return (a&sign)?Checked{sign,4}:Checked{sign-1,3};
    return {r,0};
}
struct Product { U low; U high; };
__device__ Product product(U a,U b) {
    U a0=static_cast<unsigned int>(a),a1=a>>32,b0=static_cast<unsigned int>(b),b1=b>>32;
    U base=a0*b0;
    U cross=a1*b0+(base>>32);
    U upper=cross>>32;
    cross=(cross&0xffffffffull)+a0*b1;
    return {(cross<<32)|(base&0xffffffffull),a1*b1+upper+(cross>>32)};
}
__device__ Checked mul(U a,U b) {
    if (!SIGNED) { Product p=product(a,b); return p.high?Checked{~0ull,3}:Checked{p.low,0}; }
    bool negative=((a^b)>>63)!=0;
    U x=(a>>63)?0ull-a:a, y=(b>>63)?0ull-b:b;
    U bound=negative?(1ull<<63):((1ull<<63)-1);
    Product p=product(x,y);
    if (p.high || p.low>bound) return negative?Checked{1ull<<63,4}:Checked{(1ull<<63)-1,3};
    return {negative?0ull-p.low:p.low,0};
}
extern "C" __global__ void native_helpers(U* stage,U* output,const U* input,U* status) {
    U id=static_cast<U>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (id>=INVOCATIONS) return;
    for (U lane=id;lane<EXTENT;lane+=INVOCATIONS) {
        U original=input[lane]; Checked sum=add(original,original);
        if (sum.code && !CLAMP) { atomicMin(status,(lane<<3)|sum.code); continue; }
        stage[lane]=sum.value; Checked product=mul(stage[lane],original);
        if (product.code && !CLAMP) { atomicMin(status,(lane<<3)|product.code); continue; }
        output[lane]=product.value;
        unsigned code=sum.code?sum.code:product.code;
        if (code) atomicMin(status,(1ull<<63)|(lane<<3)|code);
    }
}
"#;
