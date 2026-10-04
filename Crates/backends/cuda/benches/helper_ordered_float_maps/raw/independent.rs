//! Independently authored checked Add then Mul; no PCU emitter source is imported.
#[rustfmt::skip]
use fusion_pcu::{
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
};
pub fn source(
    bytes: usize,
    extent: usize,
    invocations: u32,
    range: PcuRangePolicy,
    uf: PcuFloatUnderflowPolicy,
) -> String {
    let (ty, word, fraction, exponent, bias, sign, maximum, minimum) = match bytes {
        4 => (
            "float",
            "unsigned int",
            23,
            255,
            127,
            "0x80000000ull",
            "0x7f7fffffull",
            -126,
        ),
        8 => (
            "double",
            "unsigned long long",
            52,
            2047,
            1023,
            "0x8000000000000000ull",
            "0x7fefffffffffffffull",
            -1022,
        ),
        _ => panic!("handwritten helper only admits F32/F64"),
    };
    let policy = match uf {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
    };
    let clamp = u8::from(range == PcuRangePolicy::Clamp);
    format!(
        "using T={ty}; using Word={word};\n#define FRAC {fraction}\n#define EXP {exponent}ull\n#define BIAS {bias}\n#define SIGN {sign}\n#define MAXIMUM {maximum}\n#define MINIMUM {minimum}\n#define POLICY {policy}\n#define CLAMP {clamp}\n#define EXTENT {extent}ull\n#define INVOCATIONS {invocations}u\n{KERNEL}"
    )
}
const KERNEL: &str = r#"
using U = unsigned long long;
__device__ U bits(T x) { return __builtin_bit_cast(Word,x); }
__device__ T value(U x) { return __builtin_bit_cast(T,static_cast<Word>(x)); }
__device__ U mantissa(U x) { U m=x&((1ull<<FRAC)-1); return ((x>>FRAC)&EXP)?m|(1ull<<FRAC):m; }
__device__ int scale(U x) { int e=static_cast<int>((x>>FRAC)&EXP); return (e?e:1)-BIAS-FRAC; }
struct Product { U low; U high; };
__device__ Product product(U a,U b) {
    U a0=static_cast<unsigned int>(a),a1=a>>32,b0=static_cast<unsigned int>(b),b1=b>>32;
    U base=a0*b0;
    U cross=a1*b0+(base>>32);
    U upper=cross>>32;
    cross=(cross&0xffffffffull)+a0*b1;
    return {(cross<<32)|(base&0xffffffffull),a1*b1+upper+(cross>>32)};
}
__device__ bool tiny_inexact(U a,U b,U result) {
    Product p=product(mantissa(a),mantissa(b));
    if (!(p.low|p.high)) return false;
    int top=p.high?127-__clzll(p.high):63-__clzll(p.low);
    int drop=top-FRAC;
    U significand=drop>0?((p.low>>drop)|(p.high<<(64-drop))):(p.low<<(-drop));
    if (drop>0) {
        U remainder=p.low&((1ull<<drop)-1),half=1ull<<(drop-1);
        if (remainder>half||(remainder==half&&(significand&1))) ++significand;
    }
    int rounded_exponent=scale(a)+scale(b)+top;
    if (significand==(1ull<<(FRAC+1))) ++rounded_exponent;
    if (rounded_exponent>=MINIMUM) return false;
    U rm=mantissa(result);
    if (!rm) return true;
    int shift=scale(result)-scale(a)-scale(b);
    Product represented={0,0};
    if (shift>=0&&shift<128) {
        represented.low=shift<64?rm<<shift:0;
        represented.high=shift==0?0:(shift<64?rm>>(64-shift):rm<<(shift-64));
    } else if (shift<0&&shift>-64) {
        U remainder=rm&((1ull<<(-shift))-1);
        if (remainder) return true;
        represented.low=rm>>(-shift);
    } else return true;
    return represented.low!=p.low||represented.high!=p.high;
}
struct Checked { T value; unsigned code; };
__device__ Checked calculate(T a,T b,bool multiply) {
    U x=bits(a),y=bits(b);
    if (((x>>FRAC)&EXP)==EXP||((y>>FRAC)&EXP)==EXP) return {value(0),5};
    T z=multiply?a*b:a+b;
    U raw=bits(z),magnitude=raw&~SIGN;
    unsigned code=0;
    if (((raw>>FRAC)&EXP)==EXP) code=3;
    else {
        bool subnormal=magnitude&&((raw>>FRAC)&EXP)==0;
        bool inexact=multiply&&tiny_inexact(x,y,raw);
        if ((POLICY==0&&inexact)||(POLICY==1&&(subnormal||inexact))) code=4;
    }
    if (CLAMP&&code==3) z=value((raw&SIGN)|MAXIMUM);
    return {z,code};
}
__device__ bool record(Checked c,U lane,U* status,bool* recovered) {
    if (!c.code) return true;
    if (CLAMP&&(c.code==3||c.code==4)) {
        if (!*recovered) { atomicMin(status,(1ull<<63)|(lane<<3)|c.code); *recovered=true; }
        return true;
    }
    atomicMin(status,(lane<<3)|c.code);
    return false;
}
extern "C" __global__ void native_helpers(T* stage,T* output,const T* input,U* status) {
    unsigned start=blockIdx.x*blockDim.x+threadIdx.x;
    if (start>=INVOCATIONS) return;
    for (U lane=start;lane<EXTENT;lane+=INVOCATIONS) {
        bool recovered=false;
        T original=input[lane];
        Checked first=calculate(original,original,false);
        if (!record(first,lane,status,&recovered)) return;
        stage[lane]=first.value;
        T reloaded=stage[lane];
        Checked second=calculate(reloaded,original,true);
        if (!record(second,lane,status,&recovered)) return;
        output[lane]=second.value;
    }
}
"#;
