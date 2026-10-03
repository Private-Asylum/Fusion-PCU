//! Independent representation-only derivative oracle; no floating arithmetic or core result oracle.
#[rustfmt::skip]
use fusion_pcu::{PcuScalar,PcuCheckedFloat,PcuFloatUnderflowPolicy,PcuExecutionFaultKind,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
pub trait Format: PcuScalar + PcuCheckedFloat + std::fmt::Debug + PartialEq {
    const LABEL: &'static str;
    const SIGN: u64;
    const MAX: u64;
    const ONE: u64;
    const MIN_NORMAL: u64;
    fn bits(self) -> u64;
    fn from(bits: u64) -> Self;
    fn zero() -> Self {
        Self::from(0)
    }
    fn one() -> Self {
        Self::from(Self::ONE)
    }
    fn sentinel() -> Self {
        Self::from(Self::ONE + 1)
    }
}
macro_rules! formats { ($($ty:ty, $label:literal, $sign:expr, $max:expr, $one:expr, $minimum:expr;)+) => {$(
    impl Format for $ty {
        const LABEL: &'static str=$label; const SIGN:u64=$sign; const MAX:u64=$max; const ONE:u64=$one; const MIN_NORMAL:u64=$minimum;
        fn bits(self)->u64 { u64::from(self.to_bits()) }
        fn from(bits:u64)->Self { Self::from_bits(bits.try_into().unwrap()) }
    }
)+}; }
formats!(PcuF16Bits,"f16",0x8000,0x7bff,0x3c00,0x400;
PcuBf16Bits,"bf16",0x8000,0x7f7f,0x3f80,0x80;
PcuF8E4M3FnBits,"e4m3fn",0x80,0x7e,0x38,0x8;
PcuF8E5M2Bits,"e5m2",0x80,0x7b,0x3c,0x4;
f32,"f32",0x8000_0000,0x7f7f_ffff,0x3f80_0000,0x80_0000;
f64,"f64",0x8000_0000_0000_0000,0x7fef_ffff_ffff_ffff,0x3ff0_0000_0000_0000,0x10_0000_0000_0000;);
pub fn expected<T: Format>(
    input: T,
    upstream: T,
    policy: PcuFloatUnderflowPolicy,
) -> Result<T, PcuExecutionFaultKind> {
    let mask = T::SIGN - 1;
    if input.bits() & mask > T::MAX || upstream.bits() & mask > T::MAX {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    let selected = if input.bits() & T::SIGN == 0 && input.bits() & mask != 0 {
        upstream.bits()
    } else {
        0
    };
    if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
        && selected & mask != 0
        && selected & mask < T::MIN_NORMAL
    {
        return Err(PcuExecutionFaultKind::ArithmeticUnderflow);
    }
    Ok(T::from(selected))
}
pub fn inputs<T: Format>(
    count: usize,
    phase: u32,
    _op: u32,
    policy: PcuFloatUnderflowPolicy,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let representatives = [
        0,
        T::SIGN,
        1,
        T::SIGN | 1,
        T::MAX,
        T::SIGN | T::MAX,
        T::ONE,
        T::SIGN | T::ONE,
        T::MIN_NORMAL,
        T::MIN_NORMAL - 1,
    ];
    let mut a = Vec::with_capacity(count);
    let mut b = Vec::with_capacity(count);
    let mut want = Vec::with_capacity(count);
    for lane in 0..count {
        let i = lane + usize::try_from(phase).unwrap();
        let x = T::from(representatives[i % representatives.len()]);
        let mut dy = T::from(representatives[(i * 7 + 3) % representatives.len()]);
        if expected(x, dy, policy).is_err() {
            dy = T::one();
        }
        a.push(x);
        b.push(dy);
        want.push(expected(x, dy, policy).unwrap());
    }
    (a, b, want)
}
