//! Reference fault oracle plus independent host binary64-to-encoding midpoint search.
#[rustfmt::skip]
use fusion_pcu::{ PcuCheckedFloat, PcuClampedFloat, PcuExecutionFaultKind, PcuFloatUnderflowPolicy, PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits };
pub trait Format: PcuCheckedFloat + PcuClampedFloat + std::fmt::Debug + Eq {
    const LABEL: &'static str;
    const SIGN: u16;
    const MAX: u16;
    const ONE: u16;
    const FRACTION: u32;
    const BIAS: i32;
    fn bits(self) -> u16;
    fn from(bits: u16) -> Self;
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
macro_rules! formats { ($($ty:ty, $label:literal, $sign:expr, $maximum:expr, $one:expr, $fraction:expr, $bias:expr;)+) => { $(
    impl Format for $ty {
        const LABEL: &'static str = $label; const SIGN:u16=$sign; const MAX:u16=$maximum; const ONE:u16=$one; const FRACTION:u32=$fraction; const BIAS:i32=$bias;
        fn bits(self) -> u16 { u16::from(self.to_bits()) }
        fn from(bits:u16)->Self { Self::from_bits(bits.try_into().unwrap()) }
    }
)+ }; }
formats!(PcuF16Bits,"f16",0x8000,0x7bff,0x3c00,10,15;
PcuBf16Bits,"bf16",0x8000,0x7f7f,0x3f80,7,127;
PcuF8E4M3FnBits,"e4m3fn",0x80,0x7e,0x38,3,7;
PcuF8E5M2Bits,"e5m2",0x80,0x7b,0x3c,2,15;);

pub fn evaluate<T: Format>(
    value: T,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
) -> Result<(T, bool), PcuExecutionFaultKind> {
    let bits = value.bits();
    let magnitude = bits & (T::SIGN - 1);
    if magnitude > T::MAX {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    let result = if op == 0 {
        bits ^ T::SIGN
    } else if bits & T::SIGN == 0 && magnitude != 0 {
        bits
    } else {
        0
    };
    let magnitude = result & (T::SIGN - 1);
    Ok((
        T::from(result),
        policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
            && magnitude != 0
            && magnitude < (1 << T::FRACTION),
    ))
}
pub fn inputs<T: Format>(count: usize, phase: u32, op: u32) -> (Vec<T>, Vec<T>) {
    let input = (0..count)
        .map(|i| {
            let bits =
                u16::try_from((i + usize::try_from(phase).unwrap()) % usize::from(T::MAX + 1))
                    .unwrap();
            T::from(
                if bits < (1 << T::FRACTION) {
                    T::ONE
                } else {
                    bits
                } | if i.is_multiple_of(3) { T::SIGN } else { 0 },
            )
        })
        .collect::<Vec<_>>();
    let expected = input
        .iter()
        .copied()
        .map(|v| {
            evaluate(v, op, PcuFloatUnderflowPolicy::IeeeAfterRounding)
                .unwrap()
                .0
        })
        .collect();
    (input, expected)
}
pub fn verify<T: Format>(expected: &[T], output: &[T]) {
    assert_eq!(&output[..expected.len()], expected);
    assert!(output[expected.len()..].iter().all(|v| *v == T::sentinel()));
}
