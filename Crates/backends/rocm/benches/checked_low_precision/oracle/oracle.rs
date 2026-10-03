//! Reference fault oracle plus independent host binary64-to-encoding midpoint search.
#[rustfmt::skip]
use fusion_pcu::{ PcuCheckedFloat, PcuClampedFloat, PcuClampedError, PcuExecutionFaultKind, PcuFloatUnderflowPolicy, PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits };
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
pub fn reference<T: Format>(
    a: T,
    b: T,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
) -> Result<T, PcuClampedError<T>> {
    match op {
        0 => a.pcu_clamped_add_with_policy(b, policy),
        1 => a.pcu_clamped_sub_with_policy(b, policy),
        2 => a.pcu_clamped_mul_with_policy(b, policy),
        3 => a.pcu_clamped_div_with_policy(b, policy),
        _ => panic!("invalid operation"),
    }
}
pub fn unpack<T: Format>(bits: u16) -> f64 {
    let exp = (bits & (T::SIGN - 1)) >> T::FRACTION;
    let sig = (bits & ((1 << T::FRACTION) - 1)) | if exp == 0 { 0 } else { 1 << T::FRACTION };
    let scale = i32::from(exp.max(1)) - T::BIAS - i32::try_from(T::FRACTION).unwrap();
    f64::from(sig) * 2.0_f64.powi(scale) * if bits & T::SIGN == 0 { 1.0 } else { -1.0 }
}
// Destination precision <=11 and divisor significands <=11 bits: a non-tie
// quotient is separated from a destination midpoint by far more than a binary64
// ulp. Products are exact; sums near a rounding midpoint have <=23 exact bits.
// This deliberately uses no core arithmetic/packing to establish output bits.
#[allow(clippy::float_cmp)] // Exact comparisons with encoded dyadics and midpoint ties.
pub fn independent<T: Format>(a: T, b: T, op: u32) -> u16 {
    let left = unpack::<T>(a.bits());
    let right = unpack::<T>(b.bits());
    let result = match op {
        0 => left + right,
        1 => left - right,
        2 => left * right,
        3 => left / right,
        _ => unreachable!(),
    };
    let sign = if result.is_sign_negative() {
        T::SIGN
    } else {
        0
    };
    let magnitude = result.abs();
    let mut low = 0u16;
    let mut high = T::MAX;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if unpack::<T>(middle) <= magnitude {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    if low == T::MAX {
        return sign | low;
    }
    let midpoint = unpack::<T>(low).midpoint(unpack::<T>(low + 1));
    sign | if magnitude > midpoint || (magnitude == midpoint && low & 1 != 0) {
        low + 1
    } else {
        low
    }
}
pub fn expected<T: Format>(
    a: T,
    b: T,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
) -> (T, Option<PcuExecutionFaultKind>) {
    let (value, fault) = match reference(a, b, op, policy) {
        Ok(value) => (value, None),
        Err(PcuClampedError::Range(f)) => {
            let kind = f.kind();
            (f.clamped_value(), Some(kind))
        }
        Err(PcuClampedError::Fatal(kind)) => panic!("fatal fixture operand {a:?}/{b:?}: {kind:?}"),
    };
    assert_eq!(
        value.bits(),
        independent(a, b, op),
        "independent bits: {a:?}/{b:?} op{op}"
    );
    (value, fault)
}
pub fn inputs<T: Format>(count: usize, phase: u32, op: u32) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut state = phase.wrapping_add(1);
    let mut left = Vec::with_capacity(count);
    let mut right = Vec::with_capacity(count);
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let a = T::from(
            u16::try_from(state & u32::from(T::SIGN.wrapping_mul(2).wrapping_sub(1))).unwrap(),
        );
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let b = T::from(
            u16::try_from(state & u32::from(T::SIGN.wrapping_mul(2).wrapping_sub(1))).unwrap(),
        );
        let (a, b) = if reference(a, b, op, PcuFloatUnderflowPolicy::IeeeAfterRounding).is_ok() {
            (a, b)
        } else {
            (T::one(), T::one())
        };
        let value = expected(a, b, op, PcuFloatUnderflowPolicy::IeeeAfterRounding).0;
        left.push(a);
        right.push(b);
        output.push(value);
    }
    (left, right, output)
}
pub fn verify<T: Format>(expected: &[T], output: &[T]) {
    assert_eq!(&output[..expected.len()], expected);
    assert!(output[expected.len()..].iter().all(|v| *v == T::sentinel()));
}
