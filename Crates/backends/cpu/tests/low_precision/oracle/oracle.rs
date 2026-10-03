//! Independent native binary64 arithmetic with manual format decoding and midpoint search.
//! No PCU arithmetic, conversion or packing implementation participates in this oracle.
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuDispatchFloatBinaryOp,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
};
pub trait Low: PcuCheckedFloat + Eq + core::fmt::Debug {
    const FORMAT: Format;
    fn bits(self) -> u16;
    fn from_bits(bits: u16) -> Self;
}
#[derive(Clone, Copy)]
pub struct Format {
    pub fraction: u32,
    pub bias: i32,
    pub sign: u16,
    pub max: u16,
}
macro_rules! half {
    ($ty:ty, $fraction:expr, $bias:expr, $max:expr) => {
        impl Low for $ty {
            const FORMAT: Format = Format {
                fraction: $fraction,
                bias: $bias,
                sign: 0x8000,
                max: $max,
            };
            fn bits(self) -> u16 {
                self.to_bits()
            }
            fn from_bits(bits: u16) -> Self {
                Self::from_bits(bits)
            }
        }
    };
}
half!(PcuF16Bits, 10, 15, 0x7bff);
half!(PcuBf16Bits, 7, 127, 0x7f7f);
macro_rules! fp8 {
    ($ty:ty, $fraction:expr, $bias:expr, $max:expr) => {
        impl Low for $ty {
            const FORMAT: Format = Format {
                fraction: $fraction,
                bias: $bias,
                sign: 0x80,
                max: $max,
            };
            fn bits(self) -> u16 {
                u16::from(self.to_bits())
            }
            fn from_bits(bits: u16) -> Self {
                Self::from_bits(u8::try_from(bits).unwrap())
            }
        }
    };
}
fp8!(PcuF8E4M3FnBits, 3, 7, 0x7e);
fp8!(PcuF8E5M2Bits, 2, 15, 0x7b);
impl Format {
    pub fn value(self, bits: u16) -> f64 {
        let magnitude = bits & (self.sign - 1);
        let fraction = magnitude & ((1 << self.fraction) - 1);
        let exponent = magnitude >> self.fraction;
        let (significand, scale) = if exponent == 0 {
            (
                fraction,
                1 - self.bias - i32::try_from(self.fraction).unwrap(),
            )
        } else {
            (
                (1 << self.fraction) + fraction,
                i32::from(exponent) - self.bias - i32::try_from(self.fraction).unwrap(),
            )
        };
        let value = f64::from(significand) * 2.0_f64.powi(scale);
        if bits & self.sign == 0 { value } else { -value }
    }
    pub fn evaluate(
        self,
        left: u16,
        right: u16,
        op: PcuDispatchFloatBinaryOp,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<u16, PcuExecutionFaultKind> {
        match self.evaluate_clamped(left, right, op, policy)? {
            (value, None) => Ok(value),
            (_, Some(kind)) => Err(kind),
        }
    }
    #[allow(clippy::float_cmp)] // Exact dyadic ties and inexactness require equality, never epsilon.
    pub fn evaluate_clamped(
        self,
        left: u16,
        right: u16,
        op: PcuDispatchFloatBinaryOp,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<(u16, Option<PcuExecutionFaultKind>), PcuExecutionFaultKind> {
        if left & (self.sign - 1) > self.max || right & (self.sign - 1) > self.max {
            return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
        }
        let a = self.value(left);
        let b = self.value(right);
        if op == PcuDispatchFloatBinaryOp::Div && b == 0.0 {
            return Err(PcuExecutionFaultKind::DivideByZero);
        }
        let exact = match op {
            PcuDispatchFloatBinaryOp::Add => a + b,
            PcuDispatchFloatBinaryOp::Sub => a - b,
            PcuDispatchFloatBinaryOp::Mul => a * b,
            PcuDispatchFloatBinaryOp::Div => a / b,
        };
        let magnitude = exact.abs();
        let sign = if exact.is_sign_negative() {
            self.sign
        } else {
            0
        };
        let maximum = self.value(self.max);
        let overflow = maximum + (maximum - self.value(self.max - 1)) / 2.0;
        if magnitude > overflow || (magnitude == overflow && self.max & 1 != 0) {
            return Ok((
                sign | self.max,
                Some(PcuExecutionFaultKind::ArithmeticOverflow),
            ));
        }
        let mut low = 0;
        let mut high = self.max;
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if self.value(middle) <= magnitude {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        let rounded = if low == self.max {
            low
        } else {
            let midpoint = self.value(low).midpoint(self.value(low + 1));
            if magnitude > midpoint || (magnitude == midpoint && low & 1 != 0) {
                low + 1
            } else {
                low
            }
        };
        // Tininess is tested after destination-precision rounding with unbounded exponent,
        // so a packed minimum-normal result can still be tiny and inexact.
        let minimum_normal = 1 << self.fraction;
        let tiny_boundary = self.value(minimum_normal) - self.value(1) / 4.0;
        let tiny_inexact =
            magnitude != 0.0 && magnitude < tiny_boundary && magnitude != self.value(rounded);
        if (policy != PcuFloatUnderflowPolicy::AllowGradualUnderflow && tiny_inexact)
            || (policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                && rounded != 0
                && rounded < minimum_normal)
        {
            return Ok((
                sign | rounded,
                Some(PcuExecutionFaultKind::ArithmeticUnderflow),
            ));
        }
        Ok((sign | rounded, None))
    }
}
