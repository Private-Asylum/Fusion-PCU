//! Small fixture carriers; chosen arithmetic operands/results are exactly representable.
#[rustfmt::skip]
use pcu_facade::{
    PcuScalar,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
use pcu_facade::dialect::tensor::TensorElement;
pub trait Sample: PcuScalar + TensorElement {
    const FLOAT: bool;
    fn small(value: u8) -> Self;
}
macro_rules! integer {($($ty:ty),+)=>{$(impl Sample for $ty {
    const FLOAT:bool=false;
    fn small(value:u8)->Self { Self::try_from(value).unwrap() }
})+};}
integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
macro_rules! wide {
    ($ty:ty,$n:literal) => {
        impl Sample for $ty {
            const FLOAT: bool = false;
            fn small(value: u8) -> Self {
                let mut limbs = [0; $n];
                limbs[0] = u64::from(value);
                Self::from_limbs_le(limbs)
            }
        }
    };
}
wide!(PcuI256, 4);
wide!(PcuU256, 4);
wide!(PcuI512, 8);
wide!(PcuU512, 8);
impl Sample for f32 {
    const FLOAT: bool = true;
    fn small(value: u8) -> Self {
        Self::from(value)
    }
}
impl Sample for f64 {
    const FLOAT: bool = true;
    fn small(value: u8) -> Self {
        Self::from(value)
    }
}
macro_rules! low {
    ($ty:ty,$one:literal,$fraction:literal) => {
        impl Sample for $ty {
            const FLOAT: bool = true;
            fn small(value: u8) -> Self {
                let bits = if value == 0 {
                    0
                } else {
                    let exponent = value.ilog2();
                    $one + ((u16::try_from(exponent).unwrap()) << $fraction)
                        + ((u16::from(value) - (1u16 << exponent)) << $fraction)
                            / (1u16 << exponent)
                };
                Self::from_bits(bits.try_into().unwrap())
            }
        }
    };
}
low!(PcuF16Bits, 0x3c00, 10);
low!(PcuBf16Bits, 0x3f80, 7);
low!(PcuF8E4M3FnBits, 0x38, 3);
low!(PcuF8E5M2Bits, 0x3c, 2);
