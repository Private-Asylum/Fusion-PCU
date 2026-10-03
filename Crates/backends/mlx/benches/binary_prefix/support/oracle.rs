//! Exact dyadic controls and byte-level tails; reference work is outside observed calls.
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuDispatchFloatBinaryOp as Op,
};
pub trait Sample: PcuCheckedFloat {
    fn value(value: f32) -> Self;
}
macro_rules! sample {
    ($ty:ty, $convert:expr) => {
        impl Sample for $ty {
            fn value(value: f32) -> Self {
                ($convert)(value)
            }
        }
    };
}
sample!(PcuF16Bits, |v| PcuF16Bits::pcu_checked_from_f32(v).unwrap());
sample!(PcuBf16Bits, |v| PcuBf16Bits::pcu_checked_from_f32(v)
    .unwrap());
sample!(PcuF8E4M3FnBits, |v| PcuF8E4M3FnBits::pcu_checked_from_f32(
    v
)
.unwrap());
sample!(PcuF8E5M2Bits, |v| PcuF8E5M2Bits::pcu_checked_from_f32(v)
    .unwrap());
sample!(f32, |v| v);
sample!(f64, f64::from);
pub fn bank<T: Sample>(count: usize, full: usize, bank: usize, right: bool) -> Vec<T> {
    let mut input = vec![T::value(7.0); full];
    for (lane, value) in input[..count].iter_mut().enumerate() {
        let bit = ((bank >> (lane % 6)) & 1) ^ usize::from(right);
        *value = T::value([1.0, 2.0, 4.0, 8.0][(lane % 3 + bit) % 4]);
    }
    input
}
pub fn expected<T: Sample>(
    left: &[T],
    right: &[T],
    count: usize,
    op: Op,
    repeated: bool,
) -> Vec<T> {
    (0..count)
        .map(|lane| {
            let (a, b) = if repeated {
                (left[0], left[lane])
            } else {
                (left[lane], right[lane])
            };
            match op {
                Op::Add => a.pcu_checked_add(b),
                Op::Sub => a.pcu_checked_sub(b),
                Op::Mul => a.pcu_checked_mul(b),
                Op::Div => a.pcu_checked_div(b),
            }
            .unwrap()
        })
        .collect()
}
pub fn compare<T: Sample>(left: &[T], right: &[T]) {
    let binding = pcu_facade::PcuBindingRef::new(0, 0);
    let left = pcu_facade::PcuHostArgument::read(binding, left);
    let right = pcu_facade::PcuHostArgument::read(binding, right);
    assert_eq!(left.bytes(), right.bytes());
}
