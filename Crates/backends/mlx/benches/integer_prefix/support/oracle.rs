//! Small exact integer banks keep all measured operations in range; fault banks are separate native tests.
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedInteger,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuDispatchIntegerBinaryOp as Op,
};
pub trait Sample: PcuCheckedInteger {
    fn small(value: u8) -> Self;
}
macro_rules! native {($($ty:ty),+)=>{$(impl Sample for $ty {fn small(value:u8)->Self {Self::try_from(value).unwrap()}})+};}
native!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);
macro_rules! wide {
    ($ty:ty,$limbs:literal) => {
        impl Sample for $ty {
            fn small(value: u8) -> Self {
                let mut limbs = [0; $limbs];
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

pub fn bank<T: Sample>(count: usize, full: usize, bank: usize, right: bool) -> Vec<T> {
    let mut input = vec![T::small(7); full];
    for (lane, value) in input[..count].iter_mut().enumerate() {
        let bit = u8::try_from((bank >> (lane % 6)) & 1).unwrap();
        *value = T::small(if right { 1 + bit } else { 4 + bit });
    }
    if !right {
        input[0] = T::small(8);
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
            }
            .unwrap()
        })
        .collect()
}
pub fn compare<T: Sample>(left: &[T], right: &[T]) {
    let binding = pcu_facade::PcuBindingRef::new(0, 0);
    assert_eq!(
        pcu_facade::PcuHostArgument::read(binding, left).bytes(),
        pcu_facade::PcuHostArgument::read(binding, right).bytes()
    );
}
