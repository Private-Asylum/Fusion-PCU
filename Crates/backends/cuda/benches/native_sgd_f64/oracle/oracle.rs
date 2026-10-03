//! Independent integer/dyadic complete-output oracle for the benchmark domain.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuScalar,
};
#[allow(dead_code)] // Edge constructors are exercised by separate hardware acceptance.
pub trait Scalar: PcuScalar + PcuCheckedFloat + Default + std::fmt::Debug + PartialEq {
    const LABEL: &'static str;
    fn from_units(value: i16, denominator: i16) -> Self;
    fn bits(self) -> u64;
    fn tiny() -> Self;
    fn max() -> Self;
    fn infinity() -> Self;
    fn witness_gradient() -> Self;
}
macro_rules! scalar {
    ($ty:ty, $label:literal, $witness:expr) => {
        impl Scalar for $ty {
            const LABEL: &'static str = $label;
            fn from_units(value: i16, denominator: i16) -> Self {
                Self::from(value) / Self::from(denominator)
            }
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn tiny() -> Self {
                Self::from_bits(1)
            }
            fn max() -> Self {
                Self::MAX
            }
            fn infinity() -> Self {
                Self::INFINITY
            }
            fn witness_gradient() -> Self {
                Self::from_bits($witness)
            }
        }
    };
}
scalar!(f32, "f32", 0x3f7f_fffe);
scalar!(f64, "f64", 0x3fef_ffff_c000_0080);
fn units(index: usize, phase: usize) -> (i16, i16) {
    (
        i16::try_from((index + phase) % 9).unwrap() - 4,
        i16::try_from((index * 3 + phase) % 7).unwrap() - 3,
    )
}
pub fn inputs<T: Scalar, const N: usize>(phase: usize) -> (Box<[T; N]>, Box<[T; N]>, Vec<T>) {
    let mut w = vec![T::default(); N].into_boxed_slice();
    let mut g = vec![T::default(); N].into_boxed_slice();
    let mut expected = Vec::with_capacity(N);
    for i in 0..N {
        let (wu, gu) = units(i, phase);
        w[i] = T::from_units(wu, 8);
        g[i] = T::from_units(gu, 8);
        expected.push(T::from_units(4 * wu + gu, 32));
    }
    (
        w.try_into().ok().unwrap(),
        g.try_into().ok().unwrap(),
        expected,
    )
}
pub fn verify<T: Scalar>(expected: &[T], actual: &[T]) {
    assert_eq!(expected.len(), actual.len());
    for (i, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(expected.bits(), actual.bits(), "native F64 SGD lane {i}");
    }
}
