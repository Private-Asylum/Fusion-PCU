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
    fn ratio(numerator: u32, denominator: u32) -> Self;
    fn tiny() -> Self;
    fn subnormal_square_root() -> Self;
    fn large_square_root() -> Self;
    fn max() -> Self;
    fn infinity() -> Self;
    fn witness_gradient() -> Self;
}
macro_rules! scalar {
    ($ty:ty, $label:literal, $witness:expr, $tiny_root:expr, $large_root:expr) => {
        impl Scalar for $ty {
            const LABEL: &'static str = $label;
            fn from_units(value: i16, denominator: i16) -> Self {
                Self::from(value) / Self::from(denominator)
            }
            #[allow(clippy::cast_precision_loss, clippy::cast_lossless)] // Inputs are bounded exact integers; one division rounds the independently accumulated ratio.
            fn ratio(numerator: u32, denominator: u32) -> Self {
                assert!(numerator <= (1 << 24) && denominator <= (1 << 24));
                numerator as Self / denominator as Self
            }
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn subnormal_square_root() -> Self {
                Self::from_bits($tiny_root)
            }
            fn large_square_root() -> Self {
                Self::from_bits($large_root)
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
scalar!(f32, "f32", 0x3f7f_fffe, 53 << 23, (190 << 23) | (1 << 22));
scalar!(
    f64,
    "f64",
    0x3fef_ffff_c000_0080,
    486 << 52,
    (1534 << 52) | (1 << 51)
);
fn units(index: usize, phase: usize) -> (i16, i16) {
    (
        i16::try_from((index + phase) % 9).unwrap() - 4,
        i16::try_from((index * 3 + phase) % 7).unwrap() - 3,
    )
}
pub fn inputs<T: Scalar, const N: usize>(phase: usize) -> (Box<[T; N]>, Box<[T; N]>, Vec<T>) {
    let mut w = vec![T::default(); N].into_boxed_slice();
    let mut g = vec![T::default(); N].into_boxed_slice();
    let mut sum = 0_u32;
    for i in 0..N {
        let (wu, gu) = units(i, phase);
        w[i] = T::from_units(wu, 8);
        g[i] = T::from_units(gu, 8);
        let difference = i32::from(wu) - i32::from(gu);
        sum += u32::try_from(difference * difference).unwrap();
    }
    let expected = vec![T::ratio(sum, u32::try_from(N * 64).unwrap())];
    (
        w.try_into().ok().unwrap(),
        g.try_into().ok().unwrap(),
        expected,
    )
}
pub fn verify<T: Scalar>(expected: &[T], actual: &[T]) {
    assert_eq!(expected.len(), actual.len());
    for (i, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(expected.bits(), actual.bits(), "derivative lane {i}");
    }
}
