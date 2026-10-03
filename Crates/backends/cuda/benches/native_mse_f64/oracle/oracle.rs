//! Independent integer/dyadic complete-output oracle for the benchmark domain.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuScalar,
};
pub trait Scalar: PcuScalar + PcuCheckedFloat + Default + std::fmt::Debug + PartialEq {
    const LABEL: &'static str;
    fn from_units(value: i16, denominator: i16) -> Self;
    fn bits(self) -> u64;
    fn native_scaled_sum(numerator: u32, count: u32) -> Self;
}
impl Scalar for f64 {
    const LABEL: &'static str = "f64";
    fn from_units(value: i16, denominator: i16) -> Self {
        Self::from(value) / Self::from(denominator)
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn native_scaled_sum(numerator: u32, count: u32) -> Self {
        assert!(numerator <= (1 << 28) && count > 0 && count <= (1 << 24));
        // Integer difference-square accumulation is independent of the device kernel.
        // Every dyadic square and positive partial sum fits exactly in F64. Core's
        // integer-significand arithmetic independently rounds the native reciprocal/scale.
        Self::from(numerator)
            .pcu_checked_div(64.0)
            .unwrap()
            .pcu_checked_mul(1.0.pcu_checked_div(Self::from(count)).unwrap())
            .unwrap()
    }
}
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
    let expected = vec![T::native_scaled_sum(sum, u32::try_from(N).unwrap())];
    (
        w.try_into().ok().unwrap(),
        g.try_into().ok().unwrap(),
        expected,
    )
}
pub fn verify<T: Scalar>(expected: &[T], actual: &[T]) {
    assert_eq!(expected.len(), actual.len());
    for (i, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(expected.bits(), actual.bits(), "native loss scalar {i}");
    }
}
