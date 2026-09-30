//! Bit-exact ordered scalar law; no widened accumulation or tolerance conceals faults.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuScalar,
};

pub trait Scalar: PcuScalar + PcuCheckedFloat + Default + core::fmt::Debug {
    const LABEL: &'static str;
    fn small(value: i16) -> Self;
    fn bits(self) -> u64;
    fn min_subnormal() -> Self;
    fn max() -> Self;
    fn epsilon() -> Self;
    fn infinity() -> Self;
}

macro_rules! scalar {
    ($ty:ty, $label:literal) => {
        impl Scalar for $ty {
            const LABEL: &'static str = $label;
            fn small(value: i16) -> Self {
                Self::from(value)
            }
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn min_subnormal() -> Self {
                Self::from_bits(1)
            }
            fn max() -> Self {
                Self::MAX
            }
            fn epsilon() -> Self {
                Self::EPSILON
            }
            fn infinity() -> Self {
                Self::INFINITY
            }
        }
    };
}
scalar!(f32, "f32");
scalar!(f64, "f64");

pub type MatrixInputs<T, const R: usize, const K: usize, const C: usize> =
    (Box<[[T; K]; R]>, Box<[[T; C]; K]>);

pub fn fill<T: Scalar, const R: usize, const K: usize, const C: usize>(
    job: u64,
) -> MatrixInputs<T, R, K, C> {
    let mut left = Box::new([[T::default(); K]; R]);
    let mut right = Box::new([[T::default(); C]; K]);
    for (index, value) in left.as_flattened_mut().iter_mut().enumerate() {
        let value_index = (u64::try_from(index).unwrap() + job) % 9;
        *value = T::small(i16::try_from(value_index).unwrap() - 4)
            .pcu_checked_div(T::small(8))
            .unwrap();
    }
    for (index, value) in right.as_flattened_mut().iter_mut().enumerate() {
        let value_index = (u64::try_from(index).unwrap() * 3 + job) % 7;
        *value = T::small(i16::try_from(value_index).unwrap() - 3)
            .pcu_checked_div(T::small(8))
            .unwrap();
    }
    (left, right)
}

pub fn expected<T: Scalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Vec<T> {
    let mut data = vec![T::default(); R * C];
    for row in 0..R {
        for column in 0..C {
            for depth in 0..K {
                let product = left[row][depth]
                    .pcu_checked_mul(right[depth][column])
                    .unwrap();
                data[row * C + column] = data[row * C + column].pcu_checked_add(product).unwrap();
            }
        }
    }
    data
}

pub fn verify<T: Scalar>(expected: &[T], actual: &[T]) {
    assert_eq!(expected.len(), actual.len());
    for (index, (&wanted, &observed)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(wanted.bits(), observed.bits(), "cell {index}");
    }
}
