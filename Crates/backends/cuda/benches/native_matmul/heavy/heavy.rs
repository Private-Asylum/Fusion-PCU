//! Controlled split-depth rank-two matrices with an independent complete-output integer oracle.
use super::oracle::Scalar;

pub trait HeavyScalar: Scalar {
    fn integer(value: u32) -> Self;
}

impl HeavyScalar for f32 {
    #[allow(clippy::cast_precision_loss)] // The asserted domain contains exactly representable F32 integers.
    fn integer(value: u32) -> Self {
        assert!(value <= 1 << 24);
        value as Self
    }
}

impl HeavyScalar for f64 {
    fn integer(value: u32) -> Self {
        Self::from(value)
    }
}

fn matrix<T: Scalar, const R: usize, const C: usize>() -> Box<[[T; C]; R]> {
    // One row is the largest stack temporary; the whole matrix is constructed in heap storage.
    vec![[T::default(); C]; R]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| panic!("fixed matrix row count"))
}

fn code(index: usize, phase: u64) -> u32 {
    u32::try_from(u64::try_from(index).unwrap() + 1 + phase).unwrap()
}

fn weight(depth: usize, phase: u64, period: u64) -> u32 {
    u32::try_from((u64::try_from(depth).unwrap() + phase) % period + 1).unwrap()
}

pub fn fill<T: HeavyScalar, const R: usize, const K: usize, const C: usize>(
    phase: u64,
) -> super::oracle::MatrixInputs<T, R, K, C> {
    assert!(phase < 3);
    // The TF32 inputs are also exact: every source integer needs at most eleven significant bits.
    assert!(code(R - 1, phase).max(code(C - 1, phase)) <= 1 << 11);
    let mut left = matrix::<T, R, K>();
    let mut right = matrix::<T, K, C>();
    let split = K / 3;
    for (row, values) in left.iter_mut().enumerate() {
        for (depth, value) in values.iter_mut().enumerate() {
            *value = T::integer(if depth < split {
                code(row, phase)
            } else {
                weight(depth, phase, 2)
            });
        }
    }
    for (depth, values) in right.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = T::integer(if depth < split {
                weight(depth, phase, 3)
            } else {
                code(column, phase)
            });
        }
    }
    (left, right)
}

pub fn expected<T: HeavyScalar, const R: usize, const K: usize, const C: usize>(
    phase: u64,
) -> Vec<T> {
    let split = K / 3;
    let first: u64 = (0..split)
        .map(|depth| u64::from(weight(depth, phase, 3)))
        .sum();
    let second: u64 = (split..K)
        .map(|depth| u64::from(weight(depth, phase, 2)))
        .sum();
    let maximum = u64::from(code(R - 1, phase)) * first + u64::from(code(C - 1, phase)) * second;
    // All products and partial sums are positive integers below this bound, so any reduction
    // grouping is exact in F32/F64. This fixture proof is not a general vendor guarantee.
    assert!(maximum <= 1 << 24);
    (0..R * C)
        .map(|cell| {
            let sum = u64::from(code(cell / C, phase)) * first
                + u64::from(code(cell % C, phase)) * second;
            T::integer(u32::try_from(sum).unwrap())
        })
        .collect()
}

pub fn preflight<T: HeavyScalar>() {
    // Independent ordered PCU scalar operations verify every cell of three nonsquare witnesses.
    for phase in 0..3 {
        let (left, right) = fill::<T, 5, 11, 7>(phase);
        super::oracle::verify(
            &expected::<T, 5, 11, 7>(phase),
            &super::oracle::expected(&left, &right),
        );
    }
}
