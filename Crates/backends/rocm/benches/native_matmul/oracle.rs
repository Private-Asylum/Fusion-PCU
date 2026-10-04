//! Exact bounded integer fixtures; this does not promise general vendor bit determinism.
use fusion_pcu::PcuScalar;
pub trait Scalar: PcuScalar + Default + core::fmt::Debug {
    const LABEL: &'static str;
    fn integer(value: u32) -> Self;
    fn bits(self) -> u64;
}
#[allow(clippy::cast_precision_loss)] // The explicit bound admits only integers exactly representable in F32.
fn f32_integer(value: u32) -> f32 {
    assert!(value <= 1 << 24);
    value as f32
}
macro_rules! scalar {
    ($ty:ty, $label:literal, $convert:path) => {
        impl Scalar for $ty {
            const LABEL: &'static str = $label;
            fn integer(value: u32) -> Self {
                $convert(value)
            }
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
        }
    };
}
scalar!(f32, "f32", f32_integer);
scalar!(f64, "f64", f64::from);
pub type MatrixInputs<T, const R: usize, const K: usize, const C: usize> =
    (Box<[[T; K]; R]>, Box<[[T; C]; K]>);
#[allow(clippy::unnecessary_box_returns)] // Cold fixed matrices stay on the heap to bound stack use.
fn matrix<T: Scalar, const R: usize, const C: usize>() -> Box<[[T; C]; R]> {
    // Construct only a single row on the stack. Vec repeats that row directly into heap storage.
    vec![[T::default(); C]; R]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| panic!("fixed matrix row count"))
}
fn left_value(index: usize, phase: u64) -> u32 {
    u32::try_from((u64::try_from(index).unwrap() + phase) % 7 + 1).unwrap()
}
fn right_value(index: usize, phase: u64) -> u32 {
    u32::try_from((u64::try_from(index).unwrap() * 3 + phase) % 5 + 1).unwrap()
}
fn code(index: usize, phase: u64) -> u32 {
    u32::try_from(u64::try_from(index).unwrap() + 1 + phase).unwrap()
}
fn weight(depth: usize, phase: u64, period: u64) -> u32 {
    u32::try_from((u64::try_from(depth).unwrap() + phase) % period + 1).unwrap()
}
pub const fn phase_count<const K: usize>() -> u64 {
    if K > 256 { 3 } else { 35 }
}
pub fn fill<T: Scalar, const R: usize, const K: usize, const C: usize>(
    phase: u64,
) -> MatrixInputs<T, R, K, C> {
    let mut left = matrix::<T, R, K>();
    let mut right = matrix::<T, K, C>();
    if K > 256 {
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
    } else {
        for (index, value) in left.as_flattened_mut().iter_mut().enumerate() {
            *value = T::integer(left_value(index, phase));
        }
        for (index, value) in right.as_flattened_mut().iter_mut().enumerate() {
            *value = T::integer(right_value(index, phase));
        }
    }
    (left, right)
}
pub fn expected<T: Scalar, const R: usize, const K: usize, const C: usize>(phase: u64) -> Vec<T> {
    if K > 256 {
        // Split-depth rank two: sum(row_code * first_weight) + sum(second_weight * column_code).
        // Compute the two independent integer moments, then every output cell in O(K + R*C).
        let split = K / 3;
        let first: u64 = (0..split)
            .map(|depth| u64::from(weight(depth, phase, 3)))
            .sum();
        let second: u64 = (split..K)
            .map(|depth| u64::from(weight(depth, phase, 2)))
            .sum();
        let maximum =
            u64::from(code(R - 1, phase)) * first + u64::from(code(C - 1, phase)) * second;
        // All products/partial sums are positive integers <= maximum, so every possible
        // reduction grouping is exact. This is a controlled fixture domain, not a vendor promise.
        assert!(maximum <= 1 << 24);
        (0..R * C)
            .map(|cell| {
                let sum = u64::from(code(cell / C, phase)) * first
                    + u64::from(code(cell % C, phase)) * second;
                T::integer(u32::try_from(sum).unwrap())
            })
            .collect()
    } else {
        // Positive products <=35 and K<=256 give exact sums <=8960 for both widths.
        (0..R * C)
            .map(|cell| {
                let row = cell / C;
                let column = cell % C;
                let sum: u32 = (0..K)
                    .map(|depth| {
                        left_value(row * K + depth, phase) * right_value(depth * C + column, phase)
                    })
                    .sum();
                T::integer(sum)
            })
            .collect()
    }
}
pub fn verify<T: Scalar>(expected: &[T], actual: &[T]) {
    assert_eq!(expected.len(), actual.len());
    for (index, (&wanted, &observed)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(wanted.bits(), observed.bits(), "cell {index}");
    }
}
