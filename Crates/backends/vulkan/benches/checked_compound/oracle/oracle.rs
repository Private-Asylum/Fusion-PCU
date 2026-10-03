//! Independent small exact dyadic compounds; signed integer sums never exceed 2^20.
#[rustfmt::skip]
use pcu_facade::{
    PcuScalar,
    dialect::tensor::TensorElement,
};
pub trait Format: PcuScalar + TensorElement {
    fn value(value: f32) -> Self;
    fn maximum() -> Self;
    fn minimum() -> Self;
    fn bits(self) -> u64;
    fn third_loss() -> Self;
}
impl Format for f32 {
    fn value(value: f32) -> Self {
        value
    }
    fn maximum() -> Self {
        Self::MAX
    }
    fn minimum() -> Self {
        Self::from_bits(1)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn third_loss() -> Self {
        Self::from_bits(0x4095_5555)
    }
}
impl Format for f64 {
    fn value(value: f32) -> Self {
        Self::from(value)
    }
    fn maximum() -> Self {
        Self::MAX
    }
    fn minimum() -> Self {
        Self::from_bits(1)
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn third_loss() -> Self {
        Self::from_bits(0x4012_aaaa_aaaa_aaab)
    }
}
pub struct Bank<T> {
    pub left: [[T; 3]; 2],
    pub right: [[T; 2]; 3],
    pub weights: [[T; 2]; 2],
    pub gradient: [[T; 2]; 2],
    pub product: [T; 4],
    pub update: [T; 4],
    pub loss: T,
}
pub fn banks<T: Format>(phase: i16) -> Bank<T> {
    let mut a: [[i16; 3]; 2] = core::array::from_fn(|row| {
        core::array::from_fn(|column| i16::try_from(row * 3 + column).unwrap() - 3 + phase)
    });
    a[0][0] += phase;
    let b: [[i16; 2]; 3] = [[2, -1], [-3, 2], [1, 4]];
    let x: [[i16; 2]; 2] = [[phase + 2, -3], [4, phase - 2]];
    let y: [[i16; 2]; 2] = [[1, 2], [-1, 3]];
    // All products/sums and the /4 MSE denominator are exact small dyadics in both formats.
    let product = core::array::from_fn(|lane| {
        let integer: i16 = (0..3)
            .map(|inner| a[lane / 2][inner] * b[inner][lane % 2])
            .sum();
        T::value(f32::from(integer))
    });
    let update = core::array::from_fn(|lane| {
        T::value(f32::from(x[lane / 2][lane % 2]) - f32::from(y[lane / 2][lane % 2]) / 2.0)
    });
    let squared: i16 = x
        .iter()
        .flatten()
        .zip(y.iter().flatten())
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    Bank {
        left: a.map(|row| row.map(|x| T::value(f32::from(x)))),
        right: b.map(|row| row.map(|x| T::value(f32::from(x)))),
        weights: x.map(|row| row.map(|x| T::value(f32::from(x)))),
        gradient: y.map(|row| row.map(|x| T::value(f32::from(x)))),
        product,
        update,
        loss: T::value(f32::from(squared) / 4.0),
    }
}
