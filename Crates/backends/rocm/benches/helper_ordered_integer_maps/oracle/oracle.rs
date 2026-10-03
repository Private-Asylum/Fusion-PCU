//! Independent U128/I128 arithmetic, with representable inputs and exact signed bounds.
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuRangePolicy,
};
pub trait Format: PcuScalar + std::fmt::Debug + PartialEq {
    const LABEL: &'static str;
    const SIGNED: bool;
    const MAX: u64;
    fn bits(self) -> u64;
    fn from(bits: u64) -> Self;
    fn sentinel() -> Self {
        Self::from(17)
    }
}
impl Format for u64 {
    const LABEL: &'static str = "u64";
    const SIGNED: bool = false;
    const MAX: Self = Self::MAX;
    fn bits(self) -> u64 {
        self
    }
    fn from(bits: u64) -> Self {
        bits
    }
}
impl Format for i64 {
    const LABEL: &'static str = "i64";
    const SIGNED: bool = true;
    const MAX: u64 = (1_u64 << 63) - 1;
    fn bits(self) -> u64 {
        u64::from_ne_bytes(self.to_ne_bytes())
    }
    fn from(bits: u64) -> Self {
        Self::from_ne_bytes(bits.to_ne_bytes())
    }
}
fn calculate<T: Format>(a: T, b: T, multiply: bool) -> (T, u64) {
    if T::SIGNED {
        let a = i128::from(i64::from_ne_bytes(a.bits().to_ne_bytes()));
        let b = i128::from(i64::from_ne_bytes(b.bits().to_ne_bytes()));
        let result = if multiply { a * b } else { a + b };
        let code = if result < i128::from(i64::MIN) {
            4
        } else if result > i128::from(i64::MAX) {
            3
        } else {
            0
        };
        let bounded =
            i64::try_from(result.clamp(i128::from(i64::MIN), i128::from(i64::MAX))).unwrap();
        (T::from(u64::from_ne_bytes(bounded.to_ne_bytes())), code)
    } else {
        let a = u128::from(a.bits());
        let b = u128::from(b.bits());
        let result = if multiply { a * b } else { a + b };
        (
            T::from(u64::try_from(result.min(u128::from(u64::MAX))).unwrap()),
            <u64 as From<bool>>::from(result > u128::from(u64::MAX)) * 3,
        )
    }
}
pub fn expected<T: Format>(input: &[T], range: PcuRangePolicy) -> (u64, Vec<T>, Vec<T>) {
    let mut word = u64::MAX;
    let mut stage = Vec::new();
    let mut output = Vec::new();
    for (lane, original) in input.iter().copied().enumerate() {
        let (sum, first) = calculate(original, original, false);
        let (product, second) = calculate(sum, original, true);
        let code = if first != 0 { first } else { second };
        if code != 0 {
            let status = (u64::try_from(lane).unwrap() << 3)
                | code
                | if range == PcuRangePolicy::Clamp {
                    1_u64 << 63
                } else {
                    0
                };
            word = word.min(status);
        }
        stage.push(sum);
        output.push(product);
    }
    (word, stage, output)
}
pub fn bank<T: Format>(count: usize, phase: usize) -> (Vec<T>, Vec<T>, Vec<T>) {
    let input: Vec<T> = (0..count)
        .map(|lane| {
            let top = if T::SIGNED {
                (1_u64 << 31) - 1
            } else {
                3_037_000_499
            };
            let magnitude = match (lane + phase) % 4 {
                0 => top,
                1 => top - 1 - u64::try_from(phase).unwrap(),
                _ => 1 + u64::try_from((lane + 17 * phase) % 997).unwrap(),
            };
            T::from(if T::SIGNED && (lane + phase).is_multiple_of(2) {
                0_u64.wrapping_sub(magnitude)
            } else {
                magnitude
            })
        })
        .collect();
    let (word, stage, output) = expected(&input, PcuRangePolicy::Reject);
    assert_eq!(word, u64::MAX);
    (input, stage, output)
}
pub fn verify<T: Format>(expected: &[T], actual: &[T]) {
    assert_eq!(actual.len(), expected.len() + 2);
    assert_eq!(&actual[..expected.len()], expected);
    assert_eq!(&actual[expected.len()..], &[T::sentinel(); 2]);
}
