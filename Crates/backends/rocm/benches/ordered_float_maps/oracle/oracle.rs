//! Independent exact dyadics for both observable stores.
#[path = "../../checked_relu_backward/oracle/oracle.rs"]
#[allow(dead_code)] // Only scalar encoding metadata is needed here.
mod representation;
pub use representation::Format;
pub fn bank<T: Format>(count: usize, phase: usize) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut input = Vec::with_capacity(count);
    let mut stage = Vec::with_capacity(count);
    let mut output = Vec::with_capacity(count);
    for lane in 0..count {
        let power = (lane + phase) % 3;
        let raw = match power {
            0 => T::ONE - T::MIN_NORMAL,
            1 => T::ONE,
            _ => T::ONE + T::MIN_NORMAL,
        };
        let sign = if (lane + phase).is_multiple_of(2) {
            0
        } else {
            T::SIGN
        };
        input.push(T::from(raw | sign));
        stage.push(T::from((raw + T::MIN_NORMAL) | sign));
        output.push(T::from(match power {
            0 => T::ONE - T::MIN_NORMAL,
            1 => T::ONE + T::MIN_NORMAL,
            _ => T::ONE + 3 * T::MIN_NORMAL,
        }));
    }
    (input, stage, output)
}
pub fn verify<T: Format>(expected: &[T], actual: &[T]) {
    assert_eq!(actual.len(), expected.len() + 2);
    for (want, got) in expected.iter().zip(actual) {
        assert_eq!(want.bits(), got.bits());
    }
    for tail in &actual[expected.len()..] {
        assert_eq!(tail.bits(), T::sentinel().bits());
    }
}
