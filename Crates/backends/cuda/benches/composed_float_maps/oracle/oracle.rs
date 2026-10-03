//! Independent exact dyadic composed dyadic map bits; no backend/core result oracle.
#[path = "../../checked_relu_backward/oracle/oracle.rs"]
#[allow(dead_code)] // Derivative-only helpers are outside this scalar-map fixture.
mod representation;
pub use representation::Format;
pub fn inputs<T: Format>(count: usize, phase: usize) -> (Vec<T>, Vec<T>) {
    let mut input = Vec::with_capacity(count);
    let mut expected = Vec::with_capacity(count);
    for lane in 0..count {
        let exponent = (lane + phase) % 3;
        let raw = match exponent {
            0 => T::ONE - T::MIN_NORMAL,
            1 => T::ONE,
            _ => T::ONE + T::MIN_NORMAL,
        };
        input.push(T::from(
            raw | if (lane + phase).is_multiple_of(2) {
                0
            } else {
                T::SIGN
            },
        ));
        // (x+x)*x = 2*x^2 for x in {+/-0.5,+/-1,+/-2}.
        // Each intermediate is exactly representable in all four named formats.
        let result = match exponent {
            0 => T::ONE - T::MIN_NORMAL,
            1 => T::ONE + T::MIN_NORMAL,
            _ => T::ONE + 3 * T::MIN_NORMAL,
        };
        expected.push(T::from(result));
    }
    (input, expected)
}
pub fn verify<T: Format>(expected: &[T], observed: &[T]) {
    assert_eq!(observed.len(), expected.len() + 2);
    for (want, actual) in expected.iter().zip(observed) {
        assert_eq!(actual.bits(), want.bits());
    }
    for tail in &observed[expected.len()..] {
        assert_eq!(tail.bits(), T::sentinel().bits());
    }
}

#[path = "independent/independent.rs"]
pub mod independent;
