//! Independent base-256 results are formed cold, never via the backend arithmetic.
#[path = "../../wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)] // Shared frozen high-limb oracle also supplies exceptional native fixtures.
mod shared;
pub use shared::Integer as Format;
pub fn inputs<T: Format>(count: usize, phase: u64, kind: u32) -> (Vec<T>, Vec<T>, Vec<T>, Vec<T>) {
    let (mut left, right) = shared::inputs::<T>(count, phase);
    for value in &mut left {
        if *value == T::ZERO {
            *value = T::ONE;
        }
    }
    if matches!(kind, 3 | 4) {
        left[0] = T::ONE;
    }
    let mut quotient = Vec::with_capacity(count);
    let mut remainder = Vec::with_capacity(count);
    for id in 0..count {
        let (lhs, rhs) = match kind {
            0 | 1 => (left[id], left[id]),
            2 => (left[id], right[id]),
            3 => (left[id], left[0]),
            4 => (left[0], left[id]),
            _ => unreachable!(),
        };
        let pair = shared::evaluate(lhs, rhs).unwrap();
        quotient.push(pair.0);
        remainder.push(pair.1);
    }
    (left, right, quotient, remainder)
}
pub fn verify<T: Format>(expected_q: &[T], expected_r: &[T], quotient: &[T], remainder: &[T]) {
    assert_eq!(expected_q, &quotient[..expected_q.len()]);
    assert_eq!(expected_r, &remainder[..expected_r.len()]);
    assert!(
        quotient[expected_q.len()..]
            .iter()
            .all(|v| *v == T::SENTINEL)
    );
    assert!(
        remainder[expected_r.len()..]
            .iter()
            .all(|v| *v == T::SENTINEL)
    );
}
