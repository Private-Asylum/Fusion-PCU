//! Independent encoding-only selection, without core/backend arithmetic results.
#[path = "../../checked_relu_backward/oracle/oracle.rs"]
#[allow(dead_code)] // Derivative-only reference helpers are outside this unary proof.
mod representation;
pub use representation::Format;
#[rustfmt::skip]
use fusion_pcu::{
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
};
pub fn expected<T: Format>(
    input: T,
    operation: u32,
    uf: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> (u64, Option<T>) {
    let raw = input.bits();
    if raw & (T::SIGN - 1) > T::MAX {
        return (5, None);
    }
    let selected = if operation == 0 {
        raw ^ T::SIGN
    } else if raw & T::SIGN == 0 && raw & (T::SIGN - 1) != 0 {
        raw
    } else {
        0
    };
    let magnitude = selected & (T::SIGN - 1);
    if uf == PcuFloatUnderflowPolicy::RejectSubnormalResult
        && magnitude != 0
        && magnitude < T::MIN_NORMAL
    {
        if range == PcuRangePolicy::Reject {
            (4, None)
        } else {
            ((1 << 63) | 4, Some(T::from(selected)))
        }
    } else {
        (u64::MAX, Some(T::from(selected)))
    }
}
pub fn inputs<T: Format>(
    count: usize,
    phase: usize,
    operation: u32,
    uf: PcuFloatUnderflowPolicy,
) -> (Vec<T>, Vec<T>) {
    let samples = [
        0,
        T::SIGN,
        T::ONE,
        T::SIGN | T::ONE,
        T::MAX,
        T::SIGN | T::MAX,
        T::MIN_NORMAL,
        T::SIGN | T::MIN_NORMAL,
        1,
        T::SIGN | 1,
    ];
    let mut inputs = Vec::with_capacity(count);
    let mut results = Vec::with_capacity(count);
    for lane in 0..count {
        let mut input = T::from(samples[(lane + phase) % samples.len()]);
        if expected(input, operation, uf, PcuRangePolicy::Reject).0 != u64::MAX {
            input = T::one();
        }
        inputs.push(input);
        results.push(
            expected(input, operation, uf, PcuRangePolicy::Reject)
                .1
                .unwrap(),
        );
    }
    (inputs, results)
}
pub fn verify<T: Format>(expected: &[T], actual: &[T]) {
    assert_eq!(actual.len(), expected.len() + 2);
    for (want, have) in expected.iter().zip(actual) {
        assert_eq!(want.bits(), have.bits());
    }
    for tail in &actual[expected.len()..] {
        assert_eq!(tail.bits(), T::sentinel().bits());
    }
}

/// Full encodings for the four small formats; deterministic full-width samples for F32/F64.
pub fn corpus<T: Format, const N: usize>() -> Vec<T> {
    let specials = [
        0,
        T::SIGN,
        1,
        T::SIGN | 1,
        T::MIN_NORMAL - 1,
        T::SIGN | (T::MIN_NORMAL - 1),
        T::MIN_NORMAL,
        T::ONE,
        T::MAX,
        T::SIGN | T::MAX,
        T::MAX + 1,
        T::SIGN | (T::MAX + 1),
        T::SIGN - 1,
        (T::SIGN - 1) | T::SIGN,
    ];
    let mut state = 0x71d2_94ae_663c_a9f1_u64;
    (0..N)
        .map(|lane| {
            let raw = if T::SIGN <= 0x8000 {
                u64::try_from(lane).unwrap()
            } else if lane < specials.len() {
                specials[lane]
            } else {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state & (T::SIGN | (T::SIGN - 1))
            };
            T::from(raw)
        })
        .collect()
}
