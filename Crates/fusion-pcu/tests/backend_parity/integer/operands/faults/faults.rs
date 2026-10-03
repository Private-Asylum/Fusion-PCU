//! First-fault and publication checks do not use the arithmetic implementation as an oracle.
use super::*;

fn fault(error: &global::PcuExecutionError, kind: PcuExecutionFaultKind, range: PcuRangePolicy) {
    let observed = error.arithmetic_fault().expect("range fault");
    assert_eq!(observed.kind, kind);
    assert_eq!(observed.invocation_id, 2);
    assert_eq!(observed.recovered, range == PcuRangePolicy::Clamp);
}

fn publication<T: Sample>(
    output: &[T; 9],
    before: &[T; 9],
    clamped: &[T; 7],
    range: PcuRangePolicy,
) {
    if range == PcuRangePolicy::Clamp {
        bits(&output[..7], clamped);
    } else {
        bits(output, before);
    }
    bits(&output[7..], &before[7..]);
}

pub(super) fn repeated<T: Sample>(range: PcuRangePolicy, input: &[T; 7], output: &mut [T; 9]) {
    let empty: [T; 0] = [];
    let mut bad = [T::small(2); 7];
    bad[2] = T::MAX;
    bad[6] = T::MAX;
    let mut clamped = [T::small(4); 7];
    clamped[2] = T::MAX;
    clamped[6] = T::MAX;
    let before = *output;
    fault(
        &source::doubled::<T, 7>(output, &bad).unwrap_err(),
        PcuExecutionFaultKind::ArithmeticOverflow,
        range,
    );
    publication(output, &before, &clamped, range);
    source::doubled::<T, 7>(output, input).unwrap();
    let before = *output;
    fault(
        &source::squared::<T, 7>(&empty, output, &bad).unwrap_err(),
        PcuExecutionFaultKind::ArithmeticOverflow,
        range,
    );
    publication(output, &before, &clamped, range);
    source::squared::<T, 7>(&empty, output, input).unwrap();
}

pub(super) fn broadcast<T: Sample>(range: PcuRangePolicy, input: &[T; 7], output: &mut [T; 9]) {
    let empty: [T; 0] = [];
    let mut bad = [T::small(2); 7];
    bad[2] = T::MIN;
    bad[6] = T::MIN;
    let mut clamped = [T::small(0); 7];
    clamped[2] = T::MIN;
    clamped[6] = T::MIN;
    let before = *output;
    fault(
        &source::independent::<T, 7>(&bad, output).unwrap_err(),
        PcuExecutionFaultKind::ArithmeticUnderflow,
        range,
    );
    publication(output, &before, &clamped, range);
    source::independent::<T, 7>(input, output).unwrap();
    let before = *output;
    let mut bad = [T::MIN; 7];
    bad[2] = T::small(1);
    bad[6] = T::small(2);
    fault(
        &source::grid::<T, 7>(&empty, output, &bad).unwrap_err(),
        PcuExecutionFaultKind::ArithmeticUnderflow,
        range,
    );
    publication(output, &before, &clamped, range);
    source::grid::<T, 7>(&empty, output, &[T::small(3); 7]).unwrap();
    bits(&output[..7], &[T::small(0); 7]);
}
