//! Automatic borrowed unary prefixes preserve escaped full-capacity owners.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use source::{
    direct_neg,
    direct_relu,
    grid_neg,
    grid_relu,
    broadcast_neg,
    broadcast_relu,
    clamp_direct_neg,
    clamp_direct_relu,
    clamp_grid_neg,
    clamp_grid_relu,
    clamp_broadcast_neg,
    clamp_broadcast_relu,
};

// Source-local flags cover borrowed Clamp without a global owned-graph Clamp request.
macro_rules! invoke {
    ($neg:ident, $relu:ident, $clamp_neg:ident, $clamp_relu:ident, $input:expr, $output:expr, $negate:expr, $range:expr) => {
        match ($negate, $range) {
            (true, PcuRangePolicy::Clamp) => $clamp_neg($input, $output),
            (false, PcuRangePolicy::Clamp) => $clamp_relu($input, $output),
            (true, _) => $neg($input, $output),
            (false, _) => $relu($input, $output),
        }
    };
}

fn check<T: Sample>(values: &[T], expected: &[T; 7], sentinel: T) {
    bits(&values[..7], expected);
    bits(&values[7..], &vec![sentinel; values.len() - 7]);
}

fn publication(
    result: Result<(), PcuExecutionError>,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> bool {
    if underflow != PcuFloatUnderflowPolicy::RejectSubnormalResult {
        result.unwrap();
        return true;
    }
    let error = result.unwrap_err();
    assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow
        && fault.invocation_id == 4 && fault.recovered == (range == PcuRangePolicy::Clamp)));
    range == PcuRangePolicy::Clamp
}

fn variant<T: Sample>(
    input: &PcuTensor<T>,
    full: &[T; 11],
    expected: &[T; 7],
    negate: bool,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> (PcuTensor<T>, T) {
    let sentinel = T::from_raw(T::NORMAL);
    let unchanged = [sentinel; 7];
    let mut stack = [sentinel; 9];
    let result = invoke!(
        direct_neg,
        direct_relu,
        clamp_direct_neg,
        clamp_direct_relu,
        input,
        &mut stack,
        negate,
        range
    );
    let publish = publication(result, underflow, range);
    check(
        &stack,
        if publish { expected } else { &unchanged },
        sentinel,
    );
    stack.fill(sentinel);
    let result = invoke!(
        grid_neg,
        grid_relu,
        clamp_grid_neg,
        clamp_grid_relu,
        input,
        &mut stack,
        negate,
        range
    );
    let publish = publication(result, underflow, range);
    check(
        &stack,
        if publish { expected } else { &unchanged },
        sentinel,
    );

    stack.fill(sentinel);
    invoke!(
        broadcast_neg,
        broadcast_relu,
        clamp_broadcast_neg,
        clamp_broadcast_relu,
        input,
        &mut stack,
        negate,
        range
    )
    .unwrap();
    check(&stack, &[expected[0]; 7], sentinel);

    owned_prefix(input, full, expected, negate, underflow, range)
}

fn owned_prefix<T: Sample>(
    input: &PcuTensor<T>,
    full: &[T; 11],
    expected: &[T; 7],
    negate: bool,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> (PcuTensor<T>, T) {
    let sentinel = T::from_raw(T::NORMAL);
    let unchanged = [sentinel; 7];
    let mut output = retain(&[sentinel; 11]).unwrap();
    let result = invoke!(
        direct_neg,
        direct_relu,
        clamp_direct_neg,
        clamp_direct_relu,
        input,
        &mut output,
        negate,
        range
    );
    let publish = publication(result, underflow, range);
    let mut host = [sentinel; 13];
    if publish {
        output.read_into(&mut host).unwrap();
        check(&host, expected, sentinel);
    } else if super::super::publication::discarded_after_terminal_fault(
        &output, &mut host, bits::<T>,
    ) {
        // An actually written failed resident value cannot be read or retried.
        // A fresh owner restores a usable output without exposing faulted bytes.
        output = retain(&[sentinel; 11]).unwrap();
    } else {
        check(&host, &unchanged, sentinel);
    }
    assert_eq!(output.shape(), &[11]);

    // Host-minimum calls must not reuse a full resident specialization.
    let healthy = [full[0]; 7];
    let healthy_expected = [expected[0]; 7];
    invoke!(
        direct_neg,
        direct_relu,
        clamp_direct_neg,
        clamp_direct_relu,
        &healthy,
        &mut output,
        negate,
        range
    )
    .unwrap();
    host.fill(sentinel);
    output.read_into(&mut host).unwrap();
    check(&host, &healthy_expected, sentinel);
    let mut short = [sentinel; 6];
    let result = invoke!(
        direct_neg,
        direct_relu,
        clamp_direct_neg,
        clamp_direct_relu,
        input,
        &mut short,
        negate,
        range
    );
    assert!(result.is_err());
    bits(&short, &[sentinel; 6]);

    fatal_rollback(full, &mut output, negate, underflow, range);
    host.fill(sentinel);
    if super::super::publication::discarded_after_terminal_fault(&output, &mut host, bits::<T>) {
        let restored = core::array::from_fn::<_, 11, _>(|index| {
            if index < 7 {
                healthy_expected[index]
            } else {
                sentinel
            }
        });
        output = retain(&restored).unwrap();
    } else {
        check(&host, &healthy_expected, sentinel);
    }
    global::clear_thread_cache().unwrap();
    output.read_into(&mut host).unwrap();
    check(&host, &healthy_expected, sentinel);
    (output, healthy_expected[0])
}

fn fatal_rollback<T: Sample>(
    full: &[T; 11],
    output: &mut PcuTensor<T>,
    negate: bool,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) {
    let mut bad = *full;
    bad[0] = T::from_raw(1); // Tight Clamp can recover before the later fatal lane.
    bad[2] = T::from_raw(T::NONFINITE);
    let bad = retain(&bad).unwrap();
    let result = invoke!(
        direct_neg,
        direct_relu,
        clamp_direct_neg,
        clamp_direct_relu,
        &bad,
        output,
        negate,
        range
    );
    let error = result.unwrap_err();
    // Reject makes the earlier tiny lane fatal itself. Clamp lets it
    // recover, so the later nonfinite operand becomes the fatal notice.
    let (kind, invocation) = if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult
        && range == PcuRangePolicy::Reject
    {
        (PcuExecutionFaultKind::ArithmeticUnderflow, 0)
    } else {
        (PcuExecutionFaultKind::InvalidFloatingOperand, 2)
    };
    assert!(
        matches!(error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == kind && fault.invocation_id == invocation && !fault.recovered),
        "{error:?}; scalar={:?}, negate={negate}, underflow={underflow:?}, range={range:?}",
        T::TYPE
    );
    drop(bad);
}

fn format<T: Sample>(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    let sentinel = T::from_raw(T::NORMAL);
    for phase in 0..3 {
        let raw = [
            T::NORMAL + phase,
            T::SIGN | T::NORMAL,
            0,
            T::SIGN,
            1,
            T::SIGN | 1,
            T::NORMAL,
            T::NONFINITE,
            T::NONFINITE | T::SIGN,
            T::NORMAL,
            0,
        ];
        let full = raw.map(T::from_raw);
        let input = retain(&full).unwrap();
        let survivors = [false, true].map(|negate| {
            let expected = core::array::from_fn(|id| {
                T::from_raw(if negate {
                    raw[id] ^ T::SIGN
                } else if raw[id] & T::SIGN == 0 {
                    raw[id]
                } else {
                    0
                })
            });
            variant(&input, &full, &expected, negate, underflow, range)
        });
        let mut original = [sentinel; 13];
        input.read_into(&mut original).unwrap();
        bits(&original[..11], &full);
        bits(&original[11..], &[sentinel; 2]);
        drop(input);
        global::clear_thread_cache().unwrap();
        for (owner, expected) in survivors {
            let mut host = [sentinel; 13];
            owner.read_into(&mut host).unwrap();
            check(&host, &[expected; 7], sentinel);
        }
    }
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for float_underflow in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            global::configure(global::PcuExecutionPolicy {
                backend,
                float_underflow,
                ..Default::default()
            })
            .unwrap();
            format::<PcuF16Bits>(float_underflow, range_policy);
            format::<PcuBf16Bits>(float_underflow, range_policy);
            format::<PcuF8E4M3FnBits>(float_underflow, range_policy);
            format::<PcuF8E5M2Bits>(float_underflow, range_policy);
            format::<f32>(float_underflow, range_policy);
            format::<f64>(float_underflow, range_policy);
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
