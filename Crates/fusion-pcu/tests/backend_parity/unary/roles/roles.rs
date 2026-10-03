//! Unused borrows and reordered declarations must not invent native input roles.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    describe_checked_float_unary_map,
    describe_portable_v1_unary_map,
    PcuBindingRef,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
};

#[path = "source/source.rs"]
mod source;

fn invoke<T: Sample>(
    role: usize,
    input: &[T],
    output: &mut [T],
) -> Result<(), global::PcuExecutionError> {
    let unread: &[T] = &[];
    // A NaN in an unread scalar must not become an actual arithmetic operand.
    let ghost = T::from_raw(T::NONFINITE);
    match role {
        0 => source::direct_neg(unread, output, input, &ghost),
        1 => source::direct_relu(unread, output, input, &ghost),
        2 => source::grid_neg(unread, output, input, &ghost),
        3 => source::grid_relu(unread, output, input, &ghost),
        4 => source::broadcast_neg(unread, output, input, &ghost),
        5 => source::broadcast_relu(unread, output, input, &ghost),
        _ => unreachable!("six static source roles"),
    }
}

fn check<T: Sample>(
    role: usize,
    raw: &[u64; 11],
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) {
    let sentinel = T::from_raw(T::NORMAL);
    let input = raw.map(T::from_raw);
    let expected: [u64; 7] = core::array::from_fn(|lane| {
        let value = raw[if role >= 4 { 0 } else { lane }];
        if role.is_multiple_of(2) {
            value ^ T::SIGN
        } else if value & T::SIGN == 0 {
            value
        } else {
            0
        }
    });
    let tiny_lane = expected.iter().position(|value| {
        let magnitude = value & !T::SIGN;
        magnitude != 0 && magnitude < T::NORMAL
    });
    let mut output = [sentinel; 9];
    let result = invoke(role, &input, &mut output);
    let exceptional =
        underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult && tiny_lane.is_some();
    if exceptional {
        let fault = result.unwrap_err();
        assert!(
            matches!(fault, global::PcuExecutionError::ArithmeticFault(fault)
            if fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow
                && fault.invocation_id == u64::try_from(tiny_lane.unwrap()).unwrap()
                && fault.recovered == (range == PcuRangePolicy::Clamp))
        );
    } else {
        result.unwrap();
    }
    if exceptional && range == PcuRangePolicy::Reject {
        bits(&output, &[sentinel; 9]);
    } else {
        bits(&output[..7], &expected.map(T::from_raw));
        bits(&output[7..], &[sentinel; 2]);
    }
    bits(&input, &raw.map(T::from_raw));

    // A later invalid operand outranks a recovered notice, but not an earlier
    // fatal checked underflow. Fault selection follows invocation order.
    let mut fatal = input;
    fatal[if role >= 4 { 0 } else { 5 }] = T::from_raw(T::NONFINITE);
    output.fill(sentinel);
    let error = invoke(role, &fatal, &mut output).unwrap_err();
    let invalid_lane = if role >= 4 { 0 } else { 5 };
    let earlier_underflow = if exceptional && range == PcuRangePolicy::Reject {
        tiny_lane.filter(|lane| *lane < invalid_lane)
    } else {
        None
    };
    let (kind, lane) = earlier_underflow.map_or(
        (PcuExecutionFaultKind::InvalidFloatingOperand, invalid_lane),
        |lane| (PcuExecutionFaultKind::ArithmeticUnderflow, lane),
    );
    assert!(
        matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == kind
            && fault.invocation_id == u64::try_from(lane).unwrap()
            && !fault.recovered)
    );
    bits(&output, &[sentinel; 9]);
    let retry = invoke(role, &input, &mut output);
    assert_eq!(retry.is_err(), exceptional);
    if !exceptional || range == PcuRangePolicy::Clamp {
        bits(&output[..7], &expected.map(T::from_raw));
    }
    bits(&output[7..], &[sentinel; 2]);

    let mut short = [sentinel; 6];
    assert!(invoke(role, &input, &mut short).is_err());
    bits(&short, &[sentinel; 6]);
}

fn format<T: Sample>(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    for phase in 0..3 {
        let mut raw = [T::NONFINITE; 11];
        raw[..7].copy_from_slice(&[
            1 + phase,
            T::SIGN,
            T::SIGN | (1 + phase),
            0,
            T::NORMAL + phase,
            T::SIGN | (T::NORMAL + phase),
            T::NORMAL + 2 + phase,
        ]);
        for role in 0..6 {
            check::<T>(role, &raw, underflow, range);
        }
    }
}

fn formats(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    format::<PcuF16Bits>(underflow, range);
    format::<PcuBf16Bits>(underflow, range);
    format::<PcuF8E4M3FnBits>(underflow, range);
    format::<PcuF8E5M2Bits>(underflow, range);
    format::<f32>(underflow, range);
    format::<f64>(underflow, range);
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        global::configure(global::PcuExecutionPolicy {
                            backend,
                            numerical_mode,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                reproducibility: PcuReproducibility::Unspecified,
                            },
                            float_underflow,
                            range_policy,
                            score_invocation: Some(score),
                            ..Default::default()
                        })
                        .unwrap();
                        global::clear_thread_cache().unwrap();
                        formats(float_underflow, range_policy);
                        let cold = SCORES.load(Ordering::Relaxed);
                        formats(float_underflow, range_policy);
                        assert_eq!(SCORES.load(Ordering::Relaxed), cold);
                    }
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

fn metadata<T: PcuCheckedFloat>() {
    macro_rules! shape {
        ($bindings:ident, $ir:ident, $invocations:expr, $broadcast:expr) => {
            let bindings = source::$bindings::<T>();
            source::$ir::<T>(&bindings).unwrap().with_ir(|kernel| {
                let description = describe_checked_float_unary_map(kernel).unwrap();
                assert_eq!(description.input_binding, PcuBindingRef::new(0, 2));
                assert_eq!(description.output_binding, PcuBindingRef::new(0, 1));
                assert_eq!(description.logical_extent, 7);
                assert_eq!(description.submitted_invocations, $invocations);
                assert_eq!(description.input_extent(), if $broadcast { 1 } else { 7 });
                assert_eq!(description.requirements, kernel.numerical_requirements);
                assert_eq!(
                    description.requirements.numerical_options.reproducibility,
                    PcuReproducibility::Unspecified
                );
                assert_eq!(kernel.bindings.len(), 4);
                assert!(describe_portable_v1_unary_map(kernel).is_err());
            });
        };
    }
    shape!(direct_neg_bindings, direct_neg_ir, 7, false);
    shape!(direct_relu_bindings, direct_relu_ir, 7, false);
    shape!(grid_neg_bindings, grid_neg_ir, 3, false);
    shape!(grid_relu_bindings, grid_relu_ir, 3, false);
    shape!(broadcast_neg_bindings, broadcast_neg_ir, 7, true);
    shape!(broadcast_relu_bindings, broadcast_relu_ir, 7, true);
}

#[test]
fn ordinary_source_keeps_actual_unary_roles_and_original_header() {
    metadata::<PcuF16Bits>();
    metadata::<PcuBf16Bits>();
    metadata::<PcuF8E4M3FnBits>();
    metadata::<PcuF8E5M2Bits>();
    metadata::<f32>();
    metadata::<f64>();
}
