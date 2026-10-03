//! Logical reads and faults use the admitted prefix; retained native inputs keep full shapes.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxBinaryInput,
    MlxCheckedBinaryPlan,
    MlxEncodedCompletion,
    MlxError,
    MlxRuntime,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBindingRef,
    PcuClampedFloat,
    PcuCompoundArithmeticPolicy as Compound,
    PcuExecutionFault,
    PcuHostArgument,
    PcuImplementationRequirements,
    PcuNumericalMode as Mode,
    PcuPrecisionPolicy as Precision,
    PcuReproducibility,
};
#[rustfmt::skip]
use super::{
    graph,
    Op,
    Policy,
    Range,
    Sample,
};
#[path = "oracle.rs"]
mod oracle;
const LEFT: PcuBindingRef = PcuBindingRef::new(2, 3);
const RIGHT: PcuBindingRef = PcuBindingRef::new(3, 2);
fn permissions() -> impl Iterator<Item = (Mode, Compound, Precision)> {
    (0..8).map(|bits| {
        (
            if bits & 1 == 0 {
                Mode::Boundary
            } else {
                Mode::Strict
            },
            if bits & 2 == 0 {
                Compound::Checked
            } else {
                Compound::BackendDefined
            },
            if bits & 4 == 0 {
                Precision::Preserve
            } else {
                Precision::BackendOptimized
            },
        )
    })
}
const fn request(
    requirements: &mut PcuImplementationRequirements,
    flags: (Mode, Compound, Precision),
) {
    requirements.numerical_mode = flags.0;
    requirements.numerical_options.compound_arithmetic = flags.1;
    requirements.numerical_options.precision = flags.2;
}
const fn roles(profile: usize) -> ([bool; 2], bool) {
    match profile {
        0 => ([false; 2], false),
        1 => ([true, false], false),
        2 => ([false, true], false),
        3 => ([true, false], true),
        _ => unreachable!(),
    }
}
fn cold<T: Sample>() {
    for profile in 0..4 {
        let (broadcast, repeated) = roles(profile);
        graph::fixture_roles::<T, _>(
            5,
            Op::Div,
            Policy::RejectSubnormalResult,
            Range::Clamp,
            true,
            broadcast,
            repeated,
            |ir| {
                let plan = MlxCheckedBinaryPlan::assess(ir).unwrap();
                let count = plan.input_bindings().len();
                let minimum = plan.input_element_counts();
                assert_eq!(count, if repeated { 1 } else { 2 });
                assert_eq!(
                    plan.assess_input_extents(&minimum[..count]).unwrap(),
                    minimum
                );
                let full = [8, 11];
                let admitted = plan.assess_input_extents(&full[..count]).unwrap();
                assert_eq!(&admitted[..count], &full[..count]);
                assert!(plan.assess_input_extents(&[]).is_err());
                assert!(
                    plan.assess_input_extents(&[usize::MAX; 2][..count])
                        .is_err()
                );
                let mut short = minimum;
                short[count - 1] -= 1;
                assert!(plan.assess_input_extents(&short[..count]).is_err());
                for flags in permissions() {
                    let mut exact = *ir;
                    request(&mut exact.numerical_requirements, flags);
                    assert_eq!(
                        MlxCheckedBinaryPlan::assess(&exact).unwrap().requirements(),
                        exact.numerical_requirements
                    );
                    exact
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    assert!(MlxCheckedBinaryPlan::assess(&exact).is_err());
                }
            },
        );
    }
}
macro_rules! six {
    ($function:ident) => {
        $function::<pcu_facade::PcuF16Bits>();
        $function::<pcu_facade::PcuBf16Bits>();
        $function::<pcu_facade::PcuF8E4M3FnBits>();
        $function::<pcu_facade::PcuF8E5M2Bits>();
        $function::<f32>();
        $function::<f64>();
    };
}
#[test]
fn six_format_binary_capacity_assessment_is_detached() {
    six!(cold);
}
fn verify<T: Sample>(
    actual: Result<MlxEncodedCompletion, MlxError>,
    expected: &(Vec<T>, Option<PcuExecutionFault>),
) {
    if let Some(fatal) = expected.1.filter(|fault| !fault.recovered) {
        assert!(matches!(actual, Err(MlxError::Arithmetic(fault)) if fault == fatal));
    } else {
        let (output, notice) = actual.unwrap().into_parts();
        assert_eq!(notice, expected.1);
        let sentinel = T::from_bits(17);
        let mut values = vec![sentinel; 8];
        output.read_into(&mut values).unwrap();
        oracle::compare(&values[..5], &expected.0);
        oracle::compare(&values[5..], &[sentinel; 3]);
    }
}
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
// One frozen tuple pairs all-resident/native/mixed terminal faults with exact shape and no-write preflight.
fn profile<T: Sample + PcuClampedFloat>(
    session: &MlxSession,
    other: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    grid: bool,
    profile: usize,
    flags: (Mode, Compound, Precision),
) {
    let (broadcast, repeated) = roles(profile);
    let unique_full = if repeated { &[8][..] } else { &[8, 11][..] };
    let mut prepared =
        graph::fixture_roles::<T, _>(5, op, policy, range, grid, broadcast, repeated, |ir| {
            let mut exact = *ir;
            request(&mut exact.numerical_requirements, flags);
            session
                .checked_binary_backend()
                .prepare_host_kernel_with_input_extents(&exact, unique_full)
        })
        .unwrap();
    assert_eq!(prepared.prepared_input_element_counts(), unique_full);
    let minimum = prepared.input_element_counts();
    let operand_minimum = if repeated {
        [5, 5]
    } else {
        broadcast.map(|b| if b { 1 } else { 5 })
    };
    let mut native = session
        .prepare_checked_binary_control_with_input_extents(
            T::TYPE,
            op,
            policy,
            range,
            5,
            operand_minimum,
            broadcast,
            [8, if repeated { 8 } else { 11 }],
        )
        .unwrap();
    for phase in 0..8 {
        let (left, right) = oracle::bank::<T>(phase);
        let a = session.upload_encoded(&left).unwrap();
        let b = session.upload_encoded(&right).unwrap();
        let expected = oracle::expected(
            &left,
            if repeated { &left } else { &right },
            op,
            policy,
            range,
            broadcast,
        );
        let inputs = if repeated { vec![&a] } else { vec![&a, &b] };
        verify(prepared.execute_resident(&inputs), &expected);
        verify(
            native.execute_resident([&a, if repeated { &a } else { &b }]),
            &expected,
        );
        let foreign = other.upload_encoded(&left).unwrap();
        let foreign_inputs = if repeated {
            vec![&foreign]
        } else {
            vec![&foreign, &b]
        };
        assert!(matches!(
            prepared.execute_resident(&foreign_inputs),
            Err(MlxError::ForeignSession)
        ));
        assert!(!prepared.last_call_may_have_written());
        let wrong = session.upload_encoded(&left[..7]).unwrap();
        let wrong_inputs = if repeated {
            vec![&wrong]
        } else {
            vec![&wrong, &b]
        };
        assert!(matches!(
            prepared.execute_resident(&wrong_inputs),
            Err(MlxError::InvalidExtent)
        ));
        assert!(!prepared.last_call_may_have_written());
        let host = PcuHostArgument::read(LEFT, &left[..minimum[0]]);
        let mut arguments = vec![MlxBinaryInput::HostBytes {
            target: LEFT,
            scalar: T::TYPE,
            bytes: host.bytes(),
        }];
        if !repeated {
            arguments.push(MlxBinaryInput::Resident {
                target: RIGHT,
                array: &b,
            });
        }
        assert!(matches!(
            prepared.execute_inputs(&arguments),
            Err(MlxError::InvalidExtent)
        ));
        assert!(!prepared.last_call_may_have_written());
        let mut unchanged = vec![T::from_bits(17); 11];
        a.read_into(&mut unchanged).unwrap();
        oracle::compare(&unchanged[..8], &left);
        oracle::compare(&unchanged[8..], &[T::from_bits(17); 3]);
        let mut right_unchanged = vec![T::from_bits(17); 14];
        b.read_into(&mut right_unchanged).unwrap();
        oracle::compare(&right_unchanged[..11], &right);
        oracle::compare(&right_unchanged[11..], &[T::from_bits(17); 3]);
        // A mixed cold shape keeps host minimum and full resident capacity separately.
        if !repeated {
            let mut mixed =
                graph::fixture_roles::<T, _>(5, op, policy, range, grid, broadcast, false, |ir| {
                    let mut exact = *ir;
                    request(&mut exact.numerical_requirements, flags);
                    session
                        .checked_binary_backend()
                        .prepare_host_kernel_with_input_extents(&exact, &[minimum[0], 11])
                })
                .unwrap();
            verify(mixed.execute_inputs(&arguments), &expected);
            let mut inverse =
                graph::fixture_roles::<T, _>(5, op, policy, range, grid, broadcast, false, |ir| {
                    let mut exact = *ir;
                    request(&mut exact.numerical_requirements, flags);
                    session
                        .checked_binary_backend()
                        .prepare_host_kernel_with_input_extents(&exact, &[8, minimum[1]])
                })
                .unwrap();
            let host_right = PcuHostArgument::read(RIGHT, &right[..minimum[1]]);
            verify(
                inverse.execute_inputs(&[
                    MlxBinaryInput::HostBytes {
                        target: RIGHT,
                        scalar: T::TYPE,
                        bytes: host_right.bytes(),
                    },
                    MlxBinaryInput::Resident {
                        target: LEFT,
                        array: &a,
                    },
                ]),
                &expected,
            );
        }
    }
    let (left, right) = oracle::bank::<T>(0);
    let a = session.upload_encoded(&left).unwrap();
    let b = session.upload_encoded(&right).unwrap();
    let inputs = [&a, &b];
    let (escaped, notice) = prepared
        .execute_resident(&inputs[..if repeated { 1 } else { 2 }])
        .unwrap()
        .into_parts();
    assert!(notice.is_none());
    let expected = oracle::expected(
        &left,
        if repeated { &left } else { &right },
        op,
        policy,
        range,
        broadcast,
    );
    drop(prepared);
    drop(native);
    drop(a);
    drop(b);
    let mut values = vec![T::from_bits(17); 8];
    escaped.read_into(&mut values).unwrap();
    oracle::compare(&values[..5], &expected.0);
}
fn qualify<T: Sample + PcuClampedFloat>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let other = runtime.open_gpu(0).unwrap();
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::RejectSubnormalResult,
            Policy::AllowGradualUnderflow,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                for grid in [false, true] {
                    for profile_index in 0..4 {
                        for flags in permissions() {
                            profile::<T>(
                                &session,
                                &other,
                                op,
                                policy,
                                range,
                                grid,
                                profile_index,
                                flags,
                            );
                        }
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual MLX GPU six-format full-capacity binary completion and retention."]
fn six_format_binary_full_capacity_resident_mixed_faults_and_lifetimes() {
    six!(qualify);
}
