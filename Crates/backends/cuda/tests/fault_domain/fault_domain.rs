//! Actual later-event fault coordinates, private output rejection and successful retry.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuDeviceTensor,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuImplementationRequirements,
    PcuFloatUnderflowPolicy,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuMemoryPoolId,
    PcuNumericalMode,
    PcuScalar,
    PcuTensor,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorError,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorArithmeticStep,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaOwnedDispatchBackend as OwnedDispatchBackend,
    CudaHostKernelError as HostKernelError,
    CudaTensorAssessor as TensorAssessor,
    CudaTensorExecutionError as TensorExecutionError,
};

fn location(error: &TensorExecutionError, case: u32) {
    let TensorExecutionError::Graph(TensorError::CompoundArithmeticFault {
        element_index,
        reduction_index,
        step,
        kind,
        ..
    }) = error
    else {
        panic!("expected compound coordinates: {error:?}");
    };
    let expected = match case {
        0 => (
            0,
            2,
            TensorArithmeticStep::Multiply,
            PcuExecutionFaultKind::ArithmeticOverflow,
        ),
        1 => (
            2,
            0,
            TensorArithmeticStep::Subtract,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ),
        2 => (
            0,
            3,
            TensorArithmeticStep::Divide,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
        _ => unreachable!(),
    };
    assert_eq!((*element_index, *reduction_index, *step, *kind), expected);
}
fn source_call<T: PcuScalar>(
    case: u32,
    a: &[T; 3],
    b: &[T; 3],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    match case {
        0 => source::product(&[*a], &[[b[0]], [b[1]], [b[2]]]),
        1 => source::update(a, b),
        2 => source::loss(a, b),
        _ => unreachable!(),
    }
}
fn read<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T], sentinel: T) {
    let mut output = vec![sentinel; expected.len() + 1];
    owner.read_into(&mut output).unwrap();
    for (actual, want) in output[..expected.len()].iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), want.encode_le().as_ref());
    }
    assert_eq!(
        output[expected.len()].encode_le().as_ref(),
        sentinel.encode_le().as_ref()
    );
}
fn graph_case<T: PcuScalar>(
    session: &OwnedDispatchBackend,
    assessor: &TensorAssessor<'_>,
    case: u32,
    inputs: (&[T; 3], &[T; 3], &[T; 3]),
    expected: &[T],
    sentinel: T,
) {
    let (bad, good, other) = inputs;
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let (shape_a, shape_b): (&[usize], &[usize]) = if case == 0 {
        (&[1, 3], &[3, 1])
    } else {
        (&[3], &[3])
    };
    let a = graph.input(shape_a, T::TYPE).unwrap();
    let b = graph.input(shape_b, T::TYPE).unwrap();
    let result = match case {
        0 => graph.matmul(a, b),
        1 => graph.sgd_update(a, b, 1.0),
        2 => graph.mean_squared_error(a, b),
        _ => unreachable!(),
    }
    .unwrap();
    let program = graph
        .into_selected_program(
            &[result],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let prepared = assessor.prepare_owned_program(program).unwrap();
    let pool = PcuMemoryPoolId(0x4644_4f4d);
    let other = PcuDeviceTensor::new(
        shape_b.to_vec(),
        session.upload_buffer(pool, other).unwrap(),
    )
    .unwrap();
    let bad =
        PcuDeviceTensor::new(shape_a.to_vec(), session.upload_buffer(pool, bad).unwrap()).unwrap();
    let mut memory = session.memory_provider(pool);
    let failure = assessor
        .execute_owned_program_outputs(&prepared, &[(a, &bad), (b, &other)], pool, &mut memory)
        .err()
        .expect("faulting graph must not publish an output");
    location(&failure, case);
    let good =
        PcuDeviceTensor::new(shape_a.to_vec(), session.upload_buffer(pool, good).unwrap()).unwrap();
    let output = assessor
        .execute_owned_program_outputs(&prepared, &[(a, &good), (b, &other)], pool, &mut memory)
        .unwrap();
    let mut actual = vec![sentinel; expected.len()];
    session
        .download_buffer(pool, output[0].1.buffer(), &mut actual)
        .unwrap();
    for (got, want) in actual.iter().zip(expected) {
        assert_eq!(got.encode_le().as_ref(), want.encode_le().as_ref());
    }
}
#[derive(Clone, Copy)]
struct Values<T> {
    zero: T,
    one: T,
    two: T,
    four: T,
    negative_one: T,
    max: T,
    nan: T,
    tiny: T,
}
fn format<T: PcuScalar>(session: &OwnedDispatchBackend, values: Values<T>) {
    let assessor = TensorAssessor::new(session).unwrap();
    for case in 0..3 {
        global::clear_thread_cache().unwrap();
        let (bad, good, other, expected) = match case {
            0 => (
                [values.one, values.one, values.max],
                [values.one; 3],
                [values.one, values.one, values.two],
                vec![values.four],
            ),
            1 => (
                [values.zero, values.zero, values.nan],
                [values.zero; 3],
                [values.one; 3],
                vec![values.negative_one; 3],
            ),
            2 => (
                [values.tiny, values.zero, values.zero],
                [values.zero; 3],
                [values.zero; 3],
                vec![values.zero],
            ),
            _ => unreachable!(),
        };
        let prior = source_call(case, &good, &other).unwrap();
        read(&prior, &expected, values.max);
        let failure = source_call(case, &bad, &other).unwrap_err();
        let PcuExecutionError::CudaTensorExecution(error) = failure else {
            panic!("expected selected backend failure: {failure:?}");
        };
        location(&error, case);
        read(&prior, &expected, values.max);
        read(
            &source_call(case, &good, &other).unwrap(),
            &expected,
            values.max,
        );
        graph_case(
            session,
            &assessor,
            case,
            (&bad, &good, &other),
            &expected,
            values.max,
        );
    }
}
#[test]
#[ignore = "authorized actual selected GPU: ordered source/owned graph late-event faults and retry"]
fn checked_compound_later_events_source_graph_and_retry() {
    let (_, session, _) = selection::selected_device();
    format(
        &session,
        Values {
            zero: 0f32,
            one: 1f32,
            two: 2f32,
            four: 4f32,
            negative_one: -1f32,
            max: f32::MAX,
            nan: f32::NAN,
            tiny: f32::from_bits(53 << 23),
        },
    );
    format(
        &session,
        Values {
            zero: 0f64,
            one: 1f64,
            two: 2f64,
            four: 4f64,
            negative_one: -1f64,
            max: f64::MAX,
            nan: f64::NAN,
            tiny: f64::from_bits(487 << 52),
        },
    );
}

fn exact<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}
fn composed_routes<T: PcuCheckedFloat>(
    session: &OwnedDispatchBackend,
    zero: T,
    one: T,
    two: T,
    max: T,
    nan: T,
    sentinel: T,
) {
    for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Cuda,
            device: Some(0),
            range_policy: range,
            ..Default::default()
        })
        .unwrap();
        let bindings = source::composed_bindings::<T>();
        let requirements = PcuImplementationRequirements {
            range_policy: range,
            ..PcuImplementationRequirements::DEFAULT
        };
        let builder = source::__composed_ir_with_float_underflow_policy::<T>(
            &bindings,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range,
            requirements,
        )
        .unwrap();
        builder.with_ir(|ir| {
            let mut prepared = session.prepare_host_kernel(ir).unwrap();
            for actual_source in [true, false] {
                let mut invoke = |a: &[T], b: &[T], output: &mut [T]| {
                    if actual_source {
                        source::composed(a, b, output)
                            .map_err(|error| error.arithmetic_fault().unwrap())
                    } else {
                        prepared
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), a),
                                PcuHostArgument::read(PcuBindingRef::new(0, 1), b),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
                            ])
                            .map_err(|error| match error {
                                HostKernelError::CheckedExecutionFault(fault) => fault,
                                other => panic!("expected checked fault: {other:?}"),
                            })
                    }
                };
                let mut a = [one; 4];
                let mut b = [one; 4];
                let healthy = [two, two, two, two, sentinel];
                let mut output = [sentinel; 5];
                invoke(&a, &b, &mut output).unwrap();
                exact(&output, &healthy);
                a[2] = max;
                b[2] = zero;
                let fault = invoke(&a, &b, &mut output).unwrap_err();
                assert_eq!(
                    (fault.invocation_id, fault.kind, fault.recovered),
                    (
                        2,
                        PcuExecutionFaultKind::ArithmeticOverflow,
                        range == PcuRangePolicy::Clamp
                    )
                );
                if range == PcuRangePolicy::Clamp {
                    exact(&output, &[two, two, max, two, sentinel]);
                } else {
                    exact(&output, &healthy);
                }
                let before_fatal = output;
                a[0] = max;
                b[0] = zero;
                a[3] = nan;
                b[3] = zero;
                let fault = invoke(&a, &b, &mut output).unwrap_err();
                let lane = if range == PcuRangePolicy::Clamp { 3 } else { 0 };
                let kind = if range == PcuRangePolicy::Clamp {
                    PcuExecutionFaultKind::InvalidFloatingOperand
                } else {
                    PcuExecutionFaultKind::ArithmeticOverflow
                };
                assert_eq!(
                    (fault.invocation_id, fault.kind, fault.recovered),
                    (lane, kind, false)
                );
                exact(&output, &before_fatal);
                invoke(&[one; 4], &[one; 4], &mut output).unwrap();
                exact(&output, &healthy);
            }
        });
    }
}
#[test]
#[ignore = "authorized actual selected GPU: restored F32/F64 composed source/prepared range and fatal publication"]
fn composed_f32_f64_source_and_prepared_fault_publication() {
    let (_, session, _) = selection::selected_device();
    composed_routes(&session, 0f32, 1f32, 2f32, f32::MAX, f32::NAN, -1f32);
    composed_routes(&session, 0f64, 1f64, 2f64, f64::MAX, f64::NAN, -1f64);
}
