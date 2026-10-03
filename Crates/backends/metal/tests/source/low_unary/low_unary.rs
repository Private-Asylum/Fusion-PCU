//! Actual annotated low unary host/resident calls and direct cold endpoint admission.
#[rustfmt::skip]
use pcu_facade::{
    pcu,PcuCheckedFloat,PcuHostArgument,PcuBindingRef,PcuExecutionError,PcuExecutionFaultKind,
    PcuArgumentError,PcuTensor,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,PcuDispatchFloatUnaryOp,PcuDispatchOp,PcuDispatchDataOp,
    PcuDispatchIndex,PcuRangePolicy,PcuReproducibility,PcuHostKernelBackend,
};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalError};
use super::low_precision::{Sample, assert_bits, open_backend};
#[path = "../../../benches/support/low_unary/graph.rs"]
mod graph;
#[path = "../../../benches/checked_low_unary/source.rs"]
mod source;
#[pcu(invocations=1,crate_path=::pcu_facade)]
fn grid<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 3 {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}
#[pcu(invocations=3,flag(clamp_range),crate_path=::pcu_facade)]
fn clamp<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations=3,flag(deterministic),crate_path=::pcu_facade)]
fn portable<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations=3,crate_path=::pcu_facade)]
fn broadcast<T: PcuCheckedFloat>(input: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(*input);
}
#[pcu(invocations=3,flag(strict),flag(native_compound),flag(backend_precision),crate_path=::pcu_facade)]
fn permissions<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)] // One exact-byte source lifecycle covers independent policies, ownership and fatal publication.
fn host<T: Sample>() {
    let session = MetalSession::open(0).unwrap();
    let input = [T::raw(0), T::value(-0.0), T::raw(1)];
    let sentinel = T::value(91.0);
    let mut output = [sentinel; 5];
    let mut negate = source::negate_prepare::<T, 3, _>(&session).unwrap();
    let mut relu = source::relu_prepare::<T, 3, _>(&session).unwrap();
    negate(&input, &mut output).unwrap();
    assert_bits(
        &output,
        &[
            input[0].pcu_checked_neg().unwrap(),
            input[1].pcu_checked_neg().unwrap(),
            input[2].pcu_checked_neg().unwrap(),
            sentinel,
            sentinel,
        ],
    );
    relu(&input, &mut output).unwrap();
    assert_bits(
        &output,
        &[T::raw(0), T::raw(0), T::raw(1), sentinel, sentinel],
    );
    let before = output;
    assert!(
        matches!(relu(&[T::value(-1.0),T::raw(T::NAN),T::raw(1)],&mut output),Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) if fault.invocation_id==1&&fault.kind==PcuExecutionFaultKind::InvalidFloatingOperand)
    );
    assert_bits(&output, &before);
    assert!(negate(&input[..1], &mut output).is_err());
    assert_bits(&output, &before);
    negate(
        &[T::value(-1.0), T::value(2.0), T::value(-3.0)],
        &mut output,
    )
    .unwrap();
    assert_bits(
        &output,
        &[
            T::value(1.0),
            T::value(-2.0),
            T::value(3.0),
            sentinel,
            sentinel,
        ],
    );
    let mut tight = source::negate_tight_prepare::<T, 3, _>(&session).unwrap();
    assert!(
        matches!(tight(&input,&mut output),Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) if fault.invocation_id==2&&fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    let mut tight_relu = source::relu_tight_prepare::<T, 3, _>(&session).unwrap();
    tight_relu(
        &[T::value(-1.0), T::value(-0.0), T::value(1.0)],
        &mut output,
    )
    .unwrap();
    let mut gradual = source::negate_gradual_prepare::<T, 3, _>(&session).unwrap();
    gradual(&input, &mut output).unwrap();
    let mut gradual_relu = source::relu_gradual_prepare::<T, 3, _>(&session).unwrap();
    gradual_relu(&input, &mut output).unwrap();
    grid_prepare::<T, _>(&session).unwrap()(&input, &mut output).unwrap();
    assert_bits(
        &output,
        &[T::raw(0), T::raw(0), T::raw(1), sentinel, sentinel],
    );
    clamp_prepare::<T, _>(&session).unwrap()(&input, &mut output).unwrap();
    portable_prepare::<T, _>(&session).unwrap()(&input, &mut output).unwrap();
    broadcast_prepare::<T, _>(&session).unwrap()(&T::value(2.0), &mut output).unwrap();
    permissions_prepare::<T, _>(&session).unwrap()(&input, &mut output).unwrap();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    source::negate::<T, 3>(&input, &mut output).unwrap();
    source::relu::<T, 3>(&input, &mut output).unwrap();
    let backend = open_backend();
    let pool = pcu_facade::PcuMemoryPoolId(0);
    let input_buffer = backend.upload_buffer(pool, &input).unwrap();
    let mut raw_output = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    source::negate_prepare_device::<T, 3, _>(&backend).unwrap()(&input_buffer, &mut raw_output)
        .unwrap();
    let mut raw_values = [sentinel; 5];
    backend
        .download_buffer(pool, &raw_output, &mut raw_values)
        .unwrap();
    assert_bits(
        &raw_values,
        &[
            input[0].pcu_checked_neg().unwrap(),
            input[1].pcu_checked_neg().unwrap(),
            input[2].pcu_checked_neg().unwrap(),
            sentinel,
            sentinel,
        ],
    );
    let resident_input =
        PcuTensor::from_device_buffer(backend.clone(), input_buffer, &[3]).unwrap();
    let output_buffer = backend.upload_buffer(pool, &[sentinel; 3]).unwrap();
    let mut resident_output =
        PcuTensor::from_device_buffer(backend.clone(), output_buffer, &[3]).unwrap();
    source::relu::<T, 3>(&resident_input, &mut resident_output).unwrap();
    let mut values = [sentinel; 3];
    resident_output.read_into(&mut values).unwrap();
    assert_bits(&values, &[T::raw(0), T::raw(0), T::raw(1)]);
    source::negate::<T, 3>(&input, &mut resident_output).unwrap();
    source::relu::<T, 3>(&resident_input, &mut output).unwrap();
    assert!(source::negate::<T, 3>(&input[..1], &mut resident_output).is_err());
    resident_output.read_into(&mut values).unwrap();
    let bad = [T::value(1.0), T::raw(T::NAN), T::value(2.0)];
    assert!(source::negate::<T, 3>(&bad, &mut resident_output).is_err());
    assert!(matches!(
        resident_output.read_into(&mut values),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    assert!(matches!(
        source::relu::<T, 3>(&resident_output, &mut output),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    let fresh_buffer = backend.upload_buffer(pool, &[sentinel; 3]).unwrap();
    let mut fresh = PcuTensor::from_device_buffer(backend.clone(), fresh_buffer, &[3]).unwrap();
    source::relu::<T, 3>(&resident_input, &mut fresh).unwrap();
    fresh.read_into(&mut values).unwrap();
    resident_input.read_into(&mut values).unwrap();
    assert_bits(&values, &input);
    let foreign = MetalSession::open(0).unwrap();
    let other = foreign
        .upload_bytes(PcuHostArgument::read(PcuBindingRef::new(0, 0), &input).bytes())
        .unwrap();
    let map = session
        .prepare_low_precision_unary(
            T::TYPE,
            PcuDispatchFloatUnaryOp::Neg,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        )
        .unwrap();
    assert!(matches!(
        map.execute(&other),
        Err(MetalError::ForeignSession)
    ));
    pcu_facade::global::use_defaults().unwrap();
}
#[pcu(invocations=3,flag(clamp_range),flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn broadcast_tight<T: PcuCheckedFloat>(input: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(*input);
}
#[pcu(invocations=1,flag(clamp_range),flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn grid_clamp<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 3 {
        output[id] = -input[id];
        id += stride;
    }
}
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)] // One recovery/rollback lifecycle compares the same actual source across borrowed ownership boundaries.
fn recovery<T: Sample>() {
    let session = MetalSession::open(0).unwrap();
    let input = [T::raw(1), T::value(-1.0), T::raw(2)];
    let expected = [T::raw(1), T::raw(0), T::raw(2)];
    let sentinel = T::value(91.0);
    let mut output = [sentinel; 5];
    let mut map = source::relu_tight_clamp_prepare::<T, 3, _>(&session).unwrap();
    assert!(
        matches!(map(&input, &mut output), Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
        if fault.recovered && fault.invocation_id == 0 && fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_bits(
        &output,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let before = output;
    let fatal = [T::raw(1), T::raw(T::NAN), T::raw(2)];
    assert!(
        matches!(map(&fatal,&mut output), Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
        if !fault.recovered && fault.invocation_id == 1 && fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand)
    );
    assert_bits(&output, &before);
    assert!(map(&input[..1], &mut output).is_err());
    assert_bits(&output, &before);
    assert!(grid_clamp_prepare::<T, _>(&session).unwrap()(&input, &mut output).is_err());
    let negated = input.map(|value| {
        value
            .pcu_checked_neg_with_policy(PcuFloatUnderflowPolicy::AllowGradualUnderflow)
            .unwrap()
    });
    assert_bits(
        &output,
        &[negated[0], negated[1], negated[2], sentinel, sentinel],
    );
    let mut broadcast = broadcast_tight_prepare::<T, _>(&session).unwrap();
    assert!(
        matches!(broadcast(&T::raw(1),&mut output),Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) if fault.recovered&&fault.invocation_id==0)
    );
    assert_bits(
        &output,
        &[T::raw(1), T::raw(1), T::raw(1), sentinel, sentinel],
    );
    broadcast(&T::value(-1.0), &mut output).unwrap();
    assert_bits(
        &output,
        &[T::raw(0), T::raw(0), T::raw(0), sentinel, sentinel],
    );
    let before = output;
    assert!(broadcast(&T::raw(T::NAN), &mut output).is_err());
    assert_bits(&output, &before);
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    assert!(
        matches!(source::relu_tight_clamp::<T,3>(&input,&mut output),Err(PcuExecutionError::ArithmeticFault(fault)) if fault.recovered&&fault.invocation_id==0)
    );
    assert_bits(
        &output,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let backend = open_backend();
    let pool = pcu_facade::PcuMemoryPoolId(0);
    let input_buffer = backend.upload_buffer(pool, &input).unwrap();
    let mut raw_output = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    let mut device_call = source::relu_tight_clamp_prepare_device::<T, 3, _>(&backend).unwrap();
    assert!(device_call(&input_buffer, &mut raw_output).is_err());
    let mut read = [sentinel; 5];
    backend
        .download_buffer(pool, &raw_output, &mut read)
        .unwrap();
    assert_bits(
        &read,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let resident_input =
        PcuTensor::from_device_buffer(backend.clone(), input_buffer, &[3]).unwrap();
    let output_buffer = backend.upload_buffer(pool, &[sentinel; 3]).unwrap();
    let mut resident_output =
        PcuTensor::from_device_buffer(backend.clone(), output_buffer, &[3]).unwrap();
    assert!(
        matches!(source::relu_tight_clamp::<T,3>(&resident_input,&mut resident_output),Err(PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    let mut values = [sentinel; 3];
    resident_output.read_into(&mut values).unwrap();
    assert_bits(&values, &expected);
    assert!(source::relu_tight_clamp::<T, 3>(&input, &mut resident_output).is_err());
    resident_output.read_into(&mut values).unwrap();
    assert_bits(&values, &expected);
    assert!(source::relu_tight_clamp::<T, 3>(&resident_input, &mut output).is_err());
    assert_bits(
        &output,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let scalar_buffer = backend.upload_buffer(pool, &[T::raw(1)]).unwrap();
    let scalar = PcuTensor::from_device_buffer(backend.clone(), scalar_buffer, &[]).unwrap();
    assert!(
        matches!(broadcast_tight::<T>(&scalar,&mut resident_output),Err(PcuExecutionError::ArithmeticFault(fault)) if fault.recovered&&fault.invocation_id==0)
    );
    resident_output.read_into(&mut values).unwrap();
    assert_bits(&values, &[T::raw(1); 3]);
    assert!(
        matches!(source::relu_tight_clamp::<T,3>(&fatal,&mut resident_output),Err(PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered&&fault.invocation_id==1)
    );
    assert!(matches!(
        resident_output.read_into(&mut values),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    assert!(matches!(
        source::relu::<T, 3>(&resident_output, &mut output),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    resident_input.read_into(&mut values).unwrap();
    assert_bits(&values, &input);
    pcu_facade::global::use_defaults().unwrap();
}
#[allow(clippy::too_many_lines)] // One cold tuple matrix covers positive shapes/ranges and independent structural/header rejections.
fn admission<T: Sample>() {
    let session = MetalSession::open(0).unwrap();
    for looped in [false, true] {
        for op in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                for looped in [false, true] {
                    for broadcast in [false, true] {
                        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                            for op in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu]
                            {
                                for policy in [
                                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                                ] {
                                    graph::fixture_profile::<T, _>(
                                        3,
                                        op,
                                        policy,
                                        range,
                                        looped,
                                        broadcast,
                                        |kernel| {
                                            session.prepare_host_kernel(kernel).unwrap();
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
                graph::fixture::<T, _>(3, op, policy, looped, |kernel| {
                    session.prepare_float_unary_kernel(kernel).unwrap();
                    let mut invalid = *kernel;
                    invalid
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    assert_eq!(
                        session
                            .prepare_float_unary_kernel(&invalid)
                            .unwrap()
                            .requirements(),
                        invalid.numerical_requirements
                    );
                    let mut invalid = *kernel;
                    invalid.numerical_requirements.float_underflow =
                        if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                            PcuFloatUnderflowPolicy::IeeeAfterRounding
                        } else {
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow
                        };
                    assert!(matches!(
                        session.prepare_float_unary_kernel(&invalid),
                        Err(MetalError::Unsupported)
                    ));
                });
            }
        }
    }
    graph::fixture::<T, _>(
        3,
        PcuDispatchFloatUnaryOp::Neg,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        false,
        |kernel| {
            for variant in 2..5 {
                let mut invalid = *kernel;
                let mut body = kernel.ops.to_vec();
                match variant {
                    0 => {
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            index, ..
                        }) = &mut body[0]
                        {
                            *index = PcuDispatchIndex::BindingElementZero;
                        }
                    }
                    1 => {
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                            range_policy,
                            ..
                        }) = &mut body[1]
                        {
                            *range_policy = PcuRangePolicy::Clamp;
                            invalid.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
                        }
                    }
                    2 => {
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                            value,
                            ..
                        }) = &mut body[1]
                        {
                            *value = pcu_facade::PcuDispatchValueId(99);
                        }
                    }
                    3 => invalid.entry.logical_shape = [3, 2, 1],
                    _ => invalid.numerical_requirements.range_policy = PcuRangePolicy::Clamp,
                }
                invalid.ops = &body;
                assert!(session.prepare_host_kernel(&invalid).is_err());
            }
        },
    );
}
macro_rules! proof {
    ($host:ident,$gate:ident,$ty:ty) => {
        #[test]
        #[ignore = "Requires actual Metal low unary annotated source/ownership proof."]
        fn $host() {
            let _guard = crate::source_policy_guard();
            host::<$ty>();
        }
        #[test]
        #[ignore = "Requires actual Metal cold preparation and exact requested-header rejections."]
        fn $gate() {
            let _guard = crate::source_policy_guard();
            admission::<$ty>();
        }
    };
}
proof!(f16_unary_source, f16_unary_admission, PcuF16Bits);
proof!(bf16_unary_source, bf16_unary_admission, PcuBf16Bits);
proof!(e4m3fn_unary_source, e4m3fn_unary_admission, PcuF8E4M3FnBits);
proof!(e5m2_unary_source, e5m2_unary_admission, PcuF8E5M2Bits);
#[test]
#[ignore = "Requires actual Metal; direct Portable unary endpoint retains its exact qualified descriptor header."]
fn f32_f64_direct_unary_portable_retains_original_header() {
    let _guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    macro_rules! check {
        ($ty:ty) => {
            graph::fixture::<$ty, _>(
                3,
                PcuDispatchFloatUnaryOp::Neg,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                false,
                |kernel| {
                    let mut invalid = *kernel;
                    invalid
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    assert_eq!(
                        session
                            .prepare_float_unary_kernel(&invalid)
                            .unwrap()
                            .requirements(),
                        invalid.numerical_requirements
                    );
                },
            );
        };
    }
    check!(f32);
    check!(f64);
}

macro_rules! recover_proof {
    ($name:ident,$ty:ty) => {
        #[test]
        #[ignore = "Requires actual Metal completed Clamp payloads, scalar broadcasts and host/resident publication."]
        fn $name() { let _guard=crate::source_policy_guard(); recovery::<$ty>(); }
    };
}
recover_proof!(f16_unary_recovery, PcuF16Bits);
recover_proof!(bf16_unary_recovery, PcuBf16Bits);
recover_proof!(e4m3fn_unary_recovery, PcuF8E4M3FnBits);
recover_proof!(e5m2_unary_recovery, PcuF8E5M2Bits);
