//! Exact cast rounding, fault order, rollback/retry, mixed schemas and frozen policy offers.
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuCheckedConversion,PcuCpuHostBackend,PcuCpuHostError,PcuCpuHostOffers,PcuCpuHostOfferError};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloatConversion,
    PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,
    PcuDispatchOp,PcuDispatchDataOp,PcuDispatchControlOp,PcuDispatchIndex,
    PcuExecutionFaultKind,PcuFloatUnderflowPolicy,PcuRangePolicy,
    PcuCostBoundary,PcuDeviceIdentity,PcuExecutorId,PcuProviderId,PcuObjectRef,PcuObjectKind,
    PcuImplementationRequirements,PcuImplementationRequest,PcuImplementationOffers,
    PcuNumericalMode,PcuReproducibility,
};
#[path = "source/source.rs"]
mod source;
#[test]
fn widening_source_preserves_all_finite_bits_signs_tails_and_retries() {
    let input = [
        0.0_f32,
        -0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
        -f32::MAX,
        1.0 + f32::EPSILON,
    ];
    let mut output = [99.0_f64; 9];
    let mut call = source::widen_prepare::<7, _>(&PcuCpuHostBackend::scalar()).unwrap();
    call(&input, &mut output).unwrap();
    for index in 0..7 {
        assert_eq!(output[index].to_bits(), f64::from(input[index]).to_bits());
    }
    assert_eq!(output[7..], [99.0; 2]);
    for bad in [
        f32::from_bits(0x7f80_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let mut changed = input;
        changed[2] = bad;
        let before = output.map(f64::to_bits);
        let fault = call(&changed, &mut output).unwrap_err().fault().unwrap();
        assert_eq!(fault.invocation_id, 2);
        assert_eq!(fault.kind, PcuExecutionFaultKind::InvalidFloatingOperand);
        assert_eq!(output.map(f64::to_bits), before);
        call(&input, &mut output).unwrap();
    }
}
#[test]
fn narrowing_source_rounds_ties_even_and_preserves_signed_zero() {
    let input = [
        0.0_f64,
        -0.0,
        f64::from_bits(0x3ff0_0000_1000_0000),
        f64::from_bits(0x3ff0_0000_3000_0000),
        f64::from(f32::MAX),
        f64::from(f32::from_bits(1)),
        f64::from(f32::MIN_POSITIVE),
    ];
    let mut output = [99.0_f32; 9];
    source::narrow_prepare::<7, _>(&PcuCpuHostBackend::scalar()).unwrap()(&input, &mut output)
        .unwrap();
    for index in 0..7 {
        assert_eq!(
            output[index].to_bits(),
            input[index].pcu_checked_to_f32().unwrap().to_bits()
        );
    }
    assert_eq!(output[2].to_bits(), 1.0_f32.to_bits());
    assert_eq!(output[3].to_bits(), 0x3f80_0002);
    assert_eq!(output[7..], [99.0; 2]);
}
#[test]
fn narrowing_range_and_nonfinite_faults_preserve_all_output_and_retry() {
    let mut call = source::narrow_prepare::<4, _>(&PcuCpuHostBackend::scalar()).unwrap();
    let mut output = [99.0_f32; 6];
    for (bad, kind) in [
        (f64::MAX, PcuExecutionFaultKind::ArithmeticOverflow),
        (
            f64::from(f32::from_bits(1)) * 0.5,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
        (f64::NAN, PcuExecutionFaultKind::InvalidFloatingOperand),
        (f64::INFINITY, PcuExecutionFaultKind::InvalidFloatingOperand),
    ] {
        let fault = call(&[1.0, bad, f64::NAN, 2.0], &mut output)
            .unwrap_err()
            .fault()
            .unwrap();
        assert_eq!(fault.invocation_id, 1);
        assert_eq!(fault.kind, kind);
        assert!(!fault.recovered);
        assert_eq!(output.map(f32::to_bits), [99.0_f32; 6].map(f32::to_bits));
        call(&[1.0, 2.0, 3.0, 4.0], &mut output).unwrap();
        assert_eq!(
            output.map(f32::to_bits),
            [1.0_f32, 2.0, 3.0, 4.0, 99.0, 99.0].map(f32::to_bits)
        );
        output.fill(99.0);
    }
}
#[test]
fn grid_broadcast_underflow_and_schema_validation_are_transactional() {
    let bindings = source::narrow_bindings();
    let builder = source::narrow_ir::<5>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut body = kernel.ops[..3].to_vec();
    for op in &mut body {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { index, .. }) => {
                *index = PcuDispatchIndex::BindingElementZero;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { index, .. }) => {
                *index = PcuDispatchIndex::GridStrideId;
            }
            _ => {}
        }
    }
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
            underflow_policy,
            ..
        }) = &mut body[1]
        {
            *underflow_policy = policy;
        }
        let ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 5,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = pcu_facade::PcuDispatchKernelIr {
            ops: &ops,
            numerical_requirements: PcuImplementationRequirements {
                float_underflow: policy,
                ..kernel.numerical_requirements
            },
            entry: pcu_facade::PcuDispatchEntryPoint {
                logical_shape: [2, 1, 1],
                ..kernel.entry
            },
            ..kernel
        };
        let mut prepared = PcuCpuCheckedConversion
            .prepare_host_kernel(&kernel)
            .unwrap();
        let input = [f64::from(f32::from_bits(1)) * 0.5];
        let mut output = [99.0_f32; 7];
        let result = prepared.call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
        ]);
        if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
            result.unwrap();
            assert_eq!(
                output.map(f32::to_bits),
                [0.0_f32, 0.0, 0.0, 0.0, 0.0, 99.0, 99.0].map(f32::to_bits)
            );
        } else {
            assert_eq!(
                result.unwrap_err().fault().unwrap().kind,
                PcuExecutionFaultKind::ArithmeticUnderflow
            );
            assert_eq!(output.map(f32::to_bits), [99.0_f32; 7].map(f32::to_bits));
        }
        let before = output.map(f32::to_bits);
        assert!(
            prepared
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &[1.0_f32]),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output)
                ])
                .is_err()
        );
        assert_eq!(output.map(f32::to_bits), before);
    }
}
#[test]
fn exact_underflow_and_independent_offer_axes_reject_unproved_profiles() {
    let bindings = source::narrow_bindings();
    let builder = source::narrow_ir::<5>(&bindings).unwrap();
    let kernel = builder.ir();
    let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), identity, PcuExecutorId(0));
    let mut request = PcuImplementationRequest {
        device: identity,
        executor: PcuExecutorId(0),
        operation: &kernel,
        requirements: PcuImplementationRequirements::default(),
        boundary: PcuCostBoundary::Host,
    };
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        request.requirements.numerical_mode = mode;
        let kernel = pcu_facade::PcuDispatchKernelIr {
            numerical_requirements: request.requirements,
            ..*request.operation
        };
        let mut request = pcu_facade::PcuImplementationRequest {
            operation: &kernel,
            ..request
        };
        let mut output = [None];
        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
        assert_eq!(output[0].unwrap().implementation.local_id, 41);
        request.requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
    }
    request.requirements.float_underflow = PcuFloatUnderflowPolicy::AllowGradualUnderflow;
    let kernel = pcu_facade::PcuDispatchKernelIr {
        numerical_requirements: request.requirements,
        ..*request.operation
    };
    let request = pcu_facade::PcuImplementationRequest {
        operation: &kernel,
        ..request
    };
    assert_eq!(
        offers.implementation_offers(&request, &mut [None]),
        Err(PcuCpuHostOfferError::UnderflowMismatch)
    );
    let mut operations = kernel.ops.to_vec();
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert { range_policy, .. }) =
        &mut operations[1]
    {
        *range_policy = PcuRangePolicy::Clamp;
    }
    let kernel = pcu_facade::PcuDispatchKernelIr {
        ops: &operations,
        ..kernel
    };
    assert_eq!(
        PcuCpuCheckedConversion
            .prepare_host_kernel(&kernel)
            .unwrap_err(),
        PcuCpuHostError::UnsupportedProfile
    );
}
#[test]
fn narrowing_and_widening_reject_short_mixed_bindings_before_publication() {
    let mut call = source::widen_prepare::<3, _>(&PcuCpuHostBackend::scalar()).unwrap();
    let mut output = [99.0_f64; 5];
    assert!(call(&[1.0, 2.0], &mut output).is_err());
    assert_eq!(output.map(f64::to_bits), [99.0_f64; 5].map(f64::to_bits));
    assert!(call(&[1.0, 2.0, 3.0], &mut output[..2]).is_err());
    assert_eq!(output.map(f64::to_bits), [99.0_f64; 5].map(f64::to_bits));
}
