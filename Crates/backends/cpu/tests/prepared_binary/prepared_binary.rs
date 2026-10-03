//! Exact prepared/source admission and transactional checked float behavior.
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedBinary,
    PcuCpuHostBackend,
    PcuCpuHostOffers,
    PcuCpuHostOfferError,
    PcuCpuPreparedBinaryError,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuBindingRef,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchControlOp,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuExecutionFaultKind,
    PcuExecutorId,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    PcuObjectKind,
    PcuObjectRef,
    PcuPreparedHostKernel,
    PcuProviderId,
    PcuRangePolicy,
    PcuReproducibility,
    PcuNumericalMode,
};
#[path = "source/source.rs"]
mod source;

macro_rules! binary_case {
    ($name:ident, $module:ident, $entry:ident, $prepare:ident, $op:ident, $ty:ty) => {
        #[test]
        fn $name() {
            let backend = PcuCpuHostBackend::scalar();
            let mut call = source::$module::$prepare::<5, _>(&backend).unwrap();
            let lhs = [6.0, -12.0, 20.0, -0.0, 0.0];
            let rhs = [2.0, 3.0, -4.0, 1.0, 1.0];
            let mut output = [99.0; 7];
            call(&lhs, &rhs, &mut output).unwrap();
            for index in 0..5 {
                let expected = match PcuDispatchFloatBinaryOp::$op {
                    PcuDispatchFloatBinaryOp::Add => lhs[index] + rhs[index],
                    PcuDispatchFloatBinaryOp::Sub => lhs[index] - rhs[index],
                    PcuDispatchFloatBinaryOp::Mul => lhs[index] * rhs[index],
                    PcuDispatchFloatBinaryOp::Div => lhs[index] / rhs[index],
                };
                assert_eq!(output[index].to_bits(), expected.to_bits());
            }
            assert_eq!(output[5..], [99.0, 99.0]);
            for bad in [<$ty>::NAN, <$ty>::INFINITY, <$ty>::NEG_INFINITY] {
                let mut changed = lhs;
                changed[1] = bad;
                let before = output.map(<$ty>::to_bits);
                let error = call(&changed, &rhs, &mut output).unwrap_err();
                assert_eq!(error.fault().unwrap().invocation_id, 1);
                assert_eq!(
                    error.fault().unwrap().kind,
                    PcuExecutionFaultKind::InvalidFloatingOperand
                );
                assert_eq!(output.map(<$ty>::to_bits), before);
                call(&lhs, &rhs, &mut output).unwrap();
            }
            let before = output.map(<$ty>::to_bits);
            assert!(call(&lhs[..4], &rhs, &mut output).is_err());
            assert_eq!(output.map(<$ty>::to_bits), before);
        }
    };
}
binary_case!(f32_add, f32_source, add, add_prepare, Add, f32);
binary_case!(f32_sub, f32_source, sub, sub_prepare, Sub, f32);
binary_case!(f32_mul, f32_source, mul, mul_prepare, Mul, f32);
binary_case!(f32_div, f32_source, div, div_prepare, Div, f32);
binary_case!(f64_add, f64_source, add, add_prepare, Add, f64);
binary_case!(f64_sub, f64_source, sub, sub_prepare, Sub, f64);
binary_case!(f64_mul, f64_source, mul, mul_prepare, Mul, f64);
binary_case!(f64_div, f64_source, div, div_prepare, Div, f64);

macro_rules! edges {
    ($name:ident, $ty:ty, $module:ident, $ir:ident, $bindings:ident, $method:ident) => {
        #[test]
        fn $name() {
            let bindings = source::$module::$bindings();
            let builder = source::$module::$ir::<4>(&bindings).unwrap();
            let kernel = builder.ir();
            let mut operations = kernel.ops.to_vec();
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                for operation in &mut operations {
                    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                        underflow_policy,
                        ..
                    }) = operation
                    {
                        *underflow_policy = policy;
                    }
                }
                let kernel = pcu_facade::PcuDispatchKernelIr {
                    ops: &operations,
                    ..kernel
                };
                let mut prepared = PcuCpuCheckedBinary::<$ty>::new()
                    .prepare_host_kernel(&kernel)
                    .unwrap();
                let lhs = [<$ty>::MAX, <$ty>::MIN_POSITIVE, <$ty>::from_bits(1), 0.0];
                let rhs = [2.0, 0.5, 0.5, -1.0];
                for first in 0..4 {
                    let mut left = lhs;
                    let mut right = rhs;
                    for index in 0..first {
                        left[index] = 1.0;
                        right[index] = 1.0;
                    }
                    let expected = (0..4).find_map(|index| {
                        left[index]
                            .$method(right[index], policy)
                            .err()
                            .map(|kind| (index, kind))
                    });
                    let mut output: [$ty; 6] = [99.0; 6];
                    let result = prepared.call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
                        PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
                    ]);
                    match expected {
                        Some((index, kind)) => {
                            let Err(PcuCpuPreparedBinaryError::Fault(fault)) = result else {
                                panic!("expected checked fault")
                            };
                            assert_eq!(fault.invocation_id, index as u64);
                            assert_eq!(fault.kind, kind);
                            assert!(!fault.recovered);
                            assert_eq!(
                                output.map(<$ty>::to_bits),
                                [99.0 as $ty; 6].map(<$ty>::to_bits)
                            );
                        }
                        None => {
                            result.unwrap();
                            for index in 0..4 {
                                assert_eq!(
                                    output[index].to_bits(),
                                    left[index].$method(right[index], policy).unwrap().to_bits()
                                );
                            }
                            assert_eq!(output[4..], [99.0; 2]);
                        }
                    }
                }
            }
        }
    };
}
edges!(
    f32_mul_edges,
    f32,
    f32_source,
    mul_ir,
    mul_bindings,
    pcu_checked_mul_with_policy
);
edges!(
    f64_mul_edges,
    f64,
    f64_source,
    mul_ir,
    mul_bindings,
    pcu_checked_mul_with_policy
);

#[test]
fn first_zero_denominator_and_overflow_preserve_all_output_and_retry() {
    let mut call = source::f64_source::div_prepare::<3, _>(&PcuCpuHostBackend::scalar()).unwrap();
    let mut output = [8.0; 5];
    for rhs in [
        [1.0, 0.0, 0.0],
        [1.0, -0.0, 0.0],
        [1.0, f64::MIN_POSITIVE, 0.0],
    ] {
        let error = call(&[1.0, f64::MAX, 1.0], &rhs, &mut output).unwrap_err();
        assert_eq!(error.fault().unwrap().invocation_id, 1);
        assert_eq!(
            error.fault().unwrap().kind,
            if rhs[1] == 0.0 {
                PcuExecutionFaultKind::DivideByZero
            } else {
                PcuExecutionFaultKind::ArithmeticOverflow
            }
        );
        assert_eq!(output.map(f64::to_bits), [8.0_f64; 5].map(f64::to_bits));
        call(&[1.0, 2.0, 3.0], &[1.0; 3], &mut output).unwrap();
        assert_eq!(
            output.map(f64::to_bits),
            [1.0_f64, 2.0, 3.0, 8.0, 8.0].map(f64::to_bits)
        );
        output.fill(8.0);
    }
}

#[test]
fn detached_grid_swapped_ssa_broadcast_and_repeated_operand_preserve_layout() {
    let bindings = source::f32_source::sub_bindings();
    let builder = source::f32_source::sub_ir::<5>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut body = kernel.ops[..4].to_vec();
    for operation in &mut body {
        if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. }) =
            operation
        {
            *index = if binding.binding == 1 {
                PcuDispatchIndex::BindingElementZero
            } else {
                PcuDispatchIndex::GridStrideId
            };
        }
        if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { index, .. }) = operation {
            *index = PcuDispatchIndex::GridStrideId;
        }
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { lhs, rhs, .. }) =
            operation
        {
            core::mem::swap(lhs, rhs);
        }
    }
    let operations = [
        PcuDispatchOp::GridStrideLoop {
            extent: 5,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut prepared = {
        let kernel = pcu_facade::PcuDispatchKernelIr {
            ops: &operations,
            entry: pcu_facade::PcuDispatchEntryPoint {
                logical_shape: [2, 1, 1],
                ..kernel.entry
            },
            ..kernel
        };
        PcuCpuCheckedBinary::<f32>::new()
            .prepare_host_kernel(&kernel)
            .unwrap()
    };
    let mut output = [99.0_f32; 7];
    prepared
        .call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &[10.0_f32]),
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[1.0_f32, 2.0, 3.0, 4.0, 5.0]),
        ])
        .unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [9.0_f32, 8.0, 7.0, 6.0, 5.0, 99.0, 99.0].map(f32::to_bits)
    );
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { lhs, rhs, .. }) =
        &mut body[2]
    {
        *rhs = *lhs;
    }
    let operations = [
        PcuDispatchOp::GridStrideLoop {
            extent: 5,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = pcu_facade::PcuDispatchKernelIr {
        ops: &operations,
        ..kernel
    };
    let mut repeated = PcuCpuCheckedBinary::<f32>::new()
        .prepare_host_kernel(&kernel)
        .unwrap();
    assert!(
        repeated
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &[1.0_f32]),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &[10.0_f32]),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
            ])
            .is_err()
    );
    assert_eq!(
        output.map(f32::to_bits),
        [9.0_f32, 8.0, 7.0, 6.0, 5.0, 99.0, 99.0].map(f32::to_bits)
    );
}

#[test]
#[allow(clippy::cognitive_complexity)] // Complete format/policy/publication matrix preserves one owner lifecycle and exact case ordering.
fn offers_are_exact_independent_and_preserve_old_identifiers() {
    let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), identity, PcuExecutorId(0));
    macro_rules! check {
        ($module:ident,$ir:ident,$bindings:ident,$id:expr) => {{
            let bindings = source::$module::$bindings();
            let builder = source::$module::$ir::<3>(&bindings).unwrap();
            let kernel = builder.ir();
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
                let offer = output[0].unwrap();
                assert_eq!(offer.implementation.local_id, $id);
                assert_eq!(offer.implementation.revision, 2);
                assert_eq!(offer.requirements, request.requirements);
                request.requirements.numerical_options.reproducibility =
                    PcuReproducibility::PortableV1;
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
            }
            request.requirements.float_underflow = PcuFloatUnderflowPolicy::RejectSubnormalResult;
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
        }};
    }
    check!(f32_source, add_ir, add_bindings, 32);
    check!(f32_source, sub_ir, sub_bindings, 33);
    check!(f32_source, mul_ir, mul_bindings, 34);
    check!(f32_source, div_ir, div_bindings, 35);
    check!(f64_source, add_ir, add_bindings, 36);
    check!(f64_source, sub_ir, sub_bindings, 37);
    check!(f64_source, mul_ir, mul_bindings, 38);
    check!(f64_source, div_ir, div_bindings, 39);
}

#[test]
fn clamp_admits_but_bad_ssa_and_types_reject_cold() {
    let bindings = source::f64_source::add_bindings();
    let builder = source::f64_source::add_ir::<3>(&bindings).unwrap();
    let kernel = builder.ir();
    assert_eq!(
        PcuCpuCheckedBinary::<f32>::new()
            .prepare_host_kernel(&kernel)
            .unwrap_err(),
        PcuCpuPreparedBinaryError::UnsupportedProfile
    );
    let mut operations = kernel.ops.to_vec();
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { range_policy, .. }) =
        &mut operations[2]
    {
        *range_policy = PcuRangePolicy::Clamp;
    }
    let changed = pcu_facade::PcuDispatchKernelIr {
        ops: &operations,
        ..kernel
    };
    assert_eq!(
        PcuCpuCheckedBinary::<f64>::new()
            .prepare_host_kernel(&changed)
            .unwrap()
            .range_policy(),
        PcuRangePolicy::Clamp
    );
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { lhs, .. }) =
        &mut operations[2]
    {
        *lhs = PcuDispatchValueId(200);
    }
    let changed = pcu_facade::PcuDispatchKernelIr {
        ops: &operations,
        ..kernel
    };
    assert!(matches!(
        PcuCpuCheckedBinary::<f64>::new().prepare_host_kernel(&changed),
        Err(PcuCpuPreparedBinaryError::InvalidValueFlow(_))
    ));
}
