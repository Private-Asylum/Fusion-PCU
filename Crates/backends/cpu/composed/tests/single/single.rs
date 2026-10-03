//! One unused checked effect remains observable under a separately frozen profile.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuExecutorId,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuNumericalMode,
    PcuObjectKind,
    PcuObjectRef,
    PcuPrecisionPolicy,
    PcuProviderId,
};

#[pcu(invocations = 4, crate_path = ::pcu_facade)]
fn unused<T: PcuCheckedFloat>(input: &[T], divisor: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    let discarded = input[id] / divisor[id];
    output[id] = input[id];
}
#[pcu(invocations = 2, crate_path = ::pcu_facade)]
fn grid<T: PcuCheckedFloat>(input: &[T], divisor: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < 4 {
        let discarded = input[id] / divisor[id];
        output[id] = input[id];
        id += stride;
    }
}

fn width<T: PcuCheckedFloat>(one: T, two: T, zero: T, id: u32) {
    let backend = crate::PcuCpuHostBackend::scalar();
    let mut genuine = unused_prepare::<T, _>(&backend).unwrap();
    let mut output = [two; 6];
    genuine(&[one; 4], &[one; 4], &mut output).unwrap();
    assert_bytes(&output, &[one, one, one, one, two, two]);
    let bindings = unused_bindings::<T>();
    let direct = unused_ir::<T>(&bindings).unwrap();
    let bindings_grid = grid_bindings::<T>();
    let grid = grid_ir::<T>(&bindings_grid).unwrap();
    let cases = |base: &PcuDispatchKernelIr<'_>| {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            for compound in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    let mut requirements = base.numerical_requirements;
                    requirements.numerical_mode = mode;
                    requirements.numerical_options.compound_arithmetic = compound;
                    requirements.numerical_options.precision = precision;
                    policies::<T>(base, one, two, zero, id, requirements);
                }
            }
        }
    };
    direct.with_ir(cases);
    grid.with_ir(cases);
}

fn policies<T: PcuCheckedFloat>(
    base: &PcuDispatchKernelIr<'_>,
    one: T,
    two: T,
    zero: T,
    id: u32,
    requirements: PcuImplementationRequirements,
) {
    for underflow in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let mut ops = base.ops.to_vec();
            for op in &mut ops {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    range_policy,
                    underflow_policy,
                    ..
                }) = op
                {
                    *range_policy = range;
                    *underflow_policy = underflow;
                }
            }
            let mut kernel = *base;
            kernel.ops = &ops;
            kernel.numerical_requirements = requirements;
            kernel.numerical_requirements.range_policy = range;
            kernel.numerical_requirements.float_underflow = underflow;
            let backend = crate::PcuCpuHostBackend::scalar();
            let mut prepared = backend.prepare_host_kernel(&kernel).unwrap();
            let crate::PcuCpuPreparedHost::Composed(plan) = &prepared else {
                panic!("single effect must use the detached profile");
            };
            assert_eq!(plan.local_id(), id);
            assert_eq!(plan.requirements(), kernel.numerical_requirements);
            transaction::<T>(&mut prepared, one, two, zero);
            offer(&kernel, id, 1);
            kernel
                .numerical_requirements
                .numerical_options
                .reproducibility = PcuReproducibility::PortableV1;
            assert_eq!(
                backend.prepare_host_kernel(&kernel).unwrap_err(),
                crate::PcuCpuHostError::UnsupportedProfile
            );
        }
    }
}
fn transaction<T: PcuCheckedFloat>(plan: &mut crate::PcuCpuPreparedHost, one: T, two: T, zero: T) {
    let mut output = [two; 6];
    let mut divisor = [one; 4];
    divisor[2] = zero;
    let input = [one; 4];
    let mut call = |output: &mut [T], divisor: &[T]| {
        plan.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), divisor),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
        ])
    };
    let fault = call(&mut output, &divisor).unwrap_err().fault().unwrap();
    assert_eq!(
        fault,
        PcuExecutionFault {
            invocation_id: 2,
            kind: PcuExecutionFaultKind::DivideByZero,
            recovered: false
        }
    );
    assert_bytes(&output, &[two; 6]);
    divisor[2] = one;
    call(&mut output, &divisor).unwrap();
    assert_bytes(&output, &[one, one, one, one, two, two]);
    assert!(call(&mut output[..3], &divisor).is_err());
    assert_bytes(&output, &[one, one, one, one, two, two]);
}
fn assert_bytes<T: PcuCheckedFloat>(left: &[T], right: &[T]) {
    for (left, right) in left.iter().zip(right) {
        assert_eq!(left.encode_le().as_ref(), right.encode_le().as_ref());
    }
}
fn offer(kernel: &PcuDispatchKernelIr<'_>, id: u32, revision: u64) {
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 1,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers =
        crate::PcuCpuHostOffers::new(crate::PcuCpuHostBackend::scalar(), device, PcuExecutorId(0));
    let request = PcuImplementationRequest {
        device,
        executor: PcuExecutorId(0),
        operation: kernel,
        requirements: kernel.numerical_requirements,
        boundary: PcuCostBoundary::Host,
    };
    let mut slots = [None];
    assert_eq!(offers.implementation_offers(&request, &mut slots), Ok(1));
    let frozen = slots[0].unwrap();
    frozen.validate_request(&request).unwrap();
    assert_eq!(
        (
            frozen.implementation.local_id,
            frozen.implementation.revision
        ),
        (id, revision)
    );
    let mut wrong = request;
    wrong.requirements.range_policy = if wrong.requirements.range_policy == PcuRangePolicy::Reject {
        PcuRangePolicy::Clamp
    } else {
        PcuRangePolicy::Reject
    };
    assert_eq!(offers.implementation_offers(&wrong, &mut slots), Ok(0));
}
#[test]
fn exact_primitives_keep_old_ids_and_zero_math_uses_separate_transport() {
    let body = [
        load(0, 0, false),
        load(1, 1, false),
        binary(
            2,
            0,
            1,
            PcuDispatchFloatBinaryOp::Div,
            PcuRangePolicy::Reject,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        store(2, 2),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let binding = |index, access| {
        PcuBinding::value(
            None,
            0,
            index,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::f32(),
        )
    };
    let bindings = [
        binding(0, PcuBindingAccess::ReadOnly),
        binding(1, PcuBindingAccess::ReadOnly),
        binding(2, PcuBindingAccess::ReadWrite),
    ];
    let mut kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "single_primitive",
            logical_shape: [4, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &body,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    };
    let backend = crate::PcuCpuHostBackend::scalar();
    let crate::PcuCpuPreparedHost::F32Binary(primitive) =
        backend.prepare_host_kernel(&kernel).unwrap()
    else {
        panic!("canonical primitive must remain statically admitted");
    };
    assert_eq!(primitive.operation(), PcuDispatchFloatBinaryOp::Div);
    offer(
        &kernel,
        35,
        crate::PCU_CPU_FLOAT_BINARY_IMPLEMENTATION_REVISION,
    );
    let zero_math = [
        load(0, 0, false),
        store(2, 0),
        store(2, 0),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    kernel.ops = &zero_math;
    assert!(matches!(
        backend.prepare_host_kernel(&kernel),
        Ok(crate::PcuCpuPreparedHost::Transport(_))
    ));
    assert!(matches!(
        PcuCpuCheckedComposedMap::<f32>::new().prepare_host_kernel(&kernel),
        Err(PcuCpuComposedMapError::InvalidResources(_)
            | PcuCpuComposedMapError::UnsupportedProfile)
    ));
}
#[test]
fn single_dead_division_has_exact_identity_fault_rollback_retry_and_portable_refusal() {
    width(
        fusion_pcu::PcuF16Bits::from_bits(0x3c00),
        fusion_pcu::PcuF16Bits::from_bits(0x4000),
        fusion_pcu::PcuF16Bits::from_bits(0),
        18688,
    );
    width(
        fusion_pcu::PcuBf16Bits::from_bits(0x3f80),
        fusion_pcu::PcuBf16Bits::from_bits(0x4000),
        fusion_pcu::PcuBf16Bits::from_bits(0),
        18689,
    );
    width(
        fusion_pcu::PcuF8E4M3FnBits::from_bits(0x38),
        fusion_pcu::PcuF8E4M3FnBits::from_bits(0x40),
        fusion_pcu::PcuF8E4M3FnBits::from_bits(0),
        18690,
    );
    width(
        fusion_pcu::PcuF8E5M2Bits::from_bits(0x3c),
        fusion_pcu::PcuF8E5M2Bits::from_bits(0x40),
        fusion_pcu::PcuF8E5M2Bits::from_bits(0),
        18691,
    );
    width(1_f32, 2_f32, 0_f32, 18692);
    width(1_f64, 2_f64, 0_f64, 18693);
}
