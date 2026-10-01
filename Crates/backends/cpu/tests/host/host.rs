//! Ordinary source acceptance through the unified static CPU adapter.

#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuHostArgumentError,
    PcuCpuHostBackend,
    PcuCpuHostError,
    PcuCpuHostOfferError,
    PcuCpuHostOffers,
};
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchDataOp,
    PcuBindingRef,
    PcuHostArgument,
    PcuPreparedHostKernel,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuExecutorId,
    PcuHostKernelBackend,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderId,
};
#[path = "../checked_integer/source/source.rs"]
mod source;

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn neg<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

macro_rules! source_case {
    ($name:ident, $module:ident, $ty:ty, $ordinal:expr) => {
        #[test]
        fn $name() {
            let backend = PcuCpuHostBackend::scalar();
            let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
                provider: PcuProviderId(3),
                generation: 7,
                kind: PcuObjectKind::Device,
                id: 0,
            })
            .unwrap();
            let offers = PcuCpuHostOffers::new(backend, identity, PcuExecutorId(0));
            let bindings = source::$module::add_bindings();
            for operation in 0..3 {
                let builder = match operation {
                    0 => source::$module::add_ir::<3>(&bindings).unwrap(),
                    1 => source::$module::sub_ir::<3>(&bindings).unwrap(),
                    _ => source::$module::mul_ir::<3>(&bindings).unwrap(),
                };
                let kernel = builder.ir();
                let request = PcuImplementationRequest {
                    device: identity,
                    executor: PcuExecutorId(0),
                    operation: &kernel,
                    requirements: PcuImplementationRequirements::default(),
                    boundary: PcuCostBoundary::Host,
                };
                let mut output = [None];
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
                let offer = output[0].unwrap();
                assert_eq!(offer.implementation.local_id, 4 + $ordinal * 3 + operation);
                assert_eq!(
                    offer.implementation.revision,
                    fusion_pcu_cpu::PCU_CPU_INTEGER_IMPLEMENTATION_REVISION
                );
                assert_eq!(offer.requirements, request.requirements);
                assert_eq!(offer.workspace_bytes, Some(0));
                assert_eq!(
                    offer.cost,
                    pcu_facade::PcuImplementationCost::unknown(PcuCostBoundary::Host)
                );
            }

            let lhs = [2 as $ty, 6 as $ty, 9 as $ty];
            let rhs = [1 as $ty, 2 as $ty, 3 as $ty];
            let mut out = [99 as $ty; 4];
            source::$module::add_prepare::<3, _>(&backend).unwrap()(&lhs, &rhs, &mut out).unwrap();
            assert_eq!(out, [3, 8, 12, 99]);
            source::$module::sub_prepare::<3, _>(&backend).unwrap()(&lhs, &rhs, &mut out).unwrap();
            assert_eq!(out, [1, 4, 6, 99]);
            let mut mul = source::$module::mul_prepare::<3, _>(&backend).unwrap();
            mul(&lhs, &rhs, &mut out).unwrap();
            assert_eq!(out, [2, 12, 27, 99]);
            let failure = mul(&[2, <$ty>::MAX, 9], &rhs, &mut out).unwrap_err();
            assert_eq!(failure.fault().unwrap().invocation_id, 1);
            assert_eq!(out, [2, 12, 27, 99]);
            mul(&[1, 1, 1], &rhs, &mut out).unwrap();
            assert_eq!(out, [1, 2, 3, 99]);
        }
    };
}
source_case!(i8_source, i8_source, i8, 0);
source_case!(u8_source, u8_source, u8, 1);
source_case!(i16_source, i16_source, i16, 2);
source_case!(u16_source, u16_source, u16, 3);
source_case!(i32_source, i32_source, i32, 4);
source_case!(u32_source, u32_source, u32, 5);
source_case!(i64_source, i64_source, i64, 6);
source_case!(u64_source, u64_source, u64, 7);

#[test]
fn neg_source_retains_bits_and_transactional_faults() {
    let mut call = neg_prepare::<3, _>(&PcuCpuHostBackend::scalar()).unwrap();
    let mut output = [42.0; 4];
    call(&[0.0, -0.0, 2.0], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [-0.0_f32, 0.0, -2.0, 42.0].map(f32::to_bits)
    );
    let before = output;
    let error = call(&[1.0, f32::NAN, 2.0], &mut output).unwrap_err();
    assert_eq!(error.fault().unwrap().invocation_id, 1);
    assert_eq!(output.map(f32::to_bits), before.map(f32::to_bits));
}

#[test]
fn malformed_requests_are_errors_and_valid_broader_shapes_are_unsupported() {
    let bindings = source::u32_source::add_bindings();
    let builder = source::u32_source::add_ir::<3>(&bindings).unwrap();
    let kernel = builder.ir();
    let backend = PcuCpuHostBackend::scalar();
    let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(backend, identity, PcuExecutorId(0));
    let request = PcuImplementationRequest {
        device: identity,
        executor: PcuExecutorId(0),
        operation: &kernel,
        requirements: PcuImplementationRequirements::default(),
        boundary: PcuCostBoundary::Host,
    };
    let mut result = [None];
    assert_eq!(offers.implementation_offers(&request, &mut result), Ok(1));
    assert_eq!(result[0].unwrap().implementation.local_id, 19);
    assert_eq!(result[0].unwrap().workspace_bytes, Some(0));
    let mut malformed = kernel.ops.to_vec();
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { lhs, .. }) =
        &mut malformed[2]
    {
        *lhs = PcuDispatchValueId(99);
    }
    let bad_kernel = pcu_facade::PcuDispatchKernelIr {
        ops: &malformed,
        ..kernel
    };
    let bad = PcuImplementationRequest {
        operation: &bad_kernel,
        ..request
    };
    assert!(matches!(
        offers.implementation_offers(&bad, &mut []),
        Err(PcuCpuHostOfferError::Provider(
            PcuCpuHostError::InvalidValueFlow(_)
        ))
    ));
    let mut broad_kernel = kernel;
    broad_kernel.entry.logical_shape = [3, 2, 1];
    assert!(matches!(
        backend.prepare_host_kernel(&broad_kernel),
        Err(PcuCpuHostError::UnsupportedProfile)
    ));
    broad_kernel.entry.logical_shape = [3, 0, 1];
    assert!(matches!(
        backend.prepare_host_kernel(&broad_kernel),
        Err(PcuCpuHostError::InvalidLogicalShape([3, 0, 1]))
    ));
}

#[test]
fn unified_schema_errors_retain_details_and_leave_output_unchanged() {
    let bindings = source::u32_source::add_bindings();
    let builder = source::u32_source::add_ir::<3>(&bindings).unwrap();
    let mut prepared = PcuCpuHostBackend::scalar()
        .prepare_host_kernel(&builder.ir())
        .unwrap();
    let lhs = [1_u32, 2, 3];
    let rhs = [1_u32, 1, 1];
    let mut output = [42_u32; 4];
    assert_eq!(
        prepared.call(&mut []),
        Err(PcuCpuHostError::Arguments(PcuCpuHostArgumentError::Count {
            expected: 3,
            actual: 0
        }))
    );
    let wrong_type = [1_u16; 3];
    let error = prepared
        .call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &wrong_type),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ])
        .unwrap_err();
    assert!(
        matches!(error, PcuCpuHostError::Arguments(PcuCpuHostArgumentError::TypeMismatch { binding, expected: pcu_facade::PcuScalarType::U32, actual: pcu_facade::PcuScalarType::U16 }) if binding == PcuBindingRef::new(0, 0))
    );
    let error = prepared
        .call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs[..1]),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ])
        .unwrap_err();
    assert!(matches!(
        error,
        PcuCpuHostError::Arguments(PcuCpuHostArgumentError::BufferTooSmall {
            required_bytes: 12,
            actual_bytes: 4,
            ..
        })
    ));
    let error = prepared
        .call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ])
        .unwrap_err();
    assert!(matches!(
        error,
        PcuCpuHostError::Arguments(PcuCpuHostArgumentError::DuplicateBinding(_))
    ));
    let error = prepared
        .call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), &output),
        ])
        .unwrap_err();
    assert!(matches!(
        error,
        PcuCpuHostError::Arguments(PcuCpuHostArgumentError::AccessMismatch { .. })
    ));
    assert_eq!(output, [42; 4]);
    prepared
        .call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ])
        .unwrap();
    assert_eq!(output, [2, 3, 4, 42]);
}

#[test]
fn unified_valid_clamp_and_broader_integer_profiles_have_zero_offers() {
    let backend = PcuCpuHostBackend::scalar();
    let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(backend, identity, PcuExecutorId(0));
    let bindings = neg_bindings();
    let builder = neg_ir::<3>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut ops = kernel.ops.to_vec();
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary { range_policy, .. }) =
        &mut ops[1]
    {
        *range_policy = pcu_facade::PcuRangePolicy::Clamp;
    }
    let clamp = pcu_facade::PcuDispatchKernelIr {
        ops: &ops,
        ..kernel
    };
    let request = PcuImplementationRequest {
        device: identity,
        executor: PcuExecutorId(0),
        operation: &clamp,
        requirements: PcuImplementationRequirements {
            range_policy: pcu_facade::PcuRangePolicy::Clamp,
            ..Default::default()
        },
        boundary: PcuCostBoundary::Host,
    };
    assert_eq!(offers.implementation_offers(&request, &mut []), Ok(0));
    let mut malformed_ops = ops.clone();
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary { value, .. }) =
        &mut malformed_ops[1]
    {
        *value = PcuDispatchValueId(99);
    }
    let malformed = pcu_facade::PcuDispatchKernelIr {
        ops: &malformed_ops,
        ..kernel
    };
    assert!(matches!(
        offers.implementation_offers(
            &PcuImplementationRequest {
                operation: &malformed,
                ..request
            },
            &mut []
        ),
        Err(PcuCpuHostOfferError::Provider(
            PcuCpuHostError::InvalidValueFlow(_)
        ))
    ));
    let bindings = source::u64_source::add_bindings();
    let builder = source::u64_source::add_ir::<3>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut bindings = bindings.to_vec();
    bindings.push(pcu_facade::PcuBinding::scalar::<u64>(
        None,
        0,
        3,
        pcu_facade::PcuBindingStorageClass::Storage,
        pcu_facade::PcuBindingAccess::ReadOnly,
    ));
    let broader = pcu_facade::PcuDispatchKernelIr {
        bindings: &bindings,
        ..kernel
    };
    let request = PcuImplementationRequest {
        operation: &broader,
        requirements: PcuImplementationRequirements::default(),
        ..request
    };
    assert_eq!(offers.implementation_offers(&request, &mut []), Ok(0));
}
